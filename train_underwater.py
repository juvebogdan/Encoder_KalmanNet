import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random

# Import our custom modules
from underwater_dataset import UnderwaterDataset
from encoder_underwater import UnderwaterEncoder, UnderwaterEncoderWithPrior

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def custom_collate(batch):
    """Custom collate function to handle variable-length amplitude/delay arrays"""
    # Extract components
    basic_features = [item[0][0] for item in batch]  # List of tensors with shape [num_arrivals, 2]
    summary_features = torch.stack([item[0][1] for item in batch])  # Shape: [batch_size, 5] (now with 5 features)
    targets = torch.stack([item[1] for item in batch])  # Shape: [batch_size, 1]
    
    # Find max number of arrivals in this batch
    max_arrivals = max(features.shape[0] for features in basic_features)
    
    # Create padded tensor and mask
    padded_features = torch.zeros(len(batch), max_arrivals, 2)
    mask = torch.zeros(len(batch), max_arrivals, dtype=bool)
    
    # Fill in actual data and mask
    for i, features in enumerate(basic_features):
        num_arrivals = features.shape[0]
        padded_features[i, :num_arrivals, :] = features
        mask[i, :num_arrivals] = True
    
    return (padded_features, mask, summary_features), targets

def train_encoder(model, train_loader, val_loader, num_epochs=50, learning_rate=0.003, weight_decay=0.0001):
    """Train the basic encoder model with L2 regularization"""
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Directory for saving models
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = f"{save_dir}/best_encoder.pth"
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, ((basic_features, mask, summary_features), targets) in enumerate(train_loader):
            # Move data to device
            basic_features = basic_features.to(device)
            mask = mask.to(device)
            summary_features = summary_features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model((basic_features, mask, summary_features))
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ((basic_features, mask, summary_features), targets) in val_loader:
                basic_features = basic_features.to(device)
                mask = mask.to(device)
                summary_features = summary_features.to(device)
                targets = targets.to(device)
                
                outputs = model((basic_features, mask, summary_features))
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch [{epoch+1}/{num_epochs}], New best model saved with val_loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{save_dir}/encoder_loss_plot.png")
    plt.close()
    
    return model, train_losses, val_losses

def train_encoder_with_prior(model, dataset, train_trajectories, val_trajectories, num_epochs=50, learning_rate=0.003, batch_size=32, noise_scale=0.05, weight_decay=0.0001):
    """
    Train the encoder with prior using ground truth + noise as prior
    
    Parameters:
    - model: Encoder model with prior
    - dataset: Dataset object
    - train_trajectories: Dictionary mapping trajectory IDs to indices
    - val_trajectories: Dictionary mapping trajectory IDs to indices
    - num_epochs: Number of training epochs
    - learning_rate: Learning rate for optimizer
    - batch_size: How many trajectories to process before updating weights
    - noise_scale: Amount of noise to add to ground truth (as percentage)
    - weight_decay: L2 regularization parameter
    """
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Directory for saving models
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = f"{save_dir}/best_encoder_with_prior.pth"
    
    # Get list of trajectory IDs
    train_traj_ids = list(train_trajectories.keys())
    val_traj_ids = list(val_trajectories.keys())
    
    print(f"Training with {len(train_traj_ids)} trajectories, validating with {len(val_traj_ids)} trajectories")
    
    if len(train_traj_ids) == 0 or len(val_traj_ids) == 0:
        print("ERROR: No trajectories found. Cannot train model.")
        return model, train_losses, val_losses
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_trajectories = 0
        
        # Process training trajectories
        random.shuffle(train_traj_ids)
        
        for traj_idx, traj_id in enumerate(train_traj_ids):
            indices = train_trajectories[traj_id]
            
            # Zero gradients for this trajectory
            optimizer.zero_grad()
            
            # Process the trajectory sequentially
            traj_loss = 0.0
            valid_steps = 0
            
            for step in range(len(indices)):
                idx = indices[step]
                
                # Get data point - now these are normalized
                (basic_features, summary_features), target = dataset[idx]
                
                # Create mask for single item
                mask = torch.ones(1, basic_features.shape[0], dtype=bool)
                
                # Move data to device
                basic_features = basic_features.unsqueeze(0).to(device)
                mask = mask.to(device)
                summary_features = summary_features.unsqueeze(0).to(device)
                target = target.unsqueeze(0).to(device)
                
                # Generate prior using normalized target + noise
                prior = torch.zeros(1, 1).to(device)
                # Add noise directly to the normalized target
                noise = torch.randn(1).to(device) * noise_scale
                prior[0, 0] = target.item() + noise
                
                # Ensure prior is within reasonable bounds (-3 to +3 std is reasonable for normalized data)
                prior[0, 0] = torch.clamp(prior[0, 0], min=-3.0, max=3.0)
                
                # Forward pass
                output = model((basic_features, mask, summary_features), prior)
                step_loss = criterion(output, target)
                
                # Add to total loss
                traj_loss += step_loss
                valid_steps += 1
            
            # Backward and optimize if we have a valid trajectory
            if valid_steps > 0:
                avg_traj_loss = traj_loss / valid_steps
                avg_traj_loss.backward()
                optimizer.step()
                
                total_loss += avg_traj_loss.item()
                num_trajectories += 1
                
                # Print progress every 10 trajectories
                if (traj_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, processed {traj_idx+1}/{len(train_traj_ids)} trajectories")
        
        # Calculate average epoch loss
        epoch_train_loss = total_loss / num_trajectories if num_trajectories > 0 else float('inf')
        train_losses.append(epoch_train_loss)
        
        # Validation - perform once per epoch
        model.eval()
        val_total_loss = 0.0
        val_num_trajectories = 0
        
        with torch.no_grad():
            for traj_id in val_traj_ids:
                indices = val_trajectories[traj_id]
                
                # Initialize for sequential processing
                traj_total_loss = 0.0
                valid_steps = 0
                
                for step in range(len(indices)):
                    idx = indices[step]
                    
                    # Get data point - normalized
                    (basic_features, summary_features), target = dataset[idx]
                    
                    # Create mask for single item
                    mask = torch.ones(1, basic_features.shape[0], dtype=bool)
                    
                    # Move data to device
                    basic_features = basic_features.unsqueeze(0).to(device)
                    mask = mask.to(device)
                    summary_features = summary_features.unsqueeze(0).to(device)
                    target = target.unsqueeze(0).to(device)
                    
                    # Generate prior using normalized target + noise (same as training)
                    prior = torch.zeros(1, 1).to(device)
                    noise = torch.randn(1).to(device) * noise_scale
                    prior[0, 0] = target.item() + noise
                    prior[0, 0] = torch.clamp(prior[0, 0], min=-3.0, max=3.0)
                    prior[0, 0] = torch.max(prior[0, 0], torch.tensor(0.1).to(device))
                    prior[0, 0] = torch.max(prior[0, 0], torch.tensor(0.1).to(device))
                    
                    # Forward pass
                    output = model((basic_features, mask, summary_features), prior)
                    loss = criterion(output, target)
                    
                    traj_total_loss += loss.item()
                    valid_steps += 1
                
                # Add trajectory loss if we have valid steps
                if valid_steps > 0:
                    val_total_loss += traj_total_loss / valid_steps
                    val_num_trajectories += 1
        
        # Calculate average validation loss
        epoch_val_loss = val_total_loss / val_num_trajectories if val_num_trajectories > 0 else float('inf')
        val_losses.append(epoch_val_loss)
        
        # Save the model if validation loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch [{epoch+1}/{num_epochs}], New best model saved with val_loss: {epoch_val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (With Prior)')
    plt.legend()
    plt.savefig(f"{save_dir}/encoder_with_prior_loss_plot.png")
    plt.close()
    
    return model, train_losses, val_losses

def evaluate_model(model, dataset, test_trajectories, with_prior=False, noise_scale=0.05):
    """
    Evaluate model performance on test trajectories
    
    Parameters:
    - model: Encoder model (with or without prior)
    - dataset: Dataset object
    - test_trajectories: Dictionary mapping trajectory IDs to indices
    - with_prior: Whether to use prior in the model
    - noise_scale: Amount of noise to add for first step initialization
    
    Returns:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - all_predictions: List of model predictions
    - all_targets: List of ground truth targets
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    # For storing full trajectory data
    trajectory_data = {}
    
    test_traj_ids = list(test_trajectories.keys())
    print(f"Evaluating with {len(test_traj_ids)} test trajectories")
    
    if len(test_traj_ids) == 0:
        print("ERROR: No test trajectories found. Cannot evaluate model.")
        return 0, 0, 0, [], []
    
    # Select a few trajectories to visualize (first 3 for simplicity)
    trajectories_to_visualize = test_traj_ids[:3] if len(test_traj_ids) >= 3 else test_traj_ids
    
    # Process each test trajectory
    with torch.no_grad():
        for traj_id in test_traj_ids:
            indices = test_trajectories[traj_id]
            
            # Initialize for sequential processing
            traj_predictions = []
            traj_targets = []
            timestamps = []
            
            for step in range(len(indices)):
                idx = indices[step]
                
                # Get data point - normalized
                (basic_features, summary_features), target = dataset[idx]
                timestamps.append(dataset.all_data[idx]['timestamp'])
                
                # Store original target for metrics (denormalized)
                original_target = dataset.denormalize_distance(target.item())
                
                # Create mask for single item
                mask = torch.ones(1, basic_features.shape[0], dtype=bool)
                
                # Move data to device
                basic_features = basic_features.unsqueeze(0).to(device)
                mask = mask.to(device)
                summary_features = summary_features.unsqueeze(0).to(device)
                target = target.unsqueeze(0).to(device)
                
                if with_prior:
                    # Generate prior using ground truth + noise (same as training)
                    prior = torch.zeros(1, 1).to(device)
                    noise = torch.randn(1).to(device) * noise_scale
                    prior[0, 0] = target.item() + noise
                    prior[0, 0] = torch.clamp(prior[0, 0], min=-3.0, max=3.0)
                    prior[0, 0] = torch.max(prior[0, 0], torch.tensor(0.1).to(device))
                    
                    output = model((basic_features, mask, summary_features), prior)
                else:
                    output = model((basic_features, mask, summary_features))
                
                # Compute loss on normalized values
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Denormalize for storing and display
                prediction_denormalized = dataset.denormalize_distance(output.item())
                
                # Store results
                all_predictions.append(prediction_denormalized)
                all_targets.append(original_target)
                traj_predictions.append(prediction_denormalized)
                traj_targets.append(original_target)
            
            # Store trajectory data for visualization if this is one of the selected trajectories
            if traj_id in trajectories_to_visualize:
                trajectory_data[traj_id] = {
                    "timestamps": timestamps,
                    "predictions": traj_predictions,
                    "ground_truth": traj_targets
                }
    
    # Check if we have valid predictions
    if len(all_predictions) == 0:
        print("WARNING: No valid predictions were made during evaluation.")
        return 0, 0, 0, [], []
    
    # Calculate metrics using denormalized values
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_targets))
    
    print(f"Test Loss: {total_loss / len(all_predictions):.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Plot predictions vs targets (scatter plot)
    plt.figure(figsize=(10, 5))
    plt.scatter(all_targets, all_predictions, alpha=0.3)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel('True Distance')
    plt.ylabel('Predicted Distance')
    plt.title('Predictions vs Targets')
    plt.savefig(f"saved_models/predictions_vs_targets_{'with_prior' if with_prior else 'basic'}.png")
    plt.close()
    
    # Plot individual trajectories
    for traj_id, data in trajectory_data.items():
        plt.figure(figsize=(12, 6))
        
        # Convert timestamps to relative time if they're absolute
        relative_time = np.array(data["timestamps"]) - data["timestamps"][0]
        
        # Plot the trajectory
        plt.plot(relative_time, data["ground_truth"], 'k-', linewidth=2, label='Ground Truth')
        plt.plot(relative_time, data["predictions"], 'r--', linewidth=2, label=f'{"With Prior" if with_prior else "Basic"} Encoder')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Distance (m)')
        plt.title(f'Trajectory {traj_id}: Ground Truth vs {"With Prior" if with_prior else "Basic"} Prediction')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Print numerical values for the first few points
        print(f"\nTrajectory {traj_id} - First 10 steps:")
        print(f"{'Time':>6} | {'Ground Truth':>12} | {'Prediction':>12} | {'Error':>12}")
        print("-" * 50)
        for i in range(min(10, len(data["timestamps"]))):
            error = abs(data["ground_truth"][i] - data["predictions"][i])
            print(f"{relative_time[i]:6.1f} | {data['ground_truth'][i]:12.4f} | {data['predictions'][i]:12.4f} | {error:12.4f}")
        
        # Save the plot
        plt.savefig(f"saved_models/trajectory_{traj_id}_{'with_prior' if with_prior else 'basic'}.png")
        plt.close()
    
    return mse, rmse, mae, all_predictions, all_targets

def plot_trajectory_comparison(dataset, test_trajectories, basic_encoder, encoder_with_prior, noise_scale=0.05):
    """
    Plot comparison of ground truth vs basic encoder vs encoder with prior for specific trajectories
    
    Parameters:
    - dataset: Dataset object
    - test_trajectories: Dictionary mapping trajectory IDs to indices
    - basic_encoder: Trained basic encoder model
    - encoder_with_prior: Trained encoder with prior model
    - noise_scale: Amount of noise to add for first step initialization
    """
    basic_encoder.eval()
    encoder_with_prior.eval()
    
    # Select a few trajectories to visualize
    test_traj_ids = list(test_trajectories.keys())
    trajectories_to_visualize = test_traj_ids[:3] if len(test_traj_ids) >= 3 else test_traj_ids
    
    for traj_id in trajectories_to_visualize:
        indices = test_trajectories[traj_id]
        
        # Initialize storage
        timestamps = []
        ground_truth = []
        basic_predictions = []
        prior_predictions = []
        
        with torch.no_grad():
            for step in range(len(indices)):
                idx = indices[step]
                
                # Get data point - normalized
                (basic_features, summary_features), target = dataset[idx]
                timestamps.append(dataset.all_data[idx]['timestamp'])
                
                # Store original target for plotting
                ground_truth.append(dataset.denormalize_distance(target.item()))
                
                # Create mask and prepare inputs
                mask = torch.ones(1, basic_features.shape[0], dtype=bool)
                basic_features = basic_features.unsqueeze(0).to(device)
                mask = mask.to(device)
                summary_features = summary_features.unsqueeze(0).to(device)
                target = target.unsqueeze(0).to(device)
                
                # Basic encoder prediction
                basic_output = basic_encoder((basic_features, mask, summary_features))
                # Denormalize for display
                basic_predictions.append(dataset.denormalize_distance(basic_output.item()))
                
                # Encoder with prior prediction (using ground truth + noise)
                prior = torch.zeros(1, 1).to(device)
                noise = torch.randn(1).to(device) * noise_scale
                prior[0, 0] = target.item() + noise
                prior[0, 0] = torch.clamp(prior[0, 0], min=-3.0, max=3.0)
                
                prior_output = encoder_with_prior((basic_features, mask, summary_features), prior)
                # Denormalize for display
                prior_predictions.append(dataset.denormalize_distance(prior_output.item()))
        
        # Convert timestamps to relative time
        relative_time = np.array(timestamps) - timestamps[0]
        
        # Plot comparison
        plt.figure(figsize=(14, 7))
        plt.plot(relative_time, ground_truth, 'k-', linewidth=2, label='Ground Truth')
        plt.plot(relative_time, basic_predictions, 'b--', linewidth=2, label='Basic Encoder')
        plt.plot(relative_time, prior_predictions, 'r--', linewidth=2, label='Encoder with Prior')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Distance (m)')
        plt.title(f'Trajectory {traj_id}: Comparison of Prediction Methods')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Print numerical values for the first few points
        print(f"\nTrajectory {traj_id} - Comparison (First 10 steps):")
        print(f"{'Time':>6} | {'Ground Truth':>12} | {'Basic':>12} | {'With Prior':>12}")
        print("-" * 60)
        for i in range(min(10, len(timestamps))):
            print(f"{relative_time[i]:6.1f} | {ground_truth[i]:12.4f} | {basic_predictions[i]:12.4f} | {prior_predictions[i]:12.4f}")
        
        # Save the plot
        plt.savefig(f"saved_models/trajectory_{traj_id}_comparison.png")
        plt.close()

def main():
    # Load data
    data_dir = 'data'  # Update this to your data directory
    dataset = UnderwaterDataset(data_dir)
    
    # Save normalization parameters for future use
    os.makedirs('saved_models', exist_ok=True)
    dataset.save_normalization_params('saved_models/normalization_params.npy')
    
    # Group data by trajectory ID
    trajectories = {}
    for i, data_point in enumerate(dataset.all_data):
        traj_id = data_point['trajectory_id']
        if traj_id not in trajectories:
            trajectories[traj_id] = []
        trajectories[traj_id].append(i)
    
    # Sort indices within each trajectory by timestamp
    for traj_id in trajectories:
        trajectories[traj_id].sort(key=lambda idx: dataset.all_data[idx]['timestamp'])
    
    # Get all trajectory IDs and shuffle for random split
    all_traj_ids = list(trajectories.keys())
    random.shuffle(all_traj_ids)
    
    # Split trajectory IDs into train/val/test
    n_train = int(0.7 * len(all_traj_ids))
    n_val = int(0.15 * len(all_traj_ids))
    
    train_traj_ids = all_traj_ids[:n_train]
    val_traj_ids = all_traj_ids[n_train:n_train+n_val]
    test_traj_ids = all_traj_ids[n_train+n_val:]
    
    # Create trajectory dictionaries for each split
    train_trajectories = {traj_id: trajectories[traj_id] for traj_id in train_traj_ids}
    val_trajectories = {traj_id: trajectories[traj_id] for traj_id in val_traj_ids}
    test_trajectories = {traj_id: trajectories[traj_id] for traj_id in test_traj_ids}
    
    # Get all indices for each split (for the standard DataLoader)
    train_indices = []
    for indices in train_trajectories.values():
        train_indices.extend(indices)
    
    val_indices = []
    for indices in val_trajectories.values():
        val_indices.extend(indices)
    
    test_indices = []
    for indices in test_trajectories.values():
        test_indices.extend(indices)
    
    print(f"Split dataset into:")
    print(f"  - Training: {len(train_indices)} timesteps from {len(train_trajectories)} trajectories")
    print(f"  - Validation: {len(val_indices)} timesteps from {len(val_trajectories)} trajectories") 
    print(f"  - Testing: {len(test_indices)} timesteps from {len(test_trajectories)} trajectories")
    
    # Create data loaders for basic encoder
    train_loader = DataLoader(
        Subset(dataset, train_indices), 
        batch_size=32, 
        shuffle=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), 
        batch_size=32, 
        shuffle=False,
        collate_fn=custom_collate
    )
    
    # Train basic encoder
    print("Training basic encoder...")
    basic_encoder = UnderwaterEncoder()
    trained_basic_encoder, train_losses_basic, val_losses_basic = train_encoder(
        basic_encoder, train_loader, val_loader, num_epochs=20, weight_decay=0.0001
    )
    
    # Train encoder with prior (using ground truth + noise approach)
    print("\nTraining encoder with prior (ground truth + noise)...")
    encoder_with_prior = UnderwaterEncoderWithPrior()
    
    # Use 5% noise scale - adjust as needed
    noise_scale = 0.1
    
    trained_encoder_with_prior, train_losses_prior, val_losses_prior = train_encoder_with_prior(
        encoder_with_prior, dataset, train_trajectories, val_trajectories, 
        num_epochs=60, noise_scale=noise_scale, weight_decay=0.0001
    )
    
    # Evaluate models
    print("\nEvaluating basic encoder...")
    basic_metrics = evaluate_model(trained_basic_encoder, dataset, test_trajectories)
    
    print("\nEvaluating encoder with prior...")
    prior_metrics = evaluate_model(
        trained_encoder_with_prior, dataset, test_trajectories, 
        with_prior=True, noise_scale=noise_scale
    )
    
    # Compare models
    print("\nModel Comparison:")
    print(f"Basic Encoder RMSE: {basic_metrics[1]:.4f}")
    print(f"Encoder With Prior RMSE: {prior_metrics[1]:.4f}")
    
    # Compare training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_basic, label='Basic Encoder')
    plt.plot(train_losses_prior, label='Encoder with Prior')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses_basic, label='Basic Encoder')
    plt.plot(val_losses_prior, label='Encoder with Prior')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("saved_models/model_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
