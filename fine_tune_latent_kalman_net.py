# fine_tune_latent_kalman_net.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

from latent_kalman_net_underwater import LatentKalmanNetUnderwater
from underwater_dataset import UnderwaterDataset
from model_Underwater import m, n, m1x_0, m2x_0, delta_t

def fine_tune_latent_kalman_net():
    print("Fine-tuning Latent-KalmanNet for underwater tracking...")
    
    # Create save directory
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = UnderwaterDataset('data')
    
    # Group data by trajectory
    trajectories = {}
    for i, data_point in enumerate(dataset.all_data):
        traj_id = data_point['trajectory_id']
        if traj_id not in trajectories:
            trajectories[traj_id] = []
        trajectories[traj_id].append(i)
    
    # Sort indices within each trajectory by timestamp
    for traj_id in trajectories:
        trajectories[traj_id].sort(key=lambda idx: dataset.all_data[idx]['timestamp'])
    
    # Split into train/val/test
    all_traj_ids = list(trajectories.keys())
    random.seed(42)  # For reproducibility
    random.shuffle(all_traj_ids)
    
    n_train = int(0.7 * len(all_traj_ids))
    n_val = int(0.15 * len(all_traj_ids))
    
    train_traj_ids = all_traj_ids[:n_train]
    val_traj_ids = all_traj_ids[n_train:n_train+n_val]
    test_traj_ids = all_traj_ids[n_train+n_val:]
    
    train_trajectories = {traj_id: trajectories[traj_id] for traj_id in train_traj_ids}
    val_trajectories = {traj_id: trajectories[traj_id] for traj_id in val_traj_ids}
    test_trajectories = {traj_id: trajectories[traj_id] for traj_id in test_traj_ids}
    
    print(f"Training on {len(train_trajectories)} trajectories")
    print(f"Validating on {len(val_trajectories)} trajectories")
    print(f"Testing on {len(test_trajectories)} trajectories")
    
    # Create model
    model = LatentKalmanNetUnderwater()
    model.Build()
    
    # Load the pretrained model (KalmanNet was already trained with fixed encoder)
    model.load_state_dict(torch.load(f"{save_dir}/best_latent_kalman_net.pth"))
    model.to(device)
    
    # Set better initial state covariance
    # For km scale, use smaller covariance values
    better_m2x_0 = torch.tensor([[0.1, 0.0], [0.0, 0.01]]).to(device)  # 100m×100m -> 0.1km×0.1km
    
    # Training parameters
    num_epochs = 10  # Fewer epochs for fine-tuning
    encoder_lr = 0.00001  # Much smaller learning rate for encoder (already trained)
    kalman_lr = 0.00005  # Smaller learning rate for KalmanNet (already trained)
    batch_size = 8
    best_val_loss = float('inf')
    meters_to_km = 0.001  # For debugging prints
    
    # Enable training for both encoder and KalmanNet
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    # Use separate parameter groups with different learning rates
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.kalman_net.parameters(), 'lr': kalman_lr}
    ])
    
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    
    def process_trajectory(model, indices, get_loss=True, debug=False):
        """Process a full trajectory and return outputs and optionally loss"""
        # Get first observation to initialize state
        first_idx = indices[0]
        first_features, first_target = dataset[first_idx]
        
        # Initialize state with first observation - in meters
        initial_state = torch.zeros(m, 1).to(device)
        initial_state[0, 0] = first_target.item()  # Set distance in meters
        initial_state[1, 0] = 0.0  # Set initial velocity to 0
        
        if debug:
            print(f"Initial state (m): {initial_state.squeeze().numpy()}")
            print(f"Initial state (km): {initial_state[0,0].item() * meters_to_km:.4f} km")
        
        # Initialize sequence
        model.InitSequence(initial_state, better_m2x_0)
        
        # Arrays to store results
        predictions = []
        targets = []
        
        # Process the full trajectory at once
        for i, idx in enumerate(indices):
            # Skip first observation used for initialization
            if i == 0:
                continue
                
            features, target = dataset[idx]
            target = target.to(device)
            
            if debug and i <= 5:
                print(f"\nStep {i}:")
                print(f"Target (m): {target.item():.4f}")
                print(f"Target (km): {target.item() * meters_to_km:.6f}")
            
            # Forward pass
            try:
                state = model(features)
                
                if debug and i <= 5:
                    print(f"Model output (m): {state[0, 0, 0].item():.4f}, velocity (m/s): {state[0, 1, 0].item():.4f}")
                    print(f"Internal KalmanNet state (km): {model.kalman_net.m1x_posterior[0, 0, 0].item():.6f}")
                    print(f"Kalman gain: {model.kalman_net.KGain[0, :, 0].detach().cpu().numpy()}")
                    
                # Store results
                predictions.append(state[0, 0, 0])
                targets.append(target)
                
            except Exception as e:
                print(f"Error during forward pass at step {i}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Convert to tensors
        if predictions:
            predictions = torch.stack(predictions)
            targets = torch.stack(targets)
            
            # Make sure dimensions match
            if predictions.dim() != targets.dim():
                predictions = predictions.view(predictions.shape[0], 1)
                
            # Calculate loss if requested
            if get_loss and len(predictions) > 0:
                try:
                    loss = criterion(predictions, targets)
                    
                    if debug:
                        print(f"Loss value: {loss.item():.6f}")
                        
                    return loss, predictions.detach().cpu().numpy(), targets.detach().cpu().numpy()
                except Exception as e:
                    print(f"Error calculating loss: {e}")
                    return None, predictions.detach().cpu().numpy(), targets.detach().cpu().numpy()
                
        return None, [], []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Shuffle training trajectories
        train_traj_list = list(train_trajectories.keys())
        random.shuffle(train_traj_list)
        
        # Process trajectories in batches
        batch_losses = []
        
        # Process in batches
        for i in range(0, len(train_traj_list), batch_size):
            batch_traj_ids = train_traj_list[i:i+batch_size]
            
            # Zero gradients at batch start
            optimizer.zero_grad()
            
            # Process each trajectory in batch
            batch_loss = 0
            valid_trajectories = 0
            
            # Debug first batch in first epoch
            debug_this_batch = (epoch == 0 and i == 0)
            
            for traj_id in batch_traj_ids:
                indices = train_trajectories[traj_id]
                
                # Skip very short trajectories
                if len(indices) < 5:
                    continue
                    
                # Process trajectory with debug for first trajectory in first batch of first epoch
                debug_this_traj = debug_this_batch and (traj_id == batch_traj_ids[0])
                loss, _, _ = process_trajectory(model, indices, debug=debug_this_traj)
                
                if loss is not None:
                    # Print loss value for debugging
                    if debug_this_traj:
                        print(f"Trajectory {traj_id} loss: {loss.item():.6f}")
                    
                    batch_loss += loss
                    valid_trajectories += 1
            
            # Update only if we have valid trajectories
            if valid_trajectories > 0:
                # Average the loss
                batch_loss = batch_loss / valid_trajectories
                
                # Backward pass on batch loss
                batch_loss.backward()
                
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update model
                optimizer.step()
                
                # Store batch loss
                batch_losses.append(batch_loss.item())
                
            # Print progress
            if (i // batch_size) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size + 1}/{len(train_traj_list)//batch_size + 1}")
                if batch_losses:
                    print(f"Current average batch loss: {sum(batch_losses)/len(batch_losses):.6f}")
        
        # Average training loss for epoch
        if batch_losses:
            epoch_train_loss = sum(batch_losses) / len(batch_losses)
            train_losses.append(epoch_train_loss)
        else:
            epoch_train_loss = float('inf')
            train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_losses_epoch = []
        
        with torch.no_grad():
            for traj_id in val_traj_ids:
                indices = val_trajectories[traj_id]
                
                # Skip short trajectories
                if len(indices) < 5:
                    continue
                
                # Process trajectory
                loss, _, _ = process_trajectory(model, indices)
                
                if loss is not None:
                    val_losses_epoch.append(loss.item())
        
        # Average validation loss
        if val_losses_epoch:
            epoch_val_loss = sum(val_losses_epoch) / len(val_losses_epoch)
            val_losses.append(epoch_val_loss)
        else:
            epoch_val_loss = float('inf')
            val_losses.append(epoch_val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_fine_tuned_latent_kalman_net.pth")
            print(f"Saved new best model with val_loss: {epoch_val_loss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Fine-Tuning: Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/fine_tuning_curve.png")
    plt.close()
    
    print(f"Fine-tuning complete. Best validation loss: {best_val_loss:.6f}")
    
    # Test on the best model
    model.load_state_dict(torch.load(f"{save_dir}/best_fine_tuned_latent_kalman_net.pth"))
    model.eval()
    
    # Process a few test trajectories for visualization
    with torch.no_grad():
        for i, traj_id in enumerate(test_traj_ids[:5]):
            indices = test_trajectories[traj_id]
            
            # Skip short trajectories
            if len(indices) < 5:
                continue
            
            # Process trajectory
            loss, predictions, targets = process_trajectory(model, indices)
            
            # Plot results if we have data
            if len(predictions) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(targets, 'k-', label='Ground Truth')
                plt.plot(predictions, 'b-', label='Fine-tuned Latent-KalmanNet')
                plt.xlabel('Time Step')
                plt.ylabel('Distance (m)')
                plt.title(f'Test Trajectory {traj_id}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{save_dir}/fine_tuned_test_trajectory_{traj_id}.png")
                plt.close()
                
                # Print stats for this trajectory
                if loss is not None:
                    mse = loss.item()
                    rmse = np.sqrt(mse)
                    print(f"Trajectory {traj_id} - MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    
    print("Fine-tuning and evaluation complete!")

if __name__ == "__main__":
    fine_tune_latent_kalman_net()
