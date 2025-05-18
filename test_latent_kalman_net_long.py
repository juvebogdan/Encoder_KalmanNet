# test_long_trajectory.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from latent_kalman_net_underwater import LatentKalmanNetUnderwater
from model_Underwater import m, n, m1x_0, m2x_0

def load_long_trajectory(file_path, transmitter_name="Alice"):
    """
    Load and preprocess the long trajectory data for a specific transmitter
    """
    print(f"Loading data for transmitter {transmitter_name} from {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter for the specified transmitter
    df = df[df['Transmitter_Name'] == transmitter_name]
    
    # Get all timestamps in order
    all_timestamps = df['Timestamp'].unique()
    all_timestamps.sort()
    
    print(f"Found {len(all_timestamps)} unique timestamps for {transmitter_name}")
    
    # Create a list to store processed data for each timestamp
    processed_data = []
    ground_truth_distances = []
    
    # Process each timestamp
    for timestamp in tqdm(all_timestamps, desc="Processing timestamps"):
        # Get data for this timestamp
        timestamp_data = df[df['Timestamp'] == timestamp]
        
        # Calculate ground truth distance using the Euclidean distance formula
        # Convert to meters since our model works in meters
        tx_x = timestamp_data['Transmitter_X(km)'].iloc[0] * 1000  # km to m
        tx_y = timestamp_data['Transmitter_Y(km)'].iloc[0] * 1000  # km to m
        rx_x = timestamp_data['Receiver_X(km)'].iloc[0] * 1000     # km to m
        rx_y = timestamp_data['Receiver_Y(km)'].iloc[0] * 1000     # km to m
        
        distance = math.sqrt((tx_x - rx_x)**2 + (tx_y - rx_y)**2)
        ground_truth_distances.append(distance)
        
        # Extract amplitude and delay pairs
        amplitudes = timestamp_data['Amplitude'].values
        delays = timestamp_data['Delay(s)'].values
        
        # Calculate channel features
        num_taps = len(amplitudes)
        avg_tap_power = np.mean(np.abs(amplitudes))
        
        # Calculate delay spread
        min_delay = np.min(delays)
        weights = np.abs(amplitudes) ** 2
        weight_sum = np.sum(weights)
        
        # Avoid division by zero
        if weight_sum > 0:
            delay_spread = np.sqrt(np.sum(weights * (delays - min_delay) ** 2) / weight_sum)
        else:
            delay_spread = 0
        
        # Calculate average path delay
        if np.sum(np.abs(amplitudes)) > 0:
            avg_path_delay = np.sum(np.abs(amplitudes) * (delays - min_delay)) / np.sum(np.abs(amplitudes))
        else:
            avg_path_delay = 0
            
        # Store processed data
        processed_data.append({
            'amplitudes': amplitudes,
            'delays': delays,
            'distance': distance,
            'timestamp': timestamp,
            'num_taps': num_taps,
            'avg_tap_power': avg_tap_power,
            'delay_spread': delay_spread,
            'avg_path_delay': avg_path_delay
        })
    
    return processed_data, ground_truth_distances, all_timestamps

def prepare_model_input(timestep_data):
    """Prepare model input from processed data point"""
    # Create basic features: amplitude and delay pairs
    amplitude_delay_pairs = np.vstack((
        timestep_data['amplitudes'],
        timestep_data['delays']
    )).T
    
    # Create tensor from amplitude-delay pairs
    basic_features = torch.FloatTensor(amplitude_delay_pairs)
    
    # Create summary features tensor
    summary_features = torch.FloatTensor([
        timestep_data['num_taps'],
        timestep_data['avg_tap_power'],
        timestep_data['delay_spread'],
        timestep_data['avg_path_delay']
    ])
    
    return basic_features, summary_features

def test_long_trajectory():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model setup
    print("Loading fine-tuned model...")
    model = LatentKalmanNetUnderwater()
    model.Build(encoder_path="saved_models/best_encoder_with_prior.pth")
    model.load_state_dict(torch.load("saved_models/best_fine_tuned_latent_kalman_net.pth", 
                                     map_location=device))
    model.to(device)
    model.eval()
    
    # If fine-tuned model not found, try the regular trained model
    if "AttributeError" in str(model):
        print("Fine-tuned model not found, trying regular trained model...")
        model = LatentKalmanNetUnderwater()
        model.Build(encoder_path="saved_models/best_encoder_with_prior.pth")
        model.load_state_dict(torch.load("saved_models/best_latent_kalman_net.pth", 
                                         map_location=device))
        model.to(device)
        model.eval()
    
    # Load data
    processed_data, ground_truth_distances, timestamps = load_long_trajectory(
        "data/all_arrivals_long.csv", transmitter_name="Alice")
    
    # Better initial state covariance
    better_m2x_0 = torch.tensor([[0.1, 0.0], [0.0, 0.01]]).to(device)
    
    # Initialize model with first observation
    first_data = processed_data[0]
    initial_state = torch.zeros(m, 1).to(device)
    initial_state[0, 0] = first_data['distance']  # Set distance in meters
    initial_state[1, 0] = 0.0  # Set initial velocity to 0
    
    print(f"Initial state: {initial_state.squeeze().numpy()}")
    model.InitSequence(initial_state, better_m2x_0)
    
    # Run the model on all timesteps
    predictions = []
    encoder_outputs = []
    
    print("Running model inference...")
    with torch.no_grad():
        for i, data_point in enumerate(tqdm(processed_data)):
            # Skip first observation (used for initialization)
            if i == 0:
                predictions.append(data_point['distance'])
                encoder_outputs.append(data_point['distance'])
                continue
                
            # Prepare model input
            basic_features, summary_features = prepare_model_input(data_point)
            
            # Get model prediction
            try:
                state = model((basic_features, summary_features))
                predictions.append(state[0, 0, 0].item())  # Extract distance
                encoder_outputs.append(model.prev_distance)  # Store encoder output
            except Exception as e:
                print(f"Error at timestep {i}: {e}")
                # Use previous prediction as fallback
                if predictions:
                    predictions.append(predictions[-1])
                    encoder_outputs.append(encoder_outputs[-1])
                else:
                    predictions.append(data_point['distance'])
                    encoder_outputs.append(data_point['distance'])
    
    # Calculate error metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth_distances)
    encoder_outputs = np.array(encoder_outputs)
    
    # MSE
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)
    
    encoder_mse = np.mean((encoder_outputs - ground_truth) ** 2)
    encoder_rmse = np.sqrt(encoder_mse)
    
    print(f"Latent-KalmanNet - MSE: {mse:.2f}, RMSE: {rmse:.2f}")
    print(f"Encoder Only - MSE: {encoder_mse:.2f}, RMSE: {encoder_rmse:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(timestamps, ground_truth, 'k-', linewidth=2, label='Ground Truth')
    plt.plot(timestamps, predictions, 'b-', linewidth=2, label='Latent-KalmanNet')
    plt.plot(timestamps, encoder_outputs, 'r--', linewidth=1.5, label='Encoder Only')
    plt.xlabel('Timestamp')
    plt.ylabel('Distance (m)')
    plt.title(f'Alice Trajectory: Ground Truth vs Prediction\nLatent-KalmanNet RMSE: {rmse:.2f}m, Encoder RMSE: {encoder_rmse:.2f}m')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("long_trajectory_alice.png", dpi=300)
    
    # Plot the error over time
    plt.figure(figsize=(12, 6))
    knet_error = np.abs(predictions - ground_truth)
    encoder_error = np.abs(encoder_outputs - ground_truth)
    plt.plot(timestamps, knet_error, 'b-', label='Latent-KalmanNet Error')
    plt.plot(timestamps, encoder_error, 'r--', label='Encoder Error')
    plt.xlabel('Timestamp')
    plt.ylabel('Absolute Error (m)')
    plt.title('Absolute Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("error_over_time.png", dpi=300)
    
    print("Testing complete. Results saved to long_trajectory_alice.png and error_over_time.png")

if __name__ == "__main__":
    test_long_trajectory()
