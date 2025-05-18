import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset

class UnderwaterDataset(Dataset):
    def __init__(self, data_dir):
        self.all_data = []  # Will store processed data
        
        # Find all CSV files
        files = [f for f in os.listdir(data_dir) if f.startswith('all_arrivals') and f.endswith('.csv')]
        files.sort()
        
        print(f"Found {len(files)} data files")
        
        trajectory_counter = 0
        
        for file in files:
            file_path = os.path.join(data_dir, file)
            print(f"Processing file: {file_path}")
            
            # Read the file
            df = pd.read_csv(file_path)
            
            # Get all timestamps in order as they appear in the file
            all_timestamps = df['Timestamp'].values
            
            # Find where timestamps reset from 199 back to 0
            trajectory_starts = [0]  # First trajectory starts at row 0
            
            for i in range(1, len(all_timestamps)):
                if all_timestamps[i] == 0 and all_timestamps[i-1] == 199:
                    trajectory_starts.append(i)
            
            # Add end of file as the last boundary
            trajectory_starts.append(len(all_timestamps))
            
            print(f"Detected {len(trajectory_starts)-1} trajectories in file")
            
            # Process each trajectory
            for traj_idx in range(len(trajectory_starts)-1):
                start_row = trajectory_starts[traj_idx]
                end_row = trajectory_starts[traj_idx+1]
                
                traj_data = df.iloc[start_row:end_row]
                
                # Check if this trajectory has all 200 timestamps
                unique_timestamps = traj_data['Timestamp'].unique()
                
                if len(unique_timestamps) == 200:
                    print(f"Processing trajectory {trajectory_counter}")
                    
                    # Process each timestamp for this trajectory
                    trajectory_timesteps = []
                    
                    for timestamp in range(200):  # 0-199
                        # Get data for this timestamp in this trajectory
                        timestamp_data = traj_data[traj_data['Timestamp'] == timestamp]
                        
                        # Process and store
                        timestep_data = self._process_timestep(timestamp_data, timestamp, trajectory_counter)
                        trajectory_timesteps.append(timestep_data)
                    
                    # Add complete trajectory
                    self.all_data.extend(trajectory_timesteps)
                    trajectory_counter += 1
                else:
                    print(f"Skipping trajectory with {len(unique_timestamps)} timestamps (expected 200)")
            
        print(f"Total trajectories loaded: {trajectory_counter}")
        print(f"Total dataset size: {len(self.all_data)} timesteps")

        # Initialize variables to track feature statistics
        self.amplitude_mean = 0
        self.amplitude_std = 1
        self.delay_mean = 0
        self.delay_std = 1
        self.summary_means = np.zeros(5)  # For 5 summary features
        self.summary_stds = np.ones(5)
        self.distance_mean = 0
        self.distance_std = 1
        
        # Compute normalization statistics
        self.compute_normalization_stats()

    def compute_normalization_stats(self):
        """Compute mean and std for each feature for normalization"""
        print("Computing normalization statistics...")
        
        # Collect all amplitudes, delays, and distances
        all_amplitudes = []
        all_delays = []
        all_distances = []
        all_summary_features = []
        
        for data_point in self.all_data:
            all_amplitudes.extend(data_point['amplitudes'])
            all_delays.extend(data_point['delays'])
            all_distances.append(data_point['distance'])
            
            # Collect summary features
            summary = [
                data_point['num_taps'],
                data_point['avg_tap_power'],
                data_point['delay_spread'],
                data_point['avg_path_delay'],
                data_point['power_weighted_avg_delay']
            ]
            all_summary_features.append(summary)
        
        # Convert to numpy arrays for efficient computation
        all_amplitudes = np.array(all_amplitudes)
        all_delays = np.array(all_delays)
        all_distances = np.array(all_distances)
        all_summary_features = np.array(all_summary_features)
        
        # Compute statistics for amplitude and delay
        self.amplitude_mean = np.mean(all_amplitudes)
        self.amplitude_std = np.std(all_amplitudes) + 1e-6  # Add small value to avoid division by zero
        
        self.delay_mean = np.mean(all_delays)
        self.delay_std = np.std(all_delays) + 1e-6
        
        # Compute statistics for summary features
        self.summary_means = np.mean(all_summary_features, axis=0)
        self.summary_stds = np.std(all_summary_features, axis=0) + 1e-6
        
        # Compute statistics for distance (target)
        self.distance_mean = np.mean(all_distances)
        self.distance_std = np.std(all_distances) + 1e-6
        
        print(f"Amplitude: mean={self.amplitude_mean:.4f}, std={self.amplitude_std:.4f}")
        print(f"Delay: mean={self.delay_mean:.4f}, std={self.delay_std:.4f}")
        print(f"Distance: mean={self.distance_mean:.4f}, std={self.distance_std:.4f}")
        print(f"Summary features means: {self.summary_means}")
        print(f"Summary features stds: {self.summary_stds}")
    
    def _process_timestep(self, timestamp_data, timestamp, trajectory_id):
        # Extract amplitude and delay pairs
        amplitudes = timestamp_data['Amplitude'].values
        delays = timestamp_data['Delay'].values
        
        # Get target (distance)
        distance = timestamp_data['Distance_GT'].values[0]
        
        # Calculate channel features
        num_taps = len(amplitudes)
        avg_tap_power = np.mean(np.abs(amplitudes))
        
        # Power weighting
        weights = np.abs(amplitudes) ** 2
        weight_sum = np.sum(weights)
        
        # Using absolute delays (no min_delay subtraction)
        if weight_sum > 0:
            # Power weighted average delay using absolute delays
            power_weighted_avg_delay = np.sum(weights * delays) / weight_sum
            
            # For delay spread, calculate deviation from the weighted average
            delay_spread = np.sqrt(np.sum(weights * (delays - power_weighted_avg_delay) ** 2) / weight_sum)
        else:
            print("Warning: Weight sum is zero")
            power_weighted_avg_delay = 0
            delay_spread = 0
        
        # Average path delay using absolute delays and amplitude weighting
        if np.sum(np.abs(amplitudes)) > 0:
            avg_path_delay = np.sum(np.abs(amplitudes) * delays) / np.sum(np.abs(amplitudes))
        else:
            avg_path_delay = 0
        
        # Store the processed data
        return {
            'amplitudes': amplitudes,
            'delays': delays,
            'distance': distance,
            'timestamp': timestamp,
            'trajectory_id': trajectory_id,
            'num_taps': num_taps,
            'avg_tap_power': avg_tap_power,
            'delay_spread': delay_spread,
            'avg_path_delay': avg_path_delay,
            'power_weighted_avg_delay': power_weighted_avg_delay
        }
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        # Get the data for this timestep
        timestep_data = self.all_data[idx]
        
        # Normalize amplitude and delay values
        amplitudes = (timestep_data['amplitudes'] - self.amplitude_mean) / self.amplitude_std
        delays = (timestep_data['delays'] - self.delay_mean) / self.delay_std
        
        # Create normalized basic features
        amplitude_delay_pairs = np.vstack((amplitudes, delays)).T
        basic_features = torch.FloatTensor(amplitude_delay_pairs)
        
        # Create normalized summary features tensor
        summary_raw = np.array([
            timestep_data['num_taps'],
            timestep_data['avg_tap_power'],
            timestep_data['delay_spread'],
            timestep_data['avg_path_delay'],
            timestep_data['power_weighted_avg_delay']
        ])
        
        # Normalize summary features
        summary_normalized = (summary_raw - self.summary_means) / self.summary_stds
        summary_features = torch.FloatTensor(summary_normalized)
        
        # Normalize target distance
        normalized_distance = (timestep_data['distance'] - self.distance_mean) / self.distance_std
        target = torch.FloatTensor([normalized_distance])
        
        return (basic_features, summary_features), target
    
    def denormalize_distance(self, normalized_distance):
        """Convert normalized distance back to original scale"""
        return normalized_distance * self.distance_std + self.distance_mean

    def normalize_distance(self, distance):
        """Convert a distance value to normalized scale"""
        return (distance - self.distance_mean) / self.distance_std

    def save_normalization_params(self, filepath):
        """Save normalization parameters to a file"""
        params = {
            'amplitude_mean': self.amplitude_mean,
            'amplitude_std': self.amplitude_std,
            'delay_mean': self.delay_mean,
            'delay_std': self.delay_std,
            'summary_means': self.summary_means,
            'summary_stds': self.summary_stds,
            'distance_mean': self.distance_mean,
            'distance_std': self.distance_std
        }
        np.save(filepath, params)
        print(f"Saved normalization parameters to {filepath}")

    def load_normalization_params(self, filepath):
        """Load normalization parameters from a file"""
        params = np.load(filepath, allow_pickle=True).item()
        self.amplitude_mean = params['amplitude_mean']
        self.amplitude_std = params['amplitude_std']
        self.delay_mean = params['delay_mean']
        self.delay_std = params['delay_std']
        self.summary_means = params['summary_means']
        self.summary_stds = params['summary_stds']
        self.distance_mean = params['distance_mean']
        self.distance_std = params['distance_std']
        print(f"Loaded normalization parameters from {filepath}")
