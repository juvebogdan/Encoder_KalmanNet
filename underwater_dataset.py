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
    
    def _process_timestep(self, timestamp_data, timestamp, trajectory_id):
        """Process data for a single timestep in a trajectory"""
        # Extract amplitude and delay pairs
        amplitudes = timestamp_data['Amplitude'].values
        delays = timestamp_data['Delay'].values
        
        # Get target (distance) - should be the same for all rows at this timestep
        distance = timestamp_data['Distance_GT'].values[0]
        
        # Calculate channel features
        # 1. Number of channel taps
        num_taps = len(amplitudes)
        
        # 2. Average tap power
        avg_tap_power = np.mean(np.abs(amplitudes))
        
        # 4. Relative RMS delay spread
        min_delay = np.min(delays)
        weights = np.abs(amplitudes) ** 2
        weight_sum = np.sum(weights)
        
        # Avoid division by zero
        if weight_sum > 0:
            delay_spread = np.sqrt(np.sum(weights * (delays - min_delay) ** 2) / weight_sum)
        else:
            delay_spread = 0
        
        # 5. Average path delay
        if np.sum(np.abs(amplitudes)) > 0:
            avg_path_delay = np.sum(np.abs(amplitudes) * (delays - min_delay)) / np.sum(np.abs(amplitudes))
        else:
            avg_path_delay = 0
        
        # Store the processed data for this timestep
        return {
            'amplitudes': amplitudes,
            'delays': delays,
            'distance': distance,
            'timestamp': timestamp,
            'trajectory_id': trajectory_id,
            'num_taps': num_taps,
            'avg_tap_power': avg_tap_power,
            'delay_spread': delay_spread,
            'avg_path_delay': avg_path_delay
        }
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        # Get the data for this timestep
        timestep_data = self.all_data[idx]
        
        # Basic features: amplitude and delay pairs
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
        
        # Target distance
        target = torch.FloatTensor([timestep_data['distance']])
        
        return (basic_features, summary_features), target
