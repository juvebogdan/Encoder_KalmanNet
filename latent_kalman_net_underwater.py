# latent_kalman_net_underwater.py
import torch
import torch.nn as nn
from kalman_net_underwater import KalmanNetUnderwater
from encoder_underwater import UnderwaterEncoderWithPrior
from model_Underwater import m, n, f_function, H

class LatentKalmanNetUnderwater(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 1 meter = 0.001 kilometers
        self.meters_to_km = 0.001
        
    def Build(self, encoder_path="saved_models/best_encoder_with_prior.pth"):
        """Initialize the Latent-KalmanNet for underwater tracking"""
        # Create and load the encoder - works in meters
        self.encoder = UnderwaterEncoderWithPrior().to(self.device)
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()  # Set to evaluation mode
        
        # Create the KalmanNet - will work in kilometers
        self.kalman_net = KalmanNetUnderwater()
        self.kalman_net.Build(m=m, n=n, f_function=f_function, H=H)
        
        print("Latent-KalmanNet built successfully")
    
    def InitSequence(self, m1x_0, m2x_0):
        """Initialize the KalmanNet sequence with values in kilometers"""
        # Convert initial distance from meters to kilometers
        km_state = m1x_0.clone()
        km_state[0, 0] = m1x_0[0, 0] * self.meters_to_km  # Convert distance to km
        
        # Initialize KalmanNet with km values
        self.kalman_net.InitSequence(km_state, m2x_0)
        self.prev_distance = None  # Track previous distance estimate in meters
        
    def forward(self, observation, timestamp=None):
        """
        Process one timestep of underwater channel data
        
        Args:
            observation: tuple of (basic_features, summary_features)
            timestamp: optional timestamp for logging/debugging
            
        Returns:
            Estimated state [distance (m), velocity (m/s)]
        """
        # Ensure inputs are on the right device
        basic_features, summary_features = observation
        basic_features = basic_features.to(self.device)
        summary_features = summary_features.to(self.device)
        
        # Create mask for encoder input
        mask = torch.ones(1, basic_features.shape[0], dtype=bool).to(self.device)
        
        # Prepare inputs for encoder
        basic_features = basic_features.unsqueeze(0)  # Add batch dimension
        summary_features = summary_features.unsqueeze(0)  # Add batch dimension
        
        # Generate prior using constant velocity model (in meters)
        if self.prev_distance is not None:
            # Get current Kalman state (in kilometers)
            current_state = self.kalman_net.m1x_posterior
            # Convert back to meters for encoder prior
            prior = torch.zeros(1, 1).to(self.device)
            prior[0, 0] = current_state[0, 0, 0] / self.meters_to_km  # Convert km to m
        else:
            # First timestep - initialize with a default value in meters
            prior = torch.tensor([[5000.0]]).to(self.device)  # 5 km in meters
        
        # Run encoder with prior to get distance estimate (in meters)
        # Note: encoder was trained with meters, so we don't convert prior
        distance_est = self.encoder((basic_features, mask, summary_features), prior)
        
        # Convert distance from meters to kilometers for KalmanNet
        km_distance = distance_est * self.meters_to_km
        
        # Update KalmanNet with the distance in kilometers
        km_state = self.kalman_net(km_distance)
        
        # Convert state back to meters for output
        m_state = km_state.clone()
        m_state[0, 0, 0] = km_state[0, 0, 0] / self.meters_to_km  # Convert distance back to meters
        m_state[0, 1, 0] = km_state[0, 1, 0] / self.meters_to_km  # Convert velocity back to m/s
        
        # Store distance for next iteration's prior (in meters)
        self.prev_distance = distance_est.item()
        
        return m_state
