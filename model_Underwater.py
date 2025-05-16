import torch
import numpy as np

# System dimensions
m = 2  # State dimensions (distance, velocity)
n = 1  # Observation dimension (distance estimate from encoder)

# Initial state mean and covariance
m1x_0 = torch.zeros(m, 1)  # Will be set during initialization with first observation
m2x_0 = 0.1 * torch.eye(m)  # Initial uncertainty

# Time step
delta_t = 1.0  # Update based on your data sampling rate

# Observation matrix (used by KalmanNet architecture)
H = torch.tensor([[1.0, 0.0]])  # Only extract distance from state

# Parameters for the model
real_q2 = 0.01  # Process noise variance
y_size = 2      # Size of basic observation (amplitude, delay)
d = 1           # Dimension of encoder output (distance estimate)

def f_function(x_prev):
    """
    State transition function for constant velocity model.
    x_prev: Previous state [distance, velocity]
    Returns: State transition matrix F
    """
    # Constant velocity model
    F = torch.tensor([
        [1.0, delta_t],  # distance = distance + velocity*delta_t
        [0.0, 1.0]       # velocity = velocity (constant)
    ], dtype=torch.float32)
    
    return F

def h_function(state):
    """
    Simple observation function - extracts distance component
    (Only used for architecture compatibility)
    """
    return state[0].item()  # Return only the distance component
