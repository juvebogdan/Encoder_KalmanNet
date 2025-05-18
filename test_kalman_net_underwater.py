# test_kalman_net_underwater.py
import torch
import numpy as np
from kalman_net_underwater import KalmanNetUnderwater
from model_Underwater import f_function, H, m, n, m1x_0, m2x_0

def test_kalman_net():
    print("Testing KalmanNetUnderwater class...")
    
    # Create instance
    knet = KalmanNetUnderwater()
    print("KalmanNetUnderwater instance created successfully.")
    
    # Build model
    try:
        knet.Build(m=m, n=n, f_function=f_function, H=H)
        print(f"Build successful with m={m}, n={n}")
    except Exception as e:
        print(f"Error during Build: {e}")
        return
    
    # Initialize sequence
    try:
        # Make sure m1x_0 is in correct shape
        if m1x_0.dim() == 1:
            m1x_0_tensor = m1x_0.clone().unsqueeze(1)  # [m, 1]
        else:
            m1x_0_tensor = m1x_0.clone()
            
        print(f"m1x_0 shape: {m1x_0_tensor.shape}")
        print(f"m2x_0 shape: {m2x_0.shape}")
        
        knet.InitSequence(m1x_0=m1x_0_tensor, m2x_0=m2x_0)
        print("Sequence initialization successful")
    except Exception as e:
        print(f"Error during InitSequence: {e}")
        return
    
    # Test forward pass with dummy data
    try:
        # Create a dummy observation (distance estimate)
        dummy_obs = torch.tensor([5.0], dtype=torch.float32)
        
        print(f"Input observation shape: {dummy_obs.shape}")
        
        # Pass through network
        output = knet(dummy_obs)
        
        print(f"Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output value: {output}")
        
        # Try a second forward pass to test sequential processing
        dummy_obs2 = torch.tensor([5.2], dtype=torch.float32)
        output2 = knet(dummy_obs2)
        print(f"Second forward pass successful!")
        print(f"Output shape: {output2.shape}")
        print(f"Output value: {output2}")
        
        # Print the Kalman gain
        print(f"Calculated Kalman gain: {knet.KGain}")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("All tests passed!")

if __name__ == "__main__":
    test_kalman_net()
