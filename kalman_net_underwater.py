import torch
import torch.nn as nn
import torch.nn.functional as func

class KalmanNetUnderwater(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def Build(self, m, n, f_function, H):
        """Initialize the KalmanNet for underwater tracking"""
        # Store system parameters
        self.m = m  # State dimension (2)
        self.n = n  # Observation dimension (1)
        
        # Set state evolution function
        self.f = lambda x: torch.matmul(f_function(x), x)
        
        # Set observation function
        self.H = H
        self.h = lambda x: torch.matmul(self.H, x)
        
        # Initialize Kalman Gain Network
        self.InitKGainNet()
        
    def InitKGainNet(self):
        """Initialize the neural network for Kalman gain estimation"""
        # Parameters
        self.seq_len_input = 1
        self.batch_size = 1
        in_mult = 5
        out_mult = 40
        
        # Initialize prior covariances
        self.prior_Q = 0.01 * torch.eye(self.m).to(self.device)
        self.prior_Sigma = 0.1 * torch.eye(self.m).to(self.device)
        self.prior_S = 0.1 * torch.eye(self.n).to(self.device)
        
        # GRU to track Q
        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m**2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)
        self.h_Q = torch.zeros(self.seq_len_input, self.batch_size, self.d_hidden_Q).to(self.device)
        
        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m**2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
        self.h_Sigma = torch.zeros(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).to(self.device)
        
        # GRU to track S
        self.d_input_S = self.n**2 + 2 * self.n * in_mult
        self.d_hidden_S = self.n**2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        self.h_S = torch.zeros(self.seq_len_input, self.batch_size, self.d_hidden_S).to(self.device)
        
        # Fully connected layers
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma, self.n**2),
            nn.ReLU()).to(self.device)
        
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_hidden_Sigma, (self.d_hidden_S + self.d_hidden_Sigma) * out_mult),
            nn.ReLU(),
            nn.Linear((self.d_hidden_S + self.d_hidden_Sigma) * out_mult, self.n * self.m)).to(self.device)
        
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.n * self.m, self.m**2),
            nn.ReLU()).to(self.device)
        
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma + self.m**2, self.d_hidden_Sigma),
            nn.ReLU()).to(self.device)
        
        self.FC5 = nn.Sequential(
            nn.Linear(self.m, self.m * in_mult),
            nn.ReLU()).to(self.device)
        
        self.FC6 = nn.Sequential(
            nn.Linear(self.m, self.m * in_mult),
            nn.ReLU()).to(self.device)
        
        self.FC7 = nn.Sequential(
            nn.Linear(2 * self.n, 2 * self.n * in_mult),
            nn.ReLU()).to(self.device)
    
    def InitSequence(self, m1x_0, m2x_0):
        """Initialize sequence with initial state and covariance"""
        # m1x_0: initial state mean [m, 1]
        # m2x_0: initial state covariance [m, m]
        
        # Make sure the state is in the right form
        if m1x_0.dim() == 1:
            m1x_0 = m1x_0.unsqueeze(1)  # Make it [m, 1]
            
        # Add batch dimension if not present
        if m1x_0.dim() == 2:
            m1x_0 = m1x_0.unsqueeze(0)  # Make it [1, m, 1]
            
        self.m1x_posterior = m1x_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior.clone()
        self.m1x_prior_previous = self.m1x_posterior.clone()
        self.y_previous = self.h(self.m1x_posterior)
        
        # Initialize hidden states
        self.init_hidden()
    
    def step_prior(self):
        """Compute prior state estimate"""
        # Apply state transition
        self.m1x_prior = self.f(self.m1x_posterior)
        
        # Predict observation
        self.m1y = self.h(self.m1x_prior)
    
    def KGain_step(self, obs_diff, obs_innov_diff, state_diff, update_diff):
        """Compute Kalman gain using neural network"""
        def expand_dim(x):
            expanded = torch.zeros(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, 0, :] = x
            return expanded
        
        # Prepare inputs for GRUs
        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        state_diff = expand_dim(state_diff)
        update_diff = expand_dim(update_diff)
        
        # Forward flow
        out_FC5 = self.FC5(update_diff)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)
        
        out_FC6 = self.FC6(state_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)
        
        out_FC1 = self.FC1(out_Sigma)
        
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)
        
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
        
        # Compute Kalman gain
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)
        
        # Backward flow
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)
        
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)
        
        # Update hidden state
        self.h_Sigma = out_FC4
        
        return out_FC2
    
    def step_KGain_est(self, y):
        """Estimate Kalman gain based on observation"""
        # Ensure y is in the right shape [batch, n, 1]
        if y.dim() == 1:
            y = y.unsqueeze(0).unsqueeze(2)  # [1, n, 1]
        elif y.dim() == 2:
            y = y.unsqueeze(2)  # [batch, n, 1]
            
        # Calculate differences
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        state_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)
        update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)
        
        # Normalize
        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        state_diff = func.normalize(state_diff, p=2, dim=1, eps=1e-12)
        update_diff = func.normalize(update_diff, p=2, dim=1, eps=1e-12)
        
        # Compute Kalman gain
        KG = self.KGain_step(obs_diff, obs_innov_diff, state_diff, update_diff)
        
        # Reshape Kalman gain to a matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))
    
    def KNet_step(self, y):
        """Perform one step of KalmanNet filtering"""
        # Ensure y is in the right shape [batch, n, 1]
        if y.dim() == 1:
            y = y.unsqueeze(0).unsqueeze(2)  # [1, n, 1]
        elif y.dim() == 2:
            y = y.unsqueeze(2)  # [batch, n, 1]
            
        # Compute prior
        self.step_prior()
        
        # Compute Kalman gain
        self.step_KGain_est(y)
        
        # Innovation
        dy = y - self.m1y
        
        # Update state estimate
        self.m1x_posterior_previous = self.m1x_posterior.clone()
        innovation = torch.bmm(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + innovation
        
        # Store values for next step
        self.m1x_prior_previous = self.m1x_prior.clone()
        self.y_previous = y
        
        return self.m1x_posterior
    
    def forward(self, y):
        """Forward pass for a single observation"""
        y = y.to(self.device)
        return self.KNet_step(y)
    
    def init_hidden(self):
        """Initialize hidden states with prior values"""
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1)
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1)
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1)
