import torch
import torch.nn as nn

class UnderwaterEncoder(nn.Module):
    def __init__(self):
        super(UnderwaterEncoder, self).__init__()
        
        # Process amplitude and delay pairs with a small MLP
        self.feature_extractor = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Process channel summary features
        self.summary_processor = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Global pooling for variable number of taps
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final regression layer
        self.regressor = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # Output distance estimate
        )

    def forward(self, x):
        if isinstance(x, tuple) and len(x) == 3:
            # Unpack with mask
            basic_features, mask, summary_features = x
        else:
            # For backward compatibility
            basic_features, summary_features = x
            mask = torch.ones(basic_features.shape[0], basic_features.shape[1], dtype=bool).to(basic_features.device)
        
        # Process amplitude-delay pairs
        batch_size, max_arrivals, _ = basic_features.shape
        reshaped = basic_features.reshape(-1, 2)
        processed = self.feature_extractor(reshaped)
        processed = processed.reshape(batch_size, max_arrivals, -1)
        
        # Apply mask to zero out padding
        mask = mask.unsqueeze(2).expand(-1, -1, processed.shape[2])
        processed = processed * mask.float()
        
        # Global pooling across arrivals dimension (only counting non-padded elements)
        processed = processed.transpose(1, 2)  # Shape: batch x features x arrivals
        
        # Sum and divide by count of non-padded elements (per batch item)
        arrivals_count = mask[:, :, 0].sum(dim=1).unsqueeze(1)  # Count of arrivals per batch
        pooled = processed.sum(dim=2) / (arrivals_count + 1e-10)  # Shape: batch x features
        
        # Process summary features
        summary = self.summary_processor(summary_features)
        
        # Combine features
        combined = torch.cat([pooled, summary], dim=1)
        
        # Final regression
        distance = self.regressor(combined)
        
        return distance


class UnderwaterEncoderWithPrior(nn.Module):
    def __init__(self):
        super(UnderwaterEncoderWithPrior, self).__init__()
        
        # Process amplitude and delay pairs with a small MLP
        self.feature_extractor = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Process channel summary features
        self.summary_processor = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Process prior state information (only distance)
        self.prior_processor = nn.Sequential(
            nn.Linear(1, 16),  # Takes state vector [distance]
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Global pooling for variable number of taps
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final regression layer
        self.regressor = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32),  # +16 for prior features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # Output distance estimate
        )
    
    def forward(self, x, prior):
        if isinstance(x, tuple) and len(x) == 3:
            # Unpack with mask
            basic_features, mask, summary_features = x
        else:
            # For backward compatibility
            basic_features, summary_features = x
            mask = torch.ones(basic_features.shape[0], basic_features.shape[1], dtype=bool).to(basic_features.device)
        
        # Process amplitude-delay pairs
        batch_size, max_arrivals, _ = basic_features.shape
        reshaped = basic_features.reshape(-1, 2)
        processed = self.feature_extractor(reshaped)
        processed = processed.reshape(batch_size, max_arrivals, -1)
        
        # Apply mask to zero out padding
        mask = mask.unsqueeze(2).expand(-1, -1, processed.shape[2])
        processed = processed * mask.float()
        
        # Global pooling across arrivals dimension (only counting non-padded elements)
        processed = processed.transpose(1, 2)  # Shape: batch x features x arrivals
        
        # Sum and divide by count of non-padded elements (per batch item)
        arrivals_count = mask[:, :, 0].sum(dim=1).unsqueeze(1)  # Count of arrivals per batch
        pooled = processed.sum(dim=2) / (arrivals_count + 1e-10)  # Shape: batch x features
        
        # Process summary features
        summary = self.summary_processor(summary_features)
        
        # Process prior state (just distance)
        prior_features = self.prior_processor(prior)
        
        # Combine all features
        combined = torch.cat([pooled, summary, prior_features], dim=1)
        
        # Final regression
        distance = self.regressor(combined)
        
        return distance
