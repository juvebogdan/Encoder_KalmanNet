import torch
import torch.nn as nn

# small helper MLP ---------------------------------------------------------
class _MLP(nn.Sequential):
    def __init__(self, inp: int, hid: int):
        super().__init__(
            nn.Linear(inp, hid), nn.ReLU(), nn.LayerNorm(hid),
            nn.Linear(hid, hid), nn.ReLU(), nn.LayerNorm(hid)
        )

# --------------------------------------------------------------------------
class UnderwaterEncoder(nn.Module):
    """Baseline encoder *without* explicit prior."""
    def __init__(self):
        super().__init__()
        self.feature_extractor = _MLP(2, 64)
        self.summary_processor = _MLP(5, 16)
        self.regressor = nn.Sequential(
            nn.Linear(64 + 16, 32), nn.ReLU(), nn.LayerNorm(32), nn.Linear(32, 1)
        )

    def forward(self, x):
        if len(x) == 3:
            basic, mask, summary = x
        else:
            basic, summary = x
            mask = torch.ones(basic.size(0), basic.size(1), dtype=torch.bool, device=basic.device)

        B, A, _ = basic.shape
        proc = self.feature_extractor(basic.view(-1, 2)).view(B, A, 64)
        proc = proc * mask.unsqueeze(2).float()
        pooled = proc.sum(1) / mask.sum(1).clamp(min=1).unsqueeze(1)
        return self.regressor(torch.cat([pooled, self.summary_processor(summary)], 1))

# --------------------------------------------------------------------------
class UnderwaterEncoderWithPrior(nn.Module):
    """Encoder that consumes a *scalar* prior (e.g. previous‑step distance).

    Uses a residual connection:  output = prior + f(features).
    This makes the identity mapping (copy‑prior) trivial when noise=0 and
    allows the network to learn small corrections when noise>0.
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = _MLP(2, 64)
        self.summary_processor = _MLP(5, 16)
        self.prior_processor = _MLP(1, 16)
        self.regressor = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32), nn.ReLU(), nn.LayerNorm(32), nn.Linear(32, 1)
        )

    def forward(self, x, prior):
        if len(x) == 3:
            basic, mask, summary = x
        else:
            basic, summary = x
            mask = torch.ones(basic.size(0), basic.size(1), dtype=torch.bool, device=basic.device)

        B, A, _ = basic.shape
        proc = self.feature_extractor(basic.view(-1, 2)).view(B, A, 64)
        proc = proc * mask.unsqueeze(2).float()
        pooled = proc.sum(1) / mask.sum(1).clamp(min=1).unsqueeze(1)
        combined = torch.cat([
            pooled,
            self.summary_processor(summary),
            self.prior_processor(prior)
        ], 1)
        delta = self.regressor(combined)
        return prior + delta  # skip connection
