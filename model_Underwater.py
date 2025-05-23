import torch
import math

# ───────────── units ─────────────
KM = 1e3              # 1 km = 1000 m
delta_t_sec = 1.0     # sampling period in seconds
delta_t = delta_t_sec / KM   # → 0.001 km per second

# ───────────── system dimensions ─────────────
m = 2   # [distance (km), velocity (km/s)]
n = 1   # learned scalar measurement (km)

# ───────────── initial state (will be overwritten at run-time) ─────────────
m1x_0 = torch.zeros(m, 1)          # km, km/s
m2x_0 = 0.1 * torch.eye(m)         # broad covariance

# ───────────── constant matrices ─────────────
H = torch.tensor([[1.0, 0.0]])     # observe distance only (km)

# process-noise variance in km²
real_q2 = 0.01 / KM**2             # was 0.01 m² → now 1.0 e-8 km²

# ───────────── transition function ─────────────
def f_function(x_prev: torch.Tensor) -> torch.Tensor:
    """
    Return the constant transition matrix F (2×2) on the same device
    as x_prev.  Ignoring x_prev because our model is linear time-invariant.
    """
    return torch.tensor([[1.0, delta_t],
                         [0.0, 1.0]], dtype=torch.float32, device=x_prev.device)

# ───────────── (optional) linear observation for reference ─────────────
def h_function(x: torch.Tensor) -> torch.Tensor:
    """Return the expected measurement (distance in km)."""
    return x[0:1]  # first row = distance component
