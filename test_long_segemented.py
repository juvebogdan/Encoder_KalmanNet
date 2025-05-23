#!/usr/bin/env python3
"""
Evaluate a long trajectory using saved train-time norms and segmentation
=======================================================================
• Does *not* reload the full dataset, only the saved norm_stats.npz
• Segments into 200-step blocks to match training
• Filters for Alice→Apex and normalizes exactly as during training
"""

import argparse, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from encoder_underwater     import UnderwaterEncoderWithPrior
from kalman_net_underwater  import KalmanNetUnderwater
from model_Underwater       import f_function, H, m, n, m2x_0   # km model

# ───────── constants ─────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KM          = 1e3
BLOCK_LEN   = 200
PLOT_DIR    = Path("plots"); PLOT_DIR.mkdir(exist_ok=True)

# ───────── helpers ─────────
def to_raw(norm, μ, σ): 
    return (norm * σ + μ) / KM

def to_norm(km, μ, σ):   
    return ((km * KM) - μ) / σ

# ───────── CLI ─────────
pa = argparse.ArgumentParser()
pa.add_argument("--csv",        default="data/all_arrivals_long.csv")
pa.add_argument("--norm_stats", default="saved_models/norm_stats.npz")
args = pa.parse_args()

# ───────── 1. load normalization stats ─────────
stats = np.load(args.norm_stats)
dist_mu, dist_sigma = stats["dist_mean"].item(), stats["dist_std"].item()
amp_mu, amp_sigma   = stats["amp_mean"].item(),  stats["amp_std"].item()
dly_mu, dly_sigma   = stats["dly_mean"].item(),  stats["dly_std"].item()
summ_mu             = stats["summ_mean"]   # shape (5,)
summ_sigma          = stats["summ_std"]    # shape (5,)

print(f"[stats] distance μ={dist_mu:.1f} m  σ={dist_sigma:.1f} m")

# ───────── 2. load models ─────────
enc = UnderwaterEncoderWithPrior().to(DEVICE)
enc.load_state_dict(torch.load("saved_models/enc_finetuned.pth", map_location=DEVICE))
enc.eval()

knet = KalmanNetUnderwater().to(DEVICE)
knet.Build(m=m, n=n, f_function=f_function, H=H.to(DEVICE))
knet.load_state_dict(torch.load("saved_models/knet_best.pth", map_location=DEVICE))
knet.eval()

# ───────── 3. read & filter CSV ─────────
csv = pd.read_csv(args.csv)
csv = csv[(csv["Transmitter_Name"]=="Alice") & (csv["Sensor_ID"]=="Apex")]
if csv.empty:
    raise RuntimeError("No Alice→Apex rows found")
ts_list = sorted(csv["Timestamp"].unique())
T = len(ts_list)
print(f"[eval] trajectory length = {T} steps (first ts={ts_list[0]})")

# 3-D Euclid ground‐truth (m)
def dist_tx_rx(r):
    dx = r["Transmitter_X(km)"] - r["Receiver_X(km)"]
    dy = r["Transmitter_Y(km)"] - r["Receiver_Y(km)"]
    dz = (r["Transmitter_Depth(m)"] - r["Receiver_Depth(m)"])/1000.0
    return math.sqrt(dx*dx + dy*dy + dz*dz) * KM

# per‐step feature builder + normalization
DELAY_COL = "Delay(s)"
def proc_step(group):
    amp = group["Amplitude"].values
    dly = group[DELAY_COL].values
    w   = np.abs(amp)**2
    if w.sum()==0 or np.abs(amp).sum()==0:
        raise ValueError("zero‐power frame")
    # raw summaries
    num_taps = len(amp)
    avg_pow  = np.mean(np.abs(amp))
    pw_avg   = (w*dly).sum()/w.sum()
    d_spd    = math.sqrt(((w*(dly-pw_avg)**2).sum())/w.sum())
    avg_path = (np.abs(amp)*dly).sum()/np.abs(amp).sum()
    # normalize basic features
    basic = np.stack([
        (amp - amp_mu) / amp_sigma,
        (dly - dly_mu) / dly_sigma
    ], 1)
    # normalize summary
    raw_summ = np.array([num_taps, avg_pow, d_spd, avg_path, pw_avg], dtype=np.float32)
    summ = (raw_summ - summ_mu) / summ_sigma
    return basic.astype(np.float32), summ

frames, gt_list = [], []
for ts in ts_list:
    g = csv[csv["Timestamp"]==ts]
    b, s = proc_step(g)
    frames.append((b, s))
    gt_list.append(dist_tx_rx(g.iloc[0]))
gt_m = np.array(gt_list)

# ───────── 4. roll with segmentation ─────────
pred_m = []
with torch.no_grad():
    # t=0 init
    b0, s0 = frames[0]
    b0 = torch.tensor(b0, dtype=torch.float32).to(DEVICE)
    s0 = torch.tensor(s0, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    tgt0_n = torch.tensor([(gt_m[0]-dist_mu)/dist_sigma], dtype=torch.float32).to(DEVICE)

    z0_km = to_raw(
        enc((b0.unsqueeze(0),
             torch.ones(1,b0.size(0),dtype=torch.bool,device=DEVICE),
             s0),
            tgt0_n.unsqueeze(1)).squeeze(0),
        dist_mu, dist_sigma
    )
    knet.InitSequence(torch.tensor([[z0_km.item()],[0.0]], device=DEVICE),
                      m2x_0.to(DEVICE))
    prev_km = knet.m1x_posterior[:,0:1,0]
    pred_m.append(z0_km.item()*KM)

    # t=1…T−1
    for i in range(1, T):
        if i % BLOCK_LEN == 0:
            knet.InitSequence(torch.tensor([[prev_km.item()],[0.0]], device=DEVICE),
                              m2x_0.to(DEVICE))
            prev_km = knet.m1x_posterior[:,0:1,0]

        b, s = frames[i]
        bt = torch.tensor(b, dtype=torch.float32).to(DEVICE)
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        z_km = to_raw(
            enc((bt.unsqueeze(0),
                 torch.ones(1,bt.size(0),dtype=torch.bool,device=DEVICE),
                 st),
                to_norm(prev_km, dist_mu, dist_sigma)
            ).squeeze(0),
            dist_mu, dist_sigma
        )
        est = knet(z_km)
        prev_km = knet.m1x_prior[:,0:1,0]
        pred_m.append(est[:,0,0].item()*KM)

pred_m = np.array(pred_m)

# ───────── 5. metrics & plot ─────────
mae  = np.abs(pred_m - gt_m).mean()
rmse = np.sqrt(((pred_m - gt_m)**2).mean())
print(f"[eval] MAE  = {mae:.2f} m   RMSE = {rmse:.2f} m")

plt.figure(figsize=(10,4))
plt.plot(gt_m,  label="ground truth")
plt.plot(pred_m, label="Latent-KalmanNet")
plt.xlabel("timestep"); plt.ylabel("distance [m]"); plt.legend(); plt.tight_layout()
out = PLOT_DIR/"long_traj_segmented_normed.png"
plt.savefig(out)
print(f"[eval] plot saved → {out}")
