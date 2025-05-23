#!/usr/bin/env python
"""
Evaluate one trajectory with the trained Latent-KalmanNet
Filters rows to:  Transmitter_Name == 'Alice'   AND   Sensor_ID == 'Apex'
"""

import argparse, random, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from underwater_dataset    import UnderwaterDataset
from encoder_underwater     import UnderwaterEncoderWithPrior
from kalman_net_underwater  import KalmanNetUnderwater
from model_Underwater       import f_function, H, m, n, m2x_0        # km model

# ───────── constants ─────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KM       = 1e3
TRAIN_RATIO, VAL_RATIO, RNG_SEED = 0.7, 0.15, 42
PLOT_DIR = Path("plots"); PLOT_DIR.mkdir(exist_ok=True)

def build_traj_map(ds):
    mp = {}
    for i, d in enumerate(ds.all_data):
        mp.setdefault(d["trajectory_id"], []).append(i)
    return mp

def to_raw(norm, ds):  return (norm * ds.distance_std + ds.distance_mean) / KM
def to_norm(km, ds):   return ((km * KM) - ds.distance_mean) / ds.distance_std

# ───────── CLI ─────────
pa = argparse.ArgumentParser()
pa.add_argument("--csv", default="data/all_arrivals_long.csv")
args = pa.parse_args()

# ───────── 1. dataset & µ/σ on pseudo-train split ─────────
ds = UnderwaterDataset("data")
tids = list({d["trajectory_id"] for d in ds.all_data})
random.Random(RNG_SEED).shuffle(tids)
train_tids = tids[:int(TRAIN_RATIO * len(tids))]
train_idx  = [i for tid in train_tids for i in build_traj_map(ds)[tid]]
ds.compute_normalisation_stats(indices=train_idx)
print(f"[stats] μ={ds.distance_mean:.1f} m  σ={ds.distance_std:.1f} m")

# ───────── 2. models ─────────
enc  = UnderwaterEncoderWithPrior().to(DEVICE)
enc.load_state_dict(torch.load("saved_models/enc_finetuned.pth", map_location=DEVICE))
knet = KalmanNetUnderwater().to(DEVICE)
knet.Build(m=m, n=n, f_function=f_function, H=H.to(DEVICE))
knet.load_state_dict(torch.load("saved_models/knet_best.pth", map_location=DEVICE))
enc.eval(); knet.eval()

# ───────── 3. read evaluation CSV & filter Alice → Apex ─────────
csv = pd.read_csv(args.csv)
csv = csv[(csv["Transmitter_Name"] == "Alice") & (csv["Sensor_ID"] == "Apex")]
if csv.empty:
    raise ValueError("No rows with Transmitter_Name='Alice' and Sensor_ID='Apex'")
ts_list = sorted(csv["Timestamp"].unique())
T = len(ts_list)
print(f"[eval] trajectory length = {T} steps  (first ts={ts_list[0]})")

# distance helper
def dist_tx_rx(row):
    dx = row["Transmitter_X(km)"] - row["Receiver_X(km)"]
    dy = row["Transmitter_Y(km)"] - row["Receiver_Y(km)"]
    dz = (row["Transmitter_Depth(m)"] - row["Receiver_Depth(m)"]) / 1000.0
    return math.sqrt(dx*dx + dy*dy + dz*dz) * KM   # metres

# per-step feature builder
DELAY_COL = "Delay(s)"
def proc_step(group):
    amp = group["Amplitude"].values
    dly = group[DELAY_COL].values
    w   = np.abs(amp)**2
    if w.sum() == 0 or np.abs(amp).sum() == 0:
        raise ValueError("zero-power frame")
    pw_avg = (w*dly).sum()/w.sum()
    d_spread = math.sqrt(((w*(dly-pw_avg)**2).sum())/w.sum())
    avg_path = (np.abs(amp)*dly).sum()/np.abs(amp).sum()
    return dict(
        basic=np.stack([(amp-ds.amplitude_mean)/ds.amplitude_std,
                        (dly-ds.delay_mean   )/ds.delay_std   ], 1),
        summary=np.array([len(amp), np.mean(np.abs(amp)), d_spread,
                          avg_path, pw_avg], dtype=np.float32)
    )

frames, gt_list = [], []
for ts in ts_list:
    g = csv[csv["Timestamp"] == ts]
    frames.append(proc_step(g))
    gt_list.append(dist_tx_rx(g.iloc[0]))

gt_m = np.array(gt_list)

# ───────── 4. roll Latent-KalmanNet ─────────
pred_m = []
with torch.no_grad():
    b0 = torch.tensor(frames[0]["basic"], dtype=torch.float32).to(DEVICE)
    s0 = torch.tensor(frames[0]["summary"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    tgt0_n = torch.tensor([(gt_m[0]-ds.distance_mean)/ds.distance_std], dtype=torch.float32).to(DEVICE)

    z0_km = to_raw(enc((b0.unsqueeze(0),
                        torch.ones(1,b0.size(0),dtype=torch.bool,device=DEVICE),
                        s0), tgt0_n.unsqueeze(1)).squeeze(0), ds)
    knet.InitSequence(torch.tensor([[z0_km.item()], [0.0]], device=DEVICE), m2x_0.to(DEVICE))
    prev_pred_km = knet.m1x_posterior[:,0:1,0]
    pred_m.append(z0_km.item()*KM)

    for t in range(1, T):
        b = torch.tensor(frames[t]["basic"], dtype=torch.float32).to(DEVICE)
        s = torch.tensor(frames[t]["summary"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        z_km = to_raw(enc((b.unsqueeze(0),
                           torch.ones(1,b.size(0),dtype=torch.bool,device=b.device),
                           s), to_norm(prev_pred_km, ds)).squeeze(0), ds)
        est = knet(z_km)
        prev_pred_km = knet.m1x_prior[:,0:1,0]
        pred_m.append(est[:,0,0].item()*KM)

pred_m = np.array(pred_m)

# ───────── 5. metrics & plot ─────────
mae  = np.abs(pred_m-gt_m).mean()
rmse = np.sqrt(((pred_m-gt_m)**2).mean())
print(f"[eval] MAE  = {mae:.2f} m   RMSE = {rmse:.2f} m")

plt.figure(figsize=(10,4))
plt.plot(gt_m,  label="ground truth")
plt.plot(pred_m, label="Latent-KalmanNet")
plt.xlabel("timestep"); plt.ylabel("distance [m]"); plt.legend(); plt.tight_layout()
out = PLOT_DIR/"long_traj.png"; plt.savefig(out)
print(f"[eval] plot saved → {out}")
plt.figure(figsize=(10,4))
plt.plot(gt_m, color="steelblue")
plt.xlabel("timestep"); plt.ylabel("ground-truth distance  [m]")
plt.title("Ground-truth trajectory (Alice → Apex)")
plt.tight_layout()
out_gt = PLOT_DIR / "long_traj_ground_truth_only.png"
plt.savefig(out_gt)
print(f"[eval] ground-truth plot saved → {out_gt}")
