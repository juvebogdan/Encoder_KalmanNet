import os, random
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from underwater_dataset import UnderwaterDataset
from encoder_underwater import UnderwaterEncoderWithPrior

# ────────────────────────── config ──────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("saved_models"); SAVE_DIR.mkdir(exist_ok=True)
PLOT_DIR = Path("plots"); PLOT_DIR.mkdir(exist_ok=True)
EPOCHS, LR, NOISE_LEVEL, TRAJ_BATCH = 40, 3e-3, 0.1, 4
PRINT_EVERY, TRAIN_RATIO, VAL_RATIO = 100, 0.7, 0.15
RNG_SEED = 42
stats_filename = SAVE_DIR / "encoder_normalization_stats.npz"
print(f"[train-seq] device = {DEVICE}\n")

# ───────────────────── visualisation utils ─────────────────────

def _ensure_dir(d: Path):
    d.mkdir(exist_ok=True)

def save_loss_plot(tr: List[float], va: List[float]):
    _ensure_dir(PLOT_DIR)
    plt.figure(figsize=(8, 4))
    plt.plot(tr, label="train"); plt.plot(va, label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOT_DIR / "loss_curve.png"); plt.close()

def save_traj_plots(ds: UnderwaterDataset, traj_map: Dict[int, List[int]], preds: Dict[int, List[float]]):
    _ensure_dir(PLOT_DIR)
    for tid, pr in preds.items():
        gts = [ds.all_data[i]["distance"] for i in traj_map[tid]]
        ts = list(range(len(gts)))
        plt.figure(); plt.plot(ts, gts, label="GT"); plt.plot(ts, pr, label="pred")
        plt.xlabel("timestep"); plt.ylabel("distance [m]")
        plt.title(f"Trajectory {tid}"); plt.legend(); plt.tight_layout()
        plt.savefig(PLOT_DIR / f"traj_{tid}.png"); plt.close()

# ───────────────────── helpers ─────────────────────

def make_noisy_prior(tgt: torch.Tensor, sigma: float) -> torch.Tensor:
    prior = tgt.clone()
    if sigma > 0.0:
        prior += torch.randn_like(prior) * sigma
    return prior if prior.dim() == 2 else prior.unsqueeze(1)

def build_traj_map(ds: UnderwaterDataset) -> Dict[int, List[int]]:
    mp: Dict[int, List[int]] = {}
    for idx, d in enumerate(ds.all_data):
        mp.setdefault(d["trajectory_id"], []).append(idx)
    for tid in mp:
        mp[tid].sort(key=lambda i: ds.all_data[i]["timestamp"])
    return mp

# ─────────────────── trajectory trainer ───────────────────

def train_by_trajectory(model: nn.Module, ds: UnderwaterDataset, train_tids: List[int], val_tids: List[int]):
    traj_map = build_traj_map(ds)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    mse = nn.MSELoss()
    best_val, tr_hist, va_hist = float("inf"), [], []
    print(f"[train-seq] {len(train_tids)} train traj | {len(val_tids)} val traj\n")
    for ep in range(1, EPOCHS + 1):
        model.train(); random.shuffle(train_tids)
        seen = 0; acc_loss = 0.0
        for tid in train_tids:
            traj_loss = 0.0
            for idx in traj_map[tid]:
                (basic, summary), tgt = ds[idx]
                pad = basic.unsqueeze(0).to(DEVICE)
                mask = torch.ones(1, basic.size(0), dtype=torch.bool, device=DEVICE)
                summary = summary.unsqueeze(0).to(DEVICE)
                tgt = tgt.to(DEVICE)
                prior = make_noisy_prior(tgt, NOISE_LEVEL)
                out = model((pad, mask, summary), prior)
                traj_loss += mse(out, tgt.view_as(out))
            acc_loss += traj_loss / len(traj_map[tid]); seen += 1
            if seen % TRAJ_BATCH == 0 or tid == train_tids[-1]:
                opt.zero_grad(); acc_loss.backward(); opt.step(); acc_loss = 0.0
            if seen % PRINT_EVERY == 0:
                print(f"  [ep {ep}] processed {seen}/{len(train_tids)} trajectories")
        tr_hist.append(evaluate(model, ds, train_tids[:32])); va_hist.append(evaluate(model, ds, val_tids))
        if va_hist[-1] < best_val:
            best_val = va_hist[-1]; torch.save(model.state_dict(), SAVE_DIR / "best_encoder_seq.pth")
        print(f"Epoch {ep:03d}/{EPOCHS} | train {tr_hist[-1]:.4e} | val {va_hist[-1]:.4e}{' *' if va_hist[-1]==best_val else ''}")
    save_loss_plot(tr_hist, va_hist)


def evaluate(model: nn.Module, ds: UnderwaterDataset, tids: List[int]) -> float:
    traj_map = build_traj_map(ds); mse = nn.MSELoss(); model.eval(); total, cnt = 0.0, 0
    with torch.no_grad():
        for tid in tids:
            for idx in traj_map[tid]:
                (basic, summary), tgt = ds[idx]
                pad = basic.unsqueeze(0).to(DEVICE)
                mask = torch.ones(1, basic.size(0), dtype=torch.bool, device=DEVICE)
                summary = summary.unsqueeze(0).to(DEVICE)
                tgt = tgt.to(DEVICE)
                prior = make_noisy_prior(tgt, NOISE_LEVEL)
                out = model((pad, mask, summary), prior)
                total += mse(out, tgt.view_as(out)).item(); cnt += 1
    return total / max(cnt, 1)

# ──────────────── trajectory visualisation ────────────────

def visualise_some_trajectories(model: nn.Module, ds: UnderwaterDataset, tids: List[int], num: int = 3):
    sel = tids[:num]; traj_map = build_traj_map(ds); preds: Dict[int, List[float]] = {}
    model.eval();
    with torch.no_grad():
        for tid in sel:
            prev_pred, pr = None, []
            for idx in traj_map[tid]:
                (basic, summary), tgt = ds[idx]
                pad = basic.unsqueeze(0).to(DEVICE)
                mask = torch.ones(1, basic.size(0), dtype=torch.bool, device=DEVICE)
                summary = summary.unsqueeze(0).to(DEVICE)
                prior = make_noisy_prior(tgt, NOISE_LEVEL)
                out = model((pad, mask, summary), prior)
                pr.append(ds.denormalise_distance(out.squeeze().cpu().numpy()))
            preds[tid] = pr
    save_traj_plots(ds, traj_map, preds)

# ────────────────────────── main ──────────────────────────
if __name__ == "__main__":
    #random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)
    ds = UnderwaterDataset("data")
    tids = list({d["trajectory_id"] for d in ds.all_data}); random.shuffle(tids)
    n_tr = int(TRAIN_RATIO * len(tids)); n_val = int(VAL_RATIO * len(tids))
    tr_tids, val_tids, te_tids = tids[:n_tr], tids[n_tr:n_tr+n_val], tids[n_tr+n_val:]
    train_idx = [i for tid in tr_tids for i in build_traj_map(ds)[tid]]
    ds.compute_normalisation_stats(indices=train_idx)
    print(f"[split] {len(tr_tids)} train | {len(val_tids)} val | {len(te_tids)} test\n")

    # Save all the normalization parameters
    np.savez(stats_filename,
        # Distance normalization (what the encoder predicts)
        distance_mean=ds.distance_mean,
        distance_std=ds.distance_std,
        
        # Input feature normalization
        amplitude_mean=ds.amplitude_mean,
        amplitude_std=ds.amplitude_std,
        delay_mean=ds.delay_mean,
        delay_std=ds.delay_std,
        
        # Summary statistics normalization
        summary_means=ds.summary_means,  # This is an array of 5 values
        summary_stds=ds.summary_stds,    # This is an array of 5 values
        
        # Also save metadata for verification
        num_train_samples=len(train_idx),
        train_trajectory_ids=tr_tids  # So we can verify same split if needed
    )

    print(f"[IMPORTANT] Normalization stats saved to: {stats_filename}")
    print(f"  Distance: mean={ds.distance_mean:.1f}m, std={ds.distance_std:.1f}m")

    # ───── baseline: always predict training-set mean distance ─────
    traj_map   = build_traj_map(ds)
    test_idx   = [i for tid in te_tids for i in traj_map[tid]]
    mean_dist  = ds.distance_mean                    # metres
    dummy_rmse = np.sqrt(
        np.mean([(ds.all_data[i]["distance"] - mean_dist) ** 2
                 for i in test_idx])
    )
    print(f"[baseline] always-mean RMSE {dummy_rmse:.2f} m\n")
    # ───────────────────────────────────────────────────────────────
    model = UnderwaterEncoderWithPrior().to(DEVICE)
    train_by_trajectory(model, ds, tr_tids, val_tids)
    model.load_state_dict(torch.load(SAVE_DIR / "best_encoder_seq.pth", map_location=DEVICE))
    test_mse = evaluate(model, ds, te_tids)
    rmse = np.sqrt(test_mse * ds.distance_std ** 2)
    print(f"[test] RMSE {rmse:.2f} m (denormalised)")
    visualise_some_trajectories(model, ds, te_tids)
