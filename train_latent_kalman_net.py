"""
Latent-KalmanNet training for the underwater multipath-arrival dataset
=====================================================================
A   (optional) : warm-up encoder on normalised distances
B               : train KalmanNet (encoder frozen, km units, grad-clip)
C               : fine-tune encoder (KalmanNet frozen)

Typical run
-----------
python train_latent_kalman_net.py --data_dir data
"""

import argparse, random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

from underwater_dataset     import UnderwaterDataset
from encoder_underwater      import UnderwaterEncoderWithPrior
from kalman_net_underwater   import KalmanNetUnderwater
from model_Underwater        import f_function, H, m, n, m2x_0   # constant-velocity model

# ───────────── hyper-parameters ─────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG_SEED     = 42
TRAIN_RATIO, VAL_RATIO = 0.7, 0.15

ENC_EPOCHS, ENC_LR, ENC_SIGMA, ENC_ACCUMULATION_TRAJ = 40, 3e-3, 0.1, 4             # Phase-A
KNET_EPOCHS, KNET_LR         = 20, 3e-4                   # Phase-B  (↓ LR)
FT_EPOCHS                    = 10                         # Phase-C
CLIP_NORM                    = 10.0                       # grad-clip
HEARTBEAT                    = 50                         # print every N traj

#random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)

# ─────────── misc helpers ───────────
def build_traj_map(ds: UnderwaterDataset) -> Dict[int, List[int]]:
    mp: Dict[int, List[int]] = {}
    for i, d in enumerate(ds.all_data):
        mp.setdefault(d["trajectory_id"], []).append(i)
    for tid in mp: mp[tid].sort(key=lambda j: ds.all_data[j]["timestamp"])
    return mp

def make_noisy(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + sigma * torch.randn_like(x) if sigma > 0 else x

# ─────────── unit converters (km ⇆ z-score) ───────────
KM   = 1e3  # 1 km = 1000 m   – KalmanNet works in kilometres
to_raw  = lambda norm, ds: (norm * ds.distance_std + ds.distance_mean) / KM          # → km
to_norm = lambda km,   ds: ((km * KM) - ds.distance_mean) / ds.distance_std          # → z-score

# ─────────── Phase-A : encoder warm-up (normalised) ───────────
def train_encoder(enc, ds, tr, val, save, accumulation_steps=4):
    """
    Train encoder with gradient accumulation support.
    
    Args:
        accumulation_steps: Number of trajectories to process before updating weights.
                           Set to 1 for original behavior (update after each trajectory).
                           Set to 4 to match train_underwater.py behavior.
    """
    mse = nn.MSELoss()
    opt = optim.AdamW(enc.parameters(), lr=ENC_LR, weight_decay=1e-2)
    best = float("inf")
    tm = build_traj_map(ds)
    
    for ep in range(1, ENC_EPOCHS+1):
        print(f"\n[ENC] epoch {ep}/{ENC_EPOCHS} – {len(tr)} trajectories")
        enc.train()
        random.shuffle(tr)
        
        # Initialize accumulation tracking
        accumulated_loss = 0.0
        trajectories_in_batch = 0
        running_loss = 0.0
        trajectories_seen = 0
        
        # Important: Clear gradients at start
        opt.zero_grad()
        
        for i, tid in enumerate(tr):
            # Process one trajectory
            loss_t = 0.0
            for idx in tm[tid]:
                (basic, summary), tgt_n = ds[idx]
                prior_n = make_noisy(tgt_n, ENC_SIGMA).unsqueeze(1).to(DEVICE)
                out = enc((basic.unsqueeze(0).to(DEVICE),
                          torch.ones(1,basic.size(0),dtype=torch.bool,device=DEVICE),
                          summary.unsqueeze(0).to(DEVICE)), prior_n)
                loss_t += mse(out, tgt_n.to(DEVICE).view_as(out))
            
            # Average loss for this trajectory
            trajectory_avg_loss = loss_t / len(tm[tid])
            
            # Scale by 1/accumulation_steps to maintain same effective learning rate
            # This is crucial - without this, larger batches would have larger gradients
            scaled_loss = trajectory_avg_loss / accumulation_steps
            
            # Accumulate gradients (backward pass adds to existing gradients)
            scaled_loss.backward()
            
            # Track for logging
            accumulated_loss += trajectory_avg_loss.item()
            trajectories_in_batch += 1
            running_loss += trajectory_avg_loss.item()
            trajectories_seen += 1
            
            # Check if we should update weights
            if trajectories_in_batch == accumulation_steps or i == len(tr) - 1:
                # Perform optimization step
                opt.step()
                opt.zero_grad()
                
                # Reset batch tracking
                trajectories_in_batch = 0
                accumulated_loss = 0.0
            
            # Progress logging
            if trajectories_seen % HEARTBEAT == 0:
                avg_running = running_loss / trajectories_seen
                print(f"    …{trajectories_seen}/{len(tr)} traj, running loss {avg_running:.3e}")
        
        # Validation
        val_mse = evaluate_encoder(enc, ds, val)
        if val_mse < best:
            best = val_mse
            torch.save(enc.state_dict(), save/"best_encoder_seq.pth")
        print(f"[ENC] epoch {ep} done | val {val_mse:.3e}{' *' if val_mse==best else ''}")

def evaluate_encoder(enc, ds, tids):
    mse = nn.MSELoss(); tm = build_traj_map(ds); tot = cnt = 0
    enc.eval()
    with torch.no_grad():
        for tid in tids:
            for idx in tm[tid]:
                (basic, summary), tgt_n = ds[idx]
                out = enc((basic.unsqueeze(0).to(DEVICE),
                           torch.ones(1,basic.size(0),dtype=torch.bool,device=DEVICE),
                           summary.unsqueeze(0).to(DEVICE)), tgt_n.unsqueeze(1).to(DEVICE))
                tot += mse(out, tgt_n.to(DEVICE).view_as(out)).item(); cnt += 1
    return tot/cnt

# ─────────── Phase-B : KalmanNet training (km) ───────────
def train_knet(knet, enc, ds, tr, val, save):
    for p in enc.parameters(): p.requires_grad_(False)
    mse = nn.MSELoss(); opt = optim.Adam(knet.parameters(), lr=KNET_LR)
    best = float("inf"); tm = build_traj_map(ds)
    for ep in range(1, KNET_EPOCHS+1):
        print(f"\n[KNET] epoch {ep}/{KNET_EPOCHS} – {len(tr)} trajectories")
        knet.train(); random.shuffle(tr); run = seen = 0; tr_sum = tr_cnt = 0
        for tid in tr:
            opt.zero_grad(); loss_t = 0.0
            first = tm[tid][0]
            (b0,s0),t0_n = ds[first]
            z0_km = to_raw(enc((b0.unsqueeze(0).to(DEVICE),
                                torch.ones(1,b0.size(0),dtype=torch.bool,device=DEVICE),
                                s0.unsqueeze(0).to(DEVICE)), t0_n.unsqueeze(1).to(DEVICE)).squeeze(0), ds)
            knet.InitSequence(torch.tensor([[z0_km.item()],[0.0]], device=DEVICE), m2x_0.to(DEVICE))
            prev_pred_km = knet.m1x_posterior[:,0:1,0]  # km

            for idx in tm[tid]:
                (basic, summary), tgt_n = ds[idx]
                prior_n = to_norm(prev_pred_km, ds)
                z_km = to_raw(enc((basic.unsqueeze(0).to(DEVICE),
                                   torch.ones(1,basic.size(0),dtype=torch.bool,device=DEVICE),
                                   summary.unsqueeze(0).to(DEVICE)), prior_n).squeeze(0), ds)
                est = knet(z_km); pred_km = est[:,0,0]
                loss = mse(pred_km, to_raw(tgt_n, ds))
                loss_t += loss; tr_sum += loss.item(); tr_cnt += 1
                prev_pred_km = knet.m1x_prior[:,0:1,0]
            (loss_t/len(tm[tid])).backward()
            torch.nn.utils.clip_grad_norm_(knet.parameters(), CLIP_NORM)  # <-- stabiliser
            opt.step()
            run += loss_t.item(); seen += 1
            if seen % HEARTBEAT == 0:
                print(f"    …{seen}/{len(tr)} traj, running loss {run/seen:.3e}")
        val_mse = evaluate_knet(knet, enc, ds, val)
        if val_mse < best:
            best = val_mse; torch.save(knet.state_dict(), save/"knet_best.pth")
        print(f"[KNET] epoch {ep} done | train {tr_sum/tr_cnt:.3e} | val {val_mse:.3e}{' *' if val_mse==best else ''}")

def evaluate_knet(knet, enc, ds, tids):
    mse = nn.MSELoss(); tm = build_traj_map(ds); tot = cnt = 0
    knet.eval(); enc.eval()
    with torch.no_grad():
        for tid in tids:
            first = tm[tid][0]
            (b0,s0),t0_n = ds[first]
            z0_km = to_raw(enc((b0.unsqueeze(0).to(DEVICE),
                                torch.ones(1,b0.size(0),dtype=torch.bool,device=DEVICE),
                                s0.unsqueeze(0).to(DEVICE)), t0_n.unsqueeze(1).to(DEVICE)).squeeze(0), ds)
            knet.InitSequence(torch.tensor([[z0_km.item()],[0.0]], device=DEVICE), m2x_0.to(DEVICE))
            prev_pred_km = knet.m1x_posterior[:,0:1,0]
            for idx in tm[tid]:
                (basic, summary), tgt_n = ds[idx]
                z_km = to_raw(enc((basic.unsqueeze(0).to(DEVICE),
                                   torch.ones(1,basic.size(0),dtype=torch.bool,device=DEVICE),
                                   summary.unsqueeze(0).to(DEVICE)),
                                  to_norm(prev_pred_km, ds)).squeeze(0), ds)
                est = knet(z_km); prev_pred_km = knet.m1x_prior[:,0:1,0]
                tot += mse(est[:,0,0], to_raw(tgt_n, ds)).item(); cnt += 1
    return tot/cnt

# ─────────── Phase-C : encoder fine-tune (mixed) ───────────
def finetune_encoder(enc, knet, ds, tr, val, save):
    for p in knet.parameters(): p.requires_grad_(False)
    for p in enc.parameters():  p.requires_grad_(True)
    opt = optim.Adam(enc.parameters(), lr=1e-4); mse = nn.MSELoss(); best=float("inf")
    tm = build_traj_map(ds)
    for ep in range(1, FT_EPOCHS+1):
        print(f"\n[FT]  epoch {ep}/{FT_EPOCHS} – {len(tr)} trajectories")
        enc.train(); random.shuffle(tr); run = seen = 0
        for tid in tr:
            opt.zero_grad(); loss_t = 0.0
            first = tm[tid][0]
            (b0,s0),t0_n = ds[first]
            z0_km = to_raw(enc((b0.unsqueeze(0).to(DEVICE),
                                torch.ones(1,b0.size(0),dtype=torch.bool,device=DEVICE),
                                s0.unsqueeze(0).to(DEVICE)), t0_n.unsqueeze(1).to(DEVICE)).squeeze(0), ds)
            knet.InitSequence(torch.tensor([[z0_km.item()],[0.0]], device=DEVICE), m2x_0.to(DEVICE))
            prev_pred_km = knet.m1x_posterior[:,0:1,0]

            for idx in tm[tid]:
                (basic, summary), tgt_n = ds[idx]
                z_km = to_raw(enc((basic.unsqueeze(0).to(DEVICE),
                                   torch.ones(1,basic.size(0),dtype=torch.bool,device=DEVICE),
                                   summary.unsqueeze(0).to(DEVICE)),
                                  to_norm(prev_pred_km, ds)).squeeze(0), ds)
                est = knet(z_km); prev_pred_km = knet.m1x_prior[:,0:1,0]
                loss_t += mse(est[:,0,0], to_raw(tgt_n, ds))
            (loss_t/len(tm[tid])).backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), CLIP_NORM)
            opt.step()
            run += loss_t.item(); seen += 1
            if seen % HEARTBEAT == 0:
                print(f"    …{seen}/{len(tr)} traj, running loss {run/seen:.3e}")
        val_mse = evaluate_knet(knet, enc, ds, val)
        if val_mse < best:
            best = val_mse; torch.save(enc.state_dict(), save/"enc_finetuned.pth")
        print(f"[FT]  epoch {ep} done | val {val_mse:.3e}{' *' if val_mse==best else ''}")


def plot_some_trajectories(knet, enc, ds, tids, num=3):
    tm = build_traj_map(ds); knet.eval(); enc.eval()
    with torch.no_grad():
        for tid in tids[:num]:
            idx0 = tm[tid][0]
            (b0,s0),t0_n = ds[idx0]
            z0_km = to_raw(enc((b0.unsqueeze(0).to(DEVICE),
                                torch.ones(1,b0.size(0),dtype=torch.bool,device=DEVICE),
                                s0.unsqueeze(0).to(DEVICE)), t0_n.unsqueeze(1).to(DEVICE)).squeeze(0), ds)
            knet.InitSequence(torch.tensor([[z0_km.item()],[0.0]], device=DEVICE), m2x_0.to(DEVICE))
            pred, gt = [], []
            prev_pred_km = knet.m1x_posterior[:,0:1,0]
            for idx in tm[tid]:
                (basic, summary), tgt_n = ds[idx]
                z_km = to_raw(enc((basic.unsqueeze(0).to(DEVICE),
                                   torch.ones(1,basic.size(0),dtype=torch.bool,device=DEVICE),
                                   summary.unsqueeze(0).to(DEVICE)),
                                  to_norm(prev_pred_km, ds)).squeeze(0), ds)
                est = knet(z_km); prev_pred_km = knet.m1x_prior[:,0:1,0]
                pred.append(est[:,0,0].item()*KM)              # km → m
                gt.append(to_raw(tgt_n, ds).item()*KM)
            plt.figure()
            plt.plot(gt, label='GT'); plt.plot(pred, label='KNet')
            plt.xlabel('timestep'); plt.ylabel('distance [m]')
            plt.title(f'Trajectory {tid}'); plt.legend(); plt.tight_layout()
            plt.savefig(Path('plots')/f'traj_{tid}.png')

# ─────────── main ───────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_dir", default="data"); pa.add_argument("--encoder_ckpt")
    pa.add_argument("--force_pretrain", action="store_true"); pa.add_argument("--save_dir", default="saved_models")
    args = pa.parse_args(); save = Path(args.save_dir); save.mkdir(exist_ok=True)
    print(f"[setup] device={DEVICE}")

    # dataset ---------------------------------------------------------------
    ds = UnderwaterDataset(args.data_dir)
    tids = list({d["trajectory_id"] for d in ds.all_data}); random.shuffle(tids)
    n_tr, n_val = int(TRAIN_RATIO*len(tids)), int(VAL_RATIO*len(tids))
    tr_tids, val_tids, te_tids = tids[:n_tr], tids[n_tr:n_tr+n_val], tids[n_tr+n_val:]
    ds.compute_normalisation_stats(indices=[i for tid in tr_tids for i in build_traj_map(ds)[tid]])

    stats_file = save / "norm_stats.npz"
    np.savez(stats_file,
            dist_mean=ds.distance_mean,
            dist_std =ds.distance_std,
            amp_mean =ds.amplitude_mean,
            amp_std  =ds.amplitude_std,
            dly_mean =ds.delay_mean,
            dly_std  =ds.delay_std,
            summ_mean=ds.summary_means,
            summ_std =ds.summary_stds)
    print(f"[norm] statistics saved to {stats_file}")

    # models ---------------------------------------------------------------
    enc  = UnderwaterEncoderWithPrior().to(DEVICE)
    knet = KalmanNetUnderwater().to(DEVICE); knet.Build(m=m, n=n, f_function=f_function,
                                                       H=H.to(DEVICE))

    # warm-start encoder ----------------------------------------------------
    ckpt = Path(args.encoder_ckpt) if args.encoder_ckpt else Path("saved_models")/"best_encoder_seq.pth"
    if ckpt.exists():
        enc.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"[init] loaded {ckpt}")
        pretrained = True
    else:
        pretrained = False

    if not pretrained or args.force_pretrain:
        print("\n==== Phase-A : encoder warm-up ====")
        train_encoder(enc, ds, tr_tids, val_tids, save, 
                    accumulation_steps=ENC_ACCUMULATION_TRAJ)  # Pass the parameter
        enc.load_state_dict(torch.load(save/"best_encoder_seq.pth", map_location=DEVICE))
    else:
        print("\n[skip] Phase-A")

    print("\n==== Phase-B : KalmanNet training ====")
    train_knet(knet, enc, ds, tr_tids, val_tids, save)
    knet.load_state_dict(torch.load(save/"knet_best.pth", map_location=DEVICE))

    print("\n==== Phase-C : encoder fine-tune ====")
    finetune_encoder(enc, knet, ds, tr_tids, val_tids, save)

    # test -----------------------------------------------------------------
    test_mse = evaluate_knet(knet, enc, ds, te_tids)
    rmse_m   = np.sqrt(test_mse) * KM   # convert km → m for final metric
    print(f"\n[test] RMSE = {rmse_m:.2f} m")
    plot_some_trajectories(knet, enc, ds, te_tids)

if __name__ == "__main__":
    main()
