import os
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TRAJECTORY_LENGTH = 199

class UnderwaterDataset(Dataset):
    """Dataset that loads per‑timestep underwater channel data and exposes
    normalised tensors for the encoder network.

    After you define the train/val/test split, call
    :py:meth:`compute_normalisation_stats(indices=training_indices)` so that
    mean/std are fitted **only on the training set** (avoids data leakage).
    """

    def __init__(self, data_dir: str):
        self.all_data: List[dict] = []

        csv_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("all_arrivals") and f.endswith(".csv")
        )
        print(f"[UnderwaterDataset] found {len(csv_files)} CSV files")

        total_traj = kept_traj = 0
        for csv in csv_files:
            df = pd.read_csv(os.path.join(data_dir, csv))

            # find blocks that go timestamp 0 … 199
            ts = df["Timestamp"].values
            cuts = [0] + [i for i in range(1, len(ts))
                        if ts[i] == 0 and ts[i - 1] == TRAJECTORY_LENGTH] + [len(ts)]

            for beg, end in zip(cuts[:-1], cuts[1:]):
                total_traj += 1
                block = df.iloc[beg:end]

                # skip if missing timesteps
                if len(block["Timestamp"].unique()) != TRAJECTORY_LENGTH + 1:
                    continue

                traj_samples = []
                valid = True
                for t in range(TRAJECTORY_LENGTH + 1):
                    sample = self._process_timestep(block[block["Timestamp"] == t],
                                                    t, kept_traj)   # tentative id
                    if sample is None:          # zero-power frame
                        valid = False
                        break
                    traj_samples.append(sample)

                if valid:
                    self.all_data.extend(traj_samples)
                    kept_traj += 1              # final id only increments if kept

        print(f"[UnderwaterDataset] kept {kept_traj}/{total_traj} trajectories "
            f"→ {len(self.all_data)} samples "
            f"({total_traj - kept_traj} dropped for zero power or missing steps)")

        # placeholders for normalisation (unchanged)
        self.amplitude_mean = self.amplitude_std = 0.0
        self.delay_mean = self.delay_std = 0.0
        self.summary_means = np.zeros(5); self.summary_stds = np.ones(5)
        self.distance_mean = self.distance_std = 0.0


    # ────────────────────────────────────────────────────────────────────
    def compute_normalisation_stats(self, *, indices: Optional[Sequence[int]] = None):
        """Compute mean/σ using `indices` (training set). If *None*, use all."""
        if indices is None:
            iterable = self.all_data; tag = "ALL samples"
        else:
            iterable = (self.all_data[i] for i in indices); tag = f"{len(indices)} training samples"

        amps, dels, dists, summaries = [], [], [], []
        for d in iterable:
            amps.extend(d["amplitudes"])
            dels.extend(d["delays"])
            dists.append(d["distance"])
            summaries.append([
                d["num_taps"], d["avg_tap_power"], d["delay_spread"],
                d["avg_path_delay"], d["power_weighted_avg_delay"],
            ])
        amps, dels = np.asarray(amps), np.asarray(dels)
        dists = np.asarray(dists)
        summaries = np.asarray(summaries)

        self.amplitude_mean, self.amplitude_std = amps.mean(), amps.std() + 1e-6
        self.delay_mean, self.delay_std = dels.mean(), dels.std() + 1e-6
        self.summary_means = summaries.mean(0)
        self.summary_stds = summaries.std(0) + 1e-6
        self.distance_mean, self.distance_std = dists.mean(), dists.std() + 1e-6
        print(f"[Normalisation] fitted on {tag}: distance σ ≈ {self.distance_std:.2f} m")

    @staticmethod
    def _process_timestep(rows: pd.DataFrame, timestamp: int, traj_id: int):
        amp = rows["Amplitude"].values
        dly = rows["Delay"].values
        dist = float(rows["Distance_GT"].values[0])

        num_taps = len(amp)
        avg_pow = np.mean(np.abs(amp))

        w = np.abs(amp) ** 2
        w_sum = w.sum()
        mag_sum = np.abs(amp).sum()

        # ───────── skip silent-zero tap blocks ─────────
        if w_sum == 0.0 or mag_sum == 0.0:
            return None

        pw_avg_delay = (w * dly).sum() / w_sum
        delay_spread = np.sqrt(((w * (dly - pw_avg_delay) ** 2).sum()) / w_sum)
        avg_path_delay = (np.abs(amp) * dly).sum() / mag_sum

        return {
            "amplitudes": amp,
            "delays": dly,
            "distance": dist,
            "timestamp": timestamp,
            "trajectory_id": traj_id,
            "num_taps": num_taps,
            "avg_tap_power": avg_pow,
            "delay_spread": delay_spread,
            "avg_path_delay": avg_path_delay,
            "power_weighted_avg_delay": pw_avg_delay,
        }


    # ────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        d = self.all_data[idx]
        amp = (d["amplitudes"] - self.amplitude_mean) / self.amplitude_std
        dly = (d["delays"] - self.delay_mean) / self.delay_std
        basic = torch.tensor(np.stack([amp, dly], 1), dtype=torch.float32)

        raw_summary = np.array([
            d["num_taps"], d["avg_tap_power"], d["delay_spread"],
            d["avg_path_delay"], d["power_weighted_avg_delay"],
        ])
        summary = torch.tensor((raw_summary - self.summary_means) / self.summary_stds, dtype=torch.float32)
        tgt = torch.tensor([(d["distance"] - self.distance_mean) / self.distance_std], dtype=torch.float32)
        return (basic, summary), tgt

    # ────────────────────────────────────────────────────────────────────
    def denormalise_distance(self, x: Union[float, np.ndarray, torch.Tensor]):
        return x * self.distance_std + self.distance_mean
