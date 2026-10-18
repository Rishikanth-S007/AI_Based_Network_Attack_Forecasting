"""
Synthetic traffic-window generator, used ONLY to smoke-test the world
model before real data is wired in.

Produces sequences of window-level feature vectors shaped like what you'd
get from windowing a real CICFlowMeter CSV: flow counts, unique dest
ports/hosts, SYN:ACK ratio, IAT stats, byte volume, TTL variance,
retransmits. Each sequence has some probability of containing an injected,
gradually-progressing attack -- reconnaissance (rising unique-port count,
lots of unanswered SYNs) -> lateral movement (rising internal fan-out to
new hosts) -> command & control (regular small beacon-sized transfers,
IAT variance collapses).

This is NOT real traffic and none of these numeric ranges are calibrated
to any real dataset -- it exists purely so the encoder/transition/decoder
code can be built and tested end to end. Swap `make_dataset` for a loader
over real windowed CIC-IDS2018 / CTU-13 features and nothing in model.py
or train.py needs to change, as long as it returns the same shapes.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_NAMES = [
    "flow_count", "unique_dst_ports", "unique_dst_hosts",
    "syn_ack_ratio", "iat_mean", "iat_var",
    "bytes_out", "bytes_in", "ttl_var", "retransmits",
]
N_FEATURES = len(FEATURE_NAMES)

STAGE_BENIGN, STAGE_RECON, STAGE_LATERAL, STAGE_C2, STAGE_EXFIL = range(5)


def _benign_window(rng):
    return np.array([
        rng.poisson(20),
        rng.poisson(3),
        rng.poisson(2),
        rng.normal(1.0, 0.15),
        rng.normal(0.5, 0.1),
        rng.normal(0.05, 0.02),
        rng.normal(50_000, 8_000),
        rng.normal(60_000, 8_000),
        rng.normal(1.4, 0.2),
        rng.poisson(1),
    ], dtype=np.float32)


def _recon_window(rng, intensity):
    w = _benign_window(rng)
    w[0] += rng.poisson(40 * intensity)             # flow_count spikes
    w[1] += rng.poisson(60 * intensity)             # unique_dst_ports -- the scan signature
    w[3] = rng.normal(3.5, 0.4) * intensity + 1      # SYNs >> ACKs, many unanswered probes
    return w


def _lateral_window(rng, intensity):
    w = _benign_window(rng)
    w[2] += rng.poisson(15 * intensity)              # unique_dst_hosts -- internal fan-out
    w[0] += rng.poisson(15 * intensity)
    return w


def _c2_window(rng, intensity):
    w = _benign_window(rng)
    w[5] = rng.normal(0.005, 0.001)                  # iat_var collapses -- regular beacon timing
    w[6] = rng.normal(4_000, 300) * intensity         # small, steady bytes_out
    return w


def make_dataset(n_samples=512, seq_len=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, seq_len, N_FEATURES), dtype=np.float32)
    y_stage = np.zeros((n_samples, seq_len), dtype=np.int64)

    for i in range(n_samples):
        for t in range(seq_len):
            X[i, t] = _benign_window(rng)

        if rng.random() < 0.6:  # 60% of sequences contain an injected attack
            recon_len, lateral_len, c2_len = 5, 5, 6
            total = recon_len + lateral_len + c2_len
            latest_start = seq_len - total - 1
            if latest_start > 3:
                start = rng.integers(3, latest_start)
                intensity = rng.uniform(0.6, 1.4)
                for k in range(recon_len):
                    X[i, start + k] = _recon_window(rng, intensity)
                    y_stage[i, start + k] = STAGE_RECON
                for k in range(lateral_len):
                    X[i, start + recon_len + k] = _lateral_window(rng, intensity)
                    y_stage[i, start + recon_len + k] = STAGE_LATERAL
                for k in range(c2_len):
                    X[i, start + recon_len + lateral_len + k] = _c2_window(rng, intensity)
                    y_stage[i, start + recon_len + lateral_len + k] = STAGE_C2

    y_attack = (y_stage != STAGE_BENIGN).astype(np.float32)

    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std

    return X, y_attack, y_stage


class TrafficWindowDataset(Dataset):
    def __init__(self, X, y_attack, y_stage):
        self.X = torch.from_numpy(X)
        self.y_attack = torch.from_numpy(y_attack)
        self.y_stage = torch.from_numpy(y_stage)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_attack[idx], self.y_stage[idx]

    def raw(self):
        return self.X, self.y_attack, self.y_stage
