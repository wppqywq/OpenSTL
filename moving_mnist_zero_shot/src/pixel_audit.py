import os
import json
import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def find_mmnist_file(data_root: str) -> Path:
    root = Path(data_root)
    candidates = []
    for p in root.glob("**/*.npy"):
        name = p.name.lower()
        if "mnist" in name:
            candidates.append(p)
    if not candidates:
        for p in root.glob("**/*.npy"):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No .npy files found under {data_root}")
    preferred = [p for p in candidates if "test" in p.name.lower()]
    if preferred:
        return preferred[0]
    return candidates[0]


def standardize_layout(arr: np.ndarray):
    # Normalize arrays to accessor returning 2D frames (H, W)
    if arr.ndim == 5:
        # Expecting (T, N, H, W, C) or (N, T, H, W, C)
        T_first = arr.shape[0] < arr.shape[1]
        C = arr.shape[-1]
        if T_first:
            T, N = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                frame = arr[t, seq_idx]
                # frame: (H, W, C) -> use channel 0 to preserve integer intensities
                return frame[..., 0] if C > 1 else frame.squeeze()
            return get_frame, N, T
        else:
            N, T = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                frame = arr[seq_idx, t]
                return frame[..., 0] if C > 1 else frame.squeeze()
            return get_frame, N, T
    elif arr.ndim == 4:
        # (T, N, H, W) or (N, T, H, W)
        T_first = arr.shape[0] < arr.shape[1]
        if T_first:
            T, N = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return arr[t, seq_idx]
            return get_frame, N, T
        else:
            N, T = arr.shape[0], arr.shape[1]
            def get_frame(seq_idx, t):
                return arr[seq_idx, t]
            return get_frame, N, T
    elif arr.ndim == 3:
        # (T, H, W) or (N, H, W)
        if arr.shape[0] >= 10 and arr.shape[1] == 64 and arr.shape[2] == 64:
            T = arr.shape[0]
            def get_frame(seq_idx, t):
                return arr[t]
            return get_frame, 1, T
        else:
            N = arr.shape[0]
            def get_frame(seq_idx, t):
                return arr[seq_idx]
            return get_frame, N, 1
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def sample_indices(num_sequences: int, seq_len: int, num_seq_sample: int, frames_per_seq: int):
    rng = random.Random(0)
    seq_indices = list(range(num_sequences))
    if num_seq_sample < num_sequences:
        seq_indices = rng.sample(seq_indices, num_seq_sample)
    frame_indices_per_seq = {}
    for s in seq_indices:
        if frames_per_seq >= seq_len:
            frames = list(range(seq_len))
        else:
            step = max(1, math.floor(seq_len / frames_per_seq))
            frames = [min(i * step, seq_len - 1) for i in range(frames_per_seq)]
        frame_indices_per_seq[s] = frames
    return seq_indices, frame_indices_per_seq


def compute_unique_stats(frame: np.ndarray):
    values, counts = np.unique(frame, return_counts=True)
    unique_stats = {int(v): int(c) for v, c in zip(values, counts)}
    has_gray = len(values) > 2 or (len(values) == 2 and not ({0, 255} == set(int(v) for v in values)))
    return unique_stats, has_gray


def save_histogram(frame: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(3, 2))
    plt.hist(frame.flatten(), bins=256, range=(0, 255), color="black")
    plt.title(title)
    plt.xlabel("intensity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ensure_dir(cfg.OUTPUT_ROOT)
    audit_dir = Path(cfg.OUTPUT_ROOT) / "audit"
    ensure_dir(str(audit_dir))

    npy_path = find_mmnist_file(cfg.DATA_ROOT)
    arr = np.load(npy_path)
    get_frame, num_sequences, seq_len = standardize_layout(arr)

    S = min(cfg.AUDIT_SAMPLE_SEQUENCES, num_sequences)
    K = min(cfg.AUDIT_SAMPLE_FRAMES_PER_SEQUENCE, seq_len)
    seq_indices, frames_map = sample_indices(num_sequences, seq_len, S, K)

    results = {
        "data_file": str(npy_path),
        "num_sequences": int(num_sequences),
        "sequence_length": int(seq_len),
        "sampled_sequences": int(len(seq_indices)),
        "frames_per_sequence": int(K),
        "per_frame_stats": [],
        "any_gray_detected": False,
    }

    gray_found = False
    max_examples = 10
    saved = 0

    for s in seq_indices:
        for t in frames_map[s]:
            frame = get_frame(s, t)
            stats, has_gray = compute_unique_stats(frame)
            results["per_frame_stats"].append({
                "seq": int(s),
                "t": int(t),
                "unique_values": stats,
                "has_gray": bool(has_gray),
            })
            gray_found = gray_found or has_gray
            if cfg.AUDIT_SAVE_PLOTS and saved < max_examples:
                hist_path = audit_dir / f"hist_seq{s:04d}_t{t:03d}.png"
                save_histogram(frame, hist_path, f"seq {s} t {t}")
                saved += 1

    results["any_gray_detected"] = bool(gray_found)

    metrics_path = audit_dir / "unique_values.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(str(metrics_path))


if __name__ == "__main__":
    main()
