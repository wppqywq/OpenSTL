import json
import math
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path
CURRENT_DIR = _Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(mse_val: float, max_val: float = 255.0) -> float:
    if mse_val == 0:
        return float('inf')
    return float(20 * math.log10(max_val / math.sqrt(mse_val)))


def load_original_frames():
    # Load original test set
    root = Path(cfg.DATA_ROOT)
    candidates = list(root.glob("**/*.npy"))
    test_files = [p for p in candidates if "test" in p.name.lower() and "mnist" in p.name.lower()]
    if not test_files:
        raise FileNotFoundError("No test file found")
    arr = np.load(test_files[0])
    # Handle 5D array (T, N, H, W, C) or (N, T, H, W, C)
    if arr.ndim == 5:
        if arr.shape[0] < arr.shape[1]:  # T first
            arr = arr[:, :, :, :, 0]  # Take first channel
        else:  # N first
            arr = arr[:, :, :, :, 0].transpose(1, 0, 2, 3)  # to (T, N, H, W)
    elif arr.ndim == 4:
        if arr.shape[0] < arr.shape[1]:  # T first
            pass
        else:  # N first
            arr = arr.transpose(1, 0, 2, 3)
    return arr  # (T, N, H, W)


def main():
    original = load_original_frames()  # (T, N, H, W)
    T_orig, N_orig = original.shape[0], original.shape[1]
    
    variant_dir = Path(cfg.VARIANTS_ROOT) / "baseline"
    seq_files = sorted(variant_dir.glob("seq*.npy"))
    
    results = []
    for seq_file in seq_files[:5]:  # Sample first 5 sequences
        seq_idx = int(seq_file.stem[3:])
        if seq_idx >= N_orig:
            continue
        
        generated = np.load(seq_file)  # (T, H, W)
        T_gen = min(generated.shape[0], T_orig)
        
        frame_mses = []
        frame_psnrs = []
        for t in range(T_gen):
            orig_frame = original[t, seq_idx].astype(np.float32)
            gen_frame = generated[t].astype(np.float32)
            
            m = mse(orig_frame, gen_frame)
            p = psnr(m)
            frame_mses.append(m)
            frame_psnrs.append(p if p != float('inf') else 100.0)
        
        results.append({
            "seq": int(seq_idx),
            "avg_mse": float(np.mean(frame_mses)),
            "avg_psnr": float(np.mean(frame_psnrs)),
            "mse_by_frame": frame_mses[:10],  # First 10 frames
            "psnr_by_frame": frame_psnrs[:10],
        })
    
    out_path = Path(cfg.OUTPUT_ROOT) / "eval" / "frame_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"frame_metrics": results}, f, indent=2)
    
    print(str(out_path))


if __name__ == "__main__":
    main()
