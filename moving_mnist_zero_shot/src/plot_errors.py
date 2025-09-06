import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _Path
CURRENT_DIR = _Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def main():
    metrics_path = Path(cfg.OUTPUT_ROOT) / "eval" / "frame_metrics.json"
    with open(metrics_path, "r") as f:
        data = json.load(f)
    
    results = data["frame_metrics"]
    
    # Plot error vs time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    for r in results:
        seq = r["seq"]
        mses = r["mse_by_frame"]
        psnrs = r["psnr_by_frame"]
        times = list(range(len(mses)))
        
        ax1.plot(times, mses, label=f"seq{seq}", alpha=0.7)
        ax2.plot(times, psnrs, label=f"seq{seq}", alpha=0.7)
    
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE vs Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("PSNR vs Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(cfg.OUTPUT_ROOT) / "eval" / "error_vs_time.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(str(out_path))


if __name__ == "__main__":
    main()
