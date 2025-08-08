#!/usr/bin/env python3
"""
Run a curated set of experiments with controlled parallelism and per-experiment logs.

This script launches the training script (``train.py``) for a fixed list of
experiments while ensuring that no more than two experiments run in parallel.
All stdout/stderr from each run is streamed into a dedicated log file under
``results/<exp_name>/train.log`` so that progress can be monitored in real-time.

Experiments
===========
1. Gaussian dataset, pixel representation, MSE loss
2. Geometry dataset, coord representation, Polar loss (w_dir=1, w_mag=2)
3. Geometry dataset, coord representation, Polar loss (w_dir=2, w_mag=1)
4. Geometry dataset, heatmap representation, Weighted-MSE loss (pos_weight=200)
5. Geometry dataset, pixel representation, Dice-BCE loss
6. Geometry dataset, pixel representation, Focal-BCE loss
7. Geometry dataset, pixel representation, Weighted-BCE loss
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import argparse
import os
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Common training hyper-parameters
COMMON_ARGS = [
    "--epochs", "100",
    "--patience", "10",
    "--generate_data",  # Reuse existing data or generate if needed
]

EXPERIMENTS = [
    # {
    #     "name": "gauss_pixel_MSE",
    #     "data": "gauss",
    #     "repr": "pixel",
    #     "loss": "mse",
    #     "extra": {},
    # },
    {
        "name": "geom_pixel_DiceBCE",
        "data": "geom_simple",
        "repr": "pixel",
        "loss": "dice_bce",
        "extra": {},
    },
    {
        "name": "geom_pixel_FocalBCE",
        "data": "geom_simple",
        "repr": "pixel",
        "loss": "focal_bce",
        "extra": {},
    },
    {
        "name": "geom_pixel_WeightedBCE",
        "data": "geom_simple",
        "repr": "pixel",
        "loss": "weighted_bce",
        "extra": {"pos_weight": 2000.0},
    },
    # Coordinate representation: add Hybrid variants aligned with current experiments
    {
        "name": "geom_coord_Huber",
        "data": "geom_simple",
        "repr": "coord",
        "loss": "huber",
        "extra": {},  # absolute coordinates (default)
    },
    {
        "name": "geom_coord_HybridL1Disp",
        "data": "geom_simple",
        "repr": "coord",
        "loss": "l1",
        "extra": {"coord_mode": "displacement"},
    },
    {
        "name": "geom_coord_Polar11",
        "data": "geom_simple",
        "repr": "coord",
        "loss": "polar_decoupled",
        "extra": {"w_dir": 1.5, "w_mag": 1.0},
    },
    # Heatmap representation: add EMD and KL losses
    {
        "name": "geom_heat_EMD",
        "data": "geom_simple",
        "repr": "heat",
        "loss": "emd",
        "extra": {},
    },
    {
        "name": "geom_heat_KL",
        "data": "geom_simple",
        "repr": "heat",
        "loss": "kl",
        "extra": {},
    },
    {
        "name": "geom_heat_WeightedMSE100",
        "data": "geom_simple",
        "repr": "heat",
        "loss": "weighted_mse",
        "extra": {"pos_weight": 100.0},
    },
]

# Default concurrency (can be overridden via CLI)
MAX_WORKERS = 2  # default sequential execution to keep RAM usage low on laptops


def parse_cli() -> argparse.Namespace:
    """Parse command-line arguments for this launcher."""
    p = argparse.ArgumentParser(description="Run a curated sweep of experiments")
    p.add_argument("--max_workers", type=int, default=MAX_WORKERS, help="Maximum concurrent experiments")
    p.add_argument("--num_workers", type=int, default=max(0, (os.cpu_count() or 2) // 2), help="PyTorch DataLoader workers per experiment")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size to pass to each training job")
    # Optional dataset size overrides (geom_simple only)
    p.add_argument("--train_size", type=int, default=2000, help="Override number of training samples (geom)")
    p.add_argument("--val_size", type=int, default=200, help="Override number of validation samples (geom)")
    p.add_argument("--test_size", type=int, default=200, help="Override number of test samples (geom)")
    return p.parse_args()


def build_cmd(exp: Dict[str, Any], cli_args: argparse.Namespace) -> List[str]:
    """Convert experiment definition to a CLI command list."""
    cmd = [
        "python", "train.py",
        "--data", exp["data"],
        "--repr", exp["repr"],
        "--loss", exp["loss"],
        "--exp_name", exp["name"],
    ]
    cmd.extend(COMMON_ARGS)

    # Append extra parameters (if any)
    for key, value in exp["extra"].items():
        cmd.extend([f"--{key}", str(value)])

    # Resource-related overrides (num_workers only – batch_size already added above)
    cmd.extend(["--num_workers", str(cli_args.num_workers)])

    # Dataset size overrides (geom_simple only)
    if cli_args.train_size is not None:
        cmd.extend(["--train_size", str(cli_args.train_size)])
    if cli_args.val_size is not None:
        cmd.extend(["--val_size", str(cli_args.val_size)])
    if cli_args.test_size is not None:
        cmd.extend(["--test_size", str(cli_args.test_size)])

    # Resume from best checkpoint in results/<name> if it exists
    best_ckpt = RESULTS_DIR / exp["name"] / "best_checkpoint.pth"
    if best_ckpt.exists():
        cmd.extend(["--resume", str(best_ckpt)])

    return cmd


def run_experiment(exp: Dict[str, Any], cli_args: argparse.Namespace):
    """Launch a single training subprocess and stream output to log file."""
    cmd = build_cmd(exp, cli_args)
    log_path = RESULTS_DIR / exp["name"] / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    with log_path.open("w") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, stdout=log_file, stderr=log_file)
    duration = time.time() - start_time

    return exp["name"], proc.returncode == 0, duration


def main() -> None:
    cli_args = parse_cli()

    print(
        f"Launching {len(EXPERIMENTS)} experiments with up to {cli_args.max_workers} concurrent",
        "runners…"
    )
    results: List[Tuple[str, bool, float]] = []
    # Monkey-patch global COMMON_ARGS used by build_cmd
    global COMMON_ARGS  # noqa: PLW0603  # runtime patch acceptable for this small script

    # Update batch size in COMMON_ARGS dynamically (remove old if present)
    dynamic_common: List[str] = [
        "--batch_size",
        str(cli_args.batch_size),
    ] + COMMON_ARGS
    COMMON_ARGS = dynamic_common

    with ThreadPoolExecutor(max_workers=cli_args.max_workers) as executor:
        future_to_name = {
            executor.submit(run_experiment, exp, cli_args): exp["name"] for exp in EXPERIMENTS
        }
        for fut in as_completed(future_to_name):
            name, ok, duration = fut.result()
            status = "SUCCESS" if ok else "FAIL"
            print(f"{name}: {status} ({duration:.1f}s)")
            results.append((name, ok, duration))

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n==== SUMMARY ====")
    print(f"Passed: {passed}/{len(results)}")
    if passed != len(results):
        print("Some experiments failed. Check their individual log files for details.")
    else:
        print("All experiments completed successfully!")


if __name__ == "__main__":
    main()
