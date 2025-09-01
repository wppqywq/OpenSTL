#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import argparse
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Predefined experiments from experiment_config.py
# Change this list to run different experiments
EXPERIMENTS = [
    #"gauss_pixel_standard",            
    # "gauss_pixel_focal",
    # "gauss_pixel_dice", 
    # "gauss_pixel_weighted",
    # Cross-domain experiments (uncomment to run):
    "geom_pixel_weighted_mse",    # Pixel with MSE loss
    # "gauss_heat_focal_bce",        # Heatmap with BCE loss  
    # "geom_heat_weighted_bce",     # Heatmap with BCE loss
]

# Default concurrency (can be overridden via CLI)
MAX_WORKERS = 2  # default sequential execution to keep RAM usage low on laptops


def parse_cli() -> argparse.Namespace:
    """Parse command-line arguments for this launcher."""
    p = argparse.ArgumentParser(description="Run a curated sweep of experiments")
    p.add_argument("--max_workers", type=int, default=MAX_WORKERS, help="Maximum concurrent experiments")
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs for each experiment")
    p.add_argument("--lr", type=float, default=None, help="Learning rate override")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size override")
    return p.parse_args()


def build_cmd(exp_name: str, cli_args: argparse.Namespace) -> List[str]:
    """Convert experiment name to a CLI command list."""
    cmd = [
        "python", "train.py",
        "--exp", exp_name,
    ]
    
    # Add common overrides
    cmd.extend(["--epochs", str(cli_args.epochs)])
    
    if cli_args.lr is not None:
        cmd.extend(["--lr", str(cli_args.lr)])
        
    cmd.extend(["--batch_size", str(cli_args.batch_size)])

    # Resume from best checkpoint in results/<name> if it exists
    best_ckpt = RESULTS_DIR / exp_name / "best_checkpoint.pth"
    if best_ckpt.exists():
        cmd.extend(["--resume", str(best_ckpt)])

    return cmd


def run_experiment(exp_name: str, cli_args: argparse.Namespace):
    """Launch a single training subprocess and stream output to log file."""
    cmd = build_cmd(exp_name, cli_args)
    log_path = RESULTS_DIR / exp_name / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Run training
    start_time = time.time()
    with log_path.open("w") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, stdout=log_file, stderr=log_file, cwd=ROOT)
    train_duration = time.time() - start_time
    
    training_success = proc.returncode == 0
    
    # Auto-evaluate if training succeeded
    if training_success:
        best_checkpoint = RESULTS_DIR / exp_name / "best_checkpoint.pth"
        if best_checkpoint.exists():
            print(f"[{exp_name}] Training completed, starting evaluation...")
            eval_output_dir = RESULTS_DIR / exp_name / "eval"
            
            eval_start_time = time.time()
            eval_cmd = [
                "python", "eval_clean.py",
                "--checkpoint", str(best_checkpoint),
                "--output_dir", str(eval_output_dir),
                "--num_samples", "5"
            ]
            
            try:
                eval_proc = subprocess.run(eval_cmd, capture_output=False, cwd=ROOT)
                eval_success = eval_proc.returncode == 0
                eval_duration = time.time() - eval_start_time
                
                if eval_success:
                    print(f"[{exp_name}] Evaluation completed successfully! ({eval_duration:.1f}s)")
                else:
                    print(f"[{exp_name}] Evaluation failed. Check logs for details.")
                    
            except Exception as e:
                print(f"[{exp_name}] Evaluation error: {e}")
                eval_success = False
        else:
            print(f"[{exp_name}] No best checkpoint found for evaluation.")
            eval_success = False
    else:
        print(f"[{exp_name}] Training failed, skipping evaluation.")
        eval_success = False

    total_duration = time.time() - start_time
    return exp_name, training_success, total_duration


def main() -> None:
    cli_args = parse_cli()

    print(f"Launching {len(EXPERIMENTS)} experiments with up to {cli_args.max_workers} concurrent runners...")
    print(f"Experiments: {', '.join(EXPERIMENTS)}")
    
    results: List[Tuple[str, bool, float]] = []

    with ThreadPoolExecutor(max_workers=cli_args.max_workers) as executor:
        future_to_name = {
            executor.submit(run_experiment, exp_name, cli_args): exp_name for exp_name in EXPERIMENTS
        }
        for fut in as_completed(future_to_name):
            name, training_ok, duration = fut.result()
            status = "SUCCESS" if training_ok else "FAIL"
            print(f"{name}: {status} ({duration:.1f}s total)")
            results.append((name, training_ok, duration))

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n==== SUMMARY ====")
    print(f"Passed: {passed}/{len(results)}")
    if passed != len(results):
        print("Some experiments failed. Check their individual log files for details.")
    else:
        print("All experiments completed successfully!")


if __name__ == "__main__":
    main()
