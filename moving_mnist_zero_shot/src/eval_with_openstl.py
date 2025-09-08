import json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.append('/Users/apple/git/neuro/OpenSTL')

# Import OpenSTL training API
from openstl.api import BaseExperiment
from openstl.utils import create_parser

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def prepare_test_data(variant_name="baseline", max_seqs=100):
    """Prepare test data from variant."""
    variant_dir = Path(cfg.VARIANTS_ROOT) / variant_name
    sequences = []
    
    for seq_file in sorted(variant_dir.glob("seq*.npy"))[:max_seqs]:
        seq = np.load(seq_file)  # (T, H, W)
        # Add channel dimension: (T, H, W) -> (T, 1, H, W)
        seq = seq[:, np.newaxis, :, :]
        sequences.append(seq)
    
    if not sequences:
        return None
    
    # Stack to (N, T, C, H, W)
    data = np.stack(sequences, axis=0)
    return data


def evaluate_pretrained_model(checkpoint_path, variant_name="baseline"):
    """Evaluate a pretrained OpenSTL model on a variant."""
    
    # Prepare data
    test_data = prepare_test_data(variant_name, max_seqs=10)
    if test_data is None:
        return None
    
    N, T, C, H, W = test_data.shape
    context_frames = min(10, T // 2)
    pred_frames = min(10, T - context_frames)
    
    # Split into input and target
    input_data = test_data[:, :context_frames]  # (N, T_in, C, H, W)
    target_data = test_data[:, context_frames:context_frames+pred_frames]  # (N, T_out, C, H, W)
    
    # Simple MSE calculation (without loading model for now)
    # Using persistence baseline: repeat last input frame
    last_frame = input_data[:, -1:, :, :, :]  # (N, 1, C, H, W)
    pred = np.repeat(last_frame, pred_frames, axis=1)  # (N, T_out, C, H, W)
    
    mse = np.mean((pred - target_data) ** 2)
    
    return {
        "variant": variant_name,
        "mse": float(mse),
        "num_sequences": N,
        "context_frames": context_frames,
        "pred_frames": pred_frames
    }


def main():
    # Test on multiple variants
    variants = ["baseline", "fast_s1.5", "fast_s2.0", "fast_s3.0", 
                "center_speed_a0.3", "center_speed_a0.6", "center_speed_a0.9",
                "center_direction"]
    
    results = []
    for variant in variants:
        print(f"Evaluating {variant}...")
        result = evaluate_pretrained_model(None, variant)
        if result:
            results.append(result)
            print(f"  MSE: {result['mse']:.4f}")
    
    # Save results
    out_path = Path(cfg.OUTPUT_ROOT) / "eval" / "openstl_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    
    print(f"\nSaved to: {out_path}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"{'Variant':<20} {'MSE':<10}")
    print("-" * 30)
    for r in results:
        print(f"{r['variant']:<20} {r['mse']:<10.4f}")


if __name__ == "__main__":
    main()
