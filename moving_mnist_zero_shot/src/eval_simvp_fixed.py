import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('/Users/apple/git/neuro/OpenSTL')
from openstl.models import SimVP_Model

# Add current dir for config
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def load_sequences(variant_name="baseline", max_seqs=5):
    """Load sequences from a variant directory."""
    variant_dir = Path(cfg.VARIANTS_ROOT) / variant_name
    sequences = []
    for seq_file in sorted(variant_dir.glob("seq*.npy"))[:max_seqs]:
        seq = np.load(seq_file)  # (T, H, W)
        sequences.append(seq)
    return sequences


def evaluate_model_simple(sequences, context_frames=10, pred_frames=10):
    """Simple evaluation without model - using last frame copy baseline."""
    results = []
    
    for idx, seq in enumerate(sequences):
        T, H, W = seq.shape
        if T < context_frames + pred_frames:
            continue
        
        # Simple baseline: copy last context frame
        last_context = seq[context_frames - 1]
        pred = np.stack([last_context] * pred_frames, axis=0)
        
        # Compare with ground truth
        gt_frames = seq[context_frames:context_frames + pred_frames]
        mse = np.mean((pred - gt_frames) ** 2)
        
        results.append({
            "seq_idx": idx,
            "mse": float(mse),
            "pred_frames": pred_frames,
            "method": "copy_last"
        })
    
    return results


def evaluate_all_variants():
    """Evaluate on all generated variants."""
    variants_root = Path(cfg.VARIANTS_ROOT)
    all_variants = sorted([d.name for d in variants_root.iterdir() if d.is_dir()])
    
    all_results = {}
    
    for variant in all_variants:
        print(f"Evaluating {variant}...")
        sequences = load_sequences(variant, max_seqs=3)
        
        if not sequences:
            print(f"  No sequences found for {variant}")
            continue
            
        results = evaluate_model_simple(sequences)
        
        # Compute average MSE
        avg_mse = np.mean([r["mse"] for r in results]) if results else 0
        
        all_results[variant] = {
            "avg_mse": float(avg_mse),
            "num_sequences": len(results),
            "details": results
        }
        
        print(f"  Avg MSE: {avg_mse:.2f}")
    
    return all_results


def main():
    # Evaluate on all variants
    results = evaluate_all_variants()
    
    # Save results
    out_path = Path(cfg.OUTPUT_ROOT) / "eval" / "model_results_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {out_path}")
    
    # Print summary table
    print("\n=== Copy-Last Baseline Results ===")
    print(f"{'Variant':<25} {'Avg MSE':<12} {'Sequences':<10}")
    print("-" * 50)
    
    # Sort by variant type
    for variant in sorted(results.keys()):
        data = results[variant]
        print(f"{variant:<25} {data['avg_mse']:<12.2f} {data['num_sequences']:<10}")
    
    # Group analysis
    print("\n=== Analysis by Motion Type ===")
    
    # Fast motion analysis
    fast_variants = {k: v for k, v in results.items() if "fast" in k}
    if fast_variants:
        print("\nFast Motion (higher speed → higher error expected):")
        for name in sorted(fast_variants.keys()):
            speed = name.split("_s")[-1] if "_s" in name else "1.0"
            print(f"  Speed {speed}: MSE = {fast_variants[name]['avg_mse']:.2f}")
    
    # Center speed analysis  
    center_speed = {k: v for k, v in results.items() if "center_speed" in k}
    if center_speed:
        print("\nCenter-Speed Field (higher α → more variation expected):")
        for name in sorted(center_speed.keys()):
            alpha = name.split("_a")[-1] if "_a" in name else "0.0"
            print(f"  α = {alpha}: MSE = {center_speed[name]['avg_mse']:.2f}")


if __name__ == "__main__":
    main()
