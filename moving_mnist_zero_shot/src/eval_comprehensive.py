import json
from pathlib import Path
import numpy as np

import sys
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import config as cfg


def load_sequences(variant_name, max_seqs=10):
    """Load sequences from variant directory."""
    variant_dir = Path(cfg.VARIANTS_ROOT) / variant_name
    sequences = []
    seq_files = sorted(variant_dir.glob("seq*.npy"))[:max_seqs]
    
    for seq_file in seq_files:
        seq = np.load(seq_file)  # (T, H, W)
        sequences.append(seq)
    
    return sequences


def evaluate_baseline_methods(sequences, context_frames=10, pred_frames=10):
    """Evaluate multiple baseline methods."""
    results = {}
    
    # Method 1: Copy Last Frame
    copy_last_mses = []
    for seq in sequences:
        T = seq.shape[0]
        if T < context_frames + pred_frames:
            continue
        
        last_frame = seq[context_frames - 1]
        pred = np.stack([last_frame] * pred_frames)
        gt = seq[context_frames:context_frames + pred_frames]
        mse = np.mean((pred - gt) ** 2)
        copy_last_mses.append(mse)
    
    results["copy_last"] = float(np.mean(copy_last_mses)) if copy_last_mses else 0.0
    
    # Method 2: Linear Extrapolation
    linear_mses = []
    for seq in sequences:
        T = seq.shape[0]
        if T < context_frames + pred_frames:
            continue
        
        # Use last two frames to extrapolate
        frame_t1 = seq[context_frames - 2].astype(np.float32)
        frame_t2 = seq[context_frames - 1].astype(np.float32)
        velocity = frame_t2 - frame_t1
        
        pred = []
        for t in range(pred_frames):
            next_frame = frame_t2 + velocity * (t + 1)
            next_frame = np.clip(next_frame, 0, 255)
            pred.append(next_frame)
        pred = np.stack(pred)
        
        gt = seq[context_frames:context_frames + pred_frames]
        mse = np.mean((pred - gt) ** 2)
        linear_mses.append(mse)
    
    results["linear_extrap"] = float(np.mean(linear_mses)) if linear_mses else 0.0
    
    # Method 3: Mean of Context
    mean_context_mses = []
    for seq in sequences:
        T = seq.shape[0]
        if T < context_frames + pred_frames:
            continue
        
        mean_frame = np.mean(seq[:context_frames], axis=0)
        pred = np.stack([mean_frame] * pred_frames)
        gt = seq[context_frames:context_frames + pred_frames]
        mse = np.mean((pred - gt) ** 2)
        mean_context_mses.append(mse)
    
    results["mean_context"] = float(np.mean(mean_context_mses)) if mean_context_mses else 0.0
    
    return results


def main():
    variants_root = Path(cfg.VARIANTS_ROOT)
    all_variants = sorted([d.name for d in variants_root.iterdir() if d.is_dir()])
    
    all_results = {}
    
    for variant in all_variants:
        print(f"Evaluating {variant}...")
        sequences = load_sequences(variant, max_seqs=10)
        
        if not sequences:
            continue
        
        results = evaluate_baseline_methods(sequences)
        all_results[variant] = results
        
        print(f"  Copy-last MSE: {results['copy_last']:.2f}")
        print(f"  Linear MSE: {results['linear_extrap']:.2f}")
        print(f"  Mean MSE: {results['mean_context']:.2f}")
    
    # Save results
    out_path = Path(cfg.OUTPUT_ROOT) / "eval" / "comprehensive_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {out_path}")
    
    # Print comparison table
    print("\n=== Method Comparison ===")
    print(f"{'Variant':<20} {'Copy-Last':<12} {'Linear':<12} {'Mean':<12}")
    print("-" * 60)
    
    for variant in sorted(all_results.keys()):
        r = all_results[variant]
        print(f"{variant:<20} {r['copy_last']:<12.2f} {r['linear_extrap']:<12.2f} {r['mean_context']:<12.2f}")
    
    # Analyze which method works best for each motion type
    print("\n=== Best Method by Motion Type ===")
    
    # Fast motion
    fast_variants = [v for v in all_results.keys() if "fast" in v]
    if fast_variants:
        print("\nFast Motion:")
        for v in sorted(fast_variants):
            r = all_results[v]
            best_method = min(r, key=r.get)
            print(f"  {v}: {best_method} (MSE={r[best_method]:.2f})")
    
    # Center-speed
    center_speed = [v for v in all_results.keys() if "center_speed" in v]
    if center_speed:
        print("\nCenter-Speed Field:")
        for v in sorted(center_speed):
            r = all_results[v]
            best_method = min(r, key=r.get)
            print(f"  {v}: {best_method} (MSE={r[best_method]:.2f})")
    
    # Overall best
    print("\n=== Overall Performance ===")
    for method in ["copy_last", "linear_extrap", "mean_context"]:
        avg_mse = np.mean([r[method] for r in all_results.values()])
        print(f"{method}: Average MSE = {avg_mse:.2f}")


if __name__ == "__main__":
    main()
