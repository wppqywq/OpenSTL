#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def eval_experiment(exp_name):
    """Evaluate a single experiment using eval_unified.py"""
    checkpoint_path = f"results/{exp_name}/best_checkpoint.pth"
    output_dir = f"results/{exp_name}/evaluation_unified"
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: {exp_name} - no checkpoint found")
        return False
    
    print(f"Evaluating {exp_name}...")
    
    cmd = [
        'python', 'eval_clean.py',
        '--checkpoint', checkpoint_path,
        '--output_dir', output_dir,
        '--num_samples', '3'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"SUCCESS: {exp_name} evaluation complete")
            return True
        else:
            print(f"ERROR: {exp_name} evaluation failed")
            print(f"  {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {exp_name} evaluation timed out")
        return False
    except Exception as e:
        print(f"EXCEPTION: {exp_name} evaluation error: {e}")
        return False

def main():
    # All experiments to evaluate
    all_experiments = [
        # Gauss experiments
        'Full_gauss_pixel_WeightedBCE',
        'Full_gauss_pixel_FocalBCE', 
        'Full_gauss_pixel_DiceBCE',
        'Full_gauss_heat_KL',
        'Full_gauss_heat_WeightedMSE',
        'Full_gauss_heat_EMD',
        'Full_gauss_coord_Huber',
        'Full_gauss_coord_Polar',
        'Full_gauss_coord_L1Cosine',
        
        # Geom experiments
        'Full_geom_pixel_WeightedBCE',
        'Full_geom_pixel_FocalBCE',
        'Full_geom_pixel_DiceBCE', 
        'Full_geom_heat_KL',
        'Full_geom_heat_WeightedMSE',
        'Full_geom_heat_EMD',
        'Full_geom_coord_Huber',
        'Full_geom_coord_Polar',
        'Full_geom_coord_L1Cosine'
    ]
    
    print(f"Starting evaluation of {len(all_experiments)} experiments")
    print("Using eval_clean.py with simplified metrics and representation-specific visualizations")
    print("")
    
    success_count = 0
    failed_experiments = []
    
    for exp in all_experiments:
        if eval_experiment(exp):
            success_count += 1
        else:
            failed_experiments.append(exp)
        print("")
    
    print("="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  {exp}")
    
    print(f"\nResults saved to: results/*/evaluation_unified/")
    print(f"Visualizations in: results/*/evaluation_unified/visualizations/")

if __name__ == "__main__":
    main()