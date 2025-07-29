#!/usr/bin/env python3
"""
Evaluate Gaussian Enhanced Model and Compare with Baseline
This script evaluates the enhanced model and compares performance with single_fixation baseline.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Compatibility patches
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

os.environ['WANDB_DISABLED'] = 'true'

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openstl.methods import SimVP
import config


def load_gaussian_model(model_type='with_history'):
    """Load the trained Gaussian enhanced model"""
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    
    model_suffix = f"_{model_type}" if model_type in ['with_history', 'no_history'] else ""
    model_path = os.path.join("models", f"best_gaussian_enhanced{model_suffix}.pth")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, device
    
    # Create model with same architecture as training
    model = SimVP(
        in_shape=(4, 1, 32, 32),
        pre_seq_length=4,
        aft_seq_length=16,
        hid_S=config.model_hid_S,
        hid_T=config.model_hid_T,
        N_S=config.model_N_S,
        N_T=config.model_N_T,
        model_type=config.model_type,
        lr=1e-3,
        dataname='gaussian_enhanced',
        metrics=[],
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Gaussian model loaded successfully on {device}")
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, device


def load_baseline_model():
    """Load the baseline single_fixation model for comparison"""
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    
    baseline_path = config.baseline_experiment_path + "models/simvp_method_best.pth"
    if not os.path.exists(baseline_path):
        print(f"Baseline model not found: {baseline_path}")
        return None, device
    
    # Create baseline model
    model = SimVP(
        in_shape=(4, 1, 32, 32),
        pre_seq_length=4,
        aft_seq_length=16,
        hid_S=64,
        hid_T=512,
        N_S=4,
        N_T=8,
        model_type='gSTA',
        lr=1e-3,
        dataname='single_fixation',
        metrics=[],
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(baseline_path, map_location=device))
        model.eval()
        print(f"Baseline model loaded successfully")
        return model, device
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        return None, device


def load_test_data():
    """Load test data for both enhanced and baseline"""
    # Load enhanced test data
    gaussian_data = torch.load('data/test_data.pt')
    gaussian_frames = gaussian_data['frames'].permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
    
    # Load baseline test data
    baseline_data_path = config.baseline_experiment_path + "data/test_data.pt"
    if os.path.exists(baseline_data_path):
        baseline_data = torch.load(baseline_data_path)
        baseline_frames = baseline_data['frames'].permute(0, 2, 1, 3, 4)
    else:
        print("Baseline test data not found, using same coordinates for comparison")
        baseline_frames = None
    
    # Split into input and target
    PRE_SEQ_LEN = 4
    TOTAL_FRAMES = 20
    
    gaussian_input = gaussian_frames[:, :PRE_SEQ_LEN]
    gaussian_target = gaussian_frames[:, PRE_SEQ_LEN:TOTAL_FRAMES]
    
    if baseline_frames is not None:
        baseline_input = baseline_frames[:, :PRE_SEQ_LEN]
        baseline_target = baseline_frames[:, PRE_SEQ_LEN:TOTAL_FRAMES]
    else:
        baseline_input, baseline_target = None, None
    
    # Get coordinates (should be same for both)
    coords = gaussian_data['coords']
    
    print(f"Test data loaded: {gaussian_input.shape[0]} samples")
    print(f"Gaussian input: {gaussian_input.shape}, target: {gaussian_target.shape}")
    
    return {
        'gaussian_input': gaussian_input,
        'gaussian_target': gaussian_target,
        'baseline_input': baseline_input,
        'baseline_target': baseline_target,
        'coords': coords
    }


def extract_coordinates(heatmap):
    """Extract coordinates using argmax"""
    batch_size, seq_len = heatmap.shape[:2]
    coords = []
    
    for b in range(batch_size):
        batch_coords = []
        for t in range(seq_len):
            h = heatmap[b, t].squeeze()
            flat_idx = torch.argmax(h.view(-1))
            y = flat_idx // h.shape[1]
            x = flat_idx % h.shape[1]
            batch_coords.append([x.item(), y.item()])
        coords.append(batch_coords)
    
    return torch.tensor(coords)


def extract_coordinates_center_of_mass(heatmap):
    """Extract coordinates using center of mass (for Gaussian blobs)"""
    batch_size, seq_len = heatmap.shape[:2]
    coords = []
    
    for b in range(batch_size):
        batch_coords = []
        for t in range(seq_len):
            h = heatmap[b, t].squeeze()
            
            # Create coordinate grids
            y_coords = torch.arange(h.shape[0], dtype=torch.float32)
            x_coords = torch.arange(h.shape[1], dtype=torch.float32)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Normalize heatmap
            h_norm = h / (h.sum() + 1e-8)
            
            # Calculate center of mass
            center_y = (h_norm * yy).sum()
            center_x = (h_norm * xx).sum()
            
            batch_coords.append([center_x.item(), center_y.item()])
        coords.append(batch_coords)
    
    return torch.tensor(coords)


def evaluate_model_performance():
    """Comprehensive evaluation comparing Gaussian enhanced vs baseline"""
    print("="*60)
    print("GAUSSIAN ENHANCED vs BASELINE COMPARISON")
    print("="*60)
    
    # Load models
    gaussian_model, device = load_gaussian_model()
    baseline_model, _ = load_baseline_model()
    
    if gaussian_model is None:
        print("Cannot evaluate: Gaussian model not available")
        return
    
    # Load test data
    test_data = load_test_data()
    
    # Evaluate both models
    sample_indices = torch.randperm(test_data['gaussian_input'].shape[0])[:20]
    
    gaussian_errors = []
    baseline_errors = []
    gaussian_errors_com = []  # Center of mass extraction
    
    for idx in sample_indices:
        # Test Gaussian enhanced model
        gaussian_input = test_data['gaussian_input'][idx:idx+1].to(device)
        gaussian_target = test_data['gaussian_target'][idx:idx+1].to(device)
        
        with torch.no_grad():
            gaussian_pred = gaussian_model(gaussian_input)
        
        # Extract coordinates using both methods
        pred_coords_argmax = extract_coordinates(gaussian_pred.cpu())
        target_coords_argmax = extract_coordinates(gaussian_target.cpu())
        
        pred_coords_com = extract_coordinates_center_of_mass(gaussian_pred.cpu())
        target_coords_com = extract_coordinates_center_of_mass(gaussian_target.cpu())
        
        # Calculate errors for current sample
        for t in range(min(pred_coords_argmax.shape[1], target_coords_argmax.shape[1])):
            # Argmax method
            px, py = pred_coords_argmax[0, t]
            tx, ty = target_coords_argmax[0, t]
            error = torch.sqrt((px - tx)**2 + (py - ty)**2).item()
            gaussian_errors.append(error)
            
            # Center of mass method
            px_com, py_com = pred_coords_com[0, t]
            tx_com, ty_com = target_coords_com[0, t]
            error_com = torch.sqrt((px_com - tx_com)**2 + (py_com - ty_com)**2).item()
            gaussian_errors_com.append(error_com)
        
        # Test baseline model if available
        if baseline_model is not None and test_data['baseline_input'] is not None:
            baseline_input = test_data['baseline_input'][idx:idx+1].to(device)
            baseline_target = test_data['baseline_target'][idx:idx+1].to(device)
            
            with torch.no_grad():
                baseline_pred = baseline_model(baseline_input)
            
            baseline_pred_coords = extract_coordinates(baseline_pred.cpu())
            baseline_target_coords = extract_coordinates(baseline_target.cpu())
            
            for t in range(min(baseline_pred_coords.shape[1], baseline_target_coords.shape[1])):
                bpx, bpy = baseline_pred_coords[0, t]
                btx, bty = baseline_target_coords[0, t]
                baseline_error = torch.sqrt((bpx - btx)**2 + (bpy - bty)**2).item()
                baseline_errors.append(baseline_error)
    
    # Calculate baseline using repeat last frame
    repeat_baseline_errors = []
    for idx in sample_indices:
        gaussian_input = test_data['gaussian_input'][idx:idx+1]
        gaussian_target = test_data['gaussian_target'][idx:idx+1]
        
        # Get last input frame coordinates
        last_input_coord = extract_coordinates_center_of_mass(gaussian_input[:, -1:])
        target_coords = extract_coordinates_center_of_mass(gaussian_target)
        
        for t in range(target_coords.shape[1]):
            lx, ly = last_input_coord[0, 0]
            tx, ty = target_coords[0, t]
            error = torch.sqrt((lx - tx)**2 + (ly - ty)**2).item()
            repeat_baseline_errors.append(error)
    
    # Print results
    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"Gaussian Enhanced (argmax): {np.mean(gaussian_errors):.2f} ± {np.std(gaussian_errors):.2f} pixels")
    print(f"Gaussian Enhanced (center of mass): {np.mean(gaussian_errors_com):.2f} ± {np.std(gaussian_errors_com):.2f} pixels")
    print(f"Repeat last frame baseline: {np.mean(repeat_baseline_errors):.2f} ± {np.std(repeat_baseline_errors):.2f} pixels")
    
    if baseline_errors:
        print(f"Single fixation baseline: {np.mean(baseline_errors):.2f} ± {np.std(baseline_errors):.2f} pixels")
    
    # Calculate improvements
    gaussian_best = min(np.mean(gaussian_errors), np.mean(gaussian_errors_com))
    repeat_baseline = np.mean(repeat_baseline_errors)
    
    improvement_vs_repeat = (repeat_baseline - gaussian_best) / repeat_baseline * 100
    print(f"\nImprovement over repeat baseline: {improvement_vs_repeat:.1f}%")
    
    if baseline_errors:
        baseline_mean = np.mean(baseline_errors)
        improvement_vs_baseline = (baseline_mean - gaussian_best) / baseline_mean * 100
        print(f"Improvement over single fixation baseline: {improvement_vs_baseline:.1f}%")
    
    # Accuracy at different thresholds
    print("\n=== ACCURACY AT DIFFERENT THRESHOLDS ===")
    thresholds = [1.0, 2.0, 3.0, 5.0]
    
    for threshold in thresholds:
        gaussian_acc = np.mean(np.array(gaussian_errors_com) < threshold) * 100
        repeat_acc = np.mean(np.array(repeat_baseline_errors) < threshold) * 100
        
        print(f"< {threshold} pixel accuracy:")
        print(f"  Gaussian Enhanced: {gaussian_acc:.1f}%")
        print(f"  Repeat baseline: {repeat_acc:.1f}%")
        
        if baseline_errors:
            baseline_acc = np.mean(np.array(baseline_errors) < threshold) * 100
            print(f"  Single fixation baseline: {baseline_acc:.1f}%")
        print()
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist([gaussian_errors_com, repeat_baseline_errors], 
             bins=20, alpha=0.7, label=['Gaussian Enhanced', 'Repeat Baseline'])
    plt.xlabel('Pixel Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Comparison')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    frame_errors_gaussian = {}
    frame_errors_repeat = {}
    
    for i, error in enumerate(gaussian_errors_com):
        frame_idx = i % 16
        if frame_idx not in frame_errors_gaussian:
            frame_errors_gaussian[frame_idx] = []
        frame_errors_gaussian[frame_idx].append(error)
    
    for i, error in enumerate(repeat_baseline_errors):
        frame_idx = i % 16
        if frame_idx not in frame_errors_repeat:
            frame_errors_repeat[frame_idx] = []
        frame_errors_repeat[frame_idx].append(error)
    
    frames = sorted(frame_errors_gaussian.keys())
    gaussian_means = [np.mean(frame_errors_gaussian[f]) for f in frames]
    repeat_means = [np.mean(frame_errors_repeat[f]) for f in frames]
    
    plt.plot(frames, gaussian_means, 'o-', label='Gaussian Enhanced')
    plt.plot(frames, repeat_means, 's-', label='Repeat Baseline')
    plt.xlabel('Prediction Frame')
    plt.ylabel('Mean Pixel Error')
    plt.title('Temporal Error Evolution')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    accuracies_gaussian = [np.mean(np.array(gaussian_errors_com) < t) * 100 for t in thresholds]
    accuracies_repeat = [np.mean(np.array(repeat_baseline_errors) < t) * 100 for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    plt.bar(x - width/2, accuracies_gaussian, width, label='Gaussian Enhanced', alpha=0.8)
    plt.bar(x + width/2, accuracies_repeat, width, label='Repeat Baseline', alpha=0.8)
    
    plt.xlabel('Accuracy Threshold (pixels)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy at Different Thresholds')
    plt.xticks(x, [f'{t}px' for t in thresholds])
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.scatter(repeat_baseline_errors[:100], gaussian_errors_com[:100], alpha=0.6)
    plt.plot([0, max(repeat_baseline_errors)], [0, max(repeat_baseline_errors)], 'r--', label='Equal Performance')
    plt.xlabel('Repeat Baseline Error (pixels)')
    plt.ylabel('Gaussian Enhanced Error (pixels)')
    plt.title('Per-Frame Error Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/gaussian_vs_baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'gaussian_errors': gaussian_errors_com,
        'baseline_errors': baseline_errors,
        'repeat_baseline_errors': repeat_baseline_errors,
        'improvement_vs_repeat': improvement_vs_repeat
    }


def main():
    """Run comprehensive evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate Gaussian Enhanced model")
    parser.add_argument('--model_type', type=str, choices=['with_history', 'no_history'], 
                       default='with_history', help='Type of model to evaluate')
    args = parser.parse_args()
    
    os.makedirs('results', exist_ok=True)
    
    # Update function call to pass model_type
    results = evaluate_model_performance_with_type(args.model_type)
    
    if results:
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: results/gaussian_vs_baseline_comparison_{args.model_type}.png")
        print("="*60)


if __name__ == "__main__":
    main() 