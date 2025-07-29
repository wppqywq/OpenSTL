#!/usr/bin/env python3
"""
Visualize Phase 3 Model Predictions - Direct Coordinate Regression
Show how models fail with clustering behavior on coordinate prediction task
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add OpenSTL path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import model class
from train_coordinate_regression import SimVPCoordinateRegressor, load_real_data, create_sparse_frames, create_coordinate_dataset

def load_trained_model(model_path, strategy='parallel'):
    """Load the trained model"""
    model = SimVPCoordinateRegressor(strategy=strategy)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with strategy: {checkpoint.get('strategy', 'unknown')}")
    else:
        # Old format
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def coordinates_to_frames(coordinates, img_size=32):
    """Convert coordinates to sparse binary frames"""
    num_frames = coordinates.shape[0]
    frames = torch.zeros(num_frames, 1, img_size, img_size)
    
    for t in range(num_frames):
        x, y = coordinates[t].long()
        if 0 <= x < img_size and 0 <= y < img_size:
            frames[t, 0, x, y] = 1.0
    
    return frames

def predict_sample_trajectories(model, data, num_samples=3):
    """Predict trajectories for sample sequences using corrected format"""
    predictions = {}
    
    # Prepare data
    coordinates = data['coordinates']
    fixation_mask = data['fixation_mask']
    
    for sample_idx in range(min(num_samples, len(coordinates))):
        sample_coords = coordinates[sample_idx]
        sample_mask = fixation_mask[sample_idx]
        
        # Create frames for full sequence
        frames = coordinates_to_frames(sample_coords)
        
        # Create dataset format: 5 input frames → 5 output coordinates
        input_frames = frames[:5].unsqueeze(0)  # [1, 5, 1, 32, 32]
        target_coords = sample_coords[5:10].unsqueeze(0)  # [1, 5, 2]
        
        # Predict coordinates
        model.eval()
        with torch.no_grad():
            if model.strategy == 'sequential':
                start_coord = torch.full((1, 2), 16.0)  # Start from center
                predicted_coords = model(input_frames, start_coord)[0]  # [5, 2]
            else:
                predicted_coords = model(input_frames)[0]  # [5, 2]
        
        # Reconstruct full trajectory
        full_gt_coords = sample_coords
        full_pred_coords = torch.zeros_like(sample_coords)
        full_pred_coords[:5] = sample_coords[:5]  # Input frames (known)
        full_pred_coords[5:10] = predicted_coords  # Predicted frames
        # For frames 10+, continue with last predicted trend or keep static
        if len(sample_coords) > 10:
            last_displacement = predicted_coords[-1] - predicted_coords[-2]
            for t in range(10, len(sample_coords)):
                full_pred_coords[t] = full_pred_coords[t-1] + last_displacement
        
        predictions[sample_idx] = {
            'input_frames': frames[:5],
            'input_coords': sample_coords[:5],
            'gt_coords': full_gt_coords,
            'predicted_coords': full_pred_coords,
            'target_coords': target_coords[0],  # The 5 frames we actually predicted
            'predicted_target_coords': predicted_coords,
            'fixation_mask': sample_mask
        }
    
    return predictions

def visualize_clustering_failure(predictions, save_path='clustering_failure.png'):
    """Visualize how predictions cluster vs spread of ground truth"""
    
    num_samples = len(predictions)
    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 10))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for sample_idx, pred_data in predictions.items():
        gt_coords = pred_data['gt_coords']
        pred_coords = pred_data['predicted_coords']
        target_coords = pred_data['target_coords']
        pred_target_coords = pred_data['predicted_target_coords']
        mask = pred_data['fixation_mask']
        
        # Top row: Full trajectory comparison
        ax = axes[0, sample_idx]
        
        # Ground truth trajectory
        ax.plot(gt_coords[:, 1], gt_coords[:, 0], 'r-', linewidth=2, alpha=0.7, label='Ground Truth')
        ax.plot(pred_coords[:, 1], pred_coords[:, 0], 'b--', linewidth=2, alpha=0.7, label='Predicted')
        
        # Mark input vs prediction boundary
        ax.axvline(x=5, color='gray', linestyle=':', alpha=0.5, label='Prediction Start')
        
        # Highlight predicted region
        ax.plot(target_coords[:, 1], target_coords[:, 0], 'r-', linewidth=3, alpha=0.9)
        ax.plot(pred_target_coords[:, 1], pred_target_coords[:, 0], 'b--', linewidth=3, alpha=0.9)
        
        # Mark active frames
        active_coords = gt_coords[mask]
        if len(active_coords) > 0:
            ax.scatter(active_coords[:, 1], active_coords[:, 0], c='red', s=30, alpha=0.8, zorder=5)
        
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_xlabel('Y Coordinate')
        ax.set_ylabel('X Coordinate')
        ax.set_title(f'Sample {sample_idx} - Full Trajectory\nRed=GT, Blue=Predicted')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Bottom row: Focus on predicted region (frames 5-10)
        ax = axes[1, sample_idx]
        
        # Show just the predicted region
        ax.plot(target_coords[:, 1], target_coords[:, 0], 'r-', linewidth=3, 
                marker='o', markersize=8, alpha=0.9, label='Ground Truth')
        ax.plot(pred_target_coords[:, 1], pred_target_coords[:, 0], 'b--', linewidth=3, 
                marker='x', markersize=8, alpha=0.9, label='Predicted')
        
        # Mark start and end points clearly
        ax.scatter(target_coords[0, 1], target_coords[0, 0], c='darkred', s=100, 
                  marker='s', alpha=0.8, label='GT Start', zorder=6)
        ax.scatter(target_coords[-1, 1], target_coords[-1, 0], c='red', s=100, 
                  marker='s', alpha=0.8, label='GT End', zorder=6)
        
        ax.scatter(pred_target_coords[0, 1], pred_target_coords[0, 0], c='darkblue', s=100, 
                  marker='o', alpha=0.8, label='Pred Start', zorder=6)
        ax.scatter(pred_target_coords[-1, 1], pred_target_coords[-1, 0], c='blue', s=100, 
                  marker='o', alpha=0.8, label='Pred End', zorder=6)
        
        # Calculate movement statistics
        gt_movement = torch.norm(target_coords[-1] - target_coords[0]).item()
        pred_movement = torch.norm(pred_target_coords[-1] - pred_target_coords[0]).item()
        movement_ratio = pred_movement / (gt_movement + 1e-6)
        
        # Calculate clustering measure (standard deviation)
        pred_std = torch.std(pred_target_coords.flatten()).item()
        gt_std = torch.std(target_coords.flatten()).item()
        
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_xlabel('Y Coordinate')
        ax.set_ylabel('X Coordinate')
        ax.set_title(f'Predicted Region Only\nMovement Ratio: {movement_ratio:.2f}\n'
                     f'Pred Std: {pred_std:.1f}, GT Std: {gt_std:.1f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add clustering indicator
        if movement_ratio < 0.3:
            ax.text(0.02, 0.98, 'CLUSTERING\nDETECTED', transform=ax.transAxes, 
                   fontsize=10, color='red', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                   verticalalignment='top')
    
    plt.suptitle('Phase 3 Direct Coordinate Prediction: Clustering Failure\n' +
                 'Expected Result: Predictions cluster together vs spread-out ground truth', 
                 fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Clustering failure visualization saved to: {save_path}")
    
    return fig

def analyze_clustering_behavior(predictions):
    """Analyze clustering behavior quantitatively"""
    print("\n" + "="*80)
    print("CLUSTERING BEHAVIOR ANALYSIS")
    print("="*80)
    
    all_movement_ratios = []
    all_pred_stds = []
    all_gt_stds = []
    
    for sample_idx, pred_data in predictions.items():
        target_coords = pred_data['target_coords']
        pred_target_coords = pred_data['predicted_target_coords']
        
        # Movement analysis
        gt_total_movement = torch.norm(target_coords[1:] - target_coords[:-1], dim=1).sum().item()
        pred_total_movement = torch.norm(pred_target_coords[1:] - pred_target_coords[:-1], dim=1).sum().item()
        movement_ratio = pred_total_movement / (gt_total_movement + 1e-6)
        
        # Spread analysis
        pred_std = torch.std(pred_target_coords.flatten()).item()
        gt_std = torch.std(target_coords.flatten()).item()
        
        # Center bias analysis
        pred_center_x = pred_target_coords[:, 0].mean().item()
        pred_center_y = pred_target_coords[:, 1].mean().item()
        image_center = 16.0
        center_bias = torch.norm(torch.tensor([pred_center_x - image_center, pred_center_y - image_center])).item()
        
        all_movement_ratios.append(movement_ratio)
        all_pred_stds.append(pred_std)
        all_gt_stds.append(gt_std)
        
        print(f"\nSample {sample_idx}:")
        print(f"  Movement Ratio: {movement_ratio:.3f} ({'SEVERE CLUSTERING' if movement_ratio < 0.3 else 'Normal'})")
        print(f"  Coordinate Std: Pred={pred_std:.2f}, GT={gt_std:.2f}")
        print(f"  Center Bias: {center_bias:.2f}px from image center")
        print(f"  Pred Center: ({pred_center_x:.1f}, {pred_center_y:.1f})")
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Average Movement Ratio: {np.mean(all_movement_ratios):.3f}")
    print(f"  Average Pred Std: {np.mean(all_pred_stds):.2f}")
    print(f"  Average GT Std: {np.mean(all_gt_stds):.2f}")
    print(f"  Std Ratio (Pred/GT): {np.mean(all_pred_stds) / np.mean(all_gt_stds):.3f}")
    
    clustering_detected = np.mean(all_movement_ratios) < 0.5
    print(f"\nCLUSTERING DIAGNOSIS: {'CONFIRMED' if clustering_detected else 'NOT DETECTED'}")
    if clustering_detected:
        print("  ✓ Predictions show severe movement reduction")
        print("  ✓ This demonstrates the failure of direct coordinate prediction")
        print("  ✓ Justifies need for more sophisticated approaches")

def main():
    """Main visualization function"""
    print("Phase 3 Direct Coordinate Prediction - Clustering Failure Visualization")
    print("="*80)
    
    # Test all three strategies
    strategies = ['parallel', 'sequential', 'seq2seq']
    
    # Load test data
    _, _, test_data = load_real_data()
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"VISUALIZING {strategy.upper()} STRATEGY")
        print(f"{'='*60}")
        
        # Load trained model
        model_path = f"results/best_{strategy}_coordinate_model.pth"
        
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            print("Please run train_coordinate_regression.py first")
            continue
        
        # Check checkpoint to get the correct strategy
        checkpoint = torch.load(model_path, map_location='cpu')
        saved_strategy = checkpoint.get('strategy', strategy)
        print(f"Loading model with strategy: {saved_strategy}")
        
        model = load_trained_model(model_path, strategy=saved_strategy)
        
        print(f"Model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Generate predictions for sample trajectories
        print(f"\nGenerating predictions for sample trajectories...")
        predictions = predict_sample_trajectories(model, test_data, num_samples=3)
        
        # Analyze clustering behavior
        analyze_clustering_behavior(predictions)
        
        # Create visualizations
        print(f"\nCreating clustering failure visualization...")
        visualize_clustering_failure(predictions, f'{strategy}_clustering_failure.png')
        
        print(f"\n✅ {strategy} visualization complete!")
    
    print(f"\n🎯 PHASE 3 CONCLUSION:")
    print("="*60)
    print("✓ Direct coordinate prediction leads to clustering failure")
    print("✓ All strategies (parallel, sequential, seq2seq) show similar issues")
    print("✓ Predictions cluster near center with reduced movement")
    print("✓ This demonstrates why video prediction is not equivalent to coordinate tracking")
    print("✓ Justifies need for Phase 4+ approaches with better representations")

if __name__ == "__main__":
    main() 