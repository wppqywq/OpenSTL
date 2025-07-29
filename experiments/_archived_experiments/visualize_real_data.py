#!/usr/bin/env python3
"""
Visualize Real Eye Movement Data Used in Phases 1-3
Show coordinate distribution and sample trajectories
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_real_data():
    """Load the real eye movement data"""
    data_path = Path("data")
    
    train_data = torch.load(data_path / "train_data.pt")
    val_data = torch.load(data_path / "val_data.pt") 
    test_data = torch.load(data_path / "test_data.pt")
    
    print(f"Loaded real eye movement data:")
    print(f"  Train: {train_data['coordinates'].shape}")
    print(f"  Val: {val_data['coordinates'].shape}")
    print(f"  Test: {test_data['coordinates'].shape}")
    
    return train_data, val_data, test_data

def analyze_coordinate_distribution(all_coords, all_masks, title_prefix=""):
    """Analyze and plot coordinate distribution"""
    
    # Flatten all coordinates where mask is True
    valid_coords = []
    for coords, mask in zip(all_coords, all_masks):
        for t in range(len(mask)):
            if mask[t]:  # Only include active fixations
                valid_coords.append(coords[t].numpy())
    
    if len(valid_coords) == 0:
        print("No valid coordinates found!")
        return None, None, None
        
    valid_coords = np.array(valid_coords)
    x_coords = valid_coords[:, 0]
    y_coords = valid_coords[:, 1]
    
    print(f"\n{title_prefix} Coordinate Statistics:")
    print(f"  Total valid points: {len(valid_coords):,}")
    print(f"  X range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
    print(f"  Y range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
    print(f"  X mean: {x_coords.mean():.2f} ± {x_coords.std():.2f}")
    print(f"  Y mean: {y_coords.mean():.2f} ± {y_coords.std():.2f}")
    
    return valid_coords, x_coords, y_coords

def plot_data_analysis():
    """Create comprehensive visualization of the real data"""
    
    # Load data
    train_data, val_data, test_data = load_real_data()
    
    # Combine all data for overall analysis
    all_coords = torch.cat([
        train_data['coordinates'], 
        val_data['coordinates'], 
        test_data['coordinates']
    ], dim=0)
    all_masks = torch.cat([
        train_data['fixation_mask'],
        val_data['fixation_mask'], 
        test_data['fixation_mask']
    ], dim=0)
    
    print(f"Combined dataset: {all_coords.shape[0]} sequences, {all_coords.shape[1]} frames each")
    
    # Analyze coordinate distribution
    result = analyze_coordinate_distribution(all_coords, all_masks, "Overall")
    if result[0] is None:
        print("No valid coordinates found, cannot create visualization")
        return
    valid_coords, x_coords, y_coords = result
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Overall coordinate distribution (2D scatter)
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(x_coords, y_coords, alpha=0.1, s=1, c='black')
    plt.xlim(0, 32)
    plt.ylim(0, 32)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Overall Coordinate Distribution\n(All Valid Fixations)')
    plt.grid(True, alpha=0.3)
    
    # 2. X coordinate histogram
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(x_coords, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('X Coordinate')
    plt.ylabel('Frequency')
    plt.title('X Coordinate Distribution')
    plt.grid(True, alpha=0.3)
    
    # 3. Y coordinate histogram  
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(y_coords, bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Y Coordinate') 
    plt.ylabel('Frequency')
    plt.title('Y Coordinate Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. Sample trajectory 1
    ax4 = plt.subplot(2, 3, 4)
    sample_idx = 5  # Choose a good sample
    sample_coords = train_data['coordinates'][sample_idx]
    sample_mask = train_data['fixation_mask'][sample_idx]
    
    # Plot full trajectory
    plt.plot(sample_coords[:, 0], sample_coords[:, 1], 'b-', alpha=0.5, linewidth=1, label='Full sequence')
    
    # Highlight active fixations
    active_coords = sample_coords[sample_mask]
    if len(active_coords) > 0:
        plt.scatter(active_coords[:, 0], active_coords[:, 1], 
                   c=range(len(active_coords)), cmap='viridis', s=30, label='Active fixations')
        
        # Add arrows to show direction
        if len(active_coords) > 1:
            for i in range(len(active_coords) - 1):
                dx = active_coords[i+1, 0] - active_coords[i, 0]
                dy = active_coords[i+1, 1] - active_coords[i, 1]
                plt.arrow(active_coords[i, 0], active_coords[i, 1], dx, dy,
                         head_width=0.5, head_length=0.5, fc='red', ec='red', alpha=0.7)
    
    plt.xlim(0, 32)
    plt.ylim(0, 32)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate') 
    plt.title(f'Sample Trajectory 1\n({sample_mask.sum().item()}/{len(sample_mask)} active frames)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Sample trajectory 2
    ax5 = plt.subplot(2, 3, 5)
    sample_idx = 15  # Choose another sample
    sample_coords = train_data['coordinates'][sample_idx]
    sample_mask = train_data['fixation_mask'][sample_idx]
    
    # Plot full trajectory
    plt.plot(sample_coords[:, 0], sample_coords[:, 1], 'b-', alpha=0.5, linewidth=1, label='Full sequence')
    
    # Highlight active fixations
    active_coords = sample_coords[sample_mask]
    if len(active_coords) > 0:
        plt.scatter(active_coords[:, 0], active_coords[:, 1], 
                   c=range(len(active_coords)), cmap='plasma', s=30, label='Active fixations')
        
        # Add arrows to show direction
        if len(active_coords) > 1:
            for i in range(len(active_coords) - 1):
                dx = active_coords[i+1, 0] - active_coords[i, 0]
                dy = active_coords[i+1, 1] - active_coords[i, 1]
                plt.arrow(active_coords[i, 0], active_coords[i, 1], dx, dy,
                         head_width=0.5, head_length=0.5, fc='red', ec='red', alpha=0.7)
    
    plt.xlim(0, 32)
    plt.ylim(0, 32)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Sample Trajectory 2\n({sample_mask.sum().item()}/{len(sample_mask)} active frames)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Fixation density map
    ax6 = plt.subplot(2, 3, 6)
    
    # Create 2D histogram/heatmap
    hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=32, range=[[0, 32], [0, 32]])
    
    im = plt.imshow(hist.T, origin='lower', extent=(0, 32, 0, 32), cmap='hot', alpha=0.8)
    plt.colorbar(im, label='Fixation Count')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Fixation Density Heatmap\n(Higher = More Fixations)')
    
    plt.tight_layout()
    
    # Calculate displacement statistics
    print(f"\nDisplacement Analysis:")
    all_displacements = []
    for coords, mask in zip(all_coords, all_masks):
        # Calculate displacements for active frames
        for t in range(len(mask) - 1):
            if mask[t] and mask[t+1]:  # Both frames active
                displacement = coords[t+1] - coords[t]
                all_displacements.append(displacement.numpy())
    
    if len(all_displacements) > 0:
        all_displacements = np.array(all_displacements)
        displacement_magnitudes = np.linalg.norm(all_displacements, axis=1)
        
        print(f"  Total displacements: {len(all_displacements):,}")
        print(f"  Magnitude range: [{displacement_magnitudes.min():.2f}, {displacement_magnitudes.max():.2f}]")
        print(f"  Magnitude mean: {displacement_magnitudes.mean():.2f} ± {displacement_magnitudes.std():.2f}")
        print(f"  X displacement mean: {all_displacements[:, 0].mean():.2f} ± {all_displacements[:, 0].std():.2f}")
        print(f"  Y displacement mean: {all_displacements[:, 1].mean():.2f} ± {all_displacements[:, 1].std():.2f}")
    
    # Save the plot
    plt.savefig('real_data_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: real_data_analysis.png")
    
    plt.show()

def analyze_sparsity():
    """Analyze the sparsity characteristics"""
    train_data, val_data, test_data = load_real_data()
    
    # Combine all data
    all_masks = torch.cat([
        train_data['fixation_mask'],
        val_data['fixation_mask'], 
        test_data['fixation_mask']
    ], dim=0)
    
    total_frames = all_masks.numel()
    active_frames = all_masks.sum().item()
    sparsity_ratio = (total_frames - active_frames) / total_frames
    
    print(f"\nSparsity Analysis:")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Active frames: {active_frames:,}")
    print(f"  Inactive frames: {total_frames - active_frames:,}")
    print(f"  Sparsity ratio: {sparsity_ratio:.4f} ({sparsity_ratio*100:.1f}% empty)")
    print(f"  Class imbalance: {(total_frames - active_frames)/active_frames:.1f}:1 (empty:active)")
    
    # Per-sequence sparsity
    seq_sparsity = []
    for mask in all_masks:
        seq_active = mask.sum().item()
        seq_total = len(mask)
        seq_sparsity.append(seq_active / seq_total)
    
    seq_sparsity = np.array(seq_sparsity)
    print(f"  Per-sequence active ratio: {seq_sparsity.mean():.3f} ± {seq_sparsity.std():.3f}")
    print(f"  Min/Max active ratio: [{seq_sparsity.min():.3f}, {seq_sparsity.max():.3f}]")

if __name__ == "__main__":
    print("Analyzing Real Eye Movement Data Used in Phases 1-3")
    print("=" * 60)
    
    try:
        # Analyze sparsity first
        analyze_sparsity()
        
        # Create comprehensive visualization
        plot_data_analysis()
        
        print(f"\n✅ Analysis complete! Key characteristics of Phase 1-3 data:")
        print(f"   - Real eye movement sequences with extreme sparsity")
        print(f"   - 32x32 pixel coordinate space")
        print(f"   - Sparse active fixations (~10-20% of frames)")
        print(f"   - Variable displacement patterns")
        print(f"   - This is the challenging data that revealed velocity undershooting")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}") 