import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import math
import sys
sys.path.append('.')
from geom_simple import GeometricDataset, coordinates_to_heatmap, coordinates_to_pixel

def plot_trajectory_overview(dataset, num_samples=8, save_path=None):
    """
    Plot trajectory overview for different pattern types.
    
    Args:
        dataset: GeometricDataset instance
        num_samples: Number of samples to plot per pattern type
        save_path: Path to save the plot
    """
    pattern_types = ['line', 'arc']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for i, pattern in enumerate(pattern_types):
        ax = axes[i]
        ax.set_title(f'{pattern.capitalize()} Trajectories')
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Find samples of this pattern type
        pattern_indices = [idx for idx, ptype in enumerate(dataset.pattern_types) if ptype == pattern]
        
        for j in range(min(num_samples, len(pattern_indices))):
            idx = pattern_indices[j]
            coords = dataset.coordinates[idx]
            
            # Plot trajectory
            x_coords = coords[:, 0].numpy()
            y_coords = coords[:, 1].numpy()
            
            # Use different color for each sample
            color = plt.cm.tab10(j % 10)  # Use tab10 colormap for 10 different colors
            
            # Plot trajectory lines
            for k in range(len(x_coords) - 1):
                ax.plot(x_coords[k:k+2], y_coords[k:k+2], 
                       color=color, linewidth=2, alpha=0.8)
            
            # Mark ALL points along the trajectory
            ax.scatter(x_coords, y_coords, color=color, s=20, alpha=0.8, zorder=4)
            
            # Mark start and end points with special symbols
            ax.scatter(x_coords[0], y_coords[0], c='blue', s=100, marker='o', 
                      edgecolors='black', linewidth=2, zorder=5, label='Start' if j == 0 else "")
            ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='s', 
                      edgecolors='black', linewidth=2, zorder=5, label='End' if j == 0 else "")
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_distribution_analysis(dataset, save_path=None):
    """
    Plot distribution analysis for speed and angle.
    
    Args:
        dataset: GeometricDataset instance
        save_path: Path to save the plot
    """
    pattern_types = ['line', 'arc']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Collect data for all patterns
    all_speeds = []
    all_angles = []
    all_patterns = []
    
    for pattern in pattern_types:
        pattern_indices = [idx for idx, ptype in enumerate(dataset.pattern_types) if ptype == pattern]
        
        for idx in pattern_indices:
            coords = dataset.coordinates[idx]
            target_speed = dataset.target_speeds[idx]
            
            coords_np = coords.numpy()
            
            # Calculate initial direction angle
            if len(coords_np) > 1:
                dx = coords_np[1, 0] - coords_np[0, 0]
                dy = coords_np[1, 1] - coords_np[0, 1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                all_speeds.append(target_speed)
                all_angles.append(angle)
                all_patterns.append(pattern)
    
    # Speed distribution
    for i, pattern in enumerate(pattern_types):
        pattern_speeds = [all_speeds[j] for j, p in enumerate(all_patterns) if p == pattern]
        if pattern_speeds:
            axes[0, 0].hist(pattern_speeds, alpha=0.7, label=pattern.capitalize(), bins=20)
    
    axes[0, 0].set_title('Speed Distribution')
    axes[0, 0].set_xlabel('Speed (pixels/frame)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Angle distribution
    for i, pattern in enumerate(pattern_types):
        pattern_angles = [all_angles[j] for j, p in enumerate(all_patterns) if p == pattern]
        if pattern_angles:
            axes[0, 1].hist(pattern_angles, alpha=0.7, label=pattern.capitalize(), bins=20)
    
    axes[0, 1].set_title('Angle Distribution')
    axes[0, 1].set_xlabel('Angle (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Pattern count
    pattern_counts = {}
    for pattern in all_patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())
    axes[1, 0].bar(patterns, counts, color=['blue', 'orange'])
    axes[1, 0].set_title('Pattern Distribution')
    axes[1, 0].set_ylabel('Count')
    
    # Speed vs Angle scatter
    for i, pattern in enumerate(pattern_types):
        pattern_speeds = [all_speeds[j] for j, p in enumerate(all_patterns) if p == pattern]
        pattern_angles = [all_angles[j] for j, p in enumerate(all_patterns) if p == pattern]
        if pattern_speeds and pattern_angles:
            axes[1, 1].scatter(pattern_angles, pattern_speeds, alpha=0.6, label=pattern.capitalize())
    
    axes[1, 1].set_title('Speed vs Angle')
    axes[1, 1].set_xlabel('Angle (degrees)')
    axes[1, 1].set_ylabel('Speed (pixels/frame)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run visualizations.
    """
    # Create dataset with 1000 train samples
    dataset = GeometricDataset('train', num_samples=1000, img_size=32, sequence_length=20)
    
    # Create output directory  
    output_dir = Path('../data/geometric_patterns')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating trajectory overview...")
    plot_trajectory_overview(dataset, save_path=output_dir / 'trajectory_overview.png')
    
    print("Generating distribution analysis...")
    plot_distribution_analysis(dataset, save_path=output_dir / 'distribution_analysis.png')
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main() 