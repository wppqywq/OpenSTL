#!/usr/bin/env python3
"""
Clean MVP Visualization - Copy to Jupyter
Uses real 2000-sample dataset, simplified plots
"""

# =============================================================================
# Cell 0: Setup & Imports
# =============================================================================
import math, json, os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.bbox'] = 'tight'
torch.set_default_dtype(torch.float32)

# =============================================================================
# Cell 1: Generate Full Dataset (2000/200/200)
# =============================================================================

# EDIT MODE HERE
MODE = 'center_inward'  # 'center_inward' | 'global_right' | 'global_down'

print(f"Generating full dataset with mode: {MODE}")

# Import 
import sys
sys.path.append('../..')
from experiments_refactored.datasets.position_dependent_gaussian import GaussianFieldGenerator
from experiments_refactored.datasets.config import FIELD_CONFIG

# Use default 2000/200/200 from config but with enhanced exploration
CONFIG = FIELD_CONFIG.copy()
CONFIG.update({
    'mode': MODE,
    'step_scale': 4.0,        # higher for exploration
    'lambda_iso': 0.8,        # more noise for coverage
    'sequence_length': 25,    # longer sequences
    'seed': 42,
})

print(f"Config: step_scale={CONFIG['step_scale']}, train_size={CONFIG.get('train_size', 2000)}")

# Generate full dataset
gen = GaussianFieldGenerator("data/clean_viz", CONFIG)  
datasets = gen.generate_datasets()

# Extract data
coords_train = datasets['train']['sparse']['coordinates']  # [2000, 25, 2]
frames_train_sparse = datasets['train']['sparse']['frames']
frames_train_gauss = datasets['train']['gaussian']['frames']

print(f"✅ Generated {coords_train.shape[0]} train sequences")

# =============================================================================
# Cell 2: Three Individual Sample Trajectories
# =============================================================================

def plot_single_trajectory(coords, seq_idx, subplot_pos=None):
    """Plot one clean trajectory."""
    if subplot_pos is None:
        plt.figure(figsize=(5,5))
        ax = plt.gca()
    else:
        ax = plt.subplot(*subplot_pos)
    
    y = coords[seq_idx,:,0].numpy()
    x = coords[seq_idx,:,1].numpy()
    
    # Simple trajectory line
    ax.plot(x, y, '-', color='blue', lw=2, alpha=0.8)
    ax.plot(x, y, 'o', color='blue', ms=3, alpha=0.6)
    
    # Mark start and end only
    ax.plot(x[0], y[0], 's', color='green', markersize=8, label='Start (0,0)')
    ax.plot(x[-1], y[-1], '^', color='red', markersize=8, label='End')
    
    ax.invert_yaxis()
    ax.set_xlim(-2, 34); ax.set_ylim(34, -2)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{MODE}: Sample {seq_idx}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if seq_idx == 0: ax.legend()
    
    if subplot_pos is None:
        plt.show()

# Plot 3 samples in separate subplots
plt.figure(figsize=(15, 4))
plot_single_trajectory(coords_train, 0, (1, 3, 1))
plot_single_trajectory(coords_train, 1, (1, 3, 2))
plot_single_trajectory(coords_train, 2, (1, 3, 3))
plt.tight_layout()
plt.show()

# =============================================================================
# Cell 3: Simplified Empirical Field (Less Dense)
# =============================================================================

def plot_clean_empirical_field(coords, H=32, grid_size=6):
    """Clean field plot using real data, less dense arrows."""
    B, T, _ = coords.shape
    grid_step = H / grid_size
    
    # Collect displacements by grid
    field_data = {}
    for b in range(B):
        for t in range(T-1):
            y_curr, x_curr = coords[b, t]
            y_next, x_next = coords[b, t+1]
            
            gy = int(y_curr / grid_step)
            gx = int(x_curr / grid_step)
            gy = max(0, min(grid_size-1, gy))
            gx = max(0, min(grid_size-1, gx))
            
            dy = float(y_next - y_curr)
            dx = float(x_next - x_curr)
            
            key = (gy, gx)
            if key not in field_data:
                field_data[key] = []
            field_data[key].append([dy, dx])
    
    # Compute mean displacements
    mean_disps = {}
    for key, disps in field_data.items():
        disps_array = np.array(disps)
        mean_disps[key] = {
            'mean': disps_array.mean(axis=0),
            'count': len(disps)
        }
    
    # Plot clean field
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    # Background grid
    for i in range(grid_size + 1):
        x_line = i * grid_step
        y_line = i * grid_step
        ax.axvline(x_line, alpha=0.2, color='lightgray', lw=0.5)
        ax.axhline(y_line, alpha=0.2, color='lightgray', lw=0.5)
    
    # Plot arrows (mean displacement per grid cell)
    for (gy, gx), data in mean_disps.items():
        if data['count'] < 10:  # skip cells with too few samples
            continue
            
        y_pos = (gy + 0.5) * grid_step
        x_pos = (gx + 0.5) * grid_step
        
        dy_mean, dx_mean = data['mean']
        count = data['count']
        
        # Arrow scale and color based on magnitude and sample count
        arrow_scale = 3.0
        alpha = min(0.9, count / 100.0)
        alpha = max(0.4, alpha)
        
        # Arrow color based on magnitude
        mag = np.sqrt(dy_mean**2 + dx_mean**2)
        color = plt.cm.Reds(min(1.0, mag / 2.0))
        
        ax.arrow(x_pos, y_pos, dx_mean*arrow_scale, dy_mean*arrow_scale,
                 head_width=1.2, head_length=1.5,
                 fc=color, ec=color, alpha=alpha,
                 length_includes_head=True, width=0.3)
    
    ax.set_xlim(-1, H+1); ax.set_ylim(H+1, -1)
    ax.set_aspect('equal')
    ax.set_title(f'{MODE}: Mean Displacement Field\n(from real {B} sequences, {grid_size}×{grid_size} grid)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    
    # Add colorbar for arrow magnitude
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=2))
    sm.set_array([])
    plt.colorbar(sm, label='displacement magnitude')
    
    plt.show()
    
    print(f"Field computed from {sum(data['count'] for data in mean_disps.values())} real displacement samples")

plot_clean_empirical_field(coords_train, H=32, grid_size=6)

# =============================================================================
# Cell 4: Coverage Heatmap Only (Log Scale)
# =============================================================================

def plot_coverage_heatmap_log(coords, H=32):
    """Coverage heatmap with log scale, using full 2000 dataset."""
    B, T, _ = coords.shape
    
    coverage = np.zeros((H, H), dtype=np.int32)
    for b in range(B):
        for t in range(T):
            y, x = coords[b, t]
            yi, xi = int(round(float(y))), int(round(float(x)))
            yi = max(0, min(H-1, yi)); xi = max(0, min(H-1, xi))
            coverage[yi, xi] += 1
    
    # Use log scale for better visualization
    log_coverage = np.log10(coverage + 1)  # +1 to avoid log(0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(log_coverage, origin='upper', cmap='viridis')
    plt.title(f'{MODE}: Coverage Heatmap (log10 scale)\n{B} sequences × {T} steps = {B*T:,} total points')
    
    # Custom colorbar with original values
    cbar = plt.colorbar(label='log10(visits + 1)')
    
    # Add text annotations for key values
    plt.xlabel('x'); plt.ylabel('y')
    
    # Stats
    visited_pixels = np.sum(coverage > 0)
    coverage_ratio = visited_pixels / (H * H)
    max_visits = coverage.max()
    mean_visits = coverage[coverage > 0].mean()
    
    plt.text(0.02, 0.98, f'Coverage: {visited_pixels}/{H*H} pixels ({coverage_ratio:.1%})\n'
                         f'Max visits: {max_visits}\n'
                         f'Mean visits (visited): {mean_visits:.1f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.show()
    
    print(f"Coverage Statistics:")
    print(f"  Dataset size: {B:,} sequences × {T} steps = {B*T:,} position samples")
    print(f"  Visited pixels: {visited_pixels}/{H*H} ({coverage_ratio:.3f})")
    print(f"  Max visits per pixel: {max_visits}")
    print(f"  Mean visits (visited pixels only): {mean_visits:.2f}")
    
    return coverage

coverage = plot_coverage_heatmap_log(coords_train)

# =============================================================================
# Cell 5: Motion Statistics Only
# =============================================================================

def motion_statistics_only(coords, name):
    """Motion analysis using full real dataset."""
    B, T, _ = coords.shape
    displacements = coords[:, 1:, :] - coords[:, :-1, :]
    
    dy = displacements[:, :, 0].numpy().flatten()
    dx = displacements[:, :, 1].numpy().flatten()
    angles = np.arctan2(dy, dx)
    speeds = np.sqrt(dy*dy + dx*dx)
    
    plt.figure(figsize=(12, 3))
    
    # Angle histogram
    plt.subplot(1, 3, 1)
    plt.hist(angles, bins=36, density=True, alpha=0.7, color='blue')
    plt.title(f'{name}: Angle Distribution')
    plt.xlabel('angle (rad)'); plt.ylabel('density')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='→')
    plt.axvline(np.pi/2, color='green', linestyle='--', alpha=0.7, label='↓')
    plt.axvline(np.pi/4, color='orange', linestyle='--', alpha=0.7, label='↘')
    plt.legend()
    
    # Speed histogram
    plt.subplot(1, 3, 2)
    plt.hist(speeds, bins=30, density=True, alpha=0.7, color='orange')
    plt.title(f'{name}: Speed Distribution')
    plt.xlabel('displacement magnitude'); plt.ylabel('density')
    
    # Mean displacement per timestep
    plt.subplot(1, 3, 3)
    mean_dy_by_t = displacements[:, :, 0].mean(dim=0).numpy()
    mean_dx_by_t = displacements[:, :, 1].mean(dim=0).numpy()
    timesteps = range(len(mean_dy_by_t))
    
    plt.plot(timesteps, mean_dx_by_t, 'o-', label='mean dx', alpha=0.8)
    plt.plot(timesteps, mean_dy_by_t, 's-', label='mean dy', alpha=0.8)
    plt.title(f'{name}: Mean Displacement by Timestep')
    plt.xlabel('timestep'); plt.ylabel('mean displacement')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Motion Statistics ({len(speeds):,} real displacement samples):")
    print(f"  Mean angle: {math.degrees(angles.mean()):.1f}°")
    print(f"  Mean speed: {speeds.mean():.3f} ± {speeds.std():.3f}")
    print(f"  Overall bias: dx={dx.mean():.3f}, dy={dy.mean():.3f}")

motion_statistics_only(coords_train, f'{MODE} (2000 sequences)')

# =============================================================================
# Cell 6: Sparse vs Gaussian Frames Comparison
# =============================================================================

def show_frame_comparison(frames_sparse, frames_gauss, coords, seq_idx=0):
    """Show sparse vs gaussian for one sample sequence."""
    steps = [0, 5, 10, 15, 20, 24]  # sample timesteps
    n = len(steps)
    
    plt.figure(figsize=(2*n, 5))
    
    for i, t in enumerate(steps):
        if t >= frames_sparse.shape[1]:
            continue
            
        # Sparse frame
        plt.subplot(2, n, i+1)
        plt.imshow(frames_sparse[seq_idx, t, 0].numpy(), vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        plt.title(f'Sparse t={t}', fontsize=10)
        
        # Gaussian frame
        plt.subplot(2, n, n+i+1)
        plt.imshow(frames_gauss[seq_idx, t, 0].numpy(), cmap='hot')
        plt.axis('off')
        plt.title(f'Gaussian t={t}', fontsize=10)
    
    plt.suptitle(f'{MODE}: Frame Representations (sample {seq_idx})', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Show coordinates for reference
    print(f"Sample {seq_idx} coordinates at displayed timesteps:")
    for t in steps:
        if t < coords.shape[1]:
            y, x = coords[seq_idx, t]
            print(f"  t={t}: ({y:.2f}, {x:.2f})")

show_frame_comparison(frames_train_sparse, frames_train_gauss, coords_train, 0)

# =============================================================================
# Cell 7: Mode Comparison (Generate Other Modes)
# =============================================================================

def compare_modes_clean():
    """Generate and compare all modes cleanly."""
    modes = ['center_inward', 'global_right', 'global_down']
    mode_data = {}
    
    # Generate smaller samples for each mode
    for mode in modes:
        cfg = CONFIG.copy()
        cfg['mode'] = mode
        cfg['seed'] = hash(mode) % 1000
        
        from experiments_refactored.datasets.position_dependent_gaussian import sample_position_dependent_gaussian
        coords = sample_position_dependent_gaussian(100, 20, 32, cfg, start_pos=(0,0), seed=cfg['seed'])
        mode_data[mode] = coords
    
    plt.figure(figsize=(15, 5))
    
    for i, mode in enumerate(modes):
        coords = mode_data[mode]
        
        plt.subplot(1, 3, i+1)
        
        # Show 3 trajectories per mode
        for b in range(3):
            y = coords[b,:,0].numpy()
            x = coords[b,:,1].numpy()
            plt.plot(x, y, '-o', ms=2, lw=1.5, alpha=0.8, label=f'Traj {b}')
        
        plt.plot(0, 0, 'ko', markersize=8, label='Start')
        plt.gca().invert_yaxis()
        plt.xlim(-3, 35); plt.ylim(35, -3)
        plt.title(mode.replace('_', ' ').title())
        plt.xlabel('x'); plt.ylabel('y')
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compare motion statistics
    print("Mode Comparison (mean displacement from 100 samples each):")
    for mode in modes:
        coords = mode_data[mode]
        displacements = coords[:, 1:, :] - coords[:, :-1, :]
        mean_disp = displacements.mean(dim=(0,1))
        print(f"  {mode:>12}: dy={float(mean_disp[0]):+.3f}, dx={float(mean_disp[1]):+.3f}")

compare_modes_clean()

# =============================================================================
# Cell 8: Final Summary
# =============================================================================

def final_analysis():
    """Summary of the generated dataset."""
    B, T = coords_train.shape[:2]
    
    # Overall statistics
    all_y = coords_train[:, :, 0].numpy().flatten()
    all_x = coords_train[:, :, 1].numpy().flatten()
    
    displacements = coords_train[:, 1:, :] - coords_train[:, :-1, :]
    all_speeds = torch.norm(displacements, dim=2).numpy().flatten()
    
    print("="*50)
    print(f"DATASET SUMMARY: {MODE}")
    print("="*50)
    print(f"Generated: {B:,} sequences × {T} steps = {B*T:,} position samples")
    print(f"Configuration: {CONFIG['mode']}, step_scale={CONFIG['step_scale']}")
    print(f"")
    print(f"Spatial coverage:")
    print(f"  y range: [{all_y.min():.1f}, {all_y.max():.1f}]")
    print(f"  x range: [{all_x.min():.1f}, {all_x.max():.1f}]") 
    print(f"  coverage ratio: {(coverage > 0).mean():.3f}")
    print(f"")
    print(f"Motion patterns:")
    print(f"  mean speed: {all_speeds.mean():.3f} ± {all_speeds.std():.3f}")
    print(f"  total displacement samples: {len(all_speeds):,}")
    
    # Boundary behavior
    at_boundary = ((all_y <= 0.1) | (all_y >= 31.9) | (all_x <= 0.1) | (all_x >= 31.9))
    print(f"  boundary contact: {at_boundary.mean():.3f} fraction")
    
    print("\n✅ MVP dataset ready for SimVP training!")

final_analysis()

# =============================================================================
# Optional: Export Data Info
# =============================================================================

# Save dataset info
info = {
    'mode': MODE,
    'config': CONFIG,
    'train_shape': list(coords_train.shape),
    'coverage_ratio': float((coverage > 0).mean()),
    'data_path': 'data/clean_viz'
}

print(f"\nDataset info: {json.dumps(info, indent=2)}")
print(f"Data saved to: data/clean_viz/")
print(f"Ready for: from torch.utils.data import DataLoader; dataset = GaussianFieldDataset('train', 'data/clean_viz')")
