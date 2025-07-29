#!/usr/bin/env python3
"""
Generate single fixation simulation data.
Each frame contains exactly one fixation point.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import config


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_spatial_single_fixation(batch_size, T, img_size, sigma):
    """
    Sample spatial coordinates for single fixation sequence.
    Each time step gets a new fixation based on position-dependent Gaussian.
    """
    coords = torch.zeros(batch_size, T, 2)
    
    # Create position-dependent parameter grid
    y_grid = torch.linspace(0, img_size, config.grid_size)
    x_grid = torch.linspace(0, img_size, config.grid_size)
    yv, xv = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Center coordinates
    center_y, center_x = img_size / 2.0, img_size / 2.0
    
    # Distance from center for each grid point
    distance_from_center = torch.sqrt((yv - center_y)**2 + (xv - center_x)**2)
    max_distance = torch.sqrt((img_size/2)**2 + (img_size/2)**2)
    normalized_distance = distance_from_center / max_distance
    
    # Create bimodal covariance distribution
    short_step_mask = torch.rand(config.grid_size, config.grid_size) < config.short_step_ratio
    
    cov_matrices = torch.zeros(config.grid_size, config.grid_size, 2, 2)
    
    # Short steps
    short_cov_diag = torch.rand(config.grid_size, config.grid_size) * (
        config.short_step_cov_range[1] - config.short_step_cov_range[0]) + config.short_step_cov_range[0]
    cov_matrices[short_step_mask, 0, 0] = short_cov_diag[short_step_mask]
    cov_matrices[short_step_mask, 1, 1] = short_cov_diag[short_step_mask]
    
    # Long steps  
    long_cov_diag = torch.rand(config.grid_size, config.grid_size) * (
        config.long_step_cov_range[1] - config.long_step_cov_range[0]) + config.long_step_cov_range[0]
    cov_matrices[~short_step_mask, 0, 0] = long_cov_diag[~short_step_mask]
    cov_matrices[~short_step_mask, 1, 1] = long_cov_diag[~short_step_mask]
    
    # Generate sequences
    for b in range(batch_size):
        # Start near center
        image_center = torch.tensor([img_size / 2.0, img_size / 2.0])
        coords[b, 0] = torch.normal(image_center, sigma)
        
        for t in range(1, T):
            current_pos = coords[b, t-1]
            
            # Find nearest grid point
            grid_y = torch.clamp(torch.round(current_pos[0] * (config.grid_size-1) / img_size).long(), 0, config.grid_size-1)
            grid_x = torch.clamp(torch.round(current_pos[1] * (config.grid_size-1) / img_size).long(), 0, config.grid_size-1)
            
            # Get covariance matrix for this position
            cov_matrix = cov_matrices[grid_y, grid_x]
            
            # Center-biased displacement
            center_direction = torch.tensor([center_y, center_x]) - current_pos
            center_direction = center_direction / (torch.norm(center_direction) + 1e-8)
            
            # Random exploration component
            random_direction = torch.randn(2)
            random_direction = random_direction / (torch.norm(random_direction) + 1e-8)
            
            # Combine directions
            displacement_direction = (config.center_bias_strength * center_direction + 
                                    config.random_exploration_scale * random_direction)
            displacement_direction = displacement_direction / (torch.norm(displacement_direction) + 1e-8)
            
            # Sample displacement magnitude from covariance
            displacement_magnitude = torch.sqrt(torch.abs(torch.randn(1) * cov_matrix[0, 0]))
            displacement = displacement_direction * displacement_magnitude * config.displacement_scale
            
            coords[b, t] = current_pos + displacement
        
        # Clamp to image bounds
        coords[b] = torch.clamp(coords[b], 0, img_size)
    
    return coords.to(torch.float32)


def render_visual_single_fixation(coords, img_size):
    """
    Render single fixation visual frames.
    Each frame contains exactly one white dot at the fixation location.
    """
    batch_size, T, _ = coords.shape
    frames = torch.zeros(batch_size, T, img_size, img_size)
    
    for b in range(batch_size):
        for t in range(T):
            y, x = coords[b, t]
            y_int, x_int = int(round(y.item())), int(round(x.item()))
            
            # Clamp coordinates to valid range
            y_int = max(0, min(img_size - 1, y_int))
            x_int = max(0, min(img_size - 1, x_int))
            
            frames[b, t, y_int, x_int] = 1.0
    
    return frames


def generate_sequence(batch_size, T, img_size):
    """
    Generate a batch of single fixation sequences.
    Returns frames and coordinates.
    """
    # Sample spatial coordinates
    coords = sample_spatial_single_fixation(batch_size, T, img_size, config.sigma)
    
    # Render visual frames
    frames = render_visual_single_fixation(coords, img_size)
    
    # Create fixation mask (always 1 for single fixation)
    fixation_mask = torch.ones(batch_size, T)
    
    return {
        'frames': frames,
        'coords': coords,
        'fixation_mask': fixation_mask
    }


def generate_dataset(split_name, size):
    """Generate a dataset split"""
    print(f"Generating {split_name} dataset ({size} samples)...")
    
    all_frames = []
    all_coords = []
    all_masks = []
    
    # Generate in batches to manage memory
    batch_size = min(50, size)
    num_batches = (size + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc=f"Generating {split_name}"):
        current_batch_size = min(batch_size, size - i * batch_size)
        
        # Generate batch
        data = generate_sequence(current_batch_size, config.total_frames, config.img_size)
        
        all_frames.append(data['frames'])
        all_coords.append(data['coords'])
        all_masks.append(data['fixation_mask'])
    
    # Concatenate all batches
    frames = torch.cat(all_frames, dim=0)
    coords = torch.cat(all_coords, dim=0)
    masks = torch.cat(all_masks, dim=0)
    
    # Reshape frames to (B, C, T, H, W) format for SimVP
    frames = frames.unsqueeze(1)  # Add channel dimension
    
    return {
        'frames': frames,
        'coords': coords,
        'fixation_mask': masks
    }


def main():
    """Generate all datasets"""
    print("Starting single fixation data generation")
    print(f"Configuration: {config.total_frames} frames, {config.img_size}x{config.img_size} resolution")
    
    # Set random seed
    set_seed(config.random_seed)
    
    # Create data directory
    os.makedirs(config.data_dir, exist_ok=True)
    
    # Generate datasets
    datasets = {
        'train': generate_dataset('train', config.train_size),
        'val': generate_dataset('validation', config.val_size),
        'test': generate_dataset('test', config.test_size)
    }
    
    # Save datasets
    for split_name, dataset in datasets.items():
        file_path = getattr(config, f'{split_name}_data_file')
        torch.save(dataset, file_path)
        print(f"Saved {split_name} dataset: {file_path}")
        print(f"  Frames shape: {dataset['frames'].shape}")
        print(f"  Coords shape: {dataset['coords'].shape}")
    
    print("Data generation completed successfully!")
    
    # Verify data
    print("\nData verification:")
    for split_name, dataset in datasets.items():
        frames = dataset['frames']
        coords = dataset['coords']
        
        # Check each frame has exactly one fixation
        frames_2d = frames.squeeze(1)  # Remove channel dimension
        fixation_counts = frames_2d.sum(dim=(-2, -1))  # Sum over spatial dimensions
        
        print(f"{split_name}:")
        print(f"  Fixations per frame - min: {fixation_counts.min():.1f}, max: {fixation_counts.max():.1f}, mean: {fixation_counts.mean():.1f}")
        print(f"  Coordinate range - x: [{coords[:,:,1].min():.1f}, {coords[:,:,1].max():.1f}], y: [{coords[:,:,0].min():.1f}, {coords[:,:,0].max():.1f}]")


if __name__ == "__main__":
    main() 