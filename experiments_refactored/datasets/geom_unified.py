#!/usr/bin/env python3
"""
Unified geometric pattern generation module.

This module provides unified functions for generating geometric patterns
with shared parameters for angles, step sizes, and starting positions.
"""

import torch
import numpy as np
import math
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .constants import GEOM_CONFIG, IMG_SIZE, SEQUENCE_LENGTH, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, HEATMAP_SIGMA, RANDOM_SEED

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configuration Parameters - using constants from constants.py
LOCAL_CONFIG = {
    'img_size': IMG_SIZE,
    'sequence_length': SEQUENCE_LENGTH,
    'train_samples': TRAIN_SIZE,
    'val_samples': VAL_SIZE,
    'test_samples': TEST_SIZE,
    'batch_size': 32,
    
    # Unified sampling parameters
    'angle_range': (-180, 180),  # degrees, wider range  
    'line_ratio': GEOM_CONFIG['line_ratio'],
    
    # Boundary handling
    'max_boundary_attempts': 10,  # max attempts to handle boundary collision
    
    # Representation parameters
    'heatmap_sigma_default': HEATMAP_SIGMA,
}

def sample_unified_parameters(pattern_idx=None, force_pattern_type=None):
    """
    Sample parameters for structured geometric patterns.
    
    Line patterns: 3 speeds × 8 directions × 8 origins  
    Arc patterns: 3 radii × 2 directions (CW/CCW)
    
    Args:
        pattern_idx: Index to determine pattern type (for deterministic generation)
        force_pattern_type: Force specific pattern type ('line' or 'arc')
    
    Returns:
        dict: Dictionary containing sampled parameters
    """
    img_size = LOCAL_CONFIG['img_size']
    
    # Determine pattern type deterministically if pattern_idx is provided
    if force_pattern_type:
        pattern_type = force_pattern_type
    elif pattern_idx is not None:
        pattern_type = 'line' if pattern_idx % 2 == 0 else 'arc'
    else:
        pattern_type = 'line' if random.random() < LOCAL_CONFIG['line_ratio'] else 'arc'
    
    if pattern_type == 'line':
        # Line pattern parameters: constant speed + bounce reflection
        speed = random.choice(GEOM_CONFIG['line_speeds'])
        direction = random.choice(GEOM_CONFIG['line_directions'])  # degrees
        
        # Generate 8 starting positions from 4x4 grid subset  
        grid_positions = []
        for i in range(1, 4):  # Skip edges, use interior 3x3 but sample 8 points
            for j in range(1, 4):
                if len(grid_positions) < 8:  # Take first 8 positions
                    x = (j / 4) * img_size
                    y = (i / 4) * img_size
                    grid_positions.append((x, y))
        
        start_pos = random.choice(grid_positions)
        
        return {
            'pattern_type': 'line',
            'start_pos': start_pos,
            'speed': speed,
            'direction': direction,  # degrees
        }
        
    else:  # arc pattern
        # Arc pattern parameters: circular motion
        radius = random.choice(GEOM_CONFIG['arc_radii'])
        arc_direction = random.choice(GEOM_CONFIG['arc_directions'])  # 'CW' or 'CCW'
        
        # Generate random arc center ensuring trajectory stays within bounds
        margin = GEOM_CONFIG['arc_center_margin']
        center_x = random.uniform(margin + radius, img_size - margin - radius)
        center_y = random.uniform(margin + radius, img_size - margin - radius)
        
        # Random starting angle on the circle
        start_angle = random.uniform(0, 2 * math.pi)
        
        return {
            'pattern_type': 'arc',
            'center_pos': (center_x, center_y),
            'radius': radius,
            'direction': arc_direction,  # 'CW' or 'CCW'
            'start_angle': start_angle,  # radians
        }

def check_boundary_collision(x, y, img_size):
    """Check if position is outside image boundaries."""
    return x < 0 or x >= img_size or y < 0 or y >= img_size

def handle_line_boundary_collision(x, y, angle, img_size):
    """
    Handle boundary collision for line patterns using proper reflection.
    
    Args:
        x, y: Current position
        angle: Current angle in degrees
        img_size: Image size
        
    Returns:
        tuple: (new_x, new_y, new_angle)
    """
    # Determine which boundary was hit
    hit_left = x <= 0
    hit_right = x >= img_size - 1
    hit_top = y <= 0
    hit_bottom = y >= img_size - 1
    
    # Clamp position to valid boundaries
    x = max(0, min(x, img_size - 1))
    y = max(0, min(y, img_size - 1))
    
    # Convert angle to radians for calculation
    angle_rad = math.radians(angle)
    
    # Calculate reflection based on which boundary was hit
    if hit_left or hit_right:
        # Reflect across vertical boundary (negate x-component)
        angle_rad = math.pi - angle_rad
    elif hit_top or hit_bottom:
        # Reflect across horizontal boundary (negate y-component)
        angle_rad = -angle_rad
    
    # Handle corner cases (hit multiple boundaries)
    if (hit_left or hit_right) and (hit_top or hit_bottom):
        # Corner collision: reverse direction
        angle_rad = angle_rad + math.pi
    
    # Convert back to degrees and normalize
    new_angle = math.degrees(angle_rad) % 360
    
    return x, y, new_angle

def generate_unified_pattern(params, sequence_length):
    """
    Generate a trajectory using structured geometric parameters.
    
    Line patterns: constant speed + physics reflection
    Arc patterns: circular motion with fixed radius and angular velocity
    
    Args:
        params (dict): Parameters from sample_unified_parameters()
        sequence_length (int): Number of frames to generate
        
    Returns:
        torch.Tensor: Coordinates of shape [sequence_length, 2]
    """
    coords = torch.zeros(sequence_length, 2)
    img_size = LOCAL_CONFIG['img_size']
    pattern_type = params['pattern_type']
    
    if pattern_type == 'line':
        # Line pattern: constant speed + bounce reflection
        x, y = params['start_pos']
        speed = params['speed']  # pixels per step
        direction = params['direction']  # degrees
        
        # Convert direction to velocity components
        vx = speed * math.cos(math.radians(direction))
        vy = speed * math.sin(math.radians(direction))
        
        coords[0] = torch.tensor([x, y])
        
        for t in range(1, sequence_length):
            # Move with constant velocity
            next_x = x + vx
            next_y = y + vy
            
            # Check boundary collision and reflect
            if check_boundary_collision(next_x, next_y, img_size):
                # Determine which boundary was hit and reflect velocity
                hit_left = next_x <= 0
                hit_right = next_x >= img_size - 1
                hit_top = next_y <= 0
                hit_bottom = next_y >= img_size - 1
                
                # Clamp position to boundaries
                next_x = max(0, min(next_x, img_size - 1))
                next_y = max(0, min(next_y, img_size - 1))
                
                # Reflect velocity components
                if hit_left or hit_right:
                    vx = -vx  # Reflect horizontal velocity
                if hit_top or hit_bottom:
                    vy = -vy  # Reflect vertical velocity
            
            x, y = next_x, next_y
            coords[t] = torch.tensor([x, y])
            
    elif pattern_type == 'arc':
        # Arc pattern: circular motion
        center_x, center_y = params['center_pos']
        radius = params['radius']
        direction = params['direction']  # 'CW' or 'CCW'
        start_angle = params['start_angle']  # radians
        
        # Calculate angular velocity with three different speeds
        # Base angular velocity: complete one revolution in 20 steps
        base_angular_velocity = 2 * math.pi / 20
        
        # Three speed multipliers: [0.3, 0.5, 1.0]
        speed_multipliers = [0.3, 0.6, 1.0]
        speed_multiplier = random.choice(speed_multipliers)
        
        angular_velocity = base_angular_velocity * speed_multiplier
        if direction == 'CW':
            angular_velocity = -angular_velocity  # Clockwise = negative
        
        for t in range(sequence_length):
            # Calculate current angle
            current_angle = start_angle + t * angular_velocity
            
            # Calculate position on circle
            x = center_x + radius * math.cos(current_angle)
            y = center_y + radius * math.sin(current_angle)
            
            # Clamp to image boundaries (shouldn't happen with proper center placement)
            x = max(0, min(x, img_size - 1))
            y = max(0, min(y, img_size - 1))
            
            coords[t] = torch.tensor([x, y])
    
    return coords

def create_sparse_representation(coordinates, mask, img_size):
    """
    Create sparse binary representation from coordinates.
    
    Args:
        coordinates (torch.Tensor): Shape (B, T, 2) or (T, 2)
        mask (torch.Tensor): Boolean mask, shape (B, T) or (T,)
        img_size (int): Image size
        
    Returns:
        torch.Tensor: Binary frames of shape (B, T, 1, H, W) or (T, 1, H, W)
    """
    if coordinates.dim() == 2:  # Single sequence
        coordinates = coordinates.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, T, _ = coordinates.shape
    device = coordinates.device
    frames = torch.zeros(B, T, 1, img_size, img_size, device=device)
    
    for b in range(B):
        for t in range(T):
            if mask[b, t]:
                x, y = coordinates[b, t]
                x_int = int(torch.clamp(x, 0, img_size - 1))
                y_int = int(torch.clamp(y, 0, img_size - 1))
                frames[b, t, 0, y_int, x_int] = 1.0
    
    if squeeze_batch:
        frames = frames.squeeze(0)
    
    return frames

def create_gaussian_representation(coordinates, mask, img_size, sigma=2.0):
    """
    Create dense Gaussian representation from coordinates.
    
    Args:
        coordinates (torch.Tensor): Shape (B, T, 2) or (T, 2)
        mask (torch.Tensor): Boolean mask, shape (B, T) or (T,)
        img_size (int): Image size
        sigma (float): Gaussian standard deviation
        
    Returns:
        torch.Tensor: Gaussian heatmaps of shape (B, T, 1, H, W) or (T, 1, H, W)
    """
    if coordinates.dim() == 2:  # Single sequence
        coordinates = coordinates.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, T, _ = coordinates.shape
    device = coordinates.device
    frames = torch.zeros(B, T, 1, img_size, img_size, device=device)
    
    # Create coordinate grids on same device as coordinates
    y_grid, x_grid = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32, device=device),
        torch.arange(img_size, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    for b in range(B):
        for t in range(T):
            if mask[b, t]:
                x, y = coordinates[b, t]
                
                # Calculate Gaussian heatmap
                dist_sq = (x_grid - x)**2 + (y_grid - y)**2
                gaussian = torch.exp(-dist_sq / (2 * sigma**2))
                
                # CRITICAL FIX: Normalize Gaussian to sum to 1
                gaussian_sum = gaussian.sum()
                if gaussian_sum > 1e-8:  # Avoid division by zero
                    gaussian_normalized = gaussian / gaussian_sum
                else:
                    gaussian_normalized = gaussian
                    
                frames[b, t, 0] = gaussian_normalized
    
    if squeeze_batch:
        frames = frames.squeeze(0)
    
    return frames

class UnifiedGeometricDataset(Dataset):
    """
    Unified dataset for geometric patterns using shared sampling parameters.
    """
    
    def __init__(self, split='train', sequence_length=20, num_samples=None, representation='sparse'):
        """
        Initialize the dataset.
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            sequence_length (int): Length of each sequence
            num_samples (int): Number of samples to generate
            representation (str): Type of representation ('sparse', 'gaussian', 'coord')
        """
        self.split = split
        self.sequence_length = sequence_length
        self.representation = representation
        
        if num_samples is None:
            if split == 'train':
                num_samples = LOCAL_CONFIG['train_samples']
            elif split == 'val':
                num_samples = LOCAL_CONFIG['val_samples']
            else:
                num_samples = LOCAL_CONFIG['test_samples']
        
        self.num_samples = num_samples
        
        # Generate all trajectories with deterministic pattern distribution
        print(f"Generating {num_samples} {split} geometric patterns...")
        self.coordinates = []
        self.pattern_types = []
        
        # Generate exactly half lines, half arcs
        num_lines = num_samples // 2
        num_arcs = num_samples - num_lines
        
        # Generate line patterns
        for i in range(num_lines):
            params = sample_unified_parameters(pattern_idx=i*2, force_pattern_type='line')
            coords = generate_unified_pattern(params, sequence_length)
            self.coordinates.append(coords)
            self.pattern_types.append('line')
        
        # Generate arc patterns
        for i in range(num_arcs):
            params = sample_unified_parameters(pattern_idx=i*2+1, force_pattern_type='arc')
            coords = generate_unified_pattern(params, sequence_length)
            self.coordinates.append(coords)
            self.pattern_types.append('arc')
        
        self.coordinates = torch.stack(self.coordinates)  # (N, T, 2)
        
        # Create mask (all frames are valid for geometric patterns)
        self.mask = torch.ones(num_samples, sequence_length, dtype=torch.bool)
        
        print(f"Generated {num_samples} {split} sequences: "
              f"{sum(1 for p in self.pattern_types if p == 'line')} lines, "
              f"{sum(1 for p in self.pattern_types if p == 'arc')} arcs")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        coords = self.coordinates[idx]
        mask = self.mask[idx]
        
        # Generate frames based on representation
        if hasattr(self, 'representation'):
            repr_type = self.representation
        else:
            # Default to sparse for backward compatibility
            repr_type = 'sparse'
        
        if repr_type == 'sparse':
            # Create sparse binary frames
            frames = create_sparse_representation(coords.unsqueeze(0), mask.unsqueeze(0), LOCAL_CONFIG['img_size']).squeeze(0)
        elif repr_type == 'gaussian':
            # Create Gaussian heatmap frames
            frames = create_gaussian_representation(coords.unsqueeze(0), mask.unsqueeze(0), LOCAL_CONFIG['img_size'], LOCAL_CONFIG['heatmap_sigma_default']).squeeze(0)
        elif repr_type == 'coord':
            # For coord representation, frames are not used by the model
            frames = coords  # Placeholder, will be handled by prepare_batch_for_task
        else:
            raise ValueError(f"Unknown representation type: {repr_type}")
        
        return {
            'coordinates': coords,
            'mask': mask,
            'pattern_type': self.pattern_types[idx],
            'frames': frames
        }

def create_unified_geom_loaders(batch_size=32, sequence_length=20, num_train=None, num_val=None, num_test=None, representation='sparse'):
    """
    Create data loaders for unified geometric patterns.
    
    Args:
        batch_size (int): Batch size
        sequence_length (int): Sequence length
        num_train (int): Number of training samples
        num_val (int): Number of validation samples  
        num_test (int): Number of test samples
        representation (str): Type of representation ('sparse', 'gaussian', 'coord')
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = UnifiedGeometricDataset('train', sequence_length, num_train, representation)
    val_dataset = UnifiedGeometricDataset('val', sequence_length, num_val, representation)
    test_dataset = UnifiedGeometricDataset('test', sequence_length, num_test, representation)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def visualize_samples(dataset, num_samples=5, save_path=None):
    """
    Visualize sample trajectories from the dataset.
    
    Args:
        dataset: UnifiedGeometricDataset instance
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        sample = dataset[i]
        coords = sample['coordinates'].numpy()
        pattern_type = sample['pattern_type']
        
        ax = axes[i]
        ax.plot(coords[:, 0], coords[:, 1], 'o-', markersize=3, linewidth=1)
        ax.set_xlim(0, LOCAL_CONFIG['img_size'])
        ax.set_ylim(LOCAL_CONFIG['img_size'], 0)  # Flip y-axis
        ax.set_aspect('equal')
        ax.set_title(f'{pattern_type.capitalize()}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Test the unified geometric dataset
    print("Testing unified geometric pattern generation...")
    
    # Create dataset
    dataset = UnifiedGeometricDataset('train', sequence_length=20, num_samples=100)
    
    # Create visualization
    output_dir = Path("experiments_refactored/data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_samples(dataset, num_samples=5, save_path=output_dir / "unified_geom_samples.png")
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_unified_geom_loaders(
        batch_size=8, sequence_length=20
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches") 
    print(f"  Test: {len(test_loader)} batches")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch shapes:")
    print(f"  coordinates: {batch['coordinates'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    
    print("Unified geometric pattern generation test completed!")