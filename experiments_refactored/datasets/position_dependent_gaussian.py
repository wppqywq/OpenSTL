#!/usr/bin/env python3
"""
Position-dependent Gaussian sampling module.

This module provides core functions for sampling position-dependent Gaussian coordinates
and rendering them into sparse or dense frame representations.

Key Parameters:
- lambda1, lambda2: Eigenvalues of the covariance matrix (lambda1 > lambda2)
  - lambda1: Major axis variance (larger value = more spread in that direction)
  - lambda2: Minor axis variance (smaller value = more elliptical shape)
  - Ratio lambda2/lambda1 controls ellipticity: closer to 1 = more circular, closer to 0 = more elliptical

For visualization and dataset generation, see position_dependent_gaussian_dataset.py
"""

import torch
import math
import numpy as np
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    'grid_size': 32,
    'sigma': 4.5,
    'displacement_scale': 8.0,
    'center_bias_strength': 0.55,
    'random_exploration_scale': 0.65,
    'short_step_ratio': 0.4,
    'short_step_cov_range': (0.2, 1.5),  # Modified: more reasonable values
    'long_step_cov_range': (4.0, 6.0),   # Modified: more reasonable values
    'short_step_lambda_ratio': 0.4,      # lambda2/lambda1 for short steps (0.4 = more elliptical)
    'long_step_lambda_ratio': 0.2,       # lambda2/lambda1 for long steps (0.2 = more elliptical)
    'max_attempts': 5,  # Maximum attempts for resampling when hitting boundaries
    'orientation_type': 'center_directed'  # Options: 'random', 'center_directed', 'boundary_tangent'
}

def build_cov_matrix(theta, lambda1, lambda2):
    """
    Build a 2D covariance matrix with orientation theta and eigenvalues lambda1, lambda2.
    
    Args:
        theta (float): Rotation angle in radians.
        lambda1 (float): First eigenvalue (variance along the primary axis).
        lambda2 (float): Second eigenvalue (variance along the secondary axis).
        
    Returns:
        torch.Tensor: 2x2 covariance matrix.
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    R = torch.tensor([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])
    Lambda = torch.diag(torch.tensor([lambda1, lambda2]))
    return R @ Lambda @ R.T  # shape: (2, 2)

def get_orientation_angle(pos, img_size, orientation_type):
    """
    Determine the orientation angle based on position and orientation type.
    
    Args:
        pos (torch.Tensor): Current position [y, x].
        img_size (int): Size of the image.
        orientation_type (str): Type of orientation pattern.
        
    Returns:
        float: Orientation angle in radians.
    """
    center = img_size / 2
    y, x = pos
    
    if orientation_type == 'random':
        # Random orientation
        return torch.rand(1).item() * 2 * math.pi
    
    elif orientation_type == 'center_directed':
        # Orientation pointing towards/away from center
        dy, dx = center - y, center - x
        return math.atan2(dy, dx)
    
    elif orientation_type == 'boundary_tangent':
        # Orientation tangential to the boundary
        # Normalize position to [-1, 1] relative to center
        y_norm, x_norm = (y - center) / center, (x - center) / center
        # Get angle to center
        angle_to_center = math.atan2(y_norm, x_norm)
        # Make tangent by adding π/2
        return angle_to_center + math.pi/2
    
    # Default: random orientation
    return torch.rand(1).item() * 2 * math.pi

def sample_position_dependent_gaussian(batch_size, sequence_length, img_size, config=None):
    """
    Sample spatial coordinates using position-dependent Gaussian distribution.
    
    Args:
        batch_size (int): Number of sequences to generate.
        sequence_length (int): Length of each sequence.
        img_size (int): Size of the image (img_size x img_size).
        config (dict, optional): Configuration parameters. If None, uses DEFAULT_CONFIG.
        
    Returns:
        torch.Tensor: Coordinates tensor of shape (batch_size, sequence_length, 2).
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    coords = torch.zeros(batch_size, sequence_length, 2)
    
    # Create position-dependent parameter grid
    grid_size = config.get('grid_size', 32)
    y_grid = torch.linspace(0, img_size, grid_size)
    x_grid = torch.linspace(0, img_size, grid_size)
    yv, xv = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Center coordinates
    center_y, center_x = img_size / 2.0, img_size / 2.0
    
    # Distance from center for each grid point
    distance_from_center = torch.sqrt((yv - center_y)**2 + (xv - center_x)**2)
    max_distance = torch.sqrt(torch.tensor((img_size/2)**2 + (img_size/2)**2))
    normalized_distance = distance_from_center / max_distance
    
    # Create bimodal covariance distribution
    short_step_ratio = config.get('short_step_ratio', 0.4)
    short_step_mask = torch.rand(grid_size, grid_size) < short_step_ratio
    
    # Initialize primary and secondary eigenvalues for covariance matrices
    # lambda1: Major axis variance (larger value = more spread in that direction)
    # lambda2: Minor axis variance (smaller value = more elliptical shape)
    # Ratio lambda2/lambda1 controls ellipticity: closer to 1 = more circular, closer to 0 = more elliptical
    lambda1 = torch.zeros(grid_size, grid_size)
    lambda2 = torch.zeros(grid_size, grid_size)
    
    # Short steps
    short_step_cov_range = config.get('short_step_cov_range', (0.2, 1.5))
    short_lambda1 = torch.rand(grid_size, grid_size) * (
        short_step_cov_range[1] - short_step_cov_range[0]) + short_step_cov_range[0]
    # Make more elliptical by reducing lambda2 relative to lambda1
    short_lambda2 = short_lambda1 * config.get('short_step_lambda_ratio', 0.4)
    
    lambda1[short_step_mask] = short_lambda1[short_step_mask]
    lambda2[short_step_mask] = short_lambda2[short_step_mask]
    
    # Long steps
    long_step_cov_range = config.get('long_step_cov_range', (4.0, 6.0))
    long_lambda1 = torch.rand(grid_size, grid_size) * (
        long_step_cov_range[1] - long_step_cov_range[0]) + long_step_cov_range[0]
    # Make more elliptical by reducing lambda2 relative to lambda1
    long_lambda2 = long_lambda1 * config.get('long_step_lambda_ratio', 0.3)
    
    lambda1[~short_step_mask] = long_lambda1[~short_step_mask]
    lambda2[~short_step_mask] = long_lambda2[~short_step_mask]
    
    # Generate sequences
    sigma = config.get('sigma', 4.5)
    displacement_scale = config.get('displacement_scale', 9.0)
    center_bias_strength = config.get('center_bias_strength', 0.55)
    random_exploration_scale = config.get('random_exploration_scale', 0.65)
    orientation_type = config.get('orientation_type', 'center_directed')
    max_attempts = config.get('max_attempts', 5)
    
    for b in range(batch_size):
        # Start near center
        image_center = torch.tensor([img_size / 2.0, img_size / 2.0])
        coords[b, 0] = torch.normal(image_center, sigma)
        
        for t in range(1, sequence_length):
            current_pos = coords[b, t-1]
            
            # Find nearest grid point
            grid_y = torch.clamp(torch.round(current_pos[0] * (grid_size-1) / img_size).long(), 0, grid_size-1)
            grid_x = torch.clamp(torch.round(current_pos[1] * (grid_size-1) / img_size).long(), 0, grid_size-1)
            
            # Get eigenvalues for this position
            lambda1_pos = lambda1[grid_y, grid_x]
            lambda2_pos = lambda2[grid_y, grid_x]
            
            # Get orientation angle for this position
            theta = get_orientation_angle(current_pos, img_size, orientation_type)
            
            # Build the covariance matrix with orientation
            cov_matrix = build_cov_matrix(theta, lambda1_pos, lambda2_pos)
            
            # Center-biased displacement
            center_direction = torch.tensor([center_y, center_x]) - current_pos
            center_direction = center_direction / (torch.norm(center_direction) + 1e-8)
            
            # Random exploration component
            random_direction = torch.randn(2)
            random_direction = random_direction / (torch.norm(random_direction) + 1e-8)
            
            # Combine directions
            displacement_direction = (center_bias_strength * center_direction + 
                                   random_exploration_scale * random_direction)
            displacement_direction = displacement_direction / (torch.norm(displacement_direction) + 1e-8)
            
            # Sample directly from multivariate normal with proper covariance matrix
            # Loop to handle boundary constraints with resampling
            valid_position = False
            attempts = 0
            
            # Initialize displacement with a default value
            displacement = torch.distributions.MultivariateNormal(
                torch.zeros(2), cov_matrix).sample() * displacement_scale
            
            while not valid_position and attempts < max_attempts:
                # Sample from multivariate normal
                displacement = torch.distributions.MultivariateNormal(
                    torch.zeros(2), cov_matrix).sample() * displacement_scale
                
                # Apply center bias and random exploration components
                bias_factor = torch.dot(displacement, displacement_direction) / (torch.norm(displacement) + 1e-8)
                bias_adjustment = 0.2 * bias_factor  # Subtle influence to preferred direction
                displacement = displacement * (1 + bias_adjustment)
                
                # Check if new position is within bounds
                new_pos = current_pos + displacement
                if (0 <= new_pos[0] < img_size and 0 <= new_pos[1] < img_size):
                    valid_position = True
                    coords[b, t] = new_pos
                else:
                    attempts += 1
            
            # If no valid position found after max attempts, clamp to image bounds
            if not valid_position:
                coords[b, t] = torch.clamp(current_pos + displacement, 0, img_size - 1)
    
    return coords.to(torch.float32)

def render_sparse_frames(coords, img_size):
    """
    Render sparse binary frames from coordinates.
    
    Args:
        coords (torch.Tensor): Coordinates tensor of shape (batch_size, sequence_length, 2).
        img_size (int): Size of the image (img_size x img_size).
        
    Returns:
        torch.Tensor: Binary frames of shape (batch_size, sequence_length, 1, img_size, img_size).
    """
    batch_size, sequence_length, _ = coords.shape
    frames = torch.zeros(batch_size, sequence_length, 1, img_size, img_size)
    
    for b in range(batch_size):
        for t in range(sequence_length):
            y, x = coords[b, t]
            y_int, x_int = int(round(y.item())), int(round(x.item()))
            
            # Clamp coordinates to valid range
            y_int = max(0, min(img_size - 1, y_int))
            x_int = max(0, min(img_size - 1, x_int))
            
            frames[b, t, 0, y_int, x_int] = 1.0
    
    return frames

def render_gaussian_frames(coords, img_size, sigma=1.5):
    """
    Render Gaussian heatmap frames from coordinates.
    
    Args:
        coords (torch.Tensor): Coordinates tensor of shape (batch_size, sequence_length, 2).
        img_size (int): Size of the image (img_size x img_size).
        sigma (float): Standard deviation of the Gaussian kernel.
        
    Returns:
        torch.Tensor: Gaussian frames of shape (batch_size, sequence_length, 1, img_size, img_size).
    """
    batch_size, sequence_length, _ = coords.shape
    frames = torch.zeros(batch_size, sequence_length, 1, img_size, img_size)
    
    # Create coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32),
        torch.arange(img_size, dtype=torch.float32),
        indexing='ij'
    )
    
    for b in range(batch_size):
        for t in range(sequence_length):
            y, x = coords[b, t]
            
            # Compute Gaussian
            gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            
            # Normalize to sum to 1
            if gaussian.sum() > 0:
                gaussian = gaussian / gaussian.sum()
            
            frames[b, t, 0] = gaussian
    
    return frames

def generate_dataset(batch_size, sequence_length, img_size, config=None, representation='sparse'):
    """
    Generate a dataset of position-dependent Gaussian sequences.
    
    Args:
        batch_size (int): Number of sequences to generate.
        sequence_length (int): Length of each sequence.
        img_size (int): Size of the image (img_size x img_size).
        config (dict, optional): Configuration parameters. If None, uses DEFAULT_CONFIG.
        representation (str): Type of representation ('sparse' or 'gaussian').
        
    Returns:
        dict: Dictionary containing 'frames', 'coords', and 'mask'.
    """
    # Sample coordinates
    coords = sample_position_dependent_gaussian(batch_size, sequence_length, img_size, config)
    
    # Create mask (all valid for this dataset)
    mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    
    # Render frames based on representation
    if representation == 'sparse':
        frames = render_sparse_frames(coords, img_size)
    elif representation == 'gaussian':
        sigma = config.get('gaussian_sigma', 1.5) if config else 1.5
        frames = render_gaussian_frames(coords, img_size, sigma)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    return {
        'frames': frames,
        'coordinates': coords,
        'mask': mask
    } 