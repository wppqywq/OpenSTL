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
from .constants import FIELD_CONFIG, IMG_SIZE, RANDOM_SEED

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Default configuration
DEFAULT_CONFIG = {
    'grid_size': FIELD_CONFIG['grid_size'],
    'sigma': FIELD_CONFIG['sigma_start'],
    'displacement_scale': FIELD_CONFIG['displacement_scale'],
    'max_attempts': 5,
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

def init_covariance_field(grid_size, img_size):
    """
    Initialize a structured covariance field based on position-dependent rules.
    
    Args:
        grid_size (int): Size of the parameter grid
        img_size (int): Size of the image
        
    Returns:
        tuple: (theta_field, lambda1_field, lambda2_field) each of shape (grid_size, grid_size)
    """
    theta_field = torch.zeros(grid_size, grid_size)
    lambda1_field = torch.zeros(grid_size, grid_size)
    lambda2_field = torch.zeros(grid_size, grid_size)
    
    # Create coordinate grids
    y_coords = torch.linspace(0, img_size, grid_size)
    x_coords = torch.linspace(0, img_size, grid_size)
    yv, xv = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Center coordinates
    center_y, center_x = img_size / 2.0, img_size / 2.0
    
    # Distance from center (normalized by max possible distance)
    distance_from_center = torch.sqrt((yv - center_y)**2 + (xv - center_x)**2)
    max_distance = torch.sqrt(torch.tensor((img_size/2)**2 + (img_size/2)**2))
    normalized_distance = distance_from_center / max_distance
    
    for gy in range(grid_size):
        for gx in range(grid_size):
            y, x = yv[gy, gx], xv[gy, gx]
            dist_norm = normalized_distance[gy, gx]
            
            # Determine region and set parameters accordingly
            if dist_norm < FIELD_CONFIG['center_threshold']:
                # Center region: isotropic (random)
                theta_field[gy, gx] = torch.rand(1).item() * 2 * math.pi
                lambda1_field[gy, gx] = FIELD_CONFIG['center_lambda1']
                lambda2_field[gy, gx] = FIELD_CONFIG['center_lambda2']
                
            elif dist_norm > FIELD_CONFIG['corner_threshold']:
                # Corner regions: toward center
                dy, dx = center_y - y, center_x - x
                theta_field[gy, gx] = math.atan2(dy, dx)
                lambda1_field[gy, gx] = FIELD_CONFIG['corner_lambda1']
                lambda2_field[gy, gx] = FIELD_CONFIG['corner_lambda2']
                
            elif dist_norm > FIELD_CONFIG['edge_threshold']:
                # Edge regions: along the border (tangential)
                dy, dx = center_y - y, center_x - x
                center_angle = math.atan2(dy, dx)
                theta_field[gy, gx] = center_angle + math.pi/2  # Perpendicular to center direction
                lambda1_field[gy, gx] = FIELD_CONFIG['edge_lambda1']
                lambda2_field[gy, gx] = FIELD_CONFIG['edge_lambda2']
                
            else:
                # Quadrant regions: rotating patterns (CW/CCW)
                y_norm, x_norm = (y - center_y) / (img_size/2), (x - center_x) / (img_size/2)
                
                # Determine quadrant and rotation direction
                if y_norm > 0 and x_norm > 0:  # Top-right: CW
                    base_angle = math.atan2(-x_norm, y_norm)  # Clockwise rotation
                elif y_norm > 0 and x_norm < 0:  # Top-left: CCW
                    base_angle = math.atan2(x_norm, y_norm)  # Counter-clockwise rotation
                elif y_norm < 0 and x_norm < 0:  # Bottom-left: CW
                    base_angle = math.atan2(-x_norm, y_norm)  # Clockwise rotation
                else:  # Bottom-right: CCW
                    base_angle = math.atan2(x_norm, y_norm)  # Counter-clockwise rotation
                
                theta_field[gy, gx] = base_angle
                lambda1_field[gy, gx] = FIELD_CONFIG['quadrant_lambda1']
                lambda2_field[gy, gx] = FIELD_CONFIG['quadrant_lambda2']
    
    return theta_field, lambda1_field, lambda2_field

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
    
    elif orientation_type == 'structured_field':
        # Structured field with strong directional preferences
        y_norm, x_norm = (y - center) / center, (x - center) / center
        dist_from_center = math.sqrt(y_norm**2 + x_norm**2)
        
        # Define regions with clearer boundaries
        edge_threshold = 0.7  # Distance from center to be considered "edge"
        corner_threshold = 0.85  # Distance from center to be considered "corner"
        
        # Corner regions: ALWAYS toward center (no randomness)
        if dist_from_center > corner_threshold:
            dy, dx = center - y, center - x
            return math.atan2(dy, dx)
        
        # Edge regions: mix of toward center and tangential
        elif dist_from_center > edge_threshold:
            dy, dx = center - y, center - x
            center_angle = math.atan2(dy, dx)
            # 70% toward center, 30% tangential
            if hash((int(x*10), int(y*10))) % 10 < 7:
                return center_angle  # Toward center
            else:
                return center_angle + math.pi/2  # Tangential
        
        # Center region: weak center bias (not random)
        elif dist_from_center < 0.2:
            dy, dx = center - y, center - x
            center_angle = math.atan2(dy, dx)
            # Add small random perturbation but still generally toward center
            perturbation = (hash((int(x*100), int(y*100))) % 360 - 180) * math.pi / 180 * 0.5
            return center_angle + perturbation
        
        # Quadrant regions: strong toward center with slight rotation
        else:
            dy, dx = center - y, center - x
            center_angle = math.atan2(dy, dx)
            
            # Add quadrant-specific rotation bias
            if y_norm > 0 and x_norm > 0:  # Top-right: slight CW bias
                rotation_bias = -math.pi/6  # -30 degrees
            elif y_norm > 0 and x_norm < 0:  # Top-left: slight CCW bias
                rotation_bias = math.pi/6   # +30 degrees
            elif y_norm < 0 and x_norm < 0:  # Bottom-left: slight CW bias
                rotation_bias = -math.pi/6  # -30 degrees
            else:  # Bottom-right: slight CCW bias
                rotation_bias = math.pi/6   # +30 degrees
            
            return center_angle + rotation_bias
    
    # Default: random orientation
    return torch.rand(1).item() * 2 * math.pi

def sample_position_dependent_gaussian(batch_size, sequence_length, img_size, config=None):
    """
    Sample spatial coordinates using structured position-dependent Gaussian field.
    
    Each location has a covariance matrix that encodes directional preference:
    - Corners: toward center (strong directional bias)
    - Edges: along border (consistent low-entropy movement)
    - Center: isotropic (high uncertainty)
    - Quadrants: rotating patterns (CW/CCW spatial variation)
    
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
    
    # Initialize structured covariance field
    grid_size = config.get('grid_size', 32)
    theta_field, lambda1_field, lambda2_field = init_covariance_field(grid_size, img_size)
    
    # Parameters
    sigma = config.get('sigma', 2.0)
    displacement_scale = config.get('displacement_scale', 3.0)
    max_attempts = config.get('max_attempts', 5)
    
    for b in range(batch_size):
        # Start near center with Gaussian spread
        image_center = torch.tensor([img_size / 2.0, img_size / 2.0])
        coords[b, 0] = torch.normal(image_center, sigma)
        # Clamp to valid bounds
        coords[b, 0] = torch.clamp(coords[b, 0], 0, img_size - 1)
        
        for t in range(1, sequence_length):
            current_pos = coords[b, t-1]
            
            # Find nearest grid point for covariance lookup
            grid_y = torch.clamp(torch.round(current_pos[0] * (grid_size-1) / img_size).long(), 0, grid_size-1)
            grid_x = torch.clamp(torch.round(current_pos[1] * (grid_size-1) / img_size).long(), 0, grid_size-1)
            
            # Get covariance parameters for this position
            theta_pos = theta_field[grid_y, grid_x]
            lambda1_pos = lambda1_field[grid_y, grid_x]
            lambda2_pos = lambda2_field[grid_y, grid_x]
            
            # Build covariance matrix
            cov_matrix = build_cov_matrix(theta_pos, lambda1_pos, lambda2_pos)
            
            # Sample displacement from 2D multivariate normal
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < max_attempts:
                # Sample displacement from the structured covariance
                displacement_raw = torch.distributions.MultivariateNormal(
                    torch.zeros(2), cov_matrix
                ).sample()
                
                # Scale displacement
                displacement = displacement_raw * displacement_scale
                
                # Check if new position is within bounds
                new_pos = current_pos + displacement
                if (0 <= new_pos[0] < img_size and 0 <= new_pos[1] < img_size):
                    valid_position = True
                    coords[b, t] = new_pos
                else:
                    # Reduce displacement scale and try again
                    displacement_scale *= 0.8
                    attempts += 1
            
            # If no valid position found after max attempts, clamp to image bounds
            if not valid_position:
                coords[b, t] = torch.clamp(current_pos + displacement, 0, img_size - 1)
            
            # Reset displacement scale for next step
            displacement_scale = config.get('displacement_scale', 3.0)
    
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