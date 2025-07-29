"""
COMPATIBILITY LAYER - FOR LEGACY CODE ONLY

This module provides backward compatibility for code that still uses
the original eye_gauss.py interfaces. For new code, please directly use
the position_dependent_gaussian_dataset.py module.

This module will be deprecated in future versions.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Import from the new unified module
from .position_dependent_gaussian_dataset import (
    PositionDependentGaussianDataset,
    calculate_displacement_vectors,
    create_data_loaders
)

# For backward compatibility, re-import the necessary functions from position_dependent_gaussian
from .position_dependent_gaussian import (
    sample_position_dependent_gaussian,
    render_sparse_frames,
    render_gaussian_frames
)

# Keep these functions for backward compatibility
def create_sparse_representation(coordinates, fixation_mask, img_size=32):
    """
    Convert coordinates to sparse binary frames.
    
    Args:
        coordinates (torch.Tensor): Tensor of shape [T, 2] containing (x, y) coordinates.
        fixation_mask (torch.Tensor): Binary mask of shape [T] indicating valid points.
        img_size (int): Size of the output frames (img_size x img_size).
        
    Returns:
        torch.Tensor: Binary frames of shape [T, 1, img_size, img_size].
    """
    T = coordinates.shape[0]
    frames = torch.zeros(T, 1, img_size, img_size)
    
    for t in range(T):
        if fixation_mask[t]:
            # Round and clip coordinates to valid pixel positions
            x, y = coordinates[t]
            x_pixel = torch.clamp(torch.round(x).long(), 0, img_size - 1)
            y_pixel = torch.clamp(torch.round(y).long(), 0, img_size - 1)
            frames[t, 0, y_pixel, x_pixel] = 1.0
    
    return frames

def create_dense_gaussian_representation(coordinates, fixation_mask, img_size=32, sigma=1.5):
    """
    Convert coordinates to dense Gaussian heatmap frames.
    
    Args:
        coordinates (torch.Tensor): Tensor of shape [T, 2] containing (x, y) coordinates.
        fixation_mask (torch.Tensor): Binary mask of shape [T] indicating valid points.
        img_size (int): Size of the output frames (img_size x img_size).
        sigma (float): Standard deviation of the Gaussian kernel.
        
    Returns:
        torch.Tensor: Gaussian heatmap frames of shape [T, 1, img_size, img_size].
    """
    T = coordinates.shape[0]
    frames = torch.zeros(T, 1, img_size, img_size)
    
    # Create coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32),
        torch.arange(img_size, dtype=torch.float32),
        indexing='ij'
    )
    
    for t in range(T):
        if fixation_mask[t]:
            # Get coordinates for this frame
            x, y = coordinates[t]
            
            # Compute Gaussian
            gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            
            # Normalize to sum to 1
            if gaussian.sum() > 0:
                gaussian = gaussian / gaussian.sum()
            
            frames[t, 0] = gaussian
    
    return frames 