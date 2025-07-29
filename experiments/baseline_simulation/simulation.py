#!/usr/bin/env python3
"""
Baseline Simulation Module for Eye Movement Prediction
"""
import torch
import torch.distributions as dist
import numpy as np
import os
from typing import Tuple
import config


def sample_spatial_correct(durations: torch.Tensor, img_size: int, sigma: float) -> torch.Tensor:
    """
    Correct spatial sampling: generate one position per fixation, maintaining the same position throughout duration.
    Uses position-dependent 2D Gaussian for displacement sampling.
    
    Args:
        durations: Tensor of shape (batch_size, T) with countdown durations
        img_size: Size of the image (assumes square images)
        sigma: Standard deviation for fixation-to-fixation displacement (kept for compatibility)
        
    Returns:
        coords: Tensor of shape (batch_size, T, 2) with fixation coordinates
    """
    batch_size, T = durations.shape
    coords = torch.zeros(batch_size, T, 2)
    
    # Define grid with position-dependent parameters
    grid_size = config.grid_size
    
    # Initialize mean vectors: mixed center-biased and random displacement
    center = torch.tensor([grid_size / 2.0, grid_size / 2.0])
    mu = torch.zeros(grid_size, grid_size, 2)
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Vector from current position to center
            pos = torch.tensor([i, j], dtype=torch.float32)
            center_direction = (center - pos) / grid_size  # normalize by grid size
            
            # Add random exploration component
            random_component = torch.randn(2) * config.random_exploration_scale
            
            # Mix center bias with exploration using configurable strength
            center_weight = config.center_bias_strength
            random_weight = 1.0 - config.center_bias_strength
            mixed_direction = center_weight * center_direction + random_weight * random_component
            mu[i, j] = mixed_direction * config.displacement_scale
    
    # Initialize covariance matrices: bimodal distribution for step length variety
    cov_matrices = torch.zeros(grid_size, grid_size, 2, 2)
    for i in range(grid_size):
        for j in range(grid_size):
            # Create bimodal step length distribution using config parameters
            if torch.rand(1) < config.short_step_ratio:
                # Short step: small covariance
                short_min, short_max = config.short_step_cov_range
                diag_vals = torch.rand(2) * (short_max - short_min) + short_min
            else:
                # Long step: large covariance  
                long_min, long_max = config.long_step_cov_range
                diag_vals = torch.rand(2) * (long_max - long_min) + long_min
            cov_matrices[i, j] = torch.diag(diag_vals)
    
    for b in range(batch_size):
        fixation_info = []  # [(start_time, duration), ...]
        t = 0
        while t < T:
            if durations[b, t] > 0:
                duration = int(durations[b, t].item())
                fixation_info.append((t, duration))
                t += duration
            else:
                t += 1
        
        n_fixations = len(fixation_info)
        
        if n_fixations > 0:
            fixation_positions = torch.zeros(n_fixations, 2)
            
            image_center = torch.tensor([img_size / 2.0, img_size / 2.0])
            fixation_positions[0] = torch.normal(image_center, sigma)
            
            for i in range(1, n_fixations):
                # Get current position
                current_pos = fixation_positions[i-1]
                
                # Convert to grid coordinates
                grid_x = (current_pos[0] * grid_size / img_size).clamp(0, grid_size - 1)
                grid_y = (current_pos[1] * grid_size / img_size).clamp(0, grid_size - 1)
                grid_i = int(grid_x)
                grid_j = int(grid_y)
                
                # Get position-dependent parameters
                mean_vec = mu[grid_i, grid_j]
                cov_matrix = cov_matrices[grid_i, grid_j]
                
                # Sample displacement from position-dependent distribution
                mvn = torch.distributions.MultivariateNormal(mean_vec, cov_matrix)
                displacement = mvn.sample()
                
                # Apply displacement
                fixation_positions[i] = current_pos + displacement
            
            fixation_positions = torch.clamp(fixation_positions, 0, img_size)
            
            for fix_idx, (start_time, duration) in enumerate(fixation_info):
                end_time = min(start_time + duration, T)
                coords[b, start_time:end_time, :] = fixation_positions[fix_idx]
    
    return coords.to(torch.float32)


def sample_temporal(batch_size: int, T: int, p: float) -> torch.Tensor:
    """
    Sample fixation durations using ex-Gaussian distribution (Normal + Exponential).
    Fully vectorized implementation for optimal performance on large batches.
    
    Args:
        batch_size: Number of sequences in the batch
        T: Maximum number of time steps per sequence
        p: Parameter repurposed as tau (1/lambda) for exponential distribution
        
    Returns:
        durations: Tensor of shape (batch_size, T) with remaining duration at each time step
    """
    # Ex-Gaussian parameters from config
    mu = config.ex_gaussian_mu        # Mean of normal component
    sigma = config.ex_gaussian_sigma  # Std of normal component  
    tau = config.ex_gaussian_tau      # Scale parameter for exponential
    
    # Sample from ex-Gaussian distribution for each potential fixation
    max_fixations = T  # Conservative estimate
    
    # Draw normal and exponential components: shape (batch_size, max_fixations)
    normal_component = torch.normal(mu, sigma, size=(batch_size, max_fixations))
    # Clamp normal component to prevent negative values before adding exponential
    normal_component_clamped = torch.clamp(normal_component, min=0.0)
    
    exponential_dist = dist.Exponential(1/tau)
    exponential_component = exponential_dist.sample((batch_size, max_fixations))
    
    # Sum to get ex-Gaussian durations and convert to integers >= 1
    continuous_durations = normal_component_clamped + exponential_component
    fixation_durations = torch.ceil(continuous_durations).to(torch.long)
    fixation_durations = torch.clamp(fixation_durations, min=1)
    max_cap = int(3 * (mu + tau))  
    fixation_durations = torch.clamp(fixation_durations, max=max_cap)
    
    # Initialize output tensor with correct shape
    durations = torch.zeros(batch_size, T, dtype=torch.long)
    
    # Vectorized temporal fill using scatter operations
    # Compute cumulative start positions for each fixation
    cumsum_durations = torch.cumsum(fixation_durations, dim=1)
    fixation_starts = torch.cat([
        torch.zeros(batch_size, 1, dtype=torch.long),  # First fixation starts at 0
        cumsum_durations[:, :-1]  # Subsequent fixations start after previous ones
    ], dim=1)
    
    # Process each batch item with vectorized operations within the loop
    # This is a compromise between full vectorization and performance
    for b in range(batch_size):
        current_time = 0
        fixation_idx = 0
        
        while current_time < T and fixation_idx < max_fixations:
            fix_duration = fixation_durations[b, fixation_idx].item()
            actual_duration = min(fix_duration, T - current_time)
            
            if actual_duration > 0:
                # Create countdown sequence and assign vectorized
                end_time = current_time + actual_duration
                countdown_sequence = torch.arange(actual_duration, 0, -1, dtype=torch.long)
                durations[b, current_time:end_time] = countdown_sequence
            
            current_time += actual_duration
            fixation_idx += 1
    
    return durations


def render_visual(coordinates: torch.Tensor, durations: torch.Tensor, img_size: int) -> torch.Tensor:
    """
    Render white dots at fixation coordinates using duration information.
    A fixation appears at the same location for its entire duration.
    
    Args:
        coordinates: Tensor of shape (batch_size, T, 2) with fixation coordinates
        durations: Tensor of shape (batch_size, T) with remaining duration countdown at each time step
        img_size: Size of the image (assumes square images)
        
    Returns:
        frames: Tensor of shape (batch_size, 1, T, img_size, img_size) for OpenSTL input
                Values are in [0,1] range with dtype float32
    """
    batch_size, T, _ = coordinates.shape
    
    # Initialize frames with proper dtype
    frames = torch.zeros(batch_size, T, img_size, img_size, dtype=torch.float32)
    
    for b in range(batch_size):
        current_fixation_coord = None
        
        for t in range(T):
            # If duration > 0, we have an active fixation
            if durations[b, t] > 0:
                # If this is a new fixation: either first timestep, or duration increased, 
                # or previous timestep had no fixation
                if (t == 0 or 
                    durations[b, t] > durations[b, t-1] or 
                    durations[b, t-1] == 0):
                    current_fixation_coord = coordinates[b, t]
                
                # Use the current fixation coordinates (not the coordinates at time t)
                if current_fixation_coord is not None:
                    x, y = current_fixation_coord
                    x_int, y_int = int(x.item()), int(y.item())
                    
                    # Ensure coordinates are within bounds
                    x_int = max(0, min(x_int, img_size - 1))
                    y_int = max(0, min(y_int, img_size - 1))
                    
                    # Set white dot (value = 1.0)
                    frames[b, t, y_int, x_int] = 1.0
    
    # Add channel dimension and reorder for OpenSTL: (B, T, H, W) -> (B, C, T, H, W)
    frames = frames.unsqueeze(2)                    # (B, T, 1, H, W)
    frames = frames.permute(0, 2, 1, 3, 4)          # (B, C=1, T, H, W)
    
    return frames


def generate_sequence(batch_size: int, T: int, img_size: int, sigma: float, p: float) -> dict:
    """
    Generate complete eye movement sequences formatted for OpenSTL input.
    
    Args:
        batch_size: Number of sequences in the batch
        T: Number of time steps per sequence
        img_size: Size of the image (assumes square images)
        sigma: Standard deviation for spatial displacement sampling
        p: Parameter for temporal sampling (repurposed as tau for exponential)
        
    Returns:
        Dictionary with 'frames' key containing tensor of shape (batch_size, 1, T, img_size, img_size)
        Ready for direct input to OpenSTL models
    """
    # Sample temporal durations first
    durations = sample_temporal(batch_size, T, p)
    
    # Sample spatial coordinates based on durations (correct method)
    coordinates = sample_spatial_correct(durations, img_size, sigma)
    
    # Render visual frames with proper OpenSTL formatting
    frames = render_visual(coordinates, durations, img_size)
    
    # Extract actual fixation durations for easy analysis
    fixation_durations = []
    for b in range(batch_size):
        batch_fixations = []
        t = 0
        while t < T:
            if durations[b, t] > 0:
                # Found start of a fixation, get its duration
                fixation_duration = int(durations[b, t].item())
                batch_fixations.append(fixation_duration)
                # Skip to the end of this fixation
                t += fixation_duration
            else:
                t += 1
        fixation_durations.append(batch_fixations)
    
    metadata = {
        'img_size': img_size,
        'sigma': sigma,
        'p': p,
        'ex_gaussian_mu': config.ex_gaussian_mu,
        'ex_gaussian_sigma': config.ex_gaussian_sigma,
        'ex_gaussian_tau': config.ex_gaussian_tau,
        'max_cap': int(3 * (config.ex_gaussian_mu + config.ex_gaussian_tau)),
        'method': 'bimodal_position_dependent_gaussian_sampling',
        'center_bias_strength': config.center_bias_strength,
        'displacement_scale': config.displacement_scale,
        'random_exploration_scale': config.random_exploration_scale,
        'short_step_ratio': config.short_step_ratio,
        'short_step_cov_range': config.short_step_cov_range,
        'long_step_cov_range': config.long_step_cov_range
    }
    
    # Package output with all relevant data
    return {
        "frames": frames,
        "durations": durations,
        "coordinates": coordinates,
        "fixation_durations": fixation_durations,
        "metadata": metadata
    }


def save_data(data: dict, filename: str):
    """Save data dictionary to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(data, filename)
    frames_shape = data["frames"].shape if "frames" in data else "N/A"
    print(f"Saved data to {filename}, frames shape: {frames_shape}")


def generate_all_datasets():
    """Generate all required datasets using config parameters."""
    print("Generating datasets with sigma =", config.sigma)
    
    # Generate training data
    print(f"Generating {config.train_size} training sequences...")
    train_data = generate_sequence(config.train_size, config.sequence_length, config.img_size, config.sigma, config.p)
    save_data(train_data, f"{config.data_dir}train_data.pt")
    
    # Generate validation data
    print(f"Generating {config.val_size} validation sequences...")
    val_data = generate_sequence(config.val_size, config.sequence_length, config.img_size, config.sigma, config.p)
    save_data(val_data, f"{config.data_dir}val_data.pt")
    
    # Generate test data
    print(f"Generating {config.test_size} test sequences...")
    test_data = generate_sequence(config.test_size, config.sequence_length, config.img_size, config.sigma, config.p)
    save_data(test_data, f"{config.data_dir}test_data.pt")
    
    print("All datasets generated successfully!")


if __name__ == "__main__":
    generate_all_datasets() 