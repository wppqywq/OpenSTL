#!/usr/bin/env python3
"""
Clean Data Loader for Coordinate Regression
Extracts coordinates from heatmap frames and provides train/test splits
"""

import torch
import numpy as np
from typing import Tuple, Dict


def extract_coordinates_from_frames(frames: torch.Tensor) -> torch.Tensor:
    """
    Extract coordinates from frame tensors using argmax
    
    Args:
        frames: [batch_size, seq_len, channels, height, width]
    
    Returns:
        coords: [batch_size, seq_len, 2] - (x, y) coordinates
    """
    batch_size, seq_len = frames.shape[:2]
    coords = []
    
    for b in range(batch_size):
        batch_coords = []
        for t in range(seq_len):
            frame = frames[b, t].squeeze()  # [H, W]
            flat_idx = torch.argmax(frame.view(-1))
            y = flat_idx // frame.shape[1]
            x = flat_idx % frame.shape[1]
            batch_coords.append([x.item(), y.item()])
        coords.append(batch_coords)
    
    return torch.tensor(coords, dtype=torch.float32)


def load_coordinate_data(data_path: str = '../single_fixation_experiment/data/test_data.pt') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and convert frame data to coordinate sequences
    
    Args:
        data_path: Path to the data file
    
    Returns:
        input_coords: [N, 4, 2] - input coordinate sequences
        target_coords: [N, 16, 2] - target coordinate sequences
    """
    # Load data
    test_data = torch.load(data_path)
    frames = test_data['frames'].permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
    
    print(f"Raw data shape: {frames.shape}")
    
    # Extract coordinates for all frames
    all_coords = []
    for i in range(frames.shape[0]):
        sample_frames = frames[i, :20]  # Use first 20 frames
        coords = extract_coordinates_from_frames(sample_frames.unsqueeze(0))[0]  # [20, 2]
        all_coords.append(coords)
    
    all_coords = torch.stack(all_coords)  # [B, 20, 2]
    
    # Split input/target
    input_coords = all_coords[:, :4]    # [B, 4, 2] - first 4 frames
    target_coords = all_coords[:, 4:20] # [B, 16, 2] - next 16 frames
    
    print(f"Input coords shape: {input_coords.shape}")
    print(f"Target coords shape: {target_coords.shape}")
    
    return input_coords, target_coords


def get_trajectory_statistics(coords: torch.Tensor) -> Dict[str, float]:
    """
    Analyze trajectory statistics for data quality assessment
    
    Args:
        coords: [N, T, 2] coordinate sequences
    
    Returns:
        stats: Dictionary with trajectory statistics
    """
    all_displacements = []
    
    for i in range(coords.shape[0]):
        traj = coords[i]  # [T, 2]
        for t in range(1, len(traj)):
            dx = traj[t, 0] - traj[t-1, 0]
            dy = traj[t, 1] - traj[t-1, 1]
            displacement = torch.sqrt(dx**2 + dy**2).item()
            all_displacements.append(displacement)
    
    displacements = np.array(all_displacements)
    
    stats = {
        'mean_displacement': np.mean(displacements),
        'std_displacement': np.std(displacements),
        'median_displacement': np.median(displacements),
        'max_displacement': np.max(displacements),
        'pct_large_jumps_5px': np.mean(displacements > 5.0) * 100,
        'pct_large_jumps_8px': np.mean(displacements > 8.0) * 100,
        'pct_large_jumps_10px': np.mean(displacements > 10.0) * 100,
    }
    
    return stats


def train_test_split(input_coords: torch.Tensor, target_coords: torch.Tensor, 
                     train_ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into train and test sets
    
    Args:
        input_coords: [N, 4, 2]
        target_coords: [N, 16, 2]
        train_ratio: Fraction for training
    
    Returns:
        train_input, train_target, test_input, test_target
    """
    n_samples = input_coords.shape[0]
    n_train = int(n_samples * train_ratio)
    
    # Random permutation for split
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_input = input_coords[train_indices]
    train_target = target_coords[train_indices]
    test_input = input_coords[test_indices]
    test_target = target_coords[test_indices]
    
    return train_input, train_target, test_input, test_target


if __name__ == "__main__":
    # Test data loading
    input_coords, target_coords = load_coordinate_data()
    
    # Analyze trajectory statistics
    all_coords = torch.cat([input_coords, target_coords], dim=1)  # [N, 20, 2]
    stats = get_trajectory_statistics(all_coords)
    
    print("\n=== Trajectory Statistics ===")
    for key, value in stats.items():
        if 'pct' in key:
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value:.2f} pixels")
    
    # Train/test split
    train_input, train_target, test_input, test_target = train_test_split(input_coords, target_coords)
    
    print(f"\n=== Data Split ===")
    print(f"Train: {train_input.shape[0]} samples")
    print(f"Test: {test_input.shape[0]} samples") 