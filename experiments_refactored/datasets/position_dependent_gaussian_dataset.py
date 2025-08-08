#!/usr/bin/env python3
"""
Position-dependent Gaussian dataset module.

This module provides a PyTorch Dataset class and data generation functions
for position-dependent Gaussian trajectories.

For visualization utilities, see position_dependent_gaussian_visualization.ipynb
"""

import os
import torch
import numpy as np
from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader

# Import the position_dependent_gaussian module
from .position_dependent_gaussian import (
    sample_position_dependent_gaussian,
    render_sparse_frames,
    render_gaussian_frames,
    generate_dataset,
    DEFAULT_CONFIG
)

# Default parameters
DEFAULT_IMG_SIZE = 32
DEFAULT_SEQUENCE_LENGTH = 20
TRAIN_SIZE = 2500
VAL_SIZE = 100
TEST_SIZE = 100

class PositionDependentGaussianDataset(Dataset):
    """
    Dataset for position-dependent Gaussian data.
    
    This dataset loads coordinate data from .pt files or generates new data
    using position-dependent Gaussian sampling.
    """
    
    def __init__(self, split='train', data_path=None, generate=False, config=None, 
                representation='gaussian', orientation_type='center_directed'):
        """
        Initialize the dataset.
        
        Args:
            split (str): One of 'train', 'val', or 'test'.
            data_path (str, optional): Path to the data directory. If None and generate=False,
                                       defaults to '../../data/position_dependent_gaussian/'.
            generate (bool): Whether to generate new data or load from files.
            config (dict, optional): Configuration for data generation.
            representation (str): Type of representation ('sparse' or 'gaussian').
            orientation_type (str): Type of orientation ('center_directed', 'boundary_tangent', 'random').
        """
        self.split = split
        self.generate = generate
        self.config = config or {}
        self.representation = representation
        
        if 'orientation_type' not in self.config and orientation_type:
            self.config['orientation_type'] = orientation_type
        
        if generate:
            # Generate new data
            batch_size = self.config.get('batch_size', TRAIN_SIZE if split == 'train' else 
                                        VAL_SIZE if split == 'val' else TEST_SIZE)
            sequence_length = self.config.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
            img_size = self.config.get('img_size', DEFAULT_IMG_SIZE)
            
            data = self._generate_data(batch_size, sequence_length, img_size)
            self.frames = data['frames']
            self.coordinates = data['coordinates']
            self.mask = data['mask']
        else:
            # Set default data path if not provided
            if data_path is None:
                data_path = Path("data/position_dependent_gaussian")
            else:
                data_path = Path(data_path)
            
            # Load data based on split and representation
            file_path = data_path / f"{split}_{representation}_data.pt"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            data = torch.load(file_path)
            
            # Extract data
            self.frames = data['frames']
            self.coordinates = data['coordinates']
            self.mask = data['mask']
        
        print(f"Loaded {split} data ({representation}): {self.coordinates.shape[0]} sequences, "
              f"{self.coordinates.shape[1]} frames each")
    
    def _generate_data(self, batch_size, sequence_length, img_size):
        """Generate new data using position-dependent Gaussian sampling."""
        return generate_dataset(
            batch_size, sequence_length, img_size,
            config=self.config, representation=self.representation
        )
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        """
        Get a single sequence.
        
        Returns:
            dict: Contains 'frames', 'coordinates', and 'mask'
        """
        return {
            'frames': self.frames[idx],
            'coordinates': self.coordinates[idx],
            'mask': self.mask[idx]
        }


class DatasetGenerator:
    """Class for generating and saving position-dependent Gaussian datasets."""
    
    def __init__(self, output_dir=None, config=None):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir (str, optional): Directory to save datasets.
            config (dict, optional): Configuration for data generation.
        """
        self.config = config or DEFAULT_CONFIG.copy()
        
        # Dataset parameters
        self.img_size = self.config.get('img_size', DEFAULT_IMG_SIZE)
        self.sequence_length = self.config.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
        self.train_size = self.config.get('train_size', TRAIN_SIZE)
        self.val_size = self.config.get('val_size', VAL_SIZE)
        self.test_size = self.config.get('test_size', TEST_SIZE)
        
        # Get orientation type
        self.orientation_type = self.config.get('orientation_type', 'center_directed')
        
        # Output directory
        if output_dir is None:
            output_dir = Path("data/position_dependent_gaussian")
        self.output_dir = Path(output_dir)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_datasets(self):
        """Generate train, validation, and test datasets."""
        print(f"Generating datasets with orientation type: {self.orientation_type}")
        print(f"Training size: {self.train_size}, Validation size: {self.val_size}, Test size: {self.test_size}")
        
        print("Generating training dataset...")
        train_data_sparse = generate_dataset(self.train_size, self.sequence_length, self.img_size, 
                                     config=self.config, representation='sparse')
        train_data_gaussian = generate_dataset(self.train_size, self.sequence_length, self.img_size, 
                                       config=self.config, representation='gaussian')
        
        print("Generating validation dataset...")
        val_data_sparse = generate_dataset(self.val_size, self.sequence_length, self.img_size, 
                                   config=self.config, representation='sparse')
        val_data_gaussian = generate_dataset(self.val_size, self.sequence_length, self.img_size, 
                                     config=self.config, representation='gaussian')
        
        print("Generating test dataset...")
        test_data_sparse = generate_dataset(self.test_size, self.sequence_length, self.img_size, 
                                  config=self.config, representation='sparse')
        test_data_gaussian = generate_dataset(self.test_size, self.sequence_length, self.img_size, 
                                    config=self.config, representation='gaussian')
        
        # Save datasets
        print("Saving datasets...")
        torch.save(train_data_sparse, self.output_dir / "train_sparse_data.pt")
        torch.save(train_data_gaussian, self.output_dir / "train_gaussian_data.pt")
        torch.save(val_data_sparse, self.output_dir / "val_sparse_data.pt")
        torch.save(val_data_gaussian, self.output_dir / "val_gaussian_data.pt")
        torch.save(test_data_sparse, self.output_dir / "test_sparse_data.pt")
        torch.save(test_data_gaussian, self.output_dir / "test_gaussian_data.pt")
        
        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
            config_dict = {k: v if not isinstance(v, tuple) else list(v) 
                           for k, v in self.config.items()}
            json.dump(config_dict, f, indent=4)
        
        print(f"All datasets saved to {self.output_dir}")
        
        return {
            'train': {'sparse': train_data_sparse, 'gaussian': train_data_gaussian},
            'val': {'sparse': val_data_sparse, 'gaussian': val_data_gaussian},
            'test': {'sparse': test_data_sparse, 'gaussian': test_data_gaussian}
        }


def calculate_displacement_vectors(coordinates, mask):
    """
    Calculate displacement vectors between consecutive valid points.
    
    Args:
        coordinates (torch.Tensor): Tensor of shape [T, 2] containing (x, y) coordinates.
        mask (torch.Tensor): Binary mask of shape [T] indicating valid points.
        
    Returns:
        tuple: (displacements, mask) where displacements is a tensor of shape [T-1, 2]
               and mask is a binary tensor of shape [T-1] indicating valid displacements.
    """
    T = coordinates.shape[0]
    displacements = torch.zeros(T-1, 2)
    displacement_mask = torch.zeros(T-1, dtype=torch.bool)
    
    for t in range(T-1):
        if mask[t] and mask[t+1]:
            displacements[t] = coordinates[t+1] - coordinates[t]
            displacement_mask[t] = True
    
    return displacements, displacement_mask

def create_data_loaders(batch_size=32, data_path=None, generate=False, config=None,
                      representation='gaussian', orientation_type='center_directed'):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        batch_size (int): Batch size for the data loaders.
        data_path (str, optional): Path to the data directory.
        generate (bool): Whether to generate new data or load from files.
        config (dict, optional): Configuration for data generation.
        representation (str): Type of representation ('sparse' or 'gaussian').
        orientation_type (str): Type of orientation ('center_directed', 'boundary_tangent', 'random').
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if config is None:
        config = {}
    
    if 'orientation_type' not in config:
        config['orientation_type'] = orientation_type
    
    train_dataset = PositionDependentGaussianDataset('train', data_path, generate, config, representation)
    val_dataset = PositionDependentGaussianDataset('val', data_path, generate, config, representation)
    test_dataset = PositionDependentGaussianDataset('test', data_path, generate, config, representation)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Main execution function
def main():
    """Main function for generating position-dependent Gaussian datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate position-dependent Gaussian datasets')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save datasets')
    parser.add_argument('--train_size', type=int, default=TRAIN_SIZE, 
                        help='Number of training sequences')
    parser.add_argument('--val_size', type=int, default=VAL_SIZE, 
                        help='Number of validation sequences')
    parser.add_argument('--test_size', type=int, default=TEST_SIZE, 
                        help='Number of test sequences')
    parser.add_argument('--sequence_length', type=int, default=DEFAULT_SEQUENCE_LENGTH, 
                        help='Length of each sequence')
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE, 
                        help='Size of the image grid')
    parser.add_argument('--orientation_type', type=str, default='center_directed',
                        choices=['center_directed', 'boundary_tangent', 'random'],
                        help='Type of directional orientation to use')
    args = parser.parse_args()
    
    # Create configuration from args
    config = DEFAULT_CONFIG.copy()
    config.update({
        'img_size': args.img_size,
        'sequence_length': args.sequence_length,
        'train_size': args.train_size,
        'val_size': args.val_size,
        'test_size': args.test_size,
        'orientation_type': args.orientation_type
    })
    
    # Create dataset generator and run
    generator = DatasetGenerator(args.output_dir, config)
    generator.generate_datasets()

if __name__ == "__main__":
    main() 