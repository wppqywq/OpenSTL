#!/usr/bin/env python3
"""
Generate and freeze datasets for reproducible experiments.

This script creates the datasets once and saves them to disk,
ensuring that training and evaluation use the same data.
"""

import torch
from pathlib import Path
import json
from datasets import create_position_dependent_gaussian_loaders, create_geom_simple_loaders

def generate_gauss_datasets():
    """Generate and save gauss datasets"""
    print("Generating gauss datasets...")
    
    data_dir = Path("data/gauss")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets for both representations
    for repr_type in ['sparse', 'gaussian']:
        print(f"  Generating {repr_type} representation...")
        
        train_loader, val_loader, test_loader = create_position_dependent_gaussian_loaders(
            batch_size=32,
            representation=repr_type,
            generate=True
        )
        
        # Save datasets
        torch.save({
            'train_loader': train_loader,
            'val_loader': val_loader, 
            'test_loader': test_loader
        }, data_dir / f"{repr_type}_datasets.pt")
    
    print("Gauss datasets generated successfully!")

def generate_geom_datasets():
    """Generate and save geom datasets"""
    print("Generating geom datasets...")
    
    data_dir = Path("data/geom")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets for all representations
    for repr_type in ['sparse', 'gaussian', 'coord']:
        print(f"  Generating {repr_type} representation...")
        
        train_loader, val_loader, test_loader = create_geom_simple_loaders(
            batch_size=32,
            sequence_length=20,
            representation=repr_type
        )
        
        # Save datasets
        torch.save({
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }, data_dir / f"{repr_type}_datasets.pt")
    
    print("Geom datasets generated successfully!")

def main():
    """Generate all datasets"""
    print("Generating frozen datasets for reproducible experiments...")
    
    generate_gauss_datasets()
    generate_geom_datasets()
    
    print("\nAll datasets generated and saved!")
    print("Training and evaluation will now use these frozen datasets.")

if __name__ == "__main__":
    main() 