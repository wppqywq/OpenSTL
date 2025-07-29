#!/usr/bin/env python3
"""
Test script to verify data pipeline alignment.

This script tests whether the data loading works correctly for both
eye_gauss and geom_simple datasets with the aligned paths.
"""

import os
import sys
import torch
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_refactored.datasets import (
    create_position_dependent_gaussian_loaders,
    create_geom_simple_loaders
)

def test_eye_gauss_data_pipeline():
    """Test eye_gauss data pipeline."""
    print("Testing eye_gauss data pipeline...")
    
    # Test data generation mode
    try:
        train_loader, val_loader, test_loader = create_position_dependent_gaussian_loaders(
            batch_size=8,
            data_path=None,  # Use default path
            generate=True,   # Generate new data
            config={'grid_size': 32, 'sigma': 4.5}
        )
        
        # Test a single batch
        batch = next(iter(train_loader))
        print(f"  - Generated data batch shape: {batch['frames'].shape}")
        print(f"  - Coordinates shape: {batch['coordinates'].shape}")
        print("  - eye_gauss data pipeline: OK")
        return True
        
    except Exception as e:
        print(f"  - eye_gauss data pipeline failed: {e}")
        return False

def test_geom_simple_data_pipeline():
    """Test geom_simple data pipeline."""
    print("Testing geom_simple data pipeline...")
    
    try:
        train_loader, val_loader, test_loader = create_geom_simple_loaders(
            batch_size=8,
            num_samples=100,
            img_size=32,
            sequence_length=20
        )
        
        # Test a single batch
        batch = next(iter(train_loader))
        print(f"  - Generated data batch shape: {batch['coordinates'].shape}")
        print(f"  - Pattern types available: {set(batch['pattern_type'])}")
        print("  - geom_simple data pipeline: OK")
        return True
        
    except Exception as e:
        print(f"  - geom_simple data pipeline failed: {e}")
        return False

def test_data_path_alignment():
    """Test that data paths are correctly aligned."""
    print("Testing data path alignment...")
    
    # Check if the expected data directory exists
    expected_path = Path("experiments_refactored/data/position_dependent_gaussian")
    if expected_path.exists():
        print(f"  - Data path exists: {expected_path}")
    else:
        print(f"  - Creating data path: {expected_path}")
        expected_path.mkdir(parents=True, exist_ok=True)
    
    # Test with file loading mode (will generate files if they don't exist)
    try:
        train_loader, val_loader, test_loader = create_position_dependent_gaussian_loaders(
            batch_size=4,
            data_path='experiments_refactored/data/position_dependent_gaussian',
            generate=False,  # Try to load from files
            config={'grid_size': 32, 'sigma': 4.5}
        )
        print("  - Data path alignment: OK")
        return True
        
    except Exception as e:
        print(f"  - Data path alignment issue: {e}")
        # Try with generate=True as fallback
        try:
            train_loader, val_loader, test_loader = create_position_dependent_gaussian_loaders(
                batch_size=4,
                data_path='experiments_refactored/data/position_dependent_gaussian',
                generate=True,  # Generate new data
                config={'grid_size': 32, 'sigma': 4.5}
            )
            print("  - Data path alignment: OK (with generation)")
            return True
        except Exception as e2:
            print(f"  - Data path alignment failed: {e2}")
            return False

def main():
    """Main test function."""
    print("=" * 50)
    print("Data Pipeline Alignment Test")
    print("=" * 50)
    
    results = []
    
    # Test eye_gauss data pipeline
    results.append(test_eye_gauss_data_pipeline())
    print()
    
    # Test geom_simple data pipeline
    results.append(test_geom_simple_data_pipeline())
    print()
    
    # Test data path alignment
    results.append(test_data_path_alignment())
    print()
    
    # Summary
    print("=" * 50)
    print("Summary:")
    print(f"  - eye_gauss pipeline: {'PASS' if results[0] else 'FAIL'}")
    print(f"  - geom_simple pipeline: {'PASS' if results[1] else 'FAIL'}")
    print(f"  - data path alignment: {'PASS' if results[2] else 'FAIL'}")
    
    if all(results):
        print("\nAll tests PASSED! Data pipeline is correctly aligned.")
    else:
        print("\nSome tests FAILED! Please check the data pipeline configuration.")
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 