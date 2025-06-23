#!/usr/bin/env python

import os
import numpy as np
import argparse
from pathlib import Path


def load_coco_search_raw(data_dir):
    """
    Load raw COCO-Search data
    This function should be adapted based on the actual COCO-Search data format
    """
    # Placeholder for actual COCO-Search data loading
    # The actual implementation would depend on the specific format of COCO-Search18
    
    print("Note: This is a placeholder function.")
    print("Please implement actual COCO-Search data loading based on your data format.")
    
    # Expected format after loading:
    # scanpaths: List of scanpath sequences
    # Each scanpath: numpy array of shape [sequence_length, 2] with (x, y) coordinates
    
    return []


def normalize_coordinates(scanpaths, image_width=1920, image_height=1080):
    """Normalize coordinates to [0, 1] range"""
    normalized_scanpaths = []
    
    for scanpath in scanpaths:
        normalized = scanpath.copy()
        normalized[:, 0] = normalized[:, 0] / image_width   # Normalize x
        normalized[:, 1] = normalized[:, 1] / image_height  # Normalize y
        
        # Clip to [0, 1] range
        normalized = np.clip(normalized, 0, 1)
        normalized_scanpaths.append(normalized)
    
    return normalized_scanpaths


def pad_or_truncate_sequences(scanpaths, target_length=20):
    """Pad or truncate sequences to target length"""
    processed_scanpaths = []
    
    for scanpath in scanpaths:
        seq_length = len(scanpath)
        
        if seq_length >= target_length:
            # Truncate
            processed = scanpath[:target_length]
        else:
            # Pad with last coordinate
            last_coord = scanpath[-1]
            padding = np.tile(last_coord, (target_length - seq_length, 1))
            processed = np.vstack([scanpath, padding])
        
        processed_scanpaths.append(processed)
    
    return np.array(processed_scanpaths)


def split_train_test(data, test_ratio=0.2, random_seed=42):
    """Split data into train and test sets"""
    np.random.seed(random_seed)
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    n_test = int(n_samples * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    return train_data, test_data


def generate_dummy_data(n_samples=1000, seq_length=20):
    """Generate dummy scanpath data for testing"""
    print(f"Generating {n_samples} dummy scanpath sequences...")
    
    # Generate realistic-looking scanpaths
    data = []
    
    for i in range(n_samples):
        # Start from random position
        start_x, start_y = np.random.uniform(0.2, 0.8, 2)
        
        # Generate sequence with some temporal coherence
        scanpath = np.zeros((seq_length, 2))
        scanpath[0] = [start_x, start_y]
        
        for t in range(1, seq_length):
            # Add some random walk with drift toward center
            drift_x = (0.5 - scanpath[t-1, 0]) * 0.1
            drift_y = (0.5 - scanpath[t-1, 1]) * 0.1
            
            noise_x = np.random.normal(0, 0.05)
            noise_y = np.random.normal(0, 0.05)
            
            new_x = scanpath[t-1, 0] + drift_x + noise_x
            new_y = scanpath[t-1, 1] + drift_y + noise_y
            
            # Clip to valid range
            scanpath[t] = [np.clip(new_x, 0, 1), np.clip(new_y, 0, 1)]
        
        data.append(scanpath)
    
    return np.array(data)


def main():
    parser = argparse.ArgumentParser(description='Preprocess COCO-Search data for SimVP')
    parser.add_argument('--input_dir', type=str, default='./raw_coco_search',
                       help='Directory containing raw COCO-Search data')
    parser.add_argument('--output_dir', type=str, default='./data/coco_search',
                       help='Output directory for processed data')
    parser.add_argument('--seq_length', type=int, default=20,
                       help='Target sequence length')
    parser.add_argument('--dummy', action='store_true',
                       help='Generate dummy data instead of processing real data')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of dummy samples to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dummy:
        print("Generating dummy data...")
        data = generate_dummy_data(args.n_samples, args.seq_length)
    else:
        print(f"Loading COCO-Search data from {args.input_dir}")
        raw_scanpaths = load_coco_search_raw(args.input_dir)
        
        if not raw_scanpaths:
            print("No data loaded. Generating dummy data instead.")
            data = generate_dummy_data(args.n_samples, args.seq_length)
        else:
            print("Normalizing coordinates...")
            normalized_scanpaths = normalize_coordinates(raw_scanpaths)
            
            print("Padding/truncating sequences...")
            data = pad_or_truncate_sequences(normalized_scanpaths, args.seq_length)
    
    print(f"Processed data shape: {data.shape}")
    
    # Split into train and test
    train_data, test_data = split_train_test(data)
    
    # Save processed data
    train_path = os.path.join(args.output_dir, 'coco_search_train.npy')
    test_path = os.path.join(args.output_dir, 'coco_search_test.npy')
    
    np.save(train_path, train_data)
    np.save(test_path, test_data)
    
    print(f"Saved train data: {train_path} (shape: {train_data.shape})")
    print(f"Saved test data: {test_path} (shape: {test_data.shape})")
    print("Data preprocessing completed!")


if __name__ == '__main__':
    main()