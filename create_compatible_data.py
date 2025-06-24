#!/usr/bin/env python
"""
Create data files that match what the dataloader expects
"""

import os
import numpy as np
import shutil


def create_compatible_data():
    """Create symbolic links or copies for dataloader compatibility"""
    
    print("Creating compatible data files...")
    
    data_dir = 'data/coco_search18_tp/processed'
    
    # The dataloader is looking for these files
    expected_files = {
        'coco_search_train.npy': 'short_train_sequences.npy',
        'coco_search_val.npy': 'short_val_sequences.npy'
    }
    
    for expected, actual in expected_files.items():
        expected_path = os.path.join(data_dir, expected)
        actual_path = os.path.join(data_dir, actual)
        
        if os.path.exists(actual_path) and not os.path.exists(expected_path):
            # Create a copy
            shutil.copy2(actual_path, expected_path)
            print(f"  Created: {expected} (copy of {actual})")
    
    print("Done!")


if __name__ == '__main__':
    create_compatible_data()