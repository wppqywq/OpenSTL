#!/usr/bin/env python
"""
Check the structure of COCO-Search18 JSON files to understand field names
"""

import json
import os
from pathlib import Path


def check_coco_search_structure(data_root):
    """Examine the structure of COCO-Search18 JSON files"""
    
    data_root = Path(data_root)
    fixations_dir = data_root / 'fixations'
    
    # Find a JSON file
    json_files = list(fixations_dir.glob('*.json'))
    if not json_files:
        print("No JSON files found in", fixations_dir)
        return
    
    # Load first file
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {json_files[0].name}")
    print(f"Number of trials: {len(data)}")
    print("\nSample trial structure:")
    
    if data:
        # Show keys from first trial
        trial = data[0]
        print("\nAll keys in first trial:")
        for key in sorted(trial.keys()):
            value = trial[key]
            if isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0:
                    print(f"    First item: {value[0]}")
            else:
                print(f"  {key}: {value}")
        
        # Check for image information
        print("\nChecking for image identification fields:")
        for key in ['image_id', 'name', 'imagename', 'image', 'stimulus', 'img_name']:
            if key in trial:
                print(f"  Found '{key}': {trial[key]}")
        
        # Check for target information
        print("\nChecking for target information fields:")
        for key in ['target_name', 'task', 'target', 'target_object', 'category']:
            if key in trial:
                print(f"  Found '{key}': {trial[key]}")
        
        # Check if coordinates exist
        print("\nChecking coordinate data:")
        if 'X' in trial and 'Y' in trial:
            print(f"  X coordinates: {len(trial['X'])} points")
            print(f"  Y coordinates: {len(trial['Y'])} points")
            print(f"  First 5 X: {trial['X'][:5]}")
            print(f"  First 5 Y: {trial['Y'][:5]}")
        
        # Show all unique values for important fields across dataset
        print("\nAnalyzing field variations across dataset:")
        field_values = {}
        for field in ['task', 'condition', 'dataset', 'subject']:
            if field in trial:
                values = set()
                for t in data[:100]:  # Check first 100 trials
                    if field in t:
                        values.add(str(t[field]))
                field_values[field] = sorted(list(values))
                print(f"\n  Unique values for '{field}':")
                for v in field_values[field][:10]:  # Show first 10
                    print(f"    - {v}")
                if len(field_values[field]) > 10:
                    print(f"    ... and {len(field_values[field])-10} more")


def check_image_directory(data_root):
    """Check the structure of the images directory"""
    
    data_root = Path(data_root)
    images_dir = data_root / 'images'
    
    if not images_dir.exists():
        print(f"\nImages directory not found at {images_dir}")
        return
    
    print(f"\n\nImage Directory Structure:")
    print(f"Location: {images_dir}")
    
    # List all subdirectories (categories)
    categories = [d for d in images_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(categories)} categories:")
    
    for cat_dir in sorted(categories):
        # Count images in each category
        image_files = list(cat_dir.glob('*.jpg')) + list(cat_dir.glob('*.png'))
        print(f"  {cat_dir.name}: {len(image_files)} images")
        
        # Show sample image names
        if image_files:
            print(f"    Sample: {image_files[0].name}")


def create_data_mapping(data_root):
    """Create a mapping between trial data and images"""
    
    data_root = Path(data_root)
    
    # Load a sample of trials
    fixations_dir = data_root / 'fixations'
    json_files = list(fixations_dir.glob('*train*.json'))
    
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        # Try to find image references
        print("\n\nAttempting to map trials to images:")
        
        for i, trial in enumerate(data[:5]):
            print(f"\nTrial {i}:")
            
            # Print any field that might contain image info
            for key in trial.keys():
                if any(term in key.lower() for term in ['image', 'img', 'name', 'file', 'stim']):
                    print(f"  {key}: {trial[key]}")
            
            # Also check task/target
            if 'task' in trial:
                print(f"  task (target): {trial['task']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check COCO-Search18 data structure')
    parser.add_argument('--data_root', type=str, 
                       default='./data/coco_search18_tp',
                       help='Root directory of COCO-Search18 data')
    
    args = parser.parse_args()
    
    print("Checking COCO-Search18 Data Structure")
    print("=" * 60)
    
    check_coco_search_structure(args.data_root)
    check_image_directory(args.data_root)
    create_data_mapping(args.data_root)