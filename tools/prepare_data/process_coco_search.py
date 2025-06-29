#!/usr/bin/env python
"""
Process COCO-Search18 TP dataset for eye tracking prediction
Handles both coordinate sequences and background image associations
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm


class COCOSearchProcessor:
    """Process COCO-Search18 Target Present (TP) data"""
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.fixations_dir = self.data_root / 'fixations'
        self.images_dir = self.data_root / 'images'
        self.processed_dir = self.data_root / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
    def load_fixations(self, split='train'):
        """Load fixation data from JSON files"""
        fixations_data = []
        
        # Load all split files
        for split_num in [1, 2]:
            if split == 'train':
                filename = f'coco_search18_fixations_TP_train_split{split_num}.json'
            else:
                filename = f'coco_search18_fixations_TP_validation_split{split_num}.json'
            
            filepath = self.fixations_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    fixations_data.extend(data)
                print(f"Loaded {len(data)} trials from {filename}")
        
        return fixations_data
    
    def extract_sequences(self, fixations_data, min_length=10, max_length=20):
        """Extract coordinate sequences from fixation data"""
        sequences = []
        metadata = []
        
        for trial in tqdm(fixations_data, desc="Processing trials"):
            # Extract fixation coordinates
            X = trial.get('X', [])
            Y = trial.get('Y', [])
            
            if len(X) < min_length or len(X) != len(Y):
                continue
            
            # Normalize coordinates (assuming 1920x1080 resolution)
            img_width = 1920
            img_height = 1080
            
            coords = np.array([(x/img_width, y/img_height) for x, y in zip(X, Y)])
            
            # Clip to valid range
            coords = np.clip(coords, 0, 1)
            
            # Truncate or pad to max_length
            if len(coords) > max_length:
                coords = coords[:max_length]
            elif len(coords) < max_length:
                # Pad with last coordinate
                padding = np.tile(coords[-1], (max_length - len(coords), 1))
                coords = np.vstack([coords, padding])
            
            sequences.append(coords)
            
            # Store metadata - check various possible field names
            # COCO-Search18 might use different field names
            image_id = trial.get('image_id') or trial.get('name') or trial.get('imagename') or 'unknown'
            
            # Extract numeric image ID if it's in a path format
            if isinstance(image_id, str) and '/' in image_id:
                image_id = image_id.split('/')[-1].split('.')[0]
            
            meta = {
                'subject': trial.get('subject', 'unknown'),
                'target_name': trial.get('task') or trial.get('target_name') or trial.get('target_object') or 'unknown',
                'image_id': image_id,
                'target_found': trial.get('correct') if 'correct' in trial else trial.get('target_found', False),
                'original_length': len(X),
                'dataset': trial.get('dataset', 'coco_search18'),
                'condition': trial.get('condition', 'TP')  # Target Present
            }
            metadata.append(meta)
        
        return np.array(sequences), metadata
    
    def create_image_mapping(self):
        """Create mapping between image IDs and file paths"""
        image_mapping = {}
        
        for category_dir in self.images_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                for img_file in category_dir.glob('*.jpg'):
                    # Extract image ID from filename
                    img_id = img_file.stem  # Remove .jpg extension
                    image_mapping[img_id] = {
                        'path': str(img_file.relative_to(self.data_root)),
                        'category': category
                    }
        
        # Save mapping
        mapping_path = self.processed_dir / 'image_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(image_mapping, f, indent=2)
        
        print(f"Created image mapping with {len(image_mapping)} images")
        return image_mapping
    
    def process_dataset(self, config_name='standard'):
        """Process dataset with different configurations"""
        configs = {
            'standard': {'min_length': 10, 'max_length': 20},
            'short': {'min_length': 10, 'max_length': 10},
            'medium': {'min_length': 15, 'max_length': 30},
            'success_only': {'min_length': 10, 'max_length': 20, 'filter_success': True}
        }
        
        config = configs.get(config_name, configs['standard'])
        
        # Process train and validation sets
        for split in ['train', 'val']:
            print(f"\nProcessing {split} set with {config_name} config...")
            
            # Load fixations
            fixations = self.load_fixations('train' if split == 'train' else 'validation')
            
            # Filter for success only if specified
            if config.get('filter_success', False):
                fixations = [f for f in fixations if f.get('target_found', False)]
                print(f"Filtered to {len(fixations)} successful trials")
            
            # Extract sequences
            sequences, metadata = self.extract_sequences(
                fixations,
                min_length=config['min_length'],
                max_length=config['max_length']
            )
            
            # Save processed data
            prefix = f"{config_name}_" if config_name != 'standard' else ''
            
            np.save(self.processed_dir / f'{prefix}{split}_sequences.npy', sequences)
            
            with open(self.processed_dir / f'{prefix}{split}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved {len(sequences)} sequences for {split} set")
        
        # Save config
        with open(self.processed_dir / f'{config_name}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def compute_statistics(self):
        """Compute dataset statistics"""
        stats = {}
        
        for config in ['standard', 'short', 'medium', 'success_only']:
            config_stats = {}
            
            for split in ['train', 'val']:
                prefix = f"{config}_" if config != 'standard' else ''
                seq_file = self.processed_dir / f'{prefix}{split}_sequences.npy'
                
                if seq_file.exists():
                    sequences = np.load(seq_file)
                    
                    # Check if we have valid sequences
                    if len(sequences) > 0 and sequences.ndim == 3:
                        # Compute statistics
                        config_stats[split] = {
                            'num_sequences': len(sequences),
                            'sequence_shape': sequences.shape,
                            'mean_x': float(np.mean(sequences[:, :, 0])),
                            'mean_y': float(np.mean(sequences[:, :, 1])),
                            'std_x': float(np.std(sequences[:, :, 0])),
                            'std_y': float(np.std(sequences[:, :, 1]))
                        }
                    else:
                        config_stats[split] = {
                            'num_sequences': 0,
                            'sequence_shape': sequences.shape if len(sequences) > 0 else (0,),
                            'mean_x': 0.0,
                            'mean_y': 0.0,
                            'std_x': 0.0,
                            'std_y': 0.0
                        }
                        print(f"Warning: No valid sequences for {config} {split}")
            
            if config_stats:
                stats[config] = config_stats
        
        # Save statistics
        with open(self.processed_dir / 'data_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def create_compatibility_links(self):
        """Create symbolic links for OpenSTL compatibility"""
        # Use short sequences as default
        train_src = self.processed_dir / 'short_train_sequences.npy'
        val_src = self.processed_dir / 'short_val_sequences.npy'
        
        train_dst = self.processed_dir / 'coco_search_train.npy'
        val_dst = self.processed_dir / 'coco_search_val.npy'
        
        # Copy files for compatibility
        if train_src.exists():
            np.save(train_dst, np.load(train_src))
            print(f"Created {train_dst}")
        
        if val_src.exists():
            np.save(val_dst, np.load(val_src))
            print(f"Created {val_dst}")


def main():
    parser = argparse.ArgumentParser(description='Process COCO-Search18 data')
    parser.add_argument('--data_root', type=str, 
                       default='./data/coco_search18_tp',
                       help='Root directory of COCO-Search18 data')
    parser.add_argument('--configs', nargs='+', 
                       default=['standard', 'short', 'medium', 'success_only'],
                       help='Configurations to process')
    
    args = parser.parse_args()
    
    processor = COCOSearchProcessor(args.data_root)
    
    # Create image mapping
    print("Creating image mapping...")
    processor.create_image_mapping()
    
    # Process each configuration
    for config in args.configs:
        processor.process_dataset(config)
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = processor.compute_statistics()
    
    # Create compatibility links
    print("\nCreating compatibility links...")
    processor.create_compatibility_links()
    
    print("\nProcessing complete!")
    
    # Print summary
    print("\nDataset Statistics:")
    for config, config_stats in stats.items():
        print(f"\n{config.upper()}:")
        for split, split_stats in config_stats.items():
            print(f"  {split}: {split_stats['num_sequences']} sequences")


if __name__ == '__main__':
    main()