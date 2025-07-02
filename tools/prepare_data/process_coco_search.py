#!/usr/bin/env python
"""
Fixed COCO-Search18 processor with correct image-trial mapping
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm


class FixedCOCOSearchProcessor:
    """Fixed processor with proper image-trial mapping"""
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.fixations_dir = self.data_root / 'fixations'
        self.images_dir = self.data_root / 'images'
        self.processed_dir = self.data_root / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        # Create proper image mapping first
        self.image_id_to_path = self._create_image_mapping()
        
    def _create_image_mapping(self):
        """Create mapping from image IDs to file paths"""
        mapping = {}
        
        if not self.images_dir.exists():
            print(f"Warning: Images directory not found: {self.images_dir}")
            return mapping
            
        for category_dir in self.images_dir.iterdir():
            if category_dir.is_dir():
                for img_file in category_dir.glob('*.jpg'):
                    # Extract various possible ID formats
                    img_name = img_file.stem
                    
                    # Store multiple formats
                    mapping[img_name] = img_file
                    mapping[img_name.lower()] = img_file
                    
                    # Try with category prefix
                    full_name = f"{category_dir.name}_{img_name}"
                    mapping[full_name] = img_file
                    mapping[full_name.lower()] = img_file
                    
                    # Try COCO format variations
                    if img_name.isdigit():
                        # Pad with zeros (common COCO format)
                        for pad_len in [6, 12]:
                            padded = img_name.zfill(pad_len)
                            mapping[padded] = img_file
                            mapping[f"COCO_train2014_{padded}"] = img_file
                            mapping[f"COCO_val2014_{padded}"] = img_file
        
        print(f"Created image mapping with {len(mapping)} entries")
        return mapping
    
    def _extract_image_id_from_trial(self, trial):
        """Extract image ID from trial data using multiple strategies"""
        
        # Strategy 1: Direct image ID fields
        for field in ['image_id', 'img_id', 'imgid', 'imageid']:
            if field in trial and trial[field]:
                return str(trial[field])
        
        # Strategy 2: Name/filename fields
        for field in ['name', 'imagename', 'img_name', 'filename', 'stimulus']:
            if field in trial and trial[field]:
                value = str(trial[field])
                # Extract filename from path
                if '/' in value:
                    value = value.split('/')[-1]
                if '.' in value:
                    value = value.split('.')[0]
                return value
        
        # Strategy 3: Task field might contain image info
        if 'task' in trial and isinstance(trial['task'], str):
            task = trial['task']
            # Check if task contains image ID
            if any(c.isdigit() for c in task):
                return task
        
        # Strategy 4: Check for any numeric fields
        for key, value in trial.items():
            if isinstance(value, (int, str)) and str(value).isdigit():
                if len(str(value)) >= 4:  # Reasonable image ID length
                    return str(value)
        
        return None
    
    def _find_image_path(self, image_id):
        """Find actual image path for given ID"""
        if not image_id:
            return None
            
        # Try exact match first
        if image_id in self.image_id_to_path:
            return self.image_id_to_path[image_id]
        
        # Try lowercase
        if image_id.lower() in self.image_id_to_path:
            return self.image_id_to_path[image_id.lower()]
        
        # Try partial matches
        for mapped_id, path in self.image_id_to_path.items():
            if image_id in mapped_id or mapped_id in image_id:
                return path
        
        return None
    
    def load_and_validate_fixations(self, split='train'):
        """Load fixations and validate image mapping"""
        fixations_data = []
        
        # Load split files
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
        
        # Validate image mapping
        mapped_count = 0
        for trial in fixations_data[:100]:  # Check first 100
            image_id = self._extract_image_id_from_trial(trial)
            image_path = self._find_image_path(image_id)
            if image_path and image_path.exists():
                mapped_count += 1
        
        mapping_rate = mapped_count / min(100, len(fixations_data))
        print(f"Image mapping validation: {mapping_rate:.1%} success rate")
        
        if mapping_rate < 0.3:
            print("WARNING: Low image mapping rate. Checking trial structure...")
            self._debug_trial_structure(fixations_data[:5])
        
        return fixations_data
    
    def _debug_trial_structure(self, sample_trials):
        """Debug trial structure to find correct image ID field"""
        print("\nDEBUGGING TRIAL STRUCTURE:")
        print("=" * 50)
        
        for i, trial in enumerate(sample_trials):
            print(f"\nTrial {i}:")
            for key, value in trial.items():
                if any(term in key.lower() for term in ['image', 'img', 'name', 'file', 'id', 'stim']):
                    print(f"  {key}: {value}")
            
            # Show all string/numeric fields that might be image IDs
            potential_ids = []
            for key, value in trial.items():
                if isinstance(value, (str, int)):
                    val_str = str(value)
                    if len(val_str) >= 4 and (val_str.isdigit() or any(c.isdigit() for c in val_str)):
                        potential_ids.append(f"{key}: {value}")
            
            if potential_ids:
                print("  Potential image IDs:")
                for pid in potential_ids:
                    print(f"    {pid}")
    
    def extract_sequences_with_validation(self, fixations_data, min_length=5, max_length=10):
        """Extract sequences with image validation"""
        sequences = []
        metadata = []
        valid_mapping_count = 0
        
        # Get display resolution from first trial if available
        display_width = 1920
        display_height = 1080
        
        if fixations_data and 'screen_width' in fixations_data[0]:
            display_width = fixations_data[0]['screen_width']
            display_height = fixations_data[0]['screen_height']
            print(f"Using display resolution: {display_width}x{display_height}")
        
        for trial in tqdm(fixations_data, desc="Processing trials with validation"):
            # Extract coordinates
            X = trial.get('X', [])
            Y = trial.get('Y', [])
            
            if len(X) < min_length or len(X) != len(Y):
                continue
            
            # Find corresponding image
            image_id = self._extract_image_id_from_trial(trial)
            image_path = self._find_image_path(image_id)
            
            # Skip trials without valid images for now
            if not image_path or not image_path.exists():
                continue
            
            valid_mapping_count += 1
            
            # Normalize coordinates
            coords = np.array([(x/display_width, y/display_height) for x, y in zip(X, Y)])
            coords = np.clip(coords, 0, 1)
            
            # Adjust length
            if len(coords) > max_length:
                coords = coords[:max_length]
            elif len(coords) < max_length:
                # Repeat last coordinate
                padding = np.tile(coords[-1], (max_length - len(coords), 1))
                coords = np.vstack([coords, padding])
            
            sequences.append(coords)
            
            # Store validated metadata
            meta = {
                'subject': trial.get('subject', 'unknown'),
                'target_name': trial.get('task', trial.get('target_name', 'unknown')),
                'image_id': image_id,
                'image_path': str(image_path.relative_to(self.data_root)),
                'target_found': trial.get('correct', trial.get('target_found', False)),
                'original_length': len(X),
                'display_resolution': f"{display_width}x{display_height}"
            }
            metadata.append(meta)
        
        print(f"Successfully mapped {valid_mapping_count} trials to images")
        print(f"Processed {len(sequences)} valid sequences")
        
        return np.array(sequences), metadata
    
    def process_with_validation(self, config_name='short'):
        """Process dataset with image validation"""
        configs = {
            'short': {'min_length': 6, 'max_length': 10},
            'standard': {'min_length': 10, 'max_length': 15},
            'medium': {'min_length': 15, 'max_length': 40}
        }
        
        config = configs.get(config_name, configs['short'])
        
        for split in ['train', 'val']:
            print(f"\nProcessing {split} set with {config_name} config...")
            
            # Load and validate fixations
            fixations = self.load_and_validate_fixations('train' if split == 'train' else 'validation')
            
            # Extract sequences with validation
            sequences, metadata = self.extract_sequences_with_validation(
                fixations,
                min_length=config['min_length'],
                max_length=config['max_length']
            )
            
            if len(sequences) == 0:
                print(f"ERROR: No valid sequences found for {split}")
                continue
            
            # Save processed data
            prefix = f"{config_name}_" if config_name != 'standard' else ''
            
            np.save(self.processed_dir / f'{prefix}{split}_sequences.npy', sequences)
            
            with open(self.processed_dir / f'{prefix}{split}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved {len(sequences)} validated sequences for {split} set")
            
            # Create validation plot
            self._create_validation_plot(sequences[:6], metadata[:6], f"{config_name}_{split}")
    
    def _create_validation_plot(self, sequences, metadata, name):
        """Create validation plot with correct image backgrounds"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (seq, meta) in enumerate(zip(sequences, metadata)):
            if i >= 6:
                break
                
            ax = axes[i]
            
            # Load and display background image
            image_path = self.data_root / meta['image_path']
            if image_path.exists():
                img = Image.open(image_path)
                ax.imshow(img)
                
                # Convert coordinates to image space
                img_w, img_h = img.size
                x_coords = seq[:, 0] * img_w
                y_coords = seq[:, 1] * img_h
                
                # Plot scanpath
                ax.plot(x_coords, y_coords, 'r-', linewidth=3, alpha=0.9)
                ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.8, 
                          edgecolors='white', linewidth=2)
                ax.scatter(x_coords[0], y_coords[0], c='green', s=80, 
                          marker='s', label='Start')
                ax.scatter(x_coords[-1], y_coords[-1], c='blue', s=80, 
                          marker='*', label='End')
                
                title = f"VALIDATED - Target: {meta['target_name']}"
                ax.set_title(title, color='green', fontweight='bold', fontsize=10)
            else:
                ax.text(0.5, 0.5, f"Image not found:\n{meta['image_path']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Missing image", color='red')
            
            ax.axis('off')
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        output_path = self.processed_dir / f'validated_{name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Validation plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fixed COCO-Search18 processor')
    parser.add_argument('--data_root', type=str, 
                       default='./data/coco_search18_tp',
                       help='Root directory of COCO-Search18 data')
    parser.add_argument('--configs', nargs='+', 
                       default=['short'],
                       help='Configurations to process')
    
    args = parser.parse_args()
    
    processor = FixedCOCOSearchProcessor(args.data_root)
    
    # Process each configuration
    for config in args.configs:
        print(f"\n{'='*60}")
        print(f"PROCESSING {config.upper()} CONFIG")
        print(f"{'='*60}")
        processor.process_with_validation(config)
    
    print(f"\nFixed processing complete!")


if __name__ == '__main__':
    main()