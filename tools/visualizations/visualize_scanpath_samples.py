# Save as: tools/visualization/visualize_scanpath_samples.py

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def visualize_coco_samples(data_root='./data/coco_search18_tp', n_samples=10):
    """Visualize COCO-Search18 samples with backgrounds"""
    
    data_root = Path(data_root)
    
    # Load processed data
    sequences = np.load(data_root / 'processed' / 'short_train_sequences.npy')
    with open(data_root / 'processed' / 'short_train_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load original trial data to get image names
    fixations_file = data_root / 'fixations' / 'coco_search18_fixations_TP_train_split1.json'
    with open(fixations_file, 'r') as f:
        trials = json.load(f)
    
    # Create visualizations
    for i in range(min(n_samples, len(sequences))):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot raw scanpath
        coords = sequences[i]
        ax1.plot(coords[:, 0], coords[:, 1], 'b-', alpha=0.7, linewidth=2)
        for j, (x, y) in enumerate(coords):
            ax1.scatter(x, y, s=100, c='red', zorder=5)
            ax1.text(x, y, str(j+1), ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(1, 0)
        ax1.set_aspect('equal')
        ax1.set_title(f"Scanpath (normalized coordinates)")
        ax1.grid(True, alpha=0.3)
        
        # Try to find and display background image
        meta = metadata[i]
        target = meta.get('target_name', 'unknown')
        
        # Look for image in the target category folder
        if target != 'unknown':
            img_dir = data_root / 'images' / target
            if img_dir.exists():
                # Get first image from category (as example)
                img_files = list(img_dir.glob('*.jpg'))
                if img_files:
                    img = Image.open(img_files[0])
                    ax2.imshow(img)
                    
                    # Overlay scanpath on image
                    img_coords = coords.copy()
                    img_coords[:, 0] *= img.width
                    img_coords[:, 1] *= img.height
                    
                    ax2.plot(img_coords[:, 0], img_coords[:, 1], 
                            'r-', alpha=0.7, linewidth=3)
                    for j, (x, y) in enumerate(img_coords):
                        ax2.scatter(x, y, s=150, c='yellow', 
                                  edgecolor='red', linewidth=2, zorder=5)
                        ax2.text(x, y, str(j+1), ha='center', va='center',
                                fontsize=10, fontweight='bold', color='black')
                    
                    ax2.set_title(f"Target: {target} (example background)")
                else:
                    ax2.text(0.5, 0.5, f"No images found for {target}", 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title(f"Target: {target}")
            else:
                ax2.text(0.5, 0.5, f"Category folder not found: {target}", 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f"Target: {target}")
        else:
            ax2.text(0.5, 0.5, "Target unknown - check metadata", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Background not found")
        
        ax2.axis('off')
        
        plt.suptitle(f"Sample {i+1}: Sequence length = {len(coords)}", fontsize=14)
        plt.tight_layout()
        
        # Save
        output_dir = Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'coco_sample_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization {i+1}/{n_samples}")


if __name__ == '__main__':
    visualize_coco_samples(data_root='./data/coco_search18_tp', n_samples=10)
    print("Visualizations saved in 'visualizations' directory.")