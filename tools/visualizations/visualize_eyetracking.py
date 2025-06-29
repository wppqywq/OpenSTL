#!/usr/bin/env python
"""
Visualize COCO-Search18 scanpath samples with background images
This script creates visualizations to debug the data processing pipeline
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def visualize_coco_samples(data_root, n_samples=5, output_dir='visualizations'):
    """Visualize COCO-Search18 samples with backgrounds"""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check if processed data exists
    sequences_file = data_root / 'processed' / 'short_train_sequences.npy'
    metadata_file = data_root / 'processed' / 'short_train_metadata.json'
    
    if not sequences_file.exists():
        print(f"Error: Processed sequences not found at {sequences_file}")
        print("Please run process_coco_search.py first")
        return
    
    # Load processed data
    sequences = np.load(sequences_file)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    
    # Load original trial data to get more info if needed
    fixations_file = data_root / 'fixations' / 'coco_search18_fixations_TP_train_split1.json'
    if fixations_file.exists():
        with open(fixations_file, 'r') as f:
            trials = json.load(f)
        print(f"Loaded {len(trials)} original trials for reference")
    else:
        trials = None
    
    # Create visualizations
    for i in range(min(n_samples, len(sequences))):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Get data for this sample
        coords = sequences[i]
        meta = metadata[i]
        
        # Plot 1: Raw scanpath in normalized coordinates
        ax1 = axes[0]
        ax1.plot(coords[:, 0], coords[:, 1], 'b-', alpha=0.7, linewidth=2)
        for j, (x, y) in enumerate(coords):
            color = 'green' if j == 0 else 'red' if j == len(coords)-1 else 'blue'
            ax1.scatter(x, y, s=100, c=color, zorder=5, edgecolor='black', linewidth=1)
            ax1.text(x+0.01, y+0.01, str(j+1), fontsize=8, fontweight='bold')
        
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(1.05, -0.05)
        ax1.set_aspect('equal')
        ax1.set_title(f"Normalized Scanpath (n={len(coords)} fixations)")
        ax1.set_xlabel("X (normalized)")
        ax1.set_ylabel("Y (normalized)")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Heatmap representation
        ax2 = axes[1]
        # Create a simple heatmap by accumulating fixations
        heatmap = np.zeros((32, 32))
        for x, y in coords:
            px = int(x * 31)
            py = int(y * 31)
            px = np.clip(px, 0, 31)
            py = np.clip(py, 0, 31)
            # Add Gaussian-like spread
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if 0 <= px+dx < 32 and 0 <= py+dy < 32:
                        dist = np.sqrt(dx**2 + dy**2)
                        heatmap[py+dy, px+dx] += np.exp(-dist**2 / 4)
        
        im = ax2.imshow(heatmap, cmap='hot', origin='upper')
        ax2.set_title("Heatmap Representation (32x32)")
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        # Plot 3: Background image with scanpath overlay (if available)
        ax3 = axes[2]
        target = meta.get('target_name', 'unknown')
        image_id = meta.get('image_id', 'unknown')
        found_image = False
        
        # Try to find corresponding image
        if target != 'unknown':
            img_dir = data_root / 'images' / target
            if img_dir.exists():
                # Look for specific image ID or use first available
                img_files = list(img_dir.glob('*.jpg'))
                
                # Try to match image ID
                if image_id != 'unknown':
                    matching_files = [f for f in img_files if image_id in f.stem]
                    if matching_files:
                        img_file = matching_files[0]
                    else:
                        img_file = img_files[0] if img_files else None
                else:
                    img_file = img_files[0] if img_files else None
                
                if img_file:
                    img = Image.open(img_file)
                    ax3.imshow(img)
                    
                    # Overlay scanpath
                    img_coords = coords.copy()
                    img_coords[:, 0] *= img.width
                    img_coords[:, 1] *= img.height
                    
                    ax3.plot(img_coords[:, 0], img_coords[:, 1], 
                            'r-', alpha=0.7, linewidth=3)
                    
                    for j, (x, y) in enumerate(img_coords):
                        color = 'lime' if j == 0 else 'red' if j == len(img_coords)-1 else 'yellow'
                        ax3.scatter(x, y, s=150, c=color, 
                                  edgecolor='black', linewidth=2, zorder=5)
                        ax3.text(x+10, y-10, str(j+1), fontsize=10, 
                                fontweight='bold', color='white',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                    
                    ax3.set_title(f"Target: {target}\nImage: {img_file.name}")
                    found_image = True
        
        if not found_image:
            # No image found, show metadata info
            info_text = f"Target: {target}\n"
            info_text += f"Image ID: {image_id}\n"
            info_text += f"Subject: {meta.get('subject', 'unknown')}\n"
            info_text += f"Found: {meta.get('target_found', 'unknown')}\n"
            info_text += f"Original length: {meta.get('original_length', 'unknown')}"
            
            ax3.text(0.5, 0.5, info_text, ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray'))
            ax3.set_title("Background Image Not Found")
        
        ax3.axis('off')
        
        # Overall title
        plt.suptitle(f"Sample {i+1}/{n_samples} - Metadata: {json.dumps(meta, indent=2)[:100]}...", 
                    fontsize=12)
        plt.tight_layout()
        
        # Save
        save_path = output_dir / f'coco_scanpath_sample_{i:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
    
    print(f"\nCreated {min(n_samples, len(sequences))} visualizations in {output_dir}/")
    
    # Create a summary plot
    create_summary_plot(sequences, metadata, output_dir)


def create_summary_plot(sequences, metadata, output_dir):
    """Create a summary plot of the dataset"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sequence length distribution
    ax = axes[0, 0]
    seq_lengths = [meta.get('original_length', 10) for meta in metadata]
    ax.hist(seq_lengths, bins=30, edgecolor='black')
    ax.set_xlabel('Original Sequence Length')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Scanpath Lengths')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Target distribution
    ax = axes[0, 1]
    targets = [meta.get('target_name', 'unknown') for meta in metadata]
    target_counts = {}
    for t in targets:
        target_counts[t] = target_counts.get(t, 0) + 1
    
    # Sort by count and take top 20
    sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    labels, counts = zip(*sorted_targets)
    
    ax.barh(range(len(labels)), counts)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Count')
    ax.set_title('Top 20 Target Objects')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Average scanpath
    ax = axes[1, 0]
    # Plot all scanpaths with low alpha
    for seq in sequences[:100]:  # First 100 to avoid overcrowding
        ax.plot(seq[:, 0], seq[:, 1], 'b-', alpha=0.05)
    
    # Plot average positions
    avg_positions = np.mean(sequences, axis=0)
    ax.plot(avg_positions[:, 0], avg_positions[:, 1], 'r-', linewidth=3, label='Average')
    for i, (x, y) in enumerate(avg_positions):
        ax.scatter(x, y, s=100, c='red', zorder=5)
        ax.text(x+0.01, y+0.01, str(i+1), fontsize=8, fontweight='bold')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_aspect('equal')
    ax.set_title('Scanpath Overlay (first 100 sequences)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Success rate
    ax = axes[1, 1]
    success_rates = {}
    for meta in metadata:
        target = meta.get('target_name', 'unknown')
        found = meta.get('target_found', False)
        if target not in success_rates:
            success_rates[target] = {'found': 0, 'total': 0}
        success_rates[target]['total'] += 1
        if found:
            success_rates[target]['found'] += 1
    
    # Calculate rates and sort
    target_success = []
    for target, stats in success_rates.items():
        if stats['total'] >= 10:  # Only targets with enough samples
            rate = stats['found'] / stats['total']
            target_success.append((target, rate, stats['total']))
    
    target_success.sort(key=lambda x: x[1], reverse=True)
    
    if target_success:
        labels, rates, totals = zip(*target_success[:15])
        x = range(len(labels))
        bars = ax.bar(x, rates)
        
        # Color bars by success rate
        for bar, rate in zip(bars, rates):
            bar.set_color('green' if rate > 0.7 else 'orange' if rate > 0.4 else 'red')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Success Rate')
        ax.set_title('Target Finding Success Rate (n >= 10)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size annotations
        for i, (rate, total) in enumerate(zip(rates, totals)):
            ax.text(i, rate + 0.02, f'n={total}', ha='center', fontsize=8)
    
    plt.suptitle(f'COCO-Search18 Dataset Summary (n={len(sequences)} sequences)', fontsize=14)
    plt.tight_layout()
    
    save_path = output_dir / 'dataset_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dataset summary: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize COCO-Search18 samples')
    parser.add_argument('--data_root', type=str, 
                       default='./data/coco_search18_tp',
                       help='Root directory of COCO-Search18 data')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_coco_samples(args.data_root, args.n_samples, args.output_dir)