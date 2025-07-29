#!/usr/bin/env python3
"""
Generate Gaussian Enhanced simulation data.
Step 1 Refactoring: Use single_fixation_experiment coordinates to generate Gaussian heatmaps.
This removes edge bias by using proven coordinate generation.
"""

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import config


def generate_gaussian_blob(center, img_size, sigma, normalize=True):
    """
    Generate a 2D Gaussian blob centered at the given position.
    """
    y_center, x_center = center
    
    # Create coordinate grids
    y = torch.arange(img_size, dtype=torch.float32)
    x = torch.arange(img_size, dtype=torch.float32)
    yv, xv = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate Gaussian
    gaussian = torch.exp(-((yv - y_center)**2 + (xv - x_center)**2) / (2 * sigma**2))
    
    if normalize:
        gaussian = gaussian / gaussian.max() if gaussian.max() > 0 else gaussian
    
    return gaussian


def render_gaussian_frame_with_history(coords_sequence, current_t, img_size, use_history_trails):
    """
    Render a frame with Gaussian blob and optional history trail.
    """
    frame = torch.zeros(img_size, img_size)
    
    if use_history_trails and current_t > 0:
        # Add history trail with exponential decay
        history_start = max(0, current_t - config.history_length)
        
        for t in range(history_start, current_t):
            history_age = current_t - t  # How many steps back
            intensity = config.history_decay_gamma ** history_age
            
            # Only add if intensity is above threshold
            if intensity >= config.history_min_intensity:
                pos = coords_sequence[t]
                y, x = pos[0].item(), pos[1].item()
                
                # Clamp coordinates
                y = max(0, min(img_size - 1, y))
                x = max(0, min(img_size - 1, x))
                
                # Generate Gaussian blob with reduced intensity
                blob = generate_gaussian_blob(
                    (y, x), img_size, 
                    config.gaussian_sigma, 
                    config.gaussian_normalize
                )
                frame = torch.maximum(frame, blob * intensity)
    
    # Add current fixation with full intensity
    current_pos = coords_sequence[current_t]
    y, x = current_pos[0].item(), current_pos[1].item()
    
    # Clamp coordinates
    y = max(0, min(img_size - 1, y))
    x = max(0, min(img_size - 1, x))
    
    # Generate current Gaussian blob
    current_blob = generate_gaussian_blob(
        (y, x), img_size,
        config.gaussian_sigma,
        config.gaussian_normalize
    )
    frame = torch.maximum(frame, current_blob)
    
    return frame


def convert_dataset(input_path, output_path, use_history_trails):
    """Convert single fixation data to Gaussian enhanced format"""
    print(f"Loading data from: {input_path}")
    
    # Load original single fixation data
    original_data = torch.load(input_path)
    coords = original_data['coordinates']  # [B, T, 2]
    
    batch_size, T, _ = coords.shape
    img_size = config.img_size
    
    print(f"Converting {batch_size} sequences of length {T}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"History trails: {use_history_trails}")
    
    # Generate Gaussian frames
    frames = torch.zeros(batch_size, T, img_size, img_size)
    
    for b in tqdm(range(batch_size), desc="Converting sequences"):
        for t in range(T):
            frame = render_gaussian_frame_with_history(
                coords[b], t, img_size, use_history_trails
            )
            frames[b, t] = frame
    
    # Reshape frames to (B, C, T, H, W) format for SimVP
    frames = frames.unsqueeze(1)  # Add channel dimension
    
    # Create output data
    output_data = {
        'frames': frames,
        'coords': coords,  # Keep as 'coords' for our format
        'fixation_mask': original_data['fixation_mask']
    }
    
    # Save converted data
    torch.save(output_data, output_path)
    print(f"Saved converted data to: {output_path}")
    
    # Verify data
    print(f"Output shapes:")
    print(f"  Frames: {frames.shape}")
    print(f"  Coords: {coords.shape}")
    print(f"  Frame intensity - mean: {frames.mean():.3f}, max: {frames.max():.3f}")
    
    return output_data


def main():
    """Generate all datasets with history trails ablation support"""
    parser = argparse.ArgumentParser(description="Generate Gaussian Enhanced data from single_fixation coordinates")
    parser.add_argument('--no_history_trails', action='store_true', 
                       help='Generate data without history trails')
    args = parser.parse_args()
    
    use_history_trails = not args.no_history_trails
    history_suffix = "_no_history" if args.no_history_trails else "_with_history"
    
    print("="*60)
    print("STEP 1: GENERATING GAUSSIAN ENHANCED DATA (NO EDGE BIAS)")
    print("="*60)
    print(f"History trails: {use_history_trails}")
    print(f"Gaussian sigma: {config.gaussian_sigma}")
    print(" Using single_fixation_experiment coordinates (removes edge bias)")
    
    # Create output directory
    os.makedirs(config.data_dir, exist_ok=True)
    
    # Convert each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\nGenerating {split} data...")
        
        # Input path (from single_fixation_experiment)
        input_path = f"../single_fixation_experiment/data/{split}_data.pt"
        
        # Output path with history suffix
        output_filename = f"{split}_data{history_suffix}.pt"
        output_path = os.path.join(config.data_dir, output_filename)
        
        # Convert
        convert_dataset(input_path, output_path, use_history_trails)
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETED: DATA GENERATION")
    print("="*60)
    print(" Removed edge bias by using single_fixation coordinates")
    print(" Generated Gaussian heatmaps with proper parameters")
    print(" Maintained uniform spatial distribution")
    print(f" Created {'with' if use_history_trails else 'without'} history trails")


if __name__ == "__main__":
    main() 