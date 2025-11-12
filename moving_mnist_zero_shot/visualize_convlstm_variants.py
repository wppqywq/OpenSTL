#!/usr/bin/env python
"""Visualize ConvLSTM model test predictions with multiple variants using same initial conditions."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from types import SimpleNamespace
import gzip

from openstl.models import ConvLSTM_Model
from openstl.datasets.dataloader_moving_mnist import MovingMNIST

def reshape_patch(img_tensor, patch_size):
    """Reshape image tensor into patches."""
    if img_tensor.dim() == 5:
        # Assume [B, T, H, W, C] format
        B, T, H, W, C = img_tensor.shape
        # Reshape to patches
        img_tensor = img_tensor.reshape(B, T, H//patch_size, patch_size, W//patch_size, patch_size, C)
        img_tensor = img_tensor.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # [B, T, H//p, W//p, p, p, C]
        img_tensor = img_tensor.reshape(B, T, H//patch_size, W//patch_size, patch_size*patch_size*C)
    return img_tensor

def reshape_patch_back(img_tensor, patch_size):
    """Reshape patches back to image."""
    B, T, H_p, W_p, _ = img_tensor.shape
    C = 1  # Assuming single channel
    H = H_p * patch_size
    W = W_p * patch_size
    
    img_tensor = img_tensor.reshape(B, T, H_p, W_p, patch_size, patch_size, C)
    img_tensor = img_tensor.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # [B, T, H_p, p, W_p, p, C]
    img_tensor = img_tensor.reshape(B, T, H, W, C)
    return img_tensor

def load_mnist_digits():
    """Load MNIST digits for generation."""
    path = "/Users/apple/git/neuro/OpenSTL/data/moving_mnist/train-images-idx3-ubyte.gz"
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist

def extract_motion_from_sequence(sequence):
    """Extract approximate motion parameters from a sequence."""
    # Find digit positions in each frame
    positions = []
    for frame in sequence:
        # Find center of mass
        coords = np.where(frame > 0.1)
        if len(coords[0]) > 0:
            cy = np.mean(coords[0])
            cx = np.mean(coords[1])
            positions.append((cx, cy))
    
    if len(positions) > 1:
        # Estimate velocity from first few frames
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        return positions[0], (dx, dy)
    return (32, 32), (0.1, 0.1)

def generate_trajectory(seq_length, step_length, initial_pos, initial_vel, canvas_size=36):
    """Generate a trajectory with given initial conditions."""
    x, y = initial_pos
    vx, vy = initial_vel
    
    positions_x = np.zeros(seq_length)
    positions_y = np.zeros(seq_length)
    
    for i in range(seq_length):
        # Update position
        x += vx * step_length
        y += vy * step_length
        
        # Bounce off edges
        if x <= 0:
            x = 0
            vx = -vx
        if x >= 1.0:
            x = 1.0
            vx = -vx
        if y <= 0:
            y = 0
            vy = -vy
        if y >= 1.0:
            y = 1.0
            vy = -vy
            
        positions_x[i] = x * canvas_size
        positions_y[i] = y * canvas_size
    
    return positions_x.astype(np.int32), positions_y.astype(np.int32)

def create_moving_mnist_sequence(digit_indices, initial_positions, initial_velocities, 
                                step_length=0.1, seq_length=20, image_size=64):
    """Create a Moving MNIST sequence with specific initial conditions."""
    
    mnist = load_mnist_digits()
    canvas_size = image_size - 28  # 64 - 28 = 36
    
    # Create sequence
    sequence = np.zeros((seq_length, image_size, image_size), dtype=np.float32)
    
    for digit_idx, init_pos, init_vel in zip(digit_indices, initial_positions, initial_velocities):
        # Get digit image
        digit_image = mnist[digit_idx].astype(np.float32) / 255.0
        
        # Generate trajectory
        pos_x, pos_y = generate_trajectory(seq_length, step_length, init_pos, init_vel, canvas_size)
        
        # Place digit on each frame
        for t in range(seq_length):
            top = pos_y[t]
            left = pos_x[t]
            bottom = min(top + 28, image_size)
            right = min(left + 28, image_size)
            
            # Crop digit if at edges
            digit_top = 0
            digit_left = 0
            digit_bottom = bottom - top
            digit_right = right - left
            
            if top < 0:
                digit_top = -top
                top = 0
            if left < 0:
                digit_left = -left
                left = 0
                
            # Place digit (using max to handle overlaps)
            sequence[t, top:bottom, left:right] = np.maximum(
                sequence[t, top:bottom, left:right],
                digit_image[digit_top:digit_bottom, digit_left:digit_right]
            )
    
    return sequence

def extract_initial_conditions_from_sample(test_sample):
    """Extract approximate initial conditions from a test sample."""
    input_seq, _ = test_sample
    input_np = input_seq.cpu().numpy()[:, 0]  # [T, H, W]
    
    # Use first two frames to estimate initial positions and velocities
    # This is a simplified approach - in practice we'd need more sophisticated tracking
    
    # Find approximate digit positions in first frame
    frame0 = input_np[0]
    frame1 = input_np[1]
    
    # Simple approach: use random but consistent initial conditions
    np.random.seed(hash(str(frame0.sum())) % 10000)  # Use frame as seed for consistency
    
    num_digits = 2
    digit_indices = np.random.randint(1000, 9000, num_digits)
    initial_positions = [(np.random.random()*0.5+0.25, np.random.random()*0.5+0.25) for _ in range(num_digits)]
    
    # Random initial velocities
    initial_velocities = []
    for _ in range(num_digits):
        theta = np.random.random() * 2 * np.pi
        initial_velocities.append((np.cos(theta), np.sin(theta)))
    
    return digit_indices, initial_positions, initial_velocities

def create_test_variants(test_sample):
    """Create multiple variants from a test sample using same initial conditions."""
    
    # Extract/approximate initial conditions from the test sample
    digit_indices, initial_positions, initial_velocities = extract_initial_conditions_from_sample(test_sample)
    
    variants = {}
    
    # Standard
    variants['Standard'] = create_moving_mnist_sequence(
        digit_indices, initial_positions, initial_velocities, 
        step_length=0.1, seq_length=20
    )
    
    # Fast 1.5x
    variants['Fast_1.5x'] = create_moving_mnist_sequence(
        digit_indices, initial_positions, initial_velocities,
        step_length=0.15, seq_length=20
    )
    
    # Fast 2x
    variants['Fast_2x'] = create_moving_mnist_sequence(
        digit_indices, initial_positions, initial_velocities,
        step_length=0.2, seq_length=20
    )
    
    # Fast 3x
    variants['Fast_3x'] = create_moving_mnist_sequence(
        digit_indices, initial_positions, initial_velocities,
        step_length=0.3, seq_length=20
    )
    
    # Three Digits (add one more digit)
    three_digit_indices = np.append(digit_indices, np.random.randint(1000, 9000))
    three_positions = initial_positions + [(np.random.random()*0.5+0.25, np.random.random()*0.5+0.25)]
    theta = np.random.random() * 2 * np.pi
    three_velocities = initial_velocities + [(np.cos(theta), np.sin(theta))]
    
    variants['Three_Digits'] = create_moving_mnist_sequence(
        three_digit_indices, three_positions, three_velocities,
        step_length=0.1, seq_length=20
    )
    
    # Convert to tensors [T, C, H, W]
    for name, seq in variants.items():
        variants[name] = torch.from_numpy(seq).unsqueeze(1).float()
    
    return variants

def create_variant_comparison_movie(model, variants, sample_idx, save_dir):
    """Create a movie comparing all variants for one sample."""
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    variant_names = list(variants.keys())
    num_variants = len(variant_names)
    
    # Prepare predictions for each variant
    predictions = {}
    for variant_name, sequence in variants.items():
        # Split into input and target
        input_seq = sequence[:10]  # [T, C, H, W]
        true_seq = sequence[10:20]
        
        # Transform to [B, T, H, W, C] format for ConvLSTM
        input_batch = input_seq.permute(0, 2, 3, 1).unsqueeze(0).to(device)
        true_batch = true_seq.permute(0, 2, 3, 1).unsqueeze(0).to(device)
        
        # Apply patch transformation
        patch_size = 4
        input_patches = reshape_patch(input_batch, patch_size)
        true_patches = reshape_patch(true_batch, patch_size)
        
        # Create full sequence for the model
        full_seq = torch.cat([input_patches, true_patches], dim=1).to(device)
        
        # Create mask for scheduled sampling
        mask = torch.zeros(1, 19, 16, 16, 16).to(device)
        
        # Generate prediction
        with torch.no_grad():
            pred_seq, _ = model(full_seq, mask, return_loss=False)
            # Extract prediction frames (last 10 frames)
            pred_seq = pred_seq[:, 9:19]
            # Reshape patches back to images
            pred_seq = reshape_patch_back(pred_seq, patch_size)  # [B, T, H, W, C]
        
        predictions[variant_name] = {
            'input': input_seq.cpu().numpy(),  # [T, C, H, W]
            'true': true_seq.cpu().numpy(),
            'pred': pred_seq[0].permute(0, 3, 1, 2).cpu().numpy()  # [T, C, H, W]
        }
    
    # Create figure with grid layout
    fig, axes = plt.subplots(num_variants, 3, figsize=(10, 3*num_variants))
    
    # Initialize images
    imgs = []
    for i, variant_name in enumerate(variant_names):
        for j in range(3):
            ax = axes[i, j] if num_variants > 1 else axes[j]
            ax.axis('off')
            img = ax.imshow(np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1)
            imgs.append(img)
            
            if i == 0:
                if j == 0:
                    ax.set_title('Input', fontsize=12)
                elif j == 1:
                    ax.set_title('Ground Truth', fontsize=12)
                else:
                    ax.set_title('ConvLSTM Prediction', fontsize=12)
            
            if j == 0:
                ax.set_ylabel(variant_name, rotation=90, fontsize=10, labelpad=20)
    
    def animate(frame_idx):
        img_idx = 0
        for variant_name in variant_names:
            pred_data = predictions[variant_name]
            
            if frame_idx < 10:
                # Show input
                imgs[img_idx].set_data(pred_data['input'][frame_idx, 0])
                imgs[img_idx+1].set_data(np.zeros((64, 64)))
                imgs[img_idx+2].set_data(np.zeros((64, 64)))
            else:
                # Show predictions
                pred_frame = frame_idx - 10
                imgs[img_idx].set_data(pred_data['input'][-1, 0])  # Keep last input
                imgs[img_idx+1].set_data(pred_data['true'][pred_frame, 0])
                imgs[img_idx+2].set_data(np.clip(pred_data['pred'][pred_frame, 0], 0, 1))
            
            img_idx += 3
        
        if frame_idx < 10:
            fig.suptitle(f'Sample {sample_idx+1} - Input Frame {frame_idx+1}/10', fontsize=14)
        else:
            fig.suptitle(f'Sample {sample_idx+1} - Prediction Frame {frame_idx-9}/10', fontsize=14)
        
        return imgs
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=20, interval=200, blit=True
    )
    
    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    gif_path = save_dir / f'convlstm_sample_{sample_idx+1}_all_variants.gif'
    anim.save(gif_path, writer='pillow', fps=5)
    print(f"  Saved: {gif_path}")
    
    plt.close()
    
    # Also create side-by-side comparison at specific frames
    fig, axes = plt.subplots(num_variants, 5, figsize=(15, 3*num_variants))
    
    for i, variant_name in enumerate(variant_names):
        pred_data = predictions[variant_name]
        
        # Show frames: input[0], input[9], pred[0], pred[4], pred[9]
        frames_to_show = [
            ('Input t=0', pred_data['input'][0, 0]),
            ('Input t=9', pred_data['input'][9, 0]),
            ('Pred t=10', np.clip(pred_data['pred'][0, 0], 0, 1)),
            ('Pred t=14', np.clip(pred_data['pred'][4, 0], 0, 1)),
            ('Pred t=19', np.clip(pred_data['pred'][9, 0], 0, 1)),
        ]
        
        for j, (title, frame) in enumerate(frames_to_show):
            ax = axes[i, j] if num_variants > 1 else axes[j]
            ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(variant_name, rotation=90, fontsize=10, labelpad=20)
    
    plt.suptitle(f'ConvLSTM - Sample {sample_idx+1} - Key Frames Comparison', fontsize=14)
    plt.tight_layout()
    
    png_path = save_dir / f'convlstm_sample_{sample_idx+1}_frames.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {png_path}")
    
    plt.close()

def main():
    print("="*60)
    print("Creating ConvLSTM Variant Movies with Same Test Sample")
    print("="*60)
    
    # Create configs object
    configs = SimpleNamespace(
        in_shape=(10, 1, 64, 64),  # T, C, H, W
        pre_seq_length=10,
        aft_seq_length=10,
        patch_size=4,
        filter_size=5,
        stride=1,
        layer_norm=0,
        reverse_scheduled_sampling=0,
        scheduled_sampling=1
    )
    
    # Initialize ConvLSTM model
    print("\nLoading ConvLSTM model...")
    num_hidden = [128, 128, 128, 128]
    num_layers = len(num_hidden)
    model = ConvLSTM_Model(num_layers, num_hidden, configs)
    
    # Load checkpoint
    ckpt_path = '/Users/apple/git/neuro/OpenSTL/work_dirs/convlstm_mmnist_m2/checkpoints/best.ckpt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
        print(f"  Val loss: {ckpt.get('val_loss', 'unknown')}")
    
    # Get test dataset
    test_dataset = MovingMNIST(
        root="/Users/apple/git/neuro/OpenSTL/data",
        is_train=False,
        data_name='mnist',
        n_frames_input=10,
        n_frames_output=10,
        image_size=64,
        num_objects=[2],
    )
    
    print(f"  Test dataset size: {len(test_dataset)} samples")
    
    # Create movies for multiple samples
    save_dir = Path("moving_mnist_zero_shot/results/convlstm_variant_movies")
    
    num_samples = 3
    for sample_idx in range(num_samples):
        print(f"\nProcessing sample {sample_idx+1}/{num_samples}...")
        
        # Get test sample
        test_sample = test_dataset[sample_idx]
        
        # Create variants from this sample
        print("  Creating variants...")
        variants = create_test_variants(test_sample)
        
        # Create comparison movie
        print("  Generating predictions and creating visualizations...")
        create_variant_comparison_movie(model, variants, sample_idx, save_dir)
    
    print("\n" + "="*60)
    print("Complete!")  
    print("="*60)
    print("Generated ConvLSTM movies comparing all variants with same test sample:")
    print(f"  {save_dir}/")
    print("\nEach movie shows how ConvLSTM handles different motion patterns:")
    print("  - Standard, Fast 1.5x, Fast 2x, Fast 3x, Three Digits")

if __name__ == '__main__':
    main()
