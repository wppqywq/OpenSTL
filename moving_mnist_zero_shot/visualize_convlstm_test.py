#!/usr/bin/env python
"""Visualize ConvLSTM model test predictions as movies/GIFs."""

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

from openstl.models import ConvLSTM_Model
from openstl.datasets.dataloader_moving_mnist import MovingMNIST

def reshape_patch(img_tensor, patch_size):
    """Reshape image tensor into patches.
    
    Args:
        img_tensor: shape [B, T, C, H, W] or [B, T, H, W, C]
        patch_size: int, size of each patch
    
    Returns:
        Reshaped tensor with patches
    """
    if img_tensor.dim() == 5:
        # Assume [B, T, H, W, C] format
        B, T, H, W, C = img_tensor.shape
        # Reshape to patches
        img_tensor = img_tensor.reshape(B, T, H//patch_size, patch_size, W//patch_size, patch_size, C)
        img_tensor = img_tensor.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # [B, T, H//p, W//p, p, p, C]
        img_tensor = img_tensor.reshape(B, T, H//patch_size, W//patch_size, patch_size*patch_size*C)
    return img_tensor

def reshape_patch_back(img_tensor, patch_size):
    """Reshape patches back to image.
    
    Args:
        img_tensor: shape [B, T, H//p, W//p, p*p*C]
        patch_size: int, size of each patch
    
    Returns:
        Image tensor [B, T, H, W, C]
    """
    B, T, H_p, W_p, _ = img_tensor.shape
    C = 1  # Assuming single channel
    H = H_p * patch_size
    W = W_p * patch_size
    
    img_tensor = img_tensor.reshape(B, T, H_p, W_p, patch_size, patch_size, C)
    img_tensor = img_tensor.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # [B, T, H_p, p, W_p, p, C]
    img_tensor = img_tensor.reshape(B, T, H, W, C)
    return img_tensor

def create_test_movie(model, test_dataset, sample_idx=0, save_dir="moving_mnist_zero_shot/results/convlstm_test_movies"):
    """Generate test prediction movie showing Input, Ground Truth, and Prediction side by side."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nProcessing test sample {sample_idx}...")
    
    # Get test sample
    input_seq, true_seq = test_dataset[sample_idx]
    
    # Transform to [B, T, H, W, C] format
    # [T, C, H, W] -> [B, T, H, W, C]
    input_batch = input_seq.permute(0, 2, 3, 1).unsqueeze(0).to(device)
    true_batch = true_seq.permute(0, 2, 3, 1).unsqueeze(0).to(device)
    
    # Apply patch transformation
    patch_size = 4
    input_patches = reshape_patch(input_batch, patch_size)
    true_patches = reshape_patch(true_batch, patch_size)
    
    # Create full sequence for the model
    full_seq = torch.cat([input_patches, true_patches], dim=1).to(device)  # [B, 20, H//p, W//p, p*p*C]
    
    # Create mask for scheduled sampling (zeros during testing to use predicted frames)
    mask = torch.zeros(1, 19, 16, 16, 16).to(device)  # Shape matches patched frames
    
    # Generate prediction
    with torch.no_grad():
        pred_seq, _ = model(full_seq, mask, return_loss=False)
        # Extract prediction frames (last 10 frames) 
        pred_seq = pred_seq[:, 9:19]  # Skip first 9, get next 10
        
        # Reshape patches back to images
        pred_seq = reshape_patch_back(pred_seq, patch_size)  # [B, T, H, W, C]
    
    # Convert to numpy
    input_np = input_seq.cpu().numpy()
    true_np = true_seq.cpu().numpy()
    pred_np = pred_seq[0].permute(0, 3, 1, 2).cpu().numpy()  # [T, H, W, C] -> [T, C, H, W]
    
    # Clip predictions to valid range
    pred_np = np.clip(pred_np, 0, 1)
    
    # Create figure with 3 columns for each frame
    fig = plt.figure(figsize=(12, 5))
    
    # Create axes for Input, Ground Truth, and Prediction
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    
    ax1.set_title('Input', fontsize=14, pad=10)
    ax2.set_title('Ground Truth', fontsize=14, pad=10)
    ax3.set_title('Prediction', fontsize=14, pad=10)
    
    # Initialize images
    img1 = ax1.imshow(np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1)
    img2 = ax2.imshow(np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1)
    img3 = ax3.imshow(np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1)
    
    # Animation function
    def animate(frame_idx):
        if frame_idx < 10:
            # Show input frames
            img1.set_data(input_np[frame_idx, 0])
            img2.set_data(np.zeros((64, 64)))  # Empty during input phase
            img3.set_data(np.zeros((64, 64)))  # Empty during input phase
            fig.suptitle(f'Sample {sample_idx + 1} - Input Frame {frame_idx + 1}/10', fontsize=16, fontweight='bold')
        else:
            # Show predictions
            pred_frame = frame_idx - 10
            img1.set_data(input_np[-1, 0])  # Keep last input frame
            img2.set_data(true_np[pred_frame, 0])
            img3.set_data(pred_np[pred_frame, 0])
            fig.suptitle(f'Sample {sample_idx + 1} - Prediction Frame {pred_frame + 1}/10', fontsize=16, fontweight='bold')
        
        return [img1, img2, img3]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=20, interval=300, blit=True
    )
    
    # Save as GIF
    gif_path = save_dir / f'convlstm_test_sample_{sample_idx:02d}.gif'
    anim.save(gif_path, writer='pillow', fps=4)
    print(f"  Saved: {gif_path}")
    
    plt.close()
    
    return gif_path

def create_test_frames_grid(model, test_dataset, sample_idx=0, save_dir="moving_mnist_zero_shot/results/convlstm_test_movies"):
    """Create a static frame grid showing key frames from the test."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # Get test sample
    input_seq, true_seq = test_dataset[sample_idx]
    
    # Transform to [B, T, H, W, C] format
    # [T, C, H, W] -> [B, T, H, W, C]
    input_batch = input_seq.permute(0, 2, 3, 1).unsqueeze(0).to(device)
    true_batch = true_seq.permute(0, 2, 3, 1).unsqueeze(0).to(device)
    
    # Apply patch transformation
    patch_size = 4
    input_patches = reshape_patch(input_batch, patch_size)
    true_patches = reshape_patch(true_batch, patch_size)
    
    # Create full sequence for the model
    full_seq = torch.cat([input_patches, true_patches], dim=1).to(device)  # [B, 20, H//p, W//p, p*p*C]
    
    # Create mask for scheduled sampling (zeros during testing to use predicted frames)
    mask = torch.zeros(1, 19, 16, 16, 16).to(device)  # Shape matches patched frames
    
    # Generate prediction
    with torch.no_grad():
        pred_seq, _ = model(full_seq, mask, return_loss=False)
        # Extract prediction frames (last 10 frames)
        pred_seq = pred_seq[:, 9:19]  # Skip first 9, get next 10
        
        # Reshape patches back to images
        pred_seq = reshape_patch_back(pred_seq, patch_size)  # [B, T, H, W, C]
    
    # Convert to numpy
    input_np = input_seq.cpu().numpy()
    true_np = true_seq.cpu().numpy()
    pred_np = pred_seq[0].permute(0, 3, 1, 2).cpu().numpy()  # [T, H, W, C] -> [T, C, H, W]
    
    # Clip predictions to valid range
    pred_np = np.clip(pred_np, 0, 1)
    
    # Select key frames to show
    input_frames = [0, 3, 6, 9]  # 4 input frames
    pred_frames = [0, 3, 6, 9]   # 4 prediction frames
    
    # Create figure
    fig, axes = plt.subplots(len(input_frames) + len(pred_frames), 3, figsize=(9, 20))
    
    # Plot input frames
    for i, t in enumerate(input_frames):
        axes[i, 0].imshow(input_np[t, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')  # Empty during input
        axes[i, 2].axis('off')  # Empty during input
        
        if i == 0:
            axes[i, 0].set_title('Input', fontsize=12, fontweight='bold')
            axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 2].set_title('Prediction', fontsize=12, fontweight='bold')
        
        # Add frame number on the left
        axes[i, 0].text(-0.15, 0.5, f'Frame {t+1}', transform=axes[i, 0].transAxes,
                       fontsize=10, ha='right', va='center')
    
    # Plot prediction frames
    for i, t in enumerate(pred_frames):
        row_idx = len(input_frames) + i
        axes[row_idx, 0].imshow(input_np[-1, 0], cmap='gray', vmin=0, vmax=1)  # Keep last input
        axes[row_idx, 1].imshow(true_np[t, 0], cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 2].imshow(pred_np[t, 0], cmap='gray', vmin=0, vmax=1)
        
        for j in range(3):
            axes[row_idx, j].axis('off')
        
        # Add frame number on the left
        axes[row_idx, 0].text(-0.15, 0.5, f'Pred {t+1}', transform=axes[row_idx, 0].transAxes,
                             fontsize=10, ha='right', va='center')
    
    plt.suptitle(f'Sample {sample_idx + 1} - ConvLSTM Test Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    png_path = save_dir / f'convlstm_test_sample_{sample_idx:02d}_frames.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved frame grid: {png_path}")
    plt.close()
    
    return png_path

def main():
    print("Loading ConvLSTM model and test data...")
    
    # Create configs object similar to training configs
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
    
    # Initialize ConvLSTM model with proper parameters
    num_hidden = [128, 128, 128, 128]  # 4 layers with 128 hidden units each
    num_layers = len(num_hidden)
    
    model = ConvLSTM_Model(num_layers, num_hidden, configs)
    
    # Load best checkpoint
    ckpt_path = '/Users/apple/git/neuro/OpenSTL/work_dirs/convlstm_mmnist_m2/checkpoints/best.ckpt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # Handle 'model.' prefix in state dict
        state_dict = ckpt['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        
        # Load state dict
        model.load_state_dict(new_state_dict, strict=False)
        print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
        print(f"  Val loss: {ckpt.get('val_loss', 'unknown')}")
    else:
        print(f"  Warning: Checkpoint not found at {ckpt_path}")
    
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
    
    # Generate visualizations for multiple samples
    print("\n" + "="*60)
    print("Generating ConvLSTM test movies...")
    print("="*60)
    
    num_samples = 3  # Generate for first 3 test samples
    for i in range(num_samples):
        # Create animated GIF
        gif_path = create_test_movie(model, test_dataset, sample_idx=i)
        
        # Create static frame grid
        png_path = create_test_frames_grid(model, test_dataset, sample_idx=i)
    
    print("\n" + "="*60)
    print("  Visualizations saved to:")
    print("  - moving_mnist_zero_shot/results/convlstm_test_movies/")
    print("="*60)

if __name__ == '__main__':
    main()