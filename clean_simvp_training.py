#!/usr/bin/env python
"""
Clean SimVP training using OpenSTL's standard SimVP_Model
Based on working approach from paste.txt, with validated data
"""

import os
import torch
import numpy as np
from openstl.models import SimVP_Model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from pathlib import Path


def load_validated_coco_data(data_root, config='short'):
    """Load validated COCO data"""
    data_root = Path(data_root)
    
    train_file = data_root / f'{config}_train_sequences.npy'
    val_file = data_root / f'{config}_val_sequences.npy'
    
    if not train_file.exists():
        print(f"Data not found. Run: python clean_data_processor.py --data_root {data_root.parent} --config {config}")
        return None, None
    
    train_sequences = np.load(train_file)
    val_sequences = np.load(val_file)
    
    print(f"Loaded: train {train_sequences.shape}, val {val_sequences.shape}")
    return train_sequences, val_sequences


def coords_to_spatial_batch(coords_batch, spatial_size=32):
    """Convert coordinates to spatial heatmaps - same as paste.txt"""
    B, T, _ = coords_batch.shape
    spatial = np.zeros((B, T, spatial_size, spatial_size), dtype=np.float32)
    
    sigma = 2.0
    for b in range(B):
        for t in range(T):
            x = int(coords_batch[b, t, 0] * (spatial_size - 1))
            y = int(coords_batch[b, t, 1] * (spatial_size - 1))
            x = np.clip(x, 0, spatial_size - 1)
            y = np.clip(y, 0, spatial_size - 1)
            
            for i in range(max(0, x-5), min(spatial_size, x+6)):
                for j in range(max(0, y-5), min(spatial_size, y+6)):
                    dist = np.sqrt((i - x)**2 + (j - y)**2)
                    spatial[b, t, j, i] = np.exp(-dist**2 / (2 * sigma**2))
    
    return spatial


def spatial_to_coords_batch(spatial_batch):
    """Convert spatial heatmaps back to coordinates"""
    B, T, H, W = spatial_batch.shape
    coords = np.zeros((B, T, 2))
    
    for b in range(B):
        for t in range(T):
            heatmap = spatial_batch[b, t]
            y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            coords[b, t, 0] = x_idx / (W - 1)
            coords[b, t, 1] = y_idx / (H - 1)
    
    return coords


def train_simvp_coco_clean(data_root='./data/coco_search18_tp/processed', 
                          config='short', 
                          epochs=50, 
                          batch_size=16, 
                          device='mps'):
    """
    Clean SimVP training using OpenSTL's standard model
    
    Explanation:
    1. Uses OpenSTL's SimVP_Model without modifications
    2. Converts eye coordinates to 32x32 heatmaps (spatial representation)
    3. Splits 10-frame sequences into 5 input + 5 target frames
    4. Uses standard training loop with MSE loss
    """
    
    print("Training OpenSTL SimVP on validated COCO-Search18 data")
    
    # Load validated data
    train_sequences, val_sequences = load_validated_coco_data(data_root, config)
    if train_sequences is None:
        return
    
    # Convert coordinates to spatial heatmaps
    print("Converting coordinates to spatial heatmaps...")
    train_spatial = coords_to_spatial_batch(train_sequences)
    val_spatial = coords_to_spatial_batch(val_sequences)
    
    # Split into input/target (5 frames each for 'short' config)
    seq_len = train_spatial.shape[1]
    input_len = seq_len // 2
    
    train_inputs = train_spatial[:, :input_len]
    train_targets = train_spatial[:, input_len:]
    val_inputs = val_spatial[:, :input_len]
    val_targets = val_spatial[:, input_len:]
    
    print(f"Sequence split: {input_len} input -> {seq_len - input_len} target frames")
    
    # Create standard OpenSTL SimVP model
    # Key: in_shape must match actual input dimensions
    model = SimVP_Model(
        in_shape=(input_len, 1, 32, 32),  # (T=5, C=1, H=32, W=32)
        hid_S=32,
        hid_T=128,
        N_S=2,
        N_T=4,
        model_type='gSTA'  # This uses the gSTA variant
    )
    
    # Setup device
    if device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Convert to tensors with correct shape: [B, T, C, H, W]
    train_x = torch.from_numpy(train_inputs).float().unsqueeze(2)
    train_y = torch.from_numpy(train_targets).float().unsqueeze(2)
    val_x = torch.from_numpy(val_inputs).float().unsqueeze(2)
    val_y = torch.from_numpy(val_targets).float().unsqueeze(2)
    
    print(f"Tensor shapes: train_x={train_x.shape}, train_y={train_y.shape}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size].to(device)
            batch_y = train_y[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        # Validate
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_x), batch_size):
                batch_x = val_x[i:i+batch_size].to(device)
                batch_y = val_y[i:i+batch_size].to(device)
                
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                n_val_batches += 1
        
        scheduler.step()
        
        avg_train_loss = train_loss / n_batches
        avg_val_loss = val_loss / n_val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'simvp_coco_{config}_best.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
    
    # Evaluate coordinate prediction accuracy
    print("\nEvaluating coordinate prediction accuracy...")
    model.load_state_dict(torch.load(f'simvp_coco_{config}_best.pth'))
    model.eval()
    
    sample_size = min(100, len(val_x))
    with torch.no_grad():
        val_sample = val_x[:sample_size].to(device)
        val_pred = model(val_sample).cpu()
    
    # Convert predictions back to coordinates
    val_pred_spatial = val_pred.squeeze(2).numpy()
    val_target_coords = val_sequences[:sample_size, input_len:]
    val_pred_coords = spatial_to_coords_batch(val_pred_spatial)
    
    # Calculate coordinate error
    coord_errors = np.sqrt(np.sum((val_pred_coords - val_target_coords)**2, axis=-1))
    mean_coord_error = np.mean(coord_errors)
    
    print(f"Final coordinate MAE: {mean_coord_error:.4f}")
    print(f"Coordinate error as % of screen: {mean_coord_error*100:.1f}%")
    
    # Create results plot
    plot_results(train_losses, val_losses, 
                val_sequences[:5], val_pred_coords[:5], val_target_coords[:5], 
                config)
    
    return mean_coord_error


def plot_results(train_losses, val_losses, input_seqs, pred_seqs, target_seqs, config):
    """Plot training results and sample predictions"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Learning curves
    axes[0, 0].plot(train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation', linewidth=2) 
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample predictions
    for i in range(min(5, len(input_seqs))):
        row = i // 3 if i >= 2 else 0
        col = (i % 3) + 1 if i < 2 else i - 2
        ax = axes[row, col]
        
        # Plot full input sequence
        full_seq = input_seqs[i]  # All 10 frames
        input_part = full_seq[:5]  # First 5 frames (input)
        target_seq = target_seqs[i]  # Last 5 frames (target)
        pred_seq = pred_seqs[i]  # Predicted last 5 frames
        
        ax.plot(full_seq[:, 0], full_seq[:, 1], 'b.-', markersize=6, 
                linewidth=2, label='Full sequence')
        ax.plot(target_seq[:, 0], target_seq[:, 1], 'g.--', markersize=8, 
                linewidth=2, label='True future')
        ax.plot(pred_seq[:, 0], pred_seq[:, 1], 'r.--', markersize=8, 
                linewidth=2, label='Predicted')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_title(f'Sample {i+1}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'simvp_coco_{config}_results.png', dpi=150, bbox_inches='tight')
    print(f"Results saved: simvp_coco_{config}_results.png")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SimVP on COCO-Search18')
    parser.add_argument('--data_root', default='./data/coco_search18_tp/processed')
    parser.add_argument('--config', default='short', choices=['short', 'standard', 'medium'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    coord_error = train_simvp_coco_clean(
        data_root=args.data_root,
        config=args.config, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if coord_error:
        print(f"\nFinal coordinate error: {coord_error:.4f}")
        print("Training complete")