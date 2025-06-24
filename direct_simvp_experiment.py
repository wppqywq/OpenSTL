#!/usr/bin/env python
"""
Fixed Direct SimVP experiment
"""

import os
import torch
import numpy as np
from openstl.models import SimVP_Model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def load_coco_data():
    """Load COCO-Search data directly"""
    
    train_sequences = np.load('data/coco_search18_tp/processed/short_train_sequences.npy')
    val_sequences = np.load('data/coco_search18_tp/processed/short_val_sequences.npy')
    
    print(f"Loaded data: train {train_sequences.shape}, val {val_sequences.shape}")
    
    train_inputs = train_sequences[:, :5]
    train_targets = train_sequences[:, 5:10]
    
    val_inputs = val_sequences[:, :5]
    val_targets = val_sequences[:, 5:10]
    
    return (train_inputs, train_targets), (val_inputs, val_targets)


def coords_to_spatial_batch(coords_batch, spatial_size=32):
    """Convert batch of coordinates to spatial heatmaps"""
    
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


def train_simvp_direct():
    """Train SimVP directly with correct format"""
    
    print("Training SimVP on COCO-Search18")
    print("=" * 60)
    
    # Load data
    (train_inputs, train_targets), (val_inputs, val_targets) = load_coco_data()
    
    # Convert to spatial
    print("Converting to spatial representation...")
    train_inputs_spatial = coords_to_spatial_batch(train_inputs)
    train_targets_spatial = coords_to_spatial_batch(train_targets)
    val_inputs_spatial = coords_to_spatial_batch(val_inputs)
    val_targets_spatial = coords_to_spatial_batch(val_targets)
    
    # Create model with correct shape
    # SimVP expects (T, C, H, W) where C=1 for grayscale
    model = SimVP_Model(
        in_shape=(5, 1, 32, 32),  # T=5, C=1, H=32, W=32
        hid_S=32,
        hid_T=128,
        N_S=2,
        N_T=4,
        model_type='gSTA'
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Convert to tensors with correct shape
    # Add channel dimension: [B, T, H, W] -> [B, T, C, H, W]
    train_x = torch.from_numpy(train_inputs_spatial).float().unsqueeze(2)  # [B, T, 1, H, W]
    train_y = torch.from_numpy(train_targets_spatial).float().unsqueeze(2)
    val_x = torch.from_numpy(val_inputs_spatial).float().unsqueeze(2)
    val_y = torch.from_numpy(val_targets_spatial).float().unsqueeze(2)
    
    print(f"Data shapes: train_x={train_x.shape}, train_y={train_y.shape}")
    
    # Training
    batch_size = 32
    n_epochs = 20
    train_losses = []
    val_losses = []
    
    print("\nTraining...")
    for epoch in range(n_epochs):
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
        
        avg_train_loss = train_loss / n_batches
        avg_val_loss = val_loss / n_val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    
    # Get predictions on validation set
    sample_size = min(100, len(val_x))
    with torch.no_grad():
        val_x_sample = val_x[:sample_size].to(device)
        val_pred = model(val_x_sample)
    
    # Convert back to coordinates
    val_pred_spatial = val_pred.squeeze(2).cpu().numpy()  # Remove channel dimension
    val_target_spatial = val_targets_spatial[:sample_size]
    
    val_pred_coords = spatial_to_coords_batch(val_pred_spatial)
    val_target_coords = val_targets[:sample_size]
    
    # Calculate coordinate MAE
    coord_mae = np.mean(np.abs(val_pred_coords - val_target_coords))
    print(f"Coordinate MAE: {coord_mae:.4f}")
    
    # Compare with LSTM baseline
    lstm_mae = 0.1704  # From previous experiment
    print(f"LSTM baseline MAE: {lstm_mae:.4f}")
    print(f"SimVP vs LSTM: {coord_mae/lstm_mae:.2f}x")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Learning curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Spatial)')
    plt.title('SimVP Training Curves')
    plt.legend()
    plt.grid(True)
    
    # Sample predictions
    plt.subplot(1, 3, 2)
    idx = 0
    input_seq = val_inputs[idx]
    target_seq = val_target_coords[idx]
    pred_seq = val_pred_coords[idx]
    
    plt.plot(input_seq[:, 0], input_seq[:, 1], 'b.-', label='Input', markersize=10, linewidth=2)
    plt.plot(target_seq[:, 0], target_seq[:, 1], 'g.--', label='Target', markersize=10, linewidth=2)
    plt.plot(pred_seq[:, 0], pred_seq[:, 1], 'r.--', label='SimVP', markersize=10, linewidth=2)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Sample Prediction')
    plt.grid(True)
    
    # Error by timestep
    plt.subplot(1, 3, 3)
    timestep_errors = []
    for t in range(5):
        t_error = np.mean(np.abs(val_pred_coords[:, t] - val_target_coords[:, t]))
        timestep_errors.append(t_error)
    
    plt.plot(range(1, 6), timestep_errors, 'o-', markersize=10, linewidth=2)
    plt.xlabel('Prediction Step')
    plt.ylabel('MAE')
    plt.title('Error Growth Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simvp_fixed_results.png', dpi=150)
    print("\nSaved results to simvp_fixed_results.png")
    
    return coord_mae, train_losses[-1], val_losses[-1]


if __name__ == '__main__':
    if not os.path.exists('data/coco_search18_tp/processed/short_train_sequences.npy'):
        print("Error: Processed data not found")
        exit(1)
    
    mae, final_train, final_val = train_simvp_direct()
    
    print("\n" + "=" * 60)
    print("SimVP Experiment Complete")
    print("=" * 60)
    print(f"Coordinate MAE: {mae:.4f} ({mae*100:.1f}% of screen)")
    print(f"Final train loss: {final_train:.4f}")
    print(f"Final val loss: {final_val:.4f}")