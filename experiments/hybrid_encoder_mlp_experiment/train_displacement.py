#!/usr/bin/env python3
"""
Displacement-based training for hybrid SimVP encoder + MLP regression model.
Instead of predicting absolute coordinates, predict relative displacement vectors.
This should solve the center bias problem.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

# Compatibility patches for NumPy 2.0
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

# Add OpenSTL to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model import create_hybrid_model
import config


def extract_coordinates_from_heatmaps(heatmaps):
    """Extract (x, y) coordinates from heatmap frames using argmax"""
    B, T, C, H, W = heatmaps.shape
    coords = torch.zeros(B, T, 2)
    
    for b in range(B):
        for t in range(T):
            frame = heatmaps[b, t, 0]
            flat_idx = torch.argmax(frame.view(-1))
            y = flat_idx // W
            x = flat_idx % W
            coords[b, t, 0] = x.float()
            coords[b, t, 1] = y.float()
    
    return coords


def create_displacement_dataloaders():
    """
    Create data loaders that return displacement vectors as targets.
    Target = next_position - last_input_position
    """
    # Load existing heatmap data from single_fixation_experiment
    single_fix_data_dir = Path(__file__).parent.parent / 'single_fixation_experiment' / 'data'
    
    train_data = torch.load(single_fix_data_dir / 'train_data.pt')
    val_data = torch.load(single_fix_data_dir / 'val_data.pt')
    test_data = torch.load(single_fix_data_dir / 'test_data.pt')
    
    # Convert frame format: (B, C, T, H, W) -> (B, T, C, H, W)
    train_frames = train_data['frames'].permute(0, 2, 1, 3, 4)
    val_frames = val_data['frames'].permute(0, 2, 1, 3, 4)
    test_frames = test_data['frames'].permute(0, 2, 1, 3, 4)
    
    # Split into input sequences and displacement targets
    train_input = train_frames[:, :config.input_frames_regression]
    val_input = val_frames[:, :config.input_frames_regression]
    test_input = test_frames[:, :config.input_frames_regression]
    
    # Extract coordinates for displacement calculation
    # Last input frame coordinate
    train_last_input_frame = train_frames[:, config.input_frames_regression-1:config.input_frames_regression]
    val_last_input_frame = val_frames[:, config.input_frames_regression-1:config.input_frames_regression]
    test_last_input_frame = test_frames[:, config.input_frames_regression-1:config.input_frames_regression]
    
    train_last_coords = extract_coordinates_from_heatmaps(train_last_input_frame).squeeze(1)
    val_last_coords = extract_coordinates_from_heatmaps(val_last_input_frame).squeeze(1)
    test_last_coords = extract_coordinates_from_heatmaps(test_last_input_frame).squeeze(1)
    
    # Target frame coordinate (next frame after input sequence)
    train_target_frame = train_frames[:, config.input_frames_regression:config.input_frames_regression+1]
    val_target_frame = val_frames[:, config.input_frames_regression:config.input_frames_regression+1]
    test_target_frame = test_frames[:, config.input_frames_regression:config.input_frames_regression+1]
    
    train_target_coords = extract_coordinates_from_heatmaps(train_target_frame).squeeze(1)
    val_target_coords = extract_coordinates_from_heatmaps(val_target_frame).squeeze(1)
    test_target_coords = extract_coordinates_from_heatmaps(test_target_frame).squeeze(1)
    
    # Calculate displacement vectors: target - last_input
    train_displacements = train_target_coords - train_last_coords
    val_displacements = val_target_coords - val_last_coords
    test_displacements = test_target_coords - test_last_coords
    
    print(f"Displacement-based data shapes:")
    print(f"  Train input: {train_input.shape}")
    print(f"  Train displacements: {train_displacements.shape}")
    print(f"  Train last coords: {train_last_coords.shape}")
    print(f"  Val input: {val_input.shape}")
    print(f"  Val displacements: {val_displacements.shape}")
    
    # Show displacement statistics
    print(f"\nDisplacement statistics:")
    print(f"  Train displacement range: X[{train_displacements[:, 0].min():.1f}, {train_displacements[:, 0].max():.1f}], "
          f"Y[{train_displacements[:, 1].min():.1f}, {train_displacements[:, 1].max():.1f}]")
    print(f"  Train displacement mean: X={train_displacements[:, 0].mean():.2f}, Y={train_displacements[:, 1].mean():.2f}")
    print(f"  Train displacement std: X={train_displacements[:, 0].std():.2f}, Y={train_displacements[:, 1].std():.2f}")
    
    # Create datasets with input and displacement targets
    # We also store last_coords for inference
    train_dataset = TensorDataset(train_input, train_displacements, train_last_coords)
    val_dataset = TensorDataset(val_input, val_displacements, val_last_coords)
    test_dataset = TensorDataset(test_input, test_displacements, test_last_coords)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def calculate_displacement_error(predicted_displacements, target_displacements, last_coords):
    """
    Calculate pixel error for displacement predictions.
    Convert displacements back to absolute coordinates for error calculation.
    """
    # Convert to absolute coordinates
    predicted_coords = last_coords + predicted_displacements
    target_coords = last_coords + target_displacements
    
    # Calculate L2 pixel distance
    pixel_distances = torch.sqrt(torch.sum((predicted_coords - target_coords)**2, dim=1))
    return torch.mean(pixel_distances)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with displacement targets"""
    model.train()
    total_loss = 0.0
    total_error = 0.0
    num_batches = 0
    
    for input_video, target_displacements, last_coords in tqdm(train_loader, desc="Training"):
        input_video = input_video.to(device)
        target_displacements = target_displacements.to(device)
        last_coords = last_coords.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - model predicts displacement
        predicted_displacements = model(input_video)
        
        # Calculate loss (displacement regression)
        loss = criterion(predicted_displacements, target_displacements)
        
        # Calculate pixel error for monitoring
        with torch.no_grad():
            pixel_error = calculate_displacement_error(predicted_displacements, target_displacements, last_coords)
            total_error += pixel_error.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_error = total_error / num_batches
    
    return avg_loss, avg_error


def validate(model, val_loader, criterion, device):
    """Validate the model with displacement targets"""
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_video, target_displacements, last_coords in val_loader:
            input_video = input_video.to(device)
            target_displacements = target_displacements.to(device)
            last_coords = last_coords.to(device)
            
            # Forward pass
            predicted_displacements = model(input_video)
            
            # Calculate loss and error
            loss = criterion(predicted_displacements, target_displacements)
            pixel_error = calculate_displacement_error(predicted_displacements, target_displacements, last_coords)
            
            total_loss += loss.item()
            total_error += pixel_error.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_error = total_error / num_batches
    
    return avg_loss, avg_error


def main():
    """Main training function for displacement-based model"""
    print("Starting hybrid SimVP encoder + MLP displacement regression training")
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create data loaders with displacement targets
    print("Loading displacement-based data...")
    train_loader, val_loader, test_loader = create_displacement_dataloaders()
    print(f"Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
    
    # Create model - same architecture, but now predicts displacements
    print("Creating hybrid model for displacement prediction...")
    model = create_hybrid_model(
        in_shape=(config.input_frames_regression, 1, config.img_size, config.img_size),
        hid_S=config.model_hid_S,
        hid_T=config.model_hid_T,
        N_S=config.model_N_S,
        N_T=config.model_N_T,
        model_type=config.model_type,
        device=str(device)
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function - L1 loss for displacement regression
    criterion = nn.L1Loss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate_regression)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=config.lr_scheduler_patience, factor=0.5
    )
    
    # Training loop
    best_val_error = float('inf')
    
    # Setup logging
    log_file = 'logs/training_log_displacement.txt'
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Error,Val_Loss,Val_Error,LR,Time\n")
    
    print(f"Starting displacement training for {config.max_epochs_regression} epochs...")
    
    for epoch in range(config.max_epochs_regression):
        start_time = time.time()
        
        # Train
        train_loss, train_error = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_error = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_error)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.max_epochs_regression} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.6f}, Error: {train_error:.2f}px")
        print(f"  Val   - Loss: {val_loss:.6f}, Error: {val_error:.2f}px")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            torch.save(model.state_dict(), 'models/hybrid_displacement_best.pth')
            print(f"  New best model saved! Val error: {val_error:.2f}px")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_error:.2f},{val_loss:.6f},"
                   f"{val_error:.2f},{current_lr:.2e},{epoch_time:.1f}\n")
        
        # Early stopping
        if current_lr < 1e-6:
            print("Learning rate too small, stopping training")
            break
    
    # Final test evaluation
    print("\nEvaluating displacement model on test set...")
    model.load_state_dict(torch.load('models/hybrid_displacement_best.pth'))
    test_loss, test_error = validate(model, test_loader, criterion, device)
    
    print(f"\nDisplacement Model Results:")
    print(f"  Best validation error: {best_val_error:.2f} pixels")
    print(f"  Test error: {test_error:.2f} pixels")
    print(f"  Test loss: {test_loss:.6f}")
    
    # Save final results
    with open('logs/displacement_results.txt', 'w') as f:
        f.write(f"Hybrid SimVP Displacement Regression Results\n")
        f.write(f"Best validation error: {best_val_error:.2f} pixels\n")
        f.write(f"Test error: {test_error:.2f} pixels\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
    
    # Compare with absolute coordinate approach
    print(f"\n=== COMPARISON ===")
    print(f"Absolute coordinates: 5.57px test error")
    print(f"Displacement vectors: {test_error:.2f}px test error")
    improvement = 5.57 - test_error
    if improvement > 0:
        print(f" Improvement: {improvement:.2f}px ({improvement/5.57*100:.1f}% better)")
    else:
        print(f" Regression: {-improvement:.2f}px ({-improvement/5.57*100:.1f}% worse)")


if __name__ == "__main__":
    main() 