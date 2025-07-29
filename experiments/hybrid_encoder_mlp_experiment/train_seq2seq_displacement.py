#!/usr/bin/env python3
"""
Seq2Seq Displacement Prediction Training Script
Trains SimVP Encoder + GRU Decoder with Teacher Forcing for multi-step eye movement prediction.
This is the CORRECT implementation that uses real encoder features (not random noise).
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

from model import create_seq2seq_model
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


def create_seq2seq_displacement_dataloaders(input_frames=5, prediction_frames=3):
    """
    Create data loaders for Seq2Seq displacement prediction.
    Target = sequence of displacement vectors from last input frame.
    """
    # Load existing heatmap data
    single_fix_data_dir = Path(__file__).parent.parent / 'single_fixation_experiment' / 'data'
    
    train_data = torch.load(single_fix_data_dir / 'train_data.pt')
    val_data = torch.load(single_fix_data_dir / 'val_data.pt')
    test_data = torch.load(single_fix_data_dir / 'test_data.pt')
    
    # Convert frame format: (B, C, T, H, W) -> (B, T, C, H, W)
    train_frames = train_data['frames'].permute(0, 2, 1, 3, 4)
    val_frames = val_data['frames'].permute(0, 2, 1, 3, 4)
    test_frames = test_data['frames'].permute(0, 2, 1, 3, 4)
    
    # Check if we have enough frames
    total_frames = train_frames.shape[1]
    required_frames = input_frames + prediction_frames
    if required_frames > total_frames:
        print(f"Warning: Need {required_frames} frames, have {total_frames}")
        prediction_frames = total_frames - input_frames
        print(f"Adjusted to predict {prediction_frames} frames")
    
    print(f"Seq2Seq data configuration:")
    print(f"  Input frames: {input_frames}")
    print(f"  Prediction frames: {prediction_frames}")
    print(f"  Total frames required: {required_frames}")
    
    # Process each split
    datasets = []
    for split_name, frames in [('train', train_frames), ('val', val_frames), ('test', test_frames)]:
        
        # Split into input sequences and future frames
        input_sequences = frames[:, :input_frames]
        
        # Extract all coordinates for displacement calculation
        all_coords = []
        for i in range(input_frames + prediction_frames):
            frame = frames[:, i:i+1]
            coords = extract_coordinates_from_heatmaps(frame).squeeze(1)
            all_coords.append(coords)
        
        all_coords = torch.stack(all_coords, dim=1)  # [B, T, 2]
        
        # Last input coordinate (reference point for displacements)
        last_input_coord = all_coords[:, input_frames-1]  # [B, 2]
        
        # Future coordinates
        future_coords = all_coords[:, input_frames:input_frames+prediction_frames]  # [B, pred_frames, 2]
        
        # Calculate displacement sequence relative to last input frame
        displacement_sequence = future_coords - last_input_coord.unsqueeze(1)  # [B, pred_frames, 2]
        
        print(f"  {split_name} data shapes:")
        print(f"    Input: {input_sequences.shape}")
        print(f"    Displacement sequence: {displacement_sequence.shape}")
        print(f"    Last input coords: {last_input_coord.shape}")
        
        # Show displacement statistics for training set
        if split_name == 'train':
            for i in range(prediction_frames):
                disp = displacement_sequence[:, i]
                print(f"    Frame +{i+1} displacement: mean=({disp[:, 0].mean():.2f}, {disp[:, 1].mean():.2f}), "
                      f"std=({disp[:, 0].std():.2f}, {disp[:, 1].std():.2f})")
        
        # Create dataset
        dataset = TensorDataset(input_sequences, displacement_sequence, last_input_coord)
        datasets.append(dataset)
    
    # Create data loaders
    train_loader = DataLoader(datasets[0], batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets[1], batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets[2], batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, prediction_frames


def calculate_seq2seq_error(predicted_sequence, target_sequence, last_coords):
    """
    Calculate pixel error for sequence predictions.
    Convert displacement sequences back to absolute coordinates for error calculation.
    """
    # Convert to absolute coordinates
    # predicted_sequence: (B, T, 2), target_sequence: (B, T, 2), last_coords: (B, 2)
    batch_size, seq_len, _ = predicted_sequence.shape
    
    # Expand last_coords for broadcasting
    last_coords_expanded = last_coords.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, 2)
    
    predicted_coords = last_coords_expanded + predicted_sequence  # (B, T, 2)
    target_coords = last_coords_expanded + target_sequence  # (B, T, 2)
    
    # Calculate L2 pixel distance for each frame
    pixel_distances = torch.sqrt(torch.sum((predicted_coords - target_coords)**2, dim=2))  # (B, T)
    
    # Return mean error across all frames and samples
    return torch.mean(pixel_distances)


def create_weighted_loss(prediction_frames, decay_factor=0.8):
    """
    Create weighted loss function that gives more importance to nearer frames.
    
    Args:
        prediction_frames: Number of future frames being predicted
        decay_factor: Weight decay factor for distant frames
        
    Returns:
        weights: Tensor of weights for each frame [1.0, 0.8, 0.64, ...]
    """
    weights = torch.tensor([decay_factor ** i for i in range(prediction_frames)])
    # Normalize so they sum to prediction_frames (keeps loss magnitude similar)
    weights = weights * prediction_frames / weights.sum()
    return weights


def train_epoch(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio=0.8):
    """Train for one epoch with Teacher Forcing"""
    model.train()
    total_loss = 0.0
    total_error = 0.0
    num_batches = 0
    
    for input_video, target_sequence, last_coords in tqdm(train_loader, desc="Training"):
        input_video = input_video.to(device)
        target_sequence = target_sequence.to(device)
        last_coords = last_coords.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with Teacher Forcing
        predicted_sequence = model(input_video, target_sequence, teacher_forcing_ratio)
        
        # Calculate loss (sequence-level)
        loss = criterion(predicted_sequence, target_sequence)
        
        # Calculate pixel error for monitoring
        with torch.no_grad():
            pixel_error = calculate_seq2seq_error(predicted_sequence, target_sequence, last_coords)
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
    """Validate the Seq2Seq model"""
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    num_batches = 0
    
    # Track per-frame errors
    prediction_frames = None
    frame_errors = None
    
    with torch.no_grad():
        for input_video, target_sequence, last_coords in val_loader:
            input_video = input_video.to(device)
            target_sequence = target_sequence.to(device)
            last_coords = last_coords.to(device)
            
            # Forward pass (no teacher forcing during validation)
            predicted_sequence = model(input_video)
            
            # Calculate loss and error
            loss = criterion(predicted_sequence, target_sequence)
            pixel_error = calculate_seq2seq_error(predicted_sequence, target_sequence, last_coords)
            
            total_loss += loss.item()
            total_error += pixel_error.item()
            num_batches += 1
            
            # Track per-frame errors
            if frame_errors is None:
                prediction_frames = predicted_sequence.shape[1]
                frame_errors = [[] for _ in range(prediction_frames)]
            
            # Calculate per-frame errors
            batch_size, seq_len, _ = predicted_sequence.shape
            last_coords_expanded = last_coords.unsqueeze(1).expand(-1, seq_len, -1)
            
            predicted_coords = last_coords_expanded + predicted_sequence
            target_coords = last_coords_expanded + target_sequence
            
            for frame_idx in range(seq_len):
                frame_pixel_errors = torch.sqrt(torch.sum(
                    (predicted_coords[:, frame_idx] - target_coords[:, frame_idx])**2, dim=1
                ))
                frame_errors[frame_idx].extend(frame_pixel_errors.cpu().tolist())
    
    avg_loss = total_loss / num_batches
    avg_error = total_error / num_batches
    
    # Calculate per-frame average errors
    frame_avg_errors = [np.mean(errors) for errors in frame_errors]
    
    return avg_loss, avg_error, frame_avg_errors


def main():
    """Main training function for Seq2Seq displacement model"""
    print("Starting Seq2Seq SimVP Encoder + GRU Decoder Training")
    print("="*80)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Configuration for Seq2Seq
    input_frames = 5
    prediction_frames = 3
    
    # Create data loaders
    print("Loading Seq2Seq displacement data...")
    train_loader, val_loader, test_loader, actual_pred_frames = create_seq2seq_displacement_dataloaders(
        input_frames, prediction_frames
    )
    prediction_frames = actual_pred_frames
    print(f"Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
    
    # Create Seq2Seq model
    print(f"Creating Seq2Seq model...")
    model = create_seq2seq_model(
        in_shape=(input_frames, 1, config.img_size, config.img_size),
        num_future_frames=prediction_frames,
        hid_S=config.model_hid_S,
        hid_T=config.model_hid_T,
        N_S=config.model_N_S,
        N_T=config.model_N_T,
        model_type=config.model_type,
        device=str(device)
    )
    
    # Loss function with frame weighting
    frame_weights = create_weighted_loss(prediction_frames, decay_factor=0.8)
    if device.type == 'mps':
        frame_weights = frame_weights.to('mps')
    
    def weighted_mse_loss(pred_seq, target_seq):
        # pred_seq, target_seq: (B, T, 2)
        mse_per_frame = torch.mean((pred_seq - target_seq)**2, dim=(0, 2))  # (T,)
        weighted_loss = torch.sum(frame_weights * mse_per_frame)
        return weighted_loss
    
    criterion = weighted_mse_loss
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate_regression * 0.5)  # Lower LR for Seq2Seq
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=config.lr_scheduler_patience, factor=0.5
    )
    
    # Training loop
    best_val_error = float('inf')
    log_file = 'logs/training_log_seq2seq_displacement.txt'
    
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Error,Val_Loss,Val_Error,")
        for i in range(prediction_frames):
            f.write(f"Frame{i+1}_Error,")
        f.write("LR,Time\n")
    
    print(f"Starting Seq2Seq training for {config.max_epochs_regression} epochs...")
    print(f"Frame weights: {frame_weights.tolist()}")
    
    for epoch in range(config.max_epochs_regression):
        start_time = time.time()
        
        # Train
        train_loss, train_error = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            teacher_forcing_ratio=0.8  # High teacher forcing initially
        )
        
        # Validate
        val_loss, val_error, frame_errors = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_error)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{config.max_epochs_regression} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.6f}, Error: {train_error:.2f}px")
        print(f"  Val   - Loss: {val_loss:.6f}, Error: {val_error:.2f}px")
        print(f"  Frame errors: {[f'{err:.2f}' for err in frame_errors]}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            torch.save(model.state_dict(), 'models/seq2seq_displacement_best.pth')
            print(f"  🏆 New best model saved! Val error: {val_error:.2f}px")
        
        # Log to file
        with open(log_file, 'a') as f:
            log_line = f"{epoch+1},{train_loss:.6f},{train_error:.2f},{val_loss:.6f},{val_error:.2f},"
            for err in frame_errors:
                log_line += f"{err:.2f},"
            log_line += f"{current_lr:.2e},{epoch_time:.1f}\n"
            f.write(log_line)
        
        # Early stopping
        if current_lr < 1e-6:
            print("Learning rate too small, stopping training")
            break
    
    # Final test evaluation
    print("\nEvaluating Seq2Seq model on test set...")
    model.load_state_dict(torch.load('models/seq2seq_displacement_best.pth'))
    test_loss, test_error, test_frame_errors = validate(model, test_loader, criterion, device)
    
    print(f"\nSeq2Seq Displacement Model Results:")
    print(f"="*60)
    print(f"  Best validation error: {best_val_error:.2f} pixels")
    print(f"  Test error: {test_error:.2f} pixels")
    print(f"  Per-frame test errors: {[f'{err:.2f}px' for err in test_frame_errors]}")
    
    # Compare with previous methods
    print(f"\n PERFORMANCE COMPARISON:")
    print(f"  Single-step displacement (5→1): 4.05px")
    print(f"  Seq2Seq displacement (5→3): {test_error:.2f}px avg")
    print(f"    Frame +1: {test_frame_errors[0]:.2f}px")
    if len(test_frame_errors) > 1:
        print(f"    Frame +2: {test_frame_errors[1]:.2f}px")
    if len(test_frame_errors) > 2:
        print(f"    Frame +3: {test_frame_errors[2]:.2f}px")
    
    # Analysis
    frame1_vs_single = test_frame_errors[0] - 4.05
    if frame1_vs_single < 0:
        print(f"   Frame +1 improvement: {-frame1_vs_single:.2f}px better than single-step")
    else:
        print(f"   Frame +1 degradation: {frame1_vs_single:.2f}px worse than single-step")
    
    # Save results
    with open('logs/seq2seq_results.txt', 'w') as f:
        f.write(f"Seq2Seq SimVP Displacement Prediction Results\n")
        f.write(f"Best validation error: {best_val_error:.2f} pixels\n")
        f.write(f"Test error: {test_error:.2f} pixels\n")
        f.write(f"Per-frame test errors: {test_frame_errors}\n")
        f.write(f"Frame weights used: {frame_weights.tolist()}\n")


if __name__ == "__main__":
    main() 