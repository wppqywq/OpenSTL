#!/usr/bin/env python3
"""
Unified training script for position-dependent Gaussian prediction experiments.

This script serves as the entry point for all training experiments,
supporting different datasets, representations, models, and loss functions.

Example usage:
    # Train with position_dependent_gaussian dataset, pixel representation, and focal_bce loss
    python train.py --data position_dependent_gaussian --repr pixel --loss focal_bce

    # Train with geom_simple dataset, heat representation, and kl loss
    python train.py --data geom_simple --repr heat --loss kl --sigma 1.5

    # Train with position_dependent_gaussian dataset, coord representation, and polar_decoupled loss
    python train.py --data position_dependent_gaussian --repr coord --loss polar_decoupled --w_dir 1.0 --w_mag 1.0
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from tqdm import tqdm

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the refactored modules
from experiments_refactored.datasets import (
    create_position_dependent_gaussian_loaders,
    create_geom_simple_loaders,
    create_sparse_representation,
    create_dense_gaussian_representation,
    calculate_displacement_vectors
)

from experiments_refactored.models import create_model
from experiments_refactored.losses import get_loss

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train position-dependent Gaussian prediction models')
    
    # Dataset arguments
    parser.add_argument('--data', type=str, default='eye_gauss',
                        choices=['eye_gauss', 'geom_simple'],
                        help='Dataset to use')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the data directory')
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate new data instead of loading from files')
    
    # Representation arguments
    parser.add_argument('--repr', type=str, default='pixel',
                        choices=['pixel', 'heat', 'coord'],
                        help='Representation type')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Sigma for Gaussian heatmap representation')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Image size for pixel and heatmap representations')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simvp',
                        choices=['simvp'],
                        help='Model type')
    parser.add_argument('--hid_S', type=int, default=64,
                        help='Hidden dimension of spatial encoder')
    parser.add_argument('--hid_T', type=int, default=512,
                        help='Hidden dimension of temporal encoder')
    parser.add_argument('--N_S', type=int, default=4,
                        help='Number of spatial encoder blocks')
    parser.add_argument('--N_T', type=int, default=8,
                        help='Number of temporal encoder blocks')
    parser.add_argument('--model_type', type=str, default='gSTA',
                        choices=['gSTA', 'IncepU', 'ConvNeXt', 'Uniformer', 'ViT', 'Swin', 'MLP'],
                        help='Type of SimVP model')
    
    # Loss arguments
    parser.add_argument('--loss', type=str, default=None,
                        help='Loss function to use (if None, uses default for representation)')
    parser.add_argument('--alpha', type=float, default=0.25,
                        help='Alpha parameter for focal losses')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Gamma parameter for focal losses')
    parser.add_argument('--pos_weight', type=float, default=1000.0,
                        help='Positive class weight for weighted BCE loss')
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss')
    parser.add_argument('--w_dir', type=float, default=1.0,
                        help='Direction weight for polar decoupled loss')
    parser.add_argument('--w_mag', type=float, default=1.0,
                        help='Magnitude weight for polar decoupled loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, or cpu)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency (in epochs)')
    
    args = parser.parse_args()
    
    # Set default device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    # Set default loss function if not specified
    if args.loss is None:
        if args.repr == 'pixel':
            args.loss = 'focal_bce'
        elif args.repr == 'heat':
            args.loss = 'kl'
        elif args.repr == 'coord':
            args.loss = 'huber'
    
    # Set default experiment name if not specified
    if args.exp_name is None:
        args.exp_name = f"{args.data}_{args.repr}_{args.loss}"
    
    return args

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss doesn't improve.
    """
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state_dict = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def restore_best(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

def prepare_batch_for_task(batch, args, device):
    """
    Prepare a batch for the specified task.
    
    Args:
        batch (dict): Batch from the dataloader.
        args (argparse.Namespace): Command line arguments.
        device (torch.device): Device to use.
        
    Returns:
        tuple: (input_tensor, target_tensor)
    """
    # Extract coordinates and fixation mask
    coordinates = batch['coordinates'].to(device)
    # Handle both 'mask' and 'fixation_mask' keys for compatibility
    if 'fixation_mask' in batch:
        fixation_mask = batch['fixation_mask'].to(device)
    else:
        fixation_mask = batch['mask'].to(device)
    
    # Determine input and target sequence lengths
    seq_len = coordinates.shape[1]
    input_len = seq_len // 2
    
    # Split into input and target sequences
    input_coords = coordinates[:, :input_len]
    input_mask = fixation_mask[:, :input_len]
    target_coords = coordinates[:, input_len:]
    target_mask = fixation_mask[:, input_len:]
    
    # Process based on representation type
    if args.repr == 'pixel':
        # Convert to sparse binary frames
        input_frames = create_sparse_representation(input_coords, input_mask, args.img_size)
        target_frames = create_sparse_representation(target_coords, target_mask, args.img_size)
        
        # Add channel dimension if needed and move to device
        input_tensor = input_frames.to(device)
        target_tensor = target_frames.to(device)
        
    elif args.repr == 'heat':
        # Convert to dense Gaussian heatmap frames
        input_frames = create_dense_gaussian_representation(
            input_coords, input_mask, args.img_size, args.sigma
        )
        target_frames = create_dense_gaussian_representation(
            target_coords, target_mask, args.img_size, args.sigma
        )
        
        # Add channel dimension if needed and move to device
        input_tensor = input_frames.to(device)
        target_tensor = target_frames.to(device)
        
    elif args.repr == 'coord':
        # Calculate displacement vectors for the target sequence
        # We'll use the last valid input coordinate as the reference point
        
        # Find the last valid input coordinate for each batch item
        last_valid_idx = torch.zeros(input_coords.shape[0], dtype=torch.long, device=device)
        for b in range(input_coords.shape[0]):
            valid_indices = torch.where(input_mask[b])[0]
            if len(valid_indices) > 0:
                last_valid_idx[b] = valid_indices[-1]
        
        # Get the last valid input coordinates
        batch_indices = torch.arange(input_coords.shape[0], device=device)
        last_valid_coords = input_coords[batch_indices, last_valid_idx]
        
        # Calculate displacements from the last valid input coordinate
        target_displacements = torch.zeros_like(target_coords)
        for b in range(target_coords.shape[0]):
            for t in range(target_coords.shape[1]):
                if target_mask[b, t]:
                    target_displacements[b, t] = target_coords[b, t] - last_valid_coords[b]
        
        # Create target tensor (only for the first valid displacement)
        target_tensor = torch.zeros(target_coords.shape[0], 2, device=device)
        for b in range(target_coords.shape[0]):
            valid_indices = torch.where(target_mask[b])[0]
            if len(valid_indices) > 0:
                target_tensor[b] = target_displacements[b, valid_indices[0]]
        
        # Use input frames for the model input
        input_tensor = create_sparse_representation(input_coords, input_mask, args.img_size).to(device)
    
    else:
        raise ValueError(f"Unknown representation: {args.repr}")
    
    return input_tensor, target_tensor

def calculate_metrics(pred, target, args):
    """
    Calculate task-specific metrics.
    
    Args:
        pred (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth targets.
        args (argparse.Namespace): Command line arguments.
        
    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {}
    
    if args.repr in ['pixel', 'heat']:
        # For pixel and heatmap representations, calculate MSE
        mse = torch.mean((pred - target) ** 2).item()
        metrics['mse'] = mse
        
        if args.repr == 'pixel':
            # For pixel representation, calculate binary metrics
            pred_binary = (pred > 0.5).float()
            target_binary = (target > 0.5).float()
            
            # Calculate true positives, false positives, false negatives
            tp = torch.sum(pred_binary * target_binary).item()
            fp = torch.sum(pred_binary * (1 - target_binary)).item()
            fn = torch.sum((1 - pred_binary) * target_binary).item()
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
    
    elif args.repr == 'coord':
        # For coordinate representation, calculate displacement metrics
        
        # Calculate L2 distance (pixel error)
        pixel_error = torch.norm(pred - target, dim=1).mean().item()
        metrics['pixel_error'] = pixel_error
        
        # Calculate velocity ratio
        pred_magnitude = torch.norm(pred, dim=1)
        target_magnitude = torch.norm(target, dim=1)
        
        # Handle zero magnitudes
        valid_mask = (target_magnitude > 0) & (pred_magnitude > 0)
        if valid_mask.any():
            velocity_ratio = (pred_magnitude[valid_mask] / target_magnitude[valid_mask]).mean().item()
            metrics['velocity_ratio'] = velocity_ratio
        
        # Calculate direction cosine similarity
        valid_mask = (target_magnitude > 0) & (pred_magnitude > 0)
        if valid_mask.any():
            pred_normalized = pred[valid_mask] / pred_magnitude[valid_mask].unsqueeze(1)
            target_normalized = target[valid_mask] / target_magnitude[valid_mask].unsqueeze(1)
            
            direction_cos = torch.sum(pred_normalized * target_normalized, dim=1).mean().item()
            metrics['direction_cos'] = direction_cos
    
    return metrics

def train_epoch(model, train_loader, criterion, optimizer, args, device):
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        args (argparse.Namespace): Command line arguments.
        device (torch.device): Device to use.
        
    Returns:
        dict: Dictionary of training metrics.
    """
    model.train()
    epoch_loss = 0.0
    epoch_metrics = {}
    
    for batch in tqdm(train_loader, desc='Training', leave=False):
        # Prepare batch
        input_tensor, target_tensor = prepare_batch_for_task(batch, args, device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_tensor)
        
        # Calculate loss
        loss = criterion(output, target_tensor)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item() * input_tensor.size(0)
        batch_metrics = calculate_metrics(output.detach(), target_tensor.detach(), args)
        
        # Accumulate metrics
        for k, v in batch_metrics.items():
            if k not in epoch_metrics:
                epoch_metrics[k] = 0.0
            epoch_metrics[k] += v * input_tensor.size(0)
    
    # Calculate average metrics
    dataset_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * args.batch_size
    epoch_loss /= dataset_size
    for k in epoch_metrics:
        epoch_metrics[k] /= dataset_size
    
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics

def validate_epoch(model, val_loader, criterion, args, device):
    """
    Validate for one epoch.
    
    Args:
        model (nn.Module): Model to validate.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        args (argparse.Namespace): Command line arguments.
        device (torch.device): Device to use.
        
    Returns:
        dict: Dictionary of validation metrics.
    """
    model.eval()
    epoch_loss = 0.0
    epoch_metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', leave=False):
            # Prepare batch
            input_tensor, target_tensor = prepare_batch_for_task(batch, args, device)
            
            # Forward pass
            output = model(input_tensor)
            
            # Calculate loss
            loss = criterion(output, target_tensor)
            
            # Update metrics
            epoch_loss += loss.item() * input_tensor.size(0)
            batch_metrics = calculate_metrics(output, target_tensor, args)
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += v * input_tensor.size(0)
    
    # Calculate average metrics
    dataset_size = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(val_loader) * args.batch_size
    epoch_loss /= dataset_size
    for k in epoch_metrics:
        epoch_metrics[k] /= dataset_size
    
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics

def save_checkpoint(model, optimizer, epoch, metrics, args, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save.
        optimizer (optim.Optimizer): Optimizer to save.
        epoch (int): Current epoch.
        metrics (dict): Dictionary of metrics.
        args (argparse.Namespace): Command line arguments.
        is_best (bool): Whether this is the best model so far.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': args
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
    
    # Save periodic checkpoint
    if epoch % args.save_freq == 0:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, output_dir / 'best_checkpoint.pth')

def plot_metrics(train_metrics, val_metrics, args):
    """
    Plot training and validation metrics.
    
    Args:
        train_metrics (list): List of training metrics dictionaries.
        val_metrics (list): List of validation metrics dictionaries.
        args (argparse.Namespace): Command line arguments.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract epochs
    epochs = list(range(1, len(train_metrics) + 1))
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='Training Loss')
    plt.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({args.exp_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'loss.png')
    plt.close()
    
    # Plot other metrics
    all_metrics = set()
    for m in train_metrics + val_metrics:
        all_metrics.update(m.keys())
    
    all_metrics.remove('loss')  # Loss is already plotted
    
    for metric in all_metrics:
        plt.figure(figsize=(10, 6))
        
        if all(metric in m for m in train_metrics):
            plt.plot(epochs, [m[metric] for m in train_metrics], 'b-', label=f'Training {metric}')
        
        if all(metric in m for m in val_metrics):
            plt.plot(epochs, [m[metric] for m in val_metrics], 'r-', label=f'Validation {metric}')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Training and Validation {metric} ({args.exp_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f'{metric}.png')
        plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    if args.data == 'eye_gauss':
        # Create configuration for data generation
        data_config = {
            'grid_size': 32,
            'sigma': 4.5,
            'displacement_scale': 9.0,
            'center_bias_strength': 0.55,
            'random_exploration_scale': 0.65,
            'short_step_ratio': 0.4,
            'short_step_cov_range': (0.01, 0.15),
            'long_step_cov_range': (10.0, 20.0),
            'gaussian_sigma': args.sigma
        }
        
        # Set default data path for eye_gauss if not provided
        eye_gauss_data_path = args.data_path
        if eye_gauss_data_path is None:
            eye_gauss_data_path = 'experiments_refactored/data/position_dependent_gaussian'
            
        # For eye_gauss, always generate if data files don't exist
        generate_data = args.generate_data
        if not generate_data:
            # Check if data files exist, if not, generate them
            data_file_path = Path(eye_gauss_data_path) / "train_gaussian_data.pt"
            if not data_file_path.exists():
                print(f"Data files not found at {eye_gauss_data_path}, generating new data...")
                generate_data = True
        
        train_loader, val_loader, test_loader = create_position_dependent_gaussian_loaders(
            batch_size=args.batch_size,
            data_path=eye_gauss_data_path,
            generate=generate_data,
            config=data_config
        )
    elif args.data == 'geom_simple':
        train_loader, val_loader, test_loader = create_geom_simple_loaders(
            batch_size=args.batch_size,
            num_samples=1000,
            img_size=args.img_size,
            sequence_length=20
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    
    print(f"Created data loaders: {args.data}")
    try:
        print(f"  Train: {len(train_loader.dataset)} samples")
    except (AttributeError, TypeError):
        print(f"  Train: unknown samples")
    try:
        print(f"  Validation: {len(val_loader.dataset)} samples")
    except (AttributeError, TypeError):
        print(f"  Validation: unknown samples")
    try:
        print(f"  Test: {len(test_loader.dataset)} samples")
    except (AttributeError, TypeError):
        print(f"  Test: unknown samples")
    
    # Create model
    if args.model == 'simvp':
        # Determine input shape based on representation
        in_shape = (8, 1, args.img_size, args.img_size)  # (T, C, H, W)
        
        model = create_model(
            task=args.repr,
            in_shape=in_shape,
            hid_S=args.hid_S,
            hid_T=args.hid_T,
            N_S=args.N_S,
            N_T=args.N_T,
            model_type=args.model_type
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"Created model: {args.model} for {args.repr} representation")
    
    # Create loss function
    loss_kwargs = {}
    
    if args.loss == 'focal_bce':
        loss_kwargs = {'alpha': args.alpha, 'gamma': args.gamma}
    elif args.loss == 'weighted_bce':
        loss_kwargs = {'pos_weight': args.pos_weight}
    elif args.loss == 'focal_tversky':
        loss_kwargs = {'alpha': args.alpha, 'gamma': args.gamma}
    elif args.loss == 'huber':
        loss_kwargs = {'delta': args.delta}
    elif args.loss == 'polar_decoupled':
        loss_kwargs = {'direction_weight': args.w_dir, 'magnitude_weight': args.w_mag}
    
    criterion = get_loss(args.loss, **loss_kwargs)
    print(f"Created loss function: {args.loss}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Created optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    print(f"Created early stopping (patience={args.patience})")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    
    train_metrics_history = []
    val_metrics_history = []
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train and validate
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args, device)
        val_metrics = validate_epoch(model, val_loader, criterion, args, device)
        
        # Update metrics history
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Check if this is the best model
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_metrics, args, is_best)
        
        # Plot metrics
        plot_metrics(train_metrics_history, val_metrics_history, args)
        
        # Print metrics
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.2f}s):")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        
        # Print other metrics
        for k in train_metrics:
            if k != 'loss':
                print(f"  Train {k}: {train_metrics[k]:.6f}")
        
        for k in val_metrics:
            if k != 'loss':
                print(f"  Val {k}: {val_metrics[k]:.6f}")
        
        # Check early stopping
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Restore best model
    early_stopping.restore_best(model)
    
    # Evaluate on test set
    test_metrics = validate_epoch(model, test_loader, criterion, args, device)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save test metrics
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f"Training complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 