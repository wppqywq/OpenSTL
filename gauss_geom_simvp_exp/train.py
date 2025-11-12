#!/usr/bin/env python3
"""
Unified training script for position-dependent Gaussian prediction experiments.

This script uses a config-based approach for experiments. 

Example usage:
    # Train with a predefined experiment
    python train.py --exp gauss_pixel_focal
    
    # List available experiments
    python train.py --list
    
    # Override config parameters
    python train.py --exp gauss_pixel_focal --epochs 100 --lr 0.002
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
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the refactored modules
from experiments_refactored.datasets import (
    create_data_loaders,
    create_unified_geom_loaders,
    create_sparse_representation,
    create_gaussian_representation,
    calculate_displacement_vectors
)
from experiments_refactored.experiment_config import get_experiment_config, list_experiments
from experiments_refactored.models import create_model
from experiments_refactored.losses import get_loss

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train position-dependent Gaussian prediction models')
    
    # Main experiment selector
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment name (see config.py for available experiments)')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments and exit')
    
    # Essential override arguments
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda, mps, or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    
    args = parser.parse_args()
    
    # Handle list experiments
    if args.list:
        print("Available experiments:")
        for exp_name in list_experiments():
            print(f"  {exp_name}")
        sys.exit(0)
    
    # Require experiment name
    if args.exp is None:
        print("Error: Must specify experiment name with --exp")
        print("Use --list to see available experiments")
        sys.exit(1)
    
    # Load experiment config
    try:
        config = get_experiment_config(args.exp)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list to see available experiments")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['lr'] = args.lr
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.device is not None:
        config['device'] = args.device
    if args.resume is not None:
        config['resume'] = args.resume
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    
    # Set default device if not specified
    if 'device' not in config or config['device'] is None:
        if torch.cuda.is_available():
            config['device'] = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config['device'] = 'mps'
        else:
            config['device'] = 'cpu'
    
    # Calculate sequence lengths from total length
    config['pre_seq_length'] = int(config['total_seq_length'] * config['split_ratio'])
    config['aft_seq_length'] = config['total_seq_length'] - config['pre_seq_length']
    
    # Convert to namespace for backward compatibility
    from types import SimpleNamespace
    return SimpleNamespace(**config)

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
    Prepare a batch for the specified task. This version is aligned for seq-to-seq prediction.
    """
    # Handle different batch formats
    if isinstance(batch, dict):
        coordinates = batch['coordinates'].to(device)
        mask_data = batch.get('fixation_mask', batch.get('mask'))
        fixation_mask = mask_data.to(device) if mask_data is not None else None
        frames = batch.get('frames', None)
    else:
        # Handle tuple format (frames, coordinates, mask)
        frames, coordinates, fixation_mask = batch
        coordinates = coordinates.to(device)
        fixation_mask = fixation_mask.to(device)
        if frames is not None:
            frames = frames.to(device)
    
    # Create default mask if None
    if fixation_mask is None:
        fixation_mask = torch.ones(coordinates.shape[0], coordinates.shape[1], dtype=torch.bool, device=device)
    
    total_len = args.pre_seq_length + args.aft_seq_length
    if coordinates.shape[1] < total_len:
        # Pad sequences that are too short
        padding_len = total_len - coordinates.shape[1]
        padding_coords = torch.zeros(coordinates.shape[0], padding_len, 2, device=device)
        padding_mask = torch.zeros(coordinates.shape[0], padding_len, dtype=torch.bool, device=device)
        coordinates = torch.cat([coordinates, padding_coords], dim=1)
        fixation_mask = torch.cat([fixation_mask, padding_mask], dim=1)

    input_coords = coordinates[:, :args.pre_seq_length]
    target_coords = coordinates[:, args.pre_seq_length:total_len]
    input_mask = fixation_mask[:, :args.pre_seq_length]
    target_mask = fixation_mask[:, args.pre_seq_length:total_len]

    if args.repr == 'pixel':
        # For pixel representation, always create sparse frames
        input_tensor = create_sparse_representation(input_coords, input_mask, args.img_size).to(device)
        target_tensor = create_sparse_representation(target_coords, target_mask, args.img_size).to(device)
    elif args.repr == 'heat':
        # For heat representation, always create gaussian frames
        input_tensor = create_gaussian_representation(input_coords, input_mask, args.img_size, args.sigma).to(device)
        target_tensor = create_gaussian_representation(target_coords, target_mask, args.img_size, args.sigma).to(device)
    elif args.repr == 'coord':
        # For coord representation, use sparse frames as input (model expects frames)
        input_tensor = create_sparse_representation(input_coords, input_mask, args.img_size).to(device)
        if getattr(args, 'coord_mode', 'absolute') == 'displacement':
            # Predict displacement relative to last input coordinate
            last_coord = input_coords[:, -1:].detach()
            target_tensor = target_coords - last_coord  # (B, T_out, 2)
            input_tensor = create_sparse_representation(input_coords, input_mask, args.img_size).to(device)
            # also return last_coord for conditioning (B,2)
            extra = last_coord.squeeze(1)
            return input_tensor, target_tensor, extra
        else:
            # Predict absolute coordinates
            target_tensor = target_coords
    else:
        raise ValueError(f"Unknown representation: {args.repr}")
        
    return input_tensor, target_tensor

def calculate_metrics(pred, target, args):
    """
    Calculate task-specific metrics for seq-to-seq tasks.
    """
    metrics = {}
    
    if args.repr in ['pixel', 'heat']:
        mse = torch.mean((pred - target) ** 2).item()
        metrics['mse'] = mse
        
        if args.repr == 'pixel':
            pred_binary = (pred > 0.5).float()
            target_binary = (target > 0.5).float()
            tp = torch.sum(pred_binary * target_binary).item()
            fp = torch.sum(pred_binary * (1 - target_binary)).item()
            fn = torch.sum((1 - pred_binary) * target_binary).item()
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
    
    elif args.repr == 'coord':
        # pred and target shape: (B, T_out, 2)
        # Calculate L2 distance for each time step, then average
        error = torch.norm(pred - target, dim=-1) # Shape: (B, T_out)
        
        # Mean Displacement Error (MDE)
        mde = error.mean().item()
        metrics['mde'] = mde

        # FDE removed per user request

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
        prep = prepare_batch_for_task(batch, args, device)
        if args.repr == 'coord' and getattr(args, 'coord_mode', 'absolute') == 'displacement' and isinstance(prep, tuple) and len(prep) == 3:
            input_tensor, target_tensor, last_coord = prep
        else:
            input_tensor, target_tensor = prep  # type: ignore[misc]
            last_coord = None
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_tensor, last_coord) if last_coord is not None else model(input_tensor)
        
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
            prep = prepare_batch_for_task(batch, args, device)
            if args.repr == 'coord' and getattr(args, 'coord_mode', 'absolute') == 'displacement' and isinstance(prep, tuple) and len(prep) == 3:
                input_tensor, target_tensor, last_coord = prep
            else:
                input_tensor, target_tensor = prep  # type: ignore[misc]
                last_coord = None
            
            # Forward pass
            output = model(input_tensor, last_coord) if last_coord is not None else model(input_tensor)
            
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

def create_experiment_manifest(args, epoch, metrics):
    """
    Create a lightweight JSON manifest with experiment metadata.
    
    Args:
        args: Command line arguments
        epoch: Current epoch
        metrics: Training metrics
        
    Returns:
        dict: Experiment manifest
    """
    import time
    import subprocess
    
    # Get git commit hash if available
    git_hash = "unknown"
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            git_hash = result.stdout.strip()[:8]  # Short hash
    except:
        pass
    
    manifest = {
        "experiment_name": args.exp_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
        "dataset": args.data,
        "representation": args.repr,
        "loss_function": args.loss,
        "model_config": {
            "hid_S": args.hid_S,
            "hid_T": args.hid_T,
            "N_S": args.N_S,
            "N_T": args.N_T,
            "model_type": args.model_type
        },
        "training_config": {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "weight_decay": args.weight_decay
        },
        "data_config": {
            "img_size": args.img_size,
            "pre_seq_length": args.pre_seq_length,
            "aft_seq_length": args.aft_seq_length,
            "total_seq_length": args.total_seq_length
        },
        "final_epoch": epoch,
        "final_metrics": metrics
    }
    
    # Add loss-specific parameters
    loss_params = {}
    if hasattr(args, 'sigma') and args.sigma is not None:
        loss_params['sigma'] = args.sigma
    if hasattr(args, 'gamma') and args.gamma is not None:
        loss_params['gamma'] = args.gamma
    if hasattr(args, 'pos_weight') and args.pos_weight is not None:
        loss_params['pos_weight'] = args.pos_weight
    if hasattr(args, 'w_dir') and args.w_dir is not None:
        loss_params['w_dir'] = args.w_dir
    if hasattr(args, 'w_mag') and args.w_mag is not None:
        loss_params['w_mag'] = args.w_mag
    if hasattr(args, 'cosine_weight') and args.cosine_weight is not None:
        loss_params['cosine_weight'] = args.cosine_weight
    
    if loss_params:
        manifest["loss_params"] = loss_params
    
    return manifest

def save_checkpoint(model, optimizer, epoch, metrics, args, is_best=False, best_val_loss=None):
    """
    Save model checkpoint and experiment manifest.
    
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
        'args': args,
        'best_val_loss': best_val_loss if best_val_loss is not None else metrics.get('loss', float('inf'))
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
    

    
    # Save best checkpoint and manifest
    if is_best:
        torch.save(checkpoint, output_dir / 'best_checkpoint.pth')
        
        # Create and save experiment manifest
        manifest = create_experiment_manifest(args, epoch, metrics)
        with open(output_dir / 'experiment_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

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

def main(args=None):
    """Main function."""
    # Parse arguments
    if args is None:
        args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Running experiment: {args.exp_name}")
    print(f"Using device: {device}")
    print(f"Configuration: {args.data} dataset, {args.repr} representation, {args.loss} loss")
    print(f"Sequence configuration: total={args.total_seq_length}, input={args.pre_seq_length}, output={args.aft_seq_length}")
    
    # Create data loaders
    if args.data == 'gauss':
        # Determine representation for frame generation (align with geom_simple approach)
        representation = 'sparse' if args.repr == 'pixel' else ('gaussian' if args.repr == 'heat' else 'coord')
        # Build config to pass sampler settings
        gauss_cfg = {
            'use_mu_field': bool(args.use_mu_field),
            'seed': args.gauss_seed,
            'start_pos': None if args.start_x is None or args.start_y is None else (args.start_y, args.start_x)
        }
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=args.batch_size,
            representation=representation,
            generate=True,  # Always generate for consistency
            config=gauss_cfg
        )
    elif args.data == 'geom_simple':
        # Determine representation for frame generation
        representation = 'sparse' if args.repr == 'pixel' else ('gaussian' if args.repr == 'heat' else 'coord')
        train_loader, val_loader, test_loader = create_unified_geom_loaders(
            batch_size=args.batch_size,
            sequence_length=args.total_seq_length,
            representation=representation,
            num_train=args.train_size,
            num_val=args.val_size,
            num_test=args.test_size
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    
    print(f"Created data loaders: {args.data}")

    # Reconfigure DataLoaders with multiple workers if requested
    if args.num_workers > 0:
        from torch.utils.data import DataLoader  # Local import to avoid circular issues
        pin = args.device in ('cuda', 'mps')
        train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
        val_loader = DataLoader(val_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        test_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        print(f"DataLoader workers set to {args.num_workers}")
    try:
        print(f"  Train: {len(train_loader.dataset)} samples")  # type: ignore
    except (AttributeError, TypeError):
        print(f"  Train: unknown samples")
    try:
        print(f"  Validation: {len(val_loader.dataset)} samples")  # type: ignore
    except (AttributeError, TypeError):
        print(f"  Validation: unknown samples")
    try:
        print(f"  Test: {len(test_loader.dataset)} samples")  # type: ignore
    except (AttributeError, TypeError):
        print(f"  Test: unknown samples")
    
    # Create model
    if args.model == 'simvp':
        # Determine input shape based on representation
        in_shape = (args.pre_seq_length, 1, args.img_size, args.img_size)  # (T, C, H, W)
        out_shape = (args.aft_seq_length, 1, args.img_size, args.img_size)
        
        model = create_model(
            task=args.repr,
            in_shape=in_shape,
            out_shape=out_shape,
            hid_S=args.hid_S,
            hid_T=args.hid_T,
            N_S=args.N_S,
            N_T=args.N_T,
            model_type=args.model_type,
            use_gru=args.use_gru
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"Created model: {args.model} for {args.repr} representation")
    
    # Create loss function
    loss_kwargs = {'exp_name': args.exp_name}
    
    # Add loss-specific parameters from config
    loss_param_map = {
        'focal_bce': ['alpha', 'gamma'],
        'focal_tversky': ['alpha', 'gamma'],
        'huber': ['delta'],
        'polar_decoupled': [('direction_weight', 'w_dir'), ('magnitude_weight', 'w_mag')],
        'weighted_mse': ['pos_weight'],
        'weighted_bce': ['pos_weight'],
        'l1_cosine': ['cosine_weight']
    }
    
    if args.loss in loss_param_map:
        for param in loss_param_map[args.loss]:
            if isinstance(param, tuple):
                loss_key, config_key = param
                if hasattr(args, config_key):
                    loss_kwargs[loss_key] = getattr(args, config_key)
            else:
                if hasattr(args, param):
                    loss_kwargs[param] = getattr(args, param)
    
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
    
    # Resume from checkpoint if specified
    start_epoch = 1
    train_metrics_history = []
    val_metrics_history = []
    best_val_loss = float('inf')
    
    if hasattr(args, 'resume') and args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        try:
                model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Checkpoint incompatibility: {e}\nLoading state dict with strict=False (partial load).")
            _ = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    else:
        # Check for existing latest checkpoint
        latest_checkpoint = output_dir / 'latest_checkpoint.pth'
        if latest_checkpoint.exists():
            print(f"Found existing checkpoint, resuming from: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"Checkpoint incompatibility: {e}\nLoading state dict with strict=False (partial load).")
                _ = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}")
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Train model
    print(f"Starting training from epoch {start_epoch} for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs + 1):
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
        save_checkpoint(model, optimizer, epoch, val_metrics, args, is_best, best_val_loss)
        
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
    # Parse args
    args = parse_args()
    
    # Run training
    main(args)
    
    print("Training completed successfully!")
    print(f"Results saved to {args.output_dir}/{args.exp_name}") 