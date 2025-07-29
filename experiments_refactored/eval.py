#!/usr/bin/env python3
"""
Evaluation script for position-dependent Gaussian prediction experiments.

This script loads trained models and evaluates them on test data,
generating metrics and visualizations.

Example usage:
    # Evaluate a model checkpoint
    python eval.py --checkpoint results/position_dependent_gaussian_pixel_focal_bce/best_checkpoint.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import imageio

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
    parser = argparse.ArgumentParser(description='Evaluate position-dependent Gaussian prediction models')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for evaluation results (default: same as checkpoint)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, or cpu)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate new data instead of loading from files')
    
    args = parser.parse_args()
    
    # Set default device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent / 'evaluation')
    
    return args

def load_checkpoint(checkpoint_path, device):
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.
        
    Returns:
        tuple: (model, checkpoint_data)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract training arguments
    args = checkpoint['args']
    
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Model: {args.model} for {args.repr} representation")
    
    return model, checkpoint

def prepare_batch_for_task(batch, args, device):
    """
    Prepare a batch for the specified task.
    
    Args:
        batch (dict): Batch from the dataloader.
        args (argparse.Namespace): Command line arguments.
        device (torch.device): Device to use.
        
    Returns:
        tuple: (input_tensor, target_tensor, extra_data)
    """
    # Extract coordinates and fixation mask
    coordinates = batch['coordinates'].to(device)
    fixation_mask = batch['fixation_mask'].to(device)
    
    # Determine input and target sequence lengths
    seq_len = coordinates.shape[1]
    input_len = seq_len // 2
    
    # Split into input and target sequences
    input_coords = coordinates[:, :input_len]
    input_mask = fixation_mask[:, :input_len]
    target_coords = coordinates[:, input_len:]
    target_mask = fixation_mask[:, input_len:]
    
    # Extra data to return
    extra_data = {
        'input_coords': input_coords.clone(),
        'input_mask': input_mask.clone(),
        'target_coords': target_coords.clone(),
        'target_mask': target_mask.clone()
    }
    
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
        
        # Add last valid coordinates to extra data
        extra_data['last_valid_coords'] = last_valid_coords
        extra_data['target_displacements'] = target_displacements
    
    else:
        raise ValueError(f"Unknown representation: {args.repr}")
    
    return input_tensor, target_tensor, extra_data

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
            
            # Calculate velocity ratio histogram data
            ratios = (pred_magnitude[valid_mask] / target_magnitude[valid_mask]).cpu().numpy()
            metrics['velocity_ratio_data'] = ratios.tolist()
        
        # Calculate direction cosine similarity
        valid_mask = (target_magnitude > 0) & (pred_magnitude > 0)
        if valid_mask.any():
            pred_normalized = pred[valid_mask] / pred_magnitude[valid_mask].unsqueeze(1)
            target_normalized = target[valid_mask] / target_magnitude[valid_mask].unsqueeze(1)
            
            direction_cos = torch.sum(pred_normalized * target_normalized, dim=1).mean().item()
            metrics['direction_cos'] = direction_cos
            
            # Calculate direction cosine histogram data
            cosines = torch.sum(pred_normalized * target_normalized, dim=1).cpu().numpy()
            metrics['direction_cos_data'] = cosines.tolist()
    
    return metrics

def evaluate_model(model, data_loader, args, device):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model (nn.Module): Model to evaluate.
        data_loader (DataLoader): Data loader.
        args (argparse.Namespace): Command line arguments.
        device (torch.device): Device to use.
        
    Returns:
        tuple: (metrics, predictions, targets, extra_data)
    """
    model.eval()
    all_metrics = {}
    all_predictions = []
    all_targets = []
    all_extra_data = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Prepare batch
            input_tensor, target_tensor, extra_data = prepare_batch_for_task(batch, args, device)
            
            # Forward pass
            output = model(input_tensor)
            
            # Calculate metrics
            batch_metrics = calculate_metrics(output, target_tensor, args)
            
            # Store results
            all_predictions.append(output.cpu())
            all_targets.append(target_tensor.cpu())
            all_extra_data.append(extra_data)
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                if not isinstance(v, list):
                    all_metrics[k].append(v)
                else:
                    all_metrics[k].extend(v)
    
    # Calculate average metrics
    avg_metrics = {}
    for k, v in all_metrics.items():
        if k.endswith('_data'):
            avg_metrics[k] = v
        else:
            avg_metrics[k] = np.mean(v)
    
    return avg_metrics, all_predictions, all_targets, all_extra_data

def visualize_pixel_predictions(predictions, targets, extra_data, args, output_dir):
    """
    Visualize pixel-based predictions.
    
    Args:
        predictions (list): List of prediction tensors.
        targets (list): List of target tensors.
        extra_data (list): List of extra data dictionaries.
        args (argparse.Namespace): Command line arguments.
        output_dir (str): Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Concatenate predictions and targets
    all_preds = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)
    
    # Select samples to visualize
    num_samples = min(args.num_samples, all_preds.shape[0])
    sample_indices = np.random.choice(all_preds.shape[0], num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Get sample data
        pred = all_preds[idx]
        target = all_targets[idx]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot target
        axes[0].imshow(target[0], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Target')
        axes[0].axis('off')
        
        # Plot prediction
        axes[1].imshow(pred[0], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{i+1}_pixel.png', dpi=150)
        plt.close()

def visualize_heatmap_predictions(predictions, targets, extra_data, args, output_dir):
    """
    Visualize heatmap-based predictions.
    
    Args:
        predictions (list): List of prediction tensors.
        targets (list): List of target tensors.
        extra_data (list): List of extra data dictionaries.
        args (argparse.Namespace): Command line arguments.
        output_dir (str): Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Concatenate predictions and targets
    all_preds = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)
    
    # Select samples to visualize
    num_samples = min(args.num_samples, all_preds.shape[0])
    sample_indices = np.random.choice(all_preds.shape[0], num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Get sample data
        pred = all_preds[idx]
        target = all_targets[idx]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot target
        axes[0].imshow(target[0], cmap='viridis')
        axes[0].set_title('Target')
        axes[0].axis('off')
        
        # Plot prediction
        axes[1].imshow(pred[0], cmap='viridis')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{i+1}_heatmap.png', dpi=150)
        plt.close()
        
        # Extract coordinates using argmax
        if hasattr(model, 'extract_coordinates_from_heatmap'):
            # Extract coordinates from target and prediction
            target_coords = model.extract_coordinates_from_heatmap(target.unsqueeze(0).to(device))
            pred_coords = model.extract_coordinates_from_heatmap(pred.unsqueeze(0).to(device))
            
            # Create visualization with coordinates
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot target with coordinate
            axes[0].imshow(target[0], cmap='viridis')
            axes[0].plot(target_coords[0, 0].item(), target_coords[0, 1].item(), 'ro', markersize=8)
            axes[0].set_title('Target')
            axes[0].axis('off')
            
            # Plot prediction with coordinate
            axes[1].imshow(pred[0], cmap='viridis')
            axes[1].plot(pred_coords[0, 0].item(), pred_coords[0, 1].item(), 'ro', markersize=8)
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(output_dir / f'sample_{i+1}_heatmap_coords.png', dpi=150)
            plt.close()

def visualize_coord_predictions(predictions, targets, extra_data, args, output_dir):
    """
    Visualize coordinate-based predictions.
    
    Args:
        predictions (list): List of prediction tensors.
        targets (list): List of target tensors.
        extra_data (list): List of extra data dictionaries.
        args (argparse.Namespace): Command line arguments.
        output_dir (str): Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Concatenate predictions and targets
    all_preds = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)
    
    # Collect all extra data
    all_extra = {}
    for k in extra_data[0].keys():
        all_extra[k] = torch.cat([d[k] for d in extra_data], dim=0)
    
    # Select samples to visualize
    num_samples = min(args.num_samples, all_preds.shape[0])
    sample_indices = np.random.choice(all_preds.shape[0], num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Get sample data
        pred = all_preds[idx]
        target = all_targets[idx]
        last_valid_coord = all_extra['last_valid_coords'][idx]
        
        # Calculate predicted and target coordinates
        pred_coord = last_valid_coord + pred
        target_coord = last_valid_coord + target
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot input coordinates
        input_coords = all_extra['input_coords'][idx]
        input_mask = all_extra['input_mask'][idx]
        valid_input_coords = input_coords[input_mask]
        
        if len(valid_input_coords) > 0:
            ax.plot(valid_input_coords[:, 0], valid_input_coords[:, 1], 'b-o', label='Input', alpha=0.5)
        
        # Plot target and prediction
        ax.plot([last_valid_coord[0], target_coord[0]], [last_valid_coord[1], target_coord[1]], 'g-o', label='Target')
        ax.plot([last_valid_coord[0], pred_coord[0]], [last_valid_coord[1], pred_coord[1]], 'r-o', label='Prediction')
        
        # Add arrows
        ax.arrow(last_valid_coord[0], last_valid_coord[1], target[0], target[1], color='g', width=0.1, head_width=0.5, alpha=0.7)
        ax.arrow(last_valid_coord[0], last_valid_coord[1], pred[0], pred[1], color='r', width=0.1, head_width=0.5, alpha=0.7)
        
        # Set limits
        ax.set_xlim(0, args.img_size)
        ax.set_ylim(0, args.img_size)
        
        # Add labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{i+1}_coord.png', dpi=150)
        plt.close()

def plot_velocity_ratio_histogram(metrics, output_dir):
    """
    Plot velocity ratio histogram.
    
    Args:
        metrics (dict): Dictionary of metrics.
        output_dir (str): Output directory.
    """
    if 'velocity_ratio_data' not in metrics:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get velocity ratio data
    velocity_ratios = metrics['velocity_ratio_data']
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(velocity_ratios, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=1.0, color='r', linestyle='--', label='Ideal Ratio (1.0)')
    plt.axvline(x=np.mean(velocity_ratios), color='g', linestyle='-', label=f'Mean Ratio ({np.mean(velocity_ratios):.3f})')
    
    # Add labels and title
    plt.xlabel('Velocity Ratio (Predicted / Target)')
    plt.ylabel('Frequency')
    plt.title('Velocity Ratio Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / 'velocity_ratio_histogram.png', dpi=150)
    plt.close()

def plot_direction_cosine_histogram(metrics, output_dir):
    """
    Plot direction cosine histogram.
    
    Args:
        metrics (dict): Dictionary of metrics.
        output_dir (str): Output directory.
    """
    if 'direction_cos_data' not in metrics:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get direction cosine data
    direction_cosines = metrics['direction_cos_data']
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(direction_cosines, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=1.0, color='r', linestyle='--', label='Perfect Alignment (1.0)')
    plt.axvline(x=np.mean(direction_cosines), color='g', linestyle='-', label=f'Mean Cosine ({np.mean(direction_cosines):.3f})')
    
    # Add labels and title
    plt.xlabel('Direction Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Direction Cosine Similarity Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / 'direction_cosine_histogram.png', dpi=150)
    plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_args = checkpoint['args']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    # Create data loaders
    if checkpoint_args.data == 'eye_gauss':
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
            'gaussian_sigma': checkpoint_args.sigma
        }
        
        # Set default data path for eye_gauss if not provided
        eye_gauss_data_path = checkpoint_args.data_path if hasattr(checkpoint_args, 'data_path') else None
        if eye_gauss_data_path is None:
            eye_gauss_data_path = 'experiments_refactored/data/position_dependent_gaussian'
            
        _, _, test_loader = create_position_dependent_gaussian_loaders(
            batch_size=32,
            data_path=eye_gauss_data_path,
            generate=args.generate_data,
            config=data_config
        )
    elif checkpoint_args.data == 'geom_simple':
        _, _, test_loader = create_geom_simple_loaders(
            batch_size=32,
            num_samples=1000,
            img_size=checkpoint_args.img_size,
            sequence_length=20
        )
    else:
        raise ValueError(f"Unknown dataset: {checkpoint_args.data}")
    
    print(f"Created test data loader: {checkpoint_args.data}")
    try:
        print(f"  Test: {len(test_loader.dataset)} samples")
    except (AttributeError, TypeError):
        print(f"  Test: unknown samples")
    
    # Evaluate model
    metrics, predictions, targets, extra_data = evaluate_model(model, test_loader, checkpoint_args, device)
    
    # Print metrics
    print("Test metrics:")
    for k, v in metrics.items():
        if not k.endswith('_data'):
            print(f"  {k}: {v:.6f}")
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                serializable_metrics[k] = [x.tolist() for x in v]
            else:
                serializable_metrics[k] = v
        
        json.dump(serializable_metrics, f, indent=4)
    
    # Visualize predictions
    if checkpoint_args.repr == 'pixel':
        visualize_pixel_predictions(predictions, targets, extra_data, checkpoint_args, output_dir / 'visualizations')
    elif checkpoint_args.repr == 'heat':
        visualize_heatmap_predictions(predictions, targets, extra_data, checkpoint_args, output_dir / 'visualizations')
    elif checkpoint_args.repr == 'coord':
        visualize_coord_predictions(predictions, targets, extra_data, checkpoint_args, output_dir / 'visualizations')
        
        # Plot velocity ratio and direction cosine histograms
        plot_velocity_ratio_histogram(metrics, output_dir / 'visualizations')
        plot_direction_cosine_histogram(metrics, output_dir / 'visualizations')
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 