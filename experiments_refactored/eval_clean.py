#!/usr/bin/env python3
"""
Clean evaluation script with simplified metrics and representation-specific plotting.

Key improvements:
1. Simplified metrics: only 6 key values
2. Representation-specific plotters
3. Fixed coord model handling
4. Clean data flow without conversions
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from models.simvp_base import SimVPWithTaskHead
from datasets import create_position_dependent_gaussian_loaders, create_unified_geom_loaders as create_geom_simple_loaders
from plot_utils import create_plotter


def load_model_and_args(checkpoint_path, device):
    """Load model and arguments from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']
    
    # Create model
    in_shape = (10, 1, 32, 32)
    hid_S = getattr(model_args, 'hid_S', 16)
    hid_T = getattr(model_args, 'hid_T', 256) 
    N_S = getattr(model_args, 'N_S', 4)
    N_T = getattr(model_args, 'N_T', 4)
    model_type = getattr(model_args, 'model_type', 'gSTA')
    
    model = SimVPWithTaskHead(
        in_shape=in_shape,
        hid_S=hid_S,
        hid_T=hid_T,
        N_S=N_S,
        N_T=N_T,
        model_type=model_type,
        task=model_args.repr
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    except RuntimeError as e:
        if "coord_head" in str(e):
            print("Loading coord model without coord_head layers")
            state_dict = checkpoint['model_state_dict']
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('coord_head')}
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            raise e
    
    model.eval()
    return model, model_args


def create_test_loader(args):
    """Create test data loader."""
    if args.data == 'gauss':
        _, _, test_loader = create_position_dependent_gaussian_loaders(
            batch_size=32,
            representation='gaussian' if args.repr == 'heat' else 'sparse',
            generate=True
        )
    elif args.data == 'geom_simple':
        _, _, test_loader = create_geom_simple_loaders(
            batch_size=32,
            sequence_length=20,
            num_test=100
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    
    return test_loader


def extract_coordinates(frames, representation):
    """Extract coordinates using argmax for both pixel and heat representations.
    Returns (coords, valid_mask). A frame is invalid only when it is exactly all zeros.
    """
    B, T, C, H, W = frames.shape
    coords = torch.zeros(B, T, 2, device=frames.device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=frames.device)

    if representation in ['pixel', 'heat']:
        for b in range(B):
            for t in range(T):
                frame = frames[b, t, 0]
                if torch.count_nonzero(frame).item() == 0:
                    continue
                flat_idx = torch.argmax(frame.view(-1))
                y, x = divmod(flat_idx.item(), W)
                coords[b, t] = torch.tensor([x, y], device=frames.device)
                valid[b, t] = True

    return coords, valid


def calculate_metrics(pred_coords, true_coords):
    """Calculate simplified metrics.
    
    Note: displacement_error_X is the average L2 distance (in pixels) between 
    predicted and true coordinates for the first X frames of prediction.
    """
    B, T = pred_coords.shape[:2]
    
    # Displacement errors at different time windows
    errors_3 = []
    errors_6 = []
    errors_full = []
    
    # Velocity metrics
    magnitude_ratios = []
    cosine_similarities = []
    
    for b in range(B):
        # Calculate errors - L2 distance between predicted and true coordinates
        for t in range(T):
            error = torch.norm(pred_coords[b, t] - true_coords[b, t]).item()
            
            if t < 3:
                errors_3.append(error)
            if t < 6:
                errors_6.append(error)
            errors_full.append(error)
        
        # Calculate velocity metrics
        for t in range(T - 1):
            true_vel = true_coords[b, t+1] - true_coords[b, t]
            pred_vel = pred_coords[b, t+1] - pred_coords[b, t]
            
            true_mag = torch.norm(true_vel)
            pred_mag = torch.norm(pred_vel)
            
            if true_mag > 1e-6 and pred_mag > 1e-6:
                # Magnitude ratio
                mag_ratio = (pred_mag / true_mag).item()
                magnitude_ratios.append(mag_ratio)
                
                # Cosine similarity
                cos_sim = torch.dot(true_vel, pred_vel) / (true_mag * pred_mag)
                cosine_similarities.append(cos_sim.item())
    
    return {
        "valid": True,
        "avg_magnitude_ratio": np.mean(magnitude_ratios) if magnitude_ratios else 0.0,
        "avg_cosine_similarity": np.mean(cosine_similarities) if cosine_similarities else 0.0,
        "displacement_error_3": np.mean(errors_3) if errors_3 else 0.0,
        "displacement_error_6": np.mean(errors_6) if errors_6 else 0.0,
        "displacement_error_full": np.mean(errors_full) if errors_full else 0.0
    }


def evaluate_model(model, test_loader, device, model_args):
    """Evaluate model with clean data handling."""
    model.eval()
    
    all_pred_coords = []
    all_true_coords = []
    sample_data = {
        'input_data': [],
        'pred_data': [],
        'true_coords': [],
        'pred_coords': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Handle data loading
            if isinstance(batch, dict):
                coords = batch['coordinates'].to(device)
                frames = batch.get('frames')
                if frames is not None:
                    frames = frames.to(device)
                mask = batch.get('mask')
            else:
                # Fallback for tuple format
                frames, coords, mask = batch
                frames = frames.to(device) if frames is not None else None
                coords = coords.to(device) if coords is not None else None
            
            # Prepare input based on representation
            if model_args.repr == 'coord':
                # For coord models, we need to create dummy frames as input
                # The model still expects frame format but will use coord_head
                if frames is not None:
                    input_data = frames[:, :model_args.pre_seq_length]
                else:
                    # Create sparse frames from coordinates for input
                    B, T = coords.shape[:2]
                    input_frames = torch.zeros(B, model_args.pre_seq_length, 1, 32, 32, device=device)
                    for b in range(B):
                        for t in range(model_args.pre_seq_length):
                            x, y = coords[b, t]
                            x_int = int(torch.clamp(x, 0, 31))
                            y_int = int(torch.clamp(y, 0, 31))
                            input_frames[b, t, 0, y_int, x_int] = 1.0
                    input_data = input_frames
                target_coords = coords[:, model_args.pre_seq_length:]
            else:
                input_data = frames[:, :model_args.pre_seq_length]
                target_coords = coords[:, model_args.pre_seq_length:] if coords is not None else None
            
            # Forward pass
            pred_output = model(input_data)
            
            # Extract coordinates based on representation
            if model_args.repr == 'coord':
                if getattr(model_args, 'coord_mode', 'absolute') == 'displacement':
                    # Reconstruct absolute coordinates from predicted displacements
                    last_coord = coords[:, model_args.pre_seq_length-1].unsqueeze(1)
                    pred_coords = pred_output + last_coord
                else:
                    pred_coords = pred_output
                pred_valid = None
            else:
                pred_coords, pred_valid = extract_coordinates(pred_output, model_args.repr)
            
            # Store results
            if target_coords is not None:
                all_pred_coords.append(pred_coords.cpu())
                all_true_coords.append(target_coords.cpu())
                
                # Store samples for visualization (first 3 samples)
                if len(sample_data['input_data']) < 3:
                    sample_data['input_data'].append(input_data[0].cpu())
                    sample_data['pred_data'].append(pred_output[0].cpu())
                    sample_data['true_coords'].append(coords[0].cpu())
                    sample_data['pred_coords'].append(pred_coords[0].cpu())
                    # store valid mask for plotting decisions if available
                    if pred_valid is not None:
                        sample_data.setdefault('pred_valid_masks', []).append(pred_valid[0].cpu())
    
    # Calculate metrics
    if all_pred_coords and all_true_coords:
        all_pred_coords = torch.cat(all_pred_coords, dim=0)
        all_true_coords = torch.cat(all_true_coords, dim=0)
        metrics = calculate_metrics(all_pred_coords, all_true_coords)
    else:
        metrics = {"valid": False}
    
    return metrics, sample_data


def create_visualizations(sample_data, output_dir, model_args):
    """Create visualizations using appropriate plotter."""
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plotter based on representation
    plotter = create_plotter(model_args.repr, model_args.img_size, model_args.pre_seq_length)
    
    num_samples = min(3, len(sample_data['input_data']))
    
    for i in range(num_samples):
        if model_args.repr == 'coord':
            # For coord, only plot trajectory
            traj_path = viz_dir / f'sample_{i+1}_trajectory.png'
            plotter.plot_trajectory(
                true_coords=sample_data['true_coords'][i],
                pred_coords=sample_data['pred_coords'][i],
                sample_idx=i,
                save_path=traj_path
            )
        else:
            # For pixel/heat, plot both frames and trajectory
            grid_path = viz_dir / f'sample_{i+1}_input_output_grid.png'
            plotter.plot_frames(
                input_frames=sample_data['input_data'][i],
                pred_frames=sample_data['pred_data'][i],
                true_coords=sample_data['true_coords'][i],
                pred_coords=sample_data['pred_coords'][i],
                valid_mask=sample_data.get('pred_valid_masks', [None]*num_samples)[i] if 'pred_valid_masks' in sample_data else None,
                sample_idx=i,
                save_path=grid_path
            )
            
            traj_path = viz_dir / f'sample_{i+1}_trajectory.png'
            plotter.plot_trajectory(
                true_coords=sample_data['true_coords'][i],
                pred_coords=sample_data['pred_coords'][i],
                valid_mask=sample_data.get('pred_valid_masks', [None]*num_samples)[i] if 'pred_valid_masks' in sample_data else None,
                sample_idx=i,
                save_path=traj_path
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_args = load_model_and_args(args.checkpoint, device)
    
    # Create test loader
    print(f"Loading test data for {model_args.data}...")
    test_loader = create_test_loader(model_args)
    
    # Evaluate
    print("Evaluating model...")
    metrics, sample_data = evaluate_model(model, test_loader, device, model_args)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualizations
    print(f"Creating visualizations...")
    create_visualizations(sample_data, output_dir, model_args)
    
    print(f"Evaluation complete. Results saved to: {output_dir}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()