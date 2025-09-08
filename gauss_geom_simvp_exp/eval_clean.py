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
from datasets.geom_unified import create_unified_geom_loaders as create_geom_simple_loaders
from plot_utils import create_plotter


def load_model_and_args(checkpoint_path, device):
    """Load model and arguments from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']
    
    # Create model with correct parameters from checkpoint
    pre_seq_length = getattr(model_args, 'pre_seq_length', 10)
    img_size = getattr(model_args, 'img_size', 32)
    in_shape = (pre_seq_length, 1, img_size, img_size)
    
    hid_S = getattr(model_args, 'hid_S', 64)
    hid_T = getattr(model_args, 'hid_T', 512) 
    N_S = getattr(model_args, 'N_S', 4)
    N_T = getattr(model_args, 'N_T', 8)
    model_type = getattr(model_args, 'model_type', 'gSTA')
    
    # Check if we need to use GRU (from args)
    use_gru = getattr(model_args, 'use_gru', False)
    
    model = SimVPWithTaskHead(
        in_shape=in_shape,
        hid_S=hid_S,
        hid_T=hid_T,
        N_S=N_S,
        N_T=N_T,
        model_type=model_type,
        task=model_args.repr,
        use_gru=use_gru
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    except RuntimeError as e:
        if "coord_head" in str(e) or "gru_head" in str(e) or "frame_mlp" in str(e):
            print("Loading coord model with partial state dict")
            state_dict = checkpoint['model_state_dict']
            # Filter out coord_head, gru_head, or frame_mlp layers if they don't match
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                   if not (k.startswith('coord_head') or 
                                          k.startswith('gru_head') or 
                                          k.startswith('frame_mlp'))}
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            raise e
    
    model.eval()
    return model, model_args


def create_test_loader(args):
    """Create test data loader with consistent configuration."""
    if args.data == 'gauss':
        # Use the same configuration as training (from config.py)
        from datasets.config import FIELD_CONFIG
        eval_config = FIELD_CONFIG.copy()
        eval_config.update({
            'test_size': 200,
            'seed': getattr(args, 'gauss_seed', 42)  # Use consistent seed
        })
        
        from datasets.position_dependent_gaussian import create_data_loaders
        _, _, test_loader = create_data_loaders(
            batch_size=32,
            representation='gaussian' if args.repr == 'heat' else 'sparse',
            generate=True,
            config=eval_config
        )
    elif args.data == 'geom_simple':
        # For coord models, we still need frames as input, so use sparse representation
        if args.repr == 'coord':
            representation = 'sparse'
        elif args.repr == 'pixel':
            representation = 'sparse'
        else:  # heat
            representation = 'gaussian'
        
        _, _, test_loader = create_geom_simple_loaders(
            batch_size=32,
            sequence_length=20,
            num_test=200,
            representation=representation
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    
    return test_loader


def extract_coordinates_with_neighbor_check(frames, representation, neighbor_threshold=0.1):
    """Extract coordinates preferring points within clusters.
    If argmax is an isolated bright pixel, find the brightest pixel that has neighbors.
    Returns (coords, valid_mask).
    """
    B, T, C, H, W = frames.shape
    coords = torch.zeros(B, T, 2, device=frames.device)
    valid_mask = torch.zeros(B, T, dtype=torch.bool, device=frames.device)
    
    for b in range(B):
        for t in range(T):
            frame = frames[b, t, 0]  # Shape: (H, W)
            
            # Check if frame is not all zeros (valid frame)
            if representation in ['pixel', 'heat']:
                if torch.count_nonzero(frame).item() == 0:
                    continue
                
                # Find argmax location
                flat_idx = torch.argmax(frame.view(-1))
                y_max, x_max = divmod(flat_idx.item(), W)
                
                # Check if argmax has neighbors
                max_val = frame.max()
                if max_val > 0:
                    # Define a window around the argmax point
                    y_min = max(0, y_max - 2)
                    y_max_win = min(H, y_max + 3)
                    x_min = max(0, x_max - 2)
                    x_max_win = min(W, x_max + 3)
                    
                    window = frame[y_min:y_max_win, x_min:x_max_win]
                    # Count bright pixels in window (above threshold)
                    bright_count = (window > neighbor_threshold * max_val).sum().item()
                    
                    # If argmax is isolated (only 1-2 bright pixels), find clustered alternative
                    if bright_count <= 2:
                        # Find all candidate pixels above threshold
                        candidates = (frame > neighbor_threshold * max_val).nonzero()
                        
                        best_score = -1
                        best_x, best_y = x_max, y_max
                        
                        # Score each candidate by: brightness * number of neighbors
                        for y, x in candidates:
                            # Count neighbors
                            y_min_local = max(0, y - 2)
                            y_max_local = min(H, y + 3)
                            x_min_local = max(0, x - 2)
                            x_max_local = min(W, x + 3)
                            
                            local_window = frame[y_min_local:y_max_local, x_min_local:x_max_local]
                            neighbor_count = (local_window > neighbor_threshold * max_val).sum().item() - 1  # exclude self
                            
                            # Score = brightness * sqrt(neighbor_count)
                            # This favors bright pixels with more neighbors
                            if neighbor_count > 0:
                                score = frame[y, x].item() * (neighbor_count ** 0.5)
                                if score > best_score:
                                    best_score = score
                                    best_x, best_y = x.item(), y.item()
                        
                        x_max, y_max = best_x, best_y
                
                coords[b, t] = torch.tensor([x_max, y_max], device=frames.device)
                valid_mask[b, t] = True
    
    return coords, valid_mask


def extract_coordinates(frames, representation):
    """Extract coordinates using argmax for both pixel and heat representations.
    Returns (coords, valid_mask). A frame is invalid only when it is exactly all zeros.
    """
    # Use neighbor check for pixel representation to avoid corner artifacts
    if representation == 'pixel':
        return extract_coordinates_with_neighbor_check(frames, representation)
    
    # For heat representation, use simple argmax as Gaussians are more stable
    B, T, C, H, W = frames.shape
    coords = torch.zeros(B, T, 2, device=frames.device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=frames.device)

    if representation in ['pixel', 'heat']:
        for b in range(B):
            for t in range(T):
                frame = frames[b, t, 0]
                
                # Handle log probabilities for heat representation (from KL loss)
                if representation == 'heat' and frame.max() < 0:
                    frame = torch.exp(frame)
                
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
    
    # Per-frame errors for analysis
    per_frame_errors = [[] for _ in range(T)]
    
    # Velocity metrics
    magnitude_ratios = []
    cosine_similarities = []
    
    for b in range(B):
        # Calculate errors - L2 distance between predicted and true coordinates
        for t in range(T):
            error = torch.norm(pred_coords[b, t] - true_coords[b, t]).item()
            
            # Store for per-frame analysis
            per_frame_errors[t].append(error)
            
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
    
    # Calculate average per-frame errors
    avg_per_frame_errors = [np.mean(errors) if errors else 0.0 for errors in per_frame_errors]
    
    return {
        "valid": True,
        "avg_magnitude_ratio": np.mean(magnitude_ratios) if magnitude_ratios else 0.0,
        "avg_cosine_similarity": np.mean(cosine_similarities) if cosine_similarities else 0.0,
        "displacement_error_3": np.mean(errors_3) if errors_3 else 0.0,
        "displacement_error_6": np.mean(errors_6) if errors_6 else 0.0,
        "displacement_error_full": np.mean(errors_full) if errors_full else 0.0,
        "per_frame_mde": avg_per_frame_errors
    }


def evaluate_model(model, test_loader, device, model_args):
    """Evaluate model with clean data handling."""
    model.eval()
    
    all_pred_coords = []
    all_true_coords = []
    # For pattern-specific analysis (geom data)
    line_pred_coords = []
    line_true_coords = []
    arc_pred_coords = []
    arc_true_coords = []
    
    sample_data = {
        'input_data': [],
        'pred_data': [],
        'true_coords': [],
        'pred_coords': []
    }
    
    # Track pattern types for balanced visualization starting from sample 5
    saved_line_count = 0
    saved_arc_count = 0
    max_lines = 4
    max_arcs = 2
    max_samples = 6  # 4 lines + 2 arcs = 6 total
    sample_offset = 5  # Start from 5th sample to avoid early patterns
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Handle data loading
            if isinstance(batch, dict):
                coords = batch['coordinates'].to(device)
                frames = batch.get('frames')
                if frames is not None:
                    frames = frames.to(device)
                mask = batch.get('mask')
                pattern_types = batch.get('pattern_type', None)  # Get pattern types for geom data
            else:
                # Fallback for tuple format
                frames, coords, mask = batch
                frames = frames.to(device) if frames is not None else None
                coords = coords.to(device) if coords is not None else None
                pattern_types = None
            
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
                last_coord_for_model = coords[:, model_args.pre_seq_length-1]
            else:
                input_data = frames[:, :model_args.pre_seq_length]
                target_coords = coords[:, model_args.pre_seq_length:] if coords is not None else None
            
            # Forward pass
            if model_args.repr == 'coord' and getattr(model_args, 'coord_mode', 'absolute') == 'displacement':
                pred_output = model(input_data, last_coord_for_model)
            else:
                pred_output = model(input_data)
            
            # Extract coordinates based on representation
            if model_args.repr == 'coord':
                if getattr(model_args, 'coord_mode', 'absolute') == 'displacement':
                    # Reconstruct absolute coordinates from predicted displacements recursively
                    B, T = pred_output.shape[:2]
                    pred_coords = torch.zeros_like(pred_output)
                    current_pos = coords[:, model_args.pre_seq_length-1]  # (B, 2)
                    
                    for t in range(T):
                        displacement = pred_output[:, t]  # (B, 2)
                        current_pos = current_pos + displacement  # Accumulate displacement
                        pred_coords[:, t] = current_pos
                else:
                    pred_coords = pred_output
                pred_valid = None
            else:
                pred_coords, pred_valid = extract_coordinates(pred_output, model_args.repr)
            
            # Store results
            if target_coords is not None:
                all_pred_coords.append(pred_coords.cpu())
                all_true_coords.append(target_coords.cpu())
                
                # Store pattern-specific results for geom data
                if pattern_types is not None and model_args.data == 'geom_simple':
                    for i in range(len(pattern_types)):
                        if i < pred_coords.shape[0]:  # Ensure we don't go out of bounds
                            if pattern_types[i] == 'line':
                                line_pred_coords.append(pred_coords[i:i+1].cpu())
                                line_true_coords.append(target_coords[i:i+1].cpu())
                            elif pattern_types[i] == 'arc':
                                arc_pred_coords.append(pred_coords[i:i+1].cpu())
                                arc_true_coords.append(target_coords[i:i+1].cpu())
                
                # Store samples for visualization with balanced pattern types
                if len(sample_data['input_data']) < max_samples:
                    # For geom data, ensure balanced line/arc samples
                    if pattern_types is not None and model_args.data == 'geom_simple':
                        # Check all samples in batch to find desired pattern types
                        for idx in range(len(pattern_types)):
                            if len(sample_data['input_data']) >= max_samples:
                                break
                            if idx >= input_data.shape[0]:  # Safety check
                                break
                                
                            # Skip first few samples within each batch for variety
                            global_sample_idx = batch_idx * input_data.shape[0] + idx
                            if global_sample_idx < sample_offset:
                                continue
                                
                            pattern_type = pattern_types[idx]
                            should_save = False
                            
                            if pattern_type == 'line' and saved_line_count < max_lines:
                                should_save = True
                                saved_line_count += 1
                            elif pattern_type == 'arc' and saved_arc_count < max_arcs:
                                should_save = True
                                saved_arc_count += 1
                            
                            if should_save:
                                sample_data['input_data'].append(input_data[idx].cpu())
                                # For coord repr, pred_output is already coordinates, not frames
                                if model_args.repr == 'coord':
                                    sample_data['pred_data'].append(pred_coords[idx].cpu())
                                else:
                                    sample_data['pred_data'].append(pred_output[idx].cpu())
                                sample_data['true_coords'].append(coords[idx].cpu())
                                sample_data['pred_coords'].append(pred_coords[idx].cpu())
                                if pred_valid is not None:
                                    sample_data.setdefault('pred_valid_masks', []).append(pred_valid[idx].cpu())
                    else:
                        # For non-geom data, just save first 5 samples
                        sample_data['input_data'].append(input_data[0].cpu())
                        # For coord repr, pred_output is already coordinates, not frames
                        if model_args.repr == 'coord':
                            sample_data['pred_data'].append(pred_coords[0].cpu())
                        else:
                            sample_data['pred_data'].append(pred_output[0].cpu())
                        sample_data['true_coords'].append(coords[0].cpu())
                        sample_data['pred_coords'].append(pred_coords[0].cpu())
                        if pred_valid is not None:
                            sample_data.setdefault('pred_valid_masks', []).append(pred_valid[0].cpu())
    
    # Calculate metrics
    if all_pred_coords and all_true_coords:
        all_pred_coords = torch.cat(all_pred_coords, dim=0)
        all_true_coords = torch.cat(all_true_coords, dim=0)
        metrics = calculate_metrics(all_pred_coords, all_true_coords)
        
        # Calculate pattern-specific metrics for geom data
        if model_args.data == 'geom_simple' and line_pred_coords and arc_pred_coords:
            # Line metrics
            line_pred = torch.cat(line_pred_coords, dim=0)
            line_true = torch.cat(line_true_coords, dim=0)
            line_metrics = calculate_metrics(line_pred, line_true)
            
            # Arc metrics
            arc_pred = torch.cat(arc_pred_coords, dim=0)
            arc_true = torch.cat(arc_true_coords, dim=0)
            arc_metrics = calculate_metrics(arc_pred, arc_true)
            
            # Add pattern-specific metrics
            metrics['line_mde_3'] = line_metrics['displacement_error_3']
            metrics['line_mde_6'] = line_metrics['displacement_error_6']
            metrics['line_mde_full'] = line_metrics['displacement_error_full']
            metrics['arc_mde_3'] = arc_metrics['displacement_error_3']
            metrics['arc_mde_6'] = arc_metrics['displacement_error_6']
            metrics['arc_mde_full'] = arc_metrics['displacement_error_full']
            
            # Count samples
            metrics['n_line_samples'] = len(line_pred)
            metrics['n_arc_samples'] = len(arc_pred)
    else:
        metrics = {"valid": False}
    
    return metrics, sample_data


def create_visualizations(sample_data, output_dir, model_args):
    """Create visualizations using appropriate plotter."""
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plotter based on representation
    exp_name = getattr(model_args, 'exp_name', 'unknown')
    plotter = create_plotter(model_args.repr, model_args.img_size, model_args.pre_seq_length, exp_name)
    
    num_samples = min(6, len(sample_data['input_data']))
    
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
    
    # Create per-frame MDE plot
    if 'per_frame_mde' in metrics:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        frames = list(range(1, len(metrics['per_frame_mde']) + 1))
        plt.plot(frames, metrics['per_frame_mde'], 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Prediction Frame', fontsize=12)
        plt.ylabel('Mean Displacement Error (pixels)', fontsize=12)
        plt.title('Per-Frame MDE Analysis', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'per_frame_mde.png', dpi=150)
        plt.close()
    
    print(f"Evaluation complete. Results saved to: {output_dir}")
    print("Metrics:")
    
    # Print main metrics
    main_keys = ['displacement_error_3', 'displacement_error_6', 'displacement_error_full', 
                 'avg_magnitude_ratio', 'avg_cosine_similarity']
    for key in main_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    # Print per-frame MDE if available
    if 'per_frame_mde' in metrics and len(metrics['per_frame_mde']) > 0:
        print(f"  Per-frame MDE: {metrics['per_frame_mde'][0]:.4f} (frame 1) to {metrics['per_frame_mde'][-1]:.4f} (frame {len(metrics['per_frame_mde'])})")
    
    # Print pattern-specific metrics if available
    if 'line_mde_full' in metrics and 'arc_mde_full' in metrics:
        print("\nPattern-specific analysis:")
        print(f"  Lines ({metrics['n_line_samples']} samples):")
        print(f"    MDE@3: {metrics['line_mde_3']:.4f}")
        print(f"    MDE@6: {metrics['line_mde_6']:.4f}")
        print(f"    MDE@Full: {metrics['line_mde_full']:.4f}")
        print(f"  Arcs ({metrics['n_arc_samples']} samples):")
        print(f"    MDE@3: {metrics['arc_mde_3']:.4f}")
        print(f"    MDE@6: {metrics['arc_mde_6']:.4f}")
        print(f"    MDE@Full: {metrics['arc_mde_full']:.4f}")
        print(f"  Difference (Arc - Line):")
        print(f"    MDE@3: {metrics['arc_mde_3'] - metrics['line_mde_3']:+.4f}")
        print(f"    MDE@6: {metrics['arc_mde_6'] - metrics['line_mde_6']:+.4f}")
        print(f"    MDE@Full: {metrics['arc_mde_full'] - metrics['line_mde_full']:+.4f}")


if __name__ == "__main__":
    main()