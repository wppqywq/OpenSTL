#!/usr/bin/env python3
"""
Visualize parallel predictions on geometric patterns
Generate the visualization shown in the user's image
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math
import sys

# Add OpenSTL path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class ImprovedCoordinateRegressionModel(nn.Module):
    """IDENTICAL model from Phase 3/4"""
    def __init__(self, temporal_strategy='last_frame', max_future_frames=9):
        super().__init__()
        
        self.temporal_strategy = temporal_strategy
        self.max_future_frames = max_future_frames
        
        # Custom 3D CNN encoder for video sequences (MPS compatible)
        self.encoder = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Second 3D conv block with stride for downsampling
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Third 3D conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Fourth 3D conv block with stride for downsampling
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        # Frame index embedding (which future frame to predict)
        self.frame_embedding = nn.Embedding(max_future_frames, 64)
        
        # Calculate spatial features dimension based on strategy
        if temporal_strategy == 'temporal_average':
            spatial_features = 256 * 16  # [B, 256, 4, 4] flattened
        elif temporal_strategy == 'last_frame':
            spatial_features = 256 * 16  # [B, 256, 4, 4] from last frame
        elif temporal_strategy == 'multi_frame':
            spatial_features = 256 * 16 * 3  # Concatenate last 3 frames
        else:
            raise ValueError(f"Unknown temporal_strategy: {temporal_strategy}")
        
        # Enhanced MLP regression head
        self.regression_head = nn.Sequential(
            nn.Linear(spatial_features + 64, 512),  # +64 for frame embedding
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # (Δx, Δy) displacement
        )
    
    def forward(self, x, future_frame_idx):
        """
        Args:
            x: [B, T, C, H, W] video sequence (T=10)
            future_frame_idx: [B] indices indicating which future frame to predict (0-8)
        """
        batch_size = x.size(0)
        
        # 3D CNN encoder: [B, T, C, H, W] -> [B, C, T, H, W] -> [B, 256, T', H', W']
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for encoder
        encoder_output = self.encoder(x)  # [B, 256, T', H', W']
        
        # Temporal feature extraction based on strategy
        if self.temporal_strategy == 'temporal_average':
            # Average across time: [B, 256, T', H', W'] -> [B, 256, 1, 4, 4]
            temporal_pool = nn.AdaptiveAvgPool3d((1, 4, 4))
            spatial_features = temporal_pool(encoder_output)
            spatial_features = spatial_features.flatten(1)  # [B, 256*16]
            
        elif self.temporal_strategy == 'last_frame':
            # Use only the last temporal frame: [B, 256, T', H', W'] -> [B, 256, H', W']
            last_frame = encoder_output[:, :, -1, :, :]  # [B, 256, H', W']
            # Spatial pooling to fixed size
            spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
            spatial_features = spatial_pool(last_frame)  # [B, 256, 4, 4]
            spatial_features = spatial_features.flatten(1)  # [B, 256*16]
            
        elif self.temporal_strategy == 'multi_frame':
            # Use last 3 temporal frames
            if encoder_output.size(2) >= 3:
                last_frames = encoder_output[:, :, -3:, :, :]  # [B, 256, 3, H', W']
                # Spatial pooling for each frame
                spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
                frame_features = []
                for i in range(3):
                    frame_feat = spatial_pool(last_frames[:, :, i, :, :])
                    frame_features.append(frame_feat.flatten(1))
                spatial_features = torch.cat(frame_features, dim=1)  # [B, 256*16*3]
            else:
                # Fallback to last frame if not enough temporal frames
                last_frame = encoder_output[:, :, -1, :, :]
                spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
                spatial_features = spatial_pool(last_frame)
                spatial_features = spatial_features.flatten(1)
                # Repeat to match expected dimension
                spatial_features = spatial_features.repeat(1, 3)
        else:
            # Default fallback to temporal average if strategy is unexpected
            spatial_features = nn.AdaptiveAvgPool3d((1, 4, 4))(encoder_output)
            spatial_features = spatial_features.flatten(1)
        
        # Frame index embedding
        frame_embed = self.frame_embedding(future_frame_idx)  # [B, 64]
        
        # Combine spatial features and frame embedding
        combined_features = torch.cat([spatial_features, frame_embed], dim=1)
        
        # Predict displacement
        displacement = self.regression_head(combined_features)
        
        return displacement


def generate_line_pattern(start_point, velocity, num_frames=20):
    """Generate straight line movement (constant velocity)"""
    coordinates = torch.zeros(num_frames, 2)
    coordinates[0] = torch.tensor(start_point, dtype=torch.float32)
    
    for t in range(1, num_frames):
        coordinates[t] = coordinates[t-1] + torch.tensor(velocity, dtype=torch.float32)
        coordinates[t] = torch.clamp(coordinates[t], 2, 30)
    
    return coordinates


def generate_arc_pattern(center, radius, start_angle, angular_velocity, num_frames=20):
    """Generate circular arc movement (curved motion)"""
    coordinates = torch.zeros(num_frames, 2)
    
    for t in range(num_frames):
        angle = start_angle + angular_velocity * t
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        coordinates[t] = torch.tensor([x, y], dtype=torch.float32)
        coordinates[t] = torch.clamp(coordinates[t], 2, 30)
    
    return coordinates


def generate_bounce_pattern(start_point, velocity, bounds=(2, 30), num_frames=20):
    """Generate bouncing movement (velocity reversals)"""
    coordinates = torch.zeros(num_frames, 2)
    coordinates[0] = torch.tensor(start_point, dtype=torch.float32)
    current_velocity = torch.tensor(velocity, dtype=torch.float32)
    
    for t in range(1, num_frames):
        next_pos = coordinates[t-1] + current_velocity
        
        if next_pos[0] <= bounds[0] or next_pos[0] >= bounds[1]:
            current_velocity[0] = -current_velocity[0]
        if next_pos[1] <= bounds[0] or next_pos[1] >= bounds[1]:
            current_velocity[1] = -current_velocity[1]
        
        coordinates[t] = coordinates[t-1] + current_velocity
        coordinates[t] = torch.clamp(coordinates[t], bounds[0], bounds[1])
    
    return coordinates


def coordinates_to_frames(coordinates, img_size=32):
    """Convert coordinates to sparse binary frames"""
    num_frames = coordinates.shape[0]
    frames = torch.zeros(num_frames, 1, img_size, img_size)
    
    for t in range(num_frames):
        x, y = coordinates[t].long()
        if 0 <= x < img_size and 0 <= y < img_size:
            frames[t, 0, x, y] = 1.0
    
    return frames


def predict_parallel_trajectory(model, input_frames, device, num_future_frames=9):
    """Predict future trajectory using parallel predictions"""
    model.eval()
    input_frames = input_frames.to(device)
    
    predicted_displacements = []
    
    with torch.no_grad():
        for frame_idx in range(num_future_frames):
            frame_indices = torch.tensor([frame_idx], dtype=torch.long).to(device)
            displacement = model(input_frames, frame_indices)
            predicted_displacements.append(displacement.cpu())
    
    return torch.cat(predicted_displacements, dim=0)  # [num_future_frames, 2]


def reconstruct_trajectory_from_displacements(start_position, displacements):
    """Reconstruct full trajectory from displacement vectors"""
    trajectory = torch.zeros(len(displacements) + 1, 2)
    trajectory[0] = start_position
    
    for i, displacement in enumerate(displacements):
        trajectory[i + 1] = trajectory[i] + displacement
    
    return trajectory


def generate_parallel_predictions_visualization():
    """Generate the parallel predictions visualization matching user's image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the best model
    model_path = "experiments/phase4_geometric_validation/results/best_geometric_coordinate_model.pth"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    temporal_strategy = checkpoint.get('temporal_strategy', 'last_frame')
    
    model = ImprovedCoordinateRegressionModel(temporal_strategy=temporal_strategy)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model with temporal strategy: {temporal_strategy}")
    
    # Generate test patterns (3 samples of each type like in the user's image)
    test_patterns = []
    pattern_names = []
    
    # Line patterns (3 samples)
    line_configs = [
        ([5, 5], [0.8, 0.6]),  # Bottom-left to top-right
        ([15, 20], [0.5, -0.7]),  # Top-middle to bottom-right  
        ([25, 15], [-0.6, 0.4])  # Right-middle to left-top
    ]
    
    for config in line_configs:
        coords = generate_line_pattern(config[0], config[1], num_frames=20)
        test_patterns.append(coords)
        pattern_names.append('Line')
    
    # Arc patterns (3 samples)
    arc_configs = [
        ([12, 12], 6, 0, 0.2),  # Clockwise arc
        ([16, 8], 5, math.pi/2, 0.25),  # Arc starting from top
        ([20, 20], 7, math.pi, -0.18)  # Counter-clockwise arc
    ]
    
    for config in arc_configs:
        coords = generate_arc_pattern(config[0], config[1], config[2], config[3], num_frames=20)
        test_patterns.append(coords)
        pattern_names.append('Arc')
    
    # Bounce patterns (3 samples)
    bounce_configs = [
        ([8, 14], [1.2, -1.0]),  # Diagonal bounce
        ([15, 15], [0.9, -1.5]),  # Steep bounce
        ([20, 18], [-1.1, -0.8])  # Left-down bounce
    ]
    
    for config in bounce_configs:
        coords = generate_bounce_pattern(config[0], config[1], num_frames=20)
        test_patterns.append(coords)
        pattern_names.append('Bounce')
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Parallel Predictions on Geometric Patterns', fontsize=16, fontweight='bold')
    
    for i, (coords, pattern_name) in enumerate(zip(test_patterns, pattern_names)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Convert coordinates to frames for model input
        frames = coordinates_to_frames(coords)
        input_frames = frames[:10].unsqueeze(0)  # [1, 10, 1, 32, 32]
        
        # Get model predictions
        predicted_displacements = predict_parallel_trajectory(model, input_frames, device)
        
        # Reconstruct predicted trajectory
        last_input_position = coords[9]  # Last position from input sequence
        predicted_coords = reconstruct_trajectory_from_displacements(
            last_input_position, predicted_displacements
        )
        
        # Extract coordinate arrays for plotting
        true_input_coords = coords[:10].numpy()  # Frames 0-9 (input)
        true_future_coords = coords[10:].numpy()  # Frames 10-19 (ground truth future)
        pred_future_coords = predicted_coords[1:].numpy()  # Predicted future frames
        
        # Calculate prediction error
        if len(true_future_coords) > 0 and len(pred_future_coords) > 0:
            min_len = min(len(true_future_coords), len(pred_future_coords))
            error = np.mean(np.linalg.norm(
                true_future_coords[:min_len] - pred_future_coords[:min_len], axis=1
            ))
        else:
            error = 0
        
        # Plot trajectories
        ax.plot(true_input_coords[:, 0], true_input_coords[:, 1], 
                'ko-', markersize=6, linewidth=2, label='True Input (0-4)')
        ax.plot(true_future_coords[:, 0], true_future_coords[:, 1], 
                'o-', color='gray', markersize=5, linewidth=2, label='True Future (5-9)')
        ax.plot(pred_future_coords[:, 0], pred_future_coords[:, 1], 
                '^-', color='red', markersize=5, linewidth=2, label='Parallel Prediction')
        
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{pattern_name} - Error: {error:.2f}px')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = "experiments/phase4_geometric_validation/results/parallel_predictions_geometric_patterns.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Generating Parallel Predictions Visualization for Geometric Patterns")
    print("=" * 80)
    
    try:
        generate_parallel_predictions_visualization()
        print("✅ Visualization generated successfully")
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        import traceback
        traceback.print_exc() 