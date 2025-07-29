#!/usr/bin/env python3
"""
Correct Model Visualization
Phase 3 model on Phase 3 data, Phase 4 model on Phase 4 data
Following the provided format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path


class DirectionMagnitudeModel(nn.Module):
    """Model architecture for displacement prediction"""
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.displacement_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # (Δx, Δy) displacement
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        features = self.encoder(x)
        displacement = self.displacement_head(features)
        return displacement


# Phase 3 data loading
def load_phase3_data():
    """Load Phase 3 coordinate data"""
    try:
        test_data = torch.load('../phase3_coordinate_regression/data/test_data.pt')
        return test_data
    except:
        print("Phase 3 data not found!")
        return None


# Phase 4 geometric data generation
def generate_line_pattern(start_pos, velocity, num_frames=20):
    """Generate linear motion pattern"""
    coords = []
    x, y = start_pos
    vx, vy = velocity
    
    for i in range(num_frames):
        coords.append([x, y])
        x += vx
        y += vy
        x = np.clip(x, 1, 31)
        y = np.clip(y, 1, 31)
    
    return torch.tensor(coords, dtype=torch.float32)


def generate_arc_pattern(center, radius, start_angle, angular_velocity, num_frames=20):
    """Generate arc motion pattern"""
    coords = []
    cx, cy = center
    angle = start_angle
    
    for i in range(num_frames):
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        x = np.clip(x, 1, 31)
        y = np.clip(y, 1, 31)
        coords.append([x, y])
        angle += angular_velocity
    
    return torch.tensor(coords, dtype=torch.float32)


def generate_bounce_pattern(start_pos, velocity, num_frames=20):
    """Generate bouncing motion pattern"""
    coords = []
    x, y = start_pos
    vx, vy = velocity
    
    for i in range(num_frames):
        coords.append([x, y])
        x += vx
        y += vy
        
        # Bounce off boundaries
        if x <= 1 or x >= 31:
            vx = -vx
        if y <= 1 or y >= 31:
            vy = -vy
            
        x = np.clip(x, 1, 31)
        y = np.clip(y, 1, 31)
    
    return torch.tensor(coords, dtype=torch.float32)


def coordinates_to_frames(coords):
    """Convert coordinates to binary frames"""
    frames = torch.zeros(coords.shape[0], 1, 32, 32)
    
    for i, (x, y) in enumerate(coords):
        x_int = int(torch.round(x).clamp(0, 31))
        y_int = int(torch.round(y).clamp(0, 31))
        frames[i, 0, y_int, x_int] = 1.0
    
    return frames


def generate_geometric_test_data():
    """Generate test geometric patterns (same as Phase 4 training)"""
    all_frames = []
    all_coordinates = []
    pattern_types = []
    
    # Generate test samples for each pattern type
    patterns = [
        ('Line', lambda: generate_line_pattern([np.random.uniform(8, 24), np.random.uniform(8, 24)], 
                                              [np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)], 20)),
        ('Arc', lambda: generate_arc_pattern([np.random.uniform(12, 20), np.random.uniform(12, 20)], 
                                           np.random.uniform(3, 6), np.random.uniform(0, 2*math.pi), 
                                           np.random.uniform(-0.3, 0.3), 20)),
        ('Bounce', lambda: generate_bounce_pattern([np.random.uniform(10, 22), np.random.uniform(10, 22)], 
                                                  [np.random.uniform(-1.8, 1.8), np.random.uniform(-1.8, 1.8)], 20))
    ]
    
    # Generate 3 samples per pattern type
    for pattern_name, pattern_func in patterns:
        for _ in range(3):
            coords = pattern_func()
            frames = coordinates_to_frames(coords)
            
            all_frames.append(frames)
            all_coordinates.append(coords)
            pattern_types.append(pattern_name)
    
    frames_tensor = torch.stack(all_frames)
    coords_tensor = torch.stack(all_coordinates)
    
    return frames_tensor, coords_tensor, pattern_types


def predict_trajectory_iterative(model, initial_frames, initial_coord, num_steps=5):
    """Predict trajectory iteratively using the model"""
    device = next(model.parameters()).device
    model.eval()
    
    predicted_coords = [initial_coord.clone()]
    current_frames = initial_frames.clone()  # [T, C, H, W]
    current_pos = initial_coord.clone()
    
    with torch.no_grad():
        for step in range(num_steps):
            # Predict displacement
            input_frames = current_frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
            predicted_displacement = model(input_frames).cpu().squeeze(0)  # [2]
            
            # Update position
            current_pos = current_pos + predicted_displacement
            current_pos = torch.clamp(current_pos, 1, 31)
            predicted_coords.append(current_pos.clone())
            
            # Create new frame
            new_frame = torch.zeros(1, 32, 32)
            x_int = int(current_pos[0].round().clamp(0, 31))
            y_int = int(current_pos[1].round().clamp(0, 31))
            new_frame[0, y_int, x_int] = 1.0
            
            # Update frames (shift window)
            current_frames = torch.cat([current_frames[1:], new_frame.unsqueeze(0)], dim=0)
    
    return torch.stack(predicted_coords)


def visualize_phase3_predictions():
    """Visualize Phase 3 model predictions on Phase 3 data"""
    device = torch.device('cpu')
    
    # Load Phase 3 model
    phase3_model = DirectionMagnitudeModel()
    try:
        phase3_model.load_state_dict(torch.load('results/best_direction_magnitude_model.pth', map_location='cpu'))
        print("✅ Phase 3 model loaded")
    except:
        print("❌ Phase 3 model not found")
        return
    
    # Load Phase 3 test data
    test_data = load_phase3_data()
    if test_data is None:
        print("❌ Phase 3 test data not available")
        return
    
    # Prepare data
    test_frames = test_data['frames'][:9]  # Take 9 samples (3x3 grid)
    test_frames = test_frames.squeeze(1).unsqueeze(2)  # [9, 32, 1, 32, 32]
    test_coords = test_data['coordinates'][:9]  # [9, 32, 2]
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Phase 3 Model Predictions on Real Coordinate Data', fontsize=16, y=0.95)
    
    for i in range(9):
        row = i // 3
        col = i % 3
        
        # Use frames 0-9 as input, predict frames 10-14
        input_frames = test_frames[i, :10]  # [10, 1, 32, 32]
        true_coords = test_coords[i]  # [32, 2]
        
        # Get initial position from frame 9
        initial_coord = true_coords[9]
        
        # Predict trajectory
        pred_coords = predict_trajectory_iterative(phase3_model, input_frames, initial_coord, num_steps=5)
        
        # Calculate error
        true_future = true_coords[10:15]  # frames 10-14
        errors = torch.norm(pred_coords[1:] - true_future, p=2, dim=1)
        mean_error = errors.mean().item()
        
        # Plot
        ax = axes[row, col]
        ax.plot(true_coords[:10, 0], true_coords[:10, 1], 'o-', color='black', 
               markersize=4, linewidth=1, label='True Input (0-9)')
        ax.plot(true_coords[10:15, 0], true_coords[10:15, 1], 'o-', color='gray', 
               markersize=4, linewidth=1, label='True Future (10-14)')
        ax.plot(pred_coords[1:, 0], pred_coords[1:, 1], '^-', color='red', 
               markersize=6, linewidth=2, label='Predictions')
        
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sample {i+1} - Error: {mean_error:.2f}px')
        
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('phase3_model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Phase 3 predictions saved as 'phase3_model_predictions.png'")


def visualize_phase4_predictions():
    """Visualize Phase 4 model predictions on geometric patterns - following provided format"""
    device = torch.device('cpu')
    
    # Load Phase 4 model
    phase4_model = DirectionMagnitudeModel()
    try:
        phase4_model.load_state_dict(torch.load('results/best_direction_magnitude_geometric_model.pth', map_location='cpu'))
        print("✅ Phase 4 model loaded")
    except:
        print("❌ Phase 4 model not found")
        return
    
    # Generate test data
    frames, coords, pattern_types = generate_geometric_test_data()
    
    # Create visualization following the provided format
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Phase 4 Model Predictions on Geometric Patterns', fontsize=16, y=0.95)
    
    # Organize by pattern type
    pattern_indices = {
        'Line': [0, 1, 2],
        'Arc': [3, 4, 5], 
        'Bounce': [6, 7, 8]
    }
    
    for row, (pattern_name, indices) in enumerate(pattern_indices.items()):
        for col, idx in enumerate(indices):
            # Use frames 0-4 as input, predict frames 5-9
            input_frames = frames[idx, :5]  # [5, 1, 32, 32] 
            true_coords = coords[idx]  # [20, 2]
            
            # Get initial position from frame 4
            initial_coord = true_coords[4]
            
            # Predict trajectory
            pred_coords = predict_trajectory_iterative(phase4_model, input_frames, initial_coord, num_steps=5)
            
            # Calculate error
            true_future = true_coords[5:10]  # frames 5-9
            errors = torch.norm(pred_coords[1:] - true_future, p=2, dim=1)
            mean_error = errors.mean().item()
            
            # Plot following the format
            ax = axes[row, col]
            ax.plot(true_coords[:5, 0], true_coords[:5, 1], 'o-', color='black', 
                   markersize=4, linewidth=1, label='True Input (0-4)')
            ax.plot(true_coords[5:10, 0], true_coords[5:10, 1], 'o-', color='gray', 
                   markersize=4, linewidth=1, label='True Future (5-9)')
            ax.plot(pred_coords[1:, 0], pred_coords[1:, 1], '^-', color='red', 
                   markersize=6, linewidth=2, label='Parallel Prediction')
            
            ax.set_xlim(0, 32)
            ax.set_ylim(0, 32)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{pattern_name} - Error: {mean_error:.2f}px')
            
            if row == 0 and col == 2:  # Add legend to top-right plot
                ax.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('phase4_model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Phase 4 predictions saved as 'phase4_model_predictions.png'")


def visualize_frame_predictions_separate():
    """Create separate frame visualizations for each model on its own data"""
    
    # Phase 3 frame visualization
    print("Creating Phase 3 frame visualization...")
    test_data = load_phase3_data()
    if test_data is not None:
        phase3_model = DirectionMagnitudeModel()
        try:
            phase3_model.load_state_dict(torch.load('results/best_direction_magnitude_model.pth', map_location='cpu'))
            
            # Use first test sample
            test_frames = test_data['frames'][0].squeeze(1).unsqueeze(2)  # [32, 1, 32, 32]
            test_coords = test_data['coordinates'][0]  # [32, 2]
            
            # Predict
            input_frames = test_frames[:10]
            initial_coord = test_coords[9]
            pred_coords = predict_trajectory_iterative(phase3_model, input_frames, initial_coord, num_steps=5)
            
            # Create frame visualization
            fig, axes = plt.subplots(2, 8, figsize=(24, 6))
            
            for i in range(8):
                frame_idx = i + 10  # Show frames 10-17
                
                # Input frame (for reference)
                if i < 5:  # Show actual future frames for first 5
                    input_frame = test_frames[frame_idx].squeeze().numpy()
                else:
                    input_frame = np.zeros((32, 32))
                
                axes[0, i].imshow(input_frame, cmap='gray', vmin=0, vmax=1)
                axes[0, i].set_title(f'True Frame {frame_idx}')
                axes[0, i].set_xticks([])
                axes[0, i].set_yticks([])
                
                # Predicted frame
                pred_frame = np.zeros((32, 32))
                if i < 5:  # We only predict 5 steps
                    pred_coord = pred_coords[i + 1]  # +1 because pred_coords[0] is initial
                    pred_x = int(pred_coord[0].clamp(0, 31))
                    pred_y = int(pred_coord[1].clamp(0, 31))
                    pred_frame[pred_y, pred_x] = 1.0
                
                axes[1, i].imshow(pred_frame, cmap='Blues', vmin=0, vmax=1, alpha=0.8)
                
                # Overlay ground truth
                if frame_idx < len(test_coords):
                    true_coord = test_coords[frame_idx]
                    axes[1, i].scatter(true_coord[0], true_coord[1], c='red', s=80, 
                                     marker='o', edgecolors='white', linewidth=2, alpha=0.9)
                    
                    if i < 5:
                        error = torch.norm(pred_coords[i + 1] - true_coord)
                        axes[1, i].set_title(f'Pred Frame {frame_idx}\nError: {error:.1f}px')
                    else:
                        axes[1, i].set_title(f'No Prediction')
                
                axes[1, i].set_xticks([])
                axes[1, i].set_yticks([])
            
            axes[0, 0].set_ylabel('True Frames', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Phase 3 Model\n(w/ True Position)', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.suptitle('Phase 3 Model Frame Predictions on Real Coordinate Data', fontsize=16, y=1.02)
            plt.savefig('phase3_frame_predictions.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Phase 3 frame predictions saved")
            
        except:
            print("❌ Phase 3 model loading failed")
    
    # Phase 4 frame visualization  
    print("Creating Phase 4 frame visualization...")
    phase4_model = DirectionMagnitudeModel()
    try:
        phase4_model.load_state_dict(torch.load('results/best_direction_magnitude_geometric_model.pth', map_location='cpu'))
        
        # Generate test data
        frames, coords, pattern_types = generate_geometric_test_data()
        sample_idx = 0  # Use first sample (Line pattern)
        
        # Predict
        input_frames = frames[sample_idx, :5]
        true_coords = coords[sample_idx]
        initial_coord = true_coords[4]
        pred_coords = predict_trajectory_iterative(phase4_model, input_frames, initial_coord, num_steps=5)
        
        # Create frame visualization
        fig, axes = plt.subplots(2, 8, figsize=(24, 6))
        
        for i in range(8):
            frame_idx = i + 5  # Show frames 5-12
            
            # True frame
            if frame_idx < frames.shape[1]:
                true_frame = frames[sample_idx, frame_idx].squeeze().numpy()
            else:
                true_frame = np.zeros((32, 32))
            
            axes[0, i].imshow(true_frame, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'True Frame {frame_idx}')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # Predicted frame
            pred_frame = np.zeros((32, 32))
            if i < 5:  # We only predict 5 steps
                pred_coord = pred_coords[i + 1]
                pred_x = int(pred_coord[0].clamp(0, 31))
                pred_y = int(pred_coord[1].clamp(0, 31))
                pred_frame[pred_y, pred_x] = 1.0
            
            axes[1, i].imshow(pred_frame, cmap='Greens', vmin=0, vmax=1, alpha=0.8)
            
            # Overlay ground truth
            if frame_idx < len(true_coords):
                true_coord = true_coords[frame_idx]
                axes[1, i].scatter(true_coord[0], true_coord[1], c='red', s=80, 
                                 marker='o', edgecolors='white', linewidth=2, alpha=0.9)
                
                if i < 5:
                    error = torch.norm(pred_coords[i + 1] - true_coord)
                    axes[1, i].set_title(f'Pred Frame {frame_idx}\nError: {error:.1f}px')
                else:
                    axes[1, i].set_title(f'No Prediction')
            
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        axes[0, 0].set_ylabel('True Frames', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Phase 4 Model\n(w/ True Position)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle(f'Phase 4 Model Frame Predictions on {pattern_types[sample_idx]} Pattern', fontsize=16, y=1.02)
        plt.savefig('phase4_frame_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Phase 4 frame predictions saved")
        
    except:
        print("❌ Phase 4 model loading failed")


def main():
    """Main visualization function"""
    print("Correct Model Visualization")
    print("=" * 50)
    print("Phase 3 model → Phase 3 data")
    print("Phase 4 model → Phase 4 data")
    print()
    
    print("1. Phase 3 model predictions on real coordinate data...")
    visualize_phase3_predictions()
    
    print("\n2. Phase 4 model predictions on geometric patterns...")
    visualize_phase4_predictions()
    
    print("\n3. Frame-by-frame predictions...")
    visualize_frame_predictions_separate()
    
    print("\n✅ All visualizations completed!")
    print("Generated files:")
    print("  - phase3_model_predictions.png")
    print("  - phase4_model_predictions.png") 
    print("  - phase3_frame_predictions.png")
    print("  - phase4_frame_predictions.png")


if __name__ == "__main__":
    main() 