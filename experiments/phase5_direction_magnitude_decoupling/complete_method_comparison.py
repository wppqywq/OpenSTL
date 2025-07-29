#!/usr/bin/env python3
"""
Complete Method Comparison Visualization
Show all 3 methods: L1 Baseline, Direction-Magnitude, Direction Emphasis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import json
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
    """Generate test geometric patterns"""
    all_frames = []
    all_coordinates = []
    pattern_types = []
    
    # Generate one sample of each pattern for comparison
    patterns = [
        ('Line', lambda: generate_line_pattern([10, 10], [1.2, 0.8], 20)),
        ('Arc', lambda: generate_arc_pattern([16, 16], 5, 0, 0.2, 20)),
        ('Bounce', lambda: generate_bounce_pattern([15, 15], [1.5, -1.2], 20))
    ]
    
    for pattern_name, pattern_func in patterns:
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


def load_all_models(data_type):
    """Load all 3 models for given data type"""
    models = {}
    method_names = ['l1_baseline', 'direction_magnitude', 'direction_emphasis']
    
    suffix = '_geometric_model.pth' if data_type == 'phase4' else '_model.pth'
    
    for method in method_names:
        model = DirectionMagnitudeModel()
        try:
            model.load_state_dict(torch.load(f'results/best_{method}{suffix}', map_location='cpu'))
            models[method] = model
            print(f"✅ Loaded {method} model for {data_type}")
        except:
            print(f"❌ Failed to load {method} model for {data_type}")
            models[method] = None
    
    return models


def visualize_phase3_all_methods():
    """Visualize all 3 methods on Phase 3 data"""
    
    # Load all Phase 3 models
    models = load_all_models('phase3')
    
    # Load Phase 3 test data
    test_data = load_phase3_data()
    if test_data is None:
        print("❌ Phase 3 test data not available")
        return
    
    # Prepare data - use 3 samples
    test_frames = test_data['frames'][:3]
    test_frames = test_frames.squeeze(1).unsqueeze(2)  # [3, 32, 1, 32, 32]
    test_coords = test_data['coordinates'][:3]  # [3, 32, 2]
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Phase 3: All Methods on Real Coordinate Data', fontsize=16, y=0.95)
    
    method_names = ['l1_baseline', 'direction_magnitude', 'direction_emphasis']
    method_labels = ['L1 Baseline', 'Direction-Magnitude', 'Direction Emphasis']
    colors = ['blue', 'red', 'purple']
    
    for row, (method, model, label, color) in enumerate(zip(method_names, [models[m] for m in method_names], method_labels, colors)):
        if model is None:
            continue
            
        for col in range(3):
            # Use frames 0-9 as input, predict frames 10-14
            input_frames = test_frames[col, :10]  # [10, 1, 32, 32]
            true_coords = test_coords[col]  # [32, 2]
            initial_coord = true_coords[9]
            
            # Predict trajectory
            pred_coords = predict_trajectory_iterative(model, input_frames, initial_coord, num_steps=5)
            
            # Calculate error
            true_future = true_coords[10:15]  # frames 10-14
            errors = torch.norm(pred_coords[1:] - true_future, p=2, dim=1)
            mean_error = errors.mean().item()
            
            # Plot
            ax = axes[row, col]
            ax.plot(true_coords[:10, 0], true_coords[:10, 1], 'o-', color='black', 
                   markersize=3, linewidth=1, label='True Input (0-9)')
            ax.plot(true_coords[10:15, 0], true_coords[10:15, 1], 'o-', color='gray', 
                   markersize=3, linewidth=1, label='True Future (10-14)')
            ax.plot(pred_coords[1:, 0], pred_coords[1:, 1], '^-', color=color, 
                   markersize=5, linewidth=2, label=f'{label} Prediction')
            
            ax.set_xlim(0, 32)
            ax.set_ylim(0, 32)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{label} - Sample {col+1}\nError: {mean_error:.2f}px')
            
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('phase3_all_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Phase 3 all methods comparison saved")


def visualize_phase4_all_methods():
    """Visualize all 3 methods on Phase 4 geometric data"""
    
    # Load all Phase 4 models
    models = load_all_models('phase4')
    
    # Generate test data
    frames, coords, pattern_types = generate_geometric_test_data()
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Phase 4: All Methods on Geometric Patterns', fontsize=16, y=0.95)
    
    method_names = ['l1_baseline', 'direction_magnitude', 'direction_emphasis']
    method_labels = ['L1 Baseline', 'Direction-Magnitude', 'Direction Emphasis']
    colors = ['blue', 'red', 'purple']
    
    for row, (method, model, label, color) in enumerate(zip(method_names, [models[m] for m in method_names], method_labels, colors)):
        if model is None:
            continue
            
        for col in range(3):
            # Use frames 0-4 as input, predict frames 5-9
            input_frames = frames[col, :5]  # [5, 1, 32, 32] 
            true_coords = coords[col]  # [20, 2]
            initial_coord = true_coords[4]
            
            # Predict trajectory
            pred_coords = predict_trajectory_iterative(model, input_frames, initial_coord, num_steps=5)
            
            # Calculate error
            true_future = true_coords[5:10]  # frames 5-9
            errors = torch.norm(pred_coords[1:] - true_future, p=2, dim=1)
            mean_error = errors.mean().item()
            
            # Plot
            ax = axes[row, col]
            ax.plot(true_coords[:5, 0], true_coords[:5, 1], 'o-', color='black', 
                   markersize=3, linewidth=1, label='True Input (0-4)')
            ax.plot(true_coords[5:10, 0], true_coords[5:10, 1], 'o-', color='gray', 
                   markersize=3, linewidth=1, label='True Future (5-9)')
            ax.plot(pred_coords[1:, 0], pred_coords[1:, 1], '^-', color=color, 
                   markersize=5, linewidth=2, label=f'{label} Prediction')
            
            ax.set_xlim(0, 32)
            ax.set_ylim(0, 32)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{label} - {pattern_types[col]}\nError: {mean_error:.2f}px')
            
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('phase4_all_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Phase 4 all methods comparison saved")


def create_method_performance_summary():
    """Create a summary plot of method performance"""
    try:
        # Load results
        with open('results/real_decoupling_results.json', 'r') as f:
            phase3_results = json.load(f)
        with open('results/geometric_decoupling_results.json', 'r') as f:
            phase4_results = json.load(f)
        
        # Extract performance data
        methods = ['l1_baseline', 'direction_magnitude', 'direction_emphasis']
        method_labels = ['L1 Baseline', 'Dir-Mag Combined', 'Dir Emphasis']
        
        phase3_losses = [phase3_results['training_results'][m]['best_val_loss'] for m in methods]
        phase4_losses = [phase4_results['training_results'][m]['best_val_loss'] for m in methods]
        
        phase3_dir_acc = [phase3_results['training_results'][m]['final_direction_similarity'] for m in methods]
        phase4_dir_acc = [phase4_results['training_results'][m]['final_direction_similarity'] for m in methods]
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.arange(len(methods))
        width = 0.35
        
        # Loss comparison
        axes[0].bar(x - width/2, phase3_losses, width, label='Phase 3 (Real Data)', alpha=0.7)
        axes[0].bar(x + width/2, phase4_losses, width, label='Phase 4 (Geometric)', alpha=0.7)
        axes[0].set_xlabel('Methods')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Method Performance: Validation Loss')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(method_labels, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Direction accuracy comparison
        axes[1].bar(x - width/2, phase3_dir_acc, width, label='Phase 3 (Real Data)', alpha=0.7)
        axes[1].bar(x + width/2, phase4_dir_acc, width, label='Phase 4 (Geometric)', alpha=0.7)
        axes[1].set_xlabel('Methods')
        axes[1].set_ylabel('Direction Similarity')
        axes[1].set_title('Method Performance: Direction Accuracy')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(method_labels, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('method_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Method performance summary saved")
        
    except Exception as e:
        print(f"❌ Could not create performance summary: {e}")


def main():
    """Main visualization function"""
    print("COMPLETE METHOD COMPARISON VISUALIZATION")
    print("=" * 60)
    print("Showing all 3 methods: L1 Baseline, Direction-Magnitude, Direction Emphasis")
    print()
    
    print("1. Phase 3 models on real coordinate data...")
    visualize_phase3_all_methods()
    
    print("\n2. Phase 4 models on geometric patterns...")
    visualize_phase4_all_methods()
    
    print("\n3. Method performance summary...")
    create_method_performance_summary()
    
    print("\n✅ Complete method comparison visualization finished!")
    print("Generated files:")
    print("  - phase3_all_methods_comparison.png")
    print("  - phase4_all_methods_comparison.png") 
    print("  - method_performance_summary.png")
    
    print(f"\nNote: This shows the COMPLETE comparison of all 3 trained methods,")
    print(f"not just the single direction-magnitude method shown previously!")


if __name__ == "__main__":
    main() 