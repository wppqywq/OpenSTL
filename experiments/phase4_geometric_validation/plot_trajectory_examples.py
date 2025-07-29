#!/usr/bin/env python3
"""
Plot Phase 4 Geometric Pattern Trajectory Examples
Show Line, Arc, and Bounce patterns in a single row
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

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

def plot_phase4_trajectories():
    """Plot three types of geometric patterns in a single row"""
    
    # Generate sample patterns
    line_coords = generate_line_pattern([8, 8], [1.2, 0.8], num_frames=20)
    arc_coords = generate_arc_pattern([16, 16], 8, 0, 0.25, num_frames=20)
    bounce_coords = generate_bounce_pattern([12, 25], [1.5, -2.0], num_frames=20)
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Phase 4 Geometric Pattern Examples', fontsize=16, fontweight='bold')
    
    patterns = [
        (line_coords, 'Line Pattern', 'Constant velocity movement'),
        (arc_coords, 'Arc Pattern', 'Circular motion'),
        (bounce_coords, 'Bounce Pattern', 'Bouncing with velocity reversals')
    ]
    
    for i, (coords, title, description) in enumerate(patterns):
        ax = axes[i]
        coords_np = coords.numpy()
        
        # Plot the full trajectory
        ax.plot(coords_np[:, 0], coords_np[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # Mark input sequence (frames 0-9)
        input_coords = coords_np[:10]
        ax.plot(input_coords[:, 0], input_coords[:, 1], 'ko-', 
                markersize=6, linewidth=3, label='Input (frames 0-9)')
        
        # Mark future sequence (frames 10-19)
        future_coords = coords_np[10:]
        ax.plot(future_coords[:, 0], future_coords[:, 1], 'ro-', 
                markersize=6, linewidth=3, label='Future (frames 10-19)')
        
        # Mark start and end points
        ax.plot(coords_np[0, 0], coords_np[0, 1], 'gs', markersize=10, label='Start')
        ax.plot(coords_np[-1, 0], coords_np[-1, 1], 'rs', markersize=10, label='End')
        
        # Add frame numbers
        for frame_idx in [0, 9, 10, 19]:
            if frame_idx < len(coords_np):
                ax.annotate(f'{frame_idx}', 
                           (coords_np[frame_idx, 0], coords_np[frame_idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'{title}\n{description}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "experiments/phase4_geometric_validation/results/phase4_trajectory_examples.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Phase 4 trajectory examples saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Generating Phase 4 Geometric Pattern Trajectory Examples")
    print("=" * 60)
    
    try:
        plot_phase4_trajectories()
        print("✅ Trajectory visualization generated successfully")
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        import traceback
        traceback.print_exc() 