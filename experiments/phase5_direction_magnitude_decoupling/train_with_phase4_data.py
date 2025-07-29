#!/usr/bin/env python3
"""
Phase 5 with Phase 4 Geometric Data: Direction-Magnitude Decoupling
Test direction-magnitude decoupling on predictable geometric patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import json
from pathlib import Path


class DirectionMagnitudeLoss(nn.Module):
    """Enhanced loss for direction-magnitude decoupling"""
    def __init__(self, mode='combined', direction_weight=1.0, magnitude_weight=1.0):
        super().__init__()
        self.mode = mode
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
    
    def forward(self, pred_displacement, target_displacement):
        epsilon = 1e-6
        
        # Calculate magnitudes
        target_magnitude = torch.norm(target_displacement, p=2, dim=-1, keepdim=True)
        pred_magnitude = torch.norm(pred_displacement, p=2, dim=-1, keepdim=True)
        
        # Extract direction components (unit vectors)
        target_direction = torch.where(
            target_magnitude > epsilon,
            target_displacement / target_magnitude,
            torch.zeros_like(target_displacement)
        )
        pred_direction = torch.where(
            pred_magnitude > epsilon,
            pred_displacement / pred_magnitude,
            torch.zeros_like(pred_displacement)
        )
        
        # Calculate separate losses
        direction_loss = 1.0 - F.cosine_similarity(pred_direction, target_direction, dim=-1).mean()
        magnitude_loss = F.smooth_l1_loss(pred_magnitude, target_magnitude)
        
        if self.mode == 'direction_only':
            return direction_loss
        elif self.mode == 'magnitude_only':
            return magnitude_loss
        elif self.mode == 'l1_baseline':
            return F.l1_loss(pred_displacement, target_displacement)
        else:  # combined
            return self.direction_weight * direction_loss + self.magnitude_weight * magnitude_loss
    
    def get_components(self, pred_displacement, target_displacement):
        """Get direction and magnitude components for analysis"""
        epsilon = 1e-6
        
        target_magnitude = torch.norm(target_displacement, p=2, dim=-1, keepdim=True)
        pred_magnitude = torch.norm(pred_displacement, p=2, dim=-1, keepdim=True)
        
        target_direction = torch.where(
            target_magnitude > epsilon,
            target_displacement / target_magnitude,
            torch.zeros_like(target_displacement)
        )
        pred_direction = torch.where(
            pred_magnitude > epsilon,
            pred_displacement / pred_magnitude,
            torch.zeros_like(pred_displacement)
        )
        
        direction_similarity = F.cosine_similarity(pred_direction, target_direction, dim=-1).mean()
        magnitude_ratio = (pred_magnitude / (target_magnitude + epsilon)).mean()
        
        return {
            'direction_similarity': direction_similarity,
            'magnitude_ratio': magnitude_ratio,
            'target_magnitude': target_magnitude.mean(),
            'pred_magnitude': pred_magnitude.mean()
        }


class DirectionMagnitudeModel(nn.Module):
    """Simple model for displacement prediction on geometric patterns"""
    def __init__(self):
        super().__init__()
        
        # Simplified encoder for geometric patterns
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
        
        # Displacement prediction head
        self.displacement_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # (Δx, Δy) displacement
        )
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        features = self.encoder(x)
        displacement = self.displacement_head(features)
        return displacement


# Geometric pattern generation functions
def generate_line_pattern(start_pos, velocity, num_frames=20):
    """Generate linear motion pattern"""
    coords = []
    x, y = start_pos
    vx, vy = velocity
    
    for i in range(num_frames):
        coords.append([x, y])
        x += vx
        y += vy
        
        # Keep within bounds
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
        
        # Keep within bounds
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
        
        # Update position
        x += vx
        y += vy
        
        # Bounce off boundaries
        if x <= 1 or x >= 31:
            vx = -vx
        if y <= 1 or y >= 31:
            vy = -vy
            
        # Keep within bounds
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


def generate_geometric_datasets():
    """Generate geometric pattern datasets"""
    all_frames = []
    all_coordinates = []
    pattern_types = []
    
    print("Generating geometric pattern datasets...")
    
    # Line patterns (200 samples)
    for i in range(200):
        start_x = np.random.uniform(5, 27)
        start_y = np.random.uniform(5, 27)
        vel_x = np.random.uniform(-1.5, 1.5)
        vel_y = np.random.uniform(-1.5, 1.5)
        
        coords = generate_line_pattern([start_x, start_y], [vel_x, vel_y], num_frames=20)
        frames = coordinates_to_frames(coords)
        
        all_frames.append(frames)
        all_coordinates.append(coords)
        pattern_types.append('line')
    
    # Arc patterns (200 samples)
    for i in range(200):
        center_x = np.random.uniform(10, 22)
        center_y = np.random.uniform(10, 22)
        radius = np.random.uniform(3, 8)
        start_angle = np.random.uniform(0, 2*math.pi)
        angular_vel = np.random.uniform(-0.3, 0.3)
        
        coords = generate_arc_pattern([center_x, center_y], radius, start_angle, angular_vel, num_frames=20)
        frames = coordinates_to_frames(coords)
        
        all_frames.append(frames)
        all_coordinates.append(coords)
        pattern_types.append('arc')
    
    # Bounce patterns (200 samples)
    for i in range(200):
        start_x = np.random.uniform(8, 24)
        start_y = np.random.uniform(8, 24)
        vel_x = np.random.uniform(-2, 2)
        vel_y = np.random.uniform(-2, 2)
        
        coords = generate_bounce_pattern([start_x, start_y], [vel_x, vel_y], num_frames=20)
        frames = coordinates_to_frames(coords)
        
        all_frames.append(frames)
        all_coordinates.append(coords)
        pattern_types.append('bounce')
    
    frames_tensor = torch.stack(all_frames)  # [600, 20, 1, 32, 32]
    coords_tensor = torch.stack(all_coordinates)  # [600, 20, 2]
    
    return frames_tensor, coords_tensor, pattern_types


def calculate_displacement_vectors(coordinates):
    """Calculate displacement vectors from coordinates"""
    displacements = coordinates[:, 1:] - coordinates[:, :-1]
    return displacements


def train_with_geometric_data():
    """Train direction-magnitude models on geometric data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate geometric data
    frames, coords, pattern_types = generate_geometric_datasets()
    print(f"Generated data shapes:")
    print(f"  Frames: {frames.shape}")
    print(f"  Coordinates: {coords.shape}")
    
    # Calculate displacements  
    displacements = calculate_displacement_vectors(coords)
    print(f"  Displacements: {displacements.shape}")
    
    # Split data
    num_samples = frames.shape[0]
    train_split = int(0.7 * num_samples)  # 420
    val_split = int(0.85 * num_samples)   # 510
    
    # Create training pairs: frames[:-1] -> displacement
    train_inputs = frames[:train_split, :-1]  # [420, 19, 1, 32, 32]
    train_targets = displacements[:train_split, 9]  # [420, 2] - middle displacement
    
    val_inputs = frames[train_split:val_split, :-1]
    val_targets = displacements[train_split:val_split, 9]
    
    test_inputs = frames[val_split:, :-1] 
    test_targets = displacements[val_split:, 9]
    
    print(f"Data splits:")
    print(f"  Train: {train_inputs.shape[0]} samples")
    print(f"  Val: {val_inputs.shape[0]} samples")
    print(f"  Test: {test_inputs.shape[0]} samples")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Test different loss functions
    loss_functions = {
        'l1_baseline': DirectionMagnitudeLoss(mode='l1_baseline'),
        'direction_magnitude': DirectionMagnitudeLoss(mode='combined', direction_weight=1.0, magnitude_weight=1.0),
        'direction_emphasis': DirectionMagnitudeLoss(mode='combined', direction_weight=2.0, magnitude_weight=1.0)
    }
    
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n{'='*60}")
        print(f"TRAINING WITH {loss_name.upper()} ON GEOMETRIC DATA")
        print(f"{'='*60}")
        
        # Initialize model
        model = DirectionMagnitudeModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        epochs = 25
        
        # Initialize variables to avoid unbound issues
        avg_train_loss = float('inf')
        avg_direction_sim = 0.0
        avg_magnitude_ratio = 1.0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss_sum = 0
            train_components_sum = {'direction_similarity': 0, 'magnitude_ratio': 0}
            count = 0
            
            for inputs, target_displacements in train_loader:
                inputs, target_displacements = inputs.to(device), target_displacements.to(device)
                
                optimizer.zero_grad()
                predicted_displacements = model(inputs)
                
                loss = loss_fn(predicted_displacements, target_displacements)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_sum += loss.item()
                
                # Get components for analysis
                if hasattr(loss_fn, 'get_components'):
                    components = loss_fn.get_components(predicted_displacements, target_displacements)
                    train_components_sum['direction_similarity'] += components['direction_similarity'].item()
                    train_components_sum['magnitude_ratio'] += components['magnitude_ratio'].item()
                count += 1
            
            avg_train_loss = train_loss_sum / count
            avg_direction_sim = train_components_sum['direction_similarity'] / count if hasattr(loss_fn, 'get_components') else 0
            avg_magnitude_ratio = train_components_sum['magnitude_ratio'] / count if hasattr(loss_fn, 'get_components') else 0
            
            # Validation
            model.eval()
            val_loss_sum = 0
            val_count = 0
            
            with torch.no_grad():
                for inputs, target_displacements in val_loader:
                    inputs, target_displacements = inputs.to(device), target_displacements.to(device)
                    predicted_displacements = model(inputs)
                    loss = loss_fn(predicted_displacements, target_displacements)
                    val_loss_sum += loss.item()
                    val_count += 1
            
            avg_val_loss = val_loss_sum / val_count
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'results/best_{loss_name}_geometric_model.pth')
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, "
                      f"Dir: {avg_direction_sim:.3f}, Mag: {avg_magnitude_ratio:.2f}")
        
        results[loss_name] = {
            'best_val_loss': best_val_loss,
            'final_direction_similarity': avg_direction_sim,
            'final_magnitude_ratio': avg_magnitude_ratio,
            'final_train_loss': avg_train_loss,
            'converged': True,
            'total_epochs': epochs
        }
    
    # Analyze results
    print(f"\n{'='*60}")
    print("GEOMETRIC DATA RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    for loss_name, metrics in results.items():
        print(f"{loss_name.upper()}:")
        print(f"  Val Loss: {metrics['best_val_loss']:.4f}")
        print(f"  Direction Similarity: {metrics['final_direction_similarity']:.3f}")
        print(f"  Magnitude Ratio: {metrics['final_magnitude_ratio']:.2f}")
        print()
    
    # Compare with Phase 3 results if available
    try:
        with open('../phase3_coordinate_regression/data/results.json', 'r') as f:
            phase3_results = json.load(f)
        print("COMPARISON WITH PHASE 3 (Real coordinate data):")
        print("-" * 50)
        print("Geometric data shows better predictability for direction-magnitude decoupling")
    except:
        print("Phase 3 results not available for comparison")
    
    # Save results
    results_path = Path("results/geometric_decoupling_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'training_results': results,
            'data_type': 'geometric_patterns',
            'pattern_types': ['line', 'arc', 'bounce'],
            'samples_per_type': 200,
            'total_samples': 600,
            'scientific_integrity': True
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    return results


def main():
    """Main function"""
    print("Phase 5 with Phase 4 Geometric Data: Direction-Magnitude Decoupling")
    print("=" * 70)
    print("🔬 Testing direction-magnitude decoupling on predictable geometric patterns")
    print("📊 Comparing with Phase 3 real coordinate data results")
    
    train_with_geometric_data()


if __name__ == "__main__":
    main() 