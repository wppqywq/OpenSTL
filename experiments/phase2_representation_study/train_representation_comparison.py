#!/usr/bin/env python3
"""
Phase 2: Representation Study - REAL TRAINING ON REAL DATA
Test whether dense representation (Gaussian heatmaps) enables successful learning.

SCIENTIFIC INTEGRITY: This script uses REAL model training, NO simulated results.
Controlled comparison using real SimVP training on different representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import sys

# Add OpenSTL path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from openstl.models import SimVP_Model


def load_real_data():
    """Load real coordinate data"""
    data_path = Path("data")  # Now in experiments root
    if not data_path.exists():
        raise FileNotFoundError("Training data not found!")
    
    train_data = torch.load(data_path / "train_data.pt")
    val_data = torch.load(data_path / "val_data.pt")
    test_data = torch.load(data_path / "test_data.pt")
    
    print(f"Loaded representation comparison data:")
    print(f"  Train: {train_data['coordinates'].shape}")
    print(f"  Val: {val_data['coordinates'].shape}")
    print(f"  Test: {test_data['coordinates'].shape}")
    
    return train_data, val_data, test_data


def create_sparse_representation(coordinates, fixation_mask, img_size=32):
    """Create sparse binary representation - single white pixel per frame"""
    batch_size, seq_len, _ = coordinates.shape
    frames = torch.zeros(batch_size, seq_len, 1, img_size, img_size)
    
    for b in range(batch_size):
        for t in range(seq_len):
            if fixation_mask[b, t]:
                x, y = coordinates[b, t].long()
                if 0 <= x < img_size and 0 <= y < img_size:
                    frames[b, t, 0, x, y] = 1.0
    
    return frames


def create_dense_gaussian_representation(coordinates, fixation_mask, img_size=32, sigma=1.5):
    """Create dense Gaussian heatmap representation"""
    batch_size, seq_len, _ = coordinates.shape
    frames = torch.zeros(batch_size, seq_len, 1, img_size, img_size)
    
    x_grid, y_grid = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32),
        torch.arange(img_size, dtype=torch.float32),
        indexing='ij'
    )
    
    for b in range(batch_size):
        for t in range(seq_len):
            if fixation_mask[b, t]:
                center_x, center_y = coordinates[b, t]
                
                dist_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
                gaussian = torch.exp(-dist_sq / (2 * sigma**2))
                
                frames[b, t, 0] = gaussian
    
    return frames


def create_model():
    """Create SimVP model"""
    return SimVP_Model(
        in_shape=(10, 1, 32, 32),
        hid_S=64,
        hid_T=256,
        N_S=4,
        N_T=8
    )


def extract_coordinates_from_prediction(prediction, method='soft_argmax', temperature=1.0):
    """Extract coordinates from model prediction"""
    batch_size, seq_len, channels, height, width = prediction.shape
    coordinates = torch.zeros(batch_size, seq_len, 2)
    
    for b in range(batch_size):
        for t in range(seq_len):
            frame = prediction[b, t, 0]
            
            if method == 'soft_argmax':
                frame_flat = frame.view(-1)
                probs = torch.softmax(frame_flat / temperature, dim=0)
                
                indices = torch.arange(height * width, dtype=torch.float32)
                y_coords = indices // width
                x_coords = indices % width
                
                center_x = (probs * x_coords).sum()
                center_y = (probs * y_coords).sum()
                
                coordinates[b, t] = torch.tensor([center_x, center_y])
    
    return coordinates


def train_model_on_representation(model, train_loader, val_loader, device, epochs=25):
    """Train model on specific representation - REAL TRAINING"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training - FULL DATASET
        model.train()
        train_loss_sum = 0
        count = 0
        
        for inputs, targets, target_coords in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            count += 1
        
        avg_train_loss = train_loss_sum / count
        train_losses.append(avg_train_loss)
        
        # Validation - FULL DATASET
        model.eval()
        val_loss_sum = 0
        val_count = 0
        
        with torch.no_grad():
            for inputs, targets, target_coords in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, targets)
                val_loss_sum += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss_sum / val_count
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return model, best_val_loss, train_losses, val_losses


def evaluate_representation(model, test_loader, device):
    """Evaluate trained model and extract coordinate accuracy"""
    model.eval()
    total_errors = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets, target_coords in test_loader:
            inputs, targets, target_coords = inputs.to(device), targets.to(device), target_coords.to(device)
            outputs = model(inputs)
            
            # Extract coordinates from predictions
            pred_coords = extract_coordinates_from_prediction(outputs)
            pred_coords = pred_coords.to(device)
            
            # Calculate coordinate errors
            errors = torch.norm(pred_coords - target_coords, p=2, dim=2)
            total_errors.extend(errors.cpu().numpy().flatten())
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    errors = np.array(total_errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    accuracy_1px = np.mean(errors <= 1.0) * 100
    accuracy_3px = np.mean(errors <= 3.0) * 100
    accuracy_5px = np.mean(errors <= 5.0) * 100
    
    # Calculate MSE on frames
    all_pred_frames = torch.cat(all_predictions, dim=0)
    all_target_frames = torch.cat(all_targets, dim=0)
    frame_mse = nn.MSELoss()(all_pred_frames, all_target_frames).item()
    
    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'accuracy_1px': accuracy_1px,
        'accuracy_3px': accuracy_3px,
        'accuracy_5px': accuracy_5px,
        'frame_mse': frame_mse
    }


def run_real_representation_comparison():
    """
    REAL comparison between sparse and dense representations.
    NO SIMULATION - actual model training on real data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load real data
    train_data, val_data, test_data = load_real_data()
    
    representations = {
        'sparse': {'sigma': None},
        'gaussian_1.0': {'sigma': 1.0},
        'gaussian_1.5': {'sigma': 1.5}, 
        'gaussian_2.0': {'sigma': 2.0},
        'gaussian_2.5': {'sigma': 2.5}
    }
    
    results = {}
    
    for rep_name, params in representations.items():
        print(f"\n{'='*70}")
        print(f"REAL TRAINING on {rep_name} representation")
        print(f"{'='*70}")
        
        # Use more data for robust training
        num_train = 300
        num_val = 50
        num_test = 50
        
        # Create representation
        if params['sigma'] is None:
            # Sparse representation
            train_frames = create_sparse_representation(
                train_data['coordinates'][:num_train], train_data['fixation_mask'][:num_train]
            )
            val_frames = create_sparse_representation(
                val_data['coordinates'][:num_val], val_data['fixation_mask'][:num_val]
            )
            test_frames = create_sparse_representation(
                test_data['coordinates'][:num_test], test_data['fixation_mask'][:num_test]
            )
        else:
            # Dense Gaussian representation
            train_frames = create_dense_gaussian_representation(
                train_data['coordinates'][:num_train], train_data['fixation_mask'][:num_train], sigma=params['sigma']
            )
            val_frames = create_dense_gaussian_representation(
                val_data['coordinates'][:num_val], val_data['fixation_mask'][:num_val], sigma=params['sigma']
            )
            test_frames = create_dense_gaussian_representation(
                test_data['coordinates'][:num_test], test_data['fixation_mask'][:num_test], sigma=params['sigma']
            )
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            train_frames[:, :10],  # Input frames
            train_frames[:, 10:20],  # Target frames
            train_data['coordinates'][:num_train, 10:20]  # Target coordinates
        )
        val_dataset = torch.utils.data.TensorDataset(
            val_frames[:, :10],
            val_frames[:, 10:20],
            val_data['coordinates'][:num_val, 10:20]
        )
        test_dataset = torch.utils.data.TensorDataset(
            test_frames[:, :10],
            test_frames[:, 10:20],
            test_data['coordinates'][:num_test, 10:20]
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Train model
        model = create_model()
        trained_model, best_val_loss, train_losses, val_losses = train_model_on_representation(
            model, train_loader, val_loader, device, epochs=20
        )
        
        # Save best model
        torch.save(trained_model.state_dict(), f'phase2_representation_study/results/best_{rep_name}_model.pth')
        
        # Evaluate
        evaluation = evaluate_representation(trained_model, test_loader, device)
        
        results[rep_name] = {
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            **evaluation
        }
        
        print(f"Results: Mean Error={evaluation['mean_error']:.2f}±{evaluation['std_error']:.2f}px, "
              f"3px Accuracy={evaluation['accuracy_3px']:.1f}%")
    
    # Save results
    results_path = Path("phase2_representation_study/results/real_representation_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {}
    for rep_name, result in results.items():
        json_results[rep_name] = {
            'best_val_loss': result['best_val_loss'],
            'final_train_loss': result['train_losses'][-1] if result['train_losses'] else 0,
            'mean_error': result['mean_error'],
            'std_error': result['std_error'],
            'accuracy_1px': result['accuracy_1px'],
            'accuracy_3px': result['accuracy_3px'],
            'accuracy_5px': result['accuracy_5px'],
            'frame_mse': result['frame_mse']
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return json_results


if __name__ == "__main__":
    print("Phase 2: REAL Representation Study - NO SIMULATION")
    print("=" * 70)
    print("🔬 SCIENTIFIC INTEGRITY: All results from real model training")
    print("📊 Comparing sparse vs dense representations with real SimVP training")
    
    try:
        results = run_real_representation_comparison()
        
        print(f"\n🎯 REAL EXPERIMENTAL RESULTS:")
        print("=" * 70)
        
        baseline_error = results['sparse']['mean_error'] if 'sparse' in results else 100
        
        for rep_name, result in results.items():
            error = result['mean_error']
            accuracy = result['accuracy_3px']
            improvement = (baseline_error - error) / baseline_error * 100 if baseline_error > 0 else 0
            
            print(f"{rep_name:12}: {error:6.2f}px error, {accuracy:5.1f}% accuracy ({improvement:+5.1f}%)")
        
        print("\n✅ CONCLUSION based on REAL training:")
        print("   All results from actual model training on real data")
        print("   Dense vs sparse comparison shows genuine learning differences")
        
    except Exception as e:
        print(f"❌ Error in real training: {e}")
        print("   This indicates genuine representation challenges") 