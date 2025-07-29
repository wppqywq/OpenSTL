#!/usr/bin/env python3
"""
Evaluate trained models and visualize raw predictions.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import config

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from openstl.api import BaseExperiment
from openstl.models import SimVP
from torch.utils.data import DataLoader, TensorDataset


def load_model(model_type, device):
    """Load trained model"""
    model_suffix = f"_{model_type}" if model_type in ['with_history', 'no_history'] else ""
    model_path = f"models/best_gaussian_enhanced{model_suffix}.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    # Create model
    model = SimVP(
        in_shape=(4, 1, config.img_size, config.img_size),
        pre_seq_length=4,
        aft_seq_length=16,
        hid_S=config.model_hid_S,
        hid_T=config.model_hid_T,
        N_S=config.model_N_S,
        N_T=config.model_N_T,
        model_type=config.model_type,
        lr=1e-3,
        dataname='gaussian_enhanced',
        metrics=[],
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded model: {model_path}")
    return model


def load_test_data(model_type):
    """Load test data"""
    history_suffix = "_no_history" if model_type == 'no_history' else "_with_history"
    data_path = f"data/test_data{history_suffix}.pt"
    
    data = torch.load(data_path)
    frames = data['frames'].permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)
    
    # Split into input and target
    inp = frames[:, :4]   # First 4 frames as input
    tgt = frames[:, 4:20] # Next 16 frames as target
    
    return TensorDataset(inp.float(), tgt.float()), data['coords']


def evaluate_models():
    """Evaluate both models and compare"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("EVALUATING TRAINED MODELS")
    print("="*60)
    
    results = {}
    
    for model_type in ['no_history', 'with_history']:
        print(f"\nEvaluating {model_type} model...")
        
        # Load model and data
        model = load_model(model_type, device)
        if model is None:
            continue
            
        test_dataset, coords = load_test_data(model_type)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Evaluate
        model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                
                # Simple MSE loss for evaluation
                loss = torch.nn.functional.mse_loss(pred, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                num_samples += batch_x.size(0)
        
        avg_loss = total_loss / num_samples
        results[model_type] = {
            'test_loss': avg_loss,
            'model': model,
            'coords': coords,
            'test_loader': test_loader
        }
        
        print(f"  Test Loss: {avg_loss:.6f}")
    
    return results, device


def plot_training_curves():
    """Plot training curves"""
    # Load training logs
    log_no_history = pd.read_csv('training_log_gaussian_no_history.txt')
    log_with_history = pd.read_csv('training_log_gaussian_with_history.txt')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training loss
    axes[0].plot(log_no_history['Epoch'], log_no_history['Train_Loss'], 
                'b-', label='No History - Train', alpha=0.8)
    axes[0].plot(log_no_history['Epoch'], log_no_history['Val_Loss'], 
                'b--', label='No History - Val', alpha=0.8)
    axes[0].plot(log_with_history['Epoch'], log_with_history['Train_Loss'], 
                'r-', label='With History - Train', alpha=0.8)
    axes[0].plot(log_with_history['Epoch'], log_with_history['Val_Loss'], 
                'r--', label='With History - Val', alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final comparison
    final_results = {
        'No History': {
            'Best Val': log_no_history['Val_Loss'].min(),
            'Final Train': log_no_history['Train_Loss'].iloc[-1],
            'Final Val': log_no_history['Val_Loss'].iloc[-1]
        },
        'With History': {
            'Best Val': log_with_history['Val_Loss'].min(),
            'Final Train': log_with_history['Train_Loss'].iloc[-1],
            'Final Val': log_with_history['Val_Loss'].iloc[-1]
        }
    }
    
    models = list(final_results.keys())
    metrics = list(final_results['No History'].keys())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    values_no = [final_results['No History'][m] for m in metrics]
    values_with = [final_results['With History'][m] for m in metrics]
    
    axes[1].bar(x - width/2, values_no, width, label='No History', alpha=0.8)
    axes[1].bar(x + width/2, values_with, width, label='With History', alpha=0.8)
    
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Loss Value')
    axes[1].set_title('Final Performance Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(values_no):
        axes[1].text(i - width/2, v + 1, f'{v:.1f}', ha='center', va='bottom')
    for i, v in enumerate(values_with):
        axes[1].text(i + width/2, v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print numerical comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    for model_name, metrics in final_results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")


def visualize_predictions(results, device, num_samples=3):
    """Visualize raw predictions vs targets"""
    print("\n" + "="*60)
    print("VISUALIZING RAW PREDICTIONS")
    print("="*60)
    
    for model_type in ['no_history', 'with_history']:
        if model_type not in results:
            continue
            
        model = results[model_type]['model']
        test_loader = results[model_type]['test_loader']
        coords = results[model_type]['coords']
        
        print(f"\nGenerating predictions for {model_type} model...")
        
        # Get first batch
        batch_x, batch_y = next(iter(test_loader))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        with torch.no_grad():
            pred = model(batch_x).cpu()
        
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        
        # Visualize first few samples
        for sample_idx in range(min(num_samples, batch_x.size(0))):
            visualize_single_sample(batch_x[sample_idx], batch_y[sample_idx], 
                                   pred[sample_idx], coords[sample_idx], 
                                   model_type, sample_idx)


def visualize_single_sample(input_seq, target_seq, pred_seq, coords, model_type, sample_idx):
    """Visualize a single sample prediction"""
    # Remove channel dimension
    input_frames = input_seq.squeeze(1)  # [4, H, W]
    target_frames = target_seq.squeeze(1)  # [16, H, W]
    pred_frames = pred_seq.squeeze(1)  # [16, H, W]
    
    # Show key frames: input frames + first few predictions
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    
    # Show input frames (first 4)
    for t in range(4):
        axes[0, t].imshow(input_frames[t], cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f'Input {t+1}')
        axes[0, t].axis('off')
        
        # Plot trajectory point on input
        if t < len(coords):
            y, x = coords[t][0].item(), coords[t][1].item()
            axes[0, t].plot(x, y, 'ro', markersize=3)
    
    # Clear unused input slots
    for t in range(4, 8):
        axes[0, t].axis('off')
    
    # Show target vs prediction (first 8 prediction frames)
    for t in range(8):
        # Target
        axes[1, t].imshow(target_frames[t], cmap='gray', vmin=0, vmax=1)
        axes[1, t].set_title(f'Target {t+5}')
        axes[1, t].axis('off')
        
        # Plot trajectory point on target
        if t + 4 < len(coords):
            y, x = coords[t + 4][0].item(), coords[t + 4][1].item()
            axes[1, t].plot(x, y, 'go', markersize=3)
        
        # Prediction
        axes[2, t].imshow(pred_frames[t], cmap='gray', vmin=0, vmax=1)
        axes[2, t].set_title(f'Pred {t+5}')
        axes[2, t].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Input\nFrames', transform=axes[0, 0].transAxes,
                    ha='right', va='center', fontsize=12, weight='bold')
    axes[1, 0].text(-0.1, 0.5, 'Target\nFrames', transform=axes[1, 0].transAxes,
                    ha='right', va='center', fontsize=12, weight='bold')
    axes[2, 0].text(-0.1, 0.5, 'Predicted\nFrames', transform=axes[2, 0].transAxes,
                    ha='right', va='center', fontsize=12, weight='bold')
    
    plt.suptitle(f'{model_type.title()} Model - Sample {sample_idx+1}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Compute and display error statistics
    mse_error = torch.mean((target_frames[:8] - pred_frames[:8])**2)
    mae_error = torch.mean(torch.abs(target_frames[:8] - pred_frames[:8]))
    
    print(f"Sample {sample_idx+1} - {model_type}:")
    print(f"  MSE Error: {mse_error:.6f}")
    print(f"  MAE Error: {mae_error:.6f}")
    print(f"  Target intensity range: [{target_frames.min():.3f}, {target_frames.max():.3f}]")
    print(f"  Prediction intensity range: [{pred_frames.min():.3f}, {pred_frames.max():.3f}]")


def main():
    """Main evaluation and visualization"""
    print("GAUSSIAN ENHANCED MODEL EVALUATION")
    print("Post-Training Analysis and Visualization")
    
    # 1. Plot training curves
    plot_training_curves()
    
    # 2. Evaluate models
    results, device = evaluate_models()
    
    # 3. Visualize predictions
    if results:
        visualize_predictions(results, device, num_samples=2)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(" Training curves plotted")
    print(" Models evaluated on test set")
    print(" Raw predictions visualized")
    print(" Performance comparison completed")


if __name__ == "__main__":
    main() 