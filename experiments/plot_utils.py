#!/usr/bin/env python3
"""
Unified plotting utilities for different representation types.

Three specialized plotters:
- PixelPlotter: Binary frames with single pred/gt markers
- HeatPlotter: Normalized heatmaps with markers
- CoordPlotter: Trajectory-only visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


class BasePlotter:
    """Base class for representation-specific plotters."""
    
    def __init__(self, img_size: int = 32, pre_seq_length: int = 10, exp_name: str = ""):
        self.img_size = img_size
        self.pre_seq_length = pre_seq_length
        self.exp_name = exp_name
        self.dpi = 150
        
    def plot_trajectory(self, 
                       true_coords: torch.Tensor,
                       pred_coords: torch.Tensor, 
                       sample_idx: int,
                       save_path: Optional[Path] = None,
                       valid_mask: Optional[torch.Tensor] = None) -> None:
        """Plot trajectory visualization with points at each frame."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        # Convert to numpy
        true_coords_np = true_coords.numpy()
        pred_coords_np = pred_coords.numpy()
        
        # Input trajectory (first pre_seq_length frames or available length)
        input_end = min(self.pre_seq_length, len(true_coords_np))
        input_coords = true_coords_np[:input_end]
        ax.plot(input_coords[:, 0], input_coords[:, 1], '-', 
               color='black', linewidth=2, label='Input', alpha=0.8)
        # Add points for each input frame
        ax.scatter(input_coords[:, 0], input_coords[:, 1], 
                  color='black', s=30, alpha=0.6, zorder=3)
        
        # Ground truth future trajectory (dashed) - use all predicted horizon length
        pred_horizon = len(pred_coords_np)
        true_future = true_coords_np[input_end:input_end + pred_horizon]
        ax.plot(true_future[:, 0], true_future[:, 1], ':',
               color='blue', linewidth=2.5, label='Ground Truth', alpha=0.7)
        # Add points for each ground truth frame - smaller and more transparent
        ax.scatter(true_future[:, 0], true_future[:, 1],
                  color='blue', s=20, alpha=0.4, zorder=3)
        
        # Predicted trajectory - show full horizon
        ax.plot(pred_coords_np[:, 0], pred_coords_np[:, 1], '-',
               color='red', linewidth=2, label='Prediction', alpha=0.7)
        # Add points for each predicted frame - with higher zorder to be on top
        ax.scatter(pred_coords_np[:, 0], pred_coords_np[:, 1],
                  color='red', s=40, alpha=0.8, zorder=5)
        # Add frame numbers
        for i, (x, y) in enumerate(pred_coords_np):
            ax.annotate(str(i+1), (x, y), xytext=(2, 2), textcoords='offset points', 
                       fontsize=6, color='darkred', alpha=0.7)
        
        # Format
        ax.set_xlim(0, self.img_size)
        ax.set_ylim(0, self.img_size)
        ax.invert_yaxis()
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        title = f'Sample {sample_idx + 1} Trajectory ({self.__class__.__name__.replace("Plotter", "")})'
        if self.exp_name:
            title += f'\n{self.exp_name}'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class PixelPlotter(BasePlotter):
    """Plotter for pixel/binary representation."""
    
    def plot_frames(self,
                   input_frames: torch.Tensor,
                   pred_frames: torch.Tensor,
                   true_coords: torch.Tensor,
                   pred_coords: torch.Tensor,
                   sample_idx: int,
                   save_path: Optional[Path] = None,
                   valid_mask: Optional[torch.Tensor] = None) -> None:
        """Plot frame grid for pixel representation."""
        n_in = min(self.pre_seq_length, len(input_frames))
        n_out = len(pred_frames)
        n_cols = max(n_in, n_out) if max(n_in, n_out) > 0 else 1
        fig_width = max(10, 2 * n_cols)
        fig, axes = plt.subplots(3, n_cols, figsize=(fig_width, 7))
        if n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]], [axes[2]]])  # Normalize indexing when single column
        fig.suptitle(f'Sample {sample_idx + 1}: Pixel Input and Output', fontsize=14)
        
        # Input frames (raw binary)
        for t in range(n_in):
            if t < len(input_frames):
                frame = input_frames[t, 0].numpy()
                vmax = float(frame.max()) if frame.max() > 0 else 1.0
                axes[0, t].imshow(frame / (vmax if vmax > 0 else 1.0), cmap='gray', vmin=0, vmax=1)
            axes[0, t].set_title(f'Input {t+1}')
            axes[0, t].axis('off')
        
        # Raw predicted frames (without normalization)
        for t in range(n_out):
            if t < len(pred_frames):
                # Apply sigmoid for display
                frame = torch.sigmoid(pred_frames[t, 0]).numpy()
                axes[1, t].imshow(frame, cmap='gray', vmin=0, vmax=1)
            axes[1, t].set_title(f'Raw {t+1}', fontsize=10)
            axes[1, t].axis('off')
        
        # Processed predicted frames with markers
        for t in range(n_out):
            if t < len(pred_frames):
                # Apply sigmoid for display
                frame = torch.sigmoid(pred_frames[t, 0]).numpy()
                vmax = float(frame.max()) if frame.max() > 0 else 1.0
                axes[2, t].imshow(frame / (vmax if vmax > 0 else 1.0), cmap='gray', vmin=0, vmax=1)
                
                # Add single pred and gt markers (smaller)
                if (valid_mask is None or (t < len(valid_mask) and valid_mask[t])) and t < len(pred_coords):
                    pred_coord = pred_coords[t].numpy()
                    axes[2, t].plot(pred_coord[0], pred_coord[1], 'r+', 
                                  markersize=6, markeredgewidth=1, alpha=0.7, label='Prediction' if t == 0 else "_")
                
                if (self.pre_seq_length + t) < len(true_coords):
                    true_coord = true_coords[self.pre_seq_length + t].numpy()
                    axes[2, t].plot(true_coord[0], true_coord[1], 'gx',
                                  markersize=6, markeredgewidth=1, alpha=0.7, label='Ground Truth' if t == 0 else "_")
                
            axes[2, t].set_title(f'Pred {t+1}', fontsize=10)
            axes[2, t].axis('off')
        
        # Add legend inside the first prediction frame
        handles, labels = axes[2, 0].get_legend_handles_labels()
        if not handles:
            axes[2, 0].plot([], [], 'r+', label='Prediction')
            axes[2, 0].plot([], [], 'gx', label='Ground Truth')
        axes[2, 0].legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        # Add experiment name at the bottom
        if self.exp_name:
            fig.text(0.5, 0.01, self.exp_name, ha='center', va='bottom', 
                    fontsize=12, fontweight='bold',  transform=fig.transFigure)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class HeatPlotter(BasePlotter):
    """Plotter for heatmap/Gaussian representation."""
    
    def plot_frames(self,
                   input_frames: torch.Tensor,
                   pred_frames: torch.Tensor,
                   true_coords: torch.Tensor,
                   pred_coords: torch.Tensor,
                   sample_idx: int,
                   save_path: Optional[Path] = None,
                   valid_mask: Optional[torch.Tensor] = None) -> None:
        """Plot frame grid for heatmap representation with raw output."""
        n_in = min(self.pre_seq_length, len(input_frames))
        n_out = len(pred_frames)
        n_cols = max(n_in, n_out) if max(n_in, n_out) > 0 else 1
        fig_width = max(10, 2 * n_cols)
        fig, axes = plt.subplots(3, n_cols, figsize=(fig_width, 7))
        if n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]], [axes[2]]])
        fig.suptitle(f'Sample {sample_idx + 1}: Heatmap Input and Output', fontsize=14)
        
        # Input frames (show heatmaps)
        for t in range(n_in):
            if t < len(input_frames):
                frame = input_frames[t, 0].numpy()
                # Show heatmap with grayscale
                axes[0, t].imshow(frame, cmap='gray', vmin=0, vmax=frame.max() if frame.max() > 0 else 1.0)
            axes[0, t].set_title(f'Input {t+1}', fontsize=10)
            axes[0, t].axis('off')
        
        # Raw predicted frames (show raw heatmap output)
        for t in range(n_out):
            if t < len(pred_frames):
                frame = pred_frames[t, 0].numpy()
                # Handle negative values (log probabilities from KL loss)
                if frame.max() < 0:
                    frame = np.exp(frame)  # Convert log probabilities to probabilities
                axes[1, t].imshow(frame, cmap='gray', vmin=0, vmax=frame.max() if frame.max() > 0 else 1.0)
            axes[1, t].set_title(f'Raw {t+1}', fontsize=10)
            axes[1, t].axis('off')
        
        # Processed predicted frames - normalized and with markers
        for t in range(n_out):
            if t < len(pred_frames):
                frame = pred_frames[t, 0].numpy()
                # Handle negative values (log probabilities from KL loss)
                if frame.max() < 0:
                    frame = np.exp(frame)  # Convert log probabilities to probabilities
                vmax = float(frame.max()) if frame.max() > 0 else 1.0
                im = axes[2, t].imshow(frame / (vmax if vmax > 0 else 1.0), cmap='gray', vmin=0, vmax=1.0)
                # markers only on valid frames
                if (valid_mask is None or (t < len(valid_mask) and valid_mask[t])) and t < len(pred_coords):
                    pred_coord = pred_coords[t].numpy()
                    axes[2, t].plot(pred_coord[0], pred_coord[1], 'r+',
                                  markersize=6, markeredgewidth=1, alpha=0.7)
                
                if (self.pre_seq_length + t) < len(true_coords):
                    true_coord = true_coords[self.pre_seq_length + t].numpy()
                    axes[2, t].plot(true_coord[0], true_coord[1], 'gx',
                                  markersize=6, markeredgewidth=1, alpha=0.7)
                
            axes[2, t].set_title(f'Pred {t+1}', fontsize=10)
            axes[2, t].axis('off')
        
        # Legend inside first pred frame (force even if no markers in panel)
        handles, labels = axes[2, 0].get_legend_handles_labels()
        if not handles:
            axes[2, 0].plot([], [], 'r+', label='Prediction')
            axes[2, 0].plot([], [], 'gx', label='Ground Truth')
        axes[2, 0].legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        # Add experiment name at the bottom
        if self.exp_name:
            fig.text(0.5, 0.01, self.exp_name, ha='center', va='bottom', 
                    fontsize=13, fontweight='bold', transform=fig.transFigure)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class CoordPlotter(BasePlotter):
    """Plotter for coordinate representation - trajectory only."""
    
    def plot_frames(self,
                   input_coords: torch.Tensor,
                   pred_coords: torch.Tensor,
                   true_coords: torch.Tensor,
                   sample_idx: int,
                   save_path: Optional[Path] = None) -> None:
        """For coord representation, we only plot trajectory."""
        # Redirect to trajectory plot
        self.plot_trajectory(true_coords, pred_coords, sample_idx, save_path)


def create_plotter(representation: str, img_size: int = 32, pre_seq_length: int = 10, exp_name: str = "") -> BasePlotter:
    """Factory function to create appropriate plotter."""
    if representation == 'pixel':
        return PixelPlotter(img_size, pre_seq_length, exp_name)
    elif representation == 'heat':
        return HeatPlotter(img_size, pre_seq_length, exp_name)
    elif representation == 'coord':
        return CoordPlotter(img_size, pre_seq_length, exp_name)
    else:
        raise ValueError(f"Unknown representation: {representation}")