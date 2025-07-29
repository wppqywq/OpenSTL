import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the root directory to the Python path to import from openstl
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from openstl.models import SimVP_Model

class SimVPWithTaskHead(nn.Module):
    """
    SimVP model with different task heads.
    
    This model wraps the base SimVP model from OpenSTL and adds task-specific heads:
    - 'pixel': Binary pixel prediction (for sparse representation)
    - 'heat': Gaussian heatmap prediction (for dense representation)
    - 'coord': Coordinate displacement prediction (for vector representation)
    """
    
    def __init__(self, in_shape, hid_S=64, hid_T=512, N_S=4, N_T=8, 
                 model_type='gSTA', task='pixel', out_shape=None):
        """
        Initialize the model.
        
        Args:
            in_shape (tuple): Input shape (T, C, H, W).
            hid_S (int): Hidden dimension of spatial encoder.
            hid_T (int): Hidden dimension of temporal encoder.
            N_S (int): Number of spatial encoder blocks.
            N_T (int): Number of temporal encoder blocks.
            model_type (str): Type of SimVP model ('gSTA', 'IncepU', etc.).
            task (str): Task type ('pixel', 'heat', or 'coord').
            out_shape (tuple, optional): Output shape. If None, uses in_shape.
        """
        super().__init__()
        
        self.task = task
        
        # Set output shape if not provided
        if out_shape is None:
            out_shape = in_shape
        
        # Create base SimVP model
        self.base_model = SimVP_Model(
            in_shape=in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            model_type=model_type
        )
        
        # Create task-specific heads
        if task in ['pixel', 'heat']:
            # For pixel and heatmap tasks, use the default SimVP decoder
            # The base model already has a decoder for these tasks
            pass
        elif task == 'coord':
            # For coordinate task, replace the decoder with a coordinate regression head
            
            # Get the feature dimension from the encoder
            # The feature dimension is determined by the hidden dimension of the temporal encoder
            feature_dim = hid_T
            
            # Create a coordinate regression head
            self.coord_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # Output (Δx, Δy)
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor, shape depends on the task:
                - 'pixel'/'heat': (B, T_out, C, H, W)
                - 'coord': (B, 2) for displacement vector (Δx, Δy)
        """
        if self.task in ['pixel', 'heat']:
            # For pixel and heatmap tasks, use the base model directly
            return self.base_model(x)
        
        elif self.task == 'coord':
            # For coordinate task, extract features and apply coordinate head
            
            # Get features from the encoder
            # Use the forward method with mid_output flag
            features = self.base_model(x, return_mid=True)
            
            # Extract the features from the last time step
            # features shape: (B, T, C, H, W) -> (B, C, H, W)
            last_frame_features = features[:, -1]
            
            # Global average pooling to get a feature vector
            # (B, C, H, W) -> (B, C)
            pooled_features = F.adaptive_avg_pool2d(last_frame_features, (1, 1)).view(x.size(0), -1)
            
            # Apply coordinate head to get displacement vector
            displacement = self.coord_head(pooled_features)
            
            return displacement
    
    def extract_coordinates_from_heatmap(self, heatmap):
        """
        Extract coordinates from a heatmap using argmax.
        
        Args:
            heatmap (torch.Tensor): Heatmap tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Coordinates tensor of shape (B, 2).
        """
        B, C, H, W = heatmap.shape
        
        # Flatten the spatial dimensions
        flat_heatmap = heatmap.view(B, C, -1)
        
        # Find the indices of the maximum values
        max_indices = torch.argmax(flat_heatmap, dim=2)
        
        # Convert indices to (x, y) coordinates
        y_coords = max_indices // W
        x_coords = max_indices % W
        
        # Stack coordinates
        coords = torch.stack([x_coords, y_coords], dim=2).float().squeeze(1)  # (B, 2)
        
        return coords
    
    def extract_coordinates_center_of_mass(self, heatmap):
        """
        Extract coordinates from a heatmap using center of mass (weighted average).
        This is more robust than argmax but doesn't use softmax.
        
        Args:
            heatmap (torch.Tensor): Heatmap tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Coordinates tensor of shape (B, 2).
        """
        B, C, H, W = heatmap.shape
        
        # Create coordinate grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=heatmap.device),
            torch.arange(W, dtype=torch.float32, device=heatmap.device),
            indexing='ij'
        )
        
        # Reshape grid to match heatmap dimensions
        x_grid = x_grid.view(1, 1, H, W).expand(B, C, H, W)
        y_grid = y_grid.view(1, 1, H, W).expand(B, C, H, W)
        
        # Normalize heatmap to sum to 1 (like a probability distribution)
        # Add small epsilon to avoid division by zero
        norm_heatmap = heatmap / (heatmap.sum(dim=(2, 3), keepdim=True) + 1e-10)
        
        # Calculate expected coordinates (center of mass)
        expected_x = (norm_heatmap * x_grid).sum(dim=(2, 3))
        expected_y = (norm_heatmap * y_grid).sum(dim=(2, 3))
        
        # Stack coordinates
        coords = torch.stack([expected_x, expected_y], dim=2)  # (B, C, 2)
        
        return coords.squeeze(1)  # (B, 2)

def create_model(task='pixel', in_shape=(16, 1, 32, 32), out_shape=None, 
                 hid_S=64, hid_T=512, N_S=4, N_T=8, model_type='gSTA'):
    """
    Create a SimVP model with the specified task head.
    
    Args:
        task (str): Task type ('pixel', 'heat', or 'coord').
        in_shape (tuple): Input shape (T, C, H, W).
        out_shape (tuple, optional): Output shape. If None, uses in_shape.
        hid_S (int): Hidden dimension of spatial encoder.
        hid_T (int): Hidden dimension of temporal encoder.
        N_S (int): Number of spatial encoder blocks.
        N_T (int): Number of temporal encoder blocks.
        model_type (str): Type of SimVP model ('gSTA', 'IncepU', etc.).
        
    Returns:
        SimVPWithTaskHead: The created model.
    """
    return SimVPWithTaskHead(
        in_shape=in_shape,
        out_shape=out_shape,
        hid_S=hid_S,
        hid_T=hid_T,
        N_S=N_S,
        N_T=N_T,
        model_type=model_type,
        task=task
    ) 