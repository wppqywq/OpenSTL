import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the root directory to the Python path to import from openstl
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from openstl.models import SimVP_Model

class CoordinateHead(nn.Module):
    """MLP head for coordinate prediction from SimVP features."""
    def __init__(self, feature_dim, output_seq_length, hidden_dim=512):
        super().__init__()
        self.output_seq_length = output_seq_length
        self.hidden_dim = hidden_dim
        
        if feature_dim is not None:
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, output_seq_length * 2)  # 2 for (x, y) coordinates
            )
        else:
            self.mlp = None
        
    def _init_mlp(self, feature_dim, device):
        """Initialize MLP with correct feature dimension."""
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_seq_length * 2)  # 2 for (x, y) coordinates
        ).to(device)
        
    def forward(self, features):
        # features shape: (B, feature_dim)
        if self.mlp is None:
            feature_dim = features.shape[1]
            self._init_mlp(feature_dim, features.device)
        
        assert self.mlp is not None, "MLP should be initialized"
        output = self.mlp(features)  # (B, output_seq_length * 2)
        return output.view(-1, self.output_seq_length, 2)  # (B, T, 2)

class GRUCoordinateHead(nn.Module):
    """GRU-based head for coordinate prediction from SimVP features."""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        # GRU that takes features + last_coord at each timestep
        self.gru = nn.GRU(feature_dim + 2, hidden_dim, batch_first=True)
        # Output projection to predict displacements
        self.output_proj = nn.Linear(hidden_dim, 2)
        
    def forward(self, features, last_coord):
        """
        Args:
            features: (B, T, C_feat) - temporal features from SimVP
            last_coord: (B, 2) - last input coordinate
            
        Returns:
            displacements: (B, T, 2) - predicted displacements
        """
        B, T, C_feat = features.shape
        device = features.device
        
        # Initialize hidden state
        h = torch.zeros(1, B, self.hidden_dim).to(device)
        
        # Initialize outputs
        displacements = []
        current_coord = last_coord  # (B, 2)
        
        # Process each timestep recursively
        for t in range(T):
            # Concatenate current features with current coordinate
            feat_t = features[:, t:t+1, :]  # (B, 1, C_feat)
            coord_t = current_coord.unsqueeze(1)  # (B, 1, 2)
            input_t = torch.cat([feat_t, coord_t], dim=2)  # (B, 1, C_feat+2)
            
            # GRU step
            out_t, h = self.gru(input_t, h)  # out_t: (B, 1, hidden_dim)
            
            # Predict displacement
            disp_t = self.output_proj(out_t.squeeze(1))  # (B, 2)
            displacements.append(disp_t)
            
            # Update current coordinate for next step
            current_coord = current_coord + disp_t  # Accumulate displacement
        
        # Stack all displacements
        displacements = torch.stack(displacements, dim=1)  # (B, T, 2)
        
        return displacements

class SimVPWithTaskHead(nn.Module):
    """
    SimVP model with different task heads.
    For coord task: Uses SimVP encoder + MLP decoder
    For pixel/heat tasks: Uses standard SimVP
    """
    def __init__(self, in_shape, hid_S=64, hid_T=512, N_S=4, N_T=8, 
                 model_type='gSTA', task='pixel', out_shape=None, use_gru=False,
                 coord_mode='absolute', coord_sigmoid=False, img_size=32):
        super().__init__()
        
        self.task = task
        self.in_shape = in_shape
        self.out_shape = out_shape if out_shape is not None else in_shape
        self.use_gru = use_gru
        self.coord_mode = coord_mode
        self.coord_sigmoid = coord_sigmoid
        self.img_size = img_size
        
        if task == 'coord':
            # Use SimVP as feature extractor for coordinate prediction
            # Create a dummy output shape for SimVP (we'll replace the decoder)
            dummy_out_shape = (self.out_shape[0], 64, 8, 8)  # Reduced spatial size for features
            self.simvp_encoder = SimVP_Model(
                in_shape=in_shape,
                hid_S=hid_S,
                hid_T=hid_T,
                N_S=N_S,
                N_T=N_T,
                model_type=model_type,
                out_shape=dummy_out_shape,
            )
            
            if use_gru:
                # GRU-based coordinate head
                # Feature dim will be set on first forward pass
                self.gru_head = None  # Will be initialized on first forward
            else:
                # Use a fixed, reasonable feature dimension for coordinate head
                # This will be corrected if needed on first forward pass
                self.coord_head = CoordinateHead(
                    feature_dim=None,  # Will be set dynamically
                    output_seq_length=self.out_shape[0],
                    hidden_dim=512
                )
                self._coord_head_initialized = False
            
        elif task in ['pixel', 'heat']:
            self.base_model = SimVP_Model(
                in_shape=in_shape,
                hid_S=hid_S,
                hid_T=hid_T,
                N_S=N_S,
                N_T=N_T,
                model_type=model_type,
                out_shape=self.out_shape,
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self, x, last_coord: torch.Tensor | None = None):
        if self.task in ['pixel', 'heat']:
            # Ensure input tensor is contiguous to avoid view() issues in SimVP
            x = x.contiguous()
            return self.base_model(x)
        
        elif self.task == 'coord':
            B, T_in, C, H, W = x.shape
            aft_seq_length = self.out_shape[0]

            # Ensure input tensor is contiguous for SimVP encoder
            x = x.contiguous()
            
            # Use SimVP to extract spatio-temporal features
            simvp_features = self.simvp_encoder(x)  # (B, T_out, C_feat, H_feat, W_feat)
            
            # Global average pool spatial dims, keep temporal: (B,T,C)
            B, T_out, C_feat, H_feat, W_feat = simvp_features.shape
            pooled_features = F.adaptive_avg_pool2d(
                simvp_features.view(B * T_out, C_feat, H_feat, W_feat),
                (1, 1)
            ).view(B, T_out, C_feat)  # (B,T,C_feat)

            # Obtain last coordinate (B,2)
            if last_coord is None:
                last_frame = x[:, T_in - 1, 0]
                flat_idx = last_frame.view(B, -1).argmax(dim=1)
                y_last = (flat_idx // W).float()
                x_last = (flat_idx % W).float()
                last_coord_feat = torch.stack([x_last, y_last], dim=1)
            else:
                last_coord_feat = last_coord  # (B,2)

            if self.use_gru:
                # Initialize GRU head on first forward pass
                if self.gru_head is None:
                    self.gru_head = GRUCoordinateHead(feature_dim=C_feat, hidden_dim=256).to(x.device)
                
                # Use GRU for displacement prediction
                displacements = self.gru_head(pooled_features, last_coord_feat)
            else:
                # Original per-frame MLP approach
                # Repeat last_coord across all future steps and concatenate
                last_coord_rep = last_coord_feat.unsqueeze(1).repeat(1, T_out, 1)  # (B,T,2)
                cat_features = torch.cat([pooled_features, last_coord_rep], dim=2)  # (B,T,C_feat+2)

                # Reshape to (B*T, F) for shared MLP
                BT, F_dim = B * T_out, C_feat + 2
                cat_flat = cat_features.view(BT, F_dim)

                # Build per-frame MLP lazily
                if not hasattr(self, 'frame_mlp'):
                    hidden = 512
                    self.frame_mlp = nn.Sequential(
                        nn.Linear(F_dim, hidden),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden, hidden // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden // 2, 2)
                    ).to(x.device)

                disp_flat = self.frame_mlp(cat_flat)  # (B*T,2)
                displacements = disp_flat.view(B, T_out, 2)

            if not hasattr(self, '_debug_printed'):
                print(f"  SimVP features shape: {simvp_features.shape}")
                print(f"  Pooled features shape: {pooled_features.shape}")
                if self.use_gru:
                    print(f"  Using GRU decoder with hidden_dim=256")
                else:
                    print(f"  Using per-frame MLP decoder")
                self._debug_printed = True

            return displacements
    
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
                 hid_S=64, hid_T=512, N_S=4, N_T=8, model_type='gSTA', use_gru=False):
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
        use_gru (bool): Whether to use GRU decoder for coord task.
        
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
        task=task,
        use_gru=use_gru
    ) 