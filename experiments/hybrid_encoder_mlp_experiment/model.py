#!/usr/bin/env python3
"""
Seq2Seq Eye Movement Prediction Model
Architecture: SimVP Encoder + GRU Decoder with Teacher Forcing
Predicts sequences of displacement vectors (Δx, Δy) for eye movement.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Compatibility patches for NumPy 2.0
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

# Add OpenSTL to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openstl.models import SimVP_Model


class SimVPEncoder(nn.Module):
    """
    Step 1: Standalone SimVP Encoder for spatiotemporal feature extraction.
    Returns a single context vector for sequence-to-sequence prediction.
    """
    
    def __init__(self, in_shape, hid_S=64, hid_T=512, N_S=4, N_T=8, model_type='gSTA'):
        """
        Initialize SimVP encoder.
        
        Args:
            in_shape: Input shape (T, C, H, W)
            hid_S: Hidden channels in spatial layers
            hid_T: Hidden channels in temporal layers
            N_S: Number of spatial layers
            N_T: Number of temporal layers
            model_type: SimVP model type ('gSTA', 'ConvNeXt', etc.)
        """
        super().__init__()
        
        # Store configuration
        self.in_shape = in_shape
        self.hid_S = hid_S
        self.N_S = N_S
        
        # Create full SimVP model (we'll only use encoder part)
        self.simvp_model = SimVP_Model(
            in_shape=in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            model_type=model_type,
            mlp_ratio=8.0,
            drop=0.0,
            drop_path=0.0,
            spatio_kernel_enc=3,
            spatio_kernel_dec=3
        )
        
        # Calculate latent dimensions
        T, C, H, W = in_shape
        downsample_factor = 2 ** (N_S // 2)
        self.latent_height = H // downsample_factor
        self.latent_width = W // downsample_factor
        self.latent_channels = hid_S
        
        # Hidden size for decoder (flattened encoder output)
        self.hidden_size = self.latent_channels * self.latent_height * self.latent_width
        
    def forward(self, input_frames):
        """
        Extract context vector from input video sequence.
        
        Args:
            input_frames: Input video (B, T, C, H, W)
            
        Returns:
            context_vector: Flattened encoder output (B, hidden_size)
        """
        batch_size = input_frames.shape[0]
        
        # Reshape for SimVP: (B, T, C, H, W) -> (B*T, C, H, W)
        B, T, C, H, W = input_frames.shape
        input_reshaped = input_frames.view(B * T, C, H, W)
        
        # Get encoder features from SimVP
        latent_features, _ = self.simvp_model.enc(input_reshaped)
        
        # Reshape back: (B*T, C_latent, H_latent, W_latent) -> (B, T, ...)
        C_latent, H_latent, W_latent = latent_features.shape[1:]
        latent_sequence = latent_features.view(B, T, C_latent, H_latent, W_latent)
        
        # Use final timestep as context vector
        final_latent_state = latent_sequence[:, -1, ...]  # (B, C_latent, H_latent, W_latent)
        
        # Flatten to create context vector
        context_vector = final_latent_state.flatten(start_dim=1)  # (B, hidden_size)
        
        return context_vector


class GRUDecoder(nn.Module):
    """
    Step 2: GRU-based decoder for generating displacement sequences.
    Uses Teacher Forcing during training for stable learning.
    """
    
    def __init__(self, hidden_size, input_size=2, output_size=2):
        """
        Initialize GRU decoder.
        
        Args:
            hidden_size: Size of hidden state (must match encoder output)
            input_size: Size of decoder input (2 for Δx, Δy)
            output_size: Size of decoder output (2 for Δx, Δy)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # GRU cell for autoregressive generation
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        
        # Linear layer to map hidden state to output coordinates
        self.out_linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, decoder_input, hidden_state):
        """
        Single forward step of decoder.
        
        Args:
            decoder_input: Current input (B, input_size)
            hidden_state: Current hidden state (B, hidden_size)
            
        Returns:
            output: Predicted displacement (B, output_size)
            next_hidden_state: Updated hidden state (B, hidden_size)
        """
        # Update hidden state with GRU
        next_hidden_state = self.gru_cell(decoder_input, hidden_state)
        
        # Generate output from hidden state
        output = self.out_linear(next_hidden_state)
        
        return output, next_hidden_state


class Seq2Seq_Displacement_Model(nn.Module):
    """
    Step 3: Complete Seq2Seq model with Teacher Forcing.
    Combines SimVP encoder with GRU decoder for multi-step displacement prediction.
    """
    
    def __init__(self, in_shape, num_future_frames=3, hid_S=64, hid_T=512, 
                 N_S=4, N_T=8, model_type='gSTA'):
        """
        Initialize Seq2Seq displacement prediction model.
        
        Args:
            in_shape: Input shape (T, C, H, W)
            num_future_frames: Number of future frames to predict
            hid_S, hid_T, N_S, N_T, model_type: SimVP parameters
        """
        super().__init__()
        
        self.num_future_frames = num_future_frames
        
        # Initialize encoder
        self.encoder = SimVPEncoder(in_shape, hid_S, hid_T, N_S, N_T, model_type)
        
        # Initialize decoder
        self.decoder = GRUDecoder(hidden_size=self.encoder.hidden_size)
        
    def forward(self, input_video, target_displacements=None, teacher_forcing_ratio=0.5):
        """
        Forward pass with Teacher Forcing.
        
        Args:
            input_video: Input video sequence (B, T, C, H, W)
            target_displacements: Target displacement sequence (B, num_preds, 2)
                                 Used for teacher forcing during training
            teacher_forcing_ratio: Probability of using teacher forcing [0, 1]
            
        Returns:
            predicted_sequence: Predicted displacement sequence (B, num_preds, 2)
        """
        batch_size = input_video.shape[0]
        num_preds = target_displacements.shape[1] if target_displacements is not None else self.num_future_frames
        
        # 1. Encode: Get context vector from input sequence
        context_vector = self.encoder(input_video)  # (B, hidden_size)
        
        # 2. Initialize decoder
        hidden_state = context_vector  # Use context as initial hidden state
        
        # Start with zero displacement as first input (like <GO> token)
        decoder_input = torch.zeros(batch_size, 2, device=input_video.device)
        
        outputs = []
        
        # 3. Autoregressive decoding with Teacher Forcing
        for t in range(num_preds):
            # Decoder forward step
            output, hidden_state = self.decoder(decoder_input, hidden_state)
            outputs.append(output)
            
            # Decide next input: Teacher Forcing or own prediction
            if target_displacements is not None and self.training:
                # During training, use Teacher Forcing with given probability
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    decoder_input = target_displacements[:, t]  # Use ground truth
                else:
                    decoder_input = output.detach()  # Use own prediction
            else:
                # During inference, always use own prediction
                decoder_input = output.detach()
        
        # Stack outputs into sequence
        predicted_sequence = torch.stack(outputs, dim=1)  # (B, num_preds, 2)
        
        return predicted_sequence


# Legacy: Keep the original single-step model for compatibility
class SimVP_RegressionHead(nn.Module):
    """
    Original single-step hybrid model (kept for compatibility).
    """
    
    def __init__(self, in_shape, hid_S=64, hid_T=512, N_S=4, N_T=8, model_type='gSTA'):
        super().__init__()
        
        # Create encoder
        self.encoder = SimVPEncoder(in_shape, hid_S, hid_T, N_S, N_T, model_type)
        
        # Simple MLP head for single-step prediction
        self.mlp_head = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Output (x, y) coordinates
        )
        
    def forward(self, input_frames):
        # Get encoder features
        context_vector = self.encoder(input_frames)
        
        # Single-step prediction
        predicted_coords = self.mlp_head(context_vector)
        
        return predicted_coords


def create_seq2seq_model(in_shape=(5, 1, 32, 32), num_future_frames=3, 
                        hid_S=64, hid_T=512, N_S=4, N_T=8, model_type='gSTA', device='cpu'):
    """
    Factory function to create Seq2Seq displacement prediction model.
    
    Args:
        in_shape: Input shape (T, C, H, W)
        num_future_frames: Number of future displacements to predict
        hid_S, hid_T, N_S, N_T: SimVP architecture parameters
        model_type: SimVP model type
        device: Device to place model on
        
    Returns:
        Seq2Seq_Displacement_Model ready for training
    """
    
    model = Seq2Seq_Displacement_Model(
        in_shape=in_shape,
        num_future_frames=num_future_frames,
        hid_S=hid_S,
        hid_T=hid_T,
        N_S=N_S,
        N_T=N_T,
        model_type=model_type
    ).to(device)
    
    print(f"Created Seq2Seq model:")
    print(f"  Input shape: {in_shape}")
    print(f"  Encoder hidden size: {model.encoder.hidden_size}")
    print(f"  Predicting {num_future_frames} future frames")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_hybrid_model(in_shape=(4, 1, 32, 32), hid_S=64, hid_T=512, N_S=4, N_T=8, 
                       model_type='gSTA', device='cpu'):
    """
    Factory function for legacy single-step model (kept for compatibility).
    """
    # For compatibility with existing experiments
    model = SimVP_RegressionHead(in_shape, hid_S, hid_T, N_S, N_T, model_type).to(device)
    
    print(f"Created legacy hybrid model:")
    print(f"  Input shape: {in_shape}")
    print(f"  Encoder hidden size: {model.encoder.hidden_size}")
    print(f"  Single-step displacement prediction")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    # Test both models
    device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_name)
    
    print("Testing Seq2Seq Model:")
    seq2seq_model = create_seq2seq_model(
        in_shape=(5, 1, 32, 32),
        num_future_frames=3,
        device=device_name
    )
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 5, 1, 32, 32).to(device)
    test_targets = torch.randn(batch_size, 3, 2).to(device)
    
    with torch.no_grad():
        # Test training mode (with teacher forcing)
        seq2seq_model.train()
        output_train = seq2seq_model(test_input, test_targets, teacher_forcing_ratio=0.8)
        
        # Test inference mode (no teacher forcing)
        seq2seq_model.eval()
        output_infer = seq2seq_model(test_input)
        
        print(f"Training output shape: {output_train.shape}")
        print(f"Inference output shape: {output_infer.shape}")
        print(f"Test successful!")
        
    print("\nTesting Legacy Single-Step Model:")
    legacy_model = create_hybrid_model(device=device_name)
    test_input_legacy = torch.randn(batch_size, 4, 1, 32, 32).to(device)
    
    with torch.no_grad():
        output_legacy = legacy_model(test_input_legacy)
        print(f"Legacy output shape: {output_legacy.shape}")
        print(f"Legacy test successful!") 