#!/usr/bin/env python3
"""
Model Architectures for Coordinate Regression
Includes LSTM baseline and Transformer models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CoordinateLSTM(nn.Module):
    """LSTM for direct coordinate prediction"""
    
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_length=16, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_length * 2)  # output_length frames * 2 coords
        
    def forward(self, x):
        # x: [batch, seq_len=4, features=2]
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        coords = self.fc(last_output)     # [batch, output_length*2]
        coords = coords.view(-1, self.output_length, 2)   # [batch, output_length, 2]
        return coords


class CoordinateTransformer(nn.Module):
    """Transformer encoder-decoder for coordinate sequence prediction"""
    
    def __init__(self, d_model=64, nhead=8, num_layers=4, output_length=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.output_length = output_length
        
        # Input projection
        self.input_projection = nn.Linear(2, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_length * 2)
        
    def forward(self, x):
        # x: [batch, seq_len=4, features=2]
        batch_size = x.shape[0]
        
        # Project to d_model
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        encoded = self.encoder(x)  # [batch, seq_len, d_model]
        
        # Use last token for prediction
        last_token = encoded[:, -1, :]  # [batch, d_model]
        
        # Project to coordinates
        coords = self.output_projection(last_token)  # [batch, output_length*2]
        coords = coords.view(batch_size, self.output_length, 2)
        
        return coords


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return self.dropout(x)


class EnhancedCoordinateLSTM(nn.Module):
    """Enhanced LSTM with attention and residual connections"""
    
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, output_length=16, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Output layers with residual
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_length * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len=4, features=2]
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention
        combined = lstm_out + attn_out  # Residual connection
        
        # Use last output
        last_output = combined[:, -1, :]  # [batch, hidden_size]
        
        # Output projection with residual
        h1 = F.relu(self.fc1(last_output))
        h1 = self.dropout(h1)
        h1 = h1 + last_output  # Residual connection
        
        coords = self.fc2(h1)  # [batch, output_length*2]
        coords = coords.view(-1, self.output_length, 2)
        
        return coords


def create_model(model_type='lstm', **kwargs):
    """Factory function to create models"""
    
    if model_type == 'lstm':
        return CoordinateLSTM(**kwargs)
    elif model_type == 'transformer':
        return CoordinateTransformer(**kwargs)
    elif model_type == 'enhanced_lstm':
        return EnhancedCoordinateLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    batch_size = 8
    seq_len = 4
    input_coords = torch.randn(batch_size, seq_len, 2)
    
    print("Testing models...")
    
    # Test LSTM
    lstm_model = create_model('lstm', hidden_size=64, num_layers=2)
    lstm_out = lstm_model(input_coords)
    print(f"LSTM output shape: {lstm_out.shape}")
    
    # Test Transformer
    transformer_model = create_model('transformer', d_model=64, nhead=8, num_layers=4)
    transformer_out = transformer_model(input_coords)
    print(f"Transformer output shape: {transformer_out.shape}")
    
    # Test Enhanced LSTM
    enhanced_model = create_model('enhanced_lstm', hidden_size=128, num_layers=3)
    enhanced_out = enhanced_model(input_coords)
    print(f"Enhanced LSTM output shape: {enhanced_out.shape}")
    
    # Parameter count
    for name, model in [('LSTM', lstm_model), ('Transformer', transformer_model), ('Enhanced LSTM', enhanced_model)]:
        params = sum(p.numel() for p in model.parameters())
        print(f"{name} parameters: {params:,}") 