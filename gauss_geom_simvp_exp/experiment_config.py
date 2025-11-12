#!/usr/bin/env python3
"""
Centralized configuration for all experiments.
"""

# Base configuration
BASE_CONFIG = {
    # Data parameters
    'img_size': 32,
    'total_seq_length': 20,
    'split_ratio': 0.5,  # 0.5 means half input, half output
    
    # Dataset sizes (defaults preferred by user: 2000/200/200)
    'train_size': 2000,
    'val_size': 200,
    'test_size': 200,
    
    # Model parameters
    'model': 'simvp',
    'hid_S': 64,
    'hid_T': 512,
    'N_S': 4,
    'N_T': 8,
    'model_type': 'gSTA',
    'use_gru': False,
    
    # Training parameters
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'patience': 5,
    'num_workers': 0,
    
    # Representation parameters
    'sigma': 2.0,
    'coord_mode': 'absolute',
    'coord_sigmoid': False,
    'apply_mirror_loss': False,
    
    # Loss parameters
    'alpha': 0.25,
    'gamma': 2.0,
    'delta': 1.0,
    'w_dir': 1.0,
    'w_mag': 1.0,
    'pos_weight': 1.0,
    'cosine_weight': 1.0,
    
    # Output parameters
    'output_dir': 'results',
    'save_freq': 10,
    
    # Data generation parameters
    'generate_data': False,
    'use_mu_field': False,
    'gauss_seed': None,
    'start_x': None,
    'start_y': None,
}

# Experiment configurations
EXPERIMENTS = {
    # Gauss dataset experiments
    'gauss_pixel_focal': {
        'data': 'gauss',
        'repr': 'pixel',
        'loss': 'focal_bce',
        'lr': 0.001,
        'alpha': 1.0,
        'gamma': 4.0,
    },
    
    'gauss_pixel_weighted': {
        'data': 'gauss',
        'repr': 'pixel',
        'loss': 'weighted_bce',
        'lr': 0.001,
        'pos_weight': 2000.0,
    },
    
    'gauss_pixel_dice': {
        'data': 'gauss',
        'repr': 'pixel',
        'loss': 'dice_bce',
        'lr': 0.001,
    },
    
    'gauss_heat_kl': {
        'data': 'gauss',
        'repr': 'heat',
        'loss': 'kl',
        'lr': 0.001,
        'sigma': 2.0,
    },
    
    # Gaussian dataset with heat representation - 5->5 prediction
    'gauss_heat_weighted_mse': {
        'data': 'gauss',
        'repr': 'heat',
        'loss': 'weighted_mse',
        'lr': 0.001,
        'pos_weight': 100.0,
        'sigma': 2.0,
        'total_seq_length': 10,    # Use first 10 frames from 20-frame sequences
        'split_ratio': 0.5,        # 5 input frames, 5 output frames
        'epochs': 100,
    },
    
    'gauss_heat_weighted_bce': {
        'data': 'gauss',
        'repr': 'heat',
        'loss': 'weighted_bce',
        'lr': 0.001,
        'pos_weight': 100.0,
        'sigma': 2.0,
        'total_seq_length': 10,    # Use first 10 frames from 20-frame sequences
        'split_ratio': 0.5,        # 5 input frames, 5 output frames
        'epochs': 100,
    },
    
    # Corresponding pixel versions with higher weights
    'gauss_pixel_weighted_mse': {
        'data': 'gauss',
        'repr': 'pixel',
        'loss': 'weighted_mse',
        'lr': 0.001,
        'pos_weight': 1000.0,      # Higher weight for sparse pixel data
        'total_seq_length': 10,    # Use first 10 frames from 20-frame sequences
        'split_ratio': 0.5,        # 5 input frames, 5 output frames
        'epochs': 100,
    },
    
    'gauss_pixel_weighted_bce': {
        'data': 'gauss',
        'repr': 'pixel',
        'loss': 'weighted_bce',
        'lr': 0.001,
        'pos_weight': 2000.0,      # Even higher weight for BCE + sparse data
        'total_seq_length': 10,    # Use first 10 frames from 20-frame sequences
        'split_ratio': 0.5,        # 5 input frames, 5 output frames
        'epochs': 100,
    },
    
    'geom_heat_weighted_mse': {
        'data': 'geom_simple',
        'repr': 'heat',
        'loss': 'weighted_mse',
        'lr': 0.001,
        'sigma': 2.0,
        'pos_weight': 1000.0,
    },
    
    # Experimental: Heatmap with BCE losses
    'gauss_heat_focal_bce': {
        'data': 'gauss',
        'repr': 'heat', 
        'loss': 'focal_bce',
        'lr': 0.001,
        'sigma': 2.0,
        'alpha': 0.25,
        'gamma': 2.0,
    },
    
    'geom_heat_weighted_bce': {
        'data': 'geom_simple',
        'repr': 'heat',
        'loss': 'weighted_bce', 
        'lr': 0.001,
        'sigma': 2.0,
        'pos_weight': 1000.0, 
    },
    'geom_heat_mse': {
        'data': 'geom_simple',
        'repr': 'heat',
        'loss': 'mse',
        'lr': 0.001,
        'sigma': 2.0,
    },
    'geom_heat_focal_bce': {
        'data': 'geom_simple',
        'repr': 'heat',
        'loss': 'focal_bce',
        'lr': 0.001,
        'sigma': 2.0,
        'alpha': 0.75,
        'gamma': 4.0,
    },
    'geom_pixel_weighted_mse': {
        'data': 'geom_simple',
        'repr': 'pixel',
        'loss': 'weighted_mse', 
        'lr': 0.001,
        'sigma': 2.0,
        'pos_weight': 1000.0, 
    },
    
    'gauss_coord_huber': {
        'data': 'gauss',
        'repr': 'coord',
        'loss': 'huber',
        'lr': 0.001,
        'delta': 3.0,
    },
    
    'gauss_coord_polar': {
        'data': 'gauss',
        'repr': 'coord',
        'loss': 'polar_decoupled',
        'lr': 0.001,
        'w_dir': 1.0,
        'w_mag': 1.0,
    },
    
    # Geom dataset experiments
    'geom_pixel_focal': {
        'data': 'geom_simple',
        'repr': 'pixel',
        'loss': 'focal_bce',
        'lr': 0.0005,
        'alpha': 1.0,
        'gamma': 4.0,
    },
    
    'geom_pixel_weighted': {
        'data': 'geom_simple',
        'repr': 'pixel',
        'loss': 'weighted_bce',
        'lr': 0.0005,
        'pos_weight': 1000.0,
    },
    
    'geom_heat_kl': {
        'data': 'geom_simple',
        'repr': 'heat',
        'loss': 'kl',
        'lr': 0.0005,
        'sigma': 2.0,
    },
    
    'geom_coord_huber': {
        'data': 'geom_simple',
        'repr': 'coord',
        'loss': 'huber',
        'lr': 0.0005,
        'delta': 3.0,
    },
    
    'geom_coord_polar': {
        'data': 'geom_simple',
        'repr': 'coord',
        'loss': 'polar_decoupled',
        'lr': 0.0005,
        'w_dir': 1.0,
        'w_mag': 1.0,
    },
}

def get_experiment_config(exp_name):
    """Get configuration for a specific experiment."""
    if exp_name not in EXPERIMENTS:
        available = ', '.join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {exp_name}. Available: {available}")
    
    # Start with base config
    config = BASE_CONFIG.copy()
    # Override with experiment-specific settings
    config.update(EXPERIMENTS[exp_name])
    # Set experiment name
    config['exp_name'] = exp_name
    
    return config

def list_experiments():
    """List all available experiments."""
    return list(EXPERIMENTS.keys())
