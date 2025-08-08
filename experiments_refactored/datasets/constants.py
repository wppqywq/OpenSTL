#!/usr/bin/env python3
"""
Global constants for the experiments.
"""

# Image and sequence parameters
IMG_SIZE = 32
SEQUENCE_LENGTH = 20

# Dataset sizes
TRAIN_SIZE = 2500
VAL_SIZE = 100
TEST_SIZE = 100

# Structured Gaussian Field parameters
FIELD_CONFIG = {
    'grid_size': 32,
    'displacement_scale': 3.0,
    'sigma_start': 2.0,
    
    # Region thresholds (normalized distance from center)
    'center_threshold': 0.2,      # r < 0.2 = center region
    'corner_threshold': 0.85,     # r > 0.85 = corner region  
    'edge_threshold': 0.7,        # 0.7 < r <= 0.85 = edge region
    # Between center and edge = quadrant regions
    
    # Covariance parameters for each region
    'corner_lambda1': 4.0,        # Strong directional bias toward center
    'corner_lambda2': 0.3,        # Very elliptical
    
    'edge_lambda1': 3.0,          # Moderate directional bias along border
    'edge_lambda2': 0.5,          # Elliptical
    
    'center_lambda1': 1.5,        # Isotropic (random)
    'center_lambda2': 1.2,        # Nearly circular
    
    'quadrant_lambda1': 2.5,      # Rotating patterns
    'quadrant_lambda2': 0.4,      # Elliptical
    'quadrant_rotation_angle': 30, # degrees for CW/CCW rotation
}

# Geometric Pattern parameters  
GEOM_CONFIG = {
    # Line patterns: constant speed + bounce reflection
    'line_speeds': [0.5, 1.0, 1.5],        # 3 speeds (pixels/step)
    'line_directions': [0, 30, 60, 135, 150, 225, 300, 315],  # 8 directions (degrees)
    'line_origins': 8,                      # 8 starting positions (4x4 grid subset)
    
    # Arc patterns: circular motion
    'arc_radii': [5, 7, 10],               # 3 radii (pixels)
    'arc_directions': ['CW', 'CCW'],       # 2 rotation directions
    'arc_center_margin': 6,                # margin from edge for arc centers
    
    # General
    'line_ratio': 0.5,                     # Exactly 50% lines, 50% arcs
    'bounce_reflection': True,             # Use proper physics reflection
}

# Representation parameters
HEATMAP_SIGMA = 2.0

# Random seeds for reproducibility
RANDOM_SEED = 42