#!/usr/bin/env python3
"""
Centralized configuration for datasets and generators.
"""

# Image and sequence parameters
IMG_SIZE = 32
SEQUENCE_LENGTH = 20

# Dataset sizes (defaults preferred by user: 2000/200/200)
TRAIN_SIZE = 2000
VAL_SIZE = 200
TEST_SIZE = 200

# Representation parameters
HEATMAP_SIGMA = 2.0

# MVP Gaussian Field configuration (center-weak/edge-strong)
FIELD_CONFIG = {
    'step_scale': 5.0,
    'mode': 'center_inward',
    'alpha': 1.2,
    'lambda1_min': 0.2,
    'lambda1_max': 0.55,
    'lambda2_min': 0.1,
    'lambda_total': 0.65,
    'heatmap_sigma': 2.0,
}

# Geometric Pattern parameters
GEOM_CONFIG = {
    'line_speeds': [0.5, 1.0, 1.5],
    'line_directions': [0, 30, 60, 135, 150, 225, 300, 315],
    'line_origins': 8,
    'arc_radii': [5, 7, 10],
    'arc_directions': ['CW', 'CCW'],
    'arc_center_margin': 6,
    'line_ratio': 0.5,
    'bounce_reflection': True,
}

# Random seeds for reproducibility
RANDOM_SEED = 42


