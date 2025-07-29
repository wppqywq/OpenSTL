#!/usr/bin/env python3
"""Configuration file for eye movement simulation training."""

# Random seed
RANDOM_SEED = 42

# Data generation parameters
TRAIN_SIZE = 1000
VAL_SIZE = 100
TEST_SIZE = 100

# Sequence parameters
SEQUENCE_LENGTH = 32
IMG_SIZE = 32

# Spatial sampling parameters
GRID_SIZE = 32
DISPLACEMENT_SCALE = 9.0
CENTER_BIAS_STRENGTH = 0.55
RANDOM_EXPLORATION_SCALE = 0.65
COV_MIN_DIAG = 0.1
COV_MAX_DIAG = 9.0

# Bimodal step length distribution
SHORT_STEP_RATIO = 0.4
SHORT_STEP_COV_RANGE = (0.01, 0.15)
LONG_STEP_COV_RANGE = (10.0, 20.0)

# Temporal sampling parameters
FIXATION_PROB = 0.4

# Ex-Gaussian distribution parameters for fixation duration
EX_GAUSSIAN_MU = 1.0
EX_GAUSSIAN_SIGMA = 0.75
EX_GAUSSIAN_TAU = 2.0

# Training parameters
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 10

# Model configuration
MODEL_CONFIG = {
    'in_shape': (16, 1, 32, 32),
    'hid_S': 64,
    'hid_T': 256,
    'N_S': 4,
    'N_T': 8,
    'model_type': 'gSTA'
}

# Paths
DATA_DIR = "data/"
MODEL_DIR = "models/"

# Loss function weights (best performing: robust_sharp)
LOSS_WEIGHTS = {
    'focal': 1.0,
    'sparsity': 0.3,
    'concentration': 1.5,
    'kl': 0.1
} 