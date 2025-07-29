#!/usr/bin/env python3
"""
Configuration file for hybrid SimVP encoder + MLP regression experiment.
This experiment uses SimVP encoder for feature extraction and MLP head for coordinate regression.
"""

# Random seed
random_seed = 42

# Data parameters (reuse from single_fixation_experiment)
img_size = 32
input_frames_regression = 4  # Number of input frames for regression model
batch_size = 16  # Larger batch size for regression task

# Model architecture parameters
model_hid_S = 64
model_hid_T = 512
model_N_S = 4
model_N_T = 8
model_type = 'gSTA'

# Training parameters for regression
max_epochs_regression = 100
learning_rate_regression = 1e-3
lr_scheduler_patience = 10

# Teacher Forcing Annealing parameters for Seq2Seq
initial_teacher_forcing_ratio = 1.0  # Start with full teacher forcing

# Directories
models_dir = "models/"
logs_dir = "logs/"
results_dir = "results/"

# Model save paths
hybrid_model_path = f"{models_dir}hybrid_regression_best.pth"
training_log_path = f"{logs_dir}training_log_hybrid_regression.txt" 