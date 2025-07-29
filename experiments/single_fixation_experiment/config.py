#!/usr/bin/env python3
"""
Configuration file for single fixation experiment.
Each frame represents a single independent fixation.
"""

# Random seed
random_seed = 42

# Data generation parameters
train_size = 1000
val_size = 100
test_size = 100

# Sequence parameters - CORRECTED for baseline compatibility
input_frames = 16        # Input sequence length
output_frames = 16       # Output sequence length  
total_frames = 32        # Total sequence length (input + output)
img_size = 32           # Spatial resolution

# Spatial sampling parameters for single fixation generation
sigma = 4.5             # Initial fixation spread
grid_size = 32          # Position-dependent parameter grid
displacement_scale = 9.0  # Scale for center-biased displacement
center_bias_strength = 0.55  # Center bias strength
random_exploration_scale = 0.65  # Random exploration component
cov_min_diag = 0.1      # Min covariance diagonal
cov_max_diag = 9.0      # Max covariance diagonal
short_step_ratio = 0.4  # Ratio of short vs long steps
short_step_cov_range = (0.01, 0.15)  # Short step covariance range
long_step_cov_range = (10.0, 20.0)   # Long step covariance range

# Training parameters
batch_size = 8
max_epochs = 100
learning_rate = 1e-3
early_stopping_patience = 15
early_stopping_min_delta = 1e-4
lr_scheduler_patience = 8
lr_scheduler_factor = 0.5

# Loss function weights (optimal values from experiments)
sparsity_weight = 0.8
concentration_weight = 1.5
focal_alpha = 1.0
coordinate_weight = 1.0
background_weight = 0.1

# Model architecture
model_hid_S = 64
model_hid_T = 512
model_N_S = 4
model_N_T = 8
model_type = 'gSTA'
model_mlp_ratio = 8.0
model_drop = 0.0
model_drop_path = 0.0
model_spatio_kernel_enc = 3
model_spatio_kernel_dec = 3

# Directories and file paths
data_dir = "data/"
models_dir = "models/"
results_dir = "results/"

# Data files
train_data_file = f"{data_dir}train_data.pt"
val_data_file = f"{data_dir}val_data.pt"
test_data_file = f"{data_dir}test_data.pt"

# Model save paths
final_model_path = f"{models_dir}best_final_model.pth"
checkpoint_path = f"{models_dir}checkpoint.pth"
training_log_path = "training_log.txt"

# Resume training
resume_training = False
resume_checkpoint = None  # Set to checkpoint path to resume

# Evaluation parameters
eval_thresholds = [2.0, 3.0, 5.0]  # Pixel accuracy thresholds 