#!/usr/bin/env python3
"""
Configuration file for Gaussian Enhanced experiment.
Based on single_fixation_experiment with Gaussian blob rendering.
"""

# Random seed
random_seed = 42

# Data generation parameters
train_size = 1000
val_size = 100
test_size = 100

# Sequence parameters - CORRECTED for baseline compatibility
input_frames = 4         # Input sequence length
output_frames = 16       # Output sequence length  
total_frames = 20        # Total sequence length (input + output)
img_size = 32           # Spatial resolution

# Gaussian rendering parameters
gaussian_sigma = 2.0    # Gaussian blob sigma
gaussian_normalize = True  # Normalize Gaussian to [0,1]

# History trail parameters
enable_history_trails = True  # Enable history trails
history_length = 3      # Number of previous fixations to show
history_decay_gamma = 0.75  # Exponential decay factor for history
history_min_intensity = 0.1  # Minimum intensity threshold

# Step 1: Removed complex spatial sampling parameters - now use single_fixation coordinates

# Training parameters
batch_size = 8
max_epochs = 100
learning_rate = 1e-3
early_stopping_patience = 15
early_stopping_min_delta = 1e-4
lr_scheduler_patience = 8
lr_scheduler_factor = 0.5

# Step 2: Simplified loss function weights (removed complex 5-part loss)
simple_mse_weight = 1.0
simple_coord_weight = 2.0

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