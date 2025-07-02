# configs/coco_search/simvp/SimVP_GroupNorm_M2.py
"""
Fixed configuration for COCO-Search18 using GroupNorm instead of BatchNorm
This should resolve the channel mismatch issues.
"""

method = 'SimVP_GroupNorm'  # Use our custom GroupNorm method
no_display_method_info = True
# Dataset parameters - using short sequences that actually exist
dataname = 'coco_search'
data_root = './data/coco_search18_tp/processed'  # Keep original path
dataset_config = 'short'
representation = 'heatmap'  # or 'coordinate'

# Sequence parameters - CRITICAL: match your actual data
in_shape = (5, 1, 32, 32)  # T=5 for input, matches short config
pre_seq_length = 5
aft_seq_length = 5
total_length = 10

# Model parameters - optimized for eye tracking
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32    # Will become T*hid_S=5*32=160 channels - divisible by 8
hid_T = 128   # Hidden temporal dimension
N_T = 4       # Number of temporal layers
N_S = 2       # Number of spatial layers

# Training parameters for M2 Mac
lr = 1e-3
batch_size = 16
val_batch_size = 16
test_batch_size = 16
epoch = 100
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 5

# Device settings for M2 Mac
device = 'mps'
num_workers = 0
dist = False
find_unused_parameters = False

# Optimizer
opt = 'adam'
weight_decay = 0.0
min_lr = 1e-6
warmup_lr = 1e-5

# Evaluation metrics - keep original metric for best checkpoint
metrics = ['mse', 'mae']
metric_for_bestckpt = 'mse'  # Don't let this be overwritten

# Eye tracking specific settings
use_augment = False
scheduled_sampling = 0
reverse_scheduled_sampling = 0

# Visualization settings for debugging
save_visualizations = True
vis_save_dir = './visualizations'
save_predictions = True

# Extra trainer arguments
trainer_kwargs = dict(
    num_sanity_val_steps=0,  # Disable Lightning sanity check
    log_every_n_steps=10,
    check_val_every_n_epoch=5
)

# GroupNorm specific settings
use_bn = False  # Disable BatchNorm completely
use_gn = True   # Enable GroupNorm
gn_groups = 8   # Base number of groups for GroupNorm

print(f"Config loaded: {method} with GroupNorm")
print(f"Data root: {data_root}")
print(f"Sequences: {pre_seq_length}→{aft_seq_length}, Input shape: {in_shape}")
print(f"Device: {device}, Batch size: {batch_size}")
print(f"Best checkpoint metric: {metric_for_bestckpt}")