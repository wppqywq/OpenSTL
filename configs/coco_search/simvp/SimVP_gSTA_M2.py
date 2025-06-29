# configs/coco_search/simvp/SimVP_fixed.py
method = 'SimVP'

# Model parameters
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 128
N_T = 4
N_S = 2

# CRITICAL: Correct sequence lengths for your data
pre_seq_length = 5
aft_seq_length = 5
in_shape = (10, 1, 32, 32)  # This should match pre+aft length

# Training parameters
lr = 1e-3
batch_size = 16
val_batch_size = 16
test_batch_size = 16
epoch = 50
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 5

# Device settings for M2 Mac
#device = 'mps' if __import__('torch').backends.mps.is_available() else 'cpu'
device = 'mps'
num_workers = 0
dist = False

# Optimizer
opt = 'adam'
weight_decay = 0.0

# Scheduler
min_lr = 1e-6
warmup_lr = 1e-5
decay_rate = 0.1
decay_epoch = 100

# Evaluation
metrics = ['mse', 'mae']
metric_for_bestckpt = 'mse'

# Dataset specific - DO NOT CHANGE
dataname = 'coco_search'
dataset_config = 'short'  # Uses sequences of length 10
use_augment = False