# configs/coco_search/simvp/SimVP_fixed.py
"""
Fixed configuration for COCO-Search18
The key is to set in_shape[0] = pre_seq_length, not total_length
"""

method = 'SimVP'
model_type = 'gSTA'

# CRITICAL FIX: Set T=5 in in_shape
# Even though total_length=10, the model should be built for input length only
in_shape = (5, 1, 32, 32)  # Changed from (10, 1, 32, 32)

# Sequence lengths
pre_seq_length = 5
aft_seq_length = 5
total_length = 10
num_sanity_val_steps = 0

# Model parameters (keep original)
spatio_kernel_enc = 3
spatio_kernel_dec = 3
hid_S = 32
hid_T = 128
N_S = 2
N_T = 4

# Training parameters
lr = 1e-3
batch_size = 16
val_batch_size = 16
epoch = 100
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 5

# Device settings
device = 'mps'
num_workers = 0

# Optimizer
opt = 'adam'
weight_decay = 0.0

# Evaluation
metrics = ['mse', 'mae']
metric_for_bestckpt = 'mse'

# Data settings
data_root = './data/coco_search18_tp/processed'
dataset_config = 'short'

# Important: Disable any concatenation settings if they exist
scheduled_sampling = 0
reverse_scheduled_sampling = 0



###############################
# extra trainer arguments
###############################
trainer_kwargs = dict(
    num_sanity_val_steps = 0,   # 🔒 关闭 Lightning sanity-check
)

###############################
# SimVP-gSTA 特有开关
###############################
use_bn = False                  # 🔒 暂时关闭所有 BatchNorm


