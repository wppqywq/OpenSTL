# configs/coco_search/simvp/SimVP_gSTA_GPU.py
method = 'SimVP'

# dataset parameters
dataname = 'coco_search'
data_root = './data/coco_search18_tp/processed'
dataset_type = 'coordinate'  # 'coordinate' or 'heatmap'

# sequence parameters - matching your PDF
in_shape = (10, 1, 32, 32)  # T, C, H, W
pre_seq_length = 5
aft_seq_length = 5
total_length = 10

# model parameters - from your PDF
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 128
N_T = 4
N_S = 2

# training parameters for GPU
lr = 1e-3
batch_size = 64  # Larger batch for GPU
val_batch_size = 64
test_batch_size = 64
epoch = 200
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 10

# GPU specific settings
device = 'cuda'
num_workers = 4
find_unused_parameters = False
use_gpu = True

# optimizer
opt = 'adam'
weight_decay = 0.0

# evaluation
metrics = ['mse', 'mae', 'coordinate_error']
metric_for_bestckpt = 'mae'

# distributed training
dist = True
world_size = 1