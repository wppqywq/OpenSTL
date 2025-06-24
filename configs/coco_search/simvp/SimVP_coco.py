method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 256
N_T = 4
N_S = 2
# training
lr = 1e-3
batch_size = 16
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 5
# reverse scheduled sampling
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# model-specific
# dataset
in_shape = (10, 1, 32, 32)  # Input: 10 frames, 1 channel, 32x32 spatial
# training
epoch = 100
log_step = 1
# data
train_batch_size = 16
val_batch_size = 8
test_batch_size = 8
# optimizer
opt = 'adam'
opt_eps = None
opt_betas = None
momentum = 0.9
weight_decay = 0.0
# scheduler
min_lr = 1e-6
warmup_lr = 1e-5
decay_rate = 0.1
decay_epoch = 100
filter_bias_and_bn = False
# evaluation
eval_metrics = ['mse', 'mae']
test_mean = 0.0
test_std = 1.0
metric_for_bestckpt = 'mse'
# device
device = 'mps' if __import__('torch').backends.mps.is_available() else 'cpu'
# dataset
dataname = 'coco_search'
pre_seq_length = 10
aft_seq_length = 10
total_length = 20
use_augment = False
# training specifics
num_workers = 0  # Important for M2 Mac
find_unused_parameters = True