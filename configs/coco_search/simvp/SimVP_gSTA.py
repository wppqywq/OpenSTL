method = 'SimVP'

# dataset parameters
dataname = 'coco_search'
data_root = './data/coco_search'
in_shape = (20, 2, 64, 64)  # (total_length, channels, height, width)
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

# model parameters
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32  # Reduced for coordinate data
hid_T = 256
N_T = 6
N_S = 3

# training parameters
lr = 1e-3
batch_size = 16
val_batch_size = 16
epoch = 100
drop_path = 0.1
sched = 'onecycle'

# evaluation metrics
metrics = ['mse', 'mae']