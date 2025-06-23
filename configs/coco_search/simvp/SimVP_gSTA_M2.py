method = 'SimVP'

# dataset parameters
dataname = 'coco_search'
data_root = './data/coco_search'
in_shape = (20, 1, 32, 32)  # Change to 1 channel to avoid conflicts
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

# model parameters - M2 compatible
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 16   # Smaller for channel compatibility
hid_T = 32   # Much smaller for M2
N_T = 2     # Minimal layers
N_S = 2     # Minimal layers

# training parameters - M2 optimized
lr = 1e-3
batch_size = 2   # Very small batch
val_batch_size = 2
epoch = 3        # Just for testing
drop_path = 0.0  # No regularization
sched = 'onecycle'

# evaluation metrics
metrics = ['mse', 'mae']