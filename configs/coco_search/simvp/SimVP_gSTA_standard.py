method = 'SimVP'

# dataset parameters  
dataname = 'coco_search'
data_root = './data/coco_search'
in_shape = (20, 1, 64, 64)  # Standard size
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

# model parameters - use OpenSTL standard values
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64    # Standard value from OpenSTL
hid_T = 512   # Standard value from OpenSTL  
N_T = 8      # Standard value from OpenSTL
N_S = 4      # Standard value from OpenSTL

# training parameters - conservative for M2
lr = 1e-3
batch_size = 2   # Small for M2
val_batch_size = 2
epoch = 3        # Short test run
drop_path = 0.1  # Standard regularization
sched = 'onecycle'

# evaluation metrics
metrics = ['mse', 'mae']
