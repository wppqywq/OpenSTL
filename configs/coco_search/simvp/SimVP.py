method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 128
N_T = 4
N_S = 2
# training
lr = 1e-3
batch_size = 16
drop_path = 0
sched = 'onecycle'
# dataset
pre_seq_length = 5
aft_seq_length = 5
total_length = 10
# augmentation
use_augment = False