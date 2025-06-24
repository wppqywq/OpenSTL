# SimVP configuration for COCO-Search18

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
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 2
# dataset
in_shape = (10, 1, 32, 32)  # T, C, H, W
pre_seq_length = 5
aft_seq_length = 5
total_length = 10
# optimizer
opt = 'adam'
weight_decay = 0.0
# evaluation
metric_for_bestckpt = 'mse'
