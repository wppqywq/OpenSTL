# ConvLSTM configuration for COCO-Search18

method = 'ConvLSTM'
# model
kernel_size = 3
num_layers = 2
num_hidden = [64, 64]
# training
lr = 1e-3
batch_size = 16
sched = 'cosine'
# dataset
patch_size = 2
in_shape = (10, 1, 32, 32)
pre_seq_length = 5
aft_seq_length = 5
total_length = 10
# optimizer
opt = 'adam'
weight_decay = 0.0
