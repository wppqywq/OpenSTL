method = 'ConvLSTM'

# dataset parameters
dataname = 'coco_search'
data_root = './data/coco_search'
in_shape = (20, 1, 64, 64)  # ConvLSTM can handle this better
pre_seq_length = 10
aft_seq_length = 10
total_length = 20

# model parameters
patch_size = 4
rnn_hid_dim = 128
rnn_num_layers = 2
rnn_cell_type = 'LSTM'

# training parameters
lr = 1e-3
batch_size = 4
val_batch_size = 4
epoch = 5
drop_path = 0.0
sched = 'onecycle'

# evaluation metrics
metrics = ['mse', 'mae']
