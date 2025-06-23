import numpy as np
import torch


def euclidean_distance(pred, true):
    """Calculate euclidean distance between predicted and true coordinates"""
    return np.sqrt(np.sum((pred - true)**2, axis=-1))


def scanpath_similarity(pred, true, threshold=0.1):
    """Calculate scanpath similarity based on coordinate proximity"""
    distances = euclidean_distance(pred, true)
    accuracy = np.mean(distances < threshold)
    return accuracy


def DTW_distance(pred, true):
    """Dynamic Time Warping distance for sequence comparison"""
    def dtw_core(x, y):
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(x[i-1] - y[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        return dtw_matrix[n, m]
    
    # Calculate DTW for each sequence in batch
    batch_size = pred.shape[0]
    dtw_scores = []
    
    for b in range(batch_size):
        for t in range(pred.shape[1]):  # For each time step
            pred_coords = pred[b, t].reshape(-1, 2)  # Extract coordinates
            true_coords = true[b, t].reshape(-1, 2)
            dtw_score = dtw_core(pred_coords, true_coords)
            dtw_scores.append(dtw_score)
    
    return np.mean(dtw_scores)


def fixation_accuracy(pred, true, threshold=0.05):
    """Calculate fixation accuracy within threshold"""
    # Extract coordinates from spatial representation
    pred_coords = spatial_to_coords(pred)
    true_coords = spatial_to_coords(true)
    
    distances = euclidean_distance(pred_coords, true_coords)
    accuracy = np.mean(distances < threshold)
    return accuracy


def spatial_to_coords(spatial_data):
    """Convert spatial representation back to coordinates"""
    # spatial_data: [batch, time, channels, height, width]
    B, T, C, H, W = spatial_data.shape
    coords = np.zeros((B, T, 2))
    
    for b in range(B):
        for t in range(T):
            # Find maximum activation for each channel
            x_channel = spatial_data[b, t, 0]
            y_channel = spatial_data[b, t, 1]
            
            # Get coordinates of maximum activation
            h_idx, w_idx = np.unravel_index(np.argmax(x_channel), (H, W))
            coords[b, t, 0] = w_idx / (W - 1)  # Normalize to [0, 1]
            coords[b, t, 1] = h_idx / (H - 1)
            
    return coords


def sequence_length_error(pred, true):
    """Calculate error in predicted sequence length"""
    pred_lengths = np.sum(np.any(pred.reshape(pred.shape[0], pred.shape[1], -1) != 0, axis=-1), axis=1)
    true_lengths = np.sum(np.any(true.reshape(true.shape[0], true.shape[1], -1) != 0, axis=-1), axis=1)
    return np.mean(np.abs(pred_lengths - true_lengths))


def evaluate_eyetrack_metrics(pred, true, metrics=['scanpath_sim', 'fixation_acc', 'dtw']):
    """Evaluate eye tracking specific metrics"""
    eval_res = {}
    
    if 'scanpath_sim' in metrics:
        eval_res['scanpath_sim'] = scanpath_similarity(pred, true)
    
    if 'fixation_acc' in metrics:
        eval_res['fixation_acc'] = fixation_accuracy(pred, true)
        
    if 'dtw' in metrics:
        eval_res['dtw'] = DTW_distance(pred, true)
        
    if 'seq_len_err' in metrics:
        eval_res['seq_len_err'] = sequence_length_error(pred, true)
    
    return eval_res