"""
SimVP method wrapper that uses GroupNorm instead of BatchNorm
Place this file in: openstl/methods/simvp_groupnorm_method.py

This integrates the GroupNorm-based SimVP with OpenSTL framework.
"""

import torch
import torch.nn as nn
import numpy as np

# Fix import - use absolute path
import sys
import os
sys.path.append(os.path.dirname(__file__))
from simvp_groupnorm import SimVP_Model_GroupNorm


class SimVP_GroupNorm:
    """SimVP method with GroupNorm for COCO-Search18 eye tracking"""
    
    def __init__(self, steps_per_epoch, test_mean=None, test_std=None, save_dir=None, **kwargs):
        """Initialize method following OpenSTL pattern"""
        self.steps_per_epoch = steps_per_epoch
        self.test_mean = test_mean
        self.test_std = test_std
        self.save_dir = save_dir
        self.args = kwargs
        
        # Extract model parameters
        self.in_shape = kwargs.get('in_shape', (10, 1, 32, 32))
        self.hid_S = kwargs.get('hid_S', 32)
        self.hid_T = kwargs.get('hid_T', 128)
        self.N_S = kwargs.get('N_S', 2)
        self.N_T = kwargs.get('N_T', 4)
        self.device = kwargs.get('device', 'cpu')
        
        # Build model
        self.model = self._build_model()
        
        # Move to device
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            self.model = self.model.cuda()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and 'mps' in str(self.device):
            self.model = self.model.to('mps')
        
        # Loss function for eye tracking
        self.criterion = self._get_loss_function()
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

    def _build_model(self):
        """Build SimVP model with GroupNorm"""
        return SimVP_Model_GroupNorm(
            in_shape=self.in_shape,
            hid_S=self.hid_S,
            hid_T=self.hid_T,
            N_S=self.N_S,
            N_T=self.N_T
        )

    def _get_loss_function(self):
        """Eye tracking specific loss function"""
        class EyeTrackLoss(nn.Module):
            def __init__(self, coord_weight=1.0, smooth_weight=0.1):
                super().__init__()
                self.coord_weight = coord_weight
                self.smooth_weight = smooth_weight
                self.mse_loss = nn.MSELoss()
                
            def forward(self, pred, target):
                # Main coordinate prediction loss
                coord_loss = self.mse_loss(pred, target)
                
                # Smoothness regularization
                if pred.size(1) > 1:  # If sequence length > 1
                    diff_pred = pred[:, 1:] - pred[:, :-1]
                    diff_target = target[:, 1:] - target[:, :-1]
                    smooth_loss = self.mse_loss(diff_pred, diff_target)
                    total_loss = self.coord_weight * coord_loss + self.smooth_weight * smooth_loss
                else:
                    total_loss = coord_loss
                
                return total_loss
        
        return EyeTrackLoss()

    def _setup_optimizer(self):
        """Setup optimizer"""
        lr = self.args.get('lr', 1e-3)
        weight_decay = self.args.get('weight_decay', 0.0)
        opt_type = self.args.get('opt', 'adam').lower()
        
        if opt_type == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        sched_type = self.args.get('sched', None)
        if not sched_type:
            return None
            
        epochs = self.args.get('epoch', 100)
        
        if sched_type == 'cosine':
            min_lr = self.args.get('min_lr', 1e-6)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=min_lr
            )
        elif sched_type == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.args.get('lr', 1e-3),
                total_steps=epochs * self.steps_per_epoch
            )
        else:
            return None

    def train_one_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 2:
                batch_x, batch_y = batch
            else:
                raise ValueError("Expected batch to contain (input, target)")
            
            # Move to device
            if torch.cuda.is_available() and 'cuda' in str(self.device):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and 'mps' in str(self.device):
                batch_x = batch_x.to('mps')
                batch_y = batch_y.to('mps')
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_y = self.model(batch_x)
            loss = self.criterion(pred_y, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update scheduler if step-wise
            if self.scheduler and self.args.get('sched') == 'onecycle':
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        # Update scheduler if epoch-wise
        if self.scheduler and self.args.get('sched') != 'onecycle':
            self.scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}

    def vali_one_epoch(self, vali_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in vali_loader:
                if len(batch) == 2:
                    batch_x, batch_y = batch
                else:
                    continue
                
                # Move to device
                if torch.cuda.is_available() and 'cuda' in str(self.device):
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and 'mps' in str(self.device):
                    batch_x = batch_x.to('mps')
                    batch_y = batch_y.to('mps')
                
                pred_y = self.model(batch_x)
                loss = self.criterion(pred_y, batch_y)
                
                # Compute additional metrics
                mse = nn.functional.mse_loss(pred_y, batch_y)
                mae = nn.functional.l1_loss(pred_y, batch_y)
                
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_mae = total_mae / max(num_batches, 1)
        
        print(f'Validation - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}')
        
        return {
            'val_loss': avg_loss,
            'val_mse': avg_mse,
            'val_mae': avg_mae
        }

    def test_one_epoch(self, test_loader):
        """Test for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    batch_x, batch_y = batch
                else:
                    continue
                
                # Move to device
                if torch.cuda.is_available() and 'cuda' in str(self.device):
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and 'mps' in str(self.device):
                    batch_x = batch_x.to('mps')
                    batch_y = batch_y.to('mps')
                
                pred_y = self.model(batch_x)
                loss = self.criterion(pred_y, batch_y)
                
                # Compute metrics
                mse = nn.functional.mse_loss(pred_y, batch_y)
                mae = nn.functional.l1_loss(pred_y, batch_y)
                
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1
                
                # Store predictions for analysis
                all_predictions.append(pred_y.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_mae = total_mae / max(num_batches, 1)
        
        # Compute coordinate error if applicable
        coord_error = 0.0
        if all_predictions:
            pred_array = np.concatenate(all_predictions, axis=0)
            target_array = np.concatenate(all_targets, axis=0)
            
            if pred_array.shape[-1] == 2:  # Coordinate representation
                coord_diffs = pred_array - target_array
                coord_distances = np.sqrt(np.sum(coord_diffs**2, axis=-1))
                coord_error = np.mean(coord_distances)
        
        print(f'Test Results - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, Coord Error: {coord_error:.6f}')
        
        return {
            'test_loss': avg_loss,
            'test_mse': avg_mse,
            'test_mae': avg_mae,
            'test_coord_error': coord_error,
            'predictions': all_predictions,
            'targets': all_targets
        }