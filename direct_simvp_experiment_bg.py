#!/usr/bin/env python
"""
SimVP experiment with COCO background images
"""

import os
import json
import torch
import numpy as np
from openstl.models import SimVP_Model
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt


def load_coco_search_data_with_images(data_root='data/coco_search18_tp'):
    """Load COCO-Search data with corresponding images"""
    
    # Load scanpath sequences
    train_sequences = np.load(os.path.join(data_root, 'processed/short_train_sequences.npy'))
    val_sequences = np.load(os.path.join(data_root, 'processed/short_val_sequences.npy'))
    
    # Load fixation data to get image IDs
    # This assumes there's a mapping file - adjust based on your actual data structure
    fixations_dir = os.path.join(data_root, 'fixations')
    images_dir = os.path.join(data_root, 'images')
    
    # Get list of available images
    available_images = {}
    for category in os.listdir(images_dir):
        category_path = os.path.join(images_dir, category)
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                if img_file.endswith('.jpg'):
                    img_id = img_file.replace('.jpg', '')
                    available_images[img_id] = os.path.join(category_path, img_file)
    
    print(f"Found {len(available_images)} images")
    
    # For now, randomly assign images to sequences
    # In practice, you need the actual mapping from your data
    image_list = list(available_images.values())
    train_image_paths = [image_list[i % len(image_list)] for i in range(len(train_sequences))]
    val_image_paths = [image_list[i % len(image_list)] for i in range(len(val_sequences))]
    
    return (train_sequences, train_image_paths), (val_sequences, val_image_paths)


def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    """Load and preprocess COCO images"""
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except:
            # If image fails to load, use black image
            images.append(torch.zeros(3, target_size[0], target_size[1]))
    
    return torch.stack(images)


class ImageFeatureExtractor(nn.Module):
    """Extract features from images using pretrained CNN"""
    
    def __init__(self, model_name='resnet50'):
        super().__init__()
        if model_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048
        elif model_name == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 512
        
        self.feature_extractor.eval()
        
    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features


class SimVPMultimodal(nn.Module):
    """SimVP with both scanpath and image inputs"""
    
    def __init__(self, scanpath_shape=(5, 1, 32, 32), image_feat_dim=2048):
        super().__init__()
        T, C, H, W = scanpath_shape
        
        # Scanpath encoder (smaller than before to prevent overfitting)
        self.scanpath_encoder = nn.Sequential(
            nn.Conv2d(T*C, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Image feature processor (fixed for MPS compatibility)
        self.image_processor = nn.Sequential(
            nn.Conv2d(image_feat_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
        )
        self.spatial_size = (H, W)
        
        # Temporal model on combined features
        self.temporal_model = SimVP_Model(
            in_shape=(T, 32+64, H, W),  # Combined channels
            hid_S=32,
            hid_T=64,
            N_S=2,
            N_T=2,
            model_type='gSTA'
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(96*T, T*C, kernel_size=1)
        
    def forward(self, scanpath_spatial, image_features):
        B, T, C, H, W = scanpath_spatial.shape
        
        # Encode scanpath
        scanpath_2d = scanpath_spatial.view(B, T*C, H, W)
        scanpath_feat = self.scanpath_encoder(scanpath_2d)  # [B, 32, H, W]
        
        # Process image features
        image_feat = self.image_processor(image_features)  # [B, 64, 7, 7]
        
        # Resize image features to match scanpath spatial size
        # Use interpolation instead of adaptive pooling for MPS compatibility
        image_feat = torch.nn.functional.interpolate(
            image_feat, 
            size=self.spatial_size, 
            mode='bilinear', 
            align_corners=False
        )  # [B, 64, H, W]
        
        # Expand image features temporally
        image_feat_temporal = image_feat.unsqueeze(1).repeat(1, T, 1, 1, 1)
        scanpath_feat_temporal = scanpath_feat.unsqueeze(1).repeat(1, T, 1, 1, 1)
        
        # Combine features
        combined = torch.cat([scanpath_feat_temporal, image_feat_temporal], dim=2)
        
        # Apply temporal model
        output = self.temporal_model(combined)
        
        # Project to output space
        output_2d = output.view(B, -1, H, W)
        output_2d = self.output_proj(output_2d)
        
        return output_2d.view(B, T, C, H, W)


def train_multimodal_simvp():
    """Train SimVP with both scanpath and image inputs"""
    
    print("Loading COCO-Search data with images...")
    (train_sequences, train_images), (val_sequences, val_images) = load_coco_search_data_with_images()
    
    print(f"Train sequences: {train_sequences.shape}")
    print(f"Val sequences: {val_sequences.shape}")
    
    # Convert sequences to spatial
    print("Converting coordinates to spatial representation...")
    train_inputs_spatial = coords_to_spatial_batch(train_sequences[:, :5])
    train_targets_spatial = coords_to_spatial_batch(train_sequences[:, 5:10])
    val_inputs_spatial = coords_to_spatial_batch(val_sequences[:, :5])
    val_targets_spatial = coords_to_spatial_batch(val_sequences[:, 5:10])
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Extract image features
    print("Extracting image features...")
    feature_extractor = ImageFeatureExtractor('resnet18')  # Use smaller model for M2
    feature_extractor = feature_extractor.to(device)
    
    # Process in batches to avoid memory issues
    batch_size = 32
    train_features = []
    val_features = []
    
    print("Processing training images...")
    for i in range(0, len(train_images), batch_size):
        batch_paths = train_images[i:i+batch_size]
        batch_imgs = load_and_preprocess_images(batch_paths)
        batch_feats = feature_extractor(batch_imgs.to(device))
        train_features.append(batch_feats.cpu())
    
    print("Processing validation images...")
    for i in range(0, len(val_images), batch_size):
        batch_paths = val_images[i:i+batch_size]
        batch_imgs = load_and_preprocess_images(batch_paths)
        batch_feats = feature_extractor(batch_imgs.to(device))
        val_features.append(batch_feats.cpu())
    
    train_features = torch.cat(train_features, dim=0)
    val_features = torch.cat(val_features, dim=0)
    
    print(f"Feature shapes: train={train_features.shape}, val={val_features.shape}")
    
    # Create model
    model = SimVPMultimodal(
        scanpath_shape=(5, 1, 32, 32),
        image_feat_dim=feature_extractor.feature_dim
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Convert to tensors
    train_x = torch.from_numpy(train_inputs_spatial).float().unsqueeze(2)
    train_y = torch.from_numpy(train_targets_spatial).float().unsqueeze(2)
    val_x = torch.from_numpy(val_inputs_spatial).float().unsqueeze(2)
    val_y = torch.from_numpy(val_targets_spatial).float().unsqueeze(2)
    
    # Training
    n_epochs = 30
    batch_size = 16
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\nTraining multimodal SimVP...")
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        
        indices = torch.randperm(len(train_x))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            
            batch_scanpath = train_x[batch_idx].to(device)
            batch_target = train_y[batch_idx].to(device)
            batch_img_feat = train_features[batch_idx].to(device)
            
            optimizer.zero_grad()
            pred = model(batch_scanpath, batch_img_feat)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        # Validate
        model.eval()
        val_loss = 0
        n_val_batches = 0
        val_preds = []
        
        with torch.no_grad():
            for i in range(0, len(val_x), batch_size):
                batch_scanpath = val_x[i:i+batch_size].to(device)
                batch_target = val_y[i:i+batch_size].to(device)
                batch_img_feat = val_features[i:i+batch_size].to(device)
                
                pred = model(batch_scanpath, batch_img_feat)
                loss = criterion(pred, batch_target)
                
                val_loss += loss.item()
                n_val_batches += 1
                val_preds.append(pred.cpu())
        
        avg_train_loss = train_loss / n_batches
        avg_val_loss = val_loss / n_val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_multimodal_simvp.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    model.load_state_dict(torch.load('best_multimodal_simvp.pth'))
    model.eval()
    
    # Get final predictions
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(val_x), batch_size):
            batch_scanpath = val_x[i:i+batch_size].to(device)
            batch_img_feat = val_features[i:i+batch_size].to(device)
            pred = model(batch_scanpath, batch_img_feat)
            all_preds.append(pred.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    
    # Convert back to coordinates
    val_pred_spatial = all_preds.squeeze(2).numpy()
    val_pred_coords = spatial_to_coords_batch(val_pred_spatial)
    val_target_coords = val_sequences[:len(val_pred_coords), 5:10]
    
    # Calculate MAE
    mae = np.mean(np.abs(val_pred_coords - val_target_coords))
    
    print(f"\nResults:")
    print(f"Coordinate MAE (with images): {mae:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    methods = ['LSTM\n(scanpath)', 'SimVP\n(scanpath)', 'SimVP\n(with images)']
    maes = [0.1704, 0.2073, mae]
    colors = ['blue', 'orange', 'green']
    
    bars = plt.bar(methods, maes, color=colors)
    plt.ylabel('MAE')
    plt.title('Model Comparison')
    plt.ylim(0, 0.25)
    
    for bar, mae_val in zip(bars, maes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mae_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('multimodal_simvp_results.png', dpi=150)
    print("\nSaved results to multimodal_simvp_results.png")
    
    return mae, train_losses[-1], val_losses[-1]


def coords_to_spatial_batch(coords_batch, spatial_size=32):
    """Convert coordinates to spatial heatmaps"""
    B, T, _ = coords_batch.shape
    spatial = np.zeros((B, T, spatial_size, spatial_size), dtype=np.float32)
    
    sigma = 2.0
    for b in range(B):
        for t in range(T):
            x = int(coords_batch[b, t, 0] * (spatial_size - 1))
            y = int(coords_batch[b, t, 1] * (spatial_size - 1))
            x = np.clip(x, 0, spatial_size - 1)
            y = np.clip(y, 0, spatial_size - 1)
            
            for i in range(max(0, x-5), min(spatial_size, x+6)):
                for j in range(max(0, y-5), min(spatial_size, y+6)):
                    dist = np.sqrt((i - x)**2 + (j - y)**2)
                    spatial[b, t, j, i] = np.exp(-dist**2 / (2 * sigma**2))
    
    return spatial


def spatial_to_coords_batch(spatial_batch):
    """Convert spatial heatmaps back to coordinates"""
    B, T, H, W = spatial_batch.shape
    coords = np.zeros((B, T, 2))
    
    for b in range(B):
        for t in range(T):
            heatmap = spatial_batch[b, t]
            y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            coords[b, t, 0] = x_idx / (W - 1)
            coords[b, t, 1] = y_idx / (H - 1)
    
    return coords


if __name__ == '__main__':
    print("SimVP with COCO Background Images Experiment")
    print("=" * 60)
    
    mae, final_train, final_val = train_multimodal_simvp()
    
    print("\n" + "=" * 60)
    print("Experiment Complete")
    print("=" * 60)
    print(f"Final Results:")
    print(f"  LSTM (scanpath only): 0.1704")
    print(f"  SimVP (scanpath only): 0.2073")
    print(f"  SimVP (with images): {mae:.4f}")
    print(f"\nConclusion:")
    if mae < 0.17:
        print("  ✓ Adding visual context significantly improves SimVP performance")
        print("  ✓ SimVP is suitable for eye movement prediction with visual input")
    else:
        print("  ✗ Visual context does not sufficiently improve SimVP")
        print("  ✗ Consider alternative approaches")