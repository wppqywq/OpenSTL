# Configuration Parameters
# ======================

# Grid and sequence parameters
IMG_SIZE = 32
SEQUENCE_LENGTH = 20
GRID_SIZE = 8  # Increased grid size for more starting points near center

# Shared parameters for both line and arc patterns
SPEED_MIN = 0.5  # Reduced to fit within 32x32 bounds
SPEED_MAX = 1.5
ANGLE_MIN = -180
ANGLE_MAX = 180

# Line pattern specific parameters
LINE_MIN_LENGTH = 8  # Minimum trajectory length
LINE_MAX_LENGTH = 50  # Maximum trajectory length before becoming bounce

# Arc pattern specific parameters
ARC_RADIUS_MIN = 3.0  # Increased minimum radius for better trajectories
ARC_RADIUS_MAX = 12.0
ARC_FACTORS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Fixed arc factors

# Bounce pattern parameters (for trajectories that exceed line max length)
BOUNCE_SPEED_MIN = 1.5
BOUNCE_SPEED_MAX = 4.0
BOUNCE_ANGLE_MIN = -180
BOUNCE_ANGLE_MAX = 180

# Dataset parameters
TRAIN_SAMPLES = 1000
VAL_SAMPLES = 100
TEST_SAMPLES = 100
BATCH_SIZE = 32

# Pattern distribution (50% line, 50% arc)
LINE_RATIO = 0.5
ARC_RATIO = 0.5

# Boundary parameters
BOUNDARY_MARGIN = 2
MIN_BOUNDARY_DISTANCE = 0
MAX_BOUNDARY_DISTANCE = IMG_SIZE

# Heatmap parameters
HEATMAP_SIGMA_1 = 1.0
HEATMAP_SIGMA_2 = 2.0

import torch
import numpy as np
import math
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def generate_line_pattern(start_point, theta, velocity, num_frames=20):
    """
    Generate a straight line trajectory with uniform velocity.
    
    Args:
        start_point (tuple): Starting (x, y) coordinates.
        theta (float): Direction angle in radians.
        velocity (float): Speed in pixels per frame.
        num_frames (int): Number of frames to generate.
        
    Returns:
        torch.Tensor: Coordinates of shape [num_frames, 2].
    """
    coords = torch.zeros(num_frames, 2)
    
    for t in range(num_frames):
        coords[t, 0] = start_point[0] + velocity * math.cos(theta) * t
        coords[t, 1] = start_point[1] + velocity * math.sin(theta) * t
    
    return coords

def generate_arc_pattern(center, radius, start_angle, velocity, direction, num_frames=20):
    """
    Generate a circular arc trajectory with uniform linear velocity.
    
    Args:
        center (tuple): Center (x, y) of the circle.
        radius (float): Radius of the circle.
        start_angle (float): Starting angle in radians.
        velocity (float): Linear speed in pixels per frame.
        direction (int): Direction of rotation (1 for CCW, -1 for CW).
        num_frames (int): Number of frames to generate.
        
    Returns:
        torch.Tensor: Coordinates of shape [num_frames, 2].
    """
    coords = torch.zeros(num_frames, 2)
    angular_velocity = velocity / radius  # Convert linear speed to angular velocity
    
    for t in range(num_frames):
        angle = start_angle + direction * angular_velocity * t
        coords[t, 0] = center[0] + radius * math.cos(angle)
        coords[t, 1] = center[1] + radius * math.sin(angle)
    
    return coords

def generate_bounce_pattern(start_point, theta, velocity, bounds=(2, 30), num_frames=20):
    """
    Generate a trajectory with bounces off boundaries.
    
    Args:
        start_point (tuple): Starting (x, y) coordinates.
        theta (float): Initial direction angle in radians.
        velocity (float): Speed in pixels per frame.
        bounds (tuple): (min, max) coordinate bounds.
        num_frames (int): Number of frames to generate.
        
    Returns:
        torch.Tensor: Coordinates of shape [num_frames, 2] and bool indicating if at least one bounce occurred.
    """
    coords = torch.zeros(num_frames, 2)
    
    # Initialize position and velocity components
    x, y = start_point
    vx = velocity * math.cos(theta)
    vy = velocity * math.sin(theta)
    min_bound, max_bound = bounds
    
    bounce_occurred = False
    
    for t in range(num_frames):
        # Update position
        x += vx
        y += vy
        
        # Check for boundary collisions and bounce with proper reflection
        if x <= min_bound:
            vx = -vx  # Reflect x-component of velocity
            x = min_bound + (min_bound - x)  # Reflect position back inside bounds
            bounce_occurred = True
        elif x >= max_bound:
            vx = -vx  # Reflect x-component of velocity
            x = max_bound - (x - max_bound)  # Reflect position back inside bounds
            bounce_occurred = True
        
        if y <= min_bound:
            vy = -vy  # Reflect y-component of velocity
            y = min_bound + (min_bound - y)  # Reflect position back inside bounds
            bounce_occurred = True
        elif y >= max_bound:
            vy = -vy  # Reflect y-component of velocity
            y = max_bound - (y - max_bound)  # Reflect position back inside bounds
            bounce_occurred = True
        
        coords[t, 0] = x
        coords[t, 1] = y
    
    return coords, bounce_occurred

def generate_zigzag_pattern(start_point, theta, velocity, turn_interval=4, turn_angle=45, num_frames=20):
    """
    Generate a zigzag trajectory with fixed turning angles.
    
    Args:
        start_point (tuple): Starting (x, y) coordinates.
        theta (float): Initial direction angle in radians.
        velocity (float): Speed in pixels per frame.
        turn_interval (int): Frames between turns.
        turn_angle (float): Turn angle in degrees.
        num_frames (int): Number of frames to generate.
        
    Returns:
        torch.Tensor: Coordinates of shape [num_frames, 2].
    """
    coords = torch.zeros(num_frames, 2)
    
    x, y = start_point
    current_theta = theta
    turn_angle_rad = math.radians(turn_angle)
    
    for t in range(num_frames):
        # Update position
        coords[t, 0] = x
        coords[t, 1] = y
        
        # Turn direction every turn_interval frames
        if t > 0 and t % turn_interval == 0:
            # Alternate between left and right turns
            turn_direction = 1 if (t // turn_interval) % 2 == 0 else -1
            current_theta += turn_direction * turn_angle_rad
        
        # Update position for next frame
        x += velocity * math.cos(current_theta)
        y += velocity * math.sin(current_theta)
    
    return coords

def clip_inside(coords, bounds=(0, 32)):
    """
    Clip coordinates to ensure they stay within bounds.
    
    Args:
        coords: Tensor or array of coordinates.
        bounds: (min, max) bounds for both dimensions.
    
    Returns:
        clipped coordinates.
    """
    min_bound, max_bound = bounds
    return torch.clamp(coords, min=min_bound, max=max_bound-1)

def coordinates_to_heatmap(coords, img_size=32, sigma=1.0):
    """
    Convert coordinate trajectory to Gaussian heatmaps.
    
    Args:
        coords: Tensor of shape [T, 2] with x,y coordinates.
        img_size: Size of output heatmaps.
        sigma: Standard deviation of Gaussian kernel.
        
    Returns:
        Tensor of shape [T, 1, H, W] with heatmaps.
    """
    T = coords.shape[0]
    heatmaps = torch.zeros(T, 1, img_size, img_size)
    
    # Ensure coordinates are float type
    coords = coords.float()
    
    # Create coordinate grid
    y_grid, x_grid = torch.meshgrid(torch.arange(img_size, dtype=torch.float32), 
                                   torch.arange(img_size, dtype=torch.float32))
    
    for t in range(T):
        x, y = coords[t]
        
        # Calculate Gaussian
        gaussian = torch.exp(
            -((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2)
        )
        
        heatmaps[t, 0] = gaussian
        
        # Normalize if needed
        if gaussian.max() > 0:
            heatmaps[t, 0] = heatmaps[t, 0] / gaussian.max()
    
    return heatmaps

def coordinates_to_pixel(coords, img_size=32):
    """
    Convert coordinate trajectory to single-pixel binary maps.
    
    Args:
        coords: Tensor of shape [T, 2] with x,y coordinates.
        img_size: Size of output images.
        
    Returns:
        Tensor of shape [T, 1, H, W] with binary pixel maps.
    """
    T = coords.shape[0]
    pixel_maps = torch.zeros(T, 1, img_size, img_size)
    
    for t in range(T):
        x, y = coords[t]
        
        # Round to nearest integer and clip to image boundaries
        x_idx = max(0, min(img_size-1, int(round(x.item()))))
        y_idx = max(0, min(img_size-1, int(round(y.item()))))
        
        pixel_maps[t, 0, y_idx, x_idx] = 1.0
    
    return pixel_maps

class GeometricDataset(Dataset):
    """
    Dataset for synthetic geometric patterns.
    
    This dataset generates four types of geometric patterns:
    1. Line: Straight line trajectories with controlled direction and speed
    2. Arc: Circular arc trajectories with controlled radius and speed
    3. Bounce: Trajectories with boundary bounces
    4. Zigzag: Trajectories with fixed turning angles
    
    Each pattern type follows specific sampling parameters to ensure:
    - Perfect predictability (analytically generated)
    - Orthogonal parameters (speed and direction sampled independently)
    - Uniform coverage (directions uniformly sampled)
    - Boundary safety (trajectories don't hit boundaries too early, except for Bounce)
    - Reproducibility (all randomness seeded by pattern_id and sample_id)
    """
    
    def __init__(self, split='train', num_samples=1000, img_size=32, sequence_length=20, 
                 pattern_types=None, require_safe_margin=True, cache_formats=True):
        """
        Initialize the dataset.
        
        Args:
            split (str): One of 'train', 'val', or 'test'.
            num_samples (int): Number of samples to generate.
            img_size (int): Size of the image space.
            sequence_length (int): Length of each trajectory.
            pattern_types (list): List of pattern types to include ['line', 'arc', 'bounce', 'zigzag'].
            require_safe_margin (bool): Whether to ensure trajectories don't hit boundaries early.
            cache_formats (bool): Whether to cache different data formats.
        """
        self.split = split
        self.num_samples = num_samples
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.require_safe_margin = require_safe_margin
        self.cache_formats = cache_formats
        
        if pattern_types is None:
            pattern_types = ['line', 'arc', 'bounce']
        self.pattern_types_to_generate = pattern_types
        
        # Set random seed based on split for reproducibility
        seed_map = {'train': 42, 'val': 43, 'test': 44}
        base_seed = seed_map.get(split, 42)
        
        # Generate all patterns using vectorized approach
        self.coordinates, self.pattern_types, self.target_speeds = self._generate_vectorized(
            base_seed, num_samples, img_size, sequence_length, pattern_types
        )
        
        # Initialize format cache
        self._format_cache = {}
        
        print(f"Generated {split} data: {len(self.coordinates)} sequences, "
              f"{sequence_length} frames each")
        for pattern in pattern_types:
            count = self.pattern_types.count(pattern)
            print(f"  - {pattern}: {count} sequences")
    
    def _generate_vectorized(self, base_seed, num_samples, img_size, sequence_length, pattern_types):
        """Generate patterns using vectorized approach for better performance."""
        all_coordinates = []
        all_pattern_types = []
        all_target_speeds = []
        
        # Fixed samples per pattern type (50% line, 50% arc)
        target_line_samples = int(num_samples * LINE_RATIO)
        target_arc_samples = int(num_samples * ARC_RATIO)
        
        # Generate enough samples to ensure balanced distribution
        line_samples = target_line_samples * 2  # Generate 2x to account for filtering
        arc_samples = target_arc_samples * 5   # Generate 5x since arcs have lower success rate
        
        # Generate line samples
        if 'line' in pattern_types:
            coords, speeds = self._generate_line_samples_vectorized(
                base_seed, line_samples, img_size, sequence_length)
            # Take exactly the target number of samples
            coords = coords[:target_line_samples]
            speeds = speeds[:target_line_samples]
            all_coordinates.extend(coords)
            all_pattern_types.extend(['line'] * len(coords))
            all_target_speeds.extend(speeds)
        
        # Generate arc samples
        if 'arc' in pattern_types:
            coords, speeds = self._generate_arc_samples_vectorized(
                base_seed + 1000, arc_samples, img_size, sequence_length)
            # Take exactly the target number of samples
            coords = coords[:target_arc_samples]
            speeds = speeds[:target_arc_samples]
            all_coordinates.extend(coords)
            all_pattern_types.extend(['arc'] * len(coords))
            all_target_speeds.extend(speeds)
        
        return all_coordinates, all_pattern_types, all_target_speeds
    
    def _generate_line_samples_vectorized(self, base_seed, num_samples, img_size, sequence_length):
        """Generate line pattern samples using vectorized approach."""
        # Grid centers for starting points
        grid_points = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = (i + 0.5) * img_size / GRID_SIZE
                y = (j + 0.5) * img_size / GRID_SIZE
                grid_points.append((x, y))
        
        # Pre-sample all parameters using uniform distribution
        rng = random.Random(base_seed)
        start_points = [rng.choice(grid_points) for _ in range(num_samples)]
        
        # Sample angles and speeds from uniform distribution using shared parameters
        thetas = [math.radians(rng.uniform(ANGLE_MIN, ANGLE_MAX)) for _ in range(num_samples)]
        velocities = [rng.uniform(SPEED_MIN, SPEED_MAX) for _ in range(num_samples)]
        
        coords_list = []
        speed_list = []
        
        for i in range(num_samples):
            # Use deterministic seed
            seed = base_seed * 10000 + i
            rng = random.Random(seed)
            
            # Use pre-sampled parameters
            start_point = start_points[i]
            theta = thetas[i]
            velocity = velocities[i]
            
            # Round velocity to nearest pixel
            velocity = round(velocity, 1)
            
            # Generate trajectory
            coords = generate_line_pattern(start_point, theta, velocity, sequence_length)
            
            # Check if trajectory stays within bounds
            if torch.min(coords) >= 0 and torch.max(coords) <= 32:
                coords_list.append(coords)
                speed_list.append(velocity)
        
        return coords_list, speed_list
    
    def _generate_arc_samples_vectorized(self, base_seed, num_samples, img_size, sequence_length):
        """Generate arc pattern samples using vectorized approach."""
        # Grid centers for circle centers
        grid_points = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = (i + 0.5) * img_size / GRID_SIZE
                y = (j + 0.5) * img_size / GRID_SIZE
                grid_points.append((x, y))
        
        # Pre-sample all parameters using uniform distribution
        rng = random.Random(base_seed)
        centers = [rng.choice(grid_points) for _ in range(num_samples)]
        
        # Sample from uniform distributions using shared parameters
        radius_list = [rng.uniform(ARC_RADIUS_MIN, ARC_RADIUS_MAX) for _ in range(num_samples)]
        velocities = [rng.uniform(SPEED_MIN, SPEED_MAX) for _ in range(num_samples)]
        start_angles = [math.radians(rng.uniform(ANGLE_MIN, ANGLE_MAX)) for _ in range(num_samples)]
        
        directions = [1, -1]  # CCW, CW
        direction_list = [rng.choice(directions) for _ in range(num_samples)]
        arc_factor_list = [rng.choice(ARC_FACTORS) for _ in range(num_samples)]
        
        coords_list = []
        speed_list = []
        
        for i in range(num_samples):
            # Use deterministic seed
            seed = base_seed * 10000 + i + 5000
            rng = random.Random(seed)
            
            # Use pre-sampled parameters
            center = centers[i]
            radius = radius_list[i]
            velocity = velocities[i]
            start_angle = start_angles[i]
            direction = direction_list[i]
            arc_factor = arc_factor_list[i]
            
            # Round parameters to nearest pixel
            radius = round(radius, 1)
            velocity = round(velocity, 1)
            
            # Ensure the circle doesn't go out of bounds
            if (center[0] - radius < BOUNDARY_MARGIN or center[0] + radius > img_size - BOUNDARY_MARGIN or
                center[1] - radius < BOUNDARY_MARGIN or center[1] + radius > img_size - BOUNDARY_MARGIN):
                continue
            
            # Calculate angular velocity and total angle to cover
            angular_velocity = velocity / radius
            total_angle = 2 * math.pi * arc_factor  # Partial circle
            
            # Calculate number of frames needed for this arc
            frames_needed = int(total_angle / angular_velocity)
            actual_frames = min(frames_needed, sequence_length)
            
            # Ensure we have at least 20 frames
            if actual_frames < sequence_length:
                # Pad with the last position
                coords = torch.zeros(sequence_length, 2)
                for t in range(actual_frames):
                    angle = start_angle + direction * angular_velocity * t
                    coords[t, 0] = center[0] + radius * math.cos(angle)
                    coords[t, 1] = center[1] + radius * math.sin(angle)
                
                # Fill remaining frames with the last position
                last_angle = start_angle + direction * angular_velocity * (actual_frames - 1)
                last_x = center[0] + radius * math.cos(last_angle)
                last_y = center[1] + radius * math.sin(last_angle)
                for t in range(actual_frames, sequence_length):
                    coords[t, 0] = last_x
                    coords[t, 1] = last_y
            else:
                # Generate trajectory with partial arc
                coords = torch.zeros(sequence_length, 2)
                for t in range(sequence_length):
                    angle = start_angle + direction * angular_velocity * t
                    coords[t, 0] = center[0] + radius * math.cos(angle)
                    coords[t, 1] = center[1] + radius * math.sin(angle)
            
            # Check if trajectory stays within bounds
            if torch.min(coords) >= 0 and torch.max(coords) <= 32:
                coords_list.append(coords)
                speed_list.append(velocity)
        
        return coords_list, speed_list
    
    def _generate_bounce_samples_vectorized(self, base_seed, num_samples, img_size, sequence_length):
        """Generate bounce pattern samples using vectorized approach."""
        # Grid centers for starting points
        grid_points = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = (i + 0.5) * img_size / GRID_SIZE
                y = (j + 0.5) * img_size / GRID_SIZE
                grid_points.append((x, y))
        
        # Bounds for bounce (with margin)
        bounds = (BOUNDARY_MARGIN, img_size-BOUNDARY_MARGIN)
        
        # Pre-sample all parameters using uniform distribution
        rng = random.Random(base_seed)
        start_points = [rng.choice(grid_points) for _ in range(num_samples)]
        thetas = [math.radians(rng.uniform(BOUNCE_ANGLE_MIN, BOUNCE_ANGLE_MAX)) for _ in range(num_samples)]
        velocities = [rng.uniform(BOUNCE_SPEED_MIN, BOUNCE_SPEED_MAX) for _ in range(num_samples)]
        
        coords_list = []
        speed_list = []
        
        for i in range(num_samples):
            # Use deterministic seed
            seed = base_seed * 10000 + i + 10000
            rng = random.Random(seed)
            
            # Use pre-sampled parameters
            start_point = start_points[i]
            theta = thetas[i]
            velocity = velocities[i]
            
            # Round velocity to nearest pixel
            velocity = round(velocity, 1)
            
            # Generate trajectory with bounce check
            coords, bounce_occurred = generate_bounce_pattern(start_point, theta, velocity, 
                                                           bounds, sequence_length)
            
            # Only keep trajectories with at least one bounce
            if bounce_occurred and torch.min(coords) >= MIN_BOUNDARY_DISTANCE and torch.max(coords) < MAX_BOUNDARY_DISTANCE:
                coords_list.append(coords)
                speed_list.append(velocity)
        
        return coords_list, speed_list
    
    def _generate_zigzag_samples_vectorized(self, base_seed, num_samples, img_size, sequence_length):
        """Generate zigzag pattern samples using vectorized approach."""
        # Grid centers for starting points
        grid_size = 4
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) * img_size / grid_size
                y = (j + 0.5) * img_size / grid_size
                grid_points.append((x, y))
        
        # Directions (8 uniformly spaced angles)
        directions = [i * (2 * math.pi / 8) for i in range(8)]
        
        # Speeds
        speeds = [1.0, 2.0, 3.0]
        
        # Turn intervals
        turn_intervals = [4, 5]
        
        # Pre-sample all parameters
        rng = random.Random(base_seed)
        start_points = [rng.choice(grid_points) for _ in range(num_samples)]
        thetas = [rng.choice(directions) for _ in range(num_samples)]
        velocities = [rng.choice(speeds) for _ in range(num_samples)]
        turn_interval_list = [rng.choice(turn_intervals) for _ in range(num_samples)]
        
        coords_list = []
        speed_list = []
        
        for i in range(num_samples):
            # Use deterministic seed
            seed = base_seed * 10000 + i + 15000
            rng = random.Random(seed)
            
            # Use pre-sampled parameters
            start_point = start_points[i]
            theta = thetas[i]
            velocity = velocities[i]
            turn_interval = turn_interval_list[i]
            
            # Generate trajectory
            coords = generate_zigzag_pattern(start_point, theta, velocity, turn_interval, 45, sequence_length)
            
            # Check boundary safety if required
            if self.require_safe_margin:
                margin = 2
                if (torch.min(coords) >= margin and torch.max(coords) < img_size - margin):
                    coords_list.append(coords)
                    speed_list.append(velocity)
            else:
                if torch.min(coords) >= 0 and torch.max(coords) < img_size:
                    coords_list.append(coords)
                    speed_list.append(velocity)
        
        return coords_list, speed_list
    
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        """
        Get a single sequence with multiple data formats.
        
        Returns:
            dict: Contains coordinates, pattern_type, target_speed, and various data formats.
        """
        coords = self.coordinates[idx]
        pattern_type = self.pattern_types[idx]
        target_speed = self.target_speeds[idx]
        
        result = {
            'coordinates': coords,
            'pattern_type': pattern_type,
            'target_speed': target_speed,
            'fixation_mask': torch.ones(self.sequence_length, dtype=torch.bool)
        }
        
        # Generate different data formats if caching is enabled
        if self.cache_formats:
            cache_key = f"{idx}_{pattern_type}"
            
            if cache_key not in self._format_cache:
                # Generate all formats
                heatmap_sigma1 = coordinates_to_heatmap(coords, self.img_size, sigma=1.0)
                heatmap_sigma2 = coordinates_to_heatmap(coords, self.img_size, sigma=2.0)
                pixel = coordinates_to_pixel(coords, self.img_size)
                
                self._format_cache[cache_key] = {
                    'heatmap_σ1': heatmap_sigma1,
                    'heatmap_σ2': heatmap_sigma2,
                    'pixel': pixel
                }
            
            result.update(self._format_cache[cache_key])
        
        return result

def create_data_loaders(batch_size=BATCH_SIZE, num_samples=TRAIN_SAMPLES, img_size=IMG_SIZE, sequence_length=SEQUENCE_LENGTH, 
                      pattern_types=None, require_safe_margin=True):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        batch_size (int): Batch size for the data loaders.
        num_samples (int): Number of samples to generate for the train split.
        img_size (int): Size of the image space.
        sequence_length (int): Length of each trajectory.
        pattern_types (list): List of pattern types to include ['line', 'arc', 'bounce', 'zigzag'].
        require_safe_margin (bool): Whether to ensure trajectories don't hit boundaries early.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = GeometricDataset('train', num_samples, img_size, sequence_length, 
                                   pattern_types, require_safe_margin)
    val_dataset = GeometricDataset('val', VAL_SAMPLES, img_size, sequence_length, 
                                 pattern_types, require_safe_margin)
    test_dataset = GeometricDataset('test', TEST_SAMPLES, img_size, sequence_length, 
                                  pattern_types, require_safe_margin)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader 