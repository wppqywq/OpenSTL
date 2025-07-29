from experiments_refactored.datasets.position_dependent_gaussian_dataset import (
    PositionDependentGaussianDataset, 
    calculate_displacement_vectors,
    create_data_loaders as create_position_dependent_gaussian_loaders,
    DatasetGenerator as PositionDependentGaussianGenerator
)

from experiments_refactored.datasets.geom_simple import (
    GeometricDataset,
    generate_line_pattern,
    generate_arc_pattern,
    generate_bounce_pattern,
    create_data_loaders as create_geom_simple_loaders
)

from experiments_refactored.datasets.position_dependent_gaussian import (
    sample_position_dependent_gaussian,
    render_sparse_frames,
    render_gaussian_frames,
    generate_dataset as generate_position_dependent_gaussian_dataset
)

# Backward compatibility
from experiments_refactored.datasets.eye_gauss_compat import (
    create_sparse_representation,
    create_dense_gaussian_representation
)

__all__ = [
    'PositionDependentGaussianDataset',
    'GeometricDataset',
    'PositionDependentGaussianGenerator',
    'create_sparse_representation',
    'create_dense_gaussian_representation',
    'calculate_displacement_vectors',
    'generate_line_pattern',
    'generate_arc_pattern',
    'generate_bounce_pattern',
    'create_position_dependent_gaussian_loaders',
    'create_geom_simple_loaders',
    'sample_position_dependent_gaussian',
    'render_sparse_frames',
    'render_gaussian_frames',
    'generate_position_dependent_gaussian_dataset'
] 