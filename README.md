# ETDA: Evolutionary Topological Data Analysis

ETDA integrates swarm optimization algorithms with topological data analysis (TDA) to solve high-dimensional optimization problems, particularly for health data applications.

## Overview

ETDA couples persistence homology with swarm intelligence through the following workflow:
1. **Reduce large-scale multidimensional spaces with TDA** – Run Mapper / persistent homology / UMAP-PCA pipelines to capture dominant manifolds or cluster signatures, then export the reduced coordinates as the search domain.
2. **Define the optimization problem on the reduced representation** – Express objectives and constraints (e.g., Wasserstein distances, Betti targets, ROI metrics) directly in the reduced space; solutions can be lifted back to the original coordinates afterward.
3. **Run swarm optimization to find the global optimum** – Apply nature-inspired algorithms (Bat, PSO variants, Water Cycle, etc.) on the reduced manifold to explore and exploit efficiently.
4. **Validate and interpret the best solution** – Map the swarm-derived optimum back to the high-dimensional space and perform domain checks (dose limits, mechanistic plausibility, health-data constraints) before deployment.

## Key Features

- **Persistent Homology Analysis**: Uses giotto-tda to compute persistence diagrams
- **Manifold Reduction**: Reduces high-dimensional search spaces to lower-dimensional manifolds
- **Global Optima Identification**: Identifies critical points and global maxima on the topological structure
- **Swarm Optimization Integration**: Works with swarmopt (Particle Swarm Optimization)
- **Health Data Focus**: Designed for large-scale health data optimization problems

## Installation

```bash
pip install -e .
```

## Dependencies

- **giotto-tda**: Topological data analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning utilities
- **swarmopt**: Particle swarm optimization (automatically installed via pip)

## Quick Start

```python
from etda import ETDAOptimizer
from etda import create_high_dim_roi
import numpy as np

# Create a high-dimensional region of interest
roi_data = create_high_dim_roi(
    n_samples=1000,
    n_features=50,
    seed=42
)

# Initialize ETDA optimizer
optimizer = ETDAOptimizer(
    roi_data=roi_data,
    reduced_dim=10,
    n_components=3,
    persistence_threshold=0.1
)

# Run TDA preprocessing
optimizer.preprocess_tda()

# Optimize on reduced manifold
best_position, best_value = optimizer.optimize(
    algorithm='bat',
    iterations=200,
    population_size=40
)

print(f"Best value: {best_value}")
print(f"Best position (reduced space): {best_position}")
```

## Example: Large-Scale Multidimensional ROI

See `examples/high_dim_roi_example.py` for a complete example demonstrating:
- Large-scale multidimensional region of interest
- Persistence homology mapping
- TDA-based global maxima identification
- Swarm optimization on the reduced manifold

## License

This project is licensed under the [MIT License](LICENSE), the same permissive license used across our other optimization projects (e.g., swarmopt). You are free to use, modify, and distribute ETDA so long as the MIT notice is preserved.

> **Limited Liability Notice:** ETDA is provided “as-is” under the MIT license, without warranties of any kind. You assume all liability and responsibility for any systems, analyses, or decisions that use this software.

