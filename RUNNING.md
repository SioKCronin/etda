# Running the Huntington's Disease Optimization Example

## Prerequisites

Install the required dependencies:

```bash
cd /Users/siobhan/code/etda

# Install ETDA package
pip install -e .

# Install scikit-learn (for machine learning models)
pip install scikit-learn

# Install nio (Nature-Inspired Optimization) - required dependency
# If not already installed:
cd ../library-of-nature-inspired-optimization
pip install -e .
cd ../etda
```

## Running the Example

```bash
python3 examples/huntington_optimization_example.py
```

## What It Does

1. **Creates synthetic HD data** (76 samples × 250 features) based on the paper
2. **Sets up optimization problem**:
   - Feature selection (250 dimensions)
   - Feature weights (5 dimensions)  
   - Hyperparameters (3 dimensions)
   - Total: 258-dimensional optimization space
3. **Runs ETDA**:
   - Reduces 250D → 20D using PCA
   - Computes persistence homology
   - Identifies critical points
   - Runs swarm optimization (Bat Algorithm)
4. **Outputs results**:
   - Best accuracy achieved
   - Selected features
   - Optimal hyperparameters

## Expected Output

```
======================================================================
Huntington's Disease Detection as Optimization Problem
======================================================================

Feature Space Dimensions:
  acoustic: 50 features
  lexical: 100 features
  language: 50 features
  pause: 20 features
  prosodic: 30 features
  Total: 250 features

Loading data...
  Samples: 76
  Features: 250
  Premanifest: 53
  Manifest: 23

Setting up optimization problem...
  Optimization dimension: 258
    - Feature selection: 250 dims
    - Feature weights: 5 dims
    - Hyperparameters: 3 dims

Running ETDA Optimization:
  ...
  ✓ Optimization complete!
  Best accuracy: 0.XXXX
  Selected features: XX / 250
  Hyperparameters: {...}
```

## Using Real Data

To use real HD data from the paper:

1. Load your feature data (CSV, numpy array, etc.)
2. Replace `create_synthetic_hd_data()` with your data loader
3. Update feature dimensions in `HDFeatureSpace` if needed
4. Run the same workflow

## Troubleshooting

**ImportError: No module named 'sklearn'**
- Install: `pip install scikit-learn`

**ImportError: No module named 'nio'**
- Install nio: `pip install -e ../library-of-nature-inspired-optimization`

**ImportError: No module named 'gtda'**
- Install: `pip install giotto-tda`

