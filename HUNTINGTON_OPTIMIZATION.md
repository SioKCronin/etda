# Framing Huntington's Disease Detection as an Optimization Problem

## Paper Overview

**Paper:** "Detecting Manifest Huntington's Disease Using Vocal Data"  
Subramanian et al., Interspeech 2023  
[Link](https://www.isca-archive.org/interspeech_2023/subramanian23_interspeech.pdf)

**Key Findings:**
- 76 subjects (44 HD patients, 32 controls)
- Random Forest achieved 0.95 unweighted accuracy
- Used multiple feature types: acoustic, lexical, language, pause, prosodic
- Tested on different speech tasks: Stroop test, passage reading, free speech

## Why This is an Optimization Problem

The paper uses machine learning to classify HD patients, but several aspects can be optimized:

### 1. **High-Dimensional Feature Selection**
- **Problem**: 250+ features from speech data
- **Challenge**: Which features are most discriminative?
- **Optimization**: Find optimal feature subset that maximizes classification accuracy

### 2. **Feature Weighting**
- **Problem**: Different feature types (acoustic, lexical, language) have different importance
- **Challenge**: How to weight different feature types?
- **Optimization**: Find optimal weights for each feature type

### 3. **Model Hyperparameter Tuning**
- **Problem**: Random Forest has many hyperparameters (n_estimators, max_depth, etc.)
- **Challenge**: Finding optimal hyperparameter values
- **Optimization**: Tune hyperparameters to maximize performance

### 4. **Multi-Objective Optimization**
- **Problem**: Balance accuracy, sensitivity, specificity, feature count
- **Challenge**: Trade-offs between different objectives
- **Optimization**: Find Pareto-optimal solutions

## Optimization Problem Formulation

### Decision Variables (Optimization Vector)

The optimization vector `x` represents:

1. **Feature Selection Mask** (250 dimensions)
   - `x[0:250]`: Binary/continuous values indicating which features to use
   - Values: [0, 1] (thresholded to binary)

2. **Feature Type Weights** (5 dimensions)
   - `x[250:255]`: Weights for acoustic, lexical, language, pause, prosodic
   - Values: Normalized weights that sum to 1

3. **Model Hyperparameters** (3+ dimensions)
   - `x[255]`: n_estimators (normalized [0,1], mapped to [10, 200])
   - `x[256]`: max_depth (normalized [0,1], mapped to [3, 20])
   - `x[257]`: min_samples_split (normalized [0,1], mapped to [2, 20])

**Total optimization dimension: 258**

### Objective Function

```python
def objective(x):
    """
    Maximize classification accuracy while minimizing feature count.
    
    Returns:
        fitness = validation_accuracy - λ * (n_features / total_features)
    """
    # Decode optimization vector
    feature_mask = x[0:250] > 0.5  # Binary feature selection
    feature_weights = normalize(x[250:255])
    hyperparameters = decode_hyperparameters(x[255:258])
    
    # Select features
    selected_features = data[:, feature_mask]
    
    # Train model
    model = RandomForestClassifier(**hyperparameters)
    model.fit(selected_features[train_idx], labels[train_idx])
    
    # Evaluate on validation set
    predictions = model.predict(selected_features[val_idx])
    accuracy = accuracy_score(labels[val_idx], predictions)
    
    # Penalize using too many features (sparsity)
    sparsity_penalty = 0.01 * (feature_mask.sum() / 250)
    
    return accuracy - sparsity_penalty  # Maximize this
```

## Why ETDA is Perfect for This Problem

### 1. **High-Dimensional Feature Space**
- **Problem**: 250+ features create a complex search space
- **ETDA Solution**: TDA reduces dimensionality while preserving topological structure
- **Benefit**: Swarm optimization in reduced space is more efficient

### 2. **Complex Topology**
- **Problem**: Feature relationships may have complex topological structure
- **ETDA Solution**: Persistence homology identifies critical regions
- **Benefit**: Identifies regions where optimal feature combinations lie

### 3. **Multiple Local Optima**
- **Problem**: Different feature combinations may yield similar accuracy
- **ETDA Solution**: TDA identifies all promising regions
- **Benefit**: Swarm optimization explores multiple promising areas

### 4. **Global Optima Discovery**
- **Problem**: Need to find best feature combination globally
- **ETDA Solution**: TDA identifies global maxima regions, swarm optimization fine-tunes
- **Benefit**: More likely to find true optimal solution

## ETDA Workflow for HD Detection

### Step 1: Create High-Dimensional ROI
```python
# Your feature data (76 samples × 250 features)
roi_data = extract_features_from_audio_recordings()

# Create ROI
roi = create_high_dim_roi(
    data=roi_data,
    n_samples=76,
    n_features=250,
)
```

### Step 2: Initialize ETDA Optimizer
```python
optimizer = ETDAOptimizer(
    roi_data=roi_data,
    reduced_dim=20,  # Reduce from 250D to 20D
    persistence_threshold=0.1,
    reduction_method="pca",  # Or MDS, t-SNE
)
```

### Step 3: Preprocess with TDA
```python
optimizer.preprocess_tda()

# Get persistence information
persistence_info = optimizer.get_persistence_info()
print(f"Critical points: {persistence_info['critical_points']}")
```

### Step 4: Define Objective Function
```python
def hd_objective(x_reduced):
    """Objective function in reduced space."""
    # Map from reduced space to original space
    x_original = optimizer.manifold_reducer.inverse_transform(
        x_reduced.reshape(1, -1)
    )[0]
    
    # Decode optimization vector
    feature_mask = x_original[0:250] > 0.5
    # ... rest of decoding and evaluation
    
    return accuracy
```

### Step 5: Optimize
```python
best_config, best_accuracy = optimizer.optimize(
    objective=hd_objective,
    algorithm='bat',  # or 'cultural', 'philippine_eagle'
    iterations=200,
    population_size=40,
)
```

## Expected Benefits

1. **Better Feature Selection**: TDA identifies topologically significant features
2. **Faster Convergence**: Optimization in reduced space is more efficient
3. **Global Optima**: More likely to find best feature combination
4. **Interpretability**: TDA reveals structure in feature space
5. **Robustness**: Less sensitive to initial conditions

## Next Steps

1. **Load Real Data**: Replace synthetic data with actual HD dataset
2. **Extract Features**: Use the same feature extraction as the paper
3. **Implement Objective**: Create objective function that trains/evaluates models
4. **Run ETDA**: Use TDA + swarm optimization to find optimal configuration
5. **Compare Results**: Compare with paper's Random Forest baseline (0.95 accuracy)

## Example Code

See `examples/huntington_optimization_example.py` for a complete implementation.

