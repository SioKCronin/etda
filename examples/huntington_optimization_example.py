"""
ETDA Optimization for Huntington's Disease Detection

This example demonstrates how to frame the Huntington's Disease detection problem
from the paper as an optimization problem suitable for ETDA.

Paper: "Detecting Manifest Huntington's Disease Using Vocal Data"
Subramanian et al., Interspeech 2023

The optimization problem has several dimensions:
1. Feature Selection: Which features to use from the high-dimensional speech feature space
2. Feature Weighting: Optimal weights for different feature types (acoustic, lexical, language)
3. Model Hyperparameters: Random Forest, SVM, Logistic Regression parameters
4. Multi-objective: Balance accuracy, sensitivity, specificity

Why ETDA is useful:
- High-dimensional feature space (hundreds of speech features)
- Complex topological structure in feature space
- Multiple local optima (different feature combinations)
- Need to find global maxima for classification performance

Usage:
    python examples/huntington_optimization_example.py
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path to import etda
sys.path.insert(0, str(Path(__file__).parent.parent))

from etda import ETDAOptimizer, create_high_dim_roi


@dataclass
class HDFeatureSpace:
    """Huntington's Disease feature space representation."""
    
    # Feature dimensions from the paper
    acoustic_features: int = 50      # MFCCs, pitch, formants, etc.
    lexical_features: int = 100      # Language model features, word counts
    language_features: int = 50      # Readability, complexity metrics
    pause_features: int = 20         # Pause duration, frequency
    prosodic_features: int = 30      # Speech rate, rhythm
    
    # Total feature dimensions
    total_features: int = 250
    
    # Model hyperparameters
    n_estimators_range: Tuple[int, int] = (10, 200)
    max_depth_range: Tuple[int, int] = (3, 20)
    min_samples_split_range: Tuple[int, int] = (2, 20)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimension mapping."""
        return {
            'acoustic': self.acoustic_features,
            'lexical': self.lexical_features,
            'language': self.language_features,
            'pause': self.pause_features,
            'prosodic': self.prosodic_features,
        }


class HDOptimizationProblem:
    """
    Frames HD detection as an optimization problem.
    
    The optimization vector represents:
    - Feature selection (binary mask for each feature)
    - Feature weights (continuous weights for feature types)
    - Model hyperparameters (Random Forest parameters)
    
    Objective: Maximize classification performance (accuracy, F1-score, etc.)
    """
    
    def __init__(
        self,
        feature_space: HDFeatureSpace,
        data: np.ndarray,  # (n_samples, n_features)
        labels: np.ndarray,  # (n_samples,) - binary: 0=premanifest, 1=manifest
        validation_split: float = 0.2,
    ):
        """Initialize optimization problem.
        
        Args:
            feature_space: HD feature space definition
            data: Feature matrix (n_samples, n_features)
            labels: Binary labels (0=premanifest, 1=manifest)
            validation_split: Fraction for validation set
        """
        self.feature_space = feature_space
        self.data = data
        self.labels = labels
        self.validation_split = validation_split
        
        # Split data
        n_val = int(len(data) * validation_split)
        indices = np.random.permutation(len(data))
        self.train_indices = indices[n_val:]
        self.val_indices = indices[:n_val]
        
        self.train_data = data[self.train_indices]
        self.train_labels = labels[self.train_indices]
        self.val_data = data[self.val_indices]
        self.val_labels = labels[self.val_indices]
    
    def decode_optimization_vector(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Decode optimization vector into feature selection, weights, and hyperparameters.
        
        Vector structure:
        - [0:total_features]: Feature selection mask (0 or 1)
        - [total_features:total_features+5]: Feature type weights (acoustic, lexical, etc.)
        - [total_features+5]: n_estimators (normalized)
        - [total_features+6]: max_depth (normalized)
        - [total_features+7]: min_samples_split (normalized)
        
        Args:
            x: Optimization vector
            
        Returns:
            Dictionary with feature_mask, feature_weights, hyperparameters
        """
        n_features = self.feature_space.total_features
        
        # Feature selection mask (binary threshold at 0.5)
        feature_mask = (x[:n_features] > 0.5).astype(int)
        
        # Feature type weights (normalized)
        feature_weights = x[n_features:n_features+5]
        feature_weights = np.abs(feature_weights)  # Ensure positive
        feature_weights = feature_weights / (feature_weights.sum() + 1e-10)  # Normalize
        
        # Hyperparameters (denormalize from [0,1] to actual ranges)
        n_estimators = int(
            self.feature_space.n_estimators_range[0] +
            (self.feature_space.n_estimators_range[1] - self.feature_space.n_estimators_range[0]) *
            x[n_features+5]
        )
        max_depth = int(
            self.feature_space.max_depth_range[0] +
            (self.feature_space.max_depth_range[1] - self.feature_space.max_depth_range[0]) *
            x[n_features+6]
        )
        min_samples_split = int(
            self.feature_space.min_samples_split_range[0] +
            (self.feature_space.min_samples_split_range[1] - self.feature_space.min_samples_split_range[0]) *
            x[n_features+7]
        )
        
        return {
            'feature_mask': feature_mask,
            'feature_weights': feature_weights,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
            }
        }
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate optimization vector: train model and return validation accuracy.
        
        This is the objective function to maximize.
        
        Args:
            x: Optimization vector
            
        Returns:
            Validation accuracy (to maximize)
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
        except ImportError:
            raise ImportError("scikit-learn is required for this example")
        
        # Decode optimization vector
        config = self.decode_optimization_vector(x)
        
        # Apply feature selection
        selected_features = self.train_data[:, config['feature_mask'] == 1]
        val_selected_features = self.val_data[:, config['feature_mask'] == 1]
        
        # Check if any features selected
        if selected_features.shape[1] == 0:
            return 0.0  # Penalty for no features
        
        # Apply feature weights (if we want to weight different feature types)
        # For simplicity, we'll use feature selection only here
        # In a full implementation, you'd apply weights to different feature groups
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=config['hyperparameters']['n_estimators'],
            max_depth=config['hyperparameters']['max_depth'],
            min_samples_split=config['hyperparameters']['min_samples_split'],
            random_state=42,
            n_jobs=-1,
        )
        
        rf.fit(selected_features, self.train_labels)
        
        # Predict on validation set
        predictions = rf.predict(val_selected_features)
        accuracy = accuracy_score(self.val_labels, predictions)
        
        # Multi-objective: also consider feature count (sparsity)
        # Penalize using too many features (L1 regularization-like)
        n_selected = config['feature_mask'].sum()
        sparsity_penalty = 0.01 * (n_selected / self.feature_space.total_features)
        
        # Return accuracy minus sparsity penalty (to maximize)
        return accuracy - sparsity_penalty
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for optimization vector.
        
        Returns:
            List of (min, max) tuples for each dimension
        """
        n_features = self.feature_space.total_features
        bounds = []
        
        # Feature selection: [0, 1] (will be thresholded)
        bounds.extend([(0.0, 1.0)] * n_features)
        
        # Feature weights: [-1, 1] (will be normalized)
        bounds.extend([(-1.0, 1.0)] * 5)
        
        # Hyperparameters: [0, 1] (will be denormalized)
        bounds.extend([(0.0, 1.0)] * 3)
        
        return bounds


def create_synthetic_hd_data(
    n_samples: int = 76,  # From paper: 76 subjects
    n_features: int = 250,  # Total features
    n_manifest: int = 23,  # From paper: 23 manifest patients
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic HD data based on paper statistics.
    
    In practice, you would load real data from the paper's dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_manifest: Number of manifest patients
        seed: Random seed
        
    Returns:
        (data, labels) tuple
    """
    rng = np.random.RandomState(seed)
    
    # Generate synthetic features
    # Manifest patients might have different feature distributions
    data = rng.randn(n_samples, n_features)
    
    # Create labels: 0 = premanifest, 1 = manifest
    labels = np.zeros(n_samples)
    manifest_indices = rng.choice(n_samples, size=n_manifest, replace=False)
    labels[manifest_indices] = 1
    
    # Make manifest patients have slightly different feature distributions
    # (simulating the disease effect)
    for idx in manifest_indices:
        # Add some systematic differences in certain features
        data[idx, :50] += 0.5 * rng.randn(50)  # Acoustic features affected
        data[idx, 100:150] += 0.3 * rng.randn(50)  # Language features affected
    
    return data, labels


def main():
    """Demonstrate HD optimization problem setup."""
    
    print("=" * 70)
    print("Huntington's Disease Detection as Optimization Problem")
    print("=" * 70)
    print()
    
    # Create feature space
    feature_space = HDFeatureSpace()
    print("Feature Space Dimensions:")
    for name, dim in feature_space.get_feature_dimensions().items():
        print(f"  {name}: {dim} features")
    print(f"  Total: {feature_space.total_features} features")
    print()
    
    # Create synthetic data (in practice, load real data)
    print("Loading data...")
    data, labels = create_synthetic_hd_data(
        n_samples=76,
        n_features=feature_space.total_features,
        n_manifest=23,
    )
    print(f"  Samples: {len(data)}")
    print(f"  Features: {data.shape[1]}")
    print(f"  Premanifest: {(labels == 0).sum()}")
    print(f"  Manifest: {(labels == 1).sum()}")
    print()
    
    # Create optimization problem
    print("Setting up optimization problem...")
    problem = HDOptimizationProblem(
        feature_space=feature_space,
        data=data,
        labels=labels,
    )
    
    # Get optimization bounds
    bounds = problem.get_bounds()
    print(f"  Optimization dimension: {len(bounds)}")
    print(f"    - Feature selection: {feature_space.total_features} dims")
    print(f"    - Feature weights: 5 dims")
    print(f"    - Hyperparameters: 3 dims")
    print()
    
    # Test evaluation
    print("Testing objective function...")
    try:
        test_x = np.random.rand(len(bounds))
        accuracy = problem.evaluate(test_x)
        print(f"  Random configuration accuracy: {accuracy:.4f}")
        print()
    except ImportError as e:
        print(f"  ⚠ scikit-learn not available: {e}")
        print("  Install with: pip install scikit-learn")
        print("  Skipping objective function test")
        print()
    
    # Actually run ETDA optimization
    print("=" * 70)
    print("Running ETDA Optimization:")
    print("=" * 70)
    print()
    
    print("Step 1: Creating high-dimensional ROI from feature data...")
    roi_data = problem.data
    print(f"  ROI shape: {roi_data.shape}")
    print()
    
    print("Step 2: Initializing ETDA optimizer...")
    try:
        optimizer = ETDAOptimizer(
            roi_data=roi_data,
            reduced_dim=20,  # Reduce from 250D to 20D
            persistence_threshold=0.1,
            reduction_method="pca",
            random_state=42,
        )
        print("  ✓ ETDA optimizer initialized")
        print(f"  Reduced dimension: {optimizer.reduced_dim}")
        print()
        
        print("Step 3: Preprocessing with TDA...")
        optimizer.preprocess_tda()
        persistence_info = optimizer.get_persistence_info()
        print("  ✓ Persistence homology computed")
        print(f"  Persistence entropy: {persistence_info['persistence_entropy']:.4f}")
        print(f"  Critical points: {len(persistence_info['critical_points'])}")
        print()
        
        print("Step 4: Running optimization...")
        print("  (This may take a few minutes...)")
        print()
        
        # Create objective function wrapper for reduced space
        # The optimizer will automatically map from reduced to original space
        def wrapped_objective(x_original: np.ndarray) -> float:
            """Objective function that works in original space."""
            # Clamp to bounds [0, 1] for all dimensions
            x_original = np.clip(x_original, 0, 1)
            return problem.evaluate(x_original)
        
        # Run optimization
        # The optimizer handles the mapping from reduced to original space
        best_config_reduced, best_accuracy = optimizer.optimize(
            objective=wrapped_objective,
            algorithm='bat',
            iterations=100,  # Reduced for faster demo
            population_size=30,
        )
        
        # Map back to original space (if using PCA)
        if optimizer.manifold_reducer.method == "pca":
            best_config_original = optimizer.manifold_reducer.inverse_transform(
                best_config_reduced.reshape(1, -1)
            )[0]
        else:
            # For methods without inverse transform, we work in reduced space
            # In this case, we'd need to adapt the problem
            print("  ⚠ Note: Using method without inverse transform")
            print("  Results are in reduced space")
            best_config_original = best_config_reduced
        
        print("  ✓ Optimization complete!")
        print(f"  Best accuracy: {best_accuracy:.4f}")
        print()
        
        # Decode best configuration (if we have original space)
        try:
            if len(best_config_original) >= problem.feature_space.total_features:
                best_decoded = problem.decode_optimization_vector(best_config_original)
                n_selected = best_decoded['feature_mask'].sum()
                print("Best configuration:")
                print(f"  Selected features: {n_selected} / {problem.feature_space.total_features}")
                print(f"  Hyperparameters: {best_decoded['hyperparameters']}")
                print()
            else:
                print(f"Best configuration (reduced space): {best_config_reduced[:10]}...")
                print()
        except Exception as e:
            print(f"  Note: Could not decode configuration: {e}")
            print()
        
    except ImportError as e:
        print(f"  ⚠ ETDA dependencies not available: {e}")
        print("  Install with: pip install -e .")
        print("  Also install nio: pip install -e ../library-of-nature-inspired-optimization")
        print()
    
    print("=" * 70)
    print("Benefits of ETDA for this problem:")
    print("  - TDA reveals topological structure in high-D feature space")
    print("  - Identifies critical regions (optimal feature combinations)")
    print("  - Swarm optimization finds global optima in reduced space")
    print("  - Especially useful when feature space has complex topology")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

