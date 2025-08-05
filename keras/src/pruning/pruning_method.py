"""Pruning method classes for different pruning algorithms."""

import abc
from keras.src import ops
from keras.src import backend
from keras.src.api_export import keras_export


@keras_export("keras.pruning.PruningMethod")
class PruningMethod(abc.ABC):
    """Abstract base class for pruning methods.
    
    A pruning method defines the algorithm used to determine which weights
    to prune from a layer.
    """
    
    @abc.abstractmethod
    def compute_mask(self, weights, sparsity_ratio):
        """Compute a binary mask indicating which weights to prune.
        
        Args:
            weights: Weight tensor to analyze.
            sparsity_ratio: Float between 0 and 1. Fraction of weights to prune.
            
        Returns:
            Binary mask tensor with same shape as weights.
            True = keep weight, False = prune weight.
        """
        pass
    
    def apply_mask(self, weights, mask):
        """Apply pruning mask to weights.
        
        Args:
            weights: Weight tensor to prune.
            mask: Binary mask tensor.
            
        Returns:
            Pruned weight tensor.
        """
        return weights * ops.cast(mask, weights.dtype)


@keras_export("keras.pruning.MagnitudePruning")
class MagnitudePruning(PruningMethod):
    """Magnitude-based pruning method.
    
    Prunes weights with the smallest absolute values.
    """
    
    def compute_mask(self, weights, sparsity_ratio):
        """Compute mask based on weight magnitudes."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")
        
        abs_weights = ops.abs(weights)
        flat_weights = ops.reshape(abs_weights, [-1])
        
        # Find threshold using percentile
        k = int(sparsity_ratio * ops.size(flat_weights))
        if k == 0:
            return ops.ones_like(weights, dtype="bool")
        
        # Get k-th smallest element as threshold
        sorted_weights = ops.sort(flat_weights)
        threshold = sorted_weights[k]
        
        # Create mask: keep weights above threshold
        mask = abs_weights > threshold
        return mask


@keras_export("keras.pruning.StructuredPruning")
class StructuredPruning(PruningMethod):
    """Structured pruning method.
    
    Prunes entire channels/filters based on their L2 norm.
    """
    
    def __init__(self, axis=-1):
        """Initialize structured pruning.
        
        Args:
            axis: Axis along which to compute norms for structured pruning.
                Typically -1 for output channels.
        """
        self.axis = axis
    
    def compute_mask(self, weights, sparsity_ratio):
        """Compute mask based on channel/filter norms."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")
        
        # Compute L2 norms along appropriate axes
        if len(ops.shape(weights)) == 2:  # Dense layer
            norms = ops.sqrt(ops.sum(ops.square(weights), axis=0))
            axis = 1
        elif len(ops.shape(weights)) == 4:  # Conv2D layer
            norms = ops.sqrt(ops.sum(ops.square(weights), axis=(0, 1, 2)))
            axis = -1
        else:
            # Fall back to magnitude pruning for other shapes
            return MagnitudePruning().compute_mask(weights, sparsity_ratio)
        
        # Find threshold
        flat_norms = ops.reshape(norms, [-1])
        k = int(sparsity_ratio * ops.size(flat_norms))
        if k == 0:
            return ops.ones_like(weights, dtype="bool")
        
        sorted_norms = ops.sort(flat_norms)
        threshold = sorted_norms[k]
        
        # Create channel mask
        channel_mask = norms > threshold
        
        # Broadcast mask to weight tensor shape
        if len(ops.shape(weights)) == 2:
            mask = ops.broadcast_to(channel_mask[None, :], ops.shape(weights))
        elif len(ops.shape(weights)) == 4:
            mask = ops.broadcast_to(
                channel_mask[None, None, None, :], ops.shape(weights)
            )
        
        return mask


@keras_export("keras.pruning.RandomPruning")
class RandomPruning(PruningMethod):
    """Random pruning method.
    
    Randomly prunes weights regardless of their values.
    Mainly useful for research/comparison purposes.
    """
    
    def __init__(self, seed=None):
        """Initialize random pruning.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
    
    def compute_mask(self, weights, sparsity_ratio):
        """Compute random pruning mask."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")
        
        # Generate random values and threshold
        if self.seed is not None:
            # Use deterministic random generation if seed provided
            random_vals = ops.random.uniform(
                ops.shape(weights), seed=self.seed, dtype=weights.dtype
            )
        else:
            random_vals = ops.random.uniform(ops.shape(weights), dtype=weights.dtype)
        
        # Keep weights where random value > sparsity_ratio
        mask = random_vals > sparsity_ratio
        return mask
