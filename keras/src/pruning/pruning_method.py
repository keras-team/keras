"""Pruning method classes for different pruning algorithms."""

import abc

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export


@keras_export("keras.pruning.PruningMethod")
class PruningMethod(abc.ABC):
    """Abstract base class for pruning methods.

    A pruning method defines the algorithm used to determine which weights
    to prune from a layer.
    """

    @abc.abstractmethod
    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute a binary mask indicating which weights to prune.

        Args:
            weights: Weight tensor to analyze.
            sparsity_ratio: Float between 0 and 1. Fraction of weights to prune.
            **kwargs: Additional arguments like model, loss_fn, input_data, target_data.

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


@keras_export("keras.pruning.L1Pruning")
class L1Pruning(PruningMethod):
    """L1 norm-based pruning method.

    Prunes weights with smallest L1 magnitude (absolute value).
    Supports both unstructured and structured pruning.
    """

    def __init__(self, structured=False):
        """Initialize L1 pruning.

        Args:
            structured: If True, prune entire channels/filters based on L1 norm.
                       If False, prune individual weights.
        """
        self.structured = structured

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute mask based on L1 norms."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        if self.structured:
            return self._compute_structured_mask(weights, sparsity_ratio)
        else:
            return self._compute_unstructured_mask(weights, sparsity_ratio)

    def _compute_unstructured_mask(self, weights, sparsity_ratio):
        """Unstructured L1 pruning."""
        l1_weights = ops.abs(weights)
        flat_weights = ops.reshape(l1_weights, [-1])

        # Convert ops.size to int for calculation
        total_size = int(backend.convert_to_numpy(ops.size(flat_weights)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_weights = ops.sort(flat_weights)
        threshold = sorted_weights[k]

        mask = l1_weights > threshold
        return mask

    def _compute_structured_mask(self, weights, sparsity_ratio):
        """Structured L1 pruning."""
        if len(ops.shape(weights)) == 2:  # Dense layer
            l1_norms = ops.sum(ops.abs(weights), axis=0)
        elif len(ops.shape(weights)) == 4:  # Conv2D layer
            l1_norms = ops.sum(ops.abs(weights), axis=(0, 1, 2))
        else:
            # Fall back to unstructured for other shapes
            return self._compute_unstructured_mask(weights, sparsity_ratio)

        flat_norms = ops.reshape(l1_norms, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_norms)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_norms = ops.sort(flat_norms)
        threshold = sorted_norms[k]

        channel_mask = l1_norms > threshold

        # Broadcast to weight tensor shape
        if len(ops.shape(weights)) == 2:
            mask = ops.broadcast_to(channel_mask[None, :], ops.shape(weights))
        elif len(ops.shape(weights)) == 4:
            mask = ops.broadcast_to(
                channel_mask[None, None, None, :], ops.shape(weights)
            )

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

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute mask based on channel/filter norms."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        # Compute L2 norms along appropriate axes
        if len(ops.shape(weights)) == 2:  # Dense layer
            norms = ops.sqrt(ops.sum(ops.square(weights), axis=0))
        elif len(ops.shape(weights)) == 4:  # Conv2D layer
            norms = ops.sqrt(ops.sum(ops.square(weights), axis=(0, 1, 2)))
        else:
            # Fall back to L1 pruning for other shapes
            return L1Pruning(structured=False).compute_mask(
                weights, sparsity_ratio
            )

        # Find threshold
        flat_norms = ops.reshape(norms, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_norms)))
        k = int(sparsity_ratio * total_size)
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

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
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
            random_vals = ops.random.uniform(
                ops.shape(weights), dtype=weights.dtype
            )

        # Keep weights where random value > sparsity_ratio
        mask = random_vals > sparsity_ratio
        return mask


@keras_export("keras.pruning.LnPruning")
class LnPruning(PruningMethod):
    """Ln norm-based pruning method.

    Prunes weights with smallest Ln norm magnitude.
    Supports both unstructured and structured pruning.
    """

    def __init__(self, n=2, structured=False):
        """Initialize Ln pruning.

        Args:
            n: Norm order (e.g., 1 for L1, 2 for L2, etc.).
            structured: If True, prune entire channels/filters.
        """
        self.n = n
        self.structured = structured

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute mask based on Ln norms."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        if self.structured:
            return self._compute_structured_mask(weights, sparsity_ratio)
        else:
            return self._compute_unstructured_mask(weights, sparsity_ratio)

    def _compute_unstructured_mask(self, weights, sparsity_ratio):
        """Unstructured Ln pruning."""
        if self.n == 1:
            ln_weights = ops.abs(weights)
        elif self.n == 2:
            ln_weights = ops.abs(weights)  # For ranking, sqrt not needed
        else:
            ln_weights = ops.power(ops.abs(weights), self.n)

        flat_weights = ops.reshape(ln_weights, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_weights)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_weights = ops.sort(flat_weights)
        threshold = sorted_weights[k]

        mask = ln_weights > threshold
        return mask

    def _compute_structured_mask(self, weights, sparsity_ratio):
        """Structured Ln pruning."""
        if len(ops.shape(weights)) == 2:  # Dense layer
            if self.n == 1:
                ln_norms = ops.sum(ops.abs(weights), axis=0)
            elif self.n == 2:
                ln_norms = ops.sqrt(ops.sum(ops.square(weights), axis=0))
            else:
                ln_norms = ops.power(
                    ops.sum(ops.power(ops.abs(weights), self.n), axis=0),
                    1.0 / self.n,
                )
        elif len(ops.shape(weights)) == 4:  # Conv2D layer
            if self.n == 1:
                ln_norms = ops.sum(ops.abs(weights), axis=(0, 1, 2))
            elif self.n == 2:
                ln_norms = ops.sqrt(
                    ops.sum(ops.square(weights), axis=(0, 1, 2))
                )
            else:
                ln_norms = ops.power(
                    ops.sum(
                        ops.power(ops.abs(weights), self.n), axis=(0, 1, 2)
                    ),
                    1.0 / self.n,
                )
        else:
            return self._compute_unstructured_mask(weights, sparsity_ratio)

        flat_norms = ops.reshape(ln_norms, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_norms)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_norms = ops.sort(flat_norms)
        threshold = sorted_norms[k]

        channel_mask = ln_norms > threshold

        # Broadcast to weight tensor shape
        if len(ops.shape(weights)) == 2:
            mask = ops.broadcast_to(channel_mask[None, :], ops.shape(weights))
        elif len(ops.shape(weights)) == 4:
            mask = ops.broadcast_to(
                channel_mask[None, None, None, :], ops.shape(weights)
            )

        return mask


@keras_export("keras.pruning.SaliencyPruning")
class SaliencyPruning(PruningMethod):
    """Gradient-based saliency pruning method.

    Estimates weight importance using first-order gradients.
    """

    def __init__(self):
        """Initialize saliency pruning."""
        pass

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute saliency-based mask using gradients."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        # Get model and data from kwargs (passed by core.py)
        model = kwargs.get('model')
        loss_fn = kwargs.get('loss_fn')
        dataset = kwargs.get('dataset')
        
        if model is None or dataset is None:
            # Fall back to magnitude pruning if data not available
            flat_weights = ops.reshape(ops.abs(weights), [-1])
            total_size = int(backend.convert_to_numpy(ops.size(flat_weights)))
            k = int(sparsity_ratio * total_size)
            if k == 0:
                return ops.ones_like(weights, dtype="bool")
            sorted_weights = ops.sort(flat_weights)
            threshold = sorted_weights[k]
            mask = ops.abs(weights) > threshold
            return mask

        # Compute saliency scores (|weight * gradient|)
        saliency_scores = self._compute_saliency_scores(weights, model, loss_fn, dataset)

        flat_scores = ops.reshape(saliency_scores, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_scores)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_scores = ops.sort(flat_scores)
        threshold = sorted_scores[k]

        mask = saliency_scores > threshold
        return mask

    def _compute_saliency_scores(self, weights, model, loss_fn, dataset):
        """Compute saliency scores using gradients."""
        # For now, use weight magnitude as approximation
        # TODO: Implement actual gradient computation with GradientTape
        return ops.abs(weights)


@keras_export("keras.pruning.TaylorPruning")
class TaylorPruning(PruningMethod):
    """Second-order Taylor expansion based pruning method.

    Estimates weight importance using second-order Taylor expansion.
    """

    def __init__(self):
        """Initialize Taylor pruning."""
        pass

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute Taylor expansion based mask."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        # Get model and data from kwargs (passed by core.py)
        model = kwargs.get('model')
        loss_fn = kwargs.get('loss_fn')
        dataset = kwargs.get('dataset')
        
        if model is None or dataset is None:
            # Fall back to magnitude pruning if data not available
            flat_weights = ops.reshape(ops.abs(weights), [-1])
            total_size = int(backend.convert_to_numpy(ops.size(flat_weights)))
            k = int(sparsity_ratio * total_size)
            if k == 0:
                return ops.ones_like(weights, dtype="bool")
            sorted_weights = ops.sort(flat_weights)
            threshold = sorted_weights[k]
            mask = ops.abs(weights) > threshold
            return mask

        # Compute Taylor scores
        taylor_scores = self._compute_taylor_scores(weights, model, loss_fn, dataset)

        flat_scores = ops.reshape(taylor_scores, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_scores)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_scores = ops.sort(flat_scores)
        threshold = sorted_scores[k]

        mask = taylor_scores > threshold
        return mask

    def _compute_taylor_scores(self, weights, model, loss_fn, dataset):
        """Compute second-order Taylor expansion scores."""
        # For now, use weight magnitude as approximation
        # TODO: Implement actual second-order Taylor computation
        return ops.abs(weights)
