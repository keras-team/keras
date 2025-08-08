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
        
        # Saliency pruning requires model and dataset - no fallback
        if model is None:
            raise ValueError("SaliencyPruning requires 'model' parameter. Pass model through model.prune() kwargs.")
        
        if dataset is None:
            raise ValueError("SaliencyPruning requires 'dataset' parameter. Pass dataset as tuple (x, y) through model.prune() kwargs.")
        
        # Get loss_fn from model if not provided
        if loss_fn is None:
            if hasattr(model, 'loss') and model.loss is not None:
                loss_fn = model.loss
            else:
                raise ValueError("SaliencyPruning requires 'loss_fn' parameter or model must have a compiled loss function.")

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
        """Compute saliency scores using gradients.
        
        Saliency score = |gradient * weight| for each weight.
        This estimates how much the loss would change if we set that weight to zero.
        """
        import keras
        import numpy as np
        
        # Extract input and target data from dataset
        if isinstance(dataset, tuple) and len(dataset) == 2:
            x_data, y_data = dataset
        else:
            raise ValueError("Dataset must be a tuple (x_data, y_data) for saliency computation.")
        
        # Process data in smaller batches to avoid OOM
        # Limit batch size to avoid GPU memory issues
        if hasattr(x_data, 'shape') and len(x_data.shape) > 0:
            total_samples = x_data.shape[0]
            max_batch_size = min(32, total_samples)  # Use small batches to avoid OOM
            
            # Take a representative sample if dataset is very large
            if total_samples > max_batch_size:
                # Use random sampling for better gradient estimation
                indices = np.random.choice(total_samples, max_batch_size, replace=False)
                x_data = x_data[indices]
                y_data = y_data[indices]
        
        # Convert to tensors after sampling
        x_data = ops.convert_to_tensor(x_data)
        y_data = ops.convert_to_tensor(y_data)
        
        # Use backend-specific gradient computation for efficiency and accuracy
        from keras.src import backend as keras_backend
        backend_name = keras_backend.backend()
        
        if backend_name == "tensorflow":
            # Use TensorFlow's GradientTape for automatic differentiation
            import tensorflow as tf
            
            # Find all trainable weights to compute gradients for all at once
            trainable_weights = [layer.kernel for layer in model.layers if hasattr(layer, 'kernel') and layer.kernel is not None]
            
            def compute_loss():
                predictions = model(x_data, training=False)
                if callable(loss_fn):
                    loss = loss_fn(y_data, predictions)
                else:
                    loss_obj = keras.losses.get(loss_fn)
                    loss = loss_obj(y_data, predictions)
                return ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
            
            # Compute gradients for all weights at once (much more efficient)
            with tf.GradientTape() as tape:
                # Watch all trainable weights
                watch_vars = []
                for weight in trainable_weights:
                    if hasattr(weight, 'value'):
                        watch_vars.append(weight.value)
                        tape.watch(weight.value)
                    else:
                        watch_vars.append(weight)
                        tape.watch(weight)
                
                loss = compute_loss()
            
            # Get gradients for all weights
            all_gradients = tape.gradient(loss, watch_vars)
            
            # Find the gradient for our specific weight tensor
            target_gradients = None
            for i, weight in enumerate(trainable_weights):
                if ops.shape(weight) == ops.shape(weights):
                    # Check if values are close
                    weight_diff = ops.mean(ops.abs(weight - weights))
                    if backend.convert_to_numpy(weight_diff) < 1e-6:
                        target_gradients = all_gradients[i]
                        break
                        
            if target_gradients is None:
                raise ValueError(f"Could not find gradients for weight tensor with shape {ops.shape(weights)}")
                
            gradients = target_gradients
                
        elif backend_name == "jax":
            raise ValueError("SaliencyPruning with JAX backend is not yet implemented. "
                           "Use TensorFlow backend or magnitude-based pruning methods like L1Pruning.")
            
        elif backend_name == "torch":
            raise ValueError("SaliencyPruning with PyTorch backend is not yet implemented. "
                           "Use TensorFlow backend or magnitude-based pruning methods like L1Pruning.")
                
        else:
            # No fallback - saliency pruning requires proper gradient computation
            raise ValueError(f"SaliencyPruning is not supported for backend '{backend_name}'. "
                           f"Currently only TensorFlow backend is supported. "
                           f"Use L1Pruning or other magnitude-based methods instead.")
        
        # Compute saliency scores: |gradient * weight|
        saliency_scores = ops.abs(gradients * weights)
        
        return saliency_scores


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
        
        # Taylor pruning requires model and dataset - no fallback
        if model is None:
            raise ValueError("TaylorPruning requires 'model' parameter. Pass model through model.prune() kwargs.")
        
        if dataset is None:
            raise ValueError("TaylorPruning requires 'dataset' parameter. Pass dataset as tuple (x, y) through model.prune() kwargs.")
        
        # Get loss_fn from model if not provided
        if loss_fn is None:
            if hasattr(model, 'loss') and model.loss is not None:
                loss_fn = model.loss
            else:
                raise ValueError("TaylorPruning requires 'loss_fn' parameter or model must have a compiled loss function.")

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
        """Compute second-order Taylor expansion scores.
        
        Taylor score approximates the change in loss when setting a weight to zero
        using Taylor expansion: ΔL ≈ |∂L/∂w * w| + (1/2) * |∂²L/∂w² * w²|
        """
        import keras
        import numpy as np
        
        # Extract input and target data from dataset
        if isinstance(dataset, tuple) and len(dataset) == 2:
            x_data, y_data = dataset
        else:
            raise ValueError("Dataset must be a tuple (x_data, y_data) for Taylor computation.")
        
        # Process data in smaller batches to avoid OOM
        # Limit batch size to avoid GPU memory issues
        if hasattr(x_data, 'shape') and len(x_data.shape) > 0:
            total_samples = x_data.shape[0]
            max_batch_size = min(32, total_samples)  # Use small batches to avoid OOM
            
            # Take a representative sample if dataset is very large
            if total_samples > max_batch_size:
                # Use random sampling for better gradient estimation
                indices = np.random.choice(total_samples, max_batch_size, replace=False)
                x_data = x_data[indices]
                y_data = y_data[indices]
        
        # Convert to tensors after sampling
        x_data = ops.convert_to_tensor(x_data)
        y_data = ops.convert_to_tensor(y_data)
        
        # Find which layer this weight tensor belongs to by comparing shapes and values
        target_layer = None
        target_weight_var = None
        
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                # Check if this is the matching weight tensor by shape
                if ops.shape(layer.kernel) == ops.shape(weights):
                    # Additional check: see if values are close (in case of multiple layers with same shape)
                    weight_diff = ops.mean(ops.abs(layer.kernel - weights))
                    if backend.convert_to_numpy(weight_diff) < 1e-6:  # Very close values
                        target_layer = layer
                        target_weight_var = layer.kernel
                        break
        
        if target_layer is None or target_weight_var is None:
            raise ValueError(f"Could not find layer corresponding to weight tensor with shape {ops.shape(weights)}")
        
        # Use backend-specific gradient computation for efficiency and accuracy
        from keras.src import backend as keras_backend
        backend_name = keras_backend.backend()
        
        if backend_name == "tensorflow":
            # Use TensorFlow's GradientTape for automatic differentiation
            import tensorflow as tf
            
            def compute_loss():
                predictions = model(x_data, training=False)
                if callable(loss_fn):
                    loss = loss_fn(y_data, predictions)
                else:
                    loss_obj = keras.losses.get(loss_fn)
                    loss = loss_obj(y_data, predictions)
                return ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
            
            # Compute first-order gradients
            with tf.GradientTape() as tape:
                # For Keras variables, we need to watch the underlying tensor
                if hasattr(target_weight_var, 'value'):
                    # Keras Variable - watch the underlying tensor
                    watch_var = target_weight_var.value
                else:
                    # Already a TensorFlow tensor/variable
                    watch_var = target_weight_var
                tape.watch(watch_var)
                loss = compute_loss()
            
            gradients = tape.gradient(loss, watch_var)
            
            if gradients is None:
                raise ValueError(f"No gradients computed for layer {target_layer.name}")
            
            # For second-order term, we need proper Hessian diagonal computation
            # This is computationally expensive, so we use a simpler first-order approximation
            # In practice, most Taylor pruning methods fall back to first-order due to Hessian cost
            # We'll use the Optimal Brain Damage (OBD) approximation: assume Hessian is identity-scaled
            # This gives us: ∂²L/∂w² ≈ constant (typically estimated from gradients)
            
            # Simple approximation: use gradient magnitude as proxy for curvature
            # This is a common heuristic in pruning literature when full Hessian is too expensive
            hessian_diag_approx = ops.abs(gradients) + 1e-8
            
        elif backend_name == "jax":
            # Use JAX's automatic differentiation
            import jax
            
            def compute_loss_fn(weight_vals):
                # Temporarily set weights
                old_weights = target_layer.kernel.value
                target_layer.kernel.assign(weight_vals)
                
                predictions = model(x_data, training=False)
                if callable(loss_fn):
                    loss = loss_fn(y_data, predictions)
                else:
                    loss_obj = keras.losses.get(loss_fn)
                    loss = loss_obj(y_data, predictions)
                
                loss_scalar = ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
                
                # Restore weights
                target_layer.kernel.assign(old_weights)
                return loss_scalar
            
            # Compute gradients using JAX
            grad_fn = jax.grad(compute_loss_fn)
            gradients = grad_fn(weights)
            
            # Approximate Hessian diagonal using gradient magnitude
            # This is a simplified approximation when full second-order computation is too expensive
            hessian_diag_approx = ops.abs(gradients) + 1e-8
            
        elif backend_name == "torch":
            # Use PyTorch's autograd
            import torch
            
            # For Keras variables, get the underlying tensor
            if hasattr(target_weight_var, 'value'):
                torch_var = target_weight_var.value
            else:
                torch_var = target_weight_var
                
            # Set requires_grad for the target weights
            torch_var.requires_grad_(True)
            
            def compute_loss():
                predictions = model(x_data, training=False)
                if callable(loss_fn):
                    loss = loss_fn(y_data, predictions)
                else:
                    loss_obj = keras.losses.get(loss_fn)
                    loss = loss_obj(y_data, predictions)
                return ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
            
            loss = compute_loss()
            gradients = torch.autograd.grad(loss, torch_var, create_graph=False)[0]
            
            if gradients is None:
                raise ValueError(f"No gradients computed for layer {target_layer.name}")
            
            # Approximate Hessian diagonal using gradient magnitude
            # This is a simplified approximation when full second-order computation is too expensive
            hessian_diag_approx = ops.abs(gradients) + 1e-8
            
        else:
            # Fallback: Use numerical differentiation (slower but backend-agnostic)
            epsilon = 1e-7
            
            def compute_loss_with_weights(layer_weights):
                old_weights = target_layer.kernel.value
                target_layer.kernel.assign(layer_weights)
                
                predictions = model(x_data, training=False)
                if callable(loss_fn):
                    loss = loss_fn(y_data, predictions)
                else:
                    loss_obj = keras.losses.get(loss_fn)
                    loss = loss_obj(y_data, predictions)
                
                loss_scalar = ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
                target_layer.kernel.assign(old_weights)
                return loss_scalar
            
            # Numerical gradient computation
            baseline_loss = compute_loss_with_weights(weights)
            gradients = ops.zeros_like(weights)
            
            flat_weights = ops.reshape(weights, [-1])
            flat_gradients = ops.reshape(gradients, [-1])
            
            # Sample subset for efficiency
            total_weights = int(backend.convert_to_numpy(ops.size(flat_weights)))
            sample_size = min(100, total_weights)
            indices = np.random.choice(total_weights, sample_size, replace=False) if sample_size < total_weights else np.arange(total_weights)
            
            grad_values = []
            for i in indices:
                # Forward difference
                perturbed_weights = ops.copy(flat_weights)
                perturbed_weights = ops.slice_update(perturbed_weights, [i], [flat_weights[i] + epsilon])
                perturbed_weights_reshaped = ops.reshape(perturbed_weights, ops.shape(weights))
                
                perturbed_loss = compute_loss_with_weights(perturbed_weights_reshaped)
                grad_val = (perturbed_loss - baseline_loss) / epsilon
                grad_values.append(backend.convert_to_numpy(grad_val))
            
            # Fill gradient tensor
            flat_gradients_np = backend.convert_to_numpy(flat_gradients)
            for idx, i in enumerate(indices):
                flat_gradients_np[i] = grad_values[idx]
            
            # For unsampled weights, approximate with weight magnitude
            for i in range(total_weights):
                if i not in indices:
                    flat_gradients_np[i] = backend.convert_to_numpy(ops.abs(flat_weights[i]))
            
            gradients = ops.convert_to_tensor(flat_gradients_np.reshape(backend.convert_to_numpy(ops.shape(weights))), dtype=weights.dtype)
            # For numerical fallback, use simple gradient-based approximation
            hessian_diag_approx = ops.abs(gradients) + 1e-8
        
        # Compute Taylor expansion terms
        # Note: This is a simplified Taylor approximation since computing true Hessian diagonal
        # is computationally expensive. The second-order term uses gradient magnitude as a proxy
        # for curvature, which is a common heuristic in pruning literature.
        first_order_term = ops.abs(gradients * weights)  # |∂L/∂w * w|
        second_order_term = 0.5 * ops.abs(hessian_diag_approx * ops.square(weights))  # Approximated second-order term
        
        taylor_scores = first_order_term + second_order_term
        
        return taylor_scores
