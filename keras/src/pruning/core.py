"""Core pruning functionality."""

import numpy as np
import re

from keras.src import backend
from keras.src import ops


def get_model_sparsity(model):
    """Calculate the overall sparsity of a model."""
    total_weights = 0
    zero_weights = 0

    for layer in model.layers:
        if hasattr(layer, "kernel") and layer.kernel is not None:
            weights = layer.kernel.value
            total_weights += ops.size(weights)
            zero_weights += ops.sum(ops.cast(weights == 0, "int32"))

        if hasattr(layer, "bias") and layer.bias is not None:
            bias = layer.bias.value
            total_weights += ops.size(bias)
            zero_weights += ops.sum(ops.cast(bias == 0, "int32"))

    if total_weights == 0:
        return 0.0
    return float(zero_weights / total_weights)


def should_prune_layer(layer, layers_to_prune=None):
    """Determine if a layer should be pruned based on type and selection criteria.
    
    Args:
        layer: The layer to check.
        layers_to_prune: Optional specification of which layers to prune. Can be:
            - None: Prune all eligible layers (default behavior)
            - List of layer names: Only prune layers with names in the list
            - List of regex patterns: Prune layers whose names match any pattern
            - Single string: Treated as a layer name or regex pattern
    
    Returns:
        bool: True if the layer should be pruned, False otherwise.
    """
    # First check if layer is prunable by type
    layer_types = ("Dense", "Conv1D", "Conv2D", "Conv3D", "DepthwiseConv2D")
    if not (
        layer.__class__.__name__ in layer_types
        and hasattr(layer, "kernel")
        and layer.kernel is not None
    ):
        return False
    
    # If no specific layers specified, prune all eligible layers
    if layers_to_prune is None:
        return True
    
    layer_name = layer.name
    
    # Handle single string (layer name or pattern)
    if isinstance(layers_to_prune, str):
        layers_to_prune = [layers_to_prune]
    
    # Check against each specification
    for spec in layers_to_prune:
        # Try exact name match first
        if spec == layer_name:
            return True
        
        # Try regex pattern match
        try:
            if re.match(spec, layer_name):
                return True
        except re.error:
            # If regex fails, continue to next spec
            continue
    
    return False


def match_layers_by_patterns(model, patterns):
    """Helper function to find layers matching given patterns.
    
    Args:
        model: Keras model.
        patterns: List of layer names or regex patterns, or single string.
    
    Returns:
        List of matched layer names.
    """
    if patterns is None:
        return [layer.name for layer in model.layers if should_prune_layer(layer)]
    
    if isinstance(patterns, str):
        patterns = [patterns]
    
    matched_layers = []
    for layer in model.layers:
        layer_name = layer.name
        for pattern in patterns:
            # Try exact match first
            if pattern == layer_name:
                matched_layers.append(layer_name)
                break
            # Try regex match
            try:
                if re.match(pattern, layer_name):
                    matched_layers.append(layer_name)
                    break
            except re.error:
                continue
    
    return matched_layers


def compute_magnitude_mask(weights, sparsity):
    """Compute pruning mask based on weight magnitudes."""
    if sparsity <= 0:
        return ops.ones_like(weights, dtype="bool")
    if sparsity >= 1:
        return ops.zeros_like(weights, dtype="bool")

    abs_weights = ops.abs(weights)
    # Convert to numpy for percentile computation, then back
    abs_weights_np = backend.convert_to_numpy(abs_weights)
    threshold = np.percentile(abs_weights_np, sparsity * 100)
    mask = abs_weights > threshold
    return mask


def compute_structured_mask(weights, sparsity):
    """Compute structured pruning mask (prune entire channels/filters)."""
    if sparsity <= 0:
        return ops.ones_like(weights, dtype="bool")
    if sparsity >= 1:
        return ops.zeros_like(weights, dtype="bool")

    # For Conv layers: compute L2 norm across spatial dimensions
    # For Dense layers: compute L2 norm across input dimension
    if len(weights.shape) == 4:  # Conv2D
        # weights shape: (height, width, in_channels, out_channels)
        norms = ops.sqrt(ops.sum(ops.square(weights), axis=(0, 1, 2)))
    elif len(weights.shape) == 2:  # Dense
        # weights shape: (in_features, out_features)
        norms = ops.sqrt(ops.sum(ops.square(weights), axis=0))
    else:
        # Fallback to magnitude pruning for other shapes
        return compute_magnitude_mask(weights, sparsity)

    # Convert to numpy for percentile computation
    norms_np = backend.convert_to_numpy(norms)
    threshold = np.percentile(norms_np, sparsity * 100)

    # Create mask that zeros out entire channels/filters
    channel_mask = norms > threshold

    # Broadcast mask to full weight tensor shape
    if len(weights.shape) == 4:  # Conv2D
        mask = ops.broadcast_to(
            ops.reshape(channel_mask, (1, 1, 1, -1)), weights.shape
        )
    elif len(weights.shape) == 2:  # Dense
        mask = ops.broadcast_to(
            ops.reshape(channel_mask, (1, -1)), weights.shape
        )

    return mask


def get_pruning_mask(layer, sparsity, method="l1", model=None, dataset=None, loss_fn=None, **kwargs):
    """Compute and return a pruning mask for a layer without applying it.
    
    This utility function extracts the mask computation logic to enable
    advanced use cases like continual learning and selective weight freezing.
    
    Args:
        layer: Keras layer to compute mask for.
        sparsity: Float between 0 and 1. Fraction of weights to prune.
        method: Pruning method - string name or PruningMethod instance.
        model: Model (required for gradient-based methods).
        dataset: Dataset for gradient-based methods (tuple of (x, y)).
        loss_fn: Loss function for gradient-based methods.
        **kwargs: Additional arguments passed to pruning methods.
    
    Returns:
        Boolean mask tensor. True = keep weight, False = prune weight.
    """
    if not should_prune_layer(layer):
        # Return all-ones mask for non-prunable layers
        weights = layer.kernel.value
        return ops.ones_like(weights, dtype="bool")
    
    weights = layer.kernel.value

    # Handle method as string or instance
    if isinstance(method, str):
        from keras.src.pruning.pruning_method import L1Pruning
        from keras.src.pruning.pruning_method import LnPruning
        from keras.src.pruning.pruning_method import StructuredPruning
        from keras.src.pruning.pruning_method import SaliencyPruning
        from keras.src.pruning.pruning_method import TaylorPruning

        if method == "magnitude" or method == "l1":
            pruning_method = L1Pruning(structured=False)
        elif method == "structured":
            pruning_method = StructuredPruning()
        elif method == "l1_structured":
            pruning_method = L1Pruning(structured=True)
        elif method == "l2":
            pruning_method = LnPruning(n=2, structured=False)
        elif method == "l2_structured":
            pruning_method = LnPruning(n=2, structured=True)
        elif method == "saliency":
            pruning_method = SaliencyPruning()
        elif method == "taylor":
            pruning_method = TaylorPruning()
        else:
            raise ValueError(f"Unknown pruning method: {method}")
    else:
        # Assume it's a PruningMethod instance
        pruning_method = method

    # Prepare kwargs for compute_mask with enhanced loss function handling
    mask_kwargs = {
        "model": model,
        "dataset": dataset, 
        "loss_fn": loss_fn,
        **kwargs
    }
    
    # Enhanced error handling for gradient-based methods
    if method in ["saliency", "taylor"] or (hasattr(pruning_method, '__class__') and 
                                            pruning_method.__class__.__name__ in ["SaliencyPruning", "TaylorPruning"]):
        if model is None:
            raise ValueError(
                f"Method '{method}' requires 'model' parameter for gradient computation. "
                f"Usage: get_pruning_mask(layer, sparsity, method='{method}', model=your_model, dataset=(x, y), loss_fn='mse')"
            )
        if dataset is None:
            raise ValueError(
                f"Method '{method}' requires 'dataset' parameter for gradient computation. "
                f"Usage: get_pruning_mask(layer, sparsity, method='{method}', model=your_model, dataset=(x, y), loss_fn='mse')"
            )
        if loss_fn is None and not hasattr(model, 'compiled_loss') and not hasattr(model, 'loss'):
            raise ValueError(
                f"Method '{method}' requires 'loss_fn' parameter when model is not compiled or has no default loss. "
                f"Usage: get_pruning_mask(layer, sparsity, method='{method}', model=your_model, dataset=(x, y), loss_fn='mse')"
            )

    # Compute mask
    mask = pruning_method.compute_mask(weights, sparsity, **mask_kwargs)
    return mask


def get_inverted_pruning_mask(layer, sparsity, method="l1", model=None, dataset=None, loss_fn=None, **kwargs):
    """Return the inverse of the pruning mask.
    
    This function is useful for continual learning scenarios where you want to:
    1. Identify important weights that should be preserved/frozen
    2. Implement the "Pruning-then-Expanding" paradigm
    3. Selectively update only certain weights during training
    
    The inverted mask indicates which weights are IMPORTANT (not pruned).
    True = important weight (should be kept/frozen), False = unimportant weight (can be pruned/retrained).
    
    Args:
        layer: Keras layer to compute inverted mask for.
        sparsity: Float between 0 and 1. Fraction of weights to identify as unimportant.
        method: Pruning method - string name or PruningMethod instance.
        model: Model (required for gradient-based methods).
        dataset: Dataset for gradient-based methods (tuple of (x, y)).
        loss_fn: Loss function for gradient-based methods.
        **kwargs: Additional arguments passed to pruning methods.
    
    Returns:
        Boolean mask tensor. True = important weight (keep/freeze), False = unimportant weight (prune/retrain).
        
    Example:
        ```python
        # Get important weights for continual learning
        important_mask = get_inverted_pruning_mask(
            layer=model.layers[1], 
            sparsity=0.3,
            method="saliency",
            model=model,
            dataset=(x_old_tasks, y_old_tasks),
            loss_fn="categorical_crossentropy"
        )
        
        # Use mask to freeze important weights during new task training
        # (This would require additional gradient masking functionality)
        ```
    """
    pruning_mask = get_pruning_mask(
        layer=layer,
        sparsity=sparsity, 
        method=method,
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        **kwargs
    )
    # Return logical NOT of pruning mask
    # pruning_mask: True = keep, False = prune
    # inverted_mask: True = important (was kept), False = unimportant (was pruned)
    return pruning_mask


def apply_pruning_to_layer(layer, sparsity, method="l1", model=None, dataset=None, loss_fn=None, reinitialize=False, **kwargs):
    """Apply pruning to a single layer.
    
    Args:
        layer: Keras layer to prune.
        sparsity: Float between 0 and 1. Fraction of weights to prune.
        method: Pruning method - string name or PruningMethod instance.
        model: Model (required for gradient-based methods).
        dataset: Dataset for gradient-based methods (tuple of (x, y)).
        loss_fn: Loss function for gradient-based methods.
        reinitialize: Boolean. If True, reinitialize pruned weights instead of zeroing them.
                     This enables the "Pruning-then-Expanding" paradigm for continual learning.
        **kwargs: Additional arguments passed to pruning methods.
    
    Returns:
        Boolean indicating if pruning was applied.
    """
    if not should_prune_layer(layer):
        return False

    weights = layer.kernel.value

    # Use the new get_pruning_mask function for consistency
    mask = get_pruning_mask(
        layer=layer,
        sparsity=sparsity,
        method=method,
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        **kwargs
    )
    
    if reinitialize:
        # Re-initialize pruned weights instead of zeroing them out
        # This implements the "Expanding" part of the Pruning-then-Expanding paradigm
        import keras
        
        # Use He/Kaiming initialization which is good for ReLU activations
        # For other activations, Glorot/Xavier might be better
        initializer = keras.initializers.get("he_uniform")
        new_weights = initializer(shape=weights.shape, dtype=weights.dtype)
        
        # Keep the original weights where mask is True, use new weights where False
        pruned_weights = ops.where(mask, weights, new_weights)
    else:
        # Default behavior: zero out pruned weights
        # Apply mask directly
        pruned_weights = weights * ops.cast(mask, weights.dtype)
        
    layer.kernel.assign(pruned_weights)

    return True


def apply_pruning_to_model(model, sparsity, method="l1", layers_to_prune=None, 
                          dataset=None, loss_fn=None, reinitialize=False, **kwargs):
    """Apply pruning to specified layers in a model.

    Args:
        model: Keras model to prune.
        sparsity: Float between 0 and 1. Fraction of weights to prune.
        method: Pruning method - string name or PruningMethod instance.
        layers_to_prune: Optional specification of which layers to prune. Can be:
            - None: Prune all eligible layers (default)
            - List of layer names: Only prune layers with names in the list
            - List of regex patterns: Prune layers whose names match any pattern
            - Single string: Treated as a layer name or regex pattern
        dataset: Dataset for gradient-based methods (tuple of (x, y)).
        loss_fn: Loss function for gradient-based methods.
        reinitialize: Boolean. If True, reinitialize pruned weights instead of zeroing them.
                     This enables the "Pruning-then-Expanding" paradigm for continual learning.
        **kwargs: Additional arguments passed to pruning methods.

    Returns:
        Dictionary with pruning statistics.
    """
    initial_sparsity = get_model_sparsity(model)
    pruned_layers = 0
    pruned_layer_names = []
    skipped_layer_names = []

    for layer in model.layers:
        if should_prune_layer(layer, layers_to_prune):
            if apply_pruning_to_layer(
                layer=layer,
                sparsity=sparsity,
                method=method,
                model=model,
                dataset=dataset,
                loss_fn=loss_fn,
                reinitialize=reinitialize,
                **kwargs
            ):
                pruned_layers += 1
                pruned_layer_names.append(layer.name)
        elif hasattr(layer, "kernel") and layer.kernel is not None:
            # Layer has weights but was skipped due to selection criteria
            skipped_layer_names.append(layer.name)

    final_sparsity = get_model_sparsity(model)
    
    # If layers_to_prune was specified, show which layers matched
    if layers_to_prune is not None:
        matched_layers = match_layers_by_patterns(model, layers_to_prune)
        
        return {
            "initial_sparsity": initial_sparsity,
            "final_sparsity": final_sparsity,
            "pruned_layers": pruned_layers,
            "target_sparsity": sparsity,
            "method": method,
            "layers_specified": layers_to_prune,
            "layers_matched": matched_layers,
            "layers_pruned": pruned_layer_names,
            "layers_skipped": skipped_layer_names,
        }
    else:
        return {
            "initial_sparsity": initial_sparsity,
            "final_sparsity": final_sparsity,
            "pruned_layers": pruned_layers,
            "target_sparsity": sparsity,
            "method": method,
            "layers_pruned": pruned_layer_names,
        }
