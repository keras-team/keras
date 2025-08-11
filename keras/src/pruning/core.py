"""Core pruning functionality."""

import numpy as np
import re

import keras
from keras.src import backend
from keras.src import ops


def _has_kernel_weights(layer):
    """Check if a layer has kernel weights."""
    return hasattr(layer, "kernel") and layer.kernel is not None


def get_model_sparsity(model):
    """Calculate the overall sparsity of a model."""
    total_weights = 0
    zero_weights = 0

    try:
        all_layers = model._flatten_layers()
    except AttributeError:
        # Fallback for models that don't have _flatten_layers
        all_layers = model.layers

    for layer in all_layers:
        # We only want to count weights for leaf layers.
        try:
            list_of_sublayers = (
                list(layer._flatten_layers())
                if hasattr(layer, "_flatten_layers")
                else [layer]
            )
        except:
            list_of_sublayers = [layer]

        if len(list_of_sublayers) == 1:
            if _has_kernel_weights(layer):
                weights = layer.kernel.value
                total_weights += ops.size(weights)
                zero_weights += ops.sum(ops.cast(weights == 0, "int32"))

            if hasattr(layer, "bias") and layer.bias is not None:
                bias = layer.bias.value
                total_weights += ops.size(bias)
                zero_weights += ops.sum(ops.cast(bias == 0, "int32"))

    if total_weights == 0:
        return 0.0
    return float(zero_weights) / float(total_weights)


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
    layer_types = (
        "Dense",
        "Conv1D",
        "Conv2D",
        "Conv3D",
        "DepthwiseConv2D",
        "EinsumDense",
    )
    if not (
        layer.__class__.__name__ in layer_types and _has_kernel_weights(layer)
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


def _create_pruning_method(method):
    """Factory function to create pruning method instances from strings."""
    if not isinstance(method, str):
        # Assume it's already a PruningMethod instance
        return method
    
    from keras.src.pruning.pruning_method import (
        L1Pruning, LnPruning, StructuredPruning, 
        SaliencyPruning, TaylorPruning
    )
    
    method_map = {
        "magnitude": L1Pruning(structured=False),
        "l1": L1Pruning(structured=False),
        "structured": StructuredPruning(),
        "l1_structured": L1Pruning(structured=True),
        "l2": LnPruning(n=2, structured=False),
        "l2_structured": LnPruning(n=2, structured=True),
        "saliency": SaliencyPruning(),
        "taylor": TaylorPruning(),
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown pruning method: {method}")
    
    return method_map[method]


def _get_gradient_method_usage_message(method, missing_param):
    """Generate consistent error messages for gradient methods."""
    return (
        f"Method '{method}' requires '{missing_param}' parameter for gradient computation. "
        f"Usage: get_pruning_mask(layer, sparsity, method='{method}', model=your_model, dataset=(x, y), loss_fn='mse')"
    )


def _validate_gradient_method_requirements(method, model, dataset, loss_fn):
    """Validate that gradient-based methods have required parameters."""
    gradient_methods = ["saliency", "taylor"]
    method_name = method if isinstance(method, str) else method.__class__.__name__.lower()
    
    if any(gm in method_name for gm in gradient_methods):
        if model is None:
            raise ValueError(_get_gradient_method_usage_message(method, 'model'))
        if dataset is None:
            raise ValueError(_get_gradient_method_usage_message(method, 'dataset'))
        if loss_fn is None and not hasattr(model, 'compiled_loss') and not hasattr(model, 'loss'):
            raise ValueError(
                f"Method '{method}' requires 'loss_fn' parameter when model is not compiled or has no default loss. "
                f"Usage: get_pruning_mask(layer, sparsity, method='{method}', model=your_model, dataset=(x, y), loss_fn='mse')"
            )


def get_pruning_mask(layer, sparsity, method="l1", model=None, dataset=None, loss_fn=None, **kwargs):
    """Compute and return a pruning mask for a layer without applying it.
    
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
    pruning_method = _create_pruning_method(method)
    
    # Prepare kwargs for compute_mask with enhanced error handling
    _validate_gradient_method_requirements(method, model, dataset, loss_fn)
    
    mask_kwargs = {
        "model": model,
        "dataset": dataset, 
        "loss_fn": loss_fn,
        **kwargs
    }
    
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
    return ops.logical_not(pruning_mask)


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


def _build_pruning_stats(initial_sparsity, final_sparsity, pruned_layers, target_sparsity, 
                        method, pruned_layer_names, layers_to_prune=None, 
                        matched_layers=None, skipped_layer_names=None):
    """Build pruning statistics dictionary."""
    base_stats = {
        "initial_sparsity": initial_sparsity,
        "final_sparsity": final_sparsity,
        "pruned_layers": pruned_layers,
        "target_sparsity": target_sparsity,
        "method": method,
        "layers_pruned": pruned_layer_names,
    }
    
    if layers_to_prune is not None:
        base_stats.update({
            "layers_specified": layers_to_prune,
            "layers_matched": matched_layers or [],
            "layers_skipped": skipped_layer_names or [],
        })
    
    return base_stats


def apply_pruning_to_model(
    model,
    sparsity,
    method="l1",
    layers_to_prune=None,
    dataset=None,
    loss_fn=None,
    reinitialize=False,
    **kwargs,
):
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

    # Use the same layer traversal pattern as model.quantize()
    try:
        all_layers = model._flatten_layers()
    except AttributeError:
        # Fallback for models without _flatten_layers method
        all_layers = []

        def collect_layers(layer):
            all_layers.append(layer)
            if hasattr(layer, "_layers"):
                for sublayer in layer._layers:
                    collect_layers(sublayer)

        if hasattr(model, "layers"):
            for layer in model.layers:
                collect_layers(layer)
        else:
            all_layers = [model]

    for layer in all_layers:
        # Check if this is a leaf layer (like quantization does)
        try:
            list_of_sublayers = (
                list(layer._flatten_layers())
                if hasattr(layer, "_flatten_layers")
                else [layer]
            )
        except:
            list_of_sublayers = [layer]

        # Only process leaf layers to avoid double-processing
        if len(list_of_sublayers) == 1:
            if should_prune_layer(layer, layers_to_prune):
                if apply_pruning_to_layer(
                    layer=layer,
                    sparsity=sparsity,
                    method=method,
                    model=model,
                    dataset=dataset,
                    loss_fn=loss_fn,
                    reinitialize=reinitialize,
                    **kwargs,
                ):
                    pruned_layers += 1
                    pruned_layer_names.append(layer.name)
            elif _has_kernel_weights(layer):
                # Layer has weights but was skipped due to selection criteria
                skipped_layer_names.append(layer.name)

    final_sparsity = get_model_sparsity(model)

    # Build and return statistics
    matched_layers = None
    if layers_to_prune is not None:
        matched_layers = match_layers_by_patterns(model, layers_to_prune)

    return _build_pruning_stats(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        pruned_layers=pruned_layers,
        target_sparsity=sparsity,
        method=method,
        pruned_layer_names=pruned_layer_names,
        layers_to_prune=layers_to_prune,
        matched_layers=matched_layers,
        skipped_layer_names=skipped_layer_names,
    )
