import functools

from keras.src import layers
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
)


def analyze_dense_layer(layer):
    """
    Classifies a Dense layer based on its input/output dimensions.
    """
    input_dim = None
    output_dim = None

    kernel = getattr(layer, "kernel", getattr(layer, "_kernel", None))
    if kernel is not None:
        if len(kernel.shape) == 2:
            input_dim = kernel.shape[0]
            output_dim = kernel.shape[1]

    if output_dim is None and hasattr(layer, "units"):
        output_dim = layer.units

    if (
        input_dim is None
        and hasattr(layer, "input_shape")
        and layer.input_shape
        and len(layer.input_shape) > 1
    ):
        input_dim = layer.input_shape[-1]

    if input_dim is None or output_dim is None:
        return "dense"

    expansion_threshold = 1.5
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold

    if is_expansion:
        return "up_projection"
    elif is_contraction:
        return "down_projection"
    else:
        return "dense"


def _reduce_sum(x):
    return distribution_lib.all_reduce(x, op="sum", axis_name="model")


def _gather(x, axis):
    return distribution_lib.all_gather(x, axis=axis, axis_name="model")


def _get_layer_path(layer):
    """
    Returns the unique hierarchical path of the layer.
    Ex: 'model/dense_1'
    """
    return getattr(layer, "path", layer.name)


def _apply_layer_sharding_rules(layer, device_count, state_rules, output_rules):
    """
    Helper function that applies rules to a single layer instance.
    """

    def split_rule(dim):
        return functools.partial(
            split_tensor_for_parallelism, device_count=device_count, dim=dim
        )

    def gather_rule(axis):
        return functools.partial(_gather, axis=axis)

    layer_path = _get_layer_path(layer)

    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)

        if mlp_type == "up_projection":
            state_rules[layer.kernel.path] = split_rule(dim=1)
            if layer.use_bias:
                state_rules[layer.bias.path] = split_rule(dim=0)
            output_rules[layer_path] = {0: gather_rule(axis=-1)}

        elif mlp_type == "down_projection":
            state_rules[layer.kernel.path] = split_rule(dim=0)
            output_rules[layer_path] = {0: _reduce_sum}

        else:
            state_rules[layer.kernel.path] = split_rule(dim=1)
            if layer.use_bias:
                state_rules[layer.bias.path] = split_rule(dim=0)
            output_rules[layer_path] = {0: gather_rule(axis=-1)}

    elif isinstance(layer, layers.EinsumDense):
        if "attention_output" in layer.name:  # Use name check as heuristic
            state_rules[layer.kernel.path] = split_rule(dim=0)
            output_rules[layer_path] = {0: _reduce_sum}
        else:
            state_rules[layer.kernel.path] = split_rule(dim=1)
            if hasattr(layer, "bias") and layer.bias is not None:
                state_rules[layer.bias.path] = split_rule(dim=0)
            output_rules[layer_path] = {0: gather_rule(axis=-1)}

    elif (
        isinstance(layer, (layers.Embedding,))
        or "Embedding" in layer.__class__.__name__
    ):
        if hasattr(layer, "weights"):
            found_embedding = False
            for weight in layer.weights:
                if "embedding" in weight.name or "weight" in weight.name:
                    state_rules[weight.path] = split_rule(dim=1)
                    found_embedding = True

            if found_embedding:
                output_rules[layer_path] = {0: lambda x: x}


def get_default_config(model, device_ids):
    """
    Generates a default tensor parallelism configuration for a model.
    """
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}

    for layer in model._flatten_layers(recursive=True, include_self=True):
        _apply_layer_sharding_rules(
            layer, device_count, state_rules, output_rules
        )

    return LayoutMap(state_rules=state_rules, output_rules=output_rules)
