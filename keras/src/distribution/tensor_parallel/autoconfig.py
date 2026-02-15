import functools

from keras.src import layers
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
)


def analyze_dense_layer(layer):
    """Classifies a Dense layer based on its input/output dimensions.

    This function uses a heuristic to determine if a Dense layer acts as an
    'up_projection' (expansion), a 'down_projection' (contraction), or a
    standard 'dense' layer. This classification is used to determine the
    appropriate sharding strategy (e.g., column-parallel vs row-parallel).

    Args:
        layer: The Keras Dense layer instance to analyze.

    Returns:
        str: One of 'up_projection', 'down_projection', or 'dense'.
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
    """Performs an all-reduce sum operation across the 'model' mesh axis.

    Args:
        x: The input tensor to reduce.

    Returns:
        The reduced tensor, summed across all devices in the model axis.
    """
    return distribution_lib.all_reduce(x, op="sum", axis_name="model")


def _gather(x, axis):
    """Performs an all-gather operation across the 'model' mesh axis.

    Args:
        x: The input tensor shard to gather.
        axis: The axis along which to concatenate the gathered parts.

    Returns:
        The gathered tensor, concatenated along the specified axis.
    """
    return distribution_lib.all_gather(x, axis=axis, axis_name="model")


def _apply_layer_sharding_rules(layer, device_count, state_rules, output_rules):
    """Applies sharding rules to a single layer based on its type.

    This function populates `state_rules` and `output_rules` with strategies
    specific to the layer class (e.g., Dense, EinsumDense, Embedding). It
    determines how weights should be partitioned (state rules) and how outputs
    should be synchronized (output rules).

    Args:
        layer: The Keras layer instance to configure.
        device_count: The number of devices available for tensor parallelism.
        state_rules: A dictionary mapping variable paths to sharding functions.
            Updated in-place.
        output_rules: A dictionary mapping layer paths to output communication
            functions. Updated in-place.
    """

    def split_rule(dim):
        return functools.partial(
            split_tensor_for_parallelism, device_count=device_count, dim=dim
        )

    def gather_rule(axis):
        return functools.partial(_gather, axis=axis)

    layer_path = layer.path

    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)

        if mlp_type == "up_projection":
            state_rules[id(layer.kernel)] = split_rule(dim=1)
            if layer.use_bias:
                state_rules[id(layer.bias)] = split_rule(dim=0)
            output_rules[layer_path] = gather_rule(axis=-1)

        elif mlp_type == "down_projection":
            state_rules[id(layer.kernel)] = split_rule(dim=0)
            output_rules[layer_path] = _reduce_sum

        else:
            state_rules[id(layer.kernel)] = split_rule(dim=1)
            if layer.use_bias:
                state_rules[id(layer.bias)] = split_rule(dim=0)
            output_rules[layer_path] = gather_rule(axis=-1)

    elif isinstance(layer, layers.EinsumDense):
        if "attention_output" in layer.name:
            state_rules[id(layer.kernel)] = split_rule(dim=0)
            output_rules[layer_path] = _reduce_sum
        else:
            state_rules[id(layer.kernel)] = split_rule(dim=1)
            if hasattr(layer, "bias") and layer.bias is not None:
                state_rules[id(layer.bias)] = split_rule(dim=0)
            output_rules[layer_path] = gather_rule(axis=-1)

    elif (
        isinstance(layer, (layers.Embedding,))
        or "Embedding" in layer.__class__.__name__
    ):
        embeddings_var = getattr(layer, "embeddings", None)
        if embeddings_var is not None:
            state_rules[id(embeddings_var)] = split_rule(dim=1)
        output_rules[layer_path] = lambda x: x


def get_default_config(model, device_ids):
    """Generates a default tensor parallelism configuration for a model.

    This function traverses the model's layer hierarchy and
    automatically generates a `LayoutMap`. This map contains:
    1.  `state_rules`: How to shard the weights of supported layers
        across the specified devices.
    2.  `output_rules`: How to synchronize or gather the outputs of
        these layers during the forward pass.

    Args:
        model: The Keras model to configure.
        device_ids: A list of device identifiers to be used
            for distribution.

    Returns:
        LayoutMap: A configuration object containing `state_rules` and
        `output_rules` for tensor parallelism.
    """
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}

    for layer in model._flatten_layers(recursive=True, include_self=True):
        _apply_layer_sharding_rules(
            layer, device_count, state_rules, output_rules
        )

    return LayoutMap(state_rules=state_rules, output_rules=output_rules)
