from keras.src import layers
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
    LayoutMap
)

_split_fn_internal = split_tensor_for_parallelism


def _split_rule(device_count, dim):
    """
    Creates a sharding rule for a specific dimension.

    Returns a lambda function compatible with LayoutMap that defines
    how a tensor should be split across the available devices.

    Args:
        device_count (int): The total number of devices available for parallelism.
        dim (int): The dimension of the tensor to split.

    Returns:
        callable: A lambda function accepting (tensor, index) that returns the
        sharded layout.
    """
    return lambda x, index: _split_fn_internal(x, index, device_count, dim=dim)


def analyze_dense_layer(layer):
    """
    Classifies a Dense layer based on its input/output dimensions.

    This function determines if a Dense layer represents an 'up_projection'
    (expansion) or a 'down_projection' (contraction) based on a heuristic
    threshold. This classification dictates how the weights are sharded.

    Heuristic:
        - Expansion: Output dimension > (Input dimension * 1.5)
        - Contraction: Input dimension > (Output dimension * 1.5)

    Args:
        layer (keras.layers.Layer): The layer instance to analyze.

    Returns:
        str: One of 'up_projection', 'down_projection', or 'dense'.
    """
    if not isinstance(layer, layers.Dense):
        return 'dense'

    input_dim = None
    output_dim = None

    if hasattr(layer, 'kernel') and layer.kernel is not None:
        kernel_shape = layer.kernel.shape
        if len(kernel_shape) == 2:
            input_dim = kernel_shape[0]
            output_dim = kernel_shape[1]

    if input_dim is None or output_dim is None:
        if hasattr(layer, 'units'):
            output_dim = layer.units
        else:
            return 'dense'

        if hasattr(layer, 'input_shape') and layer.input_shape and len(layer.input_shape) > 1:
            input_dim = layer.input_shape[-1]
        else:
            return 'dense'

    if not input_dim or not output_dim:
        return 'dense'

    expansion_threshold = 1.5
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold

    if is_expansion:
        return 'up_projection'
    elif is_contraction:
        return 'down_projection'
    else:
        return 'dense'


def _recursive_layer_traversal(
    current_layer,
    prefix,
    device_count,
    state_rules,
    output_rules,
    processed_layers,
):
    """
    Traverses the model graph recursively to apply sharding rules.

    This function visits layers, checks their type, and populates the
    state_rules (weights) and output_rules (activations) dictionaries
    required for Tensor Parallelism.

    Args:
        current_layer (keras.layers.Layer): The current layer being visited.
        prefix (str): The naming prefix for the current layer (used for nested models).
        device_count (int): Total number of devices.
        state_rules (dict): The dictionary accumulating variable sharding rules.
        output_rules (dict): The dictionary accumulating output layout rules.
        processed_layers (set): A set of object IDs to prevent infinite recursion on cycles.
    """
    if id(current_layer) in processed_layers:
        return
    processed_layers.add(id(current_layer))

    name = current_layer.name
    full_name = f"{prefix}.{name}" if prefix else name

    if isinstance(current_layer, layers.Dense):
        mlp_type = analyze_dense_layer(current_layer)

        if mlp_type == 'up_projection':
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if current_layer.use_bias:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather"}

        elif mlp_type == 'down_projection':
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "allreduce"}

        else:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if current_layer.use_bias:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather -1"}

    elif isinstance(current_layer, layers.EinsumDense):
        if "attention_output" in full_name:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "allreduce"}
        else:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather -1"}

    elif isinstance(current_layer, (layers.Embedding,)):
        weight_name = None 

        if hasattr(current_layer, 'embeddings'):
            weight_name = 'embeddings'
        elif hasattr(current_layer, 'position_embeddings'):
            weight_name = 'position_embeddings'

        if weight_name:
            state_rules[f"{full_name}.{weight_name}"] = _split_rule(device_count, dim=1)
            output_rules[f"{full_name}"] = {0: "no_comm"}

    elif isinstance(current_layer, (layers.LayerNormalization, layers.BatchNormalization, layers.GroupNormalization)):
        pass

    if hasattr(current_layer, 'layers') and current_layer.layers:
        for sub_layer in current_layer.layers:
            _recursive_layer_traversal(
                sub_layer, full_name, device_count,
                state_rules, output_rules, processed_layers
            )

    for attr_name in dir(current_layer):
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue
        if hasattr(current_layer, attr_name):
            attr = getattr(current_layer, attr_name)

            if isinstance(attr, layers.Layer) and attr is not current_layer:
                _recursive_layer_traversal(
                    attr, full_name, device_count,
                    state_rules, output_rules, processed_layers
                )
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, layers.Layer):
                        _recursive_layer_traversal(
                            item, full_name, device_count,
                            state_rules, output_rules, processed_layers
                        )


def get_default_config_keras(module, device_ids):
    """
    Generates a default tensor parallelism configuration for a model.

    This function inspects the model structure and automatically generates
    a `LayoutMap` containing sharding rules for weights (kernels/biases) and
    outputs (activations).

    Args:
        module (keras.Model or keras.layers.Layer): The Keras model or layer to config.
        device_ids (list): A list of device identifiers (e.g., strings or Mesh IDs).

    Returns:
        keras.src.distribution.tensor_parallel.tensor_layout.LayoutMap:
        The configuration map applied to the model distribution API.
    """
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}
    
    processed_layers = set() 

    _recursive_layer_traversal(
        current_layer=module,
        prefix="",
        device_count=device_count, 
        state_rules=state_rules,
        output_rules=output_rules,
        processed_layers=processed_layers
    )

    return LayoutMap(
        state_rules=state_rules,
        output_rules=output_rules
    )