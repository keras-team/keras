from keras.src import layers
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
    LayoutMap
)

_split_fn_internal = split_tensor_for_parallelism


def _split_rule(device_count, dim):
    """
    Returns a sharding rule (lambda) that calls split_tensor_for_parallelism.
    The lambda accepts (tensor, index) as expected by LayoutMap.
    """
    return lambda x, index: _split_fn_internal(x, index, device_count, dim=dim)


def analyze_dense_layer(layer):
    """Analyzes a Keras Dense layer to classify its sharding strategy."""
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
    """Recursively traverses the model graph to apply sharding rules."""
    
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
    """Generates a default tensor parallelism sharding configuration for a model."""
    
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