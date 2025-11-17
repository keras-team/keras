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


def _apply_layer_sharding_rules(layer, full_name, device_count, state_rules, output_rules):
    """
    Helper function that applies rules to a single layer instance.
    """
    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)

        if mlp_type == 'up_projection':
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if layer.use_bias:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather"}

        elif mlp_type == 'down_projection':
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "allreduce"}

        else:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if layer.use_bias:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather -1"}

    elif isinstance(layer, layers.EinsumDense):
        if "attention_output" in full_name:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "allreduce"}
        else:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if hasattr(layer, 'bias') and layer.bias is not None:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather -1"}

    elif isinstance(layer, (layers.Embedding,)) or "Embedding" in layer.__class__.__name__:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                if "embedding" in weight.name or "weight" in weight.name:
                    key_found = False
                    for attr_candidate in ['embeddings', 'position_embeddings', 'weight']:
                        if getattr(layer, attr_candidate, None) is weight:
                            state_rules[f"{full_name}.{attr_candidate}"] = _split_rule(device_count, dim=1)
                            key_found = True
                            break
                    
                    if not key_found:
                        clean_name = weight.name.split('/')[-1].split(':')[0]
                        state_rules[f"{full_name}.{clean_name}"] = _split_rule(device_count, dim=1)

            output_rules[f"{full_name}"] = {0: "no_comm"}


def get_default_config(module, device_ids):
    """
    Generates a default tensor parallelism configuration for a model using
    iterative graph traversal (stack-based).
    """
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}
    
    processed_layers = set()
    
    stack = [(module, "")]

    while stack:
        current_layer, prefix = stack.pop()

        if id(current_layer) in processed_layers:
            continue
        processed_layers.add(id(current_layer))

        name = current_layer.name
        full_name = f"{prefix}.{name}" if prefix else name

        _apply_layer_sharding_rules(
            current_layer, full_name, device_count, state_rules, output_rules
        )

        children_to_add = []

        if hasattr(current_layer, 'layers') and current_layer.layers:
            for sub_layer in current_layer.layers:
                children_to_add.append((sub_layer, full_name))

        for specific_attr in ['token_embedding', 'embeddings', 'position_embedding']:
            if hasattr(current_layer, specific_attr):
                attr_val = getattr(current_layer, specific_attr)
                if isinstance(attr_val, layers.Layer):
                    children_to_add.append((attr_val, full_name))

        for attr_name in dir(current_layer):
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
            
            if attr_name in ['trainable_variables', 'non_trainable_variables', 'weights']:
                continue

            attr_value = getattr(current_layer, attr_name, None)

            if attr_value is None:
                continue

            if isinstance(attr_value, layers.Layer) and attr_value is not current_layer:
                children_to_add.append((attr_value, full_name))
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, layers.Layer):
                        children_to_add.append((item, full_name))
        
        stack.extend(reversed(children_to_add))

    return LayoutMap(
        state_rules=state_rules,
        output_rules=output_rules
    )