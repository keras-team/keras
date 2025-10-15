from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import Split


def analyze_dense_layer_directly(layer, module, prefix):
    """Analyzes a Keras Dense layer to classify its sharding strategy.

    This function inspects the input and output dimensions of a Dense layer
    to determine if it functions as an expansion layer ("up-projection"), a
    contraction layer ("down-projection"), or neither ("generic_dense"). This
    classification is a heuristic commonly used to apply tensor parallelism
    in Transformer-based models, such as in an MLP block where an up-projection
    is followed by a down-projection.

    Args:
        layer: The Keras `layers.Dense` instance to analyze.
        module: The parent module containing the layer (currently unused).
        prefix (str): The name prefix for the layer in the model hierarchy
            (currently unused).

    Returns:
        str: A string classifying the layer as 'up_projection',
        'down_projection', or 'generic_dense'.
    """
    from keras.src import layers

    if not isinstance(layer, layers.Dense):
        return 'generic_dense'

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
            return 'generic_dense'

        if hasattr(layer, 'input_shape') and layer.input_shape and len(layer.input_shape) > 1:
            input_dim = layer.input_shape[-1]
        else:
            return 'generic_dense'

    if not input_dim or not output_dim:
        return 'generic_dense'

    expansion_threshold = 1.5
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold

    if is_expansion:
        return 'up_projection'
    elif is_contraction:
        return 'down_projection'
    else:
        return 'generic_dense'


def _find_and_shard_layers(
    current_layer,
    prefix,
    module,
    world_size,
    state_rules,
    output_rules,
    processed_layers,
):
    """Recursively traverses the model graph to apply sharding rules.

    This function walks through all nested layers of a given Keras model or
    layer. For each encountered layer, it determines an appropriate tensor
    parallelism strategy and populates the `state_rules` and `output_rules`
    dictionaries with the corresponding sharding actions. It uses a set of
    processed layer IDs to avoid redundant processing of shared layers.

    The sharding logic is as follows:
    - `Dense` layers are sharded based on their classification (up/down proj).
      - Up-projections are split along the column axis (output features).
      - Down-projections are split along the row axis (input features).
    - `EinsumDense` layers in attention blocks are sharded similarly.
    - `Embedding` layers are sharded column-wise for vocabulary parallelism.
    - Normalization layers are ignored (replicated on all devices).

    Args:
        current_layer: The Keras layer currently being processed.
        prefix (str): The hierarchical name prefix for the `current_layer`.
        module: The top-level Keras model or layer being configured.
        world_size (int): The total number of devices for sharding.
        state_rules (Dict[str, Any]): A dictionary to be populated with rules for
            sharding layer weights (state). Keys are regex patterns matching
            weight names, values are `SplitKeras` actions.
        output_rules (Dict[str, Any]): A dictionary to be populated with rules
            for handling layer outputs. Keys are regex patterns matching layer
            names, values describe the communication op (e.g., 'allreduce').
        processed_layers (Set[int]): A set of `id()`s of layers that have
            already been processed to prevent cycles and redundant work.
    """
    from keras.src import layers

    if id(current_layer) in processed_layers:
        return
    processed_layers.add(id(current_layer))

    name = current_layer.name
    full_name = f"{prefix}.{name}" if prefix else name

    if isinstance(current_layer, layers.Dense):
        mlp_type = analyze_dense_layer_directly(current_layer, module, full_name)

        if mlp_type == 'up_projection':
            state_rules[f"^{full_name}.kernel$"] = Split(world_size, 1, "column")
            if current_layer.use_bias:
                state_rules[f"^{full_name}.bias$"] = Split(world_size, 0, "column")
            output_rules[f"^{full_name}$"] = {0: "gather"}

        elif mlp_type == 'down_projection':
            state_rules[f"^{full_name}.kernel$"] = Split(world_size, 0, "row")
            output_rules[f"^{full_name}$"] = {0: "allreduce"}

        else:
            state_rules[f"^{full_name}.kernel$"] = Split(world_size, 1, "column")
            if current_layer.use_bias:
                state_rules[f"^{full_name}.bias$"] = Split(world_size, 0, "column")
            output_rules[f"^{full_name}$"] = {0: "gather -1"}
        return

    elif isinstance(current_layer, layers.EinsumDense):
        if "attention_output" in full_name:
            state_rules[f"^{full_name}.kernel$"] = Split(world_size, 0, "row")
            if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                pass
            output_rules[f"^{full_name}$"] = {0: "allreduce"}
        else:
            state_rules[f"^{full_name}.kernel$"] = Split(world_size, 1, "column")
            if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                state_rules[f"^{full_name}.bias$"] = Split(world_size, 0, "column")
            output_rules[f"^{full_name}$"] = {0: "gather -1"}
        return

    elif isinstance(current_layer, (layers.Embedding,)):
        if hasattr(current_layer, 'token_embedding') or hasattr(current_layer, 'position_embedding'):
            pass
        else:
            weight_name = None
            if hasattr(current_layer, 'embeddings'):
                weight_name = 'embeddings'
            elif hasattr(current_layer, 'position_embeddings'):
                weight_name = 'position_embeddings'

            if weight_name:
                state_rules[f"^{full_name}\\..*{weight_name}$"] = Split(world_size, 1, "column")
                output_rules[f"^{full_name}$"] = {0: "no_comm"}
            return

    elif isinstance(current_layer, (layers.LayerNormalization, layers.BatchNormalization, layers.GroupNormalization)):
        return

    if hasattr(current_layer, 'layers') and current_layer.layers:
        for sub_layer in current_layer.layers:
            _find_and_shard_layers(
                sub_layer, full_name, module, world_size,
                state_rules, output_rules, processed_layers
            )

    for attr_name in dir(current_layer):
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue
        if hasattr(current_layer, attr_name):
            attr = getattr(current_layer, attr_name)

            if isinstance(attr, layers.Layer) and attr is not current_layer:
                _find_and_shard_layers(
                    attr, full_name, module, world_size,
                    state_rules, output_rules, processed_layers
                )
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, layers.Layer):
                        _find_and_shard_layers(
                            item, full_name, module, world_size,
                            state_rules, output_rules, processed_layers
                        )

def get_default_config_keras(module, device_ids):
    """Generates a default tensor parallelism sharding configuration for a model.

    This function serves as the entry point for automatically creating a sharding
    plan for a given Keras model or layer. It initializes the rule dictionaries
    and starts the recursive layer traversal to populate them based on a default
    set of heuristics for common architectures like Transformers.

    Example:
        ```python
        model = MyTransformerModel()
        device_ids = ["gpu:0", "gpu:1"]
        sharding_config = get_default_config_keras(model, device_ids)
        # sharding_config can now be used to distribute the model
        ```

    Args:
        module: The Keras `Model` or `Layer` to generate a config for.
        device_ids (Sequence[str]): A sequence of device IDs (e.g.,
            ["gpu:0", "gpu:1"]) across which the model will be sharded.

    Returns:
        ConfigKeras: A configuration object containing the generated sharding
        rules for model weights (`state_rules`) and layer outputs
        (`output_rules`).
    """
    world_size = len(device_ids)
    state_rules = {}
    output_rules = {}
    processed_layers = set()

    _find_and_shard_layers(
        current_layer=module,
        prefix="",
        module=module,
        world_size=world_size,
        state_rules=state_rules,
        output_rules=output_rules,
        processed_layers=processed_layers
    )

    return LayoutMap(
        state_rules=state_rules,
        output_rules=output_rules
    )