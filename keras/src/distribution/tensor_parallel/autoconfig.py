from keras.src import layers
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import Split


def analyze_dense_layer(layer):
    """Analyzes a Keras Dense layer to classify its sharding strategy.

    This function inspects the input and output dimensions of a Dense layer
    to determine if it functions as an expansion layer ("up-projection"), a
    contraction layer ("down-projection"), or neither ("dense"). This
    classification is a heuristic commonly used to apply tensor parallelism
    in Transformer-based models, such as in an MLP block where an up-projection
    is followed by a down-projection.

    The classification is based on an `expansion_threshold` (set to 1.5).

    Args:
        layer: The Keras `layers.Dense` instance to analyze.

    Returns:
        str: A string classifying the layer as 'up_projection',
        'down_projection', or 'dense'.
    """

    if not isinstance(layer, layers.Dense):
        return "dense"

    input_dim = None
    output_dim = None

    if hasattr(layer, "kernel") and layer.kernel is not None:
        kernel_shape = layer.kernel.shape
        if len(kernel_shape) == 2:
            input_dim = kernel_shape[0]
            output_dim = kernel_shape[1]

    if input_dim is None or output_dim is None:
        if hasattr(layer, "units"):
            output_dim = layer.units
        else:
            return "dense"

        if (
            hasattr(layer, "input_shape")
            and layer.input_shape
            and len(layer.input_shape) > 1
        ):
            input_dim = layer.input_shape[-1]
        else:
            return "dense"

    if not input_dim or not output_dim:
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


def _recursive_layer_traversal(
    current_layer,
    prefix,
    device_count,
    state_rules,
    output_rules,
    processed_layers,
):
    """Recursively traverses the model graph to apply sharding rules.

    This function is necessary because Keras Model.layers property does not
    recursively find all sub-layers in all architectures. It applies sharding
    rules based on layer type and heuristic classification (e.g., up/down
    projection).

    - Split Logic:
        - 'up_projection' (or general 'dense'): Column-wise sharding
          (`Split(..., 1, "column")`) on kernel. Requires output to be
          gathered (`gather`).
        - 'down_projection' (or attention output): Row-wise sharding
          (`Split(..., 0, "row")`) on kernel. Requires output to be
          reduced (`allreduce`).
        - Embedding: Column-wise sharding (`Split(..., 1, "column")`).

    Args:
        current_layer: The Keras layer instance currently being inspected.
        prefix: The fully qualified name prefix for the current layer's scope.
        device_count: The number of devices (replicas) in the parallelism group.
        state_rules: A dictionary to accumulate variable sharding rules
          (`LayoutMap.state_rules`).
        output_rules: A dictionary to accumulate layer output communication
          rules (`LayoutMap.output_rules`).
        processed_layers: A set of layer IDs to prevent infinite recursion
          in graph structures.
    """
    if id(current_layer) in processed_layers:
        return
    processed_layers.add(id(current_layer))

    name = current_layer.name
    full_name = f"{prefix}.{name}" if prefix else name

    if isinstance(current_layer, layers.Dense):
        mlp_type = analyze_dense_layer(current_layer)

        if mlp_type == "up_projection":
            # Column-wise sharding for the first MLP layer
            state_rules[f"{full_name}.kernel"] = Split(
                device_count, 1, "column"
            )
            if current_layer.use_bias:
                state_rules[f"{full_name}.bias"] = Split(
                    device_count, 0, "column"
                )
            # The result needs to be gathered back to a full tensor.
            output_rules[f"{full_name}"] = {0: "gather"}

        elif mlp_type == "down_projection":
            # Row-wise sharding for the second MLP layer (down-projection)
            state_rules[f"{full_name}.kernel"] = Split(device_count, 0, "row")
            # Results from different devices needs to be summed (all-reduced).
            output_rules[f"{full_name}"] = {0: "allreduce"}

        else:
            # Fallback for generic dense layers (treat as column-wise split)
            state_rules[f"{full_name}.kernel"] = Split(
                device_count, 1, "column"
            )
            if current_layer.use_bias:
                state_rules[f"{full_name}.bias"] = Split(
                    device_count, 0, "column"
                )
            output_rules[f"{full_name}"] = {0: "gather -1"}

    elif isinstance(current_layer, layers.EinsumDense):
        if "attention_output" in full_name:
            # Row-wise sharding for the attention output layer
            state_rules[f"{full_name}.kernel"] = Split(device_count, 0, "row")
            output_rules[f"{full_name}"] = {0: "allreduce"}
        else:
            # Column-wise sharding for key/query/value projections
            state_rules[f"{full_name}.kernel"] = Split(
                device_count, 1, "column"
            )
            if (
                hasattr(current_layer, "bias")
                and current_layer.bias is not None
            ):
                state_rules[f"{full_name}.bias"] = Split(
                    device_count, 0, "column"
                )
            output_rules[f"{full_name}"] = {0: "gather -1"}

    elif isinstance(current_layer, (layers.Embedding,)):
        weight_name = None

        if hasattr(current_layer, "embeddings"):
            weight_name = "embeddings"
        elif hasattr(current_layer, "position_embeddings"):
            weight_name = "position_embeddings"

        if weight_name:
            # Column-wise sharding on the embedding dimension
            state_rules[f"{full_name}.{weight_name}"] = Split(
                device_count, 1, "column"
            )
            # Output requires no communication
            output_rules[f"{full_name}"] = {0: "no_comm"}

    elif isinstance(
        current_layer,
        (
            layers.LayerNormalization,
            layers.BatchNormalization,
            layers.GroupNormalization,
        ),
    ):
        pass

    if hasattr(current_layer, "layers") and current_layer.layers:
        for sub_layer in current_layer.layers:
            _recursive_layer_traversal(
                sub_layer,
                full_name,
                device_count,
                state_rules,
                output_rules,
                processed_layers,
            )

    for attr_name in dir(current_layer):
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue
        if hasattr(current_layer, attr_name):
            attr = getattr(current_layer, attr_name)

            if isinstance(attr, layers.Layer) and attr is not current_layer:
                _recursive_layer_traversal(
                    attr,
                    full_name,
                    device_count,
                    state_rules,
                    output_rules,
                    processed_layers,
                )
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, layers.Layer):
                        _recursive_layer_traversal(
                            item,
                            full_name,
                            device_count,
                            state_rules,
                            output_rules,
                            processed_layers,
                        )


def get_default_config_keras(module, device_ids):
    """Generates a default tensor parallelism sharding configuration.

    This function leverages model-traversal and heuristic layer analysis to
    automatically generate sharding rules (for state and layer outputs)
    optimized for large-scale language models (Transformers).

    Args:
        module: The root Keras `Model` or `Layer` instance representing the
                module to be sharded.
        device_ids: A list of device identifiers (e.g., strings) that define
                    the parallelism group. The length of this list determines
                    the number of slices (`device_count`).

    Returns:
        LayoutMap: An object containing the generated `state_rules` (variable
                   sharding) and `output_rules` (layer communication).
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
        processed_layers=processed_layers,
    )

    return LayoutMap(state_rules=state_rules, output_rules=output_rules)
