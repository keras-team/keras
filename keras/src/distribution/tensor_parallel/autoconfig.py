from typing import Sequence

from keras.src import layers
from keras.src.distribution.tensor_parallel.config import ConfigKeras
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras
from keras.src.models import Model


def analyze_dense_layer_directly(
    layer: layers.Dense, module: Model, prefix: str
) -> str:
    """Analyzes a Dense layer to classify it for tensor parallelism sharding.

    This function inspects the layer's weight shapes to determine if it's an
    "up-projection" (expanding feature dimensions), a "down-projection"
    (contracting feature dimensions), or a generic layer. This classification
    helps in deciding whether to apply column-wise or row-wise parallelism.

    Args:
        layer: The keras.layers.Dense instance to analyze.
        module: The parent Keras model containing the layer.
        prefix: The hierarchical name prefix for the layer.

    Returns:
        A string indicating the layer's classification: 'up_projection',
        'down_projection', or 'generic_dense'.
    """
    if not isinstance(layer, layers.Dense):
        return "generic_dense"

    input_dim = None
    output_dim = None

    if hasattr(layer, "kernel"):
        kernel_shape = layer.kernel.shape
        if len(kernel_shape) == 2:
            input_dim = kernel_shape[0]
            output_dim = kernel_shape[1]
    else:
        if hasattr(layer, "units"):
            output_dim = layer.units

        if (
            hasattr(layer, "input_shape")
            and layer.input_shape
            and len(layer.input_shape) > 1
        ):
            input_dim = layer.input_shape[-1]

    if not input_dim or not output_dim:
        return "generic_dense"

    expansion_threshold = 1.5
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold

    if is_expansion:
        return "up_projection"
    elif is_contraction:
        return "down_projection"
    else:
        return "generic_dense"


def _traverse_and_shard_layer(
    current_layer: layers.Layer,
    module: Model,
    world_size: int,
    state_rules: dict,
    output_rules: dict,
    processed_layers: set,
    prefix: str = "",
):
    """Traverses a layer and its sub-layers to apply sharding rules.

    This function navigates through the model's layer hierarchy. For each
    layer, it identifies its type and applies appropriate sharding logic,
    populating the `state_rules` and `output_rules` dictionaries.

    Args:
        current_layer: The current keras.Layer object to be processed.
        module: The top-level Keras Model, used for context analysis.
        world_size: The total number of devices for sharding.
        state_rules: The dictionary of state sharding rules to populate.
        output_rules: The dictionary of output sharding rules to populate.
        processed_layers: A set of layer IDs that have already been processed
            to avoid redundant computation and infinite loops.
        prefix: The hierarchical name prefix from parent layers, used to
            construct the full unique name for the current layer.
    """
    if id(current_layer) in processed_layers:
        return
    processed_layers.add(id(current_layer))

    name = current_layer.name
    full_name = f"{prefix}.{name}" if prefix else name

    if isinstance(current_layer, layers.Dense):
        mlp_type = analyze_dense_layer_directly(
            current_layer, module, full_name
        )

        if mlp_type == "up_projection":
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                world_size, 1, "column"
            )
            if current_layer.use_bias:
                state_rules[f"^{full_name}.bias$"] = SplitKeras(
                    world_size, 0, "column"
                )
            output_rules[f"^{full_name}$"] = {0: "no_comm"}

        elif mlp_type == "down_projection":
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                world_size, 0, "row"
            )
            output_rules[f"^{full_name}$"] = {0: "allreduce"}

        else:
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                world_size, 1, "column"
            )
            if current_layer.use_bias:
                state_rules[f"^{full_name}.bias$"] = SplitKeras(
                    world_size, 0, "column"
                )
            output_rules[f"^{full_name}$"] = {0: "no_comm"}
        return

    elif isinstance(current_layer, layers.EinsumDense):
        is_row_parallel = False
        if "->" in current_layer.equation:
            equation_parts = current_layer.equation.split("->")
            if len(equation_parts) == 2:
                input_spec = equation_parts[0].split(",")[0].strip()
                output_spec = equation_parts[1].strip()
                if (
                    input_spec
                    and output_spec
                    and len(output_spec) < len(input_spec)
                ):
                    is_row_parallel = True

        if is_row_parallel:
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                world_size, 1, "row"
            )
            output_rules[f"^{full_name}$"] = {0: "allreduce"}
        else:
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                world_size, 1, "column"
            )
            if (
                hasattr(current_layer, "bias")
                and current_layer.bias is not None
            ):
                state_rules[f"^{full_name}.bias$"] = SplitKeras(
                    world_size, -1, "column"
                )
            output_rules[f"^{full_name}$"] = {0: "no_comm"}
        return

    elif isinstance(current_layer, layers.Embedding):
        weight_name = (
            "embeddings" if hasattr(current_layer, "embeddings") else None
        )
        if weight_name:
            state_rules[f"^{full_name}\.{weight_name}$"] = SplitKeras(
                world_size, 1, "column"
            )
            output_rules[f"^{full_name}$"] = {0: "no_comm"}
        return

    elif isinstance(
        current_layer,
        (
            layers.LayerNormalization,
            layers.BatchNormalization,
            layers.GroupNormalization,
        ),
    ):
        return

    if hasattr(current_layer, "layers") and current_layer.layers:
        for sub_layer in current_layer.layers:
            _traverse_and_shard_layer(
                sub_layer,
                module,
                world_size,
                state_rules,
                output_rules,
                processed_layers,
                full_name,
            )

    for attr_name in dir(current_layer):
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue

        if hasattr(current_layer, attr_name):
            attr = getattr(current_layer, attr_name)

            if isinstance(attr, layers.Layer) and attr is not current_layer:
                _traverse_and_shard_layer(
                    attr,
                    module,
                    world_size,
                    state_rules,
                    output_rules,
                    processed_layers,
                    full_name,
                )
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, layers.Layer):
                        _traverse_and_shard_layer(
                            item,
                            module,
                            world_size,
                            state_rules,
                            output_rules,
                            processed_layers,
                            full_name,
                        )


def get_default_config_keras(
    module: Model, device_ids: Sequence[str]
) -> ConfigKeras:
    """Generates a smart, recursive sharding configuration for a Keras model.

    This function traverses the layers of a given Keras model and applies a
    set of heuristics to automatically determine how each layer's weights
    and outputs should be sharded for tensor parallelism. It uses a helper
    function to perform the recursive traversal.

    Args:
        module: The Keras Model to generate a sharding configuration for.
        device_ids: A sequence of device identifiers, used to determine the
            world size (number of devices) for sharding.

    Returns:
        A ConfigKeras object containing the generated 'state_rules' (for model
        parameters) and 'output_rules' (for layer outputs).
    """
    world_size = len(device_ids)
    state_rules = {}
    output_rules = {}
    processed_layers = set()

    for layer in module.layers:
        _traverse_and_shard_layer(
            current_layer=layer,
            module=module,
            world_size=world_size,
            state_rules=state_rules,
            output_rules=output_rules,
            processed_layers=processed_layers,
            prefix="",
        )

    return ConfigKeras(state_rules=state_rules, output_rules=output_rules)
