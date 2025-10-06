from typing import Any
from typing import Dict
from typing import Sequence
from typing import Set

from keras.src.distribution.tensor_parallel.config import ConfigKeras
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras


def analyze_dense_layer_directly(layer, module, prefix: str) -> str:
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
    from keras.src import layers

    if not isinstance(layer, layers.Dense):
        return "generic_dense"

    input_dim = None
    output_dim = None

    if hasattr(layer, "kernel") and layer.kernel is not None:
        kernel_shape = layer.kernel.shape
        if len(kernel_shape) == 2:
            input_dim, output_dim = kernel_shape

    if input_dim is None or output_dim is None:
        if hasattr(layer, "units"):
            output_dim = layer.units
        else:
            return "generic_dense"

        if (
            hasattr(layer, "input_shape")
            and layer.input_shape
            and len(layer.input_shape) > 1
        ):
            input_dim = layer.input_shape[-1]
        else:
            return "generic_dense"

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


def _find_and_shard_layers(
    current_layer,
    prefix: str,
    module,
    world_size: int,
    state_rules: Dict[str, Any],
    output_rules: Dict[str, Any],
    processed_layers: Set[int],
):
    """Recursively traverses a Keras model to generate sharding rules.

    This is an internal helper function that navigates through all layers of a
    model, including nested ones. For each supported layer, it determines the
    appropriate sharding strategy and populates the `state_rules` and
    `output_rules` dictionaries. These dictionaries are modified in place.

    Args:
        current_layer: The Keras layer to be processed in the current step.
        prefix: The hierarchical name prefix for the `current_layer`.
        module: The top-level Keras model being analyzed.
        world_size: The total number of devices to shard the model across.
        state_rules: A dictionary with sharding rules for weights.
        output_rules: A dictionary with communication rules for outputs.
        processed_layers: A set of layer IDs to prevent infinite loops.
    """
    from keras.src import layers

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
            output_rules[f"^{full_name}$"] = {0: "gather"}

        elif mlp_type == "down_projection":
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                world_size, 0, "row"
            )
            if current_layer.use_bias:
                state_rules[f"^{full_name}.bias$"] = SplitKeras(
                    world_size, -1, "replicated"
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
            output_rules[f"^{full_name}$"] = {0: "gather -1"}
        return

    elif isinstance(current_layer, layers.EinsumDense):
        if "attention_output" in full_name or "out_proj" in full_name:
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                world_size, 0, "row"
            )
            if (
                hasattr(current_layer, "bias")
                and current_layer.bias is not None
            ):
                state_rules[f"^{full_name}.bias$"] = SplitKeras(
                    world_size, -1, "replicated"
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
                    world_size, 0, "column"
                )
            output_rules[f"^{full_name}$"] = {0: "gather -1"}
        return

    elif isinstance(current_layer, layers.Embedding):
        state_rules[f"^{full_name}.embeddings$"] = SplitKeras(
            world_size, 0, "vocab_parallel"
        )
        output_rules[f"^{full_name}$"] = {0: "allreduce"}
        return

    elif isinstance(current_layer, layers.MultiHeadAttention):
        for proj in ["query", "key", "value"]:
            proj_dense_name = f"_{proj}_dense"
            if hasattr(current_layer, proj_dense_name):
                state_rules[f"^{full_name}\.{proj_dense_name}\.kernel$"] = (
                    SplitKeras(world_size, 1, "column")
                )
                if getattr(current_layer, proj_dense_name).use_bias:
                    state_rules[f"^{full_name}\.{proj_dense_name}\.bias$"] = (
                        SplitKeras(world_size, 0, "column")
                    )

        output_dense_name = "_output_dense"
        if hasattr(current_layer, output_dense_name):
            state_rules[f"^{full_name}\.{output_dense_name}\.kernel$"] = (
                SplitKeras(world_size, 0, "row")
            )
            if getattr(current_layer, output_dense_name).use_bias:
                state_rules[f"^{full_name}\.{output_dense_name}\.bias$"] = (
                    SplitKeras(world_size, -1, "replicated")
                )

        output_rules[f"^{full_name}$"] = {0: "allreduce"}
        return

    elif isinstance(current_layer, layers.Dropout):
        if "rng_rules" not in state_rules:
            state_rules["rng_rules"] = {}
        state_rules["rng_rules"][full_name] = {"type": "parallel"}
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
            _find_and_shard_layers(
                sub_layer,
                full_name,
                module,
                world_size,
                state_rules,
                output_rules,
                processed_layers,
            )

    for attr_name in dir(current_layer):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(current_layer, attr_name)
            if isinstance(attr, layers.Layer) and attr is not current_layer:
                _find_and_shard_layers(
                    attr,
                    full_name,
                    module,
                    world_size,
                    state_rules,
                    output_rules,
                    processed_layers,
                )
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, layers.Layer):
                        _find_and_shard_layers(
                            item,
                            full_name,
                            module,
                            world_size,
                            state_rules,
                            output_rules,
                            processed_layers,
                        )
        except Exception:
            continue


def get_default_config_keras(module, device_ids: Sequence[str]) -> ConfigKeras:
    """Generates a default sharding configuration for a Keras model.

    This function serves as the main entry point for automatically creating a
    tensor parallel sharding configuration. It traverses the model and applies
    standard sharding patterns for common layer types like Dense, Embedding, and
    MultiHeadAttention.

    Args:
        module: The Keras model or layer to be configured for sharding.
        device_ids: A sequence of device IDs (e.g., `['gpu:0', 'gpu:1']`)
            to shard across. The number of devices determines the `world_size`.

    Returns:
        A `ConfigKeras` object containing the generated `state_rules` for
        sharding weights and `output_rules` for handling communications.
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
        processed_layers=processed_layers,
    )

    return ConfigKeras(state_rules=state_rules, output_rules=output_rules)
