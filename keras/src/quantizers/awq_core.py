"""AWQ core functionality for layer-wise quantization.

This module provides the orchestration logic for applying AWQ quantization
to transformer models in a layer-by-layer fashion.
"""

from contextlib import contextmanager

from absl import logging

from keras.src import ops
from keras.src import utils as keras_utils
from keras.src.dtype_policies.dtype_policy import AWQDTypePolicy
from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap
from keras.src.quantizers.awq import AWQ
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.gptq_core import find_layers_in_block
from keras.src.quantizers.gptq_core import get_dataloader
from keras.src.quantizers.utils import should_quantize_layer


@contextmanager
def stream_activations(layers_map, awq_objects):
    """Context manager to capture activations for AWQ calibration.

    Temporarily patches layer.call methods to capture activation statistics
    for computing per-channel scaling factors.

    Args:
        layers_map: Dict[str, Layer]. Mapping from layer names to layers.
        awq_objects: Dict[str, AWQ]. Mapping from names to AWQ instances.

    Yields:
        None: The patched state is active only within the `with` block.
    """
    original_calls = {}

    def create_hook(name, original_call_func):
        def hook(*args, **kwargs):
            inp = args[0] if args else kwargs["inputs"]
            num_features = awq_objects[name].rows
            input_2d = ops.reshape(inp, (-1, num_features))
            awq_objects[name].update_activation_magnitudes(input_2d)
            return original_call_func(*args, **kwargs)

        return hook

    try:
        for name, layer in layers_map.items():
            original_calls[name] = layer.call
            layer.call = create_hook(name, layer.call)
        yield
    finally:
        for name, layer in layers_map.items():
            layer.call = original_calls[name]


def apply_awq_layerwise(dataloader, config, structure, filters=None):
    """Apply AWQ quantization layer-by-layer to a Keras model.

    This function processes the model sequentially, one block at a time:
    1. Captures activation statistics through calibration data forward pass
    2. Uses activation magnitudes to determine weight saliency
    3. Finds optimal per-channel scales via grid search
    4. Quantizes weights with AWQ scaling

    Args:
        dataloader: Calibration data as numpy array.
        config: AWQConfig instance.
        structure: Dict with 'pre_block_layers' and 'sequential_blocks'.
        filters: Optional layer filters.
    """
    num_samples = config.num_samples
    logging.info("Starting AWQ quantization...")

    pre_layers = structure.get("pre_block_layers", [])
    transformer_blocks = structure.get("sequential_blocks", [])

    if not transformer_blocks:
        raise ValueError(
            "No sequential blocks found in the structure to quantize."
        )

    # Process inputs through pre-block layers (e.g., embedding)
    inputs = []
    for batch in dataloader:
        batch = ops.convert_to_tensor(batch, dtype="int32")
        for layer in pre_layers:
            batch = layer(batch)
        inputs.append(batch)

    num_samples = min(num_samples, len(inputs))
    progbar = keras_utils.Progbar(target=len(transformer_blocks))

    for block_idx, block in enumerate(transformer_blocks):
        logging.info(f"Quantizing Block {block_idx}")
        sub_layers_map = find_layers_in_block(block)

        # Apply filters
        final_sub_layers_map = {}
        for name, layer in sub_layers_map.items():
            if not should_quantize_layer(layer, filters):
                continue
            final_sub_layers_map[name] = layer

        sub_layers_map = final_sub_layers_map

        if not sub_layers_map:
            logging.info(
                f"  No quantizable layers found in block {block_idx}. Skipping."
            )
        else:
            logging.info(f"Found layers: {list(sub_layers_map.keys())}")

            # Create AWQ objects for each layer
            awq_objects = {
                name: AWQ(layer, config)
                for name, layer in sub_layers_map.items()
            }

            # Capture activation statistics
            with stream_activations(sub_layers_map, awq_objects):
                for sample_idx in range(num_samples):
                    current_input = inputs[sample_idx]
                    if len(current_input.shape) == 2:
                        current_input = ops.expand_dims(current_input, axis=0)
                    _ = block(current_input)

            # Quantize each layer
            for name, awq_object in awq_objects.items():
                logging.info(f"Quantizing {name}...")
                awq_object.quantize_layer()
                awq_object.free()

            del awq_objects

        # Generate inputs for next block
        if block_idx < len(transformer_blocks) - 1:
            logging.info(f"Generating inputs for block {block_idx + 1}...")
            next_block_inputs = []
            for sample_idx in range(num_samples):
                current_input = inputs[sample_idx]
                if len(current_input.shape) == 2:
                    current_input = ops.expand_dims(current_input, axis=0)
                output = block(current_input)[0]
                next_block_inputs.append(output)
            inputs = next_block_inputs

        progbar.update(current=block_idx + 1)

    logging.info("AWQ quantization complete.")


def awq_quantize(config, quantization_layer_structure, filters=None):
    """Main entry point for AWQ quantization.

    Args:
        config: AWQConfig instance.
        quantization_layer_structure: Model structure dictionary.
        filters: Optional layer filters.
    """
    if config.dataset is None or config.tokenizer is None:
        raise ValueError(
            "AWQ quantization requires a dataset and tokenizer. "
            "Please provide them in the AWQConfig."
        )

    if quantization_layer_structure is None:
        raise ValueError(
            "For 'awq' mode, a valid quantization structure must be provided "
            "either via `config.quantization_layer_structure` or by overriding "
            "`model.get_quantization_layer_structure(mode)`. The structure "
            "should be a dictionary with keys 'pre_block_layers' and "
            "'sequential_blocks'."
        )

    # Load calibration data
    dataloader = get_dataloader(
        config.tokenizer,
        config.sequence_length,
        config.dataset,
        num_samples=config.num_samples,
    )

    apply_awq_layerwise(
        dataloader[: config.num_samples],
        config,
        quantization_layer_structure,
        filters=filters,
    )


def get_group_size_for_layer(layer, config):
    """Get group size from config or dtype policy.

    Args:
        layer: The layer to get group size for.
        config: Optional AWQConfig instance.

    Returns:
        int: The group size for quantization.

    Raises:
        ValueError: If group size cannot be determined.
    """
    if config and isinstance(config, AWQConfig):
        return config.group_size
    elif isinstance(layer.dtype_policy, AWQDTypePolicy):
        return layer.dtype_policy.group_size
    elif isinstance(layer.dtype_policy, DTypePolicyMap):
        policy = layer.dtype_policy[layer.path]
        if isinstance(policy, AWQDTypePolicy):
            return policy.group_size
    raise ValueError(
        "For AWQ quantization, group_size must be specified "
        "through AWQConfig or AWQDTypePolicy."
    )
