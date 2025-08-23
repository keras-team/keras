import random

import numpy as np
from absl import logging

from keras.src import ops
from keras.src import utils as keras_utils
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.layers import Embedding
from keras.src.quantizers.gptq import GPTQ
from keras.src.quantizers.gptq_quant import GPTQQuantization


def get_dataloader(tokenizer, sequence_length, dataset, num_samples=128):
    """
    Prepares and chunks the calibration dataloader, repeating short datasets.
    """
    all_tokens = []

    if not hasattr(dataset, "__iter__") or isinstance(dataset, (str, bytes)):
        raise TypeError(
            "The `dataset` argument must be an iterable (e.g., a list of "
            "strings, a generator, or a NumPy array). Got type: "
            f"{type(dataset).__name__}. Please pass the loaded dataset "
            "directly."
        )

    dataset_list = list(dataset)

    if not dataset_list:
        raise ValueError("Provided dataset is empty.")

    if isinstance(dataset_list[0], str):
        logging.info("(Dataset contains strings, tokenizing now...)")
        full_text = "\n\n".join(dataset_list)
        all_tokens = tokenizer.tokenize(full_text)
    else:
        logging.info("(Dataset is pre-tokenized, concatenating...)")
        all_tokens = np.concatenate(
            [ops.convert_to_numpy(s).reshape(-1) for s in dataset_list], axis=0
        )

    all_tokens = np.array(all_tokens, dtype=np.int32)

    # Repeat data if it's too short
    required_tokens = num_samples * sequence_length
    if len(all_tokens) < required_tokens:
        logging.info(
            f"Warning: Dataset is too short ({len(all_tokens)} tokens)."
            " Repeating data to generate {num_samples} samples."
        )
        repeats = -(-required_tokens // len(all_tokens))  # Ceiling division
        all_tokens = np.tile(all_tokens, repeats)

    # Chunk the token list into samples

    calibration_samples = []
    for _ in range(num_samples):
        # Generate a random starting index
        start_index = random.randint(0, len(all_tokens) - sequence_length - 1)
        end_index = start_index + sequence_length
        sample = all_tokens[start_index:end_index]
        calibration_samples.append(np.reshape(sample, (1, sequence_length)))

    final_array = np.stack(calibration_samples, axis=0)
    return final_array


def _find_layers_recursive(layer, prefix, found_layers):
    """
    Recursively search for Dense and EinsumDense layers and record them.
    """
    for sub_layer in layer._layers:
        # Construct a unique name for the layer based on its hierarchy
        layer_name = f"{prefix}.{sub_layer.name}"
        if isinstance(sub_layer, (Dense, EinsumDense)):
            found_layers[layer_name] = sub_layer

        # Recurse into nested layers that are not the target types
        elif hasattr(sub_layer, "_layers") and sub_layer._layers:
            _find_layers_recursive(sub_layer, layer_name, found_layers)


def find_layers_in_block(block):
    """
    A pluggable, generic function to find all Dense and EinsumDense layers
    within any transformer block by using a recursive search.
    """
    found_layers = {}
    # Start the recursive search from the block itself
    _find_layers_recursive(block, "block", found_layers)
    return found_layers


def apply_gptq_layerwise(
    model,
    dataloader,
    num_samples,
    hessian_damping,
    group_size,
    symmetric,
    activation_order,
    weight_bits,
):
    """Applies GPTQ quantization layer-by-layer to a Keras model.

    This function is designed to work with common transformer architectures,
    like those provided by KerasHub. It automatically discovers the model's
    structure by first looking for the standard format: a `model.backbone`
    attribute that contains a `transformer_layers` list.

    If a standard backbone is not found, it falls back to a heuristic for
    custom models, where it assumes the first `keras.layers.Embedding` layer
    is the input embedding and any subsequent container layers are the
    transformer blocks to be quantized.

    The core logic operates as follows:
    1.  It automatically detects the model's structure, identifying the main
        embedding layer and a sequence of transformer blocks.
    2.  It processes the model sequentially, one block at a time. For each
        block, it uses temporary hooks to capture the input activations of
        each target layer during a forward pass with the calibration data.
    3.  These captured activations are used to compute the Hessian matrix for
        each layer's weights.
    4.  The GPTQ algorithm is then applied to each layer to find the optimal
        quantized weights that minimize the error introduced.
    5.  The output activations from the current block are then used as the
        input for the next block, ensuring that quantization errors are
        accounted for throughout the model.

    Args:
        model: The Keras model instance to be quantized. The function will
            attempt to automatically discover its structure.
        dataloader: An iterable providing calibration data. Each item should
            be a batch of token IDs suitable for the model's embedding layer.
        num_samples: (int) The number of samples from the dataloader to use for
            calibration.
        hessian_damping: (float) The percentage of dampening to add to the
            Hessian diagonal for stabilization during inverse calculation.
            A value of 0.01 is common.
        group_size: (int) The size of the groups to use for quantization. A
            value of 128 means that 128 weights will share the same scaling
            factor. Use -1 for per-channel quantization.
        symmetric: (bool) If True, symmetric quantization is used. Otherwise,
            asymmetric quantization is used.
        activation_order: (bool) If True, reorders the weight columns based on
            activation magnitude, which can improve quantization accuracy.
        weight_bits: (int) The number of bits to use for the quantized weights,
            e.g., 4 for 4-bit quantization.

    Raises:
        ValueError: If the function cannot automatically find an embedding
            layer or any transformer-like blocks to quantize within the model.
    """
    logging.info("Starting model quantization...")
    embedding_layer = None
    transformer_blocks = []
    if hasattr(model, "backbone"):
        logging.info("Detected KerasHub model structure.")
        backbone = model.backbone

        # Add the check for the 'transformer_layers' attribute.
        if hasattr(backbone, "transformer_layers"):
            transformer_blocks = backbone.transformer_layers
        else:
            # Raise a specific error if the attribute is missing.
            raise ValueError(
                "The model's backbone does not have a 'transformer_layers' "
                "attribute. Please ensure you are using a standard KerasHub "
                "transformer model."
            )
        # Find the embedding layer by checking for common names or by type.
        if hasattr(backbone, "token_embedding"):
            embedding_layer = backbone.token_embedding
        elif hasattr(backbone, "embedding"):
            embedding_layer = backbone.embedding
        else:
            raise ValueError(
                "Could not automatically find an embedding layer in the model."
            )

    else:
        logging.info("Detected custom model structure.")
        for layer in model.layers:
            # The first Embedding layer found is assumed to be the main one.
            if isinstance(layer, Embedding) and embedding_layer is None:
                embedding_layer = layer
            # A "block" is a container-like layer with its own sub-layers
            # that we can quantize. This is a heuristic that works for the
            # test.
            elif hasattr(layer, "_layers") and layer._layers:
                transformer_blocks.append(layer)

    if embedding_layer is None:
        raise ValueError(
            "Could not automatically find an embedding layer in the model."
        )
    if not transformer_blocks:
        raise ValueError(
            "Could not automatically find any transformer-like blocks to "
            "quantize."
        )

    # Initial inputs are the outputs of the token embedding layer
    inputs = [
        embedding_layer(ops.convert_to_tensor(batch, dtype="int32"))
        for batch in dataloader
    ]
    progbar = keras_utils.Progbar(target=len(transformer_blocks))

    for block_idx, block in enumerate(transformer_blocks):
        logging.info(f"Quantizing Block {block_idx}")
        sub_layers_map = find_layers_in_block(block)

        if not sub_layers_map:
            logging.info(
                f"  No Dense or EinsumDense layers found in block {block_idx}. "
                "Skipping."
            )
        else:
            logging.info(f"Found layers: {list(sub_layers_map.keys())}")
            gptq_objects = {
                name: GPTQ(layer) for name, layer in sub_layers_map.items()
            }

            captured_inputs = {name: [] for name in sub_layers_map.keys()}
            original_calls = {}

            def create_hook(name, original_call_func):
                """A factory for creating a hook to capture layer inputs."""

                def hook(*args, **kwargs):
                    if args:
                        inp = args[0]
                    else:
                        inp = kwargs["inputs"]
                    captured_inputs[name].append(inp)
                    return original_call_func(*args, **kwargs)

                return hook

            try:
                for name, layer in sub_layers_map.items():
                    original_call = layer.call
                    original_calls[name] = original_call
                    layer.call = create_hook(name, original_call)

                logging.info(f"Capturing activations for block {block_idx}...")
                for sample_idx in range(num_samples):
                    current_input = inputs[sample_idx]
                    if len(current_input.shape) == 2:
                        current_input = ops.expand_dims(current_input, axis=0)
                    _ = block(current_input)

            finally:
                for name, layer in sub_layers_map.items():
                    if name in original_calls:
                        layer.call = original_calls[name]

            logging.info(f"Building Hessians for block {block_idx}...")
            for name, gptq_object in gptq_objects.items():
                layer_inputs = ops.concatenate(captured_inputs[name], axis=0)

                # Explicitly reshape the input tensor to be 2D, with the second
                # dimension matching the number of input features expected by
                # the layer's kernel.
                # This correctly handles inputs of any dimensionality
                # (e.g., 3D or 4D).
                num_features = gptq_object.rows
                input_reshaped = ops.reshape(layer_inputs, (-1, num_features))
                gptq_object.update_hessian_with_batch(input_reshaped)

                quantizer = GPTQQuantization(
                    weight_bits,
                    per_channel=True,
                    symmetric=symmetric,
                    group_size=group_size,
                )
            for name, gptq_object in gptq_objects.items():
                logging.info(f"Quantizing {name}...")
                gptq_object.quantizer = quantizer
                gptq_object.quantize_and_correct_block(
                    hessian_damping=hessian_damping,
                    group_size=group_size,
                    activation_order=activation_order,
                )
                gptq_object.free()

            del gptq_objects, captured_inputs, original_calls

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

    logging.info("Quantization process complete.")


def quantize_model(model, config):
    """
    Top-level function to quantize a Keras model using GPTQ.
    """
    logging.info("Starting GPTQ quantization process...")

    # Load ALL data needed from the generator/source in a single call.
    total_samples_to_request = config.num_samples
    full_dataloader = get_dataloader(
        config.tokenizer,
        config.sequence_length,
        config.dataset,
        num_samples=total_samples_to_request,
    )

    # Split the materialized data. This works because full_dataloader
    # is now a NumPy array, which can be sliced and reused.
    calibration_dataloader = full_dataloader[: config.num_samples]

    apply_gptq_layerwise(
        model,
        calibration_dataloader,  # Use the calibration slice
        config.num_samples,  # Use the configured number of samples
        config.hessian_damping,
        config.group_size,
        config.symmetric,
        config.activation_order,
        config.weight_bits,
    )
