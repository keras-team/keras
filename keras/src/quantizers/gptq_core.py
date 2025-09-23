import math
from contextlib import contextmanager

import numpy as np
from absl import logging

from keras.src import ops
from keras.src import utils as keras_utils
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.layers import Embedding
from keras.src.quantizers.gptq import GPTQ


@contextmanager
def stream_hessians(layers_map, gptq_objects):
    """
    Temporarily monkey-patch each target layer's `call` method so
    that input activations are streamed into the GPTQ instance
    running Hessian estimate at capture time.

    On `__enter__`: For every (name, layer) in `layers_map`, replaces
     `layer.call` with a wrapper that:
     1) extracts the layer input from `*args`/`**kwargs`,
     2) reshapes it to 2D `[-1, rows]` where
      `rows = gptq_objects[name].rows`,
     3) calls `gptq_objects[name].update_hessian_with_batch(x2d)`
     4) delegates to the original `layer.call` and returns its
      output.

    On `__exit__`: All original `layer.call` methods are restored even if an
     exception occurs.

    * Space complexity: O(d**2) per layer (for the Hessian).
    * No weights are modified; only GPTQ statistics are updated.

    Args:
        layers_map: Dict[str, Layer]. Mapping from logical layer names to
         the Keras layers that should be patched during calibration. Keys must
         match `gptq_objects`.
        gptq_objects: Dict[str, GPTQ]. Mapping from names to GPTQ instances.

    Yields:
        None: The patched state is active only within the `with` block. After
         exit, all layers are unpatched and safe to use normally.

    Example:
    ```python
    >>> with stream_hessians(layers_map, gptq_objects):
    ...     for sample in calibration_inputs:
    ...         if len(sample.shape) == 2:
    ...             sample = ops.expand_dims(sample, 0)
    ...         _ = block(sample)   # hooks update Hessians on-the-fly
    >>> # <- original layer.call methods restored here
    ```
    """
    original_calls = {}

    def create_hook(name, original_call_func):
        def hook(*args, **kwargs):
            inp = args[0] if args else kwargs["inputs"]
            # Explicitly reshape the input tensor to be 2D, with the
            # second dimension matching the number of input features
            # expected by the layer's kernel.
            # This correctly handles inputs of any dimensionality
            # (e.g., 3D or 4D).
            num_features = gptq_objects[name].rows
            input_2d = ops.reshape(inp, (-1, num_features))
            gptq_objects[name].update_hessian_with_batch(input_2d)
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


def get_dataloader(
    tokenizer,
    sequence_length,
    dataset,
    num_samples=128,
    *,
    strategy="strided",
    seed=42,
    stride=None,
    eos_id=None,
):
    """
    Prepares and chunks the calibration dataloader, repeating short datasets.
    All processing happens on the CPU.

    Args:
        tokenizer: The tokenizer to use for text splitting.
        sequence_length: The length of each input sequence.
        dataset: The dataset to sample from.
        num_samples: The number of samples to generate.
        strategy: The sampling strategy to use. Possible values are
         1. "strided": Samples are taken at regular intervals.
         2. "linspace": Samples are taken at evenly spaced intervals.
         3. "random": Samples are taken at random positions.
        seed: The random seed for reproducibility. Used only if
         strategy="random"
        stride: The stride length for "strided" sampling.
        eos_id: The end-of-sequence token ID.

    Returns:
        np.ndarray of shape (num_samples, 1, sequence_length), dtype int32.
    """
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

    pieces = []
    if isinstance(dataset_list[0], str):
        for i, s in enumerate(dataset_list):
            toks = np.asarray(tokenizer.tokenize(s)).reshape(-1)
            pieces.append(toks)
            # avoid windows that span document boundaries
            if eos_id is not None and i < len(dataset_list) - 1:
                pieces.append(np.array([eos_id], dtype=np.int32))
    else:
        for s in dataset_list:
            toks = ops.convert_to_numpy(s).reshape(-1)
            pieces.append(toks.astype(np.int32, copy=False))

    all_tokens = (
        pieces[0].astype(np.int32, copy=False)
        if len(pieces) == 1
        else np.concatenate(pieces, axis=0).astype(np.int32, copy=False)
    )

    required_tokens = num_samples * sequence_length
    if all_tokens.size < required_tokens:
        repeats = math.ceil(required_tokens / max(1, all_tokens.size))
        all_tokens = np.tile(all_tokens, repeats)

    max_start = all_tokens.size - sequence_length
    if max_start < 0:
        raise ValueError(
            f"Not enough tokens to form one sample of length {sequence_length} "
            f"(have {all_tokens.size})."
        )

    # Choose deterministic, well-spread starts by default
    if strategy == "random":
        rng = np.random.default_rng(seed)
        starts = rng.integers(
            0, max_start + 1, size=num_samples, dtype=np.int64
        )
    elif strategy == "linspace":
        # even coverage with no RNG
        starts = np.linspace(0, max_start, num_samples, dtype=np.int64)
    elif strategy == "strided":
        # stride chosen to cover the space roughly uniformly
        if stride is None:
            stride = max(1, (max_start + 1) // num_samples)
        # offset derived deterministically from seed
        offset = (
            (abs(hash(("gptq-calib", seed))) % (max_start + 1))
            if max_start > 0
            else 0
        )
        starts = (offset + np.arange(num_samples, dtype=np.int64) * stride) % (
            max_start + 1
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Gather contiguous windows
    # sliding_window_view avoids building a big index matrix
    windows = np.lib.stride_tricks.sliding_window_view(
        all_tokens, sequence_length
    )
    samples = windows[starts]  # (num_samples, sequence_length)
    return samples.astype(np.int32)[:, None, :]


def _get_backbone_layers(model):
    """Extract embedding and transformer layers from a KerasHub model."""
    backbone = model.backbone
    if not hasattr(backbone, "transformer_layers"):
        raise ValueError(
            "The model's backbone does not have a 'transformer_layers' "
            "attribute. Please ensure you are using a standard KerasHub "
            "transformer model."
        )
    transformer_blocks = backbone.transformer_layers

    embedding_layer = None
    if hasattr(backbone, "token_embedding"):
        embedding_layer = backbone.token_embedding
    elif hasattr(backbone, "embedding"):
        embedding_layer = backbone.embedding
    return embedding_layer, transformer_blocks


def _get_custom_layers(model):
    """Heuristic for extracting embedding + transformer blocks from a custom
    model."""
    embedding_layer = None
    transformer_blocks = []
    for layer in model.layers:
        if isinstance(layer, Embedding) and embedding_layer is None:
            embedding_layer = layer
        elif getattr(layer, "_layers", None):  # container-like block
            transformer_blocks.append(layer)
    return embedding_layer, transformer_blocks


def find_layers_in_block(block):
    """
    Finds all Dense and EinsumDense layers in a transformer block.

    Args:
        block: A Keras layer representing a transformer block.
    Returns:
        A dict mapping layer paths to the corresponding Dense or EinsumDense
    """
    found_layers = {}
    for sub_layer in block._flatten_layers():
        if len(list(sub_layer._flatten_layers())) == 1:
            if isinstance(sub_layer, (Dense, EinsumDense)):
                found_layers[sub_layer.path] = sub_layer
    return found_layers


def apply_gptq_layerwise(model, dataloader, config):
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
        config: A GPTQConfiguration object.

    Raises:
        ValueError: If the function cannot automatically find an embedding
            layer or any transformer-like blocks to quantize within the model.
    """

    num_samples = config.num_samples

    logging.info("Starting model quantization...")
    embedding_layer = None
    transformer_blocks = []
    if hasattr(model, "backbone"):
        logging.info("Detected KerasHub model structure.")
        embedding_layer, transformer_blocks = _get_backbone_layers(model)
    else:
        logging.info("Detected custom model structure.")
        embedding_layer, transformer_blocks = _get_custom_layers(model)

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
    num_samples = min(num_samples, len(inputs))

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
                name: GPTQ(layer, config)
                for name, layer in sub_layers_map.items()
            }

            with stream_hessians(sub_layers_map, gptq_objects):
                for sample_idx in range(num_samples):
                    current_input = inputs[sample_idx]
                    if len(current_input.shape) == 2:
                        current_input = ops.expand_dims(current_input, axis=0)
                    _ = block(current_input)

            for name, gptq_object in gptq_objects.items():
                logging.info(f"Quantizing {name}...")
                gptq_object.quantize_and_correct_layer()
                gptq_object.free()

            del gptq_objects

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


def gptq_quantize(model, config):
    """
    Top-level function to quantize a Keras model using GPTQ.
    """
    logging.info("Starting GPTQ quantization process...")

    # Load all data needed from the generator/source in a single call.
    total_samples_to_request = config.num_samples
    dataloader = get_dataloader(
        config.tokenizer,
        config.sequence_length,
        config.dataset,
        num_samples=total_samples_to_request,
    )

    # Split the materialized data. This works because dataloader
    # is now a NumPy array, which can be sliced and reused.
    calibration_dataloader = dataloader[: config.num_samples]

    apply_gptq_layerwise(model, calibration_dataloader, config)
