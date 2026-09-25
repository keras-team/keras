import math
import warnings
from contextlib import contextmanager
from contextlib import nullcontext

import numpy as np
from absl import logging

from keras.src import backend
from keras.src import ops
from keras.src import utils as keras_utils
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.quantizers.gptq import GPTQ
from keras.src.quantizers.utils import should_quantize_layer


def get_calibration_call_attribute(layer):
    """Returns the name of the forward method dispatched for `layer`.

    `Operation.__call__` dispatches to `layer.quantized_call` (instead of
    `layer.call`) as soon as the layer has a quantization mode. During
    `model.quantize(...)`, layers are switched to their quantized dtype
    policy before calibration runs, so calibration hooks must patch the
    method that is actually invoked.

    Args:
        layer: The Keras layer to inspect.

    Returns:
        str. Either `"quantized_call"` or `"call"`.
    """
    if getattr(layer, "quantization_mode", None) is not None:
        return "quantized_call"
    return "call"


def calibration_no_grad_scope():
    """Returns a context manager that disables gradient tracking.

    Calibration is inference-only: it runs forward passes to accumulate
    statistics and then assigns quantized values to variables. On the torch
    backend those forwards would otherwise build autograd graphs, and
    because per-sample activations are retained across the whole
    calibration loop (and AWQ stashes activation samples for its clipping
    search), the graphs and every intermediate activation stay alive until
    the block completes - enough to exhaust GPU memory on models that fit
    comfortably otherwise. JAX and TensorFlow build no such graphs in eager
    mode, so this is a no-op there.
    """
    if backend.backend() == "torch":
        import torch

        return torch.no_grad()
    return nullcontext()


@contextmanager
def stream_hessians(layers_map, gptq_objects, execution_trace=None):
    """
    Temporarily monkey-patch each target layer's forward method so
    that input activations are streamed into the GPTQ instance
    running Hessian estimate at capture time.

    On `__enter__`: For every (name, layer) in `layers_map`, replaces
     the layer's dispatched forward method (`layer.quantized_call` if the
     layer is already in a quantized mode, `layer.call` otherwise) with a
     wrapper that:
     1) extracts the layer input from `*args`/`**kwargs`,
     2) reshapes it to 2D `[-1, rows]` where
      `rows = gptq_objects[name].rows`,
     3) calls `gptq_objects[name].update_hessian_with_batch(x2d)`
     4) delegates to the original forward method and returns its
      output.

    On `__exit__`: All original forward methods are restored even if an
     exception occurs.

    * Space complexity: O(d**2) per layer (for the Hessian).
    * No weights are modified; only GPTQ statistics are updated.

    Args:
        layers_map: Dict[str, Layer]. Mapping from logical layer names to
         the Keras layers that should be patched during calibration. Keys must
         match `gptq_objects`.
        gptq_objects: Dict[str, GPTQ]. Mapping from names to GPTQ instances.
        execution_trace: Optional dict. When provided, each layer's FIRST
         hook invocation records `{name: (call_index, input_tensor)}` into
         it — the block-level execution order and the identity of the input
         each layer consumes. Used to derive within-block quantization
         stages (see `apply_gptq_layerwise`). The recorded tensors are only
         kept alive for identity comparison; callers should drop the trace
         after use.

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
    >>> # <- original forward methods restored here
    ```
    """
    original_calls = {}
    call_counter = [0]

    def create_hook(name, original_call_func):
        def hook(*args, **kwargs):
            inp = args[0] if args else kwargs["inputs"]
            if execution_trace is not None and name not in execution_trace:
                # Record block-level execution order and the input tensor's
                # identity on the first call (a live reference is kept so the
                # id cannot be recycled while tracing).
                execution_trace[name] = (call_counter[0], inp)
                call_counter[0] += 1
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
            attr = get_calibration_call_attribute(layer)
            original_calls[name] = (attr, getattr(layer, attr))
            setattr(layer, attr, create_hook(name, original_calls[name][1]))
        yield
    finally:
        for name, (attr, original_call) in original_calls.items():
            setattr(layers_map[name], attr, original_call)


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
            toks = ops.convert_to_numpy(tokenizer.tokenize(s)).reshape(-1)
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
        # Offset derived deterministically from the seed. Python's
        # built-in `hash()` must not be used here: string/tuple hashes are
        # randomized per process (PYTHONHASHSEED), which silently made
        # calibration windows - and therefore every quantization result -
        # unreproducible across runs despite the fixed seed.
        offset = (
            int(np.random.default_rng(seed).integers(0, max_start + 1))
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


def _stack_calibration_batch(samples):
    """Stacks a list of per-sample calibration activations into a single batch.

    Each element may be a 2D `[sequence, features]` or 3D
    `[1, sequence, features]` tensor. Every element is normalized to a leading
    batch axis of size 1 and concatenated along axis 0, producing a
    `[batch, sequence, features]` tensor that can be run through a block in a
    single forward pass.

    Args:
        samples: List of per-sample activation tensors.

    Returns:
        A single `[batch, sequence, features]` tensor.
    """
    normalized = []
    for sample in samples:
        if ops.ndim(sample) == 2:
            sample = ops.expand_dims(sample, axis=0)
        normalized.append(sample)
    if len(normalized) == 1:
        return normalized[0]
    return ops.concatenate(normalized, axis=0)


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
        # A quantizable layer may own sub-layers (e.g. a `Layer`
        # activation), so no leaf filtering here — collect every Dense/
        # EinsumDense reachable inside the block.
        if isinstance(sub_layer, (Dense, EinsumDense)):
            found_layers[sub_layer.path] = sub_layer
    return found_layers


def _execution_stages(layer_names, execution_trace):
    """Groups a block's layers into sequential quantization stages.

    Reference GPTQ implementations quantize a block's sub-layers in
    topological order ("true sequential"): once an upstream sub-layer is
    quantized, downstream Hessians are re-estimated on the quantized
    activations. Without this, e.g. an MLP's Hessian is captured while the
    attention sub-layers are still full-precision, and the error
    corrections it derives are tuned to activations that no longer exist
    once attention is quantized too — which measurably degrades quality
    below plain round-to-nearest.

    Stages are derived from the calibration trace: layers are ordered by
    first invocation, and layers that consumed the *same* input tensor
    (e.g. the query/key/value projections, or an MLP's gate/up pair) share
    a stage since quantizing one cannot affect the others' inputs. Layers
    that never fired during tracing are placed in the first stage,
    preserving their (empty) Hessian behavior.

    Args:
        layer_names: Iterable of layer names in the block.
        execution_trace: Dict of `{name: (call_index, input_tensor)}`
            recorded by `stream_hessians`.

    Returns:
        List of lists of layer names, one list per stage, in execution
        order.
    """
    traced = [name for name in layer_names if name in execution_trace]
    untraced = [name for name in layer_names if name not in execution_trace]
    traced.sort(key=lambda name: execution_trace[name][0])

    stages = []
    current_stage, current_input_id = [], None
    for name in traced:
        input_id = id(execution_trace[name][1])
        if current_stage and input_id != current_input_id:
            stages.append(current_stage)
            current_stage = []
        current_stage.append(name)
        current_input_id = input_id
    if current_stage:
        stages.append(current_stage)

    if untraced:
        if stages:
            stages[0] = untraced + stages[0]
        else:
            stages = [untraced]
    return stages


def apply_gptq_layerwise(dataloader, config, structure, filters=None):
    """Applies GPTQ quantization layer-by-layer to a Keras model.

    This function uses the provided `structure` to identify pre-quantization
    layers and sequential blocks.

    The core logic operates as follows:

    1.  It processes the model sequentially, one block at a time. For each
        block, it uses temporary hooks to capture the input activations of
        each target layer during a forward pass with the calibration data.
    2.  These captured activations are used to compute the Hessian matrix for
        each layer's weights.
    3.  The GPTQ algorithm is then applied to each layer to find the optimal
        quantized weights that minimize the error introduced.
    4.  The output activations from the current block are then used as the
        input for the next block, ensuring that quantization errors are
        accounted for throughout the model.

    Args:
        dataloader: An iterable providing calibration data.
        config: A GPTQConfiguration object.
        structure: A dictionary with keys "pre_block_layers" and
            "sequential_blocks".
        filters: Optional filters to exclude layers from quantization.

    Raises:
        ValueError: If the function cannot automatically find an embedding
            layer or any transformer-like blocks to quantize within the model.
    """

    num_samples = config.num_samples

    logging.info("Starting model quantization...")

    pre_layers = structure.get("pre_block_layers", [])
    transformer_blocks = structure.get("sequential_blocks", [])

    if not transformer_blocks:
        raise ValueError(
            "No sequential blocks found in the provided structure to quantize."
        )

    # Initial inputs are the outputs of the pre-block layers
    inputs = []
    for batch in dataloader:
        batch = ops.convert_to_tensor(batch, dtype="int32")
        for layer in pre_layers:
            batch = layer(batch)
        inputs.append(batch)

    num_samples = min(num_samples, len(inputs))
    inputs = inputs[:num_samples]

    # Run calibration forward passes at this batch size. Because the Hessian is
    # accumulated over the flattened `[-1, features]` activations, batching is
    # mathematically identical to running one sample at a time; it only reduces
    # the number of (expensive) forward passes through each block.
    batch_size = max(1, int(config.calibration_batch_size))

    progbar = keras_utils.Progbar(target=len(transformer_blocks))

    # Layers whose Hessian saw too few calibration tokens relative to their
    # input width, collected across all blocks for a single summary warning.
    undersampled_layers = []

    for block_idx, block in enumerate(transformer_blocks):
        logging.info(f"Quantizing Block {block_idx}")
        sub_layers_map = find_layers_in_block(block)

        # Filter out layers that are not quantized with GPTQ
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
            gptq_objects = {
                name: GPTQ(layer, config)
                for name, layer in sub_layers_map.items()
            }

            execution_trace = {}
            with stream_hessians(
                sub_layers_map, gptq_objects, execution_trace=execution_trace
            ):
                for start in range(0, num_samples, batch_size):
                    batch = _stack_calibration_batch(
                        inputs[start : start + batch_size]
                    )
                    _ = block(batch)

            # Quantize the block's layers in execution-order stages ("true
            # sequential", as in reference GPTQ): after each stage is
            # quantized, downstream stages' Hessians are re-estimated so
            # their error corrections are computed against the quantized
            # upstream activations they will actually see at inference.
            stages = _execution_stages(sub_layers_map, execution_trace)
            del execution_trace

            for stage_idx, stage_names in enumerate(stages):
                if stage_idx > 0:
                    stage_map = {
                        name: sub_layers_map[name] for name in stage_names
                    }
                    stage_objects = {}
                    for name in stage_names:
                        gptq_objects[name].free()
                        gptq_objects[name] = GPTQ(sub_layers_map[name], config)
                        stage_objects[name] = gptq_objects[name]
                    with stream_hessians(stage_map, stage_objects):
                        for start in range(0, num_samples, batch_size):
                            batch = _stack_calibration_batch(
                                inputs[start : start + batch_size]
                            )
                            _ = block(batch)

                for name in stage_names:
                    gptq_object = gptq_objects[name]
                    tokens = int(gptq_object.num_samples)
                    rows = int(gptq_object.rows)
                    if tokens < 4 * rows:
                        undersampled_layers.append((name, tokens, rows))
                    logging.info(f"Quantizing {name}...")
                    gptq_object.quantize_and_correct_layer()
                    gptq_object.free()

            del gptq_objects

        if block_idx < len(transformer_blocks) - 1:
            logging.info(f"Generating inputs for block {block_idx + 1}...")
            next_block_inputs = []
            for start in range(0, num_samples, batch_size):
                chunk = inputs[start : start + batch_size]
                output = block(_stack_calibration_batch(chunk))
                if isinstance(output, (list, tuple)):
                    output = output[0]
                # Split the batched output back into per-sample activations so
                # the next block can be calibrated identically.
                for sample_idx in range(len(chunk)):
                    next_block_inputs.append(output[sample_idx])
            inputs = next_block_inputs
        progbar.update(current=block_idx + 1)

    if undersampled_layers:
        worst = min(t / r for _, t, r in undersampled_layers)
        examples = ", ".join(
            f"{name} ({tokens} tokens for {rows} input features)"
            for name, tokens, rows in undersampled_layers[:3]
        )
        warnings.warn(
            f"GPTQ calibration is undersampled for "
            f"{len(undersampled_layers)} layer(s): fewer than 4 calibration "
            "tokens per input feature (worst ratio: "
            f"{worst:.1f}). With this little data the Hessian is close to "
            "singular and GPTQ's error correction can overfit the "
            "calibration set and produce worse results than plain "
            "round-to-nearest. Increase `num_samples` and/or "
            "`sequence_length` in `GPTQConfig` (8 or more tokens per input "
            f"feature is recommended). Examples: {examples}.",
            stacklevel=2,
        )

    logging.info("Quantization process complete.")


def gptq_quantize(config, quantization_layer_structure, filters=None):
    """
    Quantizes the model using GPTQ.

    Args:
        config: The GPTQ configuration.
        quantization_layer_structure: A dictionary describing the model's layer
        structure for quantization.
        filters: Optional filters to exclude layers from quantization.
    """
    if config.dataset is None or config.tokenizer is None:
        raise ValueError(
            "GPTQ quantization requires a dataset and a tokenizer. "
            "Please provide them in the `GPTQConfig`."
        )

    if quantization_layer_structure is None:
        raise ValueError(
            "For 'gptq' mode, a valid quantization structure must be provided "
            "either via `config.quantization_layer_structure` or by overriding "
            "`model.get_quantization_layer_structure(mode)`. The structure "
            "should be a dictionary with keys 'pre_block_layers' and "
            "'sequential_blocks'."
        )

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

    with calibration_no_grad_scope():
        apply_gptq_layerwise(
            calibration_dataloader,
            config,
            quantization_layer_structure,
            filters=filters,
        )
