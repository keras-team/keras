"""CUDA graph capture for keras-on-torch inference.

Records the GPU kernel sequence of a forward pass once, then replays it on
subsequent calls without going through PyTorch's dispatcher or keras's
Python call path. Speedups of 3-25x are typical depending on the
overhead-vs-compute ratio of the model.

Constraints:

- Torch backend only. Other backends raise `ValueError`.
- Inputs must be on CUDA. CPU inputs raise `ValueError`.
- The captured graph is keyed to a single input structure, shape, and
  dtype. Calls with a different structure, shape, or dtype raise
  `ValueError`. Wrap the model again with the new structure to capture
  for it.
- Inference only. The capture is taken under `torch.no_grad`.
- No CPU sync points inside the model (`.cpu()`, `.item()`, Python
  branches on tensor values, prints). These either break capture or
  produce wrong results on replay.
- Random ops (e.g. `Dropout`) need the graph-safe RNG path or the same
  mask replays on every call.
"""

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export


@keras_export("keras.utils.cuda_graph")
def cuda_graph(model, sample_input):
    """Capture a model's forward pass as a CUDA graph and return a replay
    callable.

    The returned callable accepts an input with the same structure, shape,
    and dtype as `sample_input`, copies each leaf into a fixed capture
    buffer, replays the captured graph, and returns the model's output.
    Output tensors are views of internal capture buffers that are
    overwritten on the next call.

    Args:
        model: A `keras.Model` or any callable that runs on the torch
            backend.
        sample_input: A `torch.Tensor` on CUDA, or a nested structure
            (list, tuple, dict) whose leaves are CUDA `torch.Tensor`s,
            representing the input the graph will be captured for. The
            model is run on clones of these tensors during capture, so the
            originals are not modified.

    Returns:
        A callable `run(x)` that replays the captured graph with `x` as
        input. Output tensors alias internal capture buffers and will be
        overwritten by the next call; clone or copy them if you need to
        retain results across calls.

    Raises:
        ValueError: If the keras backend is not `torch`, if any leaf of
            `sample_input` is not a CUDA tensor, or if a call to the
            returned function receives an input whose structure, shape,
            or dtype differs from `sample_input`.

    Example:

    Requires the torch backend (set via `KERAS_BACKEND=torch` before
    importing keras).

    >>> import torch, keras
    >>> model = keras.Sequential([keras.layers.Dense(64)])
    >>> model.build((None, 32))
    >>> x = torch.randn(4, 32, device="cuda")
    >>> graphed = keras.utils.cuda_graph(model, x)
    >>> y = graphed(x)

    Nested inputs (list, tuple, dict, or nested combinations) work the
    same way; the structure passed at call time must match the one used
    at capture time.
    """
    if backend.backend() != "torch":
        raise ValueError(
            "`keras.utils.cuda_graph` requires the torch backend. "
            f"Current backend is `{backend.backend()}`."
        )

    import torch

    flat_sample = tree.flatten(sample_input)
    if not flat_sample:
        raise ValueError(
            "`sample_input` must contain at least one `torch.Tensor`."
        )
    for i, leaf in enumerate(flat_sample):
        if not isinstance(leaf, torch.Tensor):
            raise ValueError(
                "All leaves of `sample_input` must be `torch.Tensor` "
                f"instances. Leaf {i} is `{type(leaf).__name__}`."
            )
        if leaf.device.type != "cuda":
            raise ValueError(
                "All leaves of `sample_input` must be on a CUDA device. "
                f"Leaf {i} is on `{leaf.device}`."
            )

    flat_static = [torch.empty_like(t) for t in flat_sample]
    for static_t, sample_t in zip(flat_static, flat_sample):
        static_t.copy_(sample_t)
    static_input = tree.pack_sequence_as(sample_input, flat_static)

    # Run a few warm-up iterations on a side stream so allocator and
    # autotuner state are settled before we record. The recommended count
    # from torch is three.
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        with torch.no_grad():
            for _ in range(3):
                _ = model(static_input)
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        static_output = model(static_input)

    captured_shapes = [tuple(t.shape) for t in flat_sample]
    captured_dtypes = [t.dtype for t in flat_sample]

    def run(x):
        try:
            tree.assert_same_structure(sample_input, x)
        except ValueError as e:
            raise ValueError(
                "Input structure does not match the structure captured by "
                f"`cuda_graph`. {e}"
            )
        flat_x = tree.flatten(x)
        for i, leaf in enumerate(flat_x):
            if not isinstance(leaf, torch.Tensor):
                raise ValueError(
                    "All leaves of the input must be `torch.Tensor` "
                    f"instances. Leaf {i} is `{type(leaf).__name__}`."
                )
            if tuple(leaf.shape) != captured_shapes[i]:
                raise ValueError(
                    "Input shape does not match the shape captured by "
                    f"`cuda_graph`. Leaf {i}: captured "
                    f"{captured_shapes[i]}, received {tuple(leaf.shape)}. "
                    "Capture a new graph for the new shape."
                )
            if leaf.dtype != captured_dtypes[i]:
                raise ValueError(
                    "Input dtype does not match the dtype captured by "
                    f"`cuda_graph`. Leaf {i}: captured "
                    f"{captured_dtypes[i]}, received {leaf.dtype}. "
                    "Capture a new graph for the new dtype."
                )
            flat_static[i].copy_(leaf, non_blocking=True)
        graph.replay()
        return static_output

    return run
