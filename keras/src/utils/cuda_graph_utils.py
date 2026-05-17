"""CUDA graph capture for keras-on-torch inference.

Records the GPU kernel sequence of a forward pass once, then replays it on
subsequent calls without going through PyTorch's dispatcher or keras's
Python call path. Speedups of 3-25x are typical depending on the
overhead-vs-compute ratio of the model.

Constraints:

- Torch backend only. Other backends raise `ValueError`.
- Inputs must be on CUDA. CPU inputs raise `ValueError`.
- The captured graph is keyed to a single input shape. Calls with a
  different shape raise `ValueError`. Wrap the model again with the new
  shape to capture for it.
- Inference only. The capture is taken under `torch.no_grad`.
- No CPU sync points inside the model (`.cpu()`, `.item()`, Python
  branches on tensor values, prints). These either break capture or
  produce wrong results on replay.
- Random ops (e.g. `Dropout`) need the graph-safe RNG path or the same
  mask replays on every call.
"""

from keras.src import backend
from keras.src.api_export import keras_export


@keras_export("keras.utils.cuda_graph")
def cuda_graph(model, sample_input):
    """Capture a model's forward pass as a CUDA graph and return a replay
    callable.

    The returned callable accepts an input tensor of the same shape and
    dtype as `sample_input`, copies it into a fixed capture buffer, replays
    the captured graph, and returns the model's output (a view of the
    capture output buffer that is overwritten on the next call).

    Args:
        model: A `keras.Model` or any callable that runs on the torch
            backend.
        sample_input: A torch tensor on CUDA representing the input shape
            and dtype the graph will be captured for. The model is run on
            a clone of this tensor during capture, so the tensor itself is
            not modified.

    Returns:
        A callable `run(x)` that replays the captured graph with `x` as
        input. The returned tensor aliases an internal capture buffer and
        will be overwritten by the next call; clone or copy it if you need
        to retain the result across calls.

    Raises:
        ValueError: If the keras backend is not `torch`, if `sample_input`
            is not a CUDA tensor, or if a call to the returned function
            receives an input whose shape or dtype differs from
            `sample_input`.

    Example:

    Requires the torch backend (set via `KERAS_BACKEND=torch` before
    importing keras).

    >>> import torch, keras
    >>> model = keras.Sequential([keras.layers.Dense(64)])
    >>> model.build((None, 32))
    >>> x = torch.randn(4, 32, device="cuda")
    >>> graphed = keras.utils.cuda_graph(model, x)
    >>> y = graphed(x)
    """
    if backend.backend() != "torch":
        raise ValueError(
            "`keras.utils.cuda_graph` requires the torch backend. "
            f"Current backend is `{backend.backend()}`."
        )

    import torch

    if not isinstance(sample_input, torch.Tensor):
        raise ValueError(
            "`sample_input` must be a `torch.Tensor`. "
            f"Got `{type(sample_input).__name__}`."
        )
    if sample_input.device.type != "cuda":
        raise ValueError(
            "`sample_input` must be on a CUDA device. "
            f"Got device `{sample_input.device}`."
        )

    static_input = torch.empty_like(sample_input)
    static_input.copy_(sample_input)

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

    captured_shape = tuple(static_input.shape)
    captured_dtype = static_input.dtype

    def run(x):
        if not isinstance(x, torch.Tensor):
            raise ValueError(
                "Input to a `cuda_graph` callable must be a `torch.Tensor`. "
                f"Got `{type(x).__name__}`."
            )
        if tuple(x.shape) != captured_shape:
            raise ValueError(
                "Input shape does not match the shape captured by "
                "`cuda_graph`. Captured "
                f"{captured_shape}, received {tuple(x.shape)}. "
                "Capture a new graph for the new shape."
            )
        if x.dtype != captured_dtype:
            raise ValueError(
                "Input dtype does not match the dtype captured by "
                "`cuda_graph`. Captured "
                f"{captured_dtype}, received {x.dtype}. "
                "Capture a new graph for the new dtype."
            )
        static_input.copy_(x, non_blocking=True)
        graph.replay()
        return static_output

    return run
