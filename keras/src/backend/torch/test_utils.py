"""Shared test utilities for the PyTorch backend."""

import torch

from keras.src.backend.torch.ops.core import get_device

# One capture stream for the process. A new stream for each capture can be
# a stream that a live graph uses, and the cuBLAS workspace of that stream
# is freed under the live graph.
_capture_stream = None


def skip_unless_cuda(test_case):
    """Skips the test unless the torch device is a CUDA device."""
    if torch.device(get_device()).type != "cuda":
        test_case.skipTest("CUDA graphs require a CUDA device.")


def capture_cuda_graph(fn):
    """Runs `fn`, captures it in a CUDA graph and replays the graph.

    The capture fails when `fn` copies data from the host to the device,
    for example a Python number that an op makes into a tensor on the
    host.

    Args:
        fn: A function with no arguments that returns tensors.

    Returns:
        `(eager, replayed)`: the outputs of the first eager call, and the
        outputs of the graph after one replay.
    """
    global _capture_stream
    # The eager call also builds what `fn` builds lazily, outside the
    # capture.
    eager = fn()
    if _capture_stream is None:
        _capture_stream = torch.cuda.Stream()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.current_stream()
    try:
        with torch.cuda.graph(graph, stream=_capture_stream):
            replayed = fn()
    finally:
        # A failed capture leaves the capture stream current.
        torch.cuda.set_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    return eager, replayed
