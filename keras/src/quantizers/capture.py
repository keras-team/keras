"""Calibration capture: a hook the dispatch machinery runs before a forward.

The calibration modes (GPTQ, AWQ) observe the inputs of every layer they
quantize while calibration data flows through a block. Rather than
replacing a layer's `call` for the duration, a capture is registered on
the layer and `Operation._dispatch_call`, the one place that selects the
forward pass, runs it before the forward it dispatches to: `call` or
`quantized_call`, eager or rematerialized, under `__call__` or
`stateless_call`. The scope sets one slot on the layer
(`_calibration_capture`) and resets it on exit; nothing else on the layer
changes, so a dispatch change cannot orphan calibration silently.
"""

from contextlib import contextmanager


@contextmanager
def calibration_scope(captures):
    """Runs a capture on each layer's inputs before its forward pass.

    Args:
        captures: Dict mapping layers to callables. While the scope is
            active, every forward pass of a layer first passes the layer's
            input (its first positional argument, or its `inputs` keyword
            argument) to the callable, exactly as the forward receives it.
            A capture must not mutate the input. Symbolic calls run no
            capture, including the shape inference that a symbolic call
            of an enclosing layer runs on the layer.

    Raises:
        ValueError: If a layer already carries a capture, since one
            calibration observes a layer at a time.
    """
    installed = []
    try:
        for layer, capture in captures.items():
            if layer._calibration_capture is not None:
                raise ValueError(
                    f"Layer '{layer.name}' is already inside a "
                    "`calibration_scope`."
                )
            _set_capture(layer, capture)
            installed.append(layer)
        yield
    finally:
        for layer in installed:
            _set_capture(layer, None)


def _set_capture(layer, capture):
    # The slot is dispatch bookkeeping, not layer state: bypass the
    # attribute tracker and, under NNX, the module's attribute rules.
    object.__setattr__(layer, "_calibration_capture", capture)
