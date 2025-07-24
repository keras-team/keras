from keras.src import ops


def quantize(x, scale, zero, maxq):
    """The core quantization function with correct broadcasting."""
    # Ensure scale is broadcastable with the input tensor x
    if scale.shape != x.shape:
        scale = ops.broadcast_to(scale, x.shape)

    # Ensure zero-point is also broadcastable
    if zero.shape != x.shape:
        zero = ops.broadcast_to(zero, x.shape)

    scale = ops.where(ops.equal(scale, 0), 1e-8, scale)
    q = ops.round(x / scale) + zero
    q = ops.clip(q, 0, maxq)
    return scale * (q - zero)


class Quantizer:
    """
    This version contains the definitive fix for the per-tensor shape mismatch,
    as identified by the unit test. It now correctly tiles the per-tensor
    parameters to match the behavior of the original TensorFlow implementation.
    """

    def __init__(self, shape=1):
        self.scale = None
        self.zero = None
        self.maxq = None
        self.wbits = None
        self.perchannel = False
        self.sym = False
        self.groupsize = -1

    def configure(self, wbits, perchannel=True, sym=False, groupsize=-1):
        """Configures the quantizer settings."""
        self.wbits = wbits
        self.maxq = ops.cast((2**wbits) - 1, "float32")
        self.perchannel = perchannel
        self.sym = sym
        self.groupsize = groupsize

    def find_params(self, x, weight=False):
        """Finds quantization parameters (scale and zero) for a given tensor."""

        original_shape = x.shape

        if self.perchannel:
            if weight:
                if self.groupsize != -1:
                    x_reshaped = ops.reshape(x, [-1, self.groupsize])
                else:
                    x_reshaped = ops.reshape(x, [original_shape[0], -1])
        else:  # per-tensor
            x_reshaped = ops.reshape(x, [1, -1])

        # Find min/max values
        xmin = ops.min(x_reshaped, axis=1)
        xmax = ops.max(x_reshaped, axis=1)

        # Apply symmetric quantization logic if enabled
        if self.sym:
            xmax = ops.maximum(ops.abs(xmin), xmax)
            xmin = ops.where(ops.less(xmin, 0), -xmax, xmin)

        # Ensure range is not zero to avoid division errors
        tmp = ops.equal(xmin, xmax)
        xmin = ops.where(tmp, xmin - 1, xmin)
        xmax = ops.where(tmp, xmax + 1, xmax)

        # Calculate scale and zero-point
        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = ops.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = ops.round(-xmin / self.scale)

        # Ensure scale is non-zero
        self.scale = ops.where(ops.less_equal(self.scale, 0), 1e-8, self.scale)

        if weight:
            # Per-channel, non-grouped case: simple reshape is correct.
            if self.perchannel and self.groupsize == -1:
                self.scale = ops.reshape(self.scale, [-1, 1])
                self.zero = ops.reshape(self.zero, [-1, 1])
            elif not self.perchannel:
                num_rows = original_shape[0]
                self.scale = ops.tile(
                    ops.reshape(self.scale, (1, 1)), (num_rows, 1)
                )
                self.zero = ops.tile(
                    ops.reshape(self.zero, (1, 1)), (num_rows, 1)
                )
        if self.perchannel:
            self.scale = ops.reshape(self.scale, [-1, 1])
            self.zero = ops.reshape(self.zero, [-1, 1])

    def ready(self):
        """Checks if the quantization parameters have been computed."""
        return self.scale is not None and self.zero is not None
