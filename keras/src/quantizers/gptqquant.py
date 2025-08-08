from keras.src import ops


def dequantize(x, scale, zero, maxq):
    """The core quantization function with correct broadcasting."""
    # Ensure scale is broadcastable with the input tensor x
    if scale.shape != x.shape:
        scale = ops.broadcast_to(scale, x.shape)

    # Ensure zero-point is also broadcastable
    if zero.shape != x.shape:
        zero = ops.broadcast_to(zero, x.shape)

    epsilon = 1e-8
    scale = ops.where(ops.equal(scale, 0), epsilon, scale)
    quantized_x = ops.divide(x, scale)
    quantized_x = ops.round(quantized_x)
    q = ops.add(quantized_x, zero)
    q = ops.clip(q, 0, maxq)
    dequantized_x = ops.subtract(q, zero)
    return ops.multiply(scale, dequantized_x)


class GPTQQuant:
    """Initializes the GPTQQuant state.

    Args:
        shape (int, optional): This argument is currently unused.
            Defaults to 1.

    Attributes:
        scale (tensor, optional): The quantization scaling factor(s). This
            is computed during the calibration process. Defaults to `None`.
        zero (tensor, optional): The quantization zero-point(s). This is
            computed during the calibration process. Defaults to `None`.
        maxq (tensor, optional): The maximum integer value for the
            quantized weights (e.g., 15 for 4-bit quantization).
            Defaults to `None`.
        wbits (int, optional): The number of bits to quantize to (e.g., 4).
            Defaults to `None`.
        perchannel (bool): A flag indicating whether quantization is
            applied per-channel (`True`) or per-tensor (`False`).
            Defaults to `False`.
        symmetric (bool): A flag indicating whether symmetric (`True`) or
            asymmetric (`False`) quantization is used. Defaults to `False`.
        group_size (int): The size of weight groups for quantization. A
            value of -1 indicates that grouping is not used.
            Defaults to -1.
    """

    def __init__(self):
        self.scale = None
        self.zero = None
        self.maxq = None
        self.wbits = None
        self.perchannel = False
        self.symmetric = False
        self.group_size = -1

    def configure(self, wbits, perchannel=True, symmetric=False, group_size=-1):
        """Configures the quantizer settings."""
        self.wbits = wbits
        self.maxq = ops.cast((2**wbits) - 1, "float32")
        self.perchannel = perchannel
        self.symmetric = symmetric
        self.group_size = group_size

    def find_params(self, x, weight=False):
        """Finds quantization parameters (scale and zero) for a given tensor."""

        if x is None:
            raise ValueError("Input tensor 'x' cannot be None.")

        # For weights, we typically expect at least a 2D tensor.
        if weight and len(x.shape) < 2:
            raise ValueError(
                f"Input weight tensor 'x' must have a rank of at least 2, "
                f"but got rank {len(x.shape)}."
            )

        if ops.size(x) == 0:
            raise ValueError("Input tensor 'x' cannot be empty.")

        original_shape = x.shape

        if self.perchannel:
            if weight:
                if self.group_size != -1:
                    x_reshaped = ops.reshape(x, [-1, self.group_size])
                else:
                    x_reshaped = ops.reshape(x, [original_shape[0], -1])
        else:  # per-tensor
            x_reshaped = ops.reshape(x, [1, -1])

        # Find min/max values
        xmin = ops.min(x_reshaped, axis=1)
        xmax = ops.max(x_reshaped, axis=1)

        # Apply symmetric quantization logic if enabled
        if self.symmetric:
            xmax = ops.maximum(ops.abs(xmin), xmax)
            xmin = ops.where(ops.less(xmin, 0), -xmax, xmin)

        # Ensure range is not zero to avoid division errors
        tmp = ops.equal(xmin, xmax)
        xmin = ops.where(tmp, xmin - 1, xmin)
        xmax = ops.where(tmp, xmax + 1, xmax)

        # Calculate scale and zero-point
        self.scale = (xmax - xmin) / self.maxq
        if self.symmetric:
            self.zero = ops.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = ops.round(-xmin / self.scale)

        # Ensure scale is non-zero
        self.scale = ops.where(ops.less_equal(self.scale, 0), 1e-8, self.scale)

        if weight:
            # Per-channel, non-grouped case: simple reshape is correct.
            if self.perchannel and self.group_size == -1:
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
