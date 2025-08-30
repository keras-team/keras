from keras.src import ops


def dequantize(input_tensor, scale, zero, maxq):
    """The core quantization function."""
    epsilon = ops.cast(1e-8, dtype=scale.dtype)
    scale = ops.where(ops.equal(scale, 0), epsilon, scale)

    quantized_tensor = ops.divide(input_tensor, scale)
    quantized_tensor = ops.round(quantized_tensor)
    q = ops.add(quantized_tensor, zero)
    q = ops.clip(q, 0, maxq)

    dequantized_tensor = ops.subtract(q, zero)
    return ops.multiply(scale, dequantized_tensor)


class GPTQQuantization:
    """A class that handles the quantization of weights using GPTQ method.

    This class provides methods to find quantization parameters (scale and zero)
    for a given tensor and can be used to quantize weights in a GPTQ context.

    Args:
        weight_bits: (int) The number of bits to quantize to (e.g., 4).
        per_channel: (bool) A flag indicating whether quantization is
            applied per-channel (`True`) or per-tensor (`False`).
            Defaults to `False`.
        symmetric: (bool) A flag indicating whether symmetric (`True`) or
            asymmetric (`False`) quantization is used. Defaults to `False`.
        group_size: (int) The size of weight groups for quantization. A
            value of -1 indicates that grouping is not used.
            Defaults to -1.
    """

    def __init__(
        self, weight_bits, per_channel=True, symmetric=False, group_size=-1
    ):
        self.weight_bits = weight_bits
        self.maxq = ops.cast(
            ops.subtract(ops.power(2, weight_bits), 1), "float32"
        )
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.group_size = group_size

        # These are now determined later by `find_params`
        self.scale = None
        self.zero = None

    def find_params(self, input_tensor, weight=False):
        """Finds quantization parameters (scale and zero) for a given tensor."""

        if input_tensor is None:
            raise ValueError("Input tensor 'input_tensor' cannot be None.")

        # For weights, we typically expect at least a 2D tensor.
        if weight and len(input_tensor.shape) < 2:
            raise ValueError(
                f"Input weight tensor 'input_tensor' must have a rank of at "
                f"least 2, but got rank {len(input_tensor.shape)}."
            )

        if ops.size(input_tensor) == 0:
            raise ValueError("Input tensor 'input_tensor' cannot be empty.")

        original_shape = input_tensor.shape

        if self.per_channel:
            if weight:
                if self.group_size != -1:
                    input_reshaped = ops.reshape(
                        input_tensor, [-1, self.group_size]
                    )
                else:
                    input_reshaped = ops.reshape(
                        input_tensor, [original_shape[0], -1]
                    )
        else:  # per-tensor
            input_reshaped = ops.reshape(input_tensor, [1, -1])

        # Find min/max values
        min_values = ops.min(input_reshaped, axis=1)
        max_values = ops.max(input_reshaped, axis=1)

        # Apply symmetric quantization logic if enabled
        if self.symmetric:
            max_values = ops.maximum(ops.abs(min_values), max_values)
            min_values = ops.where(
                ops.less(min_values, 0), ops.negative(max_values), min_values
            )

        # Ensure range is not zero to avoid division errors
        zero_range = ops.equal(min_values, max_values)
        min_values = ops.where(
            zero_range, ops.subtract(min_values, 1), min_values
        )
        max_values = ops.where(zero_range, ops.add(max_values, 1), max_values)

        # Calculate scale and zero-point
        self.scale = ops.divide(ops.subtract(max_values, min_values), self.maxq)
        if self.symmetric:
            self.zero = ops.full_like(
                self.scale, ops.divide(ops.add(self.maxq, 1), 2)
            )
        else:
            self.zero = ops.round(
                ops.divide(ops.negative(min_values), self.scale)
            )

        # Ensure scale is non-zero
        self.scale = ops.where(ops.less_equal(self.scale, 0), 1e-8, self.scale)

        if weight:
            # Per-channel, non-grouped case: simple reshape is correct.
            if self.per_channel and self.group_size == -1:
                self.scale = ops.reshape(self.scale, [-1, 1])
                self.zero = ops.reshape(self.zero, [-1, 1])
            elif not self.per_channel:
                num_rows = original_shape[0]
                self.scale = ops.tile(
                    ops.reshape(self.scale, (1, 1)), (num_rows, 1)
                )
                self.zero = ops.tile(
                    ops.reshape(self.zero, (1, 1)), (num_rows, 1)
                )
        if self.per_channel:
            self.scale = ops.reshape(self.scale, [-1, 1])
            self.zero = ops.reshape(self.zero, [-1, 1])

    def ready(self):
        """Checks if the quantization parameters have been computed."""
        return self.scale is not None and self.zero is not None
