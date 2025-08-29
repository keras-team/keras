from keras.src import ops
from keras.src.quantizers.gptq_config import GPTQConfig


def compute_scale_zero(
    x, *, bits, symmetric=False, per_channel=False, group_size=-1, weight=False
):
    """
    Computes the scale and zero-point for quantization.

    Args:
        x: KerasTensor. The input tensor to quantize.
        bits: int. The number of bits to quantize to (e.g., 4).
        symmetric: bool. Whether to use symmetric quantization.
        per_channel: bool. Whether to quantize per channel.
        group_size: int. The group size for quantization.
        weight: bool. Whether the input tensor is a weight tensor.
    """
    if x is None:
        raise ValueError(f"Input tensor {x} cannot be None.")

    # For weights, we typically expect at least a 2D tensor.
    if weight and len(x.shape) < 2:
        raise ValueError(
            f"Input weight tensor {x} must have a rank of at "
            f"least 2, but got rank {len(x.shape)}."
        )

    if ops.size(x) == 0:
        raise ValueError("Input tensor 'x' cannot be empty.")

    original_shape = x.shape

    if per_channel:
        if weight:
            if group_size != -1:
                input_reshaped = ops.reshape(x, [-1, group_size])
            else:
                input_reshaped = ops.reshape(x, [original_shape[0], -1])
    else:  # per-tensor
        input_reshaped = ops.reshape(x, [1, -1])

    # Find min/max values
    min_values = ops.min(input_reshaped, axis=1)
    max_values = ops.max(input_reshaped, axis=1)

    # Apply symmetric quantization logic if enabled
    if symmetric:
        max_values = ops.maximum(ops.abs(min_values), max_values)
        min_values = ops.where(
            ops.less(min_values, 0), ops.negative(max_values), min_values
        )

    # Ensure range is not zero to avoid division errors
    zero_range = ops.equal(min_values, max_values)
    min_values = ops.where(zero_range, ops.subtract(min_values, 1), min_values)
    max_values = ops.where(zero_range, ops.add(max_values, 1), max_values)

    maxq = ops.cast(ops.subtract(ops.power(2, bits), 1), "float32")

    # Calculate scale and zero-point
    scale = ops.divide(ops.subtract(max_values, min_values), maxq)
    if symmetric:
        zero = ops.full_like(scale, ops.divide(ops.add(maxq, 1), 2))
    else:
        zero = ops.round(ops.divide(ops.negative(min_values), scale))

    # Ensure scale is non-zero
    scale = ops.where(ops.less_equal(scale, 0), 1e-8, scale)

    if weight:
        # Per-channel, non-grouped case: simple reshape is correct.
        if per_channel and group_size == -1:
            scale = ops.reshape(scale, [-1, 1])
            zero = ops.reshape(zero, [-1, 1])
        elif not per_channel:
            num_rows = original_shape[0]
            scale = ops.tile(ops.reshape(scale, (1, 1)), (num_rows, 1))
            zero = ops.tile(ops.reshape(zero, (1, 1)), (num_rows, 1))
    if per_channel:
        scale = ops.reshape(scale, [-1, 1])
        zero = ops.reshape(zero, [-1, 1])

    return scale, zero, maxq


def quantize(input_tensor, scale, zero, maxq):
    """Quantize a float tensor into discrete levels [0, maxq] using
    per-tensor/per-channel/grouped scaling.

    Returns `q` (same dtype as inputs/scales; float is fine) where values are in
    [0, maxq].

    Args:
        input_tensor: KerasTensor. The input tensor to quantize.
        scale: KerasTensor. The scale tensor for quantization.
        zero: KerasTensor. The zero tensor for quantization.
        maxq: KerasTensor. The maximum quantization value.

    Returns:
        KerasTensor. The quantized tensor.
    """
    # Guard against divide-by-zero
    epsilon = ops.cast(1e-8, dtype=scale.dtype)
    safe_scale = ops.where(ops.equal(scale, 0), epsilon, scale)

    quantized_tensor = ops.round(
        ops.add(ops.divide(input_tensor, safe_scale), zero)
    )
    quantized_tensor = ops.clip(quantized_tensor, 0, maxq)
    return quantized_tensor


def dequantize(input_tensor, scale, zero):
    """
    Dequantizes a quantized tensor using the provided scale and zero tensors.

    Args:
        input_tensor: KerasTensor. The quantized tensor to dequantize.
        scale: KerasTensor. The scale tensor for dequantization.
        zero: KerasTensor. The zero tensor for dequantization.

    Returns:
        KerasTensor. The dequantized tensor.
    """
    return ops.multiply(scale, (input_tensor - zero))


class GPTQQuantizer:
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

    def __init__(self, config=GPTQConfig(tokenizer=None, dataset=None)):
        self.weight_bits = config.weight_bits
        self.per_channel = config.per_channel
        self.symmetric = config.symmetric
        self.group_size = config.group_size

        # These are now determined later by `find_params`
        self.scale = None
        self.zero = None
        self.maxq = None

    def find_params(self, input_tensor, weight=False):
        """Finds quantization parameters (scale and zero) for a given tensor."""
        self.scale, self.zero, self.maxq = compute_scale_zero(
            input_tensor,
            bits=self.weight_bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            group_size=self.group_size,
            weight=weight,
        )
        return self.scale, self.zero, self.maxq

    def ready(self):
        """Checks if the quantization parameters have been computed."""
        return (
            self.scale is not None
            and self.zero is not None
            and self.maxq is not None
        )
