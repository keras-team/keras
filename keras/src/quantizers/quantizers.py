import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import standardize_axis_for_numpy
from keras.src.ops.operation import Operation
from keras.src.quantizers.gptq_config import GPTQConfig

"""Int8-related classes and methods"""


@keras_export(["keras.Quantizer", "keras.quantizers.Quantizer"])
class Quantizer:
    def __init__(self, output_dtype="int8"):
        self.output_dtype = output_dtype

    def __call__(self, x):
        """Compute a quantized output from an input tensor."""
        return x

    @classmethod
    def from_config(cls, config):
        """Creates a quantizer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same quantizer from the config
        dictionary.

        This method is used by Keras `model_to_estimator`, saving and
        loading models to HDF5 formats, Keras model cloning, some visualization
        utilities, and exporting models to and from JSON.

        Args:
            config: A Python dictionary, typically the output of get_config.

        Returns:
            A quantizer instance.
        """
        return cls(**config)

    def get_config(self):
        """Returns the config of the quantizer.

        A quantizer config is a Python dictionary (serializable)
        containing all configuration parameters of the quantizer.
        The same quantizer can be reinstantiated later
        (without any saved state) from this configuration.

        This method is optional if you are just training and executing models,
        exporting to and from SavedModels, or using weight checkpoints.

        This method is required for Keras `model_to_estimator`, saving and
        loading models to HDF5 formats, Keras model cloning, some visualization
        utilities, and exporting models to and from JSON.

        Returns:
            Python dictionary.
        """
        raise NotImplementedError(f"{self} does not implement get_config()")


@keras_export("keras.quantizers.abs_max_quantize")
def abs_max_quantize(
    inputs,
    axis,
    value_range=(-127, 127),
    dtype="int8",
    epsilon=backend.epsilon(),
    to_numpy=False,
):
    """
    Quantizes the input tensor using the absolute maximum quantization scheme.

    Args:
        inputs: Input tensor to quantize.
        axis: Axis along which to compute the quantization range.
        value_range: Tuple of the minimum and maximum values of the quantization
            range.
        dtype: Data type of the quantized output.
        epsilon: Small value to avoid division by zero.
        to_numpy: Whether to perform the quantization in numpy. This performs
            the computation on the host CPU and can be useful for saving memory
            on the device. If False, the computation is performed on the device.

    Returns:
        A tuple of the quantized tensor and the scale.
    """
    if to_numpy:
        # Save memory on the device using numpy
        original_dtype = backend.standardize_dtype(inputs.dtype)
        inputs = ops.convert_to_numpy(inputs)
        axis = standardize_axis_for_numpy(axis)
        scale = np.divide(
            value_range[1],
            np.add(np.max(np.abs(inputs), axis=axis, keepdims=True), epsilon),
        )
        outputs = np.multiply(inputs, scale)
        outputs = np.clip(np.round(outputs), value_range[0], value_range[1])
        outputs = outputs.astype(dtype)
        return ops.convert_to_tensor(outputs), ops.convert_to_tensor(
            scale, dtype=original_dtype
        )

    inputs = ops.convert_to_tensor(inputs)
    scale = ops.divide(
        value_range[1],
        ops.add(ops.max(ops.abs(inputs), axis=axis, keepdims=True), epsilon),
    )
    scale = ops.cast(scale, backend.standardize_dtype(inputs.dtype))
    outputs = ops.multiply(inputs, scale)
    outputs = ops.clip(ops.round(outputs), value_range[0], value_range[1])
    outputs = ops.cast(outputs, dtype)
    return outputs, scale


@keras_export("keras.quantizers.AbsMaxQuantizer")
class AbsMaxQuantizer(Quantizer):
    def __init__(
        self,
        axis=None,  # Deprecated, provide axis in __call__ instead.
        value_range=(-127, 127),
        epsilon=backend.epsilon(),
        output_dtype="int8",
    ):
        Quantizer.__init__(self, output_dtype=output_dtype)
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            self.axis = tuple(axis)
        else:
            self.axis = None
        self.value_range = value_range
        self.epsilon = epsilon
        if output_dtype == "int8":
            if value_range[0] < -128 or value_range[1] > 127:
                raise ValueError(
                    f"Quantizer with output_dtype='int8' requires value_range "
                    f"to be within the interval [-128, 127]. Received: "
                    f"value_range={value_range}"
                )

    def __call__(self, x, axis=None, to_numpy=False):
        """
        Quantizes the input tensor.

        Args:
            x: Input tensor to quantize.
            axis: Axis along which to compute the quantization range. If None,
                uses the axis specified in the constructor. If None and no axis
                was specified in the constructor, defaults to -1.
            to_numpy: Whether to perform the quantization in numpy. This
                performs the computation on the host CPU and can be useful for
                saving memory on the device. If False, the computation is
                performed on the device.

        Returns:
            A tuple of the quantized tensor and the scale.
        """
        if axis is None:
            axis = self.axis
        if axis is None:
            # Default to -1 if no axis is specified
            axis = -1
        quantized_x, scale = abs_max_quantize(
            x,
            axis,
            self.value_range,
            self.output_dtype,
            self.epsilon,
            to_numpy,
        )
        return quantized_x, scale

    def get_config(self):
        config = {
            "value_range": self.value_range,
            "epsilon": self.epsilon,
            "output_dtype": self.output_dtype,
        }
        if self.axis is not None:
            config["axis"] = self.axis
        return config


def adjust_and_nudge(min_range, max_range, num_bits, narrow_range):
    """Adjusts and nudges the quantization range for better accuracy."""
    # Use higher precision for the computation.
    compute_dtype = backend.result_type(min_range.dtype, "float32")
    min_range = ops.cast(min_range, compute_dtype)
    max_range = ops.cast(max_range, compute_dtype)

    quant_max = (1 << num_bits) - 1
    quant_min = 0 if not narrow_range else 1
    diff_range = ops.subtract(max_range, min_range)

    # Calculate the scale and ensure it's positive
    scale = ops.divide(diff_range, quant_max - quant_min)

    # Re-calculate the inverse to avoid loss of precision
    inv_scale = ops.divide(quant_max - quant_min, diff_range)

    # Calculate the zero point from the min range
    zero_point_from_min = quant_min - ops.divide(min_range, scale)

    # Ensure zero point is within valid range [0, quant_max]
    zero_point = ops.clip(zero_point_from_min, quant_min, quant_max)

    # Nudge zero point if it's very close to an integer
    nudged_zero_point = ops.round(zero_point)

    # Calculate nudged limits
    nudged_min = ops.multiply(ops.subtract(quant_min, nudged_zero_point), scale)
    nudged_max = ops.multiply(ops.subtract(quant_max, nudged_zero_point), scale)

    return nudged_min, nudged_max, scale, inv_scale


class FakeQuantWithMinMaxVars(Operation):
    def __init__(self, num_bits=8, narrow_range=False, axis=None):
        super().__init__()
        self.num_bits = num_bits
        self.narrow_range = narrow_range
        self.axis = axis

    def call(self, inputs, min_vals, max_vals):
        return fake_quant_with_min_max_vars(
            inputs,
            min_vals,
            max_vals,
            num_bits=self.num_bits,
            narrow_range=self.narrow_range,
            axis=self.axis,
        )

    def compute_output_spec(self, inputs, min_vals, max_vals):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_export("keras.quantizers.fake_quant_with_min_max_vars")
def fake_quant_with_min_max_vars(
    inputs,
    min_vals,
    max_vals,
    num_bits=8,
    narrow_range=False,
    axis=None,
):
    """Perform per-tensor or per-channel fake quantization.

    `[min_vals, max_vals]` define the clamping range for the `inputs`.

    The `inputs` are quantized into the quantization range:
    - `[0, 2^num_bits - 1]` when `narrow_range=False`
    - `[1, 2^num_bits - 1]` when `narrow_range=True`

    After quantization, the values are dequantized and output as floats within
    the `[min_vals, max_vals]` interval.

    This operation supports gradient computation, allowing `min_vals` and
    `max_vals` to be trained.

    Args:
        inputs: Input Keras tensor of float dtype.
        min_vals: A global minimum scalar or a per-channel minimum tensor.
        max_vals: A global maximum scalar or a per-channel maximum tensor.
        num_bits: Quantization bit width (e.g., `8` for int8). Defaults to `8`.
        narrow_range: Whether to use narrow quantization range. Defaults to
            `False`.
        axis: Axis along which to perform per-channel quantization. If `None`,
              per-tensor quantization is performed. Defaults to `None`.


    Returns:
        Tensor: A Keras tensor with fake quantization applied.
    """
    if any_symbolic_tensors((inputs,)):
        return FakeQuantWithMinMaxVars().symbolic_call(
            inputs, min_vals, max_vals
        )

    inputs = ops.convert_to_tensor(inputs)
    min_vals = ops.convert_to_tensor(min_vals)
    max_vals = ops.convert_to_tensor(max_vals)
    num_bits = int(num_bits)

    if axis is not None:
        axis = canonicalize_axis(axis, inputs.ndim)

    # Shortcut for TensorFlow backend by using `tf.quantization.fake_quant_*`
    # apis. This is necessary to be recognizable for the TFLite converter.
    if backend.backend() == "tensorflow":
        import tensorflow as tf

        # `tf.quantization.fake_quant_*` only supports float32.
        dtype = backend.standardize_dtype(inputs.dtype)
        if axis is None:
            outputs = tf.quantization.fake_quant_with_min_max_vars(
                ops.cast(inputs, "float32"),
                ops.cast(ops.reshape(min_vals, ()), "float32"),
                ops.cast(ops.reshape(max_vals, ()), "float32"),
                num_bits=num_bits,
                narrow_range=narrow_range,
            )
            return ops.cast(outputs, dtype=dtype)
        else:
            # `tf.quantization.fake_quant_with_min_max_vars_per_channel` only
            # supports the last channel for the per-channel quantization. We
            # use `ops.swapaxes` for the pre- and post-processing.
            last_axis = inputs.ndim - 1
            inputs = ops.swapaxes(inputs, axis, last_axis)
            outputs = tf.quantization.fake_quant_with_min_max_vars_per_channel(
                ops.cast(inputs, "float32"),
                ops.cast(min_vals, "float32"),
                ops.cast(max_vals, "float32"),
                num_bits=num_bits,
                narrow_range=narrow_range,
            )
            outputs = ops.cast(outputs, dtype=dtype)
            return ops.swapaxes(outputs, last_axis, axis)

    @ops.custom_gradient
    def _fake_quant_with_min_max_vars_per_channel(x, min_val, max_val):
        dtype = backend.standardize_dtype(x.dtype)

        # Calculate quantization parameters for all channels at once
        nudged_min, nudged_max, scale, inv_scale = adjust_and_nudge(
            min_val, max_val, num_bits, narrow_range
        )

        quant_zero = ops.floor(
            ops.add(ops.multiply(-nudged_min, inv_scale), 0.5)
        )
        x_clamped = ops.clip(
            ops.cast(x, nudged_min.dtype), nudged_min, nudged_max
        )
        x_clamped_shifted = ops.subtract(x_clamped, nudged_min)
        result = ops.multiply(
            ops.floor(
                ops.add(
                    ops.subtract(
                        ops.multiply(x_clamped_shifted, inv_scale), quant_zero
                    ),
                    0.5,
                )
            ),
            scale,
        )
        result = ops.cast(result, dtype=dtype)

        # Create gradient mask for all channels
        masks = ops.logical_and(
            ops.greater_equal(x, nudged_min), ops.less_equal(x, nudged_max)
        )

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args

            # Gradient for x
            dx = ops.where(masks, upstream, 0.0)
            axes = [i for i in range(len(dx.shape)) if i != axis]

            # Gradient for min_val
            # When x is clipped to min, the gradient flows to min_val
            min_mask = ops.less_equal(x, nudged_min)
            grad_min = ops.where(min_mask, upstream, 0.0)
            if axis is not None:
                grad_min = ops.sum(grad_min, axis=axes)
            else:
                grad_min = ops.sum(grad_min)
            grad_min = ops.reshape(grad_min, ops.shape(min_val))

            # Gradient for max_val
            # When x is clipped to max, the gradient flows to max_val
            max_mask = ops.greater_equal(x, nudged_max)
            grad_max = ops.where(max_mask, upstream, 0.0)
            if axis is not None:
                grad_max = ops.sum(grad_max, axis=axes)
            else:
                grad_max = ops.sum(grad_max)
            grad_max = ops.reshape(grad_max, ops.shape(max_val))

            return dx, grad_min, grad_max

        return result, grad

    return _fake_quant_with_min_max_vars_per_channel(inputs, min_vals, max_vals)


"""Float8-related methods"""


@keras_export("keras.quantizers.compute_float8_scale")
def compute_float8_scale(amax, scale, dtype_max, margin=0):
    # The algorithm for computing the new scale is sourced from
    # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.update_fp8_metas
    # wherein the `original_scale` corresponds to the reciprocal of the
    # `scale` passed in this function.
    scale = ops.reciprocal(scale)
    sf = ops.divide(ops.divide(dtype_max, amax), 2**margin)
    sf = ops.where(amax > 0.0, sf, scale)
    sf = ops.where(ops.isfinite(amax), sf, scale)
    return ops.reciprocal(sf)


@keras_export("keras.quantizers.compute_float8_amax_history")
def compute_float8_amax_history(x, amax_history):
    amax_update = ops.cast(ops.max(ops.abs(x)), amax_history.dtype)
    new_amax_history = ops.scatter_update(
        ops.roll(amax_history, shift=-1),
        [[0]],
        ops.reshape(amax_update, [1]),
    )
    return new_amax_history


@keras_export("keras.quantizers.quantize_and_dequantize")
def quantize_and_dequantize(inputs, scale, quantized_dtype, compute_dtype):
    # Quantize
    quantized_dtype_max = ops.cast(
        float(ml_dtypes.finfo(quantized_dtype).max), compute_dtype
    )
    x = ops.divide(inputs, ops.cast(scale, compute_dtype))
    x = ops.clip(x, -quantized_dtype_max, quantized_dtype_max)
    x = ops.cast(x, quantized_dtype)

    # Dequantize
    x = ops.multiply(ops.cast(x, compute_dtype), ops.cast(scale, compute_dtype))
    return x


@keras_export("keras.quantizers.pack_int4")
def pack_int4(arr, axis=0, dtype="int8"):
    """Pack an int4 tensor into an int8 tensor with packed nibbles.

    The input values must already be int8 in the signed range `[-8, 7]` and
    represent the desired int4 values. Packing is performed along the specified
    axis (default is 0).

    For every two consecutive rows, the **low nibble** of the output byte
    stores the value from the first row, and the **high nibble** stores
    the value from the second row.

    Args:
        arr: An `int8` or `uint8` tensor containing int4 values in the range
            `[-8, 7]`.
        axis: The axis along which to pack the tensor. Defaults to 0.
        dtype: The data type of the input and packed tensor. Can be
            `"int8"` or `"uint8"`. Defaults to `"int8"`.

    Returns:
        tuple: A tuple `(packed, packed_shape, orig_rows)` where `packed` is
            the packed int8 tensor with int4 values stored in nibbles,
            `packed_shape` is the shape of the packed tensor, and `orig_rows`
            is the original (unpacked) row count prior to any padding that may
            have been inserted when an odd number of rows is supplied.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_int4, unpack_int4

    # Example with axis=0
    # Original array has shape (3, 2)
    >>> original_array = np.array([[-3, 7], [2, -8], [1, 0]], dtype=np.int8)

    # Pack the array along axis 0. Since the length of axis 0 (3) is
    # odd, it will be padded to a length of 4. The packed array will
    # have a shape of (ceil(3/2), 2) = (2, 2).
    >>> packed, packed_shape, orig_len = pack_int4(original_array, axis=0)
    >>> print("Packed array:\n", packed)
    Packed array:
    [[  45 -121]
    [   1    0]]

    # Now, unpack the array back to its original form
    >>> unpacked = unpack_int4(packed, orig_len, axis=0)
    >>> print("Unpacked array:\n", unpacked)
    Unpacked array:
    [[-3  7]
    [ 2 -8]
    [ 1  0]]
    >>> np.allclose(original_array, unpacked)
    True

    # Example with axis=1
    # Original array has shape (2, 3)
    >>> original_array = np.array([[-3, 7, 2], [-8, 1, 0]], dtype=np.int8)

    # Pack along axis 1. Length of axis 1 (3) is padded to 4.
    # The new shape is (2, ceil(3/2)) = (2, 2).
    >>> packed, packed_shape, orig_len = pack_int4(original_array, axis=1)
    >>> print("Packed array:\n", packed)
    Packed array:
    [[ 125   2]
    [  24   0]]

    # Unpack the array
    >>> unpacked = unpack_int4(packed, orig_len, axis=1)
    >>> print("Unpacked array:\n", unpacked)
    Unpacked array:
    [[-3  7  2]
    [-8  1  0]]
    >>> np.allclose(original_array, unpacked)
    True
    ```
    """
    if dtype not in ("int8", "uint8"):
        raise ValueError(
            f"Expected dtype to be 'int8' or 'uint8', but got '{dtype}'."
        )
    if backend.standardize_dtype(arr.dtype) != dtype:
        raise TypeError(
            f"Expected {dtype} tensor for packing, got "
            f"{backend.standardize_dtype(arr.dtype)}."
        )

    rank = getattr(arr.shape, "rank", None) or len(arr.shape)

    if axis < 0:
        axis += rank

    # 1. Bring `axis` to the front.
    perm = [axis] + [i for i in range(rank) if i != axis]
    inv_perm = [perm.index(i) for i in range(rank)]
    transposed = ops.transpose(arr, perm)

    # 2. Pad to even length.
    rows = ops.shape(transposed)[0]
    needs_pad = ops.equal(ops.mod(rows, 2), 1)

    # Always append one zero row so the tensor shape is static for JAX. If no
    # padding is actually needed, we'll slice it away later.
    zero_row = transposed[:1, ...] * 0  # same dtype/shape (1, ...)
    padded_full = ops.concatenate([transposed, zero_row], axis=0)

    # Number of valid rows after (possible) padding:
    # rows + (1 if needs_pad else 0)
    rows_packed = rows + ops.cast(needs_pad, "int32")

    # Slice to keep only the valid rows. This keeps the shape rank static while
    # allowing the row count to be dynamic.
    padded = padded_full[:rows_packed, ...]

    # 3-4. Group in pairs and pack.
    low = padded[::2, ...]
    high = padded[1::2, ...]

    mask = ops.array(0x0F, dtype=dtype)
    low_u = ops.bitwise_and(low, mask)
    high_u = ops.bitwise_and(high, mask)

    packed = ops.bitwise_or(low_u, ops.left_shift(high_u, 4))
    packed = ops.cast(packed, dtype)

    # 5-6. Restore shape.
    packed = ops.transpose(packed, inv_perm)  # back to original order
    orig_len = rows  # number of slices before padding
    return packed, ops.shape(packed), orig_len


@keras_export("keras.quantizers.unpack_int4")
def unpack_int4(packed, orig_len, axis=0, dtype="int8"):
    """Unpack a packed int4 back to an int8 tensor in the range [-8, 7].

    This function reverses the packing performed by `pack_int4`, restoring
    the original int8 tensor (values in the range [-8, 7]) from a packed int8
    tensor where each element contains two int4 values (one in the lower nibble,
    one in the upper nibble).

    The function restores the original axis order and removes any
    padding that was added during packing.

    Args:
        packed: An int8 tensor containing packed int4 values along the
            specified axis. Each int8 value encodes two int4 values.
        orig_len: The original (unpadded) length of the axis that was
            packed. This is used to remove any padding that may have
            been added during packing to ensure an even number of rows.
        axis: The axis along which the tensor was packed. Defaults to 0.
        dtype: The data type of the input and unpacked tensor. Can be
            `"int8"` or `"uint8"`. Defaults to `"int8"`.

    Returns:
        unpacked: An int8 tensor with the same shape as the original
            (unpacked) tensor, with values in the range [-8, 7].

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_int4, unpack_int4

    # Example with axis=0
    # Original array has shape (3, 2)
    >>> original_array = np.array([[-3, 7], [2, -8], [1, 0]], dtype=np.int8)

    # Pack the array along axis 0. Since the length of axis 0 (3) is
    # odd, it will be padded to a length of 4. The packed array will
    # have a shape of (ceil(3/2), 2) = (2, 2).
    >>> packed, packed_shape, orig_len = pack_int4(original_array, axis=0)
    >>> print("Packed array:\n", packed)
    Packed array:
    [[  45 -121]
    [   1    0]]

    # Now, unpack the array back to its original form
    >>> unpacked = unpack_int4(packed, orig_len, axis=0)
    >>> print("Unpacked array:\n", unpacked)
    Unpacked array:
    [[-3  7]
    [ 2 -8]
    [ 1  0]]
    >>> np.allclose(original_array, unpacked)
    True

    # Example with axis=1
    # Original array has shape (2, 3)
    >>> original_array = np.array([[-3, 7, 2], [-8, 1, 0]], dtype=np.int8)

    # Pack along axis 1. Length of axis 1 (3) is padded to 4.
    # The new shape is (2, ceil(3/2)) = (2, 2).
    >>> packed, packed_shape, orig_len = pack_int4(original_array, axis=1)
    >>> print("Packed array:\n", packed)
    Packed array:
    [[ 125   2]
    [  24   0]]

    # Unpack the array
    >>> unpacked = unpack_int4(packed, orig_len, axis=1)
    >>> print("Unpacked array:\n", unpacked)
    Unpacked array:
    [[-3  7  2]
    [-8  1  0]]
    >>> np.allclose(original_array, unpacked)
    True
    ```
    """
    if dtype not in ("int8", "uint8"):
        raise ValueError(
            f"Expected dtype to be 'int8' or 'uint8', but got '{dtype}'."
        )

    if backend.standardize_dtype(packed.dtype) not in ("int8", "uint8"):
        raise TypeError(
            f"Expected int8 or uint8 tensor for unpacking, got {packed.dtype}"
        )

    def to_signed(x):
        """Converts unpacked nibbles [0, 15] to signed int4 [-8, 7]."""
        dtype_x = backend.standardize_dtype(x.dtype)
        eight = ops.cast(8, dtype_x)
        sixteen = ops.cast(16, dtype_x)
        return ops.where(x < eight, x, x - sixteen)

    rank = getattr(packed.shape, "rank", None) or len(packed.shape)
    if axis < 0:
        axis += rank

    # Fast path for the most common case in Dense layers
    if axis == 0 and rank == 2:
        # The result of the bitwise op is a wider dtype (e.g., int32).
        mask = ops.array(0x0F, dtype=packed.dtype)
        low_unpacked = ops.bitwise_and(packed, mask)
        high_unpacked = ops.bitwise_and(ops.right_shift(packed, 4), mask)

        if dtype == "int8":
            low_unpacked = to_signed(low_unpacked)
            high_unpacked = to_signed(high_unpacked)

        low_final = ops.cast(low_unpacked, dtype)
        high_final = ops.cast(high_unpacked, dtype)

        # Interleave and reshape
        stacked = ops.stack([low_final, high_final], axis=1)
        unpacked = ops.reshape(stacked, (-1,) + tuple(ops.shape(packed)[1:]))

        # Remove padding and return
        return unpacked[:orig_len, ...]

    # General case
    perm = [axis] + [i for i in range(rank) if i != axis]
    inv_perm = [perm.index(i) for i in range(rank)]
    transposed = ops.transpose(packed, perm)

    # 1. Split nibbles.
    mask = ops.array(0x0F, dtype=packed.dtype)
    low = ops.bitwise_and(transposed, mask)
    high = ops.bitwise_and(ops.right_shift(transposed, 4), mask)

    # 2. Conditionally convert to signed.
    if dtype == "int8":
        low = to_signed(low)
        high = to_signed(high)

    low = ops.cast(low, dtype)
    high = ops.cast(high, dtype)

    # 3. Interleave and reshape.
    stacked = ops.stack([low, high], axis=1)
    unpacked = ops.reshape(stacked, (-1,) + tuple(ops.shape(transposed)[1:]))

    # 4. Remove padding and restore original layout.
    unpacked = unpacked[:orig_len, ...]
    unpacked = ops.transpose(unpacked, inv_perm)

    return unpacked


class GPTQQuantizer(Quantizer):
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
        self,
        config=GPTQConfig(tokenizer=None, dataset=None),
        compute_dtype="float32",
    ):
        Quantizer.__init__(self)
        self.weight_bits = config.weight_bits
        self.per_channel = config.per_channel
        self.symmetric = config.symmetric
        self.group_size = config.group_size
        self.compute_dtype = compute_dtype

        # These are now determined later by `find_params`
        self.scale = None
        self.zero = None
        self.maxq = None

    def find_params(self, input_tensor, weight=True):
        """Finds quantization parameters (scale and zero) for a given tensor."""
        self.scale, self.zero, self.maxq = compute_quantization_parameters(
            input_tensor,
            bits=self.weight_bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            group_size=self.group_size,
            weight=weight,
            compute_dtype=self.compute_dtype,
        )
        return self.scale, self.zero, self.maxq

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_bits": self.weight_bits,
                "per_channel": self.per_channel,
                "symmetric": self.symmetric,
                "group_size": self.group_size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        gptq = GPTQConfig(
            tokenizer=None,
            dataset=None,
            weight_bits=config["weight_bits"],
            per_channel=config["per_channel"],
            symmetric=config["symmetric"],
            group_size=config["group_size"],
        )
        return cls(gptq)


def compute_quantization_parameters(
    x,
    *,
    bits,
    symmetric=False,
    per_channel=False,
    group_size=-1,
    weight=False,
    compute_dtype="float32",
):
    """
    Computes the scale and zero-point for quantization.

    This function calculates the scale and zero-point required for quantizing
    a given tensor `x` based on the specified parameters. It supports grouped,
    per-channel, per-tensor, symmetric, and asymmetric quantization - along
    with any combinations of these.

    Args:
        x: KerasTensor. The input tensor to quantize.
        bits: int. The number of bits to quantize to (e.g., 4).
        symmetric: bool. Whether to use symmetric quantization.
        per_channel: bool. Whether to quantize per channel.
        group_size: int. The group size for quantization.
        weight: bool. Whether the input tensor is a weight tensor.

    Returns:
        scale: KerasTensor. The scale tensor for quantization.
        zero: KerasTensor. The zero tensor for quantization.
        maxq: scalar. The maximum quantization value.
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

    maxq = ops.cast(ops.subtract(ops.power(2, bits), 1), compute_dtype)

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

    zero = ops.cast(zero, "uint8")

    return scale, zero, maxq


def quantize_with_zero_point(input_tensor, scale, zero, maxq):
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
        ops.add(
            ops.divide(input_tensor, safe_scale), ops.cast(zero, scale.dtype)
        )
    )
    quantized_tensor = ops.clip(quantized_tensor, 0, maxq)
    return quantized_tensor


def dequantize_with_zero_point(input_tensor, scale, zero):
    """
    Dequantizes a quantized tensor using the provided scale and zero tensors.

    Args:
        input_tensor: KerasTensor. The quantized tensor to dequantize.
        scale: KerasTensor. The scale tensor for dequantization.
        zero: KerasTensor. The zero tensor for dequantization.

    Returns:
        KerasTensor. The dequantized tensor.
    """
    return ops.multiply(
        scale, ops.subtract(input_tensor, ops.cast(zero, scale.dtype))
    )


def quantize_with_sz_map(weights_matrix, scale, zero, g_idx, maxq):
    """Quantize the weight matrix from group params.

    This function uses the provided scale and zero tensors to quantize the
    input weights_matrix according to the group indices. It maps each column
    of the weights_matrix to its corresponding group parameters and performs
    the quantization operation.

    Args:
        weights_matrix: 2D tensor of shape [out_features, in_features].
        scale: Per-group scale tensor of shape [out_features, n_groups].
        zero: Per-group zero-point tensor of shape [out_features, n_groups].
        g_idx: Integer tensor of shape [in_features,] mapping each column to
            its group index.
        maxq: Scalar (float) representing the maximum integer quantization
            level (e.g., 2^bits - 1).

    Returns:
        A tensor with the same shape as `weights_matrix` containing the
        quantized weights produced using the provided group parameters.
    """
    groups = ops.cast(g_idx, "int32")
    scale_cols = ops.take(scale, groups, axis=1)  # [out_features, in_features]
    zero_cols = ops.take(zero, groups, axis=1)  # [out_features, in_features]

    # Quantize elementwise, then cast to int
    return quantize_with_zero_point(weights_matrix, scale_cols, zero_cols, maxq)


def dequantize_with_sz_map(weights_matrix, scale, zero, g_idx):
    """Rebuild a dequantized weight matrix from group params.

    This function uses the provided scale and zero tensors to dequantize the
    input weights_matrix according to the group indices. It maps each column
    of the weights_matrix to its corresponding group parameters and performs
    the dequantization operation.

    Args:
        weights_matrix: 2D tensor of shape [out_features, in_features].
        scale: Per-group scale tensor of shape [out_features, n_groups].
        zero: Per-group zero-point tensor of shape [out_features, n_groups].
        g_idx: Integer tensor of shape [in_features,] mapping each column to
            its group index.
        maxq: Scalar (float) representing the maximum integer quantization
            level (e.g., 2^bits - 1).

    Returns:
        A tensor with the same shape as `weights_matrix` containing the
        dequantized weights produced using the provided group parameters.
    """
    # Map group indices to scales and zeros
    groups = ops.cast(g_idx, "int32")
    scales_mapped = ops.take(scale, groups, axis=1)
    zeros_mapped = ops.take(zero, groups, axis=1)
    zeros_mapped = ops.cast(zeros_mapped, scales_mapped.dtype)

    quantized = ops.multiply(
        ops.subtract(weights_matrix, zeros_mapped), scales_mapped
    )

    return quantized
