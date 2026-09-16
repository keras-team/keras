import math

import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common.backend_utils import standardize_axis_for_numpy


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


@keras_export("keras.quantizers.abs_max_quantize_grouped_with_zero_point")
def abs_max_quantize_grouped_with_zero_point(
    inputs,
    block_size,
    value_range=(-8, 7),
    dtype="int8",
    epsilon=backend.epsilon(),
    to_numpy=False,
):
    """Quantizes a 2D tensor using grouped asymmetric quantization with
    zero point.

    Groups are formed along axis 0 (the input/contracting dimension).
    Each group of `block_size` rows gets its own scale factor and zero point
    per column. This is useful for weight distributions that are not centered
    around zero. A group's range always includes zero, so its zero point is
    representable in `value_range` and the grid holds an exact zero.

    Args:
        inputs: Input tensor to quantize. Shape: `(input_dim, output_dim)`.
        block_size: Number of elements per group along axis 0.
        value_range: Tuple of `(min, max)` quantization range.
        dtype: Data type of quantized output.
        epsilon: Small value to avoid division by zero.
        to_numpy: Whether to compute in NumPy, which keeps the weight off
            the accelerator during quantization, rather than in backend
            ops. Both paths apply the same formula; `bfloat16` inputs can
            differ between them at the ulp level.

    Returns:
        A tuple `(quantized_tensor, scale, zero_point)` where:
            - `quantized_tensor`: Same shape as inputs, dtype=`dtype`.
            - `scale`: Shape `(n_groups, output_dim)` where
              `n_groups = ceil(input_dim / block_size)`.
            - `zero_point`: Shape `(n_groups, output_dim)`, dtype=`int8`.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import abs_max_quantize_grouped_with_zero_point
    >>> kernel = np.random.randn(512, 256).astype("float32")
    >>> quantized, scale, zero_point = abs_max_quantize_grouped_with_zero_point(
    ...     kernel, block_size=128, value_range=(-8, 7)
    ... )
    >>> quantized.shape
    (512, 256)
    >>> scale.shape  # 512 / 128 = 4 groups
    (4, 256)
    >>> zero_point.shape
    (4, 256)
    ```
    """
    if to_numpy:
        return _abs_max_quantize_grouped_with_zero_point_numpy(
            inputs, block_size, value_range, dtype, epsilon
        )
    return _abs_max_quantize_grouped_with_zero_point_tensor(
        inputs, block_size, value_range, dtype, epsilon
    )


def _abs_max_quantize_grouped_with_zero_point_numpy(
    inputs, block_size, value_range, dtype, epsilon
):
    """NumPy implementation of grouped asymmetric quantization.

    Uses NumPy for computation to reduce GPU memory usage during
    model quantization.
    """
    original_dtype = backend.standardize_dtype(inputs.dtype)
    inputs = ops.convert_to_numpy(inputs)

    input_dim, output_dim = inputs.shape
    n_groups = math.ceil(input_dim / block_size)
    qmin, qmax = value_range

    # Zero-pad rows so input_dim is divisible by block_size
    padded_input_dim = n_groups * block_size
    if padded_input_dim > input_dim:
        padding = np.zeros(
            (padded_input_dim - input_dim, output_dim), dtype=inputs.dtype
        )
        inputs_padded = np.concatenate([inputs, padding], axis=0)
    else:
        inputs_padded = inputs

    inputs_reshaped = inputs_padded.reshape(n_groups, block_size, output_dim)

    # Per-group min/max, widened to include zero: the zero point then lands
    # inside `[qmin, qmax]` and the grid represents 0 exactly. A group whose
    # values are all one sign would otherwise clip to one end of the range.
    min_val = np.minimum(np.min(inputs_reshaped, axis=1, keepdims=True), 0.0)
    max_val = np.maximum(np.max(inputs_reshaped, axis=1, keepdims=True), 0.0)

    # Scale maps the [min, max] range to [qmin, qmax]; the floor keeps an
    # all-zero group finite when `epsilon` underflows in the input dtype.
    scale = np.divide(np.subtract(max_val, min_val) + epsilon, qmax - qmin)
    scale = np.maximum(scale, ml_dtypes.finfo(scale.dtype).tiny)

    # Zero point shifts the quantized range to include the original zero
    zero_point = np.round(np.divide(-min_val, scale)) + qmin
    zero_point = np.clip(zero_point, qmin, qmax)

    # Quantize: q = round(input / scale) + zero_point
    outputs = np.round(np.divide(inputs_reshaped, scale)) + zero_point
    outputs = np.clip(outputs, qmin, qmax)
    outputs = outputs.astype(dtype)

    # Remove padding and squeeze to (n_groups, output_dim)
    outputs = outputs.reshape(padded_input_dim, output_dim)[:input_dim, :]
    scale = np.squeeze(scale, axis=1)
    zero_point = np.squeeze(zero_point, axis=1).astype("int8")

    return (
        ops.convert_to_tensor(outputs),
        ops.convert_to_tensor(scale, dtype=original_dtype),
        ops.convert_to_tensor(zero_point),
    )


def _abs_max_quantize_grouped_with_zero_point_tensor(
    inputs, block_size, value_range, dtype, epsilon
):
    """Backend-ops implementation of grouped asymmetric quantization.

    The same formula as `_abs_max_quantize_grouped_with_zero_point_numpy`,
    for callers that keep the weight on the accelerator.
    """
    original_dtype = backend.standardize_dtype(inputs.dtype)
    inputs = ops.convert_to_tensor(inputs)

    input_shape = ops.shape(inputs)
    input_dim, output_dim = int(input_shape[0]), int(input_shape[1])
    n_groups = math.ceil(input_dim / block_size)
    qmin, qmax = value_range

    # Zero-pad rows so input_dim is divisible by block_size
    padded_input_dim = n_groups * block_size
    if padded_input_dim > input_dim:
        padding = ops.zeros(
            (padded_input_dim - input_dim, output_dim), dtype=inputs.dtype
        )
        inputs = ops.concatenate([inputs, padding], axis=0)

    inputs_reshaped = ops.reshape(inputs, (n_groups, block_size, output_dim))

    # Per-group min/max, widened to include zero (see the NumPy path).
    min_val = ops.minimum(ops.min(inputs_reshaped, axis=1, keepdims=True), 0.0)
    max_val = ops.maximum(ops.max(inputs_reshaped, axis=1, keepdims=True), 0.0)

    # Scale maps the [min, max] range to [qmin, qmax]; the floor keeps an
    # all-zero group finite when `epsilon` underflows in the input dtype.
    scale = ops.divide(
        ops.add(ops.subtract(max_val, min_val), epsilon), qmax - qmin
    )
    scale = ops.maximum(scale, float(ml_dtypes.finfo(original_dtype).tiny))

    # Zero point shifts the quantized range to include the original zero
    zero_point = ops.add(
        ops.round(ops.divide(ops.negative(min_val), scale)), qmin
    )
    zero_point = ops.clip(zero_point, qmin, qmax)

    # Quantize: q = round(input / scale) + zero_point
    outputs = ops.add(ops.round(ops.divide(inputs_reshaped, scale)), zero_point)
    outputs = ops.cast(ops.clip(outputs, qmin, qmax), dtype)

    # Remove padding and squeeze to (n_groups, output_dim)
    outputs = ops.reshape(outputs, (padded_input_dim, output_dim))
    outputs = outputs[:input_dim, :]
    scale = ops.cast(ops.squeeze(scale, axis=1), original_dtype)
    zero_point = ops.cast(ops.squeeze(zero_point, axis=1), "int8")

    return outputs, scale, zero_point


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

    # Perform packing in numpy. Packing is only called during
    # quantization (not inference), and numpy correctly handles int8
    # overflow in bitwise operations. Some accelerators (e.g. TPU) may
    # produce incorrect results for int8 left_shift that overflows, so
    # using numpy avoids device-specific issues.
    arr_np = ops.convert_to_numpy(arr)
    np_dtype = np.dtype(dtype)

    rank = len(arr_np.shape)
    if axis < 0:
        axis += rank

    # Move the pack axis to the front for uniform handling.
    arr_np = np.moveaxis(arr_np, axis, 0)

    # Pad to even length along the front axis.
    n = arr_np.shape[0]
    if n % 2 == 1:
        pad_shape = (1,) + arr_np.shape[1:]
        arr_np = np.concatenate(
            [arr_np, np.zeros(pad_shape, dtype=arr_np.dtype)], axis=0
        )

    # Group in pairs and pack nibbles.
    low = arr_np[::2]
    high = arr_np[1::2]

    mask = np.array(0x0F, dtype=np_dtype)
    low_u = np.bitwise_and(low.astype(np_dtype), mask)
    high_u = np.bitwise_and(high.astype(np_dtype), mask)

    packed_np = np.bitwise_or(
        low_u, np.left_shift(high_u, np.array(4, dtype=np_dtype))
    )
    packed_np = packed_np.astype(np_dtype)

    # Move the pack axis back to its original position.
    packed_np = np.moveaxis(packed_np, 0, axis)

    packed = ops.convert_to_tensor(packed_np)
    return packed, tuple(packed_np.shape), n


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
        """Converts unpacked nibbles [0, 15] to signed int4 [-8, 7].

        Uses a branchless XOR approach: (x ^ 8) - 8
        This maps: 0->0, 1->1, ..., 7->7, 8->-8, 9->-7, ..., 15->-1
        """
        dtype_x = backend.standardize_dtype(x.dtype)
        eight = ops.cast(8, dtype_x)
        return ops.subtract(ops.bitwise_xor(x, eight), eight)

    rank = getattr(packed.shape, "rank", None) or len(packed.shape)
    if axis < 0:
        axis += rank

    # Fast path for axis==0 (common case in Dense layers)
    if axis == 0 and rank == 2:
        mask = ops.array(0x0F, dtype=packed.dtype)
        low_unpacked = ops.bitwise_and(packed, mask)
        high_unpacked = ops.bitwise_and(ops.right_shift(packed, 4), mask)

        if dtype == "int8":
            low_unpacked = to_signed(low_unpacked)
            high_unpacked = to_signed(high_unpacked)

        low_final = ops.cast(low_unpacked, dtype)
        high_final = ops.cast(high_unpacked, dtype)

        # Interleave along axis 0 and reshape
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


@keras_export("keras.quantizers.pack_int2")
def pack_int2(arr, axis=0, dtype="int8"):
    """Pack an int2 tensor into an int8 tensor with 4 values per byte.

    The input values must already be int8/uint8 representing the desired int2
    values (signed range `[-2, 1]`, or unsigned range `[0, 3]`). Packing is
    performed along the specified axis (default is 0). Four consecutive values
    along the packing axis are stored in a single output byte, from the least
    significant 2-bit field to the most significant one.

    This mirrors the design of `pack_int4` (padding + original-length trim),
    but achieves a 4x rather than 2x storage reduction, which is what makes
    2-bit storage worthwhile relative to a plain uint8 tensor.

    Args:
        arr: An `int8` or `uint8` tensor containing int2 values in the range
            `[-2, 1]` (signed) or `[0, 3]` (unsigned).
        axis: The axis along which to pack the tensor. Defaults to 0.
        dtype: The data type of the input and packed tensor. Can be
            `"int8"` or `"uint8"`. Defaults to `"int8"`.

    Returns:
        tuple: A tuple `(packed, packed_shape, orig_len)` where `packed` is
            the packed tensor with four int2 values per byte, `packed_shape`
            is the shape of the packed tensor, and `orig_len` is the original
            (unpacked) length along `axis` prior to any padding that was
            inserted to reach a multiple of four.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_int2, unpack_int2

    # Example with axis=0
    # Original array has shape (5, 2)
    >>> original_array = np.array(
    ...     [[-2, 1], [0, -1], [1, -2], [0, 1], [-1, 0]], dtype=np.int8
    ... )

    # Pack the array along axis 0. Since the length of axis 0 (5) is
    # not a multiple of 4, it will be padded to a length of 8. The packed
    # array will have a shape of (ceil(5/4), 2) = (2, 2).
    >>> packed, packed_shape, orig_len = pack_int2(original_array, axis=0)
    >>> print("Packed array:\n", packed)
    Packed array:
    [[ 18 109]
     [  3   0]]

    # Now, unpack the array back to its original form
    >>> unpacked = unpack_int2(packed, orig_len, axis=0)
    >>> print("Unpacked array:\n", unpacked)
    Unpacked array:
    [[-2  1]
     [ 0 -1]
     [ 1 -2]
     [ 0  1]
     [-1  0]]
    >>> np.allclose(original_array, unpacked)
    True

    # Example with axis=1
    # Original array has shape (2, 5)
    >>> original_array = np.array(
    ...     [[-2, 1, 0, -1, 1], [-2, 0, 1, -1, 0]], dtype=np.int8
    ... )

    # Pack along axis 1. Length of axis 1 (5) is padded to 8.
    # The new shape is (2, ceil(5/4)) = (2, 2).
    >>> packed, packed_shape, orig_len = pack_int2(original_array, axis=1)
    >>> print("Packed array:\n", packed)
    Packed array:
    [[-58   1]
     [-46   0]]

    # Unpack the array
    >>> unpacked = unpack_int2(packed, orig_len, axis=1)
    >>> print("Unpacked array:\n", unpacked)
    Unpacked array:
    [[-2  1  0 -1  1]
     [-2  0  1 -1  0]]
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

    # Perform packing in numpy for the same reasons as `pack_int4`: it is only
    # called during quantization (not inference), and numpy correctly handles
    # int8 overflow in the bitwise shifts that some accelerators mishandle.
    arr_np = ops.convert_to_numpy(arr)
    np_dtype = np.dtype(dtype)

    rank = len(arr_np.shape)
    if axis < 0:
        axis += rank

    # Move the pack axis to the front for uniform handling.
    arr_np = np.moveaxis(arr_np, axis, 0)

    # Pad to a multiple of four along the front axis.
    n = arr_np.shape[0]
    pad = (-n) % 4
    if pad:
        pad_shape = (pad,) + arr_np.shape[1:]
        arr_np = np.concatenate(
            [arr_np, np.zeros(pad_shape, dtype=arr_np.dtype)], axis=0
        )

    # Group in quadruples and pack four 2-bit fields per byte.
    mask = np.array(0x03, dtype=np_dtype)

    def field(values, shift):
        masked = np.bitwise_and(values.astype(np_dtype), mask)
        return np.left_shift(masked, np.array(shift, dtype=np_dtype))

    packed_np = np.bitwise_or(
        np.bitwise_or(field(arr_np[0::4], 0), field(arr_np[1::4], 2)),
        np.bitwise_or(field(arr_np[2::4], 4), field(arr_np[3::4], 6)),
    )
    packed_np = packed_np.astype(np_dtype)

    # Move the pack axis back to its original position.
    packed_np = np.moveaxis(packed_np, 0, axis)

    packed = ops.convert_to_tensor(packed_np)
    return packed, tuple(packed_np.shape), n


@keras_export("keras.quantizers.unpack_int2")
def unpack_int2(packed, orig_len, axis=0, dtype="int8"):
    """Unpack a packed int2 tensor back to an int8 tensor.

    This reverses `pack_int2`, restoring the original tensor whose values lie in
    `[-2, 1]` (signed) or `[0, 3]` (unsigned) from a packed tensor where each
    byte stores four int2 values. It restores the original axis order and
    removes any padding that was added during packing.

    Args:
        packed: An `int8` or `uint8` tensor with four int2 values per element
            along the specified axis.
        orig_len: The original (unpadded) length of the packed axis. Used to
            trim the padding inserted during packing.
        axis: The axis along which the tensor was packed. Defaults to 0.
        dtype: The data type of the input and unpacked tensor. Can be
            `"int8"` or `"uint8"`. Defaults to `"int8"`.

    Returns:
        unpacked: A tensor with the same shape as the original (unpacked)
            tensor.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_int2, unpack_int2

    # Example with axis=0
    # Original array has shape (5, 2)
    >>> original_array = np.array(
    ...     [[-2, 1], [0, -1], [1, -2], [0, 1], [-1, 0]], dtype=np.int8
    ... )

    # Pack the array along axis 0. Since the length of axis 0 (5) is
    # not a multiple of 4, it will be padded to a length of 8. The packed
    # array will have a shape of (ceil(5/4), 2) = (2, 2).
    >>> packed, packed_shape, orig_len = pack_int2(original_array, axis=0)
    >>> print("Packed array:\n", packed)
    Packed array:
    [[ 18 109]
     [  3   0]]

    # Now, unpack the array back to its original form
    >>> unpacked = unpack_int2(packed, orig_len, axis=0)
    >>> print("Unpacked array:\n", unpacked)
    Unpacked array:
    [[-2  1]
     [ 0 -1]
     [ 1 -2]
     [ 0  1]
     [-1  0]]
    >>> np.allclose(original_array, unpacked)
    True

    # Example with axis=1
    # Original array has shape (2, 5)
    >>> original_array = np.array(
    ...     [[-2, 1, 0, -1, 1], [-2, 0, 1, -1, 0]], dtype=np.int8
    ... )

    # Pack along axis 1. Length of axis 1 (5) is padded to 8.
    # The new shape is (2, ceil(5/4)) = (2, 2).
    >>> packed, packed_shape, orig_len = pack_int2(original_array, axis=1)
    >>> print("Packed array:\n", packed)
    Packed array:
    [[-58   1]
     [-46   0]]

    # Unpack the array
    >>> unpacked = unpack_int2(packed, orig_len, axis=1)
    >>> print("Unpacked array:\n", unpacked)
    Unpacked array:
    [[-2  1  0 -1  1]
     [-2  0  1 -1  0]]
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
        """Converts unpacked 2-bit fields [0, 3] to signed int2 [-2, 1].

        Uses the same branchless XOR approach as `unpack_int4`: (x ^ 2) - 2.
        This maps: 0->0, 1->1, 2->-2, 3->-1.
        """
        dtype_x = backend.standardize_dtype(x.dtype)
        two = ops.cast(2, dtype_x)
        return ops.subtract(ops.bitwise_xor(x, two), two)

    rank = getattr(packed.shape, "rank", None) or len(packed.shape)
    if axis < 0:
        axis += rank

    mask = ops.array(0x03, dtype=packed.dtype)

    def split_fields(x):
        fields = [
            ops.bitwise_and(ops.right_shift(x, shift), mask)
            for shift in (0, 2, 4, 6)
        ]
        if dtype == "int8":
            fields = [to_signed(f) for f in fields]
        return [ops.cast(f, dtype) for f in fields]

    # Fast path for axis==0 (common case in Dense layers).
    if axis == 0 and rank == 2:
        fields = split_fields(packed)
        # Interleave the four fields along axis 0 and reshape.
        stacked = ops.stack(fields, axis=1)
        unpacked = ops.reshape(stacked, (-1,) + tuple(ops.shape(packed)[1:]))
        return unpacked[:orig_len, ...]

    # General case.
    perm = [axis] + [i for i in range(rank) if i != axis]
    inv_perm = [perm.index(i) for i in range(rank)]
    transposed = ops.transpose(packed, perm)

    fields = split_fields(transposed)

    stacked = ops.stack(fields, axis=1)
    unpacked = ops.reshape(stacked, (-1,) + tuple(ops.shape(transposed)[1:]))

    unpacked = unpacked[:orig_len, ...]
    unpacked = ops.transpose(unpacked, inv_perm)

    return unpacked


@keras_export("keras.quantizers.pack_ternary")
def pack_ternary(arr, axis=0):
    """Pack a ternary tensor into a `uint8` tensor at ~1.6 bits per value.

    The input values must be in `{-1, 0, +1}`. Five ternary values (trits) are
    packed into a single `uint8` byte using base-3 encoding, which is exact
    because `3 ** 5 == 243 <= 256`. This is the information-theoretic floor for
    ternary weights (`log2(3) ~= 1.58` bits/value) and is strictly denser than
    any integer format: ~2.5x denser than int4 and ~5x denser than int8, with
    no loss (the stored values are exactly the original `{-1, 0, +1}`).

    Each trit `t` is shifted to an unsigned digit `d = t + 1` in `{0, 1, 2}`.
    Five consecutive digits `d0..d4` along `axis` are then combined into one
    byte as `d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4` (max value `242`). If the axis
    length is not a multiple of 5 it is padded with zero-trits, which are
    removed on unpacking.

    Args:
        arr: A tensor whose values are in `{-1, 0, +1}` (any numeric dtype;
            values are rounded to the nearest integer and clipped to
            `[-1, 1]` before packing).
        axis: The axis along which to pack the tensor. Defaults to 0.

    Returns:
        tuple: A tuple `(packed, packed_shape, orig_len)` where `packed` is the
            packed `uint8` tensor, `packed_shape` is its shape, and `orig_len`
            is the original (unpadded) length of `axis`, needed by
            `unpack_ternary` to strip padding.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_ternary, unpack_ternary
    >>> original = np.array(
    ...     [[1, -1], [0, 1], [-1, 0], [1, 1], [0, -1], [1, 0]], dtype="int8"
    ... )  # shape (6, 2)
    # Axis 0 has length 6, padded to 10; packed shape is (ceil(6/5), 2) = (2, 2)
    >>> packed, packed_shape, orig_len = pack_ternary(original, axis=0)
    >>> unpacked = unpack_ternary(packed, orig_len, axis=0)
    >>> np.allclose(original, unpacked)
    True
    ```
    """
    arr_np = ops.convert_to_numpy(arr)
    rank = len(arr_np.shape)
    if axis < 0:
        axis += rank

    # Move the pack axis to the front for uniform handling.
    arr_np = np.moveaxis(arr_np, axis, 0)

    # Pad the front axis to a multiple of 5 with zero-trits (harmless: they
    # contribute nothing to the matmul and are stripped on unpack).
    n = arr_np.shape[0]
    pad = (-n) % 5
    if pad:
        pad_shape = (pad,) + arr_np.shape[1:]
        arr_np = np.concatenate(
            [arr_np, np.zeros(pad_shape, dtype=arr_np.dtype)], axis=0
        )

    # Map {-1, 0, +1} -> digits {0, 1, 2} and combine groups of 5 in base 3.
    # Clip before shifting: rounding alone can produce 2 (e.g. round(1.6)=2),
    # which overflows a base-3 digit and corrupts the adjacent trit on unpack.
    digits = np.clip(np.round(arr_np).astype(np.int32), -1, 1) + 1
    groups = digits.reshape((-1, 5) + digits.shape[1:])
    place = np.array([1, 3, 9, 27, 81], dtype=np.int32).reshape(
        (1, 5) + (1,) * (digits.ndim - 1)
    )
    packed_np = np.sum(groups * place, axis=1).astype(np.uint8)

    # Move the pack axis back to its original position.
    packed_np = np.moveaxis(packed_np, 0, axis)

    packed = ops.convert_to_tensor(packed_np)
    return packed, tuple(packed_np.shape), n


@keras_export("keras.quantizers.unpack_ternary")
def unpack_ternary(packed, orig_len, axis=0):
    """Unpack a base-3 packed `uint8` tensor back to ternary `{-1, 0, +1}`.

    This reverses `pack_ternary`, restoring an `int8` tensor whose values are
    in `{-1, 0, +1}`. The original axis order is preserved and any padding
    added during packing is removed.

    Args:
        packed: A `uint8` tensor produced by `pack_ternary`, with five trits
            encoded in each byte along `axis`.
        orig_len: The original (unpadded) length of the packed axis, used to
            strip padding.
        axis: The axis along which the tensor was packed. Defaults to 0.

    Returns:
        An `int8` tensor with values in `{-1, 0, +1}` and the original
        (unpacked) shape along `axis`.

    Example:

    ```python
    >>> import numpy as np
    >>> from keras.quantizers import pack_ternary, unpack_ternary
    >>> original = np.array([[1, -1, 0, 1, 0, -1]], dtype="int8")  # (1, 6)
    >>> packed, packed_shape, orig_len = pack_ternary(original, axis=1)
    >>> unpacked = unpack_ternary(packed, orig_len, axis=1)
    >>> np.allclose(original, unpacked)
    True
    ```
    """
    from keras.src import backend as _backend

    packed_dtype = _backend.standardize_dtype(packed.dtype)
    if packed_dtype not in ("uint8", "int8"):
        raise TypeError(
            "`unpack_ternary` expects a `uint8` or `int8` tensor produced by "
            f"`pack_ternary`. Received dtype: {packed_dtype}"
        )

    rank = getattr(packed.shape, "rank", None) or len(packed.shape)
    if axis < 0:
        axis += rank

    # Fast path: axis=0 on a rank-2 tensor — no transposes needed.
    if axis == 0 and rank == 2:
        codes = ops.cast(packed, "int32")
        codes = ops.where(codes < 0, codes + 256, codes)
        digits = []
        for place in (1, 3, 9, 27, 81):
            digit = ops.mod(ops.floor_divide(codes, place), 3)
            digits.append(ops.subtract(digit, 1))
        stacked = ops.stack(digits, axis=1)
        unpacked = ops.reshape(stacked, (-1, ops.shape(packed)[1]))
        unpacked = unpacked[:orig_len, ...]
        return ops.cast(unpacked, "int8")

    # General path: move the pack axis to the front, decode, restore layout.
    perm = [axis] + [i for i in range(rank) if i != axis]
    inv_perm = [perm.index(i) for i in range(rank)]
    transposed = ops.transpose(packed, perm)
    codes = ops.cast(transposed, "int32")
    codes = ops.where(codes < 0, codes + 256, codes)

    digits = []
    for place in (1, 3, 9, 27, 81):
        digit = ops.mod(ops.floor_divide(codes, place), 3)
        digits.append(ops.subtract(digit, 1))  # {0, 1, 2} -> {-1, 0, +1}

    # Interleave d0..d4 along the front axis and reshape: byte j holds trits
    # [5*j, 5*j + 4], so the stacked order reproduces the original sequence.
    stacked = ops.stack(digits, axis=1)
    unpacked = ops.reshape(stacked, (-1,) + tuple(ops.shape(transposed)[1:]))

    # Strip padding and restore the original layout.
    unpacked = unpacked[:orig_len, ...]
    unpacked = ops.cast(unpacked, "int8")
    unpacked = ops.transpose(unpacked, inv_perm)
    return unpacked


def compute_quantization_parameters(
    x,
    *,
    bits,
    symmetric=False,
    per_channel=False,
    group_size=-1,
    compute_dtype="float32",
    epsilon=0.0,
):
    """
    Computes the scale and zero-point for quantizing weight tensors.

    This function calculates the scale and zero-point required for quantizing
    a given weight tensor `x` based on the specified parameters. It supports
    grouped, per-channel, per-tensor, symmetric, and asymmetric quantization.

    For grouped quantization (per_channel=True, group_size > 0), the output
    shapes are [out_features, n_groups] where n_groups is the number of groups
    along the in_features dimension.

    Args:
        x: KerasTensor. The weight tensor to quantize with shape
            [out_features, in_features].
        bits: int. The number of bits to quantize to (e.g., 4).
        symmetric: bool. Whether to use symmetric quantization.
        per_channel: bool. Whether to quantize per channel.
        group_size: int. The group size for quantization. -1 means no grouping.
        compute_dtype: str. The dtype for computation. Defaults to "float32".
        epsilon: float. Small value added to (max - min) before computing
            scale to avoid division by zero. Defaults to 0.0.

    Returns:
        scale: KerasTensor. The scale tensor for quantization.
        zero: KerasTensor. The `uint8` zero tensor for quantization.
        maxq: scalar. The maximum quantization value.
    """
    # Input validation
    if x is None:
        raise ValueError(f"Input tensor {x} cannot be None.")
    if len(x.shape) < 2:
        raise ValueError(
            f"Input weight tensor {x} must have a rank of at "
            f"least 2, but got rank {len(x.shape)}."
        )
    if ops.size(x) == 0:
        raise ValueError("Input tensor 'x' cannot be empty.")

    out_features, in_features = x.shape[0], x.shape[1]

    # Determine number of groups for quantization
    if per_channel and group_size > 0:
        n_groups = (in_features + group_size - 1) // group_size
    else:
        n_groups = 1

    # Compute min/max values based on quantization mode
    if n_groups > 1:
        # Grouped quantization: output shape [out_features, n_groups]
        remainder = in_features % group_size
        if remainder != 0:
            pad_size = group_size - remainder
            x = ops.pad(x, [[0, 0], [0, pad_size]], constant_values=0.0)

        x_grouped = ops.reshape(x, [out_features, n_groups, group_size])
        min_values = ops.min(x_grouped, axis=2)
        max_values = ops.max(x_grouped, axis=2)
    else:
        # Per-channel or per-tensor: compute stats along rows
        reduction_shape = [out_features, -1] if per_channel else [1, -1]
        x_reshaped = ops.reshape(x, reduction_shape)
        min_values = ops.min(x_reshaped, axis=1)
        max_values = ops.max(x_reshaped, axis=1)

    # Asymmetric quantization: clamp the range to include zero, matching
    # reference GPTQ/AWQ (`xmin = min(xmin, 0)`, `xmax = max(xmax, 0)`).
    # This guarantees the zero point lands in `[0, maxq]`, so it is
    # representable in `bits`-bit packed formats, and that the quantized
    # grid can represent 0 exactly, even for groups whose values are
    # all-negative or all-positive.
    if not symmetric:
        min_values = ops.minimum(min_values, 0.0)
        max_values = ops.maximum(max_values, 0.0)

    # Symmetric quantization: make range symmetric around zero
    if symmetric:
        max_abs = ops.maximum(ops.abs(min_values), max_values)
        min_values = ops.where(
            ops.less(min_values, 0), ops.negative(max_abs), min_values
        )
        max_values = max_abs

    # Ensure non-zero range to avoid division errors
    zero_range = ops.equal(min_values, max_values)
    min_values = ops.where(zero_range, ops.subtract(min_values, 1), min_values)
    max_values = ops.where(zero_range, ops.add(max_values, 1), max_values)

    # Compute scale and zero-point
    maxq = ops.cast(ops.subtract(ops.power(2, bits), 1), compute_dtype)
    range_values = ops.subtract(max_values, min_values)
    if epsilon > 0:
        range_values = ops.add(range_values, epsilon)
    scale = ops.divide(range_values, maxq)
    scale = ops.where(ops.less_equal(scale, 0), 1e-8, scale)

    # Zero point in the unsigned range [0, 2^bits-1], e.g., [0, 15] for 4-bit
    if symmetric:
        zero = ops.full_like(scale, ops.divide(ops.add(maxq, 1), 2))
    else:
        zero = ops.round(ops.divide(ops.negative(min_values), scale))
    zero = ops.clip(zero, 0, maxq)

    # Reshape output to [out_features, n_groups] or [out_features, 1]
    if n_groups > 1:
        pass  # Already [out_features, n_groups]
    elif per_channel:
        scale = ops.reshape(scale, [-1, 1])
        zero = ops.reshape(zero, [-1, 1])
    else:
        # Per-tensor: tile single value to [out_features, 1]
        scale = ops.tile(ops.reshape(scale, (1, 1)), (out_features, 1))
        zero = ops.tile(ops.reshape(zero, (1, 1)), (out_features, 1))

    return scale, ops.cast(zero, "uint8"), maxq


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


def _take_group_params(scale, zero, g_idx, group_axis):
    """Gathers each position's group scale and zero point.

    `g_idx` is a 1-D integer tensor with one entry per position along the
    quantized dimension, naming that position's group (`0` to
    `n_groups - 1`; with 128 columns and `group_size=32` it is
    `[0] * 32 + [1] * 32 + [2] * 32 + [3] * 32`). `group_axis` is the axis
    of `scale` and `zero` that holds the per-group values. The gathered
    zero point is cast to the scale's dtype.
    """
    groups = ops.cast(g_idx, "int32")
    scales = ops.take(scale, groups, axis=group_axis)
    zeros = ops.cast(ops.take(zero, groups, axis=group_axis), scales.dtype)
    return scales, zeros


def quantize_with_sz_map(
    weights_matrix, scale, zero, g_idx, maxq, group_axis=-1
):
    """Quantizes `weights_matrix` with per-group multiplier scales.

    See `_take_group_params` for `g_idx` and `group_axis`; `maxq` is the
    largest code, `2**bits - 1`.
    """
    scales, zeros = _take_group_params(scale, zero, g_idx, group_axis)
    return quantize_with_zero_point(weights_matrix, scales, zeros, maxq)


def dequantize_with_sz_map(weights_matrix, scale, zero, g_idx, group_axis=-1):
    """Dequantizes codes with per-group multiplier scales.

    The real value is `(code - zero) * scale`; see `_take_group_params`
    for `g_idx` and `group_axis`.
    """
    scales, zeros = _take_group_params(scale, zero, g_idx, group_axis)
    return dequantize_with_zero_point(weights_matrix, scales, zeros)
