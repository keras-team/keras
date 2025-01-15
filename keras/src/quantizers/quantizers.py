import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import standardize_axis_for_numpy

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
        axis,
        value_range=(-127, 127),
        epsilon=backend.epsilon(),
        output_dtype="int8",
    ):
        Quantizer.__init__(self, output_dtype=output_dtype)
        if isinstance(axis, int):
            axis = (axis,)
        self.axis = tuple(axis)
        self.value_range = value_range
        self.epsilon = epsilon

    def __call__(self, x):
        quantized_x, scale = abs_max_quantize(
            x, self.axis, self.value_range, self.output_dtype, self.epsilon
        )
        return quantized_x, scale

    def get_config(self):
        return {
            "axis": self.axis,
            "value_range": self.value_range,
            "epsilon": self.epsilon,
            "output_dtype": self.output_dtype,
        }


def adjust_and_nudge(min_range, max_range, num_bits, narrow_range):
    """Adjusts and nudges the quantization range for better accuracy."""

    quant_max = ops.cast(ops.subtract(ops.power(2, num_bits), 1.0), "float32")

    quant_min = ops.cast(0.0 if not narrow_range else 1.0, "float32")

    # Calculate the scale and ensure it's positive
    scale = ops.divide(
        ops.subtract(max_range, min_range), ops.subtract(quant_max, quant_min)
    )

    inv_scale = ops.reciprocal(scale)

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


@keras_export("keras.quantizers.fake_quant_with_min_max_vars_per_channel")
def fake_quant_with_min_max_vars(
    inputs,
    min_vals,
    max_vals,
    num_bits,
    narrow_range=False,
    axis=None,
):
    """
    Perform per-tensor or per-channel fake quantization.

    `[min_vals, max_vals]` define the clamping range for the `inputs`.

    The `inputs` are quantized into the quantization range:
    - `[0, 2^num_bits - 1]` when `narrow_range=False`
    - `[1, 2^num_bits - 1]` when `narrow_range=True`

    After quantization, the values are dequantized and output as floats within
    the `[min_vals, max_vals]` interval.

    This operation supports gradient computation, allowing `min_vals` and
    `max_vals` to be trained.

    Args:
        inputs: Input tensor of float dtype.
        min_vals: A global minimum scalar or a per-channel minimum tensor.
        max_vals: A global maximum scalar or a per-channel maximum tensor.
        num_bits: Quantization bit width (e.g., `8` for int8).
        narrow_range: Whether to use narrow quantization range.
        axis: Axis along which to perform per-channel quantization. If `None`,
              per-tensor quantization is performed. Defaults to `None`.


    Returns:
        Fake-quantized tensor
    """
    inputs = ops.convert_to_tensor(inputs)
    min_vals = ops.convert_to_tensor(min_vals)
    max_vals = ops.convert_to_tensor(max_vals)

    if axis is not None:
        axis = canonicalize_axis(axis, inputs.ndim)

    @ops.custom_gradient
    def _fake_quant_with_min_max_vars_per_channel(x, min_val, max_val):
        # Calculate quantization parameters for all channels at once
        nudged_min, nudged_max, scale, inv_scale = adjust_and_nudge(
            min_val, max_val, num_bits, narrow_range
        )

        quant_zero = ops.floor(
            ops.add(ops.multiply(-nudged_min, inv_scale), 0.5)
        )
        x_clamped = ops.clip(x, nudged_min, nudged_max)
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

        # Create gradient mask for all channels
        masks = ops.cast(
            (x >= nudged_min) & (x <= nudged_max),
            dtype="float32",
        )

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args

            # Gradient for x
            dx = ops.multiply(upstream, masks)
            axes = [i for i in range(len(dx.shape)) if i != axis]
            # Gradient for min_val
            # When x is clipped to min, the gradient flows to min_val
            min_mask = ops.cast(x <= nudged_min, dtype="float32")
            grad_min = ops.multiply(upstream, min_mask)
            if axis is not None:
                grad_min = ops.sum(grad_min, axis=axes)
            else:
                grad_min = ops.sum(grad_min)

            # Gradient for max_val
            # When x is clipped to max, the gradient flows to max_val
            max_mask = ops.cast(x >= nudged_max, dtype="float32")
            grad_max = ops.multiply(upstream, max_mask)
            if axis is not None:
                grad_max = ops.sum(grad_max, axis=axes)
            else:
                grad_max = ops.sum(grad_max)

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
