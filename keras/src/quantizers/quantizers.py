import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
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
    if num_bits < 2:
        raise ValueError("num_bits must be >= 2")

    n_steps = float(2**num_bits - 1)
    if narrow_range:
        n_steps -= 1.0

    # Handle the case where min and max are too close
    if abs(max_range - min_range) < 1e-10:
        return min_range, max_range, 1.0

    # Calculate the step size
    step_size = (max_range - min_range) / n_steps

    # Calculate the reciprocal of the step size
    inv_step_size = 1.0 / step_size

    # Round the reciprocal to get an integer
    rounded_inv_step_size = ops.round(inv_step_size)

    # Calculate the final step size
    final_step_size = 1.0 / rounded_inv_step_size

    # Calculate the quantized min/max values, ensuring accurate rounding
    quantized_min = (
        ops.round(min_range * rounded_inv_step_size) / rounded_inv_step_size
    )
    quantized_max = (
        ops.round(max_range * rounded_inv_step_size) / rounded_inv_step_size
    )

    # Convert quantization limits to float
    quant_min_float = ops.cast(quantized_min, "float32")
    quant_max_float = ops.cast(quantized_max, "float32")

    # Calculate the scale
    nudged_scale = (max_range - min_range) / (quant_max_float - quant_min_float)

    # Calculate zero point from min
    zero_point_from_min = quant_min_float - min_range / nudged_scale

    # Determine nudged zero point
    nudged_zero_point = ops.where(
        zero_point_from_min < quant_min_float,
        quantized_min,
        ops.where(
            zero_point_from_min > quant_max_float,
            quantized_max,
            ops.round(zero_point_from_min),
        ),
    )

    # Calculate nudged min and max
    nudged_min = (quant_min_float - nudged_zero_point) * nudged_scale
    nudged_max = (quant_max_float - nudged_zero_point) * nudged_scale

    return (
        nudged_min,
        nudged_max,
        final_step_size,
    )  # Returning nudged values and scale


@keras_export("keras.quantizers.fake_quant_with_min_max_args")
def fake_quant_with_min_max_args(
    inputs,
    min_range=-6.0,
    max_range=6.0,
    num_bits=8,
    narrow_range=False,
):
    """Fake quantization operation matching TensorFlow's implementation."""

    if isinstance(inputs, np.ndarray):
        inputs = ops.convert_to_tensor(inputs)

    @ops.custom_gradient
    def _fake_quant_with_min_max_args(x):
        quant_min, quant_max, step_size = adjust_and_nudge(
            min_range, max_range, num_bits, narrow_range
        )

        n_steps = 2**num_bits - 1
        if narrow_range:
            n_steps -= 1

        # Clip and nudge input to the range
        x_clipped = ops.clip(x, quant_min, quant_max)
        x_norm = (x_clipped - quant_min) / step_size
        x_quantized = ops.round(x_norm)
        x_quantized = ops.clip(x_quantized, 0.0, n_steps)
        result = x_quantized * step_size + quant_min

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            # Gradient mask: valid within the range
            mask = ops.cast(
                (x >= quant_min) & (x <= quant_max), dtype=upstream.dtype
            )
            return ops.multiply(upstream, mask)

        return result, grad

    return _fake_quant_with_min_max_args(inputs)


@keras_export("keras.quantizers.fake_quant_with_min_max_vars")
def fake_quant_with_min_max_vars(
    inputs,
    min_range=-6.0,
    max_range=6.0,
    num_bits=8,
    narrow_range=False,
):
    """Fake quantization operation matching TensorFlow's implementation."""
    return fake_quant_with_min_max_args(
        inputs, min_range, max_range, num_bits, narrow_range
    )


@keras_export("keras.quantizers.fake_quant_with_min_max_args_gradient")
def fake_quant_with_min_max_args_gradient(
    gradients,
    inputs,
    min_range=-6.0,
    max_range=6.0,
    num_bits=8,
    narrow_range=False,
):
    """Fake quantization operation with gradient,
    matching TensorFlow's implementation."""

    inputs = ops.convert_to_tensor(inputs)

    def _fake_quant_with_min_max_args_gradient(x):
        quant_min, quant_max, step_size = adjust_and_nudge(
            min_range, max_range, num_bits, narrow_range
        )

        n_steps = 2**num_bits - 1
        if narrow_range:
            n_steps -= 1

        # Clip and nudge input to the range
        x_clipped = ops.clip(x, quant_min, quant_max)
        x_norm = (x_clipped - quant_min) / step_size
        x_quantized = ops.round(x_norm)
        x_quantized = ops.clip(x_quantized, 0.0, n_steps)
        result = x_quantized * step_size + quant_min

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            # Gradient mask: valid within the range
            mask = ops.cast(
                (x >= quant_min) & (x <= quant_max), dtype=upstream.dtype
            )
            return ops.multiply(upstream, mask)

        return result, grad

    output, grad = _fake_quant_with_min_max_args_gradient(inputs)
    return output, grad(gradients)


@keras_export("keras.quantizers.fake_quant_with_min_max_vars_per_channel")
def fake_quant_with_min_max_vars_per_channel(
    inputs,
    min_vals,
    max_vals,
    num_bits,
    narrow_range,
):
    """
    Perform per-channel fake quantization with custom gradient.

    Args:
        inputs: Input tensor of float type
        min_vals: Per-channel minimum values
        max_vals: Per-channel maximum values
        num_bits: Quantization bit width (2-16)
        narrow_range: Whether to use narrow quantization range

    Returns:
        Fake-quantized tensor
    """

    inputs = ops.convert_to_tensor(inputs)
    min_vals = ops.convert_to_tensor(min_vals)
    max_vals = ops.convert_to_tensor(max_vals)

    @ops.custom_gradient
    def _fake_quant_with_min_max_vars_per_channel(x, min_val, max_val):
        # Determine the number of channels
        num_channels = min_val.shape[-1]

        # Initialize an empty list to store quantized values for each channel
        quantized_channels = []
        masks = []

        # Iterate over each channel
        for i in range(num_channels):
            # Extract min/max values for current channel
            current_min = min_val[..., i]
            current_max = max_val[..., i]

            # Calculate step size and quantized min/max using _adjust_range
            qnt_min, qnt_max, step_size = adjust_and_nudge(
                current_min, current_max, num_bits, narrow_range
            )
            # Calculate the number of steps
            n_steps = 2**num_bits - 1
            if narrow_range:
                n_steps -= 1

            # Clip and nudge input to the range for the current channel
            x_clipped = ops.clip(x[..., i], qnt_min, qnt_max)
            x_norm = (x_clipped - qnt_min) / step_size
            x_quantized = ops.round(x_norm)
            x_quantized = ops.clip(x_quantized, 0.0, n_steps)
            result_channel = x_quantized * step_size + qnt_min

            quantized_channels.append(result_channel)
            mask = ops.cast(
                (x[..., i] >= qnt_min) & (x[..., i] <= qnt_max),
                dtype=np.float32,
            )
            masks.append(mask)

        # Concatenate quantized channels
        result = ops.stack(quantized_channels, axis=-1)

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args

            # Gradient mask: valid within the range
            return ops.multiply(upstream, mask)

        return result, grad

    return _fake_quant_with_min_max_vars_per_channel(inputs, min_vals, max_vals)


@keras_export(
    "keras.quantizers.fake_quant_with_min_max_vars_per_channel_gradient"
)
def fake_quant_with_min_max_vars_per_channel_gradient(
    gradients,
    inputs,
    min_vals,
    max_vals,
    num_bits,
    narrow_range,
):
    """
    Perform per-channel fake quantization with custom gradient.

    Args:
        inputs: Input tensor of float type
        min_vals: Per-channel minimum values
        max_vals: Per-channel maximum values
        num_bits: Quantization bit width (2-16)
        narrow_range: Whether to use narrow quantization range

    Returns:
        Fake-quantized tensor
    """

    if isinstance(inputs, np.ndarray):
        inputs = ops.convert_to_tensor(inputs)
    min_vals = ops.convert_to_tensor(min_vals)
    max_vals = ops.convert_to_tensor(max_vals)

    # @ops.custom_gradient
    def _fake_quant_with_min_max_vars_per_channel_gradient(x, min_val, max_val):
        # Determine the number of channels
        num_channels = min_val.shape[-1]

        # Initialize an empty list to store quantized values for each channel
        quantized_channels = []
        between_min_max_masks = []
        below_min_masks = []
        above_max_masks = []

        # Iterate over each channel
        for i in range(num_channels):
            # Extract min/max values for current channel
            current_min = min_val[..., i]
            current_max = max_val[..., i]

            # Calculate step size and quantized min/max using _adjust_range
            qnt_min, qnt_max, step_size = adjust_and_nudge(
                current_min, current_max, num_bits, narrow_range
            )

            # Calculate the number of steps
            n_steps = 2**num_bits - 1
            if narrow_range:
                n_steps -= 1

            # Clip and nudge input to the range for the current channel
            x_clipped = ops.clip(x[..., i], qnt_min, qnt_max)
            x_norm = (x_clipped - qnt_min) / step_size
            x_quantized = ops.round(x_norm)
            x_quantized = ops.clip(x_quantized, 0.0, n_steps)
            result_channel = x_quantized * step_size + qnt_min
            between_min_max_mask = ops.cast(
                (x[..., i] >= qnt_min) & (x[..., i] <= qnt_max),
                dtype=np.float32,
            )
            below_min_mask = ops.cast((x[..., i] < qnt_min), dtype=np.float32)
            above_max_mask = ops.cast((x[..., i] > qnt_max), dtype=np.float32)
            between_min_max_masks.append(between_min_max_mask)
            below_min_masks.append(below_min_mask)
            above_max_masks.append(above_max_mask)
            quantized_channels.append(result_channel)

        # Concatenate quantized channels
        result = ops.stack(quantized_channels, axis=-1)
        between_min_max_masks = ops.stack(between_min_max_masks, axis=-1)
        below_min_masks = ops.stack(below_min_masks, axis=-1)
        above_max_masks = ops.stack(above_max_masks, axis=-1)

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            backprops_wrt_input = ops.multiply(upstream, between_min_max_masks)
            backprops_wrt_min = ops.sum(
                ops.multiply(upstream, below_min_masks), axis=0
            )
            backprops_wrt_max = ops.sum(
                ops.multiply(upstream, above_max_masks), axis=0
            )

            return backprops_wrt_input, backprops_wrt_min, backprops_wrt_max

        return result, grad

    output, grad = _fake_quant_with_min_max_vars_per_channel_gradient(
        inputs, min_vals, max_vals
    )
    backprops_wrt_input, backprops_wrt_min, backprops_wrt_max = grad(gradients)

    return output, backprops_wrt_input, backprops_wrt_min, backprops_wrt_max


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
