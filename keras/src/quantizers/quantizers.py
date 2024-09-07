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
