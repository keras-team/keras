from keras import backend
from keras import ops
from keras.api_export import keras_export


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

        An quantizer config is a Python dictionary (serializable)
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


@keras_export(["keras.quantizers.abs_max_quantize"])
def abs_max_quantize(
    inputs,
    axis,
    value_range=(-127, 127),
    dtype="int8",
    epsilon=backend.epsilon(),
):
    scale = ops.divide(
        value_range[1],
        ops.add(ops.max(ops.abs(inputs), axis=axis, keepdims=True), epsilon),
    )
    outputs = ops.multiply(inputs, scale)
    outputs = ops.clip(ops.round(outputs), value_range[0], value_range[1])
    outputs = ops.cast(outputs, dtype)
    return outputs, scale


@keras_export(["keras.AbsMaxQuantizer", "keras.quantizers.AbsMaxQuantizer"])
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
