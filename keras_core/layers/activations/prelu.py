from keras_core import activations
from keras_core import constraints
from keras_core import initializers
from keras_core import regularizers
from keras_core.api_export import keras_core_export
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.PReLU")
class PReLU(Layer):
    """Parametric Rectified Linear Unit activation layer.

    Formula:
    ``` python
    f(x) = negative_slope * x for x < 0
    f(x) = x for x >= 0
    ```
    where `negative_slope` is a learned array with the same shape as x.

    Args:
        negative_slope_initializer: Initializer function for the weights.
        negative_slope_regularizer: Regularizer for the weights.
        negative_slope_constraint: Constraint for the weights.
        shared_axes: The axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    def __init__(
        self,
        negative_slope_initializer="Zeros",
        negative_slope_regularizer=None,
        negative_slope_constraint=None,
        shared_axes=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.negative_slope_initializer = initializers.get(
            negative_slope_initializer
        )
        self.negative_slope_regularizer = regularizers.get(
            negative_slope_regularizer
        )
        self.negative_slope_constraint = constraints.get(
            negative_slope_constraint
        )
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        self.negative_slope = self.add_weight(
            shape=param_shape,
            name="negative_slope",
            initializer=self.negative_slope_initializer,
            regularizer=self.negative_slope_regularizer,
            constraint=self.negative_slope_constraint,
        )
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        pos = activations.relu(inputs)
        neg = -self.negative_slope * activations.relu(-inputs)
        return pos + neg

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "negative_slope_initializer": initializers.serialize(
                    self.negative_slope_initializer
                ),
                "negative_slope_regularizer": regularizers.serialize(
                    self.negative_slope_regularizer
                ),
                "negative_slope_constraint": constraints.serialize(
                    self.negative_slope_constraint
                ),
                "shared_axes": self.shared_axes,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
