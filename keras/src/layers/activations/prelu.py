from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@keras_export("keras.layers.PReLU")
class PReLU(Layer):
    """Parametric Rectified Linear Unit activation layer.

    Formula:
    ``` python
    f(x) = alpha * x for x < 0
    f(x) = x for x >= 0
    ```
    where `alpha` is a learned array with the same shape as x.

    Args:
        alpha_initializer: Initializer function for the weights.
        alpha_regularizer: Regularizer for the weights.
        alpha_constraint: Constraint for the weights.
        shared_axes: The axes along which to share learnable parameters for the
            activation function. For example, if the incoming feature maps are
            from a 2D convolution with output shape
            `(batch, height, width, channels)`, and you wish to share parameters
            across space so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
    """

    def __init__(
        self,
        alpha_initializer="Zeros",
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
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
        self.alpha = self.add_weight(
            shape=param_shape,
            name="alpha",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
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
        neg = -self.alpha * activations.relu(-inputs)
        return pos + neg

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_initializer": initializers.serialize(
                    self.alpha_initializer
                ),
                "alpha_regularizer": regularizers.serialize(
                    self.alpha_regularizer
                ),
                "alpha_constraint": constraints.serialize(
                    self.alpha_constraint
                ),
                "shared_axes": self.shared_axes,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
