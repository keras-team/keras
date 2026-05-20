from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@keras_export("keras.layers.TernaryDense")
class TernaryDense(Layer):
    """A densely-connected layer with weights quantized to {-1, 0, +1}.

    Drop-in replacement for `Dense` for ternary-weight models (BitNet b1.58
    and successors). On every forward pass the kernel is quantized to
    `{-1, 0, +1}` before the matrix multiply; the underlying float weights
    remain trainable so that gradient-based optimizers can update them between
    steps.

    Quantization rule (BitNet b1.58, https://arxiv.org/abs/2402.17764 §3.1):

    ```
    kernel_ternary[i,j] = sign(kernel[i,j])  if |kernel[i,j]| > threshold
                          0                   otherwise
    ```

    where `threshold` defaults to `mean(|kernel|)` per forward pass.

    Args:
        units: Positive integer, dimensionality of the output space.
        threshold: Float or `None`. Quantization boundary applied element-wise
            to `|kernel|`. `None` (default) uses `mean(|kernel|)` per forward
            pass, matching the BitNet b1.58 §3.1 rule. `0.0` maps every weight
            to ±1 (no zeros). Large values (e.g. `1e9`) yield an all-zero
            effective kernel.
        activation: Activation function to use. Defaults to no activation
            (linear: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
            Defaults to `True`.
        kernel_initializer: Initializer for the `kernel` weights matrix.
            Defaults to `"glorot_uniform"`.
        bias_initializer: Initializer for the bias vector.
            Defaults to `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel`
            weights matrix. Defaults to `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Defaults to `None`.
        activity_regularizer: Regularizer function applied to the output of
            the layer. Defaults to `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Defaults to `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Defaults to `None`.

    Input shape:
        N-D tensor with shape `(batch_size, ..., input_dim)`.

    Output shape:
        N-D tensor with shape `(batch_size, ..., units)`.

    Example:

    >>> layer = keras.layers.TernaryDense(64)
    >>> inputs = np.random.rand(4, 32).astype("float32")
    >>> outputs = layer(inputs)
    >>> outputs.shape
    (4, 64)
    """

    def __init__(
        self,
        units,
        threshold=None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        if not isinstance(units, int) or units <= 0:
            raise ValueError(
                "Received an invalid value for `units`, expected a positive "
                f"integer. Received: units={units}"
            )
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError(
                "Received an invalid value for `threshold`, expected a float "
                f"or None. Received: threshold={threshold}"
            )

        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.threshold = threshold
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def _ternary_kernel(self):
        """Return the kernel quantized to {-1, 0, +1}."""
        abs_k = ops.abs(self.kernel)
        t = ops.mean(abs_k) if self.threshold is None else self.threshold
        return ops.sign(self.kernel) * ops.cast(
            abs_k > t, dtype=self.kernel.dtype
        )

    def call(self, inputs):
        x = ops.matmul(inputs, self._ternary_kernel())
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "threshold": self.threshold,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        return {**base_config, **config}
