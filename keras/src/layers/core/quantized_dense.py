import math

from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@ops.custom_gradient
def _fake_quant_with_ste(inputs, min_val, max_val, num_bits=8, narrow_range=True):
    quant_min = 1.0 if narrow_range else 0.0
    quant_max = (2.0 ** num_bits) - 1.0
    
    # Avoid division by zero
    scale = (max_val - min_val) / (quant_max - quant_min)
    
    quantized = ops.round((inputs - min_val) / scale)
    quantized = ops.clip(quantized, quant_min, quant_max)
    dequantized = (quantized * scale) + min_val
    
    def grad(upstream):
        # Straight-Through Estimator (STE)
        # We pass gradients through to inputs.
        return upstream, None, None
        
    return dequantized, grad


class QuantizedDense(Layer):
    """A densely-connected layer with fake weight quantization for QAT.

    This layer acts like a standard Dense layer but simulates
    4-bit or 8-bit weight quantization during the forward pass using
    fake quantization. It is designed for Quantization-Aware Training (QAT).

    Args:
        units: Positive integer, dimensionality of the output space.
        bits: Integer, bit width for weight quantization (4 or 8).
        activation: Activation function to use.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        bits=8,
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
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = int(units)
        self.bits = int(bits)
        if self.bits not in [4, 8]:
            raise ValueError("Only 4-bit and 8-bit quantization are supported.")
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `QuantizedDense` "
                "should be defined. Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        # Fake quantize the kernel
        kernel = ops.cast(self.kernel, self.compute_dtype)
        min_val = ops.min(kernel)
        max_val = ops.max(kernel)
        max_val = ops.maximum(max_val, min_val + 1e-5)

        quantized_kernel = _fake_quant_with_ste(
            kernel, min_val, max_val, self.bits, True
        )

        outputs = ops.matmul(inputs, quantized_kernel)

        if self.use_bias:
            outputs = ops.add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "bits": self.bits,
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
