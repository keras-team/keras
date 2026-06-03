from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
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
    output = beta * matmul(inputs, kernel_ternary) + bias
    ```

    where `threshold` defaults to `0.5 * mean(|kernel|)` and `beta =
    mean(|kernel|)` per forward pass (both per the BitNet b1.58 §3.1 spec).
    Gradients flow through the quantization step via a Straight-Through
    Estimator (STE), keeping the underlying float kernel trainable.

    For export and inference, call `layer.quantize("ternary")` after training.
    This replaces the float kernel with a packed representation that stores the
    `{-1, 0, +1}` weights at the information-theoretic floor of `log2(3) ~=
    1.58` bits/value: five trits are packed into each byte, since
    `3 ** 5 == 243 <= 256`. The result is ~1.6 bits/weight on disk, denser than
    int4 (4 bits) or int8 (8 bits) can express, and lossless (the exact
    ternary values are recovered). Inference then runs from the packed kernel.

    Args:
        units: Positive integer, dimensionality of the output space.
        threshold: Non-negative float or `None`. Quantization boundary applied
            element-wise to `|kernel|`. `None` (default) uses
            `0.5 * mean(|kernel|)` per forward pass, matching the BitNet b1.58
            §3.1 rule. `0.0` maps every weight to ±1 (no zeros). Large values
            (e.g. `1e9`) yield an all-zero effective kernel.
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

    Quantize for export and inference (kernel stored at ~1.6 bits/value):

    >>> layer = keras.layers.TernaryDense(64)
    >>> _ = layer(np.random.rand(4, 32).astype("float32"))
    >>> layer.quantize("ternary")
    >>> outputs = layer(np.random.rand(4, 32).astype("float32"))
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
        if threshold is not None and threshold < 0:
            raise ValueError(
                "Received an invalid value for `threshold`, expected a "
                "non-negative float or None. "
                f"Received: threshold={threshold}"
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
        kernel_shape = (input_dim, self.units)
        if self.quantization_mode:
            # Packed kernel + scale are created here; the float `kernel` is not.
            self.quantized_build(kernel_shape, mode=self.quantization_mode)
        else:
            self.kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
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
        """Forward value is in {-1, 0, +1}; gradients flow via STE."""
        abs_k = ops.abs(self.kernel)
        t = 0.5 * ops.mean(abs_k) if self.threshold is None else self.threshold
        k_ternary = ops.sign(self.kernel) * ops.cast(
            abs_k > t, dtype=self.kernel.dtype
        )
        return self.kernel + ops.stop_gradient(k_ternary - self.kernel)

    def call(self, inputs):
        k = self._ternary_kernel()
        x = ops.matmul(inputs, k)
        if self.threshold is None:
            beta = ops.mean(ops.abs(self.kernel))
            x = ops.multiply(x, beta)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    # Quantization: real ternary export + inference.
    #
    # During training the float `kernel` is kept and ternarized on the fly
    # (the `call` path above). `quantize("ternary")` freezes that result into a
    # packed representation: the `{-1, 0, +1}` kernel is stored at the
    # information-theoretic floor of ~1.6 bits/value (five trits per byte,
    # `3 ** 5 == 243 <= 256`), denser than int4 or int8 can express. Inference
    # then runs from the packed kernel via `_ternary_call`.

    def quantized_build(self, kernel_shape, mode):
        if mode == "ternary":
            self._ternary_build(kernel_shape)
        else:
            raise self._quantization_mode_error(mode)
        self._is_quantized = True

    def _ternary_build(self, kernel_shape):
        input_dim, units = kernel_shape
        # Five trits per byte -> ceil(input_dim / 5) packed rows.
        packed_rows = (input_dim + 4) // 5
        self._packed_kernel = self.add_weight(
            name="kernel",
            shape=(packed_rows, units),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )
        # Scalar BitNet b1.58 scale (beta); 1.0 in fixed-threshold mode.
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(),
            initializer="ones",
            trainable=False,
        )
        self._orig_input_dim = input_dim

    def _ternary_call(self, inputs):
        kernel = quantizers.unpack_ternary(
            self._packed_kernel, self._orig_input_dim, axis=0
        )
        kernel = ops.cast(kernel, self.compute_dtype)
        x = ops.matmul(inputs, kernel)
        x = ops.multiply(x, ops.cast(self.kernel_scale, self.compute_dtype))
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def quantize(self, mode="ternary", type_check=True, config=None):
        # Prevent quantization of subclasses with a different kernel layout.
        if type_check and type(self) is not TernaryDense:
            raise self._not_implemented_error(self.quantize)
        if mode != "ternary":
            raise self._quantization_mode_error(mode)

        kernel_shape = self.kernel.shape
        # Hard ternary values in {-1, 0, +1}. This is exactly the forward value
        # of the straight-through kernel used in training, so freezing it does
        # not change the layer's outputs.
        kernel_ternary = ops.convert_to_numpy(self._ternary_kernel())
        if self.threshold is None:
            beta = float(ops.convert_to_numpy(ops.mean(ops.abs(self.kernel))))
        else:
            beta = 1.0
        packed_kernel, _, _ = quantizers.pack_ternary(kernel_ternary, axis=0)

        del self.kernel
        self.quantized_build(kernel_shape, mode=mode)
        self._packed_kernel.assign(packed_kernel)
        self.kernel_scale.assign(beta)

        # Switch to a ternary dtype policy so future calls take the packed path.
        if self.dtype_policy.quantization_mode is None:
            from keras.src import dtype_policies

            policy = dtype_policies.get(
                f"ternary_from_{self.dtype_policy.name}"
            )
            self.dtype_policy = policy

    @property
    def variable_serialization_spec(self):
        """Maps each quantization mode to its ordered variable names."""
        return {
            None: ["kernel", "bias"],
            "ternary": ["kernel", "bias", "kernel_scale"],
        }

    def _serialization_targets(self, mode):
        return {
            "kernel": (
                self._packed_kernel if mode == "ternary" else self.kernel
            ),
            "bias": self.bias,
            "kernel_scale": getattr(self, "kernel_scale", None),
        }

    def save_own_variables(self, store):
        if not self.built:
            return
        mode = self.quantization_mode
        if mode not in self.variable_serialization_spec:
            raise self._quantization_mode_error(mode)
        targets = self._serialization_targets(mode)
        idx = 0
        for name in self.variable_serialization_spec[mode]:
            if name == "bias" and self.bias is None:
                continue
            store[str(idx)] = targets[name]
            idx += 1

    def load_own_variables(self, store):
        self._check_load_own_variables(store)
        if not self.built:
            return
        mode = self.quantization_mode
        if mode not in self.variable_serialization_spec:
            raise self._quantization_mode_error(mode)
        targets = self._serialization_targets(mode)
        idx = 0
        for name in self.variable_serialization_spec[mode]:
            if name == "bias" and self.bias is None:
                continue
            targets[name].assign(store[str(idx)])
            idx += 1

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
