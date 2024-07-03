import ml_dtypes

from keras.src import activations
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@keras_export("keras.layers.Dense")
class Dense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors). The output in this case will have
    shape `(batch_size, d0, units)`.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
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
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `Dense` layer by calling `layer.enable_lora(rank)`.

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
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_enabled = False
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.quantization_mode:
            self.quantized_build(input_shape, mode=self.quantization_mode)
        if self.quantization_mode != "int8":
            # If the layer is quantized to int8, `self._kernel` will be added
            # in `self._int8_build`. Therefore, we skip it here.
            self._kernel = self.add_weight(
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
        if self.lora_rank:
            self.enable_lora(self.lora_rank)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        if self.lora_enabled:
            return self._kernel + ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )
        return self._kernel

    def call(self, inputs, training=None):
        x = ops.matmul(inputs, self.kernel)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        if self.kernel_constraint:
            raise ValueError(
                "Lora is incompatible with kernel constraints. "
                "In order to enable lora on this layer, remove the "
                "`kernel_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. "
                "This can only be done once per layer."
            )
        self._tracker.unlock()
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(self.kernel.shape[0], rank),
            initializer=initializers.get(a_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.kernel.shape[1]),
            initializer=initializers.get(b_initializer),
            regularizer=self.kernel_regularizer,
        )
        self._kernel.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank

    def save_own_variables(self, store):
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        kernel_value, kernel_scale = self._get_kernel_with_merged_lora()
        target_variables = [kernel_value]
        if self.use_bias:
            target_variables.append(self.bias)
        if self.quantization_mode is not None:
            if self.quantization_mode == "int8":
                target_variables.append(kernel_scale)
            elif self.quantization_mode == "float8":
                target_variables.append(self.inputs_scale)
                target_variables.append(self.inputs_amax_history)
                target_variables.append(self.kernel_scale)
                target_variables.append(self.kernel_amax_history)
                target_variables.append(self.outputs_grad_scale)
                target_variables.append(self.outputs_grad_amax_history)
            else:
                raise self._quantization_mode_error(self.quantization_mode)
        for i, variable in enumerate(target_variables):
            store[str(i)] = variable

    def load_own_variables(self, store):
        if not self.lora_enabled:
            self._check_load_own_variables(store)
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        target_variables = [self._kernel]
        if self.use_bias:
            target_variables.append(self.bias)
        if self.quantization_mode is not None:
            if self.quantization_mode == "int8":
                target_variables.append(self.kernel_scale)
            elif self.quantization_mode == "float8":
                target_variables.append(self.inputs_scale)
                target_variables.append(self.inputs_amax_history)
                target_variables.append(self.kernel_scale)
                target_variables.append(self.kernel_amax_history)
                target_variables.append(self.outputs_grad_scale)
                target_variables.append(self.outputs_grad_amax_history)
            else:
                raise self._quantization_mode_error(self.quantization_mode)
        for i, variable in enumerate(target_variables):
            variable.assign(store[str(i)])
        if self.lora_enabled:
            self.lora_kernel_a.assign(ops.zeros(self.lora_kernel_a.shape))
            self.lora_kernel_b.assign(ops.zeros(self.lora_kernel_b.shape))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
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
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
        return {**base_config, **config}

    def _check_load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )

    # Quantization-related (int8 and float8) methods

    def _quantization_mode_error(self, mode):
        return NotImplementedError(
            "Invalid quantization mode. Expected one of "
            f"{dtype_policies.QUANTIZATION_MODES}. "
            f"Received: quantization_mode={mode}"
        )

    def quantized_build(self, input_shape, mode):
        if mode == "int8":
            input_dim = input_shape[-1]
            kernel_shape = (input_dim, self.units)
            self._int8_build(kernel_shape)
        elif mode == "float8":
            self._float8_build()
        else:
            raise self._quantization_mode_error(mode)

    def _int8_build(
        self,
        kernel_shape,
        kernel_initializer="zeros",
        kernel_scale_initializer="ones",
    ):
        self.inputs_quantizer = quantizers.AbsMaxQuantizer(axis=-1)
        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=kernel_initializer,
            dtype="int8",
            trainable=False,
        )
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(self.units,),
            initializer=kernel_scale_initializer,
            trainable=False,
        )
        self._is_quantized = True

    def _float8_build(self):
        from keras.src.dtype_policies import QuantizedFloat8DTypePolicy

        # If `self.dtype_policy` is not QuantizedFloat8DTypePolicy, then set
        # `amax_history_length` to its default value.
        amax_history_length = getattr(
            self.dtype_policy,
            "amax_history_length",
            QuantizedFloat8DTypePolicy.default_amax_history_length,
        )
        # We set `trainable=True` because we will use the gradients to overwrite
        # these variables
        scale_kwargs = {
            "shape": (),
            "initializer": "ones",
            "dtype": "float32",  # Always be float32
            "trainable": True,
            "autocast": False,
        }
        amax_history_kwargs = {
            "shape": (amax_history_length,),
            "initializer": "zeros",
            "dtype": "float32",  # Always be float32
            "trainable": True,
            "autocast": False,
        }
        self.inputs_scale = self.add_weight(name="inputs_scale", **scale_kwargs)
        self.inputs_amax_history = self.add_weight(
            name="inputs_amax_history", **amax_history_kwargs
        )
        self.kernel_scale = self.add_weight(name="kernel_scale", **scale_kwargs)
        self.kernel_amax_history = self.add_weight(
            name="kernel_amax_history", **amax_history_kwargs
        )
        self.outputs_grad_scale = self.add_weight(
            name="outputs_grad_scale", **scale_kwargs
        )
        self.outputs_grad_amax_history = self.add_weight(
            name="outputs_grad_amax_history", **amax_history_kwargs
        )
        # We need to set `overwrite_with_gradient=True` to instruct the
        # optimizer to directly overwrite these variables with their computed
        # gradients during training
        self.inputs_scale.overwrite_with_gradient = True
        self.inputs_amax_history.overwrite_with_gradient = True
        self.kernel_scale.overwrite_with_gradient = True
        self.kernel_amax_history.overwrite_with_gradient = True
        self.outputs_grad_scale.overwrite_with_gradient = True
        self.outputs_grad_amax_history.overwrite_with_gradient = True
        self._is_quantized = True

    def quantized_call(self, inputs, training=None):
        if self.quantization_mode == "int8":
            return self._int8_call(inputs)
        elif self.quantization_mode == "float8":
            return self._float8_call(inputs, training=training)
        else:
            raise self._quantization_mode_error(self.quantization_mode)

    def _int8_call(self, inputs):
        @ops.custom_gradient
        def matmul_with_inputs_gradient(inputs, kernel, kernel_scale):
            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=self.compute_dtype),
                    kernel_scale,
                )
                inputs_grad = ops.matmul(upstream, ops.transpose(float_kernel))
                return (inputs_grad, None, None)

            inputs, inputs_scale = self.inputs_quantizer(inputs)
            x = ops.matmul(inputs, kernel)
            # De-scale outputs
            x = ops.cast(x, self.compute_dtype)
            x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
            return x, grad_fn

        x = matmul_with_inputs_gradient(
            inputs,
            ops.convert_to_tensor(self._kernel),
            ops.convert_to_tensor(self.kernel_scale),
        )
        if self.lora_enabled:
            lora_x = ops.matmul(inputs, self.lora_kernel_a)
            lora_x = ops.matmul(lora_x, self.lora_kernel_b)
            x = ops.add(x, lora_x)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _float8_call(self, inputs, training=None):
        if self.lora_enabled:
            raise NotImplementedError(
                "Currently, `_float8_call` doesn't support LoRA"
            )

        @ops.custom_gradient
        def quantized_dequantize_inputs(inputs, scale, amax_history):
            if training:
                new_scale = quantizers.compute_float8_scale(
                    ops.max(amax_history, axis=0),
                    scale,
                    ops.cast(
                        float(ml_dtypes.finfo("float8_e4m3fn").max), "float32"
                    ),
                )
                new_amax_history = quantizers.compute_float8_amax_history(
                    inputs, amax_history
                )
            else:
                new_scale = None
                new_amax_history = None
            qdq_inputs = quantizers.quantize_and_dequantize(
                inputs, scale, "float8_e4m3fn", self.compute_dtype
            )

            def grad(*args, upstream=None, variables=None):
                if upstream is None:
                    (upstream,) = args
                return upstream, new_scale, new_amax_history

            return qdq_inputs, grad

        @ops.custom_gradient
        def quantized_dequantize_outputs(outputs, scale, amax_history):
            """Quantize-dequantize the output gradient but not the output."""

            def grad(*args, upstream=None, variables=None):
                if upstream is None:
                    (upstream,) = args
                new_scale = quantizers.compute_float8_scale(
                    ops.max(amax_history, axis=0),
                    scale,
                    ops.cast(
                        float(ml_dtypes.finfo("float8_e5m2").max), "float32"
                    ),
                )
                qdq_upstream = quantizers.quantize_and_dequantize(
                    upstream, scale, "float8_e5m2", self.compute_dtype
                )
                new_amax_history = quantizers.compute_float8_amax_history(
                    upstream, amax_history
                )
                return qdq_upstream, new_scale, new_amax_history

            return outputs, grad

        x = ops.matmul(
            quantized_dequantize_inputs(
                inputs,
                ops.convert_to_tensor(self.inputs_scale),
                ops.convert_to_tensor(self.inputs_amax_history),
            ),
            quantized_dequantize_inputs(
                ops.convert_to_tensor(self._kernel),
                ops.convert_to_tensor(self.kernel_scale),
                ops.convert_to_tensor(self.kernel_amax_history),
            ),
        )
        # `quantized_dequantize_outputs` is placed immediately after
        # `ops.matmul` for the sake of pattern matching in gemm_rewrite. That
        # way, the qdq will be adjacent to the corresponding matmul_bprop in the
        # bprop.
        x = quantized_dequantize_outputs(
            x,
            ops.convert_to_tensor(self.outputs_grad_scale),
            ops.convert_to_tensor(self.outputs_grad_amax_history),
        )
        if self.bias is not None:
            # Under non-mixed precision cases, F32 bias has to be converted to
            # BF16 first to get the biasAdd fusion support. ref. PR
            # https://github.com/tensorflow/tensorflow/pull/60306
            bias = self.bias
            if self.dtype_policy.compute_dtype == "float32":
                bias_bf16 = ops.cast(bias, "bfloat16")
                bias = ops.cast(bias_bf16, bias.dtype)
            x = ops.add(x, bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def quantize(self, mode, type_check=True):
        import gc

        # Prevent quantization of the subclasses
        if type_check and (type(self) is not Dense):
            raise self._quantize_not_implemented_error()
        self._check_quantize_args(mode, self.compute_dtype)

        self._tracker.unlock()
        if mode == "int8":
            # Quantize `self._kernel` to int8 and compute corresponding scale
            kernel_value, kernel_scale = quantizers.abs_max_quantize(
                self._kernel, axis=0
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            self._untrack_variable(self._kernel)
            kernel_shape = self._kernel.shape
            del self._kernel
            # Utilize a lambda expression as an initializer to prevent adding a
            # large constant to the computation graph.
            self._int8_build(
                kernel_shape,
                lambda shape, dtype: kernel_value,
                lambda shape, dtype: kernel_scale,
            )
        elif mode == "float8":
            self._float8_build()
        else:
            raise self._quantization_mode_error(mode)
        self._tracker.lock()

        # Set new dtype policy
        if self.dtype_policy.quantization_mode is None:
            policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
            self.dtype_policy = policy

        # Release memory manually because sometimes the backend doesn't
        gc.collect()

    def _get_kernel_with_merged_lora(self):
        if self.dtype_policy.quantization_mode is not None:
            kernel_value = self._kernel
            kernel_scale = self.kernel_scale
            if self.lora_enabled:
                # Dequantize & quantize to merge lora weights into int8 kernel
                # Note that this is a lossy compression
                kernel_value = ops.divide(kernel_value, kernel_scale)
                kernel_value = ops.add(
                    kernel_value,
                    ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
                )
                kernel_value, kernel_scale = quantizers.abs_max_quantize(
                    kernel_value, axis=0
                )
                kernel_scale = ops.squeeze(kernel_scale, axis=0)
            return kernel_value, kernel_scale
        return self.kernel, None
