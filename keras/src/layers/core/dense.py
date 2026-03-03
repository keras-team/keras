import math

import ml_dtypes

from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantization_config import get_block_size_for_layer
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.saving import serialization_lib


@keras_export("keras.layers.Dense")
class Dense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`). When this layer is
    followed by a `BatchNormalization` layer, it is recommended to set
    `use_bias=False` as `BatchNormalization` has its own bias term.

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
        lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.

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
        lora_alpha=None,
        quantization_config=None,
        **kwargs,
    ):
        if not isinstance(units, int) or units <= 0:
            raise ValueError(
                "Received an invalid value for `units`, expected a positive "
                f"integer. Received: units={units}"
            )

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
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_enabled = False
        self.quantization_config = quantization_config
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        kernel_shape = (input_shape[-1], self.units)
        if self.quantization_mode:
            self.quantized_build(
                kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        if self.quantization_mode not in ("int8", "int4", "gptq", "awq"):
            # If the layer is quantized to int8 or int4, `self._kernel` will be
            # added in `self._int8_build` or `_int4_build`. Therefore, we skip
            # it here.
            self._kernel = self.add_weight(
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
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_shape[-1]})
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)

    @property
    def kernel(self):
        from keras.src.quantizers import gptq_core

        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )

        mode = self.quantization_mode
        is_gptq = mode == "gptq"
        is_awq = mode == "awq"
        is_int4 = mode == "int4"
        gptq_calibrated = bool(getattr(self, "is_gptq_calibrated", False))
        awq_calibrated = bool(getattr(self, "is_awq_calibrated", False))
        gptq_bits = (
            gptq_core.get_weight_bits_for_layer(self, None) if is_gptq else None
        )

        # Decide the source tensor first (packed vs already-quantized vs plain
        # kernel)
        if is_gptq and gptq_calibrated and gptq_bits != 4:
            # calibrated GPTQ, not 4-bit, no unpacking needed
            kernel = self.quantized_kernel
        else:
            # Start with the stored kernel
            kernel = getattr(self, "_kernel", None)

            # Handle int4 unpacking cases in one place
            if is_int4:
                # unpack [in, ceil(out/2)] to [in, out]
                kernel = quantizers.unpack_int4(
                    kernel, self._orig_output_dim, axis=-1
                )
            elif is_gptq and gptq_calibrated and gptq_bits == 4:
                kernel = quantizers.unpack_int4(
                    self.quantized_kernel,
                    orig_len=self.units,
                    axis=0,
                    dtype="uint8",
                )
            elif is_awq and awq_calibrated:
                # AWQ always uses 4-bit quantization
                kernel = quantizers.unpack_int4(
                    self.quantized_kernel,
                    orig_len=self.units,
                    axis=0,
                    dtype="uint8",
                )

        # Apply LoRA once at the end.
        if self.lora_enabled:
            kernel = kernel + (self.lora_alpha / self.lora_rank) * ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )

        return kernel

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
        self,
        rank,
        lora_alpha=None,
        a_initializer="he_uniform",
        b_initializer="zeros",
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
                "lora is already enabled. This can only be done once per layer."
            )
        if self.quantization_mode == "gptq":
            raise NotImplementedError(
                "lora is not currently supported with GPTQ quantization."
            )
        self._tracker.unlock()
        # Determine the correct input dimension for the LoRA A matrix. When
        # the layer has been int4-quantized, `self._kernel` stores a *packed*
        # representation whose first dimension is `ceil(input_dim/2)`. We
        # saved the true, *unpacked* input dimension in `self._orig_input_dim`
        # during quantization. Use it if available; otherwise fall back to the
        # first dimension of `self.kernel`.
        if self.quantization_mode == "int4" and hasattr(
            self, "_orig_input_dim"
        ):
            input_dim_for_lora = self._orig_input_dim
        else:
            input_dim_for_lora = self.kernel.shape[0]

        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(input_dim_for_lora, rank),
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
        self.lora_alpha = lora_alpha if lora_alpha is not None else rank

    def save_own_variables(self, store):
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        mode = self.quantization_mode
        if mode not in self.variable_serialization_spec:
            raise self._quantization_mode_error(mode)

        # Kernel plus optional merged LoRA-aware scale/zero (returns
        # (kernel, None, None) for None/gptq/awq)
        kernel_value, merged_kernel_scale, merged_kernel_zero = (
            self._get_kernel_with_merged_lora()
        )
        idx = 0
        for name in self.variable_serialization_spec[mode]:
            if name == "kernel":
                store[str(idx)] = kernel_value
            elif name == "bias" and self.bias is None:
                continue
            elif name == "kernel_zero":
                if merged_kernel_zero is None:
                    # kernel_zero only exists for sub-channel int4 quantization
                    continue
                store[str(idx)] = merged_kernel_zero
            elif name == "g_idx":
                if not hasattr(self, "g_idx"):
                    # g_idx only exists for sub-channel int4 quantization
                    continue
                store[str(idx)] = self.g_idx
            elif name == "kernel_scale" and mode in ("int4", "int8"):
                # For int4/int8, the merged LoRA scale (if any) comes from
                # `_get_kernel_with_merged_lora()`
                store[str(idx)] = merged_kernel_scale
            else:
                store[str(idx)] = getattr(self, name)
            idx += 1

    def load_own_variables(self, store):
        if not self.lora_enabled:
            self._check_load_own_variables(store)
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        mode = self.quantization_mode
        if mode not in self.variable_serialization_spec:
            raise self._quantization_mode_error(mode)

        # A saved GPTQ/AWQ quantized model will always be calibrated.
        self.is_gptq_calibrated = mode == "gptq"
        self.is_awq_calibrated = mode == "awq"

        idx = 0
        for name in self.variable_serialization_spec[mode]:
            if name == "kernel":
                self._kernel.assign(store[str(idx)])
            elif name == "bias" and self.bias is None:
                continue
            elif name == "kernel_zero" and not hasattr(self, "kernel_zero"):
                # kernel_zero only exists for sub-channel int4 quantization
                continue
            elif name == "g_idx" and not hasattr(self, "g_idx"):
                # g_idx only exists for sub-channel int4 quantization
                continue
            else:
                getattr(self, name).assign(store[str(idx)])
            idx += 1
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
            "quantization_config": serialization_lib.serialize_keras_object(
                self.quantization_config
            ),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
            config["lora_alpha"] = self.lora_alpha
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["quantization_config"] = (
            serialization_lib.deserialize_keras_object(
                config.get("quantization_config", None)
            )
        )
        return super().from_config(config)

    @property
    def variable_serialization_spec(self):
        """Returns a dict mapping quantization modes to variable names in order.

        This spec is used by `save_own_variables` and `load_own_variables` to
        determine the correct ordering of variables during serialization for
        each quantization mode. `None` means no quantization.
        """
        return {
            None: [
                "kernel",
                "bias",
            ],
            "int8": [
                "kernel",
                "bias",
                "kernel_scale",
            ],
            "int4": [
                "kernel",
                "bias",
                "kernel_scale",
                "kernel_zero",
                "g_idx",
            ],
            "float8": [
                "kernel",
                "bias",
                "inputs_scale",
                "inputs_amax_history",
                "kernel_scale",
                "kernel_amax_history",
                "outputs_grad_scale",
                "outputs_grad_amax_history",
            ],
            "gptq": [
                "bias",
                "quantized_kernel",
                "kernel_scale",
                "kernel_zero",
                "g_idx",
            ],
            "awq": [
                "bias",
                "quantized_kernel",
                "kernel_scale",
                "kernel_zero",
                "awq_scales",
                "g_idx",
            ],
        }

    def quantized_build(self, kernel_shape, mode, config=None):
        if mode == "int8":
            self._int8_build(kernel_shape, config)
        elif mode == "int4":
            self._int4_build(kernel_shape, config)
        elif mode == "float8":
            self._float8_build()
        elif mode == "gptq":
            self._gptq_build(kernel_shape, config)
        elif mode == "awq":
            self._awq_build(kernel_shape, config)
        else:
            raise self._quantization_mode_error(mode)
        self._is_quantized = True

    def _int8_build(self, kernel_shape, config=None):
        self.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(
                config, quantizers.AbsMaxQuantizer()
            )
        )

        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(self.units,),
            initializer="ones",
            trainable=False,
        )

    def _gptq_build(self, kernel_shape, config):
        from keras.src.quantizers import gptq_core

        # Ensures the forward pass uses the original high-precision kernel
        # until calibration has been performed.
        self.is_gptq_calibrated = False
        self.kernel_shape = kernel_shape

        weight_bits = gptq_core.get_weight_bits_for_layer(self, config)
        # For 4-bit weights, we pack two values per byte.
        units = (
            (kernel_shape[1] + 1) // 2 if weight_bits == 4 else kernel_shape[1]
        )

        self.quantized_kernel = self.add_weight(
            name="kernel",
            shape=(units, kernel_shape[0]),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )

        group_size = gptq_core.get_group_size_for_layer(self, config)
        n_groups = (
            1
            if group_size == -1
            else math.ceil(self.kernel_shape[0] / group_size)
        )
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(self.units, n_groups),
            initializer="ones",
            trainable=False,
        )
        self.kernel_zero = self.add_weight(
            name="kernel_zero",
            shape=(self.units, n_groups),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )
        self.g_idx = self.add_weight(
            name="g_idx",
            shape=(self.kernel_shape[0],),
            initializer="zeros",
            dtype="float32",
            trainable=False,
        )

    def _gptq_call(self, inputs, training=False):
        from keras.src.quantizers import gptq_core

        if not self.is_gptq_calibrated:
            W = self._kernel
        else:
            should_unpack = (
                gptq_core.get_weight_bits_for_layer(self, config=None) == 4
            )
            W = (
                quantizers.unpack_int4(
                    self.quantized_kernel,
                    orig_len=self.units,
                    axis=0,
                    dtype="uint8",
                )
                if should_unpack
                else self.quantized_kernel
            )
            W = ops.transpose(
                dequantize_with_sz_map(
                    W,
                    self.kernel_scale,
                    self.kernel_zero,
                    self.g_idx,
                )
            )

        y = ops.matmul(inputs, W)
        if self.bias is not None:
            y = ops.add(y, self.bias)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _awq_build(self, kernel_shape, config):
        """Build variables for AWQ quantization.

        AWQ uses 4-bit quantization with per-channel AWQ scales that protect
        salient weights based on activation magnitudes.
        """
        from keras.src.quantizers import awq_core

        # Ensures the forward pass uses the original high-precision kernel
        # until calibration has been performed.
        self.is_awq_calibrated = False
        self.kernel_shape = kernel_shape

        # For 4-bit weights, we pack two values per byte.
        units = (kernel_shape[1] + 1) // 2

        self.quantized_kernel = self.add_weight(
            name="kernel",
            shape=(units, kernel_shape[0]),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )

        group_size = awq_core.get_group_size_for_layer(self, config)
        num_groups = (
            1 if group_size == -1 else math.ceil(kernel_shape[0] / group_size)
        )
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(self.units, num_groups),
            initializer="ones",
            trainable=False,
        )
        self.kernel_zero = self.add_weight(
            name="kernel_zero",
            shape=(self.units, num_groups),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )

        # Per-channel AWQ scales from activation magnitudes
        self.awq_scales = self.add_weight(
            name="awq_scales",
            shape=(kernel_shape[0],),
            initializer="ones",
            trainable=False,
        )
        self.g_idx = self.add_weight(
            name="g_idx",
            shape=(kernel_shape[0],),
            initializer="zeros",
            dtype="float32",
            trainable=False,
        )

    def _awq_call(self, inputs, training=False):
        """Forward pass for AWQ quantized layer."""
        if not self.is_awq_calibrated:
            W = self._kernel
        else:
            # Unpack 4-bit weights
            W = quantizers.unpack_int4(
                self.quantized_kernel,
                orig_len=self.units,
                axis=0,
                dtype="uint8",
            )
            # Dequantize using scale/zero maps
            W = ops.transpose(
                dequantize_with_sz_map(
                    W,
                    self.kernel_scale,
                    self.kernel_zero,
                    self.g_idx,
                )
            )
            # Apply AWQ scales by dividing to restore original magnitude
            # (We multiplied by scales before quantization, so divide to undo)
            # awq_scales has shape [input_dim], W has shape [input_dim, units]
            # Expand dims for proper broadcasting.
            W = ops.divide(W, ops.expand_dims(self.awq_scales, -1))

        y = ops.matmul(inputs, W)
        if self.bias is not None:
            y = ops.add(y, self.bias)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _int4_build(self, kernel_shape, config=None):
        """Build variables for int4 quantization.

        The kernel is packed along the last axis,
        resulting in shape `(input_dim, ceil(units/2))`.

        Args:
            kernel_shape: The original float32 kernel shape
                `(input_dim, units)`.
            config: Optional quantization config specifying block_size.
        """
        self.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(config, None)
        )
        input_dim, output_dim = kernel_shape

        # kernel is packed along last axis (output dimension)
        # Stored shape: [input_dim, ceil(output_dim/2)]
        packed_cols = (output_dim + 1) // 2

        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, packed_cols),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )

        block_size = get_block_size_for_layer(self, config)
        self._int4_block_size = block_size

        if block_size is None or block_size == -1:
            # Per-channel: one scale per output unit
            scale_shape = (self.units,)
        else:
            # Sub-channel: [n_groups, out_features]
            n_groups = math.ceil(input_dim / block_size)
            scale_shape = (n_groups, self.units)

        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )

        # Sub-channel quantization uses asymmetric quantization
        if block_size is not None and block_size > 0:

            def idx_initializer(shape, dtype):
                return ops.floor_divide(
                    ops.arange(input_dim, dtype=dtype), block_size
                )

            self.kernel_zero = self.add_weight(
                name="kernel_zero",
                shape=scale_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            self.g_idx = self.add_weight(
                name="g_idx",
                shape=(input_dim,),
                initializer=idx_initializer,
                dtype="float32",
                trainable=False,
            )

        # Record dimensions for unpacking and reshaping at runtime.
        self._orig_input_dim = input_dim
        self._orig_output_dim = output_dim

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
            "overwrite_with_gradient": True,
        }
        amax_history_kwargs = {
            "shape": (amax_history_length,),
            "initializer": "zeros",
            "dtype": "float32",  # Always be float32
            "trainable": True,
            "autocast": False,
            "overwrite_with_gradient": True,
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

    def _int8_call(self, inputs, training=None):
        @ops.custom_gradient
        def matmul_with_inputs_gradient(inputs, kernel, kernel_scale):
            """Custom gradient function to handle the int8 quantized weights.

            Automatic differentiation will not know how to handle the int8
            quantized weights. So a custom gradient function is needed to
            handle the int8 quantized weights.

            The custom gradient function will use the dequantized kernel to
            compute the gradient.
            """

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=self.compute_dtype),
                    kernel_scale,
                )
                inputs_grad = ops.matmul(upstream, ops.transpose(float_kernel))
                return (inputs_grad, None, None)

            output_scale = kernel_scale
            if self.inputs_quantizer:
                inputs, inputs_scale = self.inputs_quantizer(inputs, axis=-1)
                output_scale = ops.multiply(output_scale, inputs_scale)

            x = ops.matmul(inputs, kernel)
            # De-scale outputs
            x = ops.cast(x, self.compute_dtype)
            x = ops.divide(x, output_scale)
            return x, grad_fn

        x = matmul_with_inputs_gradient(
            inputs,
            ops.convert_to_tensor(self._kernel),
            ops.convert_to_tensor(self.kernel_scale),
        )
        if self.lora_enabled:
            lora_x = ops.matmul(inputs, self.lora_kernel_a)
            lora_x = ops.matmul(lora_x, self.lora_kernel_b)
            x = ops.add(x, (self.lora_alpha / self.lora_rank) * lora_x)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _int4_call(self, inputs, training=None):
        """Forward pass for int4 quantized Dense layer.

        Uses custom gradients to handle quantized weights since autodiff
        cannot differentiate through int4 operations.
        """
        block_size = getattr(self, "_int4_block_size", None)

        if block_size is None or block_size == -1:
            # Per-channel: symmetric quantization (no zero point needed)
            @ops.custom_gradient
            def matmul_per_channel_with_inputs_gradient(
                inputs, kernel, kernel_scale
            ):
                """Per-channel int4 forward pass with custom gradient."""
                # Unpack: stored as [in, ceil(out/2)], unpack along last axis
                unpacked_kernel = quantizers.unpack_int4(
                    kernel, self._orig_output_dim, axis=-1
                )

                def grad_fn(*args, upstream=None):
                    if upstream is None:
                        (upstream,) = args
                    # Per-channel: unpacked is [in, out]
                    float_kernel = ops.divide(
                        ops.cast(unpacked_kernel, dtype=self.compute_dtype),
                        kernel_scale,
                    )
                    inputs_grad = ops.matmul(
                        upstream, ops.transpose(float_kernel)
                    )
                    return (inputs_grad, None, None)

                # Forward pass: per-channel dequantization
                output_scale = kernel_scale
                if self.inputs_quantizer:
                    inputs, inputs_scale = self.inputs_quantizer(
                        inputs, axis=-1
                    )
                    output_scale = ops.multiply(output_scale, inputs_scale)

                x = ops.matmul(inputs, unpacked_kernel)
                x = ops.cast(x, self.compute_dtype)
                x = ops.divide(x, output_scale)
                return x, grad_fn

            x = matmul_per_channel_with_inputs_gradient(
                inputs,
                ops.convert_to_tensor(self._kernel),
                ops.convert_to_tensor(self.kernel_scale),
            )
        else:
            # Sub-channel: asymmetric quantization (with zero point)
            @ops.custom_gradient
            def matmul_sub_channel_with_inputs_gradient(
                inputs, kernel, kernel_scale, kernel_zero, g_idx
            ):
                """Sub-channel int4 forward pass with custom gradient."""
                # Unpack: stored as [in, ceil(out/2)], unpack along last axis
                unpacked_kernel = quantizers.unpack_int4(
                    kernel, self._orig_output_dim, axis=-1
                )

                def grad_fn(*args, upstream=None):
                    if upstream is None:
                        (upstream,) = args
                    float_kernel = dequantize_with_sz_map(
                        unpacked_kernel,
                        kernel_scale,
                        kernel_zero,
                        g_idx,
                        group_axis=0,
                    )
                    float_kernel = ops.cast(float_kernel, self.compute_dtype)
                    inputs_grad = ops.matmul(
                        upstream, ops.transpose(float_kernel)
                    )
                    return (inputs_grad, None, None, None, None)

                float_kernel = dequantize_with_sz_map(
                    unpacked_kernel,
                    kernel_scale,
                    kernel_zero,
                    g_idx,
                    group_axis=0,
                )
                float_kernel = ops.cast(float_kernel, self.compute_dtype)
                x = ops.matmul(inputs, float_kernel)
                return x, grad_fn

            x = matmul_sub_channel_with_inputs_gradient(
                inputs,
                ops.convert_to_tensor(self._kernel),
                ops.convert_to_tensor(self.kernel_scale),
                ops.convert_to_tensor(self.kernel_zero),
                ops.convert_to_tensor(self.g_idx),
            )

        if self.lora_enabled:
            lora_x = ops.matmul(inputs, self.lora_kernel_a)
            lora_x = ops.matmul(lora_x, self.lora_kernel_b)
            x = ops.add(x, (self.lora_alpha / self.lora_rank) * lora_x)

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

    def quantize(self, mode=None, type_check=True, config=None):
        # Prevent quantization of the subclasses
        if type_check and (type(self) is not Dense):
            raise self._not_implemented_error(self.quantize)

        self.quantization_config = config

        kernel_shape = self._kernel.shape
        if mode == "int8":
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                self.quantization_config, quantizers.AbsMaxQuantizer(axis=0)
            )
            kernel_value, kernel_scale = weight_quantizer(
                self._kernel, to_numpy=True
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            del self._kernel
            # Build variables for int8 mode
            self.quantized_build(kernel_shape, mode, self.quantization_config)
            self._kernel.assign(kernel_value)
            self.kernel_scale.assign(kernel_scale)
        elif mode == "int4":
            from keras.src.quantizers.quantization_config import (
                Int4QuantizationConfig,
            )

            block_size = None
            if isinstance(self.quantization_config, Int4QuantizationConfig):
                block_size = self.quantization_config.block_size

            if block_size is None or block_size == -1:
                # Per-channel quantization
                weight_quantizer = (
                    QuantizationConfig.weight_quantizer_or_default(
                        self.quantization_config,
                        quantizers.AbsMaxQuantizer(
                            axis=0, value_range=(-8, 7), output_dtype="int8"
                        ),
                    )
                )
                kernel_value_int4, kernel_scale = weight_quantizer(
                    self._kernel, to_numpy=True
                )
                kernel_scale = ops.squeeze(kernel_scale, axis=0)
            else:
                # Sub-channel quantization with asymmetric zero point
                # Returns kernel [in, out], scale [n_groups, out], zero
                # [n_groups, out]
                kernel_value_int4, kernel_scale, kernel_zero = (
                    quantizers.abs_max_quantize_grouped_with_zero_point(
                        self._kernel, block_size=block_size, to_numpy=True
                    )
                )

            # Pack two int4 values per int8 byte along last axis
            # Stored as [in, ceil(out/2)]
            packed_kernel_value, _, _ = quantizers.pack_int4(
                kernel_value_int4, axis=-1
            )
            del self._kernel
            self.quantized_build(kernel_shape, mode, self.quantization_config)
            self._kernel.assign(packed_kernel_value)
            self.kernel_scale.assign(kernel_scale)
            if block_size is not None and block_size > 0:
                self.kernel_zero.assign(kernel_zero)
        elif mode == "gptq":
            self.quantized_build(kernel_shape, mode, self.quantization_config)
        elif mode == "awq":
            self.quantized_build(kernel_shape, mode, self.quantization_config)
        elif mode == "float8":
            self.quantized_build(kernel_shape, mode)
        else:
            raise self._quantization_mode_error(mode)

        # Set new dtype policy only for modes that already have a policy.
        if self.dtype_policy.quantization_mode is None:
            from keras.src import dtype_policies  # local import to avoid cycle

            policy_name = mode
            if mode in ("gptq", "awq"):
                policy_name = self.quantization_config.dtype_policy_string()
            elif mode == "int4":
                # Include block_size in policy name for sub-channel quantization
                block_size = get_block_size_for_layer(self, config)
                # Use -1 for per-channel, otherwise use block_size
                block_size_value = -1 if block_size is None else block_size
                policy_name = f"int4/{block_size_value}"
            policy = dtype_policies.get(
                f"{policy_name}_from_{self.dtype_policy.name}"
            )
            self.dtype_policy = policy

    def _get_kernel_with_merged_lora(self):
        """Returns the kernel with LoRA matrices merged, for serialization.

        This method is called by `save_own_variables` to produce a single
        kernel tensor that includes the adaptations from LoRA. This is useful
        for deploying the model or for continuing training after permanently
        applying the LoRA update.

        If the layer is quantized (`int8` or `int4`), the process is:
        1. Dequantize the base kernel to float.
        2. Compute the LoRA delta (`lora_kernel_a @ lora_kernel_b`) and add
            it to the dequantized kernel.
        3. Re-quantize the merged result back to the original quantized
            type (`int8` or packed `int4`), calculating a new scale factor.

        If the layer is not quantized, this method returns the result of the
        `kernel` property (which computes the merge in floating-point) and a
        scale of `None`.

        If LoRA is not enabled, it returns the original kernel and scale
        without modification.

        Returns:
            A tuple `(kernel_value, kernel_scale, kernel_zero)`:
                `kernel_value`: The merged kernel. A quantized tensor if
                    quantization is active, otherwise a high precision tensor.
                `kernel_scale`: The quantization scale for the merged kernel.
                    This is `None` if the layer is not quantized.
                `kernel_zero`: The zero point for sub-channel int4 quantization.
                    This is `None` for per-channel or non-int4 modes.
        """
        if self.dtype_policy.quantization_mode in (None, "gptq", "awq"):
            return self.kernel, None, None

        kernel_value = self._kernel
        kernel_scale = self.kernel_scale
        kernel_zero = getattr(self, "kernel_zero", None)

        if not self.lora_enabled:
            return kernel_value, kernel_scale, kernel_zero

        # Dequantize, Merge, and Re-quantize
        block_size = getattr(self, "_int4_block_size", None)

        # Step 1: Dequantize kernel to float
        if self.quantization_mode == "int4":
            # Unpack along last axis ([in, out])
            unpacked_kernel = quantizers.unpack_int4(
                kernel_value, self._orig_output_dim, axis=-1
            )
            if block_size is None or block_size == -1:
                # Per-channel: kernel [in, out], scale [out]
                float_kernel = ops.divide(
                    ops.cast(unpacked_kernel, self.compute_dtype),
                    kernel_scale,
                )
            else:
                # Sub-channel: scale/zero are [n_groups, out]
                float_kernel = dequantize_with_sz_map(
                    unpacked_kernel,
                    kernel_scale,
                    self.kernel_zero,
                    self.g_idx,
                    group_axis=0,
                )
                float_kernel = ops.cast(float_kernel, self.compute_dtype)
            quant_range = (-8, 7)
        elif self.quantization_mode == "int8":
            float_kernel = ops.divide(
                ops.cast(kernel_value, self.compute_dtype), kernel_scale
            )
            quant_range = (-127, 127)
        else:
            raise ValueError(
                f"Unsupported quantization mode: {self.quantization_mode}"
            )

        # Step 2: Merge LoRA weights in float domain
        lora_delta = (self.lora_alpha / self.lora_rank) * ops.matmul(
            self.lora_kernel_a, self.lora_kernel_b
        )
        merged_float_kernel = ops.add(float_kernel, lora_delta)

        # Step 3: Re-quantize the merged kernel
        if (
            self.quantization_mode == "int4"
            and block_size is not None
            and block_size != -1
        ):
            # Sub-channel: returns kernel [in, out], scale [n_groups, out]
            requantized_kernel, kernel_scale, kernel_zero = (
                quantizers.abs_max_quantize_grouped_with_zero_point(
                    merged_float_kernel, block_size=block_size, to_numpy=True
                )
            )
        elif self.quantization_mode == "int4":
            # Per-channel: quantize along input axis (axis=0)
            requantized_kernel, kernel_scale = quantizers.abs_max_quantize(
                merged_float_kernel,
                axis=0,
                value_range=quant_range,
                dtype="int8",
                to_numpy=True,
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            kernel_zero = None
        else:
            requantized_kernel, kernel_scale = quantizers.abs_max_quantize(
                merged_float_kernel,
                axis=0,
                value_range=quant_range,
                dtype="int8",
                to_numpy=True,
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            kernel_zero = None

        if self.quantization_mode == "int4":
            # Pack along last axis
            kernel_value, _, _ = quantizers.pack_int4(
                requantized_kernel, axis=-1
            )
        else:
            kernel_value = requantized_kernel
        return kernel_value, kernel_scale, kernel_zero
