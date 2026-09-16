from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.geometry import ProjectionGeometry
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
        strategy = strategy_registry.get_strategy(self.quantization_mode)
        if strategy is None or not strategy.owns_weight_storage:
            # Modes that own their weight storage created the kernel in
            # quantized_build.
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
        # Resolved once when the GPTQ variables are built, so the
        # inference path never re-parses the dtype policy.
        gptq_bits = self._gptq_weight_bits if is_gptq else None

        # Decide the source tensor first (packed vs already-quantized vs plain
        # kernel)
        if mode == "ternary":
            # Ternary: unpack to int8 {-1, 0, +1} float view.
            return quantizers.unpack_ternary(
                self._packed_kernel, self._orig_input_dim, axis=0
            )
        if is_gptq and gptq_calibrated and gptq_bits not in (2, 4):
            # calibrated GPTQ, not a packed bit-width, no unpacking needed
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
            elif is_gptq and gptq_calibrated and gptq_bits == 2:
                kernel = quantizers.unpack_int2(
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
            kernel = ops.cast(
                ops.add(
                    kernel,
                    (self.lora_alpha / self.lora_rank)
                    * ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
                ),
                dtype=self.compute_dtype,
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

        # LoRA weights should be float32 to avoid the risk of underflow or
        # overflow during fine-tuning.
        # When deploying the model, these weights should be merged with the
        # original kernel while maintaining the original kernel's dtype.
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(input_dim_for_lora, rank),
            initializer=initializers.get(a_initializer),
            dtype="float32",
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.kernel.shape[1]),
            initializer=initializers.get(b_initializer),
            dtype="float32",
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

        # GPTQ/AWQ layers are only serializable after calibration. Before
        # calibration, the quantized variables hold uninitialized values
        # while the real weights live in the float `_kernel`, which has no
        # slot in the serialization spec, so saving would silently drop the
        # actual weights and produce a corrupted model on reload.
        if (
            mode == "gptq" and not getattr(self, "is_gptq_calibrated", False)
        ) or (mode == "awq" and not getattr(self, "is_awq_calibrated", False)):
            raise ValueError(
                f"Cannot save layer '{self.name}' because it is quantized "
                f"with mode '{mode}' but has never been calibrated. Its "
                "quantized weights are uninitialized, so saving would "
                "produce a corrupted model. Run calibration first, e.g. via "
                "`model.quantize(...)` with a quantization layer structure "
                "that covers this layer, or exclude the layer from "
                "quantization with `filters`."
            )

        # Kernel plus optional merged LoRA-aware scale/zero (returns
        # (kernel, None, None) for None/gptq/awq)
        kernel_value, merged_kernel_scale, merged_kernel_zero = (
            self._get_kernel_with_merged_lora()
        )
        # Variables are stored under their integer position ("0", "1", ...)
        # within the mode's serialization spec. Each branch picks the value
        # for the current spec entry (or skips it); the write happens at a
        # single point so save and load stay position-consistent.
        idx = 0
        for name in self.variable_serialization_spec[mode]:
            if name == "kernel":
                if mode == "ternary":
                    value = self._packed_kernel
                else:
                    value = kernel_value
            elif name == "bias" and self.bias is None:
                continue
            elif name == "kernel_zero" and mode == "int4":
                # For int4, the (LoRA-merged) zero point comes from
                # `_get_kernel_with_merged_lora()` and only exists for
                # sub-channel quantization.
                if merged_kernel_zero is None:
                    continue
                value = merged_kernel_zero
            elif name == "g_idx":
                if not hasattr(self, "g_idx"):
                    # g_idx only exists for sub-channel int4 quantization
                    continue
                value = self.g_idx
            elif name == "kernel_scale" and mode in ("int4", "int8"):
                # For int4/int8, the merged LoRA scale (if any) comes from
                # `_get_kernel_with_merged_lora()`
                value = merged_kernel_scale
            else:
                value = getattr(self, name)
            store[str(idx)] = value
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

        spec = self.variable_serialization_spec[mode]
        # Variables are keyed by their integer position ("0", "1", ...) within
        # the mode's serialization spec. Each branch picks the target variable
        # for the current spec entry (or skips it); the assign happens at a
        # single point so save and load stay position-consistent.
        idx = 0
        for name in spec:
            key = str(idx)
            if name == "kernel":
                target = (
                    self._packed_kernel if mode == "ternary" else self._kernel
                )
            elif name == "bias" and self.bias is None:
                continue
            elif name == "kernel_zero" and not hasattr(self, "kernel_zero"):
                # kernel_zero only exists for sub-channel int4 quantization
                continue
            elif name == "g_idx":
                if not hasattr(self, "g_idx"):
                    # g_idx only exists for sub-channel int4 quantization
                    continue
                # `g_idx` is stored as `float32` (see build). Cast to the
                # variable dtype on assign so both legacy `float32`
                # checkpoints and any `int32`-saved ones load correctly.
                self.g_idx.assign(ops.cast(store[key], self.g_idx.dtype))
                idx += 1
                continue
            elif name == "quantized_kernel" and mode == "gptq":
                # Handles legacy unpacked 2-bit layouts.
                self._assign_gptq_quantized_kernel(store[key])
                idx += 1
                continue
            else:
                target = getattr(self, name)
            target.assign(store[key])
            idx += 1
        if self.lora_enabled:
            self.lora_kernel_a.assign(ops.zeros(self.lora_kernel_a.shape))
            self.lora_kernel_b.assign(ops.zeros(self.lora_kernel_b.shape))

    def _assign_gptq_quantized_kernel(self, value):
        """Assigns a stored GPTQ quantized kernel, handling legacy layouts.

        Older checkpoints stored 2-bit GPTQ kernels unpacked (one value per
        uint8 byte). Current checkpoints pack four 2-bit values per byte. When
        a legacy unpacked 2-bit store is detected by shape, it is packed on load
        so the inference path can always unpack a packed kernel.
        """
        if self._gptq_weight_bits == 2 and tuple(value.shape) != tuple(
            self.quantized_kernel.shape
        ):
            value, _, _ = quantizers.pack_int2(
                ops.cast(value, "uint8"), axis=0, dtype="uint8"
            )
        self.quantized_kernel.assign(value)

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
            "ternary": [
                "kernel",
                "bias",
                "kernel_scale",
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

    def quantize(self, mode=None, type_check=True, config=None):
        # Prevent quantization of the subclasses.
        if type_check and type(self) is not Dense:
            raise self._not_implemented_error(self.quantize)
        self._registry_quantize(mode, config)

    def _quantization_geometry(self):
        return ProjectionGeometry(self)

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
        if self.dtype_policy.quantization_mode == "ternary":
            return self._packed_kernel, None, None

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
