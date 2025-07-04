import re
import string

import ml_dtypes
import numpy as np

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


@keras_export("keras.layers.EinsumDense")
class EinsumDense(Layer):
    """A layer that uses `einsum` as the backing computation.

    This layer can perform einsum calculations of arbitrary dimensionality.

    Args:
        equation: An equation describing the einsum to perform.
            This equation must be a valid einsum string of the form
            `ab,bc->ac`, `...ab,bc->...ac`, or
            `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
            axis expression sequence.
        output_shape: The expected shape of the output tensor
            (excluding the batch dimension and any dimensions
            represented by ellipses). You can specify `None` for any dimension
            that is unknown or can be inferred from the input shape.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied
            (that is, a "linear" activation: `a(x) = x`).
        bias_axes: A string containing the output dimension(s)
            to apply a bias to. Each character in the `bias_axes` string
            should correspond to a character in the output portion
            of the `equation` string.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix.
        bias_constraint: Constraint function applied to the bias vector.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices
            (the factorization happens on the last dimension).
            This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `EinsumDense` layer by calling `layer.enable_lora(rank)`.
         lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Examples:

    **Biased dense layer with einsums**

    This example shows how to instantiate a standard Keras dense layer using
    einsum operations. This example is equivalent to
    `keras.layers.Dense(64, use_bias=True)`.

    >>> layer = keras.layers.EinsumDense("ab,bc->ac",
    ...                                       output_shape=64,
    ...                                       bias_axes="c")
    >>> input_tensor = keras.Input(shape=[32])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 64)

    **Applying a dense layer to a sequence**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence. Here, the `output_shape` has two
    values (since there are two non-batch dimensions in the output); the first
    dimension in the `output_shape` is `None`, because the sequence dimension
    `b` has an unknown shape.

    >>> layer = keras.layers.EinsumDense("abc,cd->abd",
    ...                                       output_shape=(None, 64),
    ...                                       bias_axes="d")
    >>> input_tensor = keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 32, 64)

    **Applying a dense layer to a sequence using ellipses**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence, but uses the ellipsis notation
    instead of specifying the batch and sequence dimensions.

    Because we are using ellipsis notation and have specified only one axis, the
    `output_shape` arg is a single value. When instantiated in this way, the
    layer can handle any number of sequence dimensions - including the case
    where no sequence dimension exists.

    >>> layer = keras.layers.EinsumDense("...x,xy->...y",
    ...                                       output_shape=64,
    ...                                       bias_axes="y")
    >>> input_tensor = keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 32, 64)
    """

    def __init__(
        self,
        equation,
        output_shape,
        activation=None,
        bias_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        lora_alpha=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.equation = equation
        if isinstance(output_shape, int):
            self.partial_output_shape = (output_shape,)
        else:
            self.partial_output_shape = tuple(output_shape)
        self.bias_axes = bias_axes
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_enabled = False

    def build(self, input_shape):
        shape_data = _analyze_einsum_string(
            self.equation,
            self.bias_axes,
            input_shape,
            self.partial_output_shape,
        )
        kernel_shape, bias_shape, full_output_shape = shape_data
        self.full_output_shape = tuple(full_output_shape)
        self.input_spec = InputSpec(ndim=len(input_shape))
        if self.quantization_mode is not None:
            self.quantized_build(kernel_shape, mode=self.quantization_mode)
        if self.quantization_mode != "int8":
            # If the layer is quantized to int8, `self._kernel` will be added
            # in `self._int8_build`. Therefore, we skip it here.
            self._kernel = self.add_weight(
                name="kernel",
                shape=tuple(kernel_shape),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        if bias_shape is not None:
            self.bias = self.add_weight(
                name="bias",
                shape=tuple(bias_shape),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank, lora_alpha=self.lora_alpha)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        if self.lora_enabled:
            return self._kernel + (
                self.lora_alpha / self.lora_rank
            ) * ops.matmul(self.lora_kernel_a, self.lora_kernel_b)
        return self._kernel

    def compute_output_shape(self, _):
        return self.full_output_shape

    def call(self, inputs, training=None):
        x = ops.einsum(self.equation, inputs, self.kernel)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

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
        self._tracker.unlock()
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(self.kernel.shape[:-1] + (rank,)),
            initializer=initializers.get(a_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.kernel.shape[-1]),
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
        # The keys of the `store` will be saved as determined because the
        # default ordering will change after quantization
        kernel_value, kernel_scale = self._get_kernel_with_merged_lora()
        target_variables = [kernel_value]
        if self.bias is not None:
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
        if self.bias is not None:
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
            "output_shape": self.partial_output_shape,
            "equation": self.equation,
            "activation": activations.serialize(self.activation),
            "bias_axes": self.bias_axes,
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
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
            config["lora_alpha"] = self.lora_alpha
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

    def quantized_build(self, kernel_shape, mode):
        if mode == "int8":
            self._int8_build(kernel_shape)
        elif mode == "float8":
            self._float8_build()
        else:
            raise self._quantization_mode_error(mode)
        self._is_quantized = True

    def _int8_build(self, kernel_shape):
        (
            self._input_reduced_axes,
            self._kernel_reduced_axes,
            self._input_transpose_axes,
            self._kernel_transpose_axes,
            self._input_expand_axes,
            self._kernel_expand_axes,
            self._input_squeeze_axes,
            self._kernel_squeeze_axes,
            self._custom_gradient_equation,
            self._kernel_reverse_transpose_axes,
        ) = _analyze_quantization_info(self.equation, self.input_spec.ndim)
        self.inputs_quantizer = quantizers.AbsMaxQuantizer(
            axis=self._input_reduced_axes
        )
        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        kernel_scale_shape = np.array(kernel_shape)
        kernel_scale_shape[self._kernel_reduced_axes] = 1
        kernel_scale_shape = kernel_scale_shape[self._kernel_transpose_axes]
        kernel_scale_shape = kernel_scale_shape.tolist()
        for a in sorted(self._kernel_expand_axes):
            kernel_scale_shape.insert(a, 1)
        for a in sorted(self._kernel_squeeze_axes, reverse=True):
            kernel_scale_shape.pop(a)
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=kernel_scale_shape,
            initializer="ones",
            trainable=False,
        )

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
        def einsum_with_inputs_gradient(inputs, kernel, kernel_scale):
            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                # De-scale kernel
                _kernel_scale = kernel_scale  # Overcome UnboundLocalError
                if self._kernel_squeeze_axes:
                    _kernel_scale = ops.expand_dims(
                        _kernel_scale, axis=self._kernel_squeeze_axes
                    )
                if self._kernel_expand_axes:
                    _kernel_scale = ops.squeeze(
                        _kernel_scale, axis=self._kernel_expand_axes
                    )
                _kernel_scale = ops.transpose(
                    _kernel_scale, self._kernel_reverse_transpose_axes
                )
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=self.compute_dtype),
                    _kernel_scale,
                )
                # From https://stackoverflow.com/a/47609896
                inputs_grad = ops.einsum(
                    self._custom_gradient_equation, upstream, float_kernel
                )
                return (inputs_grad, None, None)

            inputs, inputs_scale = self.inputs_quantizer(inputs)
            x = ops.einsum(self.equation, inputs, kernel)
            # Deal with `inputs_scale`
            inputs_scale = ops.transpose(
                inputs_scale, self._input_transpose_axes
            )
            if self._input_expand_axes:
                inputs_scale = ops.expand_dims(
                    inputs_scale, axis=self._input_expand_axes
                )
            if self._input_squeeze_axes:
                inputs_scale = ops.squeeze(
                    inputs_scale, axis=self._input_squeeze_axes
                )
            # De-scale outputs
            x = ops.cast(x, self.compute_dtype)
            x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
            return x, grad_fn

        x = einsum_with_inputs_gradient(
            inputs,
            ops.convert_to_tensor(self._kernel),
            ops.convert_to_tensor(self.kernel_scale),
        )
        if self.lora_enabled:
            lora_x = ops.einsum(self.equation, inputs, self.lora_kernel_a)
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

        x = ops.einsum(
            self.equation,
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
        # `ops.einsum` for the sake of pattern matching in gemm_rewrite. That
        # way, the qdq will be adjacent to the corresponding einsum_bprop in the
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
        # Prevent quantization of the subclasses
        if type_check and (type(self) is not EinsumDense):
            raise self._not_implemented_error(self.quantize)

        kernel_shape = self._kernel.shape
        if mode == "int8":
            (
                self._input_reduced_axes,
                self._kernel_reduced_axes,
                self._input_transpose_axes,
                self._kernel_transpose_axes,
                self._input_expand_axes,
                self._kernel_expand_axes,
                self._input_squeeze_axes,
                self._kernel_squeeze_axes,
                self._custom_gradient_equation,
                self._kernel_reverse_transpose_axes,
            ) = _analyze_quantization_info(self.equation, self.input_spec.ndim)
            # Quantize `self._kernel` to int8 and compute corresponding scale
            kernel_value, kernel_scale = quantizers.abs_max_quantize(
                self._kernel, axis=self._kernel_reduced_axes, to_numpy=True
            )
            kernel_scale = ops.transpose(
                kernel_scale, self._kernel_transpose_axes
            )
            if self._kernel_expand_axes:
                kernel_scale = ops.expand_dims(
                    kernel_scale, axis=self._kernel_expand_axes
                )
            if self._kernel_squeeze_axes:
                kernel_scale = ops.squeeze(
                    kernel_scale, axis=self._kernel_squeeze_axes
                )
            del self._kernel
        self.quantized_build(kernel_shape, mode)
        if mode == "int8":
            self._kernel.assign(kernel_value)
            self.kernel_scale.assign(kernel_scale)

        # Set new dtype policy
        if self.dtype_policy.quantization_mode is None:
            policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
            self.dtype_policy = policy

    def _get_kernel_with_merged_lora(self):
        if self.dtype_policy.quantization_mode is not None:
            kernel_value = self._kernel
            kernel_scale = self.kernel_scale
            if self.lora_enabled:
                # Dequantize & quantize to merge lora weights into int8 kernel
                # Note that this is a lossy compression
                if self._kernel_squeeze_axes:
                    kernel_scale = ops.expand_dims(
                        kernel_scale, axis=self._kernel_squeeze_axes
                    )
                if self._kernel_expand_axes:
                    kernel_scale = ops.squeeze(
                        kernel_scale, axis=self._kernel_expand_axes
                    )
                if self._kernel_transpose_axes:

                    def _argsort(seq):
                        # Ref: https://stackoverflow.com/a/3382369
                        return sorted(range(len(seq)), key=seq.__getitem__)

                    reverse_transpose = _argsort(self._kernel_transpose_axes)
                    kernel_scale = ops.transpose(
                        kernel_scale, axes=reverse_transpose
                    )
                kernel_value = ops.divide(kernel_value, kernel_scale)
                kernel_value = ops.add(
                    kernel_value,
                    (self.lora_alpha / self.lora_rank)
                    * ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
                )
                kernel_value, kernel_scale = quantizers.abs_max_quantize(
                    kernel_value, axis=self._kernel_reduced_axes, to_numpy=True
                )
                kernel_scale = ops.transpose(
                    kernel_scale, self._kernel_transpose_axes
                )
                if self._kernel_expand_axes:
                    kernel_scale = ops.expand_dims(
                        kernel_scale, axis=self._kernel_expand_axes
                    )
                if self._kernel_squeeze_axes:
                    kernel_scale = ops.squeeze(
                        kernel_scale, axis=self._kernel_squeeze_axes
                    )
        else:
            kernel_value = self.kernel
            kernel_scale = None
        return kernel_value, kernel_scale


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
    """Analyzes an einsum string to determine the required weight shape."""

    dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

    # This is the case where no ellipses are present in the string.
    split_string = re.match(
        "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    # This is the case where ellipses are present on the left.
    split_string = re.match(
        "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape, left_elided=True
        )

    # This is the case where ellipses are present on the right.
    split_string = re.match(
        "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", dot_replaced_string
    )
    if split_string:
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    raise ValueError(
        f"Invalid einsum equation '{equation}'. Equations must be in the form "
        "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
    )


def _analyze_split_string(
    split_string, bias_axes, input_shape, output_shape, left_elided=False
):
    """Analyze an pre-split einsum string to find the weight shape."""
    input_spec = split_string.group(1)
    weight_spec = split_string.group(2)
    output_spec = split_string.group(3)
    elided = len(input_shape) - len(input_spec)

    if isinstance(output_shape, int):
        output_shape = [output_shape]
    else:
        output_shape = list(output_shape)

    output_shape.insert(0, input_shape[0])

    if elided > 0 and left_elided:
        for i in range(1, elided):
            # We already inserted the 0th input dimension at dim 0, so we need
            # to start at location 1 here.
            output_shape.insert(1, input_shape[i])
    elif elided > 0 and not left_elided:
        for i in range(len(input_shape) - elided, len(input_shape)):
            output_shape.append(input_shape[i])

    if left_elided:
        # If we have beginning dimensions elided, we need to use negative
        # indexing to determine where in the input dimension our values are.
        input_dim_map = {
            dim: (i + elided) - len(input_shape)
            for i, dim in enumerate(input_spec)
        }
        # Because we've constructed the full output shape already, we don't need
        # to do negative indexing.
        output_dim_map = {
            dim: (i + elided) for i, dim in enumerate(output_spec)
        }
    else:
        input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
        output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

    for dim in input_spec:
        input_shape_at_dim = input_shape[input_dim_map[dim]]
        if dim in output_dim_map:
            output_shape_at_dim = output_shape[output_dim_map[dim]]
            if (
                output_shape_at_dim is not None
                and output_shape_at_dim != input_shape_at_dim
            ):
                raise ValueError(
                    "Input shape and output shape do not match at shared "
                    f"dimension '{dim}'. Input shape is {input_shape_at_dim}, "
                    "and output shape "
                    f"is {output_shape[output_dim_map[dim]]}."
                )

    for dim in output_spec:
        if dim not in input_spec and dim not in weight_spec:
            raise ValueError(
                f"Dimension '{dim}' was specified in the output "
                f"'{output_spec}' but has no corresponding dim in the input "
                f"spec '{input_spec}' or weight spec '{output_spec}'"
            )

    weight_shape = []
    for dim in weight_spec:
        if dim in input_dim_map:
            weight_shape.append(input_shape[input_dim_map[dim]])
        elif dim in output_dim_map:
            weight_shape.append(output_shape[output_dim_map[dim]])
        else:
            raise ValueError(
                f"Weight dimension '{dim}' did not have a match in either "
                f"the input spec '{input_spec}' or the output "
                f"spec '{output_spec}'. For this layer, the weight must "
                "be fully specified."
            )

    if bias_axes is not None:
        num_left_elided = elided if left_elided else 0
        idx_map = {
            char: output_shape[i + num_left_elided]
            for i, char in enumerate(output_spec)
        }

        for char in bias_axes:
            if char not in output_spec:
                raise ValueError(
                    f"Bias dimension '{char}' was requested, but is not part "
                    f"of the output spec '{output_spec}'"
                )

        first_bias_location = min(
            [output_spec.find(char) for char in bias_axes]
        )
        bias_output_spec = output_spec[first_bias_location:]

        bias_shape = [
            idx_map[char] if char in bias_axes else 1
            for char in bias_output_spec
        ]

        if not left_elided:
            for _ in range(elided):
                bias_shape.append(1)
    else:
        bias_shape = None

    return weight_shape, bias_shape, output_shape


def _analyze_quantization_info(equation, input_shape):
    def get_specs(equation, input_shape):
        possible_labels = string.ascii_letters
        dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

        # This is the case where no ellipses are present in the string.
        split_string = re.match(
            "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", dot_replaced_string
        )
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            return input_spec, weight_spec, output_spec

        # This is the case where ellipses are present on the left.
        split_string = re.match(
            "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", dot_replaced_string
        )
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            elided = len(input_shape) - len(input_spec)
            possible_labels = sorted(
                set(possible_labels)
                - set(input_spec)
                - set(weight_spec)
                - set(output_spec)
            )
            # Pad labels on the left to `input_spec` and `output_spec`
            for i in range(elided):
                input_spec = possible_labels[i] + input_spec
                output_spec = possible_labels[i] + output_spec
            return input_spec, weight_spec, output_spec

        # This is the case where ellipses are present on the right.
        split_string = re.match(
            "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", dot_replaced_string
        )
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            elided = len(input_shape) - len(input_spec)
            possible_labels = sorted(
                set(possible_labels)
                - set(input_spec)
                - set(weight_spec)
                - set(output_spec)
            )
            # Pad labels on the right to `input_spec` and `output_spec`
            for i in range(elided):
                input_spec = input_spec + possible_labels[i]
                output_spec = output_spec + possible_labels[i]
            return input_spec, weight_spec, output_spec

        raise ValueError(
            f"Invalid einsum equation '{equation}'. Equations must be in the "
            "form [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
        )

    input_spec, weight_spec, output_spec = get_specs(equation, input_shape)

    # Determine the axes that should be reduced by the quantizer
    input_reduced_axes = []
    weight_reduced_axes = []
    for i, label in enumerate(input_spec):
        index = output_spec.find(label)
        if index == -1:
            input_reduced_axes.append(i)
    for i, label in enumerate(weight_spec):
        index = output_spec.find(label)
        if index == -1:
            weight_reduced_axes.append(i)

    # Determine the axes of `ops.expand_dims`
    input_expand_axes = []
    weight_expand_axes = []
    for i, label in enumerate(output_spec):
        index_input = input_spec.find(label)
        index_weight = weight_spec.find(label)
        if index_input == -1:
            input_expand_axes.append(i)
        if index_weight == -1:
            weight_expand_axes.append(i)

    # Determine the axes of `ops.transpose`
    input_transpose_axes = []
    weight_transpose_axes = []
    for i, label in enumerate(output_spec):
        index_input = input_spec.find(label)
        index_weight = weight_spec.find(label)
        if index_input != -1:
            input_transpose_axes.append(index_input)
        if index_weight != -1:
            weight_transpose_axes.append(index_weight)
    # Postprocess the information:
    # 1. Add dummy axes (1) to transpose_axes
    # 2. Add axis to squeeze_axes if 1. failed
    input_squeeze_axes = []
    weight_squeeze_axes = []
    for ori_index in input_reduced_axes:
        try:
            index = input_expand_axes.pop(0)
        except IndexError:
            input_squeeze_axes.append(ori_index)
        input_transpose_axes.insert(index, ori_index)
    for ori_index in weight_reduced_axes:
        try:
            index = weight_expand_axes.pop(0)
        except IndexError:
            weight_squeeze_axes.append(ori_index)
        weight_transpose_axes.insert(index, ori_index)
    # Prepare equation for `einsum_with_inputs_gradient`
    custom_gradient_equation = f"{output_spec},{weight_spec}->{input_spec}"
    weight_reverse_transpose_axes = [
        i
        for (_, i) in sorted(
            (v, i) for (i, v) in enumerate(weight_transpose_axes)
        )
    ]
    return (
        input_reduced_axes,
        weight_reduced_axes,
        input_transpose_axes,
        weight_transpose_axes,
        input_expand_axes,
        weight_expand_axes,
        input_squeeze_axes,
        weight_squeeze_axes,
        custom_gradient_equation,
        weight_reverse_transpose_axes,
    )
