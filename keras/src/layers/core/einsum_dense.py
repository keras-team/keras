import math
import re
import string

import ml_dtypes
import numpy as np

from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
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
        gptq_unpacked_column_size=None,
        quantization_config=None,
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
        self.gptq_unpacked_column_size = gptq_unpacked_column_size
        self.quantization_config = quantization_config

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
            self.quantized_build(
                kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        # Skip creating a duplicate kernel variable when the layer is already
        # quantized to int8 or int4, because `quantized_build` has created the
        # appropriate kernel variable. For other modes (e.g., float8 or no
        # quantization), we still need the floating-point kernel.
        if self.quantization_mode not in ("int8", "int4", "gptq", "awq"):
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
                # unpack [rows, ceil(columns/2)] to [rows, columns]
                kernel = quantizers.unpack_int4(
                    kernel,
                    self._int4_unpacked_column_size,
                    axis=-1,
                )
                kernel = ops.reshape(kernel, self.original_kernel_shape)
            elif is_gptq and gptq_calibrated and gptq_bits == 4:
                kernel = quantizers.unpack_int4(
                    self.quantized_kernel,
                    orig_len=self.gptq_unpacked_column_size,
                    axis=0,
                    dtype="uint8",
                )
            elif is_awq and awq_calibrated:
                # AWQ always uses 4-bit quantization
                kernel = quantizers.unpack_int4(
                    self.quantized_kernel,
                    orig_len=self.awq_unpacked_column_size,
                    axis=0,
                    dtype="uint8",
                )

        # Apply LoRA if enabled
        if self.lora_enabled:
            kernel = kernel + (self.lora_alpha / self.lora_rank) * ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )

        return kernel

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
        if self.quantization_mode == "gptq":
            raise NotImplementedError(
                "lora is not currently supported with GPTQ quantization."
            )
        self._tracker.unlock()
        # Determine the appropriate (unpacked) kernel shape for LoRA.
        if self.quantization_mode == "int4":
            # INT4 weights are stored in a flattened 2D layout that loses
            # the original N-dimensional structure required by the einsum
            # equation. We use `original_kernel_shape`` to ensure LoRA adapters
            # operate in the correct logical dimension space.
            kernel_shape_for_lora = tuple(self.original_kernel_shape)
        else:
            kernel_shape_for_lora = self.kernel.shape

        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(kernel_shape_for_lora[:-1] + (rank,)),
            initializer=initializers.get(a_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, kernel_shape_for_lora[-1]),
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
        # (kernel, None, None) for None/gptq)
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
            "quantization_config": serialization_lib.serialize_keras_object(
                self.quantization_config
            ),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
            config["lora_alpha"] = self.lora_alpha
        if self.gptq_unpacked_column_size:
            config["gptq_unpacked_column_size"] = self.gptq_unpacked_column_size
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
        self._set_quantization_info()
        self.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(
                config,
                quantizers.AbsMaxQuantizer(),
            )
        )
        # If the config provided a default AbsMaxQuantizer, we need to
        # override the axis to match the equation's reduction axes.
        self.quantization_axis = tuple(self._input_reduced_axes)
        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        kernel_scale_shape = self._get_kernel_scale_shape(kernel_shape)
        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=kernel_scale_shape,
            initializer="ones",
            trainable=False,
        )

    def _gptq_build(self, kernel_shape, config):
        """
        Allocate quantized kernel & params for EinsumDense.

        Args:
            kernel_shape: tuple/list; the layer's original kernel shape, e.g.
                [in_features, out_features] or [in_features, heads, head_dim].
            group_size: int; contiguous input-group size for quantization
                (=-1 means per-output-channel with no grouping).
        """
        from keras.src.quantizers import gptq_core

        # Ensures the forward pass uses the original high-precision kernel
        # until calibration has been performed.
        self.is_gptq_calibrated = False

        self.original_kernel_shape = kernel_shape
        if len(kernel_shape) == 2:
            rows = kernel_shape[0]
            columns = kernel_shape[1]
        elif len(kernel_shape) == 3:
            shape = list(self.original_kernel_shape)
            d_model_dim_index = shape.index(max(shape))

            if d_model_dim_index == 0:  # QKV projection case
                in_features, heads, head_dim = shape
                rows, columns = (
                    in_features,
                    heads * head_dim,
                )
            elif d_model_dim_index in [1, 2]:  # Attention Output case
                heads, head_dim, out_features = shape
                rows, columns = (
                    heads * head_dim,
                    out_features,
                )
            else:
                raise ValueError("Could not determine row/column split.")

        group_size = gptq_core.get_group_size_for_layer(self, config)
        n_groups = 1 if group_size == -1 else math.ceil(rows / group_size)

        self.gptq_unpacked_column_size = columns

        weight_bits = gptq_core.get_weight_bits_for_layer(self, config)
        # For 4-bit weights, we pack two values per byte.
        kernel_columns = (columns + 1) // 2 if weight_bits == 4 else columns

        self._set_quantization_info()

        self.quantized_kernel = self.add_weight(
            name="kernel",
            shape=(kernel_columns, rows),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )

        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(columns, n_groups),
            initializer="ones",
            trainable=False,
        )
        self.kernel_zero = self.add_weight(
            name="zero_point",
            shape=(columns, n_groups),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )

        self.g_idx = self.add_weight(
            name="g_idx",
            shape=(rows,),
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
                    orig_len=self.gptq_unpacked_column_size,
                    axis=0,
                    dtype="uint8",
                )
                if should_unpack
                else self.quantized_kernel
            )
            W = dequantize_with_sz_map(
                W,
                self.kernel_scale,
                self.kernel_zero,
                self.g_idx,
            )
            W = ops.transpose(W)

            W = ops.reshape(W, self.original_kernel_shape)

        y = ops.einsum(self.equation, inputs, W)
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

        self.original_kernel_shape = kernel_shape
        if len(kernel_shape) == 2:
            rows = kernel_shape[0]
            columns = kernel_shape[1]
        elif len(kernel_shape) == 3:
            shape = list(self.original_kernel_shape)
            d_model_dim_index = shape.index(max(shape))

            if d_model_dim_index == 0:  # QKV projection case
                in_features, heads, head_dim = shape
                rows, columns = (
                    in_features,
                    heads * head_dim,
                )
            elif d_model_dim_index in [1, 2]:  # Attention Output case
                heads, head_dim, out_features = shape
                rows, columns = (
                    heads * head_dim,
                    out_features,
                )
            else:
                raise ValueError("Could not determine row/column split.")
        else:
            raise ValueError("AWQ quantization only supports 2D or 3D kernels.")

        group_size = awq_core.get_group_size_for_layer(self, config)
        num_groups = 1 if group_size == -1 else math.ceil(rows / group_size)

        self.awq_unpacked_column_size = columns

        # For 4-bit weights, we pack two values per byte.
        kernel_columns = (columns + 1) // 2

        self._set_quantization_info()

        self.quantized_kernel = self.add_weight(
            name="kernel",
            shape=(kernel_columns, rows),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )

        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=(columns, num_groups),
            initializer="ones",
            trainable=False,
        )
        self.kernel_zero = self.add_weight(
            name="zero_point",
            shape=(columns, num_groups),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )

        # Per-channel AWQ scales from activation magnitudes
        self.awq_scales = self.add_weight(
            name="awq_scales",
            shape=(rows,),
            initializer="ones",
            trainable=False,
        )

        self.g_idx = self.add_weight(
            name="g_idx",
            shape=(rows,),
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
                orig_len=self.awq_unpacked_column_size,
                axis=0,
                dtype="uint8",
            )
            # Dequantize using scale/zero maps
            W = dequantize_with_sz_map(
                W,
                self.kernel_scale,
                self.kernel_zero,
                self.g_idx,
            )
            W = ops.transpose(W)

            # Apply AWQ scales by dividing to restore original magnitude
            # (We multiplied by scales before quantization, so divide to undo)
            # awq_scales has shape [input_dim], W has shape [input_dim, out_dim]
            # Expand dims for proper broadcasting.
            W = ops.divide(W, ops.expand_dims(self.awq_scales, -1))

            W = ops.reshape(W, self.original_kernel_shape)

        y = ops.einsum(self.equation, inputs, W)
        if self.bias is not None:
            y = ops.add(y, self.bias)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _int4_build(self, kernel_shape, config=None):
        """Build variables for int4 quantization.

        The kernel is  flattened to 2D [rows, columns]
        and packed along last axis to [rows, ceil(columns/2)].

        Args:
            kernel_shape: Original kernel shape (may be N-dimensional).
            config: Optional quantization config specifying block_size.
        """
        self._set_quantization_info()

        self.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(config, None)
        )
        self.quantization_axis = tuple(self._input_reduced_axes)
        self.original_kernel_shape = kernel_shape

        # Flatten kernel to 2D: rows = reduced dims, columns = non-reduced dims
        rows = 1
        columns = 1
        for i, dim in enumerate(kernel_shape):
            if i in self._kernel_reduced_axes:
                rows *= dim
            else:
                columns *= dim

        block_size = get_block_size_for_layer(self, config)
        use_grouped = block_size is not None and block_size != -1
        self._int4_block_size = block_size if use_grouped else None
        self._int4_unpacked_column_size = columns
        self._int4_rows = rows

        # Kernel packed along last axis (columns)
        # Stored shape: [rows, ceil(columns/2)]
        packed_cols = (columns + 1) // 2
        self._kernel = self.add_weight(
            name="kernel",
            shape=(rows, packed_cols),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )

        if use_grouped:
            # Sub-channel: [n_groups, columns]
            n_groups = math.ceil(rows / block_size)
            scale_shape = (n_groups, columns)
        else:
            scale_shape = (columns,)

        self.kernel_scale = self.add_weight(
            name="kernel_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )

        # Sub-channel quantization uses asymmetric quantization with zero point
        if use_grouped:
            self.kernel_zero = self.add_weight(
                name="kernel_zero",
                shape=scale_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            self.g_idx = self.add_weight(
                name="g_idx",
                shape=(rows,),
                initializer="zeros",
                dtype="float32",
                trainable=False,
            )
            self.g_idx.assign(
                ops.floor_divide(ops.arange(rows, dtype="float32"), block_size)
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
            """Performs int8 quantized einsum with a custom gradient.

            Computes the einsum operation with quantized inputs and a quantized
            kernel, then de-quantizes the result.

            Also computes the gradient with respect to the original,
            full-precision inputs by using a de-quantized kernel.

            Args:
                inputs: The full-precision input tensor.
                kernel: The int8 quantized kernel tensor.
                kernel_scale: The float32 scale factor for the kernel.

            Returns:
                A tuple `(output, grad_fn)`:
                    `output`: The de-quantized result of the einsum operation.
                    `grad_fn`: The custom gradient function for the backward
                        pass.

            Raises:
                ValueError: If the quantization mode is not supported.
            """

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                # De-scale kernel
                _kernel_scale = kernel_scale
                _kernel_scale = self._adjust_scale_for_dequant(_kernel_scale)
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=self.compute_dtype),
                    _kernel_scale,
                )
                # From https://stackoverflow.com/a/47609896
                inputs_grad = ops.einsum(
                    self._custom_gradient_equation, upstream, float_kernel
                )
                return (inputs_grad, None, None)

            if self.inputs_quantizer:
                inputs, inputs_scale = self.inputs_quantizer(
                    inputs, axis=self.quantization_axis
                )
                # Align `inputs_scale` axes with the output
                # for correct broadcasting
                inputs_scale = self._adjust_scale_for_quant(
                    inputs_scale, "input"
                )
                x = ops.einsum(self.equation, inputs, kernel)
                # De-scale outputs
                x = ops.cast(x, self.compute_dtype)
                x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
            else:
                # Weight-only quantization: dequantize kernel and use float
                # einsum. This is a workaround for PyTorch's einsum which
                # doesn't support mixed-precision inputs (float input,
                # int8 kernel).
                if backend.backend() == "torch":
                    kernel_scale = self._adjust_scale_for_dequant(kernel_scale)
                    float_kernel = ops.divide(
                        ops.cast(kernel, dtype=self.compute_dtype),
                        kernel_scale,
                    )
                    x = ops.einsum(self.equation, inputs, float_kernel)
                else:
                    x = ops.einsum(self.equation, inputs, kernel)
                    # De-scale outputs
                    x = ops.cast(x, self.compute_dtype)
                    x = ops.divide(x, kernel_scale)
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

    def _int4_call(self, inputs, training=None):
        """Forward pass for int4 quantized EinsumDense.

        Uses custom gradients to handle quantized weights since autodiff
        cannot differentiate through int4 operations.
        """
        block_size = getattr(self, "_int4_block_size", None)

        if block_size is None or block_size == -1:

            @ops.custom_gradient
            def einsum_per_channel_with_inputs_gradient(
                inputs, packed_kernel, kernel_scale
            ):
                """Per-channel int4 forward pass with custom gradient."""
                # Unpack: stored as [rows, ceil(columns/2)],
                # unpack along last axis
                unpacked_kernel = quantizers.unpack_int4(
                    packed_kernel,
                    self._int4_unpacked_column_size,
                    axis=-1,
                    dtype="int8",
                )

                def _dequantize_kernel(unpacked, scale):
                    # kernel is [rows, columns], scale is [columns]
                    float_kernel = ops.divide(
                        ops.cast(unpacked, dtype=self.compute_dtype),
                        scale,
                    )
                    return ops.reshape(float_kernel, self.original_kernel_shape)

                def grad_fn(*args, upstream=None):
                    if upstream is None:
                        (upstream,) = args
                    float_kernel = _dequantize_kernel(
                        unpacked_kernel, kernel_scale
                    )
                    inputs_grad = ops.einsum(
                        self._custom_gradient_equation, upstream, float_kernel
                    )
                    return (inputs_grad, None, None)

                if self.inputs_quantizer:
                    # Per-channel with input quantization
                    float_kernel = _dequantize_kernel(
                        unpacked_kernel, kernel_scale
                    )
                    inputs_q, inputs_scale = self.inputs_quantizer(
                        inputs, axis=self.quantization_axis
                    )
                    inputs_scale = self._adjust_scale_for_quant(
                        inputs_scale, "input"
                    )
                    # Cast inputs to float for einsum. This is a workaround
                    # for PyTorch's einsum which doesn't support
                    # mixed-precision inputs (int8 input, float kernel).
                    if backend.backend() == "torch":
                        x = ops.einsum(
                            self.equation,
                            ops.cast(inputs_q, self.compute_dtype),
                            float_kernel,
                        )
                        x = ops.divide(x, inputs_scale)
                    else:
                        x = ops.einsum(self.equation, inputs_q, float_kernel)
                        x = ops.cast(x, self.compute_dtype)
                        x = ops.divide(x, inputs_scale)
                else:
                    # Weight-only per-channel quantization
                    float_kernel = _dequantize_kernel(
                        unpacked_kernel, kernel_scale
                    )
                    x = ops.einsum(self.equation, inputs, float_kernel)
                return x, grad_fn

            x = einsum_per_channel_with_inputs_gradient(
                inputs,
                ops.convert_to_tensor(self._kernel),
                ops.convert_to_tensor(self.kernel_scale),
            )
        else:

            @ops.custom_gradient
            def einsum_sub_channel_with_inputs_gradient(
                inputs, packed_kernel, kernel_scale, kernel_zero, g_idx
            ):
                """Sub-channel int4 forward pass with custom gradient."""
                # Unpack: stored as [rows, ceil(columns/2)],
                # unpack along last axis
                unpacked_kernel = quantizers.unpack_int4(
                    packed_kernel,
                    self._int4_unpacked_column_size,
                    axis=-1,
                    dtype="int8",
                )

                def _dequantize_kernel(unpacked, scale, zero, g_idx_t):
                    # Dequantize with group_axis=0 since
                    # scale is [n_groups, columns]
                    float_kernel = dequantize_with_sz_map(
                        unpacked, scale, zero, g_idx_t, group_axis=0
                    )
                    float_kernel = ops.cast(float_kernel, self.compute_dtype)
                    return ops.reshape(float_kernel, self.original_kernel_shape)

                def grad_fn(*args, upstream=None):
                    if upstream is None:
                        (upstream,) = args
                    float_kernel = _dequantize_kernel(
                        unpacked_kernel, kernel_scale, kernel_zero, g_idx
                    )
                    inputs_grad = ops.einsum(
                        self._custom_gradient_equation, upstream, float_kernel
                    )
                    return (inputs_grad, None, None, None, None)

                float_kernel = _dequantize_kernel(
                    unpacked_kernel, kernel_scale, kernel_zero, g_idx
                )
                x = ops.einsum(self.equation, inputs, float_kernel)
                return x, grad_fn

            x = einsum_sub_channel_with_inputs_gradient(
                inputs,
                ops.convert_to_tensor(self._kernel),
                ops.convert_to_tensor(self.kernel_scale),
                ops.convert_to_tensor(self.kernel_zero),
                ops.convert_to_tensor(self.g_idx),
            )

        if self.lora_enabled:
            lora_x = ops.einsum(self.equation, inputs, self.lora_kernel_a)
            lora_x = ops.matmul(lora_x, self.lora_kernel_b)
            x = ops.add(x, (self.lora_alpha / self.lora_rank) * lora_x)

        # Bias & activation
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

    def quantize(self, mode=None, type_check=True, config=None):
        # Prevent quantization of the subclasses
        if type_check and (type(self) is not EinsumDense):
            raise self._not_implemented_error(self.quantize)

        self.quantization_config = config

        kernel_shape = self._kernel.shape
        if mode in ("int8", "int4", "gptq", "awq"):
            self._set_quantization_info()

        if mode == "int8":
            # Quantize `self._kernel` to int8 and compute corresponding scale
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                self.quantization_config,
                quantizers.AbsMaxQuantizer(axis=self._kernel_reduced_axes),
            )
            kernel_value, kernel_scale = weight_quantizer(
                self._kernel, to_numpy=True
            )
            kernel_scale = self._adjust_scale_for_quant(kernel_scale, "kernel")
            del self._kernel
        elif mode == "int4":
            from keras.src.quantizers.quantization_config import (
                Int4QuantizationConfig,
            )

            block_size = None
            if isinstance(self.quantization_config, Int4QuantizationConfig):
                block_size = self.quantization_config.block_size

            use_grouped = block_size is not None and block_size != -1

            # Flatten kernel to 2D: rows = reduced dims, columns = non-reduced
            rows = 1
            columns = 1
            for i, dim in enumerate(kernel_shape):
                if i in self._kernel_reduced_axes:
                    rows *= dim
                else:
                    columns *= dim

            flat_kernel = ops.reshape(self._kernel, (rows, columns))

            if not use_grouped:
                # Per-channel quantization
                kernel_value_int4, kernel_scale = quantizers.abs_max_quantize(
                    flat_kernel,
                    axis=0,
                    value_range=(-8, 7),
                    dtype="int8",
                    to_numpy=True,
                )
                kernel_scale = ops.squeeze(kernel_scale, axis=0)
            else:
                # Sub-channel quantization with asymmetric zero point
                # Returns kernel [rows, columns], scale [n_groups, columns]
                kernel_value_int4, kernel_scale, kernel_zero = (
                    quantizers.abs_max_quantize_grouped_with_zero_point(
                        flat_kernel, block_size=block_size, to_numpy=True
                    )
                )

            # Pack two int4 values per int8 byte along last axis
            # Stored as [rows, ceil(columns/2)]
            packed_kernel_value, _, _ = quantizers.pack_int4(
                kernel_value_int4, axis=-1
            )
            kernel_value = packed_kernel_value
            del self._kernel
        self.quantized_build(kernel_shape, mode, self.quantization_config)

        # Assign values to the newly created variables.
        if mode in ("int8", "int4"):
            self._kernel.assign(kernel_value)
            self.kernel_scale.assign(kernel_scale)
            # Assign zero point for sub-channel int4 quantization
            if mode == "int4" and use_grouped:
                self.kernel_zero.assign(kernel_zero)

        # Set new dtype policy
        if self.dtype_policy.quantization_mode is None:
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

    def _get_kernel_scale_shape(self, kernel_shape, block_size=None):
        """Get the shape of the kernel scale tensor.

        The kernel scale tensor is used to scale the kernel tensor.
        The shape of the kernel scale tensor is the same as the shape of the
        kernel tensor, but with the reduced axes set to 1 (for per-channel)
        or n_groups (for grouped quantization), and the transpose axes set
        to the original axes.

        Args:
            kernel_shape: The shape of the kernel tensor.
            block_size: If provided and positive, use grouped quantization
                along the reduced axes with the specified block size.

        Returns:
            The shape of the kernel scale tensor.
        """
        if block_size is not None and block_size > 0:
            # Grouped quantization: use simple 2D scale shape
            # (n_groups, non_reduced) - matches dequantize_grouped format
            total_reduced_dim = 1
            for ax in self._kernel_reduced_axes:
                total_reduced_dim *= kernel_shape[ax]
            n_groups = math.ceil(total_reduced_dim / block_size)

            total_non_reduced = 1
            for i, dim in enumerate(kernel_shape):
                if i not in self._kernel_reduced_axes:
                    total_non_reduced *= dim

            return (n_groups, total_non_reduced)
        else:
            # Per-channel quantization: use the original transformation logic
            kernel_scale_shape = np.array(kernel_shape)
            kernel_scale_shape[self._kernel_reduced_axes] = 1

            kernel_scale_shape = kernel_scale_shape[self._kernel_transpose_axes]
            kernel_scale_shape = kernel_scale_shape.tolist()
            for a in sorted(self._kernel_expand_axes):
                kernel_scale_shape.insert(a, 1)
            for a in sorted(self._kernel_squeeze_axes, reverse=True):
                kernel_scale_shape.pop(a)
            return kernel_scale_shape

    def _get_kernel_with_merged_lora(self):
        """Returns the kernel with LoRA matrices merged, for serialization.

        This method is called by `save_own_variables` to produce a single
        kernel tensor that includes the adaptations from LoRA. This is useful
        for deploying the model or for continuing training after permanently
        applying the LoRA update.

        If the layer is quantized (`int8` or `int4`), the process is:
        1. Dequantize the base kernel to float.
        2. Adjust the scale tensor layout for dequantization. This is the
            reverse order of operations used when building the layer.
        3. Compute the LoRA delta (`lora_kernel_a @ lora_kernel_b`) and add
            it to the dequantized kernel.
        4. Re-quantize the merged result back to the original quantized
            type (`int8` or packed `int4`), calculating a new scale factor.
        5. Adjust the scale tensor layout for quantization. This is the forward
            order of operations used when building the layer.

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
        # If not a quantized layer, return the full-precision kernel directly.
        if self.dtype_policy.quantization_mode in (None, "gptq", "awq"):
            return self.kernel, None, None

        kernel_zero = getattr(self, "kernel_zero", None)

        # If quantized but LoRA is not enabled, return the original quantized
        # kernel.
        if not self.lora_enabled:
            return self._kernel, self.kernel_scale, kernel_zero

        # Dequantize, Merge, and Re-quantize

        # 1. Dequantize the kernel
        if self.quantization_mode == "int4":
            # Unpack [rows, ceil(columns/2)] to [rows, columns]
            unpacked_kernel = quantizers.unpack_int4(
                self._kernel,
                self._int4_unpacked_column_size,
                axis=-1,
            )
            block_size = getattr(self, "_int4_block_size", None)
            if block_size is not None and block_size != -1:
                # Grouped dequantization with group_axis=0
                kernel_fp = dequantize_with_sz_map(
                    unpacked_kernel,
                    self.kernel_scale,
                    self.kernel_zero,
                    self.g_idx,
                    group_axis=0,
                )
            else:
                # Per-channel dequantization:
                # kernel [rows, columns], scale [columns]
                kernel_fp = ops.divide(
                    ops.cast(unpacked_kernel, self.compute_dtype),
                    self.kernel_scale,
                )
            kernel_fp = ops.reshape(kernel_fp, self.original_kernel_shape)
        elif self.quantization_mode == "int8":
            adjusted_scale = self._adjust_scale_for_dequant(self.kernel_scale)
            kernel_fp = ops.divide(self._kernel, adjusted_scale)
        else:
            raise ValueError(
                f"Unsupported quantization mode: {self.quantization_mode}"
            )

        # 2. Merge the LoRA update in the float domain
        lora_update = (self.lora_alpha / self.lora_rank) * ops.matmul(
            self.lora_kernel_a, self.lora_kernel_b
        )
        merged_kernel = ops.add(kernel_fp, lora_update)

        # 3. Re-quantize the merged float kernel back to the target format
        if self.quantization_mode == "int4":
            block_size = getattr(self, "_int4_block_size", None)
            rows = self._int4_rows
            columns = self._int4_unpacked_column_size

            # Flatten to 2D [rows, columns]
            flat_kernel = ops.reshape(merged_kernel, (rows, columns))

            if block_size is not None and block_size != -1:
                # Use abs_max_quantize_grouped_with_zero_point for proper
                # signed quantization (same as quantize() method)
                # Returns kernel [rows, columns], scale [n_groups, columns]
                kernel_quant, new_scale, new_zero = (
                    quantizers.abs_max_quantize_grouped_with_zero_point(
                        flat_kernel, block_size=block_size, to_numpy=True
                    )
                )
                kernel_zero = new_zero
            else:
                # Per-channel: quantize along rows axis
                kernel_quant, new_scale = quantizers.abs_max_quantize(
                    flat_kernel,
                    axis=0,
                    value_range=(-8, 7),
                    dtype="int8",
                    to_numpy=True,
                )
                new_scale = ops.squeeze(new_scale, axis=0)
                kernel_zero = None

            # Pack along last axis
            new_kernel, _, _ = quantizers.pack_int4(kernel_quant, axis=-1)
        elif self.quantization_mode == "int8":
            new_kernel, new_scale = quantizers.abs_max_quantize(
                merged_kernel,
                axis=self._kernel_reduced_axes,
                to_numpy=True,
            )
            new_scale = self._adjust_scale_for_quant(new_scale, "kernel")
            kernel_zero = None

        return new_kernel, new_scale, kernel_zero

    def _adjust_scale_for_dequant(self, scale):
        """Adjusts scale tensor layout for dequantization.

        Helper method to handle scale adjustments before dequantization.
        This is the reverse order of operations used when building the layer.

        Args:
            scale: The scale tensor to adjust.

        Returns:
            The adjusted scale tensor.
        """
        if self._kernel_squeeze_axes:
            scale = ops.expand_dims(scale, axis=self._kernel_squeeze_axes)
        if self._kernel_expand_axes:
            scale = ops.squeeze(scale, axis=self._kernel_expand_axes)
        if self._kernel_transpose_axes:
            # We need to reverse the transpose operation.
            reverse_transpose = sorted(
                range(len(self._kernel_transpose_axes)),
                key=self._kernel_transpose_axes.__getitem__,
            )
            scale = ops.transpose(scale, axes=reverse_transpose)
        return scale

    def _adjust_scale_for_quant(self, scale, tensor_type="kernel"):
        """Adjusts scale tensor layout after quantization.

        Helper method to handle scale adjustments after re-quantization.
        This is the forward order of operations used when building the layer.

        Args:
            scale: The scale tensor to adjust.
            tensor_type: The type of tensor to adjust the scale for.
                "kernel" or "input".
        Returns:
            The adjusted scale tensor.
        """
        if tensor_type == "kernel":
            transpose_axes = self._kernel_transpose_axes
            expand_axes = self._kernel_expand_axes
            squeeze_axes = self._kernel_squeeze_axes
        elif tensor_type == "input":
            transpose_axes = self._input_transpose_axes
            expand_axes = self._input_expand_axes
            squeeze_axes = self._input_squeeze_axes
        else:
            raise ValueError(f"Invalid tensor type: {tensor_type}")

        if transpose_axes:
            scale = ops.transpose(scale, transpose_axes)
        if expand_axes:
            scale = ops.expand_dims(scale, axis=expand_axes)
        if squeeze_axes:
            scale = ops.squeeze(scale, axis=squeeze_axes)
        return scale

    def _set_quantization_info(self):
        if hasattr(self, "_input_reduced_axes"):
            # Already set.
            return
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


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
    """Parses an einsum string to determine the shapes of the weights.

    This function is the main entry point for analyzing the einsum equation.
    It handles equations with and without ellipses (`...`) by converting them
    to a standard format and then delegating to `_analyze_split_string` for
    the core logic.

    Args:
        equation: The einsum equation string, e.g., "ab,bc->ac" or
            "...ab,bc->...ac".
        bias_axes: A string indicating which output axes to apply a bias to.
        input_shape: The shape of the input tensor.
        output_shape: The user-specified shape of the output tensor (may be
            partial).

    Returns:
        A tuple `(kernel_shape, bias_shape, full_output_shape)` where:
            `kernel_shape`: The calculated shape of the einsum kernel.
            `bias_shape`: The calculated shape of the bias, or `None`.
            `full_output_shape`: The fully-resolved shape of the output tensor.

    Raises:
        ValueError: If the einsum `equation` is not in a supported format.
    """

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
    """Computes kernel and bias shapes from a parsed einsum equation.

    This function takes the components of an einsum equation, validates them,
    and calculates the required shapes for the kernel and bias weights.

    Args:
        split_string: A regex match object containing the input, weight, and
            output specifications.
        bias_axes: A string indicating which output axes to apply a bias to.
        input_shape: The shape of the input tensor.
        output_shape: The user-specified partial shape of the output tensor.
        left_elided: A boolean indicating if the ellipsis "..." was on the
            left side of the equation.

    Returns:
        A tuple `(kernel_shape, bias_shape, full_output_shape)` where:
            `kernel_shape`: The calculated shape of the einsum kernel.
            `bias_shape`: The calculated shape of the bias, or `None`.
            `full_output_shape`: The fully-resolved shape of the output tensor.

    Raises:
        ValueError: If there are inconsistencies between the input and output
            shapes or if the equation specifications are invalid.
    """
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
    """Analyzes an einsum equation to derive information for quantization.

    This function canonicalizes the einsum equation (handling ellipses) and
    determines the necessary tensor manipulations (reduction, transposition,
    expansion, squeezing) required to correctly apply per-axis quantization
    to the inputs and kernel. It also derives the einsum equation needed for
    the custom gradient.

    Args:
        equation: The einsum equation string.
        input_shape: The shape of the input tensor.

    Returns:
        A tuple containing metadata for quantization operations:
        `input_reduced_axes`: Axes to reduce for input quantization.
        `kernel_reduced_axes`: Axes to reduce for kernel quantization.
        `input_transpose_axes`: Permutation for transposing the input scale.
        `kernel_transpose_axes`: Permutation for transposing the kernel scale.
        `input_expand_axes`: Axes to expand for the input scale.
        `kernel_expand_axes`: Axes to expand for the kernel scale.
        `input_squeeze_axes`: Axes to squeeze from the input scale.
        `kernel_squeeze_axes`: Axes to squeeze from the kernel scale.
        `custom_gradient_equation`: Einsum equation for the backward pass.
        `kernel_reverse_transpose_axes`: Permutation to reverse the kernel
            scale transpose.
    """

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
