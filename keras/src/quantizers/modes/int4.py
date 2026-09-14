import math

from keras.src import ops
from keras.src.dtype_policies.dtype_policy import Int4DTypePolicy
from keras.src.dtype_policies.dtype_policy import QuantizedDTypePolicy
from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap
from keras.src.quantizers.modes.common import GeometryDispatchStrategy
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.quantizers import (
    abs_max_quantize_grouped_with_zero_point,
)
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.quantizers import pack_int4
from keras.src.quantizers.quantizers import unpack_int4


def _is_per_channel(block_size):
    """Whether `block_size` selects per-channel (ungrouped) quantization.

    `block_size` is validated to be `None`, `-1`, or a positive integer by
    both `Int4QuantizationConfig` and the policy-string codec, so `None`
    and `-1` are the two spellings of per-channel.
    """
    return block_size is None or block_size == -1


def _is_grouped(block_size):
    """Whether `block_size` selects sub-channel (grouped) quantization."""
    return not _is_per_channel(block_size)


class Int4Strategy(GeometryDispatchStrategy):
    """W4A16 weight-only quantization (packed int4 weights)."""

    name = "int4"
    config_cls = Int4QuantizationConfig
    # Packed sub-byte storage: two int4 values per byte.
    summary_byte_multiplier = 2

    def resolve_block_size(self, layer, config):
        """Determine the block size for int4 quantization.

        The block size can be specified either through the `config` argument
        or through the `dtype_policy` if it is of type `Int4DTypePolicy`.

        The config argument is usually available when quantizing the layer
        via the `quantize` method. If the layer was deserialized from a
        saved model, the block size should be specified in the
        `dtype_policy`.

        Args:
            layer: The layer being quantized.
            config: An optional configuration object that may contain the
                `block_size` attribute.
        Returns:
            int or None. The determined block size for int4 quantization.
            Returns `None` or `-1` for per-channel quantization.
        """
        if isinstance(config, Int4QuantizationConfig):
            return config.block_size
        elif isinstance(layer.dtype_policy, Int4DTypePolicy):
            block_size = layer.dtype_policy.block_size
            # Convert -1 to None for consistency
            return None if block_size == -1 else block_size
        elif isinstance(layer.dtype_policy, DTypePolicyMap):
            policy = layer.dtype_policy[layer.path]
            if isinstance(policy, Int4DTypePolicy):
                block_size = policy.block_size
                return None if block_size == -1 else block_size
            # Fall back to None for legacy QuantizedDTypePolicy
            return None
        else:
            # For backwards compatibility with models that don't have
            # Int4DTypePolicy (legacy per-channel mode)
            return None

    def policy_from_string(self, mode_str, source_name):
        # Legacy bare "int4" policies carry no block size and stay generic
        # (they resolve to per-channel quantization on reload).
        if "/" in mode_str:
            return Int4DTypePolicy(mode_str, source_name)
        else:
            return QuantizedDTypePolicy(mode_str, source_name)

    def config_from_policy(self, policy):
        if isinstance(policy, Int4DTypePolicy):
            return Int4QuantizationConfig(block_size=policy.block_size)
        return Int4QuantizationConfig()

    def policy_suffix(self, layer, config):
        # Include block_size in policy name for sub-channel quantization.
        block_size = self.resolve_block_size(layer, config)
        # Use -1 for per-channel, otherwise use block_size
        block_size_value = -1 if block_size is None else block_size
        return f"int4/{block_size_value}"

    # --- Projection (Dense, EinsumDense) ----------------------------------
    #
    # One implementation serves every kernel contracted against its inputs.
    # The kernel is viewed as 2D `[rows, columns]` (rows: the contracted
    # axes, columns: the rest); codes are packed two per byte along the
    # columns, and the scale runs per column (per-channel) or per group of
    # rows (grouped, with a zero point and a group index). Per-channel and
    # grouped differ only in the variables they build and quantize; the
    # forward pass is one path: dequantize, then contract in float.

    def _build_projection(self, layer, geometry, kernel_shape, config):
        geometry.prepare()
        layer.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(config, None)
        )
        rows, columns = geometry.rows_columns(kernel_shape)
        block_size = self.resolve_block_size(layer, config)
        geometry.record_kernel_shape(kernel_shape)

        # Codes packed along the columns: stored as `[rows, ceil(columns/2)]`.
        layer._kernel = layer.add_weight(
            name="kernel",
            shape=(rows, (columns + 1) // 2),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        if _is_per_channel(block_size):
            scale_shape = (columns,)
        else:
            scale_shape = (math.ceil(rows / block_size), columns)
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=scale_shape,
            initializer="ones",
            trainable=False,
        )
        if _is_grouped(block_size):
            # Grouped quantization is asymmetric: a zero point per group and
            # the row-to-group index.
            def idx_initializer(shape, dtype):
                return ops.floor_divide(
                    ops.arange(rows, dtype=dtype), block_size
                )

            layer.kernel_zero = layer.add_weight(
                name="zero_point",
                shape=scale_shape,
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            # `g_idx` is stored as `float32` because TF has no GPU kernel for
            # int32 resource variables (would pin the variable to CPU and
            # break jit_compile on GPU); consumers cast to int32 on-device.
            layer.g_idx = layer.add_weight(
                name="g_idx",
                shape=(rows,),
                initializer=idx_initializer,
                dtype="float32",
                trainable=False,
            )

        # Recorded for unpacking and reshaping at runtime.
        layer._int4_block_size = block_size
        layer._orig_input_dim = rows
        layer._orig_output_dim = columns

    def _call_projection(self, layer, inputs, training=None):
        geometry = layer._quantization_geometry()
        grouped = _is_grouped(layer._int4_block_size)

        @ops.custom_gradient
        def contract_with_inputs_gradient(
            inputs, kernel, kernel_scale, *group_params
        ):
            """Dequantizes the int4 kernel and contracts in float.

            `group_params` is `(kernel_zero, g_idx)` for a grouped scheme
            and empty for per-channel. Autodiff cannot differentiate through
            the packed kernel, so the gradient with respect to the inputs is
            taken through the dequantized kernel.
            """
            unpacked = unpack_int4(kernel, layer._orig_output_dim, axis=-1)

            def dequantize():
                if group_params:
                    kernel_zero, g_idx = group_params
                    # Scale and zero point are `[n_groups, columns]`; the
                    # group index expands them over the rows.
                    float_kernel = dequantize_with_sz_map(
                        unpacked, kernel_scale, kernel_zero, g_idx, group_axis=0
                    )
                    float_kernel = ops.cast(float_kernel, layer.compute_dtype)
                else:
                    float_kernel = ops.divide(
                        ops.cast(unpacked, dtype=layer.compute_dtype),
                        kernel_scale,
                    )
                return geometry.reshape_kernel(float_kernel)

            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                inputs_grad = geometry.contract_grad(upstream, dequantize())
                return (inputs_grad, None, None) + (None,) * len(group_params)

            float_kernel = dequantize()
            if layer.inputs_quantizer:
                inputs_q, inputs_scale = layer.inputs_quantizer(
                    inputs, axis=geometry.inputs_quantization_axis
                )
                x = geometry.contract(inputs_q, float_kernel)
                x = ops.cast(x, layer.compute_dtype)
                x = ops.divide(x, geometry.align_inputs_scale(inputs_scale))
            else:
                x = geometry.contract(inputs, float_kernel)
            return x, grad_fn

        params = [
            inputs,
            ops.convert_to_tensor(layer._kernel),
            ops.convert_to_tensor(layer.kernel_scale),
        ]
        if grouped:
            params += [
                ops.convert_to_tensor(layer.kernel_zero),
                ops.convert_to_tensor(layer.g_idx),
            ]
        x = contract_with_inputs_gradient(*params)
        x = geometry.add_lora_delta(inputs, x)
        return apply_bias_activation(layer, x)

    def _quantize_projection(self, layer, geometry, config):
        kernel_shape = layer._kernel.shape
        geometry.prepare()
        # `Int4Strategy.resolve_block_size` is the single source of truth for
        # the group size, shared with the build path and the dtype-policy
        # naming, so the quantized values, the built variables, and the saved
        # policy string can never disagree. A bare `quantize("int4")` reaches
        # here with the canonical `Int4QuantizationConfig()` (grouped,
        # block_size=128); a `block_size` of `None` or `-1` selects the
        # per-channel escape hatch.
        block_size = self.resolve_block_size(layer, config)
        rows, columns = geometry.rows_columns(kernel_shape)
        flat_kernel = ops.reshape(layer._kernel, (rows, columns))

        if _is_per_channel(block_size):
            # Symmetric codes with one scale per column.
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                config,
                AbsMaxQuantizer(
                    axis=0, value_range=(-8, 7), output_dtype="int8"
                ),
            )
            kernel_value_int4, kernel_scale = weight_quantizer(
                flat_kernel, to_numpy=True
            )
            kernel_scale = ops.squeeze(kernel_scale, axis=0)
            kernel_zero = None
        else:
            # Asymmetric codes per group of rows: scale and zero point are
            # `[n_groups, columns]`.
            kernel_value_int4, kernel_scale, kernel_zero = (
                abs_max_quantize_grouped_with_zero_point(
                    flat_kernel, block_size=block_size, to_numpy=True
                )
            )

        # Pack two int4 values per int8 byte along the columns.
        packed_kernel_value, _, _ = pack_int4(kernel_value_int4, axis=-1)
        del layer._kernel
        layer.quantized_build(kernel_shape, "int4", config)
        layer._kernel.assign(packed_kernel_value)
        layer.kernel_scale.assign(kernel_scale)
        if kernel_zero is not None:
            layer.kernel_zero.assign(kernel_zero)
