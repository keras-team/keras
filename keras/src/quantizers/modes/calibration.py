"""Shared chassis for the calibration-based quantization modes.

GPTQ and AWQ allocate the same family of variables, run the same
dequantize-and-contract forward pass, and speak the same three-part policy
grammar; they differ only in how the quantized kernel is packed, in one
extra AWQ variable and its inverse scaling, and in a handful of message
fragments. Those differences are the hooks below.
"""

import math

from keras.src import ops
from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap
from keras.src.quantizers.modes.common import apply_bias_activation
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.strategy_registry import QuantizationStrategy


class CalibrationStrategy(QuantizationStrategy):
    """A post-training strategy whose values arrive from a calibration pass."""

    requires_config = True
    requires_layer_structure = True
    # Packed sub-byte storage (4-bit packs two values per byte).
    summary_byte_multiplier = 2

    def quantize(self, layer, config):
        # The quantized values arrive later, so this only allocates the
        # mode's variables from the layer's current weight shape.
        geometry = self.require_geometry(layer)
        layer.quantized_build(geometry.weight_shape, self.name, config)

    # --- Config and policy-string surface ---------------------------------

    def _missing_config_error(self):
        return (
            f"For {self.name.upper()}, the `config` argument must be of "
            f"type `{self.config_cls.__name__}`."
        )

    def policy_suffix(self, layer, config):
        del layer
        return config.dtype_policy_string()

    def resolve_group_size(self, layer, config):
        """Determine the group size from the config or the dtype policy."""
        return self._resolve_from_config_or_policy(layer, config, "group_size")

    def _resolve_from_config_or_policy(self, layer, config, attr):
        """Resolves a hyperparameter with config-over-policy precedence.

        The config argument is usually available when quantizing the layer
        via the `quantize` method. If the layer was deserialized from a
        saved model, the value comes from the mode's dtype policy.
        """
        if isinstance(config, self.config_cls):
            return getattr(config, attr)
        policy = layer.dtype_policy
        if isinstance(policy, DTypePolicyMap):
            policy = policy[layer.path]
            if policy.quantization_mode != self.name:
                self._on_policy_map_mismatch(policy)
        if policy.quantization_mode == self.name:
            return getattr(policy, attr)
        raise ValueError(self._resolution_error(attr))

    def _on_policy_map_mismatch(self, policy):
        """Hook for modes that reject a mismatched `DTypePolicyMap` entry.

        Returning lets resolution fall through to `_resolution_error`.
        """

    def _resolution_error(self, attr):
        """The error raised when a hyperparameter cannot be resolved."""
        raise NotImplementedError

    # --- Variables --------------------------------------------------------

    def build(self, layer, input_shape, config):
        """Allocates the quantized kernel and quantization parameters.

        The variables hold uninitialized values until the calibration pass
        (run by `Model.quantize`) writes the quantized weights back.
        """
        geometry = self.require_geometry(layer)

        # Ensures the forward pass uses the original high-precision kernel
        # until calibration has been performed.
        setattr(layer, f"is_{self.name}_calibrated", False)
        geometry.record_kernel_shape(input_shape)

        if len(input_shape) not in (2, 3):
            raise ValueError(
                f"{self.name.upper()} quantization only supports 2D or 3D "
                "kernels."
            )
        rows, columns = geometry.calibration_rows_columns(input_shape)

        kernel_columns = self._packed_columns(layer, columns, config)
        group_size = self.resolve_group_size(layer, config)
        n_groups = 1 if group_size == -1 else math.ceil(rows / group_size)

        geometry.store_unpacked_columns(self.name, columns)
        geometry.prepare()

        layer.quantized_kernel = layer.add_weight(
            name="kernel",
            shape=(kernel_columns, rows),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )
        layer.kernel_scale = layer.add_weight(
            name="kernel_scale",
            shape=(columns, n_groups),
            initializer="ones",
            trainable=False,
        )
        layer.kernel_zero = layer.add_weight(
            name="zero_point",
            shape=(columns, n_groups),
            initializer="zeros",
            dtype="uint8",
            trainable=False,
        )
        self._build_extra_variables(layer, rows)
        # `g_idx` is stored as `float32` because TF has no GPU kernel for
        # int32 resource variables (would pin the variable to CPU and break
        # jit_compile on GPU); consumers cast to int32 on-device.
        layer.g_idx = layer.add_weight(
            name="g_idx",
            shape=(rows,),
            initializer="zeros",
            dtype="float32",
            trainable=False,
        )

    def _packed_columns(self, layer, columns, config):
        """Column count of the packed kernel for this mode's bit-width."""
        raise NotImplementedError

    def _build_extra_variables(self, layer, rows):
        """Creates any mode-specific variables, after the zero point."""

    # --- Forward pass -----------------------------------------------------

    def call(self, layer, inputs, training=False):
        geometry = self.require_geometry(layer)
        if not getattr(layer, f"is_{self.name}_calibrated"):
            W = layer._kernel
        else:
            W = self._unpack_kernel(layer, geometry)
            W = dequantize_with_sz_map(
                W,
                layer.kernel_scale,
                layer.kernel_zero,
                layer.g_idx,
            )
            W = ops.transpose(W)
            W = self._postprocess_kernel(layer, W)
            W = geometry.reshape_kernel(W)

        y = geometry.contract(inputs, W)
        return apply_bias_activation(layer, y)

    def _unpack_kernel(self, layer, geometry):
        """Unpacks the stored kernel to one code per byte."""
        raise NotImplementedError

    def _postprocess_kernel(self, layer, W):
        """Hook applied to the dequantized, transposed kernel."""
        return W
