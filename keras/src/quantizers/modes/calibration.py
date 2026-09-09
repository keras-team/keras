"""Shared chassis for the calibration-based quantization modes.

GPTQ and AWQ speak the same three-part policy grammar and resolve their
hyperparameters the same way; they differ only in a handful of message
fragments and in which bit-widths they accept. Those differences are the
hooks below.
"""

from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap
from keras.src.quantizers.mode_registry import QuantizationMode


class CalibrationMode(QuantizationMode):
    """A post-training mode whose values arrive from a calibration pass."""

    requires_config = True
    requires_layer_structure = True
    # Packed sub-byte storage (4-bit packs two values per byte).
    summary_byte_multiplier = 2

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
