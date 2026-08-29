from keras.src.dtype_policies.dtype_policy import GPTQDTypePolicy
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.modes.calibration import CalibrationMode


class GPTQMode(CalibrationMode):
    """GPTQ post-training quantization (calibration-based, 2/3/4/8-bit)."""

    name = "gptq"
    config_cls = GPTQConfig

    def policy_from_string(self, mode_str, source_name):
        return GPTQDTypePolicy(mode_str, source_name)

    def config_from_policy(self, policy):
        raise ValueError(
            "Implicitly enabling GPTQ quantization by setting "
            f"`dtype_policy` to '{policy.name}' is not supported. "
            "GPTQ requires a calibration dataset and a "
            "`GPTQConfig` object.\n\n"
            "Please use the `.quantize('gptq', config=...)` method "
            "on the layer or model instead."
        )

    def resolve_weight_bits(self, layer, config):
        """Determine the weight bits from the config or the dtype policy."""
        return self._resolve_from_config_or_policy(layer, config, "weight_bits")

    def _on_policy_map_mismatch(self, policy):
        # This should never happen based on how we set the quantization
        # mode, but we check just in case.
        raise ValueError(
            "Expected a `dtype_policy` of type `GPTQDTypePolicy`. "
            f"Got: {type(policy)}"
        )

    def _resolution_error(self, attr):
        return (
            f"For GPTQ quantization, the {attr} must be specified "
            "either through a `dtype_policy` of type "
            "`GPTQDTypePolicy` or the `config` argument."
        )

    def finalize_model_quantization(self, model, config, structure, filters):
        from keras.src.quantizers.gptq_core import gptq_quantize

        del model
        gptq_quantize(config, structure, filters=filters)
