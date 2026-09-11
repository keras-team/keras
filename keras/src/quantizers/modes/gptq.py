from keras.src.dtype_policies.dtype_policy import GPTQDTypePolicy
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.modes.calibration import CalibrationStrategy
from keras.src.quantizers.quantizers import unpack_int2
from keras.src.quantizers.quantizers import unpack_int4


class GPTQStrategy(CalibrationStrategy):
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

    def _packed_columns(self, layer, columns, config):
        weight_bits = self.resolve_weight_bits(layer, config)
        # Cache the resolved bit-width so the forward/serialization paths
        # don't have to re-resolve it from the dtype policy.
        layer._gptq_weight_bits = weight_bits
        # 4-bit weights pack two values per byte; 2-bit weights pack four.
        # Other bit-widths (e.g. 3, 8) are stored one value per byte.
        if weight_bits == 4:
            return (columns + 1) // 2
        elif weight_bits == 2:
            return (columns + 3) // 4
        return columns

    def _unpack_kernel(self, layer, geometry):
        weight_bits = layer._gptq_weight_bits
        orig_len = geometry.unpacked_columns(self.name)
        if weight_bits == 4:
            return unpack_int4(
                layer.quantized_kernel,
                orig_len=orig_len,
                axis=0,
                dtype="uint8",
            )
        elif weight_bits == 2:
            return unpack_int2(
                layer.quantized_kernel,
                orig_len=orig_len,
                axis=0,
                dtype="uint8",
            )
        return layer.quantized_kernel

    def finalize_model_quantization(self, model, config, structure, filters):
        from keras.src.quantizers.gptq_core import gptq_quantize

        del model
        gptq_quantize(config, structure, filters=filters)
