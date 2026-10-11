from keras.src.dtype_policies.dtype_policy import AWQDTypePolicy
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.modes.calibration import CalibrationStrategy


class AWQStrategy(CalibrationStrategy):
    """AWQ post-training quantization (activation-aware, 4-bit).

    AWQ uses 4-bit quantization with per-channel AWQ scales that protect
    salient weights based on activation magnitudes.
    """

    name = "awq"
    config_cls = AWQConfig

    def policy_from_string(self, mode_str, source_name):
        return AWQDTypePolicy(mode_str, source_name)

    def _resolution_error(self, attr):
        del attr
        return (
            "For AWQ quantization, group_size must be specified "
            "through AWQConfig or AWQDTypePolicy."
        )

    def _build_extra_variables(self, layer, rows):
        # Per-channel AWQ scales from activation magnitudes. The weights
        # were multiplied by them before quantization, so the quantized
        # weight view divides them back out (`QuantizedWeight.input_scales`).
        layer.awq_scales = layer.add_weight(
            name="awq_scales",
            shape=(rows,),
            initializer="ones",
            trainable=False,
        )

    def _input_scales(self, layer):
        return layer.awq_scales

    def finalize_model_quantization(self, model, config, structure, filters):
        from keras.src.quantizers.awq_core import awq_quantize

        del model
        awq_quantize(config, structure, filters=filters)
