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
    # Per-input-row scales from the activation magnitudes.
    input_scales_name = "awq_scales"

    def policy_from_string(self, mode_str, source_name):
        return AWQDTypePolicy(mode_str, source_name)

    def _resolution_error(self, attr):
        del attr
        return (
            "For AWQ quantization, group_size must be specified "
            "through AWQConfig or AWQDTypePolicy."
        )

    def finalize_model_quantization(self, model, config, structure, filters):
        from keras.src.quantizers.awq_core import awq_quantize

        del model
        awq_quantize(config, structure, filters=filters)
