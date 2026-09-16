from keras.src import ops
from keras.src.dtype_policies.dtype_policy import AWQDTypePolicy
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.modes.calibration import CalibrationStrategy
from keras.src.quantizers.quantizers import unpack_int4


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

    def _packed_columns(self, layer, columns, config):
        # For 4-bit weights, we pack two values per byte.
        return (columns + 1) // 2

    def _build_extra_variables(self, layer, rows):
        # Per-channel AWQ scales from activation magnitudes
        layer.awq_scales = layer.add_weight(
            name="awq_scales",
            shape=(rows,),
            initializer="ones",
            trainable=False,
        )

    def _unpack_kernel(self, layer, geometry):
        return unpack_int4(
            layer.quantized_kernel,
            orig_len=geometry.unpacked_columns(self.name),
            axis=0,
            dtype="uint8",
        )

    def _postprocess_kernel(self, layer, W):
        # Apply AWQ scales by dividing to restore original magnitude
        # (We multiplied by scales before quantization, so divide to undo)
        # awq_scales has shape [input_dim], W has shape [input_dim, units]
        # Expand dims for proper broadcasting.
        return ops.divide(W, ops.expand_dims(layer.awq_scales, -1))

    def finalize_model_quantization(self, model, config, structure, filters):
        from keras.src.quantizers.awq_core import awq_quantize

        del model
        awq_quantize(config, structure, filters=filters)
