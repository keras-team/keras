from keras.src.dtype_policies.dtype_policy import QuantizedFloat8DTypePolicy
from keras.src.quantizers.mode_registry import QuantizationMode
from keras.src.quantizers.quantization_config import Float8QuantizationConfig


class Float8Mode(QuantizationMode):
    """Float8 QDQ mixed-precision training.

    Quantizing only allocates the scale/amax-history variables; the float
    kernel is kept and the fp8 casts happen dynamically during training.
    """

    name = "float8"
    config_cls = Float8QuantizationConfig

    def policy_from_string(self, mode_str, source_name):
        return QuantizedFloat8DTypePolicy(mode_str, source_name)
