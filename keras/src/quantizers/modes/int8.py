from keras.src.quantizers.mode_registry import QuantizationMode
from keras.src.quantizers.quantization_config import Int8QuantizationConfig


class Int8Mode(QuantizationMode):
    """W8A8 dynamic quantization (int8 weights times int8 activations)."""

    name = "int8"
    config_cls = Int8QuantizationConfig
