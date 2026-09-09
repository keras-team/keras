from keras.src.quantizers.mode_registry import QuantizationMode
from keras.src.quantizers.quantization_config import TernaryQuantizationConfig


class TernaryMode(QuantizationMode):
    """Ternary (BitNet b1.58) quantization: weights in `{-1, 0, +1}`.

    The quantization rule (threshold and scale) is owned by the layer, so
    the config carries no parameters.
    """

    name = "ternary"
    config_cls = TernaryQuantizationConfig
