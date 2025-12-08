from keras.src.api_export import keras_export
from keras.src.dtype_policies import QUANTIZATION_MODES
from keras.src.saving import serialization_lib


@keras_export("keras.quantizers.QuantizationConfig")
class QuantizationConfig:
    def __init__(self, weight_quantizer=None, activation_quantizer=None):
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    @property
    def mode(self):
        raise NotImplementedError(
            "Subclasses must implement this property. Do not instantiate "
            "QuantizationConfig directly."
        )

    def get_config(self):
        return {
            "weight_quantizer": serialization_lib.serialize_keras_object(
                self.weight_quantizer
            ),
            "activation_quantizer": serialization_lib.serialize_keras_object(
                self.activation_quantizer
            ),
        }

    @classmethod
    def from_config(cls, config):
        weight_quantizer = serialization_lib.deserialize_keras_object(
            config.get("weight_quantizer")
        )
        activation_quantizer = serialization_lib.deserialize_keras_object(
            config.get("activation_quantizer")
        )
        return cls(
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
        )

    @staticmethod
    def weight_quantizer_or_default(config, default):
        if config and config.weight_quantizer:
            return config.weight_quantizer
        return default

    @staticmethod
    def activation_quantizer_or_default(config, default):
        if config and config.activation_quantizer:
            return config.activation_quantizer
        elif config and config.activation_quantizer is None:
            return None
        return default


@keras_export("keras.quantizers.Int8QuantizationConfig")
class Int8QuantizationConfig(QuantizationConfig):
    def __init__(self, weight_quantizer=None, activation_quantizer="default"):
        from keras.src.quantizers.quantizers import AbsMaxQuantizer

        if activation_quantizer == "default":
            activation_quantizer = AbsMaxQuantizer(axis=-1)
        super().__init__(weight_quantizer, activation_quantizer)
        if self.weight_quantizer:
            if hasattr(self.weight_quantizer, "value_range"):
                if self.weight_quantizer.value_range != (-127, 127):
                    raise ValueError(
                        "Int8QuantizationConfig requires a weight_quantizer "
                        "with value_range=(-127, 127). Received: "
                        f"value_range={self.weight_quantizer.value_range}"
                    )

    @property
    def mode(self):
        return "int8"


@keras_export("keras.quantizers.Int4QuantizationConfig")
class Int4QuantizationConfig(QuantizationConfig):
    def __init__(self, weight_quantizer=None, activation_quantizer="default"):
        from keras.src.quantizers.quantizers import AbsMaxQuantizer

        if activation_quantizer == "default":
            activation_quantizer = AbsMaxQuantizer(axis=-1)
        super().__init__(weight_quantizer, activation_quantizer)
        if self.weight_quantizer:
            if hasattr(self.weight_quantizer, "value_range"):
                if self.weight_quantizer.value_range != (-8, 7):
                    raise ValueError(
                        "Int4QuantizationConfig requires a weight_quantizer "
                        "with value_range=(-8, 7). Received: "
                        f"value_range={self.weight_quantizer.value_range}"
                    )

    @property
    def mode(self):
        return "int4"


@keras_export("keras.quantizers.Float8QuantizationConfig")
class Float8QuantizationConfig(QuantizationConfig):
    def __init__(self, weight_quantizer=None, activation_quantizer=None):
        super().__init__(weight_quantizer, activation_quantizer)

    @property
    def mode(self):
        return "float8"


def validate_and_resolve_config(mode, config, name=None):
    # 1. Backwards Compatibility: Handle string shortcuts
    if isinstance(config, str):
        mode = config
        config = None

    # 2. Resolve "mode" into a Config object
    if config is None:
        if mode == "int8":
            config = Int8QuantizationConfig()
        elif mode == "int4":
            config = Int4QuantizationConfig()
        elif mode == "float8":
            config = Float8QuantizationConfig()
        elif mode == "gptq":
            raise ValueError(
                "For GPTQ, you must pass a GPTQConfig object explicitly."
            )
        else:
            if mode is not None:
                raise ValueError(
                    f"Invalid quantization mode. Received: mode={mode}"
                )
            raise ValueError(
                "You must provide either `mode` or `config` to `quantize`."
            )
    else:
        if not isinstance(config, QuantizationConfig):
            raise ValueError(
                "Argument `config` must be an instance of "
                "`QuantizationConfig`. "
                f"Received: config={config} (of type {type(config)})"
            )

    # 3. Validation: Prevent contradictions
    if mode is not None and config.mode != mode:
        raise ValueError(
            f"Contradictory arguments: mode='{mode}' but "
            f"config.mode='{config.mode}'"
        )

    # 4. Execution
    mode = config.mode  # Ensure mode is consistent
    if mode not in QUANTIZATION_MODES:
        raise ValueError(
            "Invalid quantization mode. "
            f"Expected one of {QUANTIZATION_MODES}. Received: mode={mode}"
        )

    if mode == "gptq":
        from keras.src.quantizers.gptq_config import GPTQConfig

        if not isinstance(config, GPTQConfig):
            raise ValueError(
                "Mode 'gptq' requires a valid `config` argument of type "
                f"`GPTQConfig`. Received: {type(config)}"
            )

    return config
