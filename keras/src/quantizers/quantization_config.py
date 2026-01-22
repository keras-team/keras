from keras.src.api_export import keras_export
from keras.src.dtype_policies import QUANTIZATION_MODES
from keras.src.saving import serialization_lib


@keras_export("keras.quantizers.QuantizationConfig")
class QuantizationConfig:
    """Base class for quantization configs.

    Subclasses must implement the `mode` property and the `get_config` and
    `from_config` class methods.

    Args:
        weight_quantizer: Quantizer for weights.
        activation_quantizer: Quantizer for activations.
    """

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
        if config is not None and config.weight_quantizer is not None:
            return config.weight_quantizer
        return default

    @staticmethod
    def activation_quantizer_or_default(config, default):
        if config is not None:
            return config.activation_quantizer
        return default


@keras_export("keras.quantizers.Int8QuantizationConfig")
class Int8QuantizationConfig(QuantizationConfig):
    """Int8 quantization config.

    Args:
        weight_quantizer: Quantizer for weights.
        activation_quantizer: Quantizer for activations. If "default", uses
            AbsMaxQuantizer with axis=-1.
    """

    def __init__(self, weight_quantizer=None, activation_quantizer="default"):
        from keras.src.quantizers.quantizers import AbsMaxQuantizer

        if activation_quantizer == "default":
            activation_quantizer = AbsMaxQuantizer()
        super().__init__(weight_quantizer, activation_quantizer)
        if self.weight_quantizer is not None:
            if self.weight_quantizer.output_dtype != "int8":
                raise ValueError(
                    "Int8QuantizationConfig requires a weight_quantizer "
                    "with output_dtype='int8'. Received: "
                    f"output_dtype={self.weight_quantizer.output_dtype}"
                )

    @property
    def mode(self):
        return "int8"


@keras_export("keras.quantizers.Int4QuantizationConfig")
class Int4QuantizationConfig(QuantizationConfig):
    """Int4 quantization config.

    Args:
        weight_quantizer: Quantizer for weights.
        activation_quantizer: Quantizer for activations. If "default", uses
            AbsMaxQuantizer with axis=-1.
        block_size: Size of groups along the input dimension for sub-channel
            quantization. If a positive integer, uses sub-channel quantization
            with `ceil(input_dim / block_size)` groups. If `None` or `-1`,
            uses per-channel quantization (one scale per output channel).
            Default: `128` (sub-channel with 128-element groups).
    """

    def __init__(
        self,
        weight_quantizer=None,
        activation_quantizer="default",
        block_size=128,
    ):
        if activation_quantizer == "default":
            # Use weight-only quantization by default for int4
            activation_quantizer = None
        super().__init__(weight_quantizer, activation_quantizer)

        # Validate block_size
        if block_size is not None and block_size != -1 and block_size <= 0:
            raise ValueError(
                f"block_size must be None, -1, or a positive integer. "
                f"Received: block_size={block_size}"
            )
        self.block_size = block_size

        # Sub-channel quantization does not support custom quantizers
        is_sub_channel = block_size is not None and block_size > 0
        has_custom_quantizer = (
            self.weight_quantizer is not None
            or self.activation_quantizer is not None
        )
        if is_sub_channel and has_custom_quantizer:
            raise ValueError(
                "Int4 sub-channel quantization (block_size > 0) does not "
                "support custom quantizers. Either set block_size to None "
                "or -1 for per-channel quantization, or remove the custom "
                f"quantizer arguments. Received: block_size={block_size}"
            )

        if self.weight_quantizer is not None:
            if self.weight_quantizer.value_range != (-8, 7):
                raise ValueError(
                    "Int4QuantizationConfig requires a weight_quantizer "
                    "with value_range=(-8, 7). Received: "
                    f"value_range={self.weight_quantizer.value_range}"
                )

            if self.weight_quantizer.output_dtype != "int8":
                raise ValueError(
                    "Int4QuantizationConfig requires a weight_quantizer "
                    "with output_dtype='int8'. Received: "
                    f"output_dtype={self.weight_quantizer.output_dtype}"
                )

    @property
    def mode(self):
        return "int4"

    def get_config(self):
        config = super().get_config()
        config["block_size"] = self.block_size
        return config

    @classmethod
    def from_config(cls, config):
        weight_quantizer = serialization_lib.deserialize_keras_object(
            config.get("weight_quantizer")
        )
        activation_quantizer = serialization_lib.deserialize_keras_object(
            config.get("activation_quantizer")
        )
        # Default to None for backwards compatibility with models saved
        # before block_size was introduced (those used per-channel mode)
        block_size = config.get("block_size", None)
        return cls(
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            block_size=block_size,
        )


@keras_export("keras.quantizers.Float8QuantizationConfig")
class Float8QuantizationConfig(QuantizationConfig):
    """FP8 quantization config.

    FP8 mixed-precision training does not support user defined quantizers.
    This config is only used to indicate that FP8 mixed-precision training
    should be used.
    """

    def __init__(self):
        super().__init__(None, None)

    @property
    def mode(self):
        return "float8"

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


def validate_and_resolve_config(mode, config):
    """Validate and resolve quantization config.

    This function validates the quantization config and resolves the mode.
    If mode is not provided, it is inferred from the config.
    If config is not provided, a default config is inferred from the mode.

    Args:
        mode: Quantization mode.
        config: Quantization config.
    """
    # 1. Backwards Compatibility: Handle string shortcuts.
    if isinstance(config, str):
        mode = config
        config = None

    _validate_mode(mode)

    # 2. Resolve "mode" into a Config object.
    if config is None:
        if mode == "int8":
            config = Int8QuantizationConfig()
        elif mode == "int4":
            config = Int4QuantizationConfig()
        elif mode == "float8":
            config = Float8QuantizationConfig()
        elif mode == "gptq":
            raise ValueError(
                "For GPTQ, you must pass a `GPTQConfig` object in the "
                "`config` argument."
            )
        elif mode == "awq":
            raise ValueError(
                "For AWQ, you must pass an `AWQConfig` object in the "
                "`config` argument."
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

    # 3. Validation: Prevent contradictions.
    if mode is not None and config.mode != mode:
        raise ValueError(
            f"Contradictory arguments: mode='{mode}' but "
            f"config.mode='{config.mode}'"
        )

    # Ensure mode is consistent.
    mode = config.mode

    # Ensure the mode derived from the config is valid.
    _validate_mode(mode)

    if mode == "gptq":
        from keras.src.quantizers.gptq_config import GPTQConfig

        if not isinstance(config, GPTQConfig):
            raise ValueError(
                "Mode 'gptq' requires a valid `config` argument of type "
                f"`GPTQConfig`. Received: {type(config)}"
            )

    if mode == "awq":
        from keras.src.quantizers.awq_config import AWQConfig

        if not isinstance(config, AWQConfig):
            raise ValueError(
                "Mode 'awq' requires a valid `config` argument of type "
                f"`AWQConfig`. Received: {type(config)}"
            )

    return config


def _validate_mode(mode):
    """Validates quantization mode."""
    if mode is not None and mode not in QUANTIZATION_MODES:
        raise ValueError(
            "Invalid quantization mode. "
            f"Expected one of {QUANTIZATION_MODES}. Received: mode={mode}"
        )


def get_block_size_for_layer(layer, config):
    """Determine the block size for int4 quantization.

    The block size can be specified either through the `config` argument
    or through the `dtype_policy` if it is of type `Int4DTypePolicy`.

    The config argument is usually available when quantizing the layer
    via the `quantize` method. If the layer was deserialized from a
    saved model, the block size should be specified in the `dtype_policy`.

    Args:
        layer: The layer being quantized.
        config: An optional configuration object that may contain the
            `block_size` attribute.
    Returns:
        int or None. The determined block size for int4 quantization.
        Returns `None` or `-1` for per-channel quantization.
    """
    from keras.src.dtype_policies.dtype_policy import Int4DTypePolicy
    from keras.src.dtype_policies.dtype_policy_map import DTypePolicyMap

    if config and isinstance(config, Int4QuantizationConfig):
        return config.block_size
    elif isinstance(layer.dtype_policy, Int4DTypePolicy):
        block_size = layer.dtype_policy.block_size
        # Convert -1 to None for consistency
        return None if block_size == -1 else block_size
    elif isinstance(layer.dtype_policy, DTypePolicyMap):
        policy = layer.dtype_policy[layer.path]
        if isinstance(policy, Int4DTypePolicy):
            block_size = policy.block_size
            return None if block_size == -1 else block_size
        # Fall back to None for legacy QuantizedDTypePolicy
        return None
    else:
        # For backwards compatibility with models that don't have
        # Int4DTypePolicy (legacy per-channel mode)
        return None
