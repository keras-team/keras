from keras.src.api_export import keras_export
from keras.src.quantizers.quantization_config import QuantizationConfig


@keras_export("keras.quantizers.AWQConfig")
class AWQConfig(QuantizationConfig):
    """Configuration class for AWQ (Activation-aware Weight Quantization).

    AWQ is a post-training quantization method that identifies and protects
    salient weights based on activation magnitudes. It applies per-channel
    scaling before quantization to minimize accuracy loss.

    Methodology:
    1. Collects activation statistics from calibration data
    2. Identifies salient weight channels based on activation magnitudes
    3. Searches for optimal per-channel scaling factors via grid search
    4. Applies scaling before quantization to protect important weights

    References:
    - Original AWQ paper: "AWQ: Activation-aware Weight Quantization for
      LLM Compression and Acceleration" (https://arxiv.org/abs/2306.00978)
    - Reference implementation: https://github.com/mit-han-lab/llm-awq

    Args:
        dataset: The calibration dataset. It can be an iterable that yields
            strings or pre-tokenized numerical tensors (e.g., a list of
            strings, a generator, or a NumPy array). This data is used to
            analyze activation patterns.
        tokenizer: A tokenizer instance (or a similar callable) that is used
            to process the `dataset`.
        weight_bits: The number of bits for weight quantization. AWQ presently
            only supports 4-bit quantization. Defaults to 4.
        num_samples: The number of calibration data samples to use from the
            dataset. Defaults to 128.
        sequence_length: The sequence length to use for each calibration
            sample. Defaults to 512.
        group_size: The size of weight groups to quantize together. A
            `group_size` of -1 indicates per-channel quantization.
            Defaults to 128.
        num_grid_points: The number of grid search points for finding optimal
            per-channel scales. Higher values may find better scales but
            take longer. Defaults to 20.
        quantization_layer_structure: A dictionary defining the model's
            quantization structure. It should contain:
            - "pre_block_layers": list of layers to run before the first
              block (e.g., embedding layer).
            - "sequential_blocks": list of transformer blocks to quantize
              sequentially.
            If not provided, the model must implement
            `get_quantization_layer_structure`.

    Example:
    ```python
    from keras.quantizers import AWQConfig

    # Create configuration for 4-bit AWQ quantization
    config = AWQConfig(
        dataset=calibration_data,          # Your calibration dataset
        tokenizer=your_tokenizer,          # Tokenizer for text data
        num_samples=128,                   # Number of calibration samples
        sequence_length=512,               # Sequence length for each sample
        group_size=128,                    # Weight grouping for quantization
        num_grid_points=20,                # Grid search points for scale search
    )

    # Apply quantization to your model
    model.quantize("awq", config=config)
    ```

    """

    def __init__(
        self,
        dataset,
        tokenizer,
        *,
        weight_bits: int = 4,
        num_samples: int = 128,
        sequence_length: int = 512,
        group_size: int = 128,
        num_grid_points: int = 20,
        quantization_layer_structure: dict = None,
    ):
        super().__init__()
        # AWQ only supports 4-bit quantization
        if weight_bits != 4:
            raise ValueError(
                f"AWQ only supports 4-bit quantization. "
                f"Received weight_bits={weight_bits}."
            )
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        if group_size < -1 or group_size == 0:
            raise ValueError(
                "Invalid group_size. Supported values are -1 (per-channel) "
                f"or a positive integer, but got {group_size}."
            )
        if num_grid_points <= 0:
            raise ValueError("num_grid_points must be a positive integer.")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.weight_bits = weight_bits
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.group_size = group_size
        self.num_grid_points = num_grid_points
        self.quantization_layer_structure = quantization_layer_structure

    @property
    def mode(self):
        return "awq"

    def dtype_policy_string(self):
        """Returns the dtype policy string for this configuration.

        Returns:
            A string representing the dtype policy, e.g. "awq/4/128".
        """
        return f"awq/{self.weight_bits}/{self.group_size}"

    def get_config(self):
        return {
            # Dataset and Tokenizer are only required for one-time
            # calibration and are not saved in the config.
            "dataset": None,
            "tokenizer": None,
            "weight_bits": self.weight_bits,
            "num_samples": self.num_samples,
            "sequence_length": self.sequence_length,
            "group_size": self.group_size,
            "num_grid_points": self.num_grid_points,
            "quantization_layer_structure": self.quantization_layer_structure,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
