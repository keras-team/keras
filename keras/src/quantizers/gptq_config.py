from keras.src.api_export import keras_export


@keras_export("keras.quantizers.GPTQConfig")
class GPTQConfig:
    """Configuration class for the GPTQ (Gradient-based Post-Training
    Quantization) algorithm.

    GPTQ is a post-training quantization method that quantizes neural network
    weights to lower precision (e.g., 4-bit) while minimizing the impact on
    model accuracy. It works by analyzing the Hessian matrix of the loss
    function with respect to the weights and applying optimal quantization
    that preserves the most important weight values.

    **When to use GPTQ:**
    - You want to reduce model size and memory usage
    - You need faster inference on hardware that supports low-precision
      operations
    - You want to maintain model accuracy as much as possible
    - You have a pre-trained model that you want to quantize without
      retraining

    **How it works:**
    1. Uses calibration data to compute the Hessian matrix for each layer
    2. Applies iterative quantization with error correction
    3. Reorders weights based on activation importance (optional)
    4. Quantizes weights while minimizing quantization error

    **Example usage:**
    ```python
    from keras.quantizers import GPTQConfig
    from keras import Model

    # Create configuration for 4-bit quantization
    config = GPTQConfig(
        dataset=calibration_data,          # Your calibration dataset
        tokenizer=your_tokenizer,          # Tokenizer for text data
        weight_bits=4,                     # Quantize to 4 bits
        num_samples=128,                   # Number of calibration samples
        sequence_length=512,               # Sequence length for each sample
        hessian_damping=0.01,             # Hessian stabilization factor
        group_size=128,                    # Weight grouping for quantization
        symmetric=False,                   # Use asymmetric quantization
        activation_order=True              # Reorder weights by importance
    )

    # Apply quantization to your model
    model = Model(...)  # Your pre-trained model
    model.quantize("gptq", config=config)

    # The model now has quantized weights and can be used for inference
    ```

    **Benefits:**
    - **Memory reduction**: 4-bit quantization reduces memory by ~8x compared
      to float32
    - **Faster inference**: Lower precision operations are faster on supported
      hardware
    - **Accuracy preservation**: Minimizes accuracy loss through optimal
      quantization
    - **No retraining required**: Works with pre-trained models

    **Advanced usage examples:**

    **Per-channel quantization (recommended for most cases):**
    ```python
    config = GPTQConfig(
        dataset=calibration_data,
        tokenizer=tokenizer,
        weight_bits=4,
        group_size=-1,  # -1 enables per-channel quantization
        symmetric=False
    )
    ```

    **Grouped quantization (for specific hardware requirements):**
    ```python
    config = GPTQConfig(
        dataset=calibration_data,
        tokenizer=tokenizer,
        weight_bits=4,
        group_size=64,  # 64 weights share the same scale factor
        symmetric=True   # Use symmetric quantization
    )
    ```

    **High-accuracy quantization with activation ordering:**
    ```python
    config = GPTQConfig(
        dataset=calibration_data,
        tokenizer=tokenizer,
        weight_bits=4,
        activation_order=True,  # Reorder weights by importance
        hessian_damping=0.005,  # Lower damping for more precise
        # quantization
        num_samples=256          # More samples for better accuracy
    )
    ```

    **References:**
    - Original GPTQ paper: "GPTQ: Accurate Post-Training Quantization
      for Generative Pre-trained Transformers"
    - Implementation based on: https://github.com/IST-DASLab/gptq
    - Suitable for: Transformer models, large language models, and other
      deep neural networks

    **Note:** The quality of quantization depends heavily on the calibration
    dataset. Use representative data that covers the expected input
    distribution for best results.

    Args:
        dataset: The calibration dataset. It can be an iterable that yields
            strings or pre-tokenized numerical tensors (e.g., a list of
            strings, a generator, or a NumPy array). This data is used to
            analyze the model's activations.
        tokenizer: A `keras_nlp.Tokenizer` instance (or a similar callable)
            that is used to process the `dataset` if it contains strings.
        weight_bits: (int, optional) The number of bits to quantize weights to.
            Defaults to 4.
        num_samples: (int, optional) The number of calibration data samples to
            use from the dataset. Defaults to 128.
        sequence_length: (int, optional) The sequence length to use for each
            calibration sample. Defaults to 512.
        hessian_damping: (float, optional) The % of Hessian damping to use for
            stabilization during inverse calculation. Defaults to 0.01.
        group_size: (int, optional) The size of weight groups to quantize
            together. A `group_size` of -1 indicates per-channel quantization.
            Defaults to 128.
        symmetric: (bool, optional) If `True`, uses symmetric quantization.
            If `False`, uses asymmetric quantization. Defaults to `False`.
        activation_order: (bool, optional) If `True`, reorders weight columns
            based on activation magnitude, which can improve quantization
            accuracy. Defaults to `False`.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        *,
        weight_bits: int = 4,
        num_samples: int = 128,
        per_channel: bool = True,
        sequence_length: int = 512,
        hessian_damping: float = 0.01,
        group_size: int = 128,
        symmetric: bool = False,
        activation_order: bool = False,
    ):
        if weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                f"Unsupported weight_bits {weight_bits}. "
                "Supported values are 2, 3, 4, and 8."
            )
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")
        if hessian_damping < 0 or hessian_damping > 1:
            raise ValueError("hessian_damping must be between 0 and 1.")
        if group_size < -1 or group_size == 0:
            raise ValueError(
                "Invalid group_size. Supported values are -1 (whole-tensor) "
                "or a positive integer, "
                f"but got {group_size}."
            )
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.per_channel = per_channel
        self.sequence_length = sequence_length
        self.hessian_damping = hessian_damping
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.activation_order = activation_order

    def dtype_policy_string(self):
        """Returns the dtype policy string for this configuration.

        Returns:
            A string representing the dtype policy, e.g. "gptq_4bit".
        """
        return f"gptq/{self.weight_bits}/{self.group_size}"
