"""AWQ (Activation-aware Weight Quantization) algorithm implementation.

AWQ protects salient weights by finding optimal per-channel scales based on
activation magnitudes, then applies those scales before quantization.

Reference: https://arxiv.org/abs/2306.00978
"""

import types

from keras.src import ops
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_sz_map
from keras.src.quantizers.quantizers import quantize_with_zero_point


def awq_search_optimal_scales(
    weights,
    activation_magnitudes,
    *,
    num_grid_points=20,
    group_size=-1,
):
    """Search for optimal AWQ scales using grid search.

    The AWQ algorithm finds scaling factors that protect salient weights.
    For each channel, we search for an optimal ratio in [0, 1] that minimizes
    the activation-weighted quantization error.

    The key insight: we MULTIPLY weights by scales before quantization to
    expand salient weights. This ensures quantization noise is small relative
    to the expanded weight magnitude. During inference, we divide by scales
    to restore the original magnitude.

    Scale formula: scales = x_max.pow(ratio).clamp(min=1e-4)
    Loss function: Activation-weighted MSE (approximates output error)

    Args:
        weights: Weight tensor [out_features, in_features] (transposed kernel).
        activation_magnitudes: Per-channel activation magnitudes [in_features].
        num_grid_points: Number of grid search points. Defaults to 20.
        group_size: Group size for quantization (-1 for per-channel).

    Returns:
        best_scales: Optimal per-channel scales [in_features].
    """
    in_features = ops.shape(weights)[1]

    # Compute per-channel activation magnitudes (x_max)
    # activations should already be per-channel max magnitudes
    x_max = ops.cast(activation_magnitudes, "float32")
    # Avoid zero or very small values
    x_max = ops.where(ops.less(x_max, 1e-8), ops.ones_like(x_max), x_max)

    best_loss = None
    best_scales = ops.ones((in_features,), dtype="float32")

    # Grid search over ratio values from 0 to 1
    for i in range(num_grid_points + 1):
        ratio = i / num_grid_points

        # Compute scales: x_max^ratio (clipped to avoid numerical issues)
        if ratio == 0:
            scales = ops.ones_like(x_max)
        else:
            scales = ops.power(x_max, ratio)
        scales = ops.maximum(scales, 1e-4)

        # Normalize scales to avoid extreme values
        scale_mean = ops.sqrt(ops.multiply(ops.max(scales), ops.min(scales)))
        scale_mean = ops.maximum(scale_mean, 1e-8)
        scales = ops.divide(scales, scale_mean)

        # Apply scales to weights by MULTIPLYING (expand salient weights)
        # weights_scaled: [out_features, in_features]
        weights_scaled = ops.multiply(weights, scales)

        if group_size == -1:
            # Per-channel quantization (no grouping)
            scale_q, zero_q, maxq = compute_quantization_parameters(
                weights_scaled,
                bits=4,
                symmetric=False,
                per_channel=True,
                group_size=-1,
                compute_dtype="float32",
            )

            # Quantize and dequantize
            quantized = quantize_with_zero_point(
                weights_scaled, scale_q, zero_q, maxq
            )
            dequantized = dequantize_with_zero_point(quantized, scale_q, zero_q)
        else:
            # Grouped quantization - use proper per-row grouping
            scale_q, zero_q, maxq = compute_quantization_parameters(
                weights_scaled,
                bits=4,
                symmetric=False,
                per_channel=True,
                group_size=group_size,
                compute_dtype="float32",
            )

            # Compute group indices: maps each input feature to its group
            g_idx = ops.cast(ops.arange(0, in_features) // group_size, "int32")

            # Quantize and dequantize using group index mapping
            quantized = quantize_with_sz_map(
                weights_scaled, scale_q, zero_q, g_idx, maxq
            )
            dequantized = dequantize_with_sz_map(
                quantized, scale_q, zero_q, g_idx
            )

        # Scale back down by DIVIDING to restore original magnitude
        reconstructed = ops.divide(dequantized, scales)

        # Compute activation-weighted MSE loss
        # This approximates the output error: ||W*X - W_hat*X||^2
        # by weighting each channel's error by x_max^2
        weight_error = ops.square(ops.subtract(weights, reconstructed))
        # Weight by activation magnitudes squared (broadcast over out_features)
        weighted_error = ops.multiply(weight_error, ops.square(x_max))
        loss = ops.mean(weighted_error)

        # Track best
        if best_loss is None:
            best_loss = loss
            best_scales = scales
        else:
            is_better = ops.less(loss, best_loss)
            if is_better:
                best_loss = loss
                best_scales = scales

    return best_scales


def awq_quantize_matrix(
    weights_transpose,
    activation_magnitudes,
    *,
    num_grid_points=20,
    group_size=-1,
):
    """Quantize a weight matrix using AWQ.

    This function performs the complete AWQ quantization process:
    1. Find optimal per-channel scales via grid search
    2. Apply scales to weights
    3. Compute quantization parameters
    4. Quantize weights

    Args:
        weights_transpose: Weight matrix [out_features, in_features].
        activation_magnitudes: Per-channel activation magnitudes [in_features].
        num_grid_points: Number of grid search points.
        group_size: Group size for quantization.

    Returns:
        quantized_weights: Quantized weights [out_features, in_features].
        scales: Quantization scales [out_features, num_groups].
        zeros: Zero points [out_features, num_groups].
        awq_scales: AWQ per-channel scales [in_features].
        g_idx: Group indices [in_features].
    """
    in_features = ops.shape(weights_transpose)[1]

    # Step 1: Find optimal AWQ scales via grid search
    awq_scales = awq_search_optimal_scales(
        weights_transpose,
        activation_magnitudes,
        num_grid_points=num_grid_points,
        group_size=group_size,
    )

    # Step 2: Apply AWQ scales by MULTIPLYING (expand salient weights)
    # weights_scaled: [out_features, in_features]
    weights_scaled = ops.multiply(weights_transpose, awq_scales)

    if group_size == -1:
        # Per-channel quantization (no grouping)
        scale_q, zero_q, maxq = compute_quantization_parameters(
            weights_scaled,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=-1,
            compute_dtype="float32",
        )

        # Quantize
        quantized = quantize_with_zero_point(
            weights_scaled, scale_q, zero_q, maxq
        )

        # Build group indices (all 0s for per-channel)
        g_idx = ops.zeros((in_features,), dtype="float32")
    else:
        # Grouped quantization - use proper per-row grouping
        scale_q, zero_q, maxq = compute_quantization_parameters(
            weights_scaled,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=group_size,
            compute_dtype="float32",
        )

        # Compute group indices: maps each input feature to its group
        g_idx = ops.cast(ops.arange(0, in_features) // group_size, "int32")

        # Quantize using group index mapping
        quantized = quantize_with_sz_map(
            weights_scaled, scale_q, zero_q, g_idx, maxq
        )

        # Convert g_idx to float for storage
        g_idx = ops.cast(g_idx, "float32")

    return quantized, scale_q, zero_q, awq_scales, g_idx


class AWQ:
    """AWQ quantizer for a single layer.

    This class accumulates activation statistics during calibration and
    performs AWQ quantization on layer weights.

    The AWQ algorithm works by:
    1. Collecting per-channel maximum activation magnitudes
    2. Using activation magnitudes to determine weight saliency
    3. Finding optimal per-channel scales via grid search
    4. Applying scales before quantization to protect salient weights

    Args:
        layer: The layer to quantize (Dense or EinsumDense).
        config: AWQConfig instance with quantization parameters.
    """

    def __init__(self, layer, config=None):
        from keras.src.quantizers.awq_config import AWQConfig

        self.original_layer = layer
        self.config = config or AWQConfig(dataset=None, tokenizer=None)
        self.num_samples = 0

        # Handle Dense and EinsumDense layers
        if isinstance(layer, Dense) or (
            isinstance(layer, EinsumDense) and layer.kernel.ndim == 2
        ):
            self.kernel_shape = layer.kernel.shape
            self.rows = self.kernel_shape[0]  # in_features
            self.columns = self.kernel_shape[1]  # out_features
            self.layer = layer
        elif isinstance(layer, EinsumDense) and layer.kernel.ndim == 3:
            # Handle 3D EinsumDense layers (typically from attention blocks)
            self.kernel_shape = layer.kernel.shape
            shape = list(self.kernel_shape)
            d_model_dim_index = shape.index(max(shape))

            if d_model_dim_index == 0:  # QKV projection case
                in_features, heads, head_dim = shape
                self.rows = in_features
                self.columns = heads * head_dim
            elif d_model_dim_index in [1, 2]:  # Attention Output case
                heads, head_dim, out_features = shape
                self.rows = heads * head_dim
                self.columns = out_features
            else:
                raise ValueError(
                    f"Cannot determine dimensions for EinsumDense kernel "
                    f"shape {shape}"
                )

            # Create a temporary object that holds a reshaped 2D version
            self.layer = types.SimpleNamespace(
                kernel=ops.reshape(layer.kernel, (self.rows, self.columns)),
            )
        else:
            raise TypeError(f"Unsupported layer type for AWQ: {type(layer)}")

        # Initialize activation magnitude accumulator (per-channel max)
        self.activation_magnitudes = ops.zeros((self.rows,), dtype="float32")

    def update_activation_magnitudes(self, input_batch):
        """Update per-channel activation magnitude statistics.

        This method tracks the maximum absolute activation value for each
        input channel across all calibration batches.

        Args:
            input_batch: Input activations tensor [batch, ..., in_features].
        """
        if input_batch is None:
            raise ValueError("Input tensor cannot be None.")
        if ops.size(input_batch) == 0:
            raise ValueError("Input tensor cannot be empty.")

        # Flatten to [batch_samples, in_features]
        if len(input_batch.shape) > 2:
            input_batch = ops.reshape(input_batch, (-1, input_batch.shape[-1]))

        x = ops.cast(input_batch, "float32")

        # Compute per-channel max absolute value for this batch
        batch_max = ops.max(ops.abs(x), axis=0)

        # Update running max
        self.activation_magnitudes = ops.maximum(
            self.activation_magnitudes, batch_max
        )
        self.num_samples = self.num_samples + int(ops.shape(x)[0])

    def quantize_layer(self):
        """Perform AWQ quantization on the layer.

        This method:
        1. Runs the AWQ grid search to find optimal scales
        2. Quantizes the layer weights
        3. Updates the layer's quantized variables
        """
        from keras.src import quantizers

        weights_matrix = ops.transpose(self.layer.kernel)

        # Perform AWQ quantization
        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights_matrix,
            self.activation_magnitudes,
            num_grid_points=self.config.num_grid_points,
            group_size=self.config.group_size,
        )

        # Cast to uint8 for storage
        # quantized is already [out_features, in_features]
        quantized = ops.cast(quantized, "uint8")

        # Pack to 4-bit along axis 0 (output features)
        quantized_packed, _, _ = quantizers.pack_int4(
            quantized, axis=0, dtype="uint8"
        )

        # Assign to layer variables
        del self.original_layer._kernel
        self.original_layer.quantized_kernel.assign(quantized_packed)
        self.original_layer.kernel_scale.assign(scale)
        self.original_layer.kernel_zero.assign(zero)
        self.original_layer.awq_scales.assign(awq_scales)
        self.original_layer.g_idx.assign(g_idx)
        self.original_layer.is_awq_calibrated = True

    def free(self):
        """Free memory used by the quantizer."""
        del self.activation_magnitudes
        del self.layer
