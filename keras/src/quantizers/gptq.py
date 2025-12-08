import types
from functools import partial

from keras.src import ops
from keras.src import quantizers
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.ops import linalg
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantizers import GPTQQuantizer
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_zero_point


def _stable_permutation(metric):
    """Return a stable permutation that sorts `metric` in descending order.
    Uses an index-based jitter to break ties deterministically."""
    n = ops.shape(metric)[0]
    idx = ops.arange(0, n, dtype="int32")
    # tiny jitter = (idx / n) * 1e-12 so it never flips a real strict ordering
    jitter = ops.divide(ops.cast(idx, "float32"), ops.cast(n, "float32"))
    metric_jittered = ops.add(metric, ops.multiply(jitter, 1e-12))
    # argsort by negative to get descending
    return ops.argsort(ops.negative(metric_jittered))


def gptq_quantize_matrix(
    weights_transpose,
    inv_hessian,
    *,
    blocksize=128,
    group_size=-1,
    activation_order=False,
    order_metric=None,
    compute_scale_zero=compute_quantization_parameters,
):
    """
    Implements the GPTQ error correction updates.

    For a single column update (column j):
        e = invH[j, j] * (w_j - q_j)
        W[:, j+1:] -= e * invH[j, j+1:]
    where:
    - w_j is the original column,
    - q_j is the quantized column,
    - invH is the inverse Hessian,
    - e is the propagated error term.

    Across entire blocks:
        W[:, future] -= E_block * invH[block, future]
    where:
    - E_block is the quantization error accumulated for the current block,
    - invH[block, future] denotes the cross-block slice of the inverse Hessian,
    - W[:, future] are the columns yet to be quantized.

    Args:
        weights_transpose: Transposed weight matrix [out_features, in_features]
         to quantize.
        inv_hessian: Inverse Hessian matrix [in_features, in_features] for
         error propagation.
        blocksize: Size of the blocks to process (default: 128).
        group_size: Size of the groups for parameter reuse
         (default: -1, no grouping).
        activation_order: Whether to apply activation-order permutation
         (default: False).
        order_metric: Metric for ordering features
         (default: None, uses 1 / diag(invH)).
        compute_scale_zero: Function to compute scale and zero for
         quantization.

    Returns:
        quantized_weights: Quantized weight matrix [out_features, in_features].
        scale: float32. Scale parameters for quantization
         [out_features, num_groups].
        zero: Zero-point parameters for quantization [out_features, num_groups].
        g_idx: int32. Group indices for each feature [in_features].
    """
    in_features = ops.shape(weights_transpose)[1]

    if activation_order:
        # Use 1 / diag(inverse_hessian) as importance proxy by default.
        if order_metric is None:
            order_metric = ops.reciprocal(
                ops.add(ops.diagonal(inv_hessian), 1e-12)
            )
        else:
            # sanitize provided metric
            order_metric = ops.cast(order_metric, "float32")
            order_metric = ops.where(
                ops.isfinite(order_metric),
                order_metric,
                ops.zeros_like(order_metric),
            )
        # Sort in descending order by importance
        perm = _stable_permutation(order_metric)
        inv_perm = ops.argsort(perm)

        weights_transpose = ops.take(weights_transpose, perm, axis=1)
        inv_hessian = ops.take(
            ops.take(inv_hessian, perm, axis=0), perm, axis=1
        )
    else:
        perm = inv_perm = None

    # weights_buffer: [out_features, in_features]
    weights_buffer = weights_transpose
    # Buffer for the final quantized matrix: [out_features, in_features]
    quantized_weights_buffer = ops.zeros_like(weights_transpose, dtype="int32")

    scale_chunks = []
    zero_chunks = []

    # Compute effective group size
    effective_group = in_features if group_size == -1 else group_size

    # Process features in blocks
    for block_start in range(0, in_features, blocksize):
        block_end = min(block_start + blocksize, in_features)
        block_size = block_end - block_start

        # Block views
        # block_weights: [out_features, block_size]
        block_weights = weights_buffer[:, block_start:block_end]
        # block_error: [out_features, block_size]
        block_error = ops.zeros_like(block_weights)
        # block_inv_hessian: [block_size, block_size]
        block_inv_hessian = inv_hessian[
            block_start:block_end, block_start:block_end
        ]

        # Per-group cached params for reuse within the group
        cached_scale = None
        cached_zero = None
        cached_maxq = None
        cached_group_start = -1

        for block_idx in range(block_size):
            # Current global column index, represents the original column
            # in the weight matrix
            global_idx = block_start + block_idx
            # weight_column: [out_features,]
            weight_column = block_weights[:, block_idx]
            # Group-wise parameter reuse (compute once per group)
            if not effective_group == in_features:  # group_size != -1
                # Determine the group start index for the current column
                group_start = (global_idx // effective_group) * effective_group
                if group_start != cached_group_start:
                    # New group encountered, compute & cache params
                    # for this group
                    group_end = min(group_start + effective_group, in_features)
                    group_slice = weights_buffer[:, group_start:group_end]
                    cached_scale, cached_zero, cached_maxq = compute_scale_zero(
                        group_slice
                    )
                    # Store params once per group (in the order encountered).
                    scale_chunks.append(cached_scale)
                    zero_chunks.append(cached_zero)
                    cached_group_start = group_start
                scale, zero, maxq = cached_scale, cached_zero, cached_maxq
            else:
                # Single global group covering all columns.
                if cached_scale is None:
                    cached_scale, cached_zero, cached_maxq = compute_scale_zero(
                        weights_buffer
                    )
                    scale_chunks.append(cached_scale)
                    zero_chunks.append(cached_zero)
                    cached_group_start = 0
                scale, zero, maxq = cached_scale, cached_zero, cached_maxq

            # Quantize column and store it.
            # quantized_column: [out_features, 1]
            quantized_column = quantize_with_zero_point(
                ops.expand_dims(weight_column, 1), scale, zero, maxq
            )

            # Store quantized column in the buffer.
            quantized_weights_buffer = ops.slice_update(
                quantized_weights_buffer,
                (0, global_idx),
                ops.cast(quantized_column, "int32"),
            )
            # Dequantize column to compute error.
            # dequantized_col: [out_features,]
            dequantized_col = dequantize_with_zero_point(
                quantized_column, scale, zero
            )[:, 0]
            # Error feedback for remaining columns within the block
            # block_inv_hessian_diag: scalar
            current_block_influence = block_inv_hessian[block_idx, block_idx]
            # We divide by current_block_influence to get the
            # correct scaling of the error term.
            err = ops.divide(
                ops.subtract(weight_column, dequantized_col),
                current_block_influence,
            )
            # Record error for propagation to future blocks
            block_error = ops.slice_update(
                block_error, (0, block_idx), ops.expand_dims(err, 1)
            )

            # Update remaining columns in the current block
            # (those before the current column have already been quantized)
            # Propagate error to remaining columns in the block.
            if block_idx < block_size - 1:
                # update: [out_features, block_size - block_idx - 1]
                update = ops.matmul(
                    ops.expand_dims(err, 1),
                    ops.expand_dims(
                        block_inv_hessian[block_idx, block_idx + 1 :], 0
                    ),
                )
                # tail is a view of the remaining columns in the block
                # to be updated
                # tail: [out_features, block_size - block_idx - 1]
                tail = block_weights[:, block_idx + 1 :]
                block_weights = ops.slice_update(
                    block_weights,
                    (0, block_idx + 1),
                    ops.subtract(tail, update),
                )

        # Propagate block errors to future features (beyond the block)
        if block_end < in_features:
            # Total update for all future columns, based on the
            # accumulated error in this block. This is calculated
            # as the matrix product of the block_error and the
            # relevant slice of the inverse Hessian.
            # total_update: [out_features, in_features - block_end]
            total_update = ops.matmul(
                block_error, inv_hessian[block_start:block_end, block_end:]
            )
            # Update the remaining weights in the buffer. This is done
            # by subtracting the total_update from the remaining columns.
            weights_buffer = ops.concatenate(
                [
                    weights_buffer[:, :block_end],
                    ops.subtract(weights_buffer[:, block_end:], total_update),
                ],
                axis=1,
            )

    # Build group indices for each (possibly permuted) column
    # base_group = effective_group (int)
    base_group = effective_group

    # g_idx in permuted domain
    g_idx = ops.arange(0, in_features, dtype="int32")
    g_idx = ops.divide(g_idx, base_group)
    g_idx = ops.cast(g_idx, "float32")

    # Map group indices and quantized weights back to original column order
    if activation_order:
        g_idx = ops.take(g_idx, inv_perm, axis=0)
        quantized_weights_buffer = ops.take(
            quantized_weights_buffer, inv_perm, axis=1
        )

    # Concatenate recorded group params
    if len(scale_chunks) == 0:
        # Edge case: no groups recorded (empty input); fall back to whole matrix
        s, z, _ = compute_scale_zero(weights_transpose)
        scale = s
        zero = z
    else:
        scale = ops.concatenate(scale_chunks, axis=1)
        zero = ops.concatenate(zero_chunks, axis=1)

    return quantized_weights_buffer, scale, zero, g_idx


class GPTQ:
    def __init__(self, layer, config=GPTQConfig(tokenizer=None, dataset=None)):
        self.original_layer = layer
        self.num_samples = 0
        self.config = config
        self.quantizer = GPTQQuantizer(
            config, compute_dtype=layer.variable_dtype
        )

        # Explicitly handle each supported layer type
        if isinstance(layer, Dense) or (
            isinstance(layer, EinsumDense) and layer.kernel.ndim == 2
        ):
            # For a standard Dense layer, the dimensions are straightforward.
            self.kernel_shape = layer.kernel.shape
            # rows: [input_features]
            self.rows = self.kernel_shape[0]
            # columns: [output_features]
            self.columns = self.kernel_shape[1]
            self.layer = layer

        # Handle 3D EinsumDense layers (typically from attention blocks).
        elif isinstance(layer, EinsumDense) and layer.kernel.ndim == 3:
            # For EinsumDense, we determine the effective 2D dimensions.
            self.kernel_shape = layer.kernel.shape
            shape = list(self.kernel_shape)
            d_model_dim_index = shape.index(max(shape))

            if d_model_dim_index == 0:  # QKV projection case
                in_features, heads, head_dim = shape
                self.rows, self.columns = (
                    in_features,
                    ops.multiply(heads, head_dim),
                )
            elif d_model_dim_index in [1, 2]:  # Attention Output case
                heads, head_dim, out_features = shape
                self.rows, self.columns = (
                    ops.multiply(heads, head_dim),
                    out_features,
                )

            # Create a temporary object that holds a reshaped
            # 2D version of the kernel.
            self.layer = types.SimpleNamespace(
                kernel=ops.reshape(layer.kernel, (self.rows, self.columns)),
            )
        else:
            # Raise an error if the layer is not supported.
            raise TypeError(f"Unsupported layer type for GPTQ: {type(layer)}")
        self.hessian = ops.zeros((self.rows, self.rows), dtype="float32")

    def update_hessian_with_batch(self, input_batch):
        """
        Updates the running average of the Hessian matrix with a new batch.

        This method computes the Hessian matrix for a given batch of input
        activations and updates the accumulated Hessian (`self.hessian`) using a
        numerically stable running average. This allows the Hessian to be
        computed over a large dataset without loading all samples into memory
        at once.

        The input tensor is first reshaped into a 2D matrix [num_samples,
        num_features] before the Hessian is calculated.

        Args:
            input_batch: A 2D or higher-dimensional tensor of input activations
                from a calibration batch.

        Raises:
            ValueError: If the feature dimension of the input tensor
                `input_batch` does not match the dimensions of the
                pre-initialized Hessian matrix `self.hessian`.
        """
        if input_batch is None:
            raise ValueError("Input tensor cannot be None.")

        if len(input_batch.shape) < 2:
            raise ValueError(
                "Input tensor must have rank >= 2 "
                f"(got rank {len(input_batch.shape)})."
            )
        if ops.size(input_batch) == 0:
            raise ValueError("Input tensor cannot be empty.")
        if len(input_batch.shape) > 2:
            # [batch, features]
            input_batch = ops.reshape(input_batch, (-1, input_batch.shape[-1]))
        x = ops.cast(input_batch, "float32")

        num_new_samples = ops.shape(x)[0]
        num_prev_samples = self.num_samples
        total_samples = ops.add(num_prev_samples, num_new_samples)

        if ops.shape(self.hessian)[0] != ops.shape(x)[-1]:
            raise ValueError(
                f"Hessian dimensions ({ops.shape(self.hessian)[0]}) do not "
                f"match input features ({ops.shape(x)[-1]})."
            )

        # gram_matrix: [features, features]
        gram_matrix = ops.matmul(ops.transpose(x), x)
        # Ensures numerical stability and symmetry in case of large floating
        # point activations.
        gram_matrix = ops.divide(
            ops.add(gram_matrix, ops.transpose(gram_matrix)), 2.0
        )

        # Decay previous mean and add current per-sample contribution
        # (factor 2/N)
        if self.num_samples > 0:
            self.hessian = ops.multiply(
                self.hessian, ops.divide(num_prev_samples, total_samples)
            )

        self.hessian = ops.add(
            self.hessian,
            ops.multiply(ops.divide(2.0, total_samples), gram_matrix),
        )

        self.num_samples = self.num_samples + ops.shape(x)[0] or 0

    def quantize_and_correct_layer(
        self,
        blocksize=128,
    ):
        """
        Performs GPTQ quantization and correction on the layer's weights.

        This method implements the core logic of the "Optimal Brain Quant"
        (OBQ) method, as applied by GPTQ, to quantize the weights of a single
        layer. It iteratively quantizes blocks of weights and corrects for the
        quantization error by updating the remaining weights.

        The algorithm follows these main steps:
        1.  Initialization: It optionally reorders the weight columns based
            on activation magnitudes (`activation_order=True`) to protect more
            salient
            weights.
        2.  Hessian Modification: The Hessian matrix, pre-computed from
            calibration data, is dampened to ensure its invertibility and
            stability.
        3.  Iterative Quantization: The function iterates through the
            weight columns in blocks (`blocksize`). In each iteration, it:
            a. Quantizes one column.
            b. Calculates the quantization error.
            c. Updates the remaining weights in the *current* block by
                distributing the error, using the inverse Hessian.
        4.  Block-wise Correction: After a block is quantized, the total
            error from that block is propagated to the *next* block of weights
            to be processed.
        5.  Finalization: The quantized weights are reordered back if
            `activation_order` was used, and the layer's weights are updated.
        This implementation is based on the official GPTQ paper and repository.
        For more details, see:
        - Paper: https://arxiv.org/abs/2210.17323
        - Original Code: https://github.com/IST-DASLab/gptq


        Args:
            blocksize: (int, optional) The size of the weight block to process
             at a time. Defaults to 128.
        """
        weights_matrix = ops.transpose(self.layer.kernel)

        # Dampen the Hessian for Stability
        hessian_diagonal = ops.diagonal(self.hessian)
        dead_diagonal = ops.equal(hessian_diagonal, 0.0)
        hessian_diagonal = ops.where(dead_diagonal, 1.0, hessian_diagonal)
        hessian_matrix = ops.add(
            self.hessian,
            ops.diag(
                ops.where(dead_diagonal, 1.0, ops.zeros_like(hessian_diagonal))
            ),
        )

        # Add dampening factor to the Hessian diagonal
        damping_factor = ops.multiply(
            self.config.hessian_damping, ops.mean(hessian_diagonal)
        )
        hessian_diagonal = ops.add(hessian_diagonal, damping_factor)
        hessian_matrix = ops.add(
            ops.subtract(
                hessian_matrix, ops.diag(ops.diagonal(hessian_matrix))
            ),
            ops.diag(hessian_diagonal),
        )

        # Compute the inverse Hessian, which is used for error correction
        inverse_hessian = linalg.inv(hessian_matrix)

        quantized, scale, zero, g_idx = gptq_quantize_matrix(
            weights_matrix,
            inv_hessian=inverse_hessian,
            blocksize=blocksize,
            group_size=self.config.group_size,
            activation_order=self.config.activation_order,
            order_metric=ops.diagonal(hessian_matrix),
            compute_scale_zero=partial(self.quantizer.find_params, weight=True),
        )
        quantized = ops.cast(
            quantized, self.original_layer.quantized_kernel.dtype
        )

        if self.config.weight_bits == 4:
            # For 4-bit weights, we need to pack them into bytes
            quantized, _, _ = quantizers.pack_int4(
                quantized, axis=0, dtype="uint8"
            )

        del self.original_layer._kernel
        self.original_layer.quantized_kernel.assign(quantized)
        self.original_layer.kernel_scale.assign(scale)
        self.original_layer.kernel_zero.assign(zero)
        self.original_layer.g_idx.assign(g_idx)
        self.original_layer.is_gptq_calibrated = True

    def free(self):
        del self.hessian
        del self.layer
