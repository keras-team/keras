from keras.src import ops
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.quantizers.gptq_quant import dequantize


class GPTQ:
    def __init__(self, layer):
        self.original_layer = layer
        self.num_samples = 0
        self.quantizer = None

        # Explicitly handle each supported layer type
        if isinstance(layer, Dense) or (
            isinstance(layer, EinsumDense) and layer.kernel.ndim == 2
        ):
            # For a standard Dense layer, the dimensions are straightforward.
            self.kernel_shape = layer.kernel.shape
            self.rows = self.kernel_shape[0]  # Input features
            self.columns = self.kernel_shape[1]  # Output features
            self.layer = layer  # The layer itself can be used directly.

        # Handle 3D EinsumDense layers (typically from attention blocks).
        elif isinstance(layer, EinsumDense) and layer.kernel.ndim == 3:
            # For EinsumDense, we determine the effective 2D dimensions.
            self.kernel_shape = layer.kernel.shape
            shape = list(self.kernel_shape)
            try:
                d_model_dim_index = shape.index(max(shape))
            except ValueError:
                raise TypeError(
                    f"Could not determine hidden dimension from shape {shape}"
                )

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
            self.layer = type(
                "temp",
                (object,),
                {
                    "kernel": ops.reshape(
                        layer.kernel, (self.rows, self.columns)
                    ),
                    "bias": layer.bias,
                },
            )()

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
            raise ValueError("Input tensor 'input_batch' cannot be None.")

        if len(input_batch.shape) < 2:
            raise ValueError(
                f"Input tensor 'input_batch' must have a rank of at least 2 "
                f"(e.g., [batch, features]), but got rank "
                f"{len(input_batch.shape)}."
            )
        if ops.size(input_batch) == 0:
            raise ValueError("Input tensor 'input_batch' cannot be empty.")

        if len(input_batch.shape) > 2:
            input_batch = ops.reshape(input_batch, (-1, input_batch.shape[-1]))
        input_batch = ops.cast(input_batch, "float32")

        if self.hessian.shape[0] != input_batch.shape[-1]:
            raise ValueError(
                f"Hessian dimensions ({self.hessian.shape[0]}) do not"
                "match input features ({input_batch.shape[-1]})."
            )

        current_hessian = ops.multiply(
            2, ops.matmul(ops.transpose(input_batch), input_batch)
        )

        if self.num_samples == 0:
            self.hessian = current_hessian
        else:
            total_samples = ops.add(self.num_samples, input_batch.shape[0])
            old_hessian_weight = ops.divide(self.num_samples, total_samples)
            current_hessian_weight = ops.divide(
                input_batch.shape[0], total_samples
            )

            # Update the accumulated Hessian
            old_term = ops.multiply(self.hessian, old_hessian_weight)
            current_term = ops.multiply(current_hessian, current_hessian_weight)
            self.hessian = ops.add(old_term, current_term)

        self.num_samples = ops.add(self.num_samples, input_batch.shape[0])

    def quantize_and_correct_block(
        self,
        blocksize=128,
        hessian_damping=0.01,
        group_size=-1,
        activation_order=False,
    ):
        """
        Performs GPTQ quantization and correction on the layer's weights.

        This method implements the core logic of the "Optimal Brain Quant"
        (OBQ) method, as applied by GPTQ, to quantize the weights of a single
        layer. It iteratively quantizes blocks of weights and corrects for the
        quantization error by updating the remaining weights.

        The algorithm follows these main steps:
        1.  **Initialization**: It optionally reorders the weight columns based
            on activation magnitudes (`activation_order=True`) to protect more
            salient
            weights.
        2.  **Hessian Modification**: The Hessian matrix, pre-computed from
            calibration data, is dampened to ensure its invertibility and
            stability.
        3.  **Iterative Quantization**: The function iterates through the
            weight columns in blocks (`blocksize`). In each iteration, it:
            a. Quantizes one column.
            b. Calculates the quantization error.
            c. Updates the remaining weights in the *current* block by
                distributing the error, using the inverse Hessian.
        4.  **Block-wise Correction**: After a block is quantized, the total
            error from that block is propagated to the *next* block of weights
            to be processed.
        5.  **Finalization**: The quantized weights are reordered back if
            `activation_order` was used, and the layer's weights are updated.

        This implementation is based on the official GPTQ paper and repository.
        For more details, see:
        - Paper: https://arxiv.org/abs/2210.17323
        - Original Code: https://github.com/IST-DASLab/gptq

        Args:
            blocksize: (int, optional) The size of the weight block to process
             at a time. Defaults to 128.
            hessian_damping: (float, optional) The percentage of dampening to
                add the
                Hessian's diagonal. A value of 0.01 is recommended.
                Defaults to 0.01.
            group_size: (int, optional) The number of weights that share the
                same quantization parameters (scale and zero-point).
                A value of -1 indicates per-channel quantization.
            activation_order: (bool, optional) If True, reorders weight columns
                based
                on their activation's second-order information.
        """

        weights_matrix = ops.transpose(ops.cast(self.layer.kernel, "float32"))
        hessian_matrix = ops.cast(self.hessian, "float32")

        if activation_order:
            permutation = ops.argsort(
                ops.negative(ops.diagonal(hessian_matrix))
            )
            weights_matrix = ops.take(weights_matrix, permutation, axis=1)
            hessian_matrix = ops.take(
                ops.take(hessian_matrix, permutation, axis=0),
                permutation,
                axis=1,
            )
            inverse_permutation = ops.argsort(permutation)

        # Dampen the Hessian for Stability
        hessian_diagonal = ops.diagonal(hessian_matrix)
        dead_diagonal = ops.equal(hessian_diagonal, 0.0)
        hessian_diagonal = ops.where(dead_diagonal, 1.0, hessian_diagonal)
        hessian_matrix = ops.add(
            hessian_matrix,
            ops.diag(
                ops.where(dead_diagonal, 1.0, ops.zeros_like(hessian_diagonal))
            ),
        )

        # Add dampening factor to the Hessian diagonal
        damping_factor = ops.multiply(
            hessian_damping, ops.mean(hessian_diagonal)
        )
        hessian_diagonal = ops.add(hessian_diagonal, damping_factor)
        hessian_matrix = ops.add(
            ops.subtract(
                hessian_matrix, ops.diag(ops.diagonal(hessian_matrix))
            ),
            ops.diag(hessian_diagonal),
        )

        # Compute the inverse Hessian, which is used for error correction
        inverse_hessian = ops.linalg.inv(hessian_matrix)
        quantized_weights = ops.zeros_like(weights_matrix)

        for block_start in range(0, self.rows, blocksize):
            block_end = min(ops.add(block_start, blocksize), self.rows)
            block_size = ops.subtract(block_end, block_start)
            # Extract the current block of weights and its corresponding
            # Hessian
            block_weights = weights_matrix[:, block_start:block_end]
            block_quantized = ops.zeros_like(block_weights)
            block_errors = ops.zeros_like(block_weights)
            block_inverse_hessian = inverse_hessian[
                block_start:block_end, block_start:block_end
            ]

            # Process one column at a time within the block
            for col_idx in range(block_size):
                weight_column = block_weights[:, col_idx]
                diagonal_element = block_inverse_hessian[col_idx, col_idx]

                if group_size != -1:
                    if ops.mod(ops.add(block_start, col_idx), group_size) == 0:
                        self.quantizer.find_params(
                            weights_matrix[
                                :,
                                (ops.add(block_start, col_idx)) : (
                                    ops.add(
                                        ops.add(block_start, col_idx),
                                        group_size,
                                    )
                                ),
                            ],
                            weight=True,
                        )
                else:
                    self.quantizer.find_params(
                        ops.expand_dims(weight_column, 1), weight=True
                    )

                # Quantize the current weight column
                quantized_column = dequantize(
                    ops.expand_dims(weight_column, 1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )[:, 0]

                block_quantized = ops.slice_update(
                    block_quantized,
                    (0, col_idx),
                    ops.expand_dims(quantized_column, axis=1),
                )
                quantization_error = ops.divide(
                    ops.subtract(weight_column, quantized_column),
                    diagonal_element,
                )
                block_errors = ops.slice_update(
                    block_errors,
                    (0, col_idx),
                    ops.expand_dims(quantization_error, axis=1),
                )

                if ops.less(col_idx, ops.subtract(block_size, 1)):
                    error_update = ops.matmul(
                        ops.expand_dims(quantization_error, 1),
                        ops.expand_dims(
                            block_inverse_hessian[
                                col_idx, ops.add(col_idx, 1) :
                            ],
                            0,
                        ),
                    )

                    # Efficiently update the remaining part of the
                    # block_weights tensor.
                    slice_to_update = block_weights[:, ops.add(col_idx, 1) :]
                    updated_slice = ops.subtract(slice_to_update, error_update)
                    block_weights = ops.slice_update(
                        block_weights, (0, ops.add(col_idx, 1)), updated_slice
                    )

            # Update the full quantized matrix with the processed block
            quantized_weights = ops.concatenate(
                [
                    quantized_weights[:, :block_start],
                    block_quantized,
                    quantized_weights[:, block_end:],
                ],
                axis=1,
            )

            if block_end < self.rows:
                total_error_update = ops.matmul(
                    block_errors,
                    inverse_hessian[block_start:block_end, block_end:],
                )
                weights_matrix = ops.concatenate(
                    [
                        weights_matrix[:, :block_end],
                        ops.subtract(
                            weights_matrix[:, block_end:], total_error_update
                        ),
                    ],
                    axis=1,
                )

        if activation_order:
            quantized_weights = ops.take(
                quantized_weights, inverse_permutation, axis=1
            )

        quantized_weights = ops.transpose(quantized_weights)

        if isinstance(self.original_layer, EinsumDense):
            quantized_weights = ops.reshape(
                quantized_weights, self.kernel_shape
            )

        # Set the new quantized weights in the original layer
        new_weights = [ops.convert_to_numpy(quantized_weights)]
        if self.original_layer.bias is not None:
            new_weights.append(ops.convert_to_numpy(self.original_layer.bias))

        self.original_layer.set_weights(new_weights)

    def free(self):
        self.hessian = None
