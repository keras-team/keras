from keras.src import ops
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.quantizers.gptq_quant import dequantize


class GPTQ:
    def __init__(self, layer):
        self.original_layer = layer
        self.nsamples = 0
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
                self.rows, self.columns = in_features, heads * head_dim
            elif d_model_dim_index in [1, 2]:  # Attention Output case
                heads, head_dim, out_features = shape
                self.rows, self.columns = heads * head_dim, out_features

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
        self.H = ops.zeros((self.rows, self.rows), dtype="float32")

    def update_hessian_with_batch(self, inp):
        """
        Updates the running average of the Hessian matrix with a new batch.

        This method computes the Hessian matrix for a given batch of input
        activations and updates the accumulated Hessian (`self.H`) using a
        numerically stable running average. This allows the Hessian to be
        computed over a large dataset without loading all samples into memory
        at once.

        The input tensor is first reshaped into a 2D matrix [num_samples,
        num_features] before the Hessian is calculated.

        Args:
            inp: A 2D or higher-dimensional tensor of input activations from a
                calibration batch.

        Raises:
            ValueError: If the feature dimension of the input tensor `inp` does
                not match the dimensions of the pre-initialized Hessian matrix
                `self.H`.
        """
        if inp is None:
            raise ValueError("Input tensor 'inp' cannot be None.")

        if len(inp.shape) < 2:
            raise ValueError(
                f"Input tensor 'inp' must have a rank of at least 2 "
                f"(e.g., [batch, features]), but got rank {len(inp.shape)}."
            )
        if ops.size(inp) == 0:
            raise ValueError("Input tensor 'inp' cannot be empty.")

        if len(inp.shape) > 2:
            inp = ops.reshape(inp, (-1, inp.shape[-1]))
        inp = ops.cast(inp, "float32")

        if self.H.shape[0] != inp.shape[-1]:
            raise ValueError(
                f"Hessian dimensions ({self.H.shape[0]}) do not"
                "match input features ({inp.shape[-1]})."
            )

        current_H = ops.multiply(2, ops.matmul(ops.transpose(inp), inp))

        if self.nsamples == 0:
            self.H = current_H
        else:
            total_samples = ops.add(self.nsamples, inp.shape[0])
            old_H_weight = ops.divide(self.nsamples, total_samples)
            current_H_weight = ops.divide(inp.shape[0], total_samples)

            # Update the accumulated Hessian
            term1 = ops.multiply(self.H, old_H_weight)
            term2 = ops.multiply(current_H, current_H_weight)
            self.H = ops.add(term1, term2)

        self.nsamples = ops.add(self.nsamples, inp.shape[0])

    def quantize_and_correct_block(
        self, blocksize=128, percdamp=0.01, group_size=-1, actorder=False
    ):
        """
        Performs GPTQ quantization and correction on the layer's weights.

        This method implements the core logic of the "Optimal Brain Quant"
        (OBQ) method, as applied by GPTQ, to quantize the weights of a single
        layer. It iteratively quantizes blocks of weights and corrects for the
        quantization error by updating the remaining weights.

        The algorithm follows these main steps:
        1.  **Initialization**: It optionally reorders the weight columns based
            on activation magnitudes (`actorder=True`) to protect more salient
            weights.
        2.  **Hessian Modification**: The Hessian matrix `H`, pre-computed from
            calibration data, is dampened to ensure its invertibility and
            stability.
        3.  **Iterative Quantization**: The function iterates through the
            weight columns in blocks (`blocksize`). In each iteration, it:
            a. Quantizes one column (`w`).
            b. Calculates the quantization error (`err`).
            c. Updates the remaining weights in the *current* block by
                distributing the error, using the inverse Hessian (`Hinv`).
        4.  **Block-wise Correction**: After a block is quantized, the total
            error from that block is propagated to the *next* block of weights
            to be processed.
        5.  **Finalization**: The quantized weights (`Q`) are reordered back if
            `actorder` was used, and the layer's weights are updated.

        This implementation is based on the official GPTQ paper and repository.
        For more details, see:
        - Paper: https://arxiv.org/abs/2210.17323
        - Original Code: https://github.com/IST-DASLab/gptq

        Args:
            blocksize (int, optional): The size of the weight block to process
             at a time. Defaults to 128.
            percdamp (float, optional): The percentage of dampening to add the
                Hessian's diagonal. A value of 0.01 is recommended.
                Defaults to 0.01.
            group_size (int, optional): The number of weights that share the
                same quantization parameters (scale and zero-point).
                A value of -1 indicates per-channel quantization.
            actorder (bool, optional): If True, reorders weight columns based
                on their activation's second-order information.
        """

        W = ops.transpose(ops.cast(self.layer.kernel, "float32"))
        H = ops.cast(self.H, "float32")

        if actorder:
            perm = ops.argsort(-ops.diagonal(H))
            W = ops.take(W, perm, axis=1)
            H = ops.take(ops.take(H, perm, axis=0), perm, axis=1)
            invperm = ops.argsort(perm)

        # Dampen the Hessian for Stability
        diag_H = ops.diagonal(H)
        dead = ops.equal(diag_H, 0.0)
        diag_H = ops.where(dead, 1.0, diag_H)
        H = ops.add(H, ops.diag(ops.where(dead, 1.0, ops.zeros_like(diag_H))))

        # Add dampening factor to the Hessian diagonal
        damp = ops.multiply(percdamp, ops.mean(diag_H))
        diag_H = ops.add(diag_H, damp)
        H = ops.add(
            ops.subtract(H, ops.diag(ops.diagonal(H))), ops.diag(diag_H)
        )

        # Compute the inverse Hessian, which is used for error correction
        Hinv = ops.linalg.inv(H)
        Q = ops.zeros_like(W)

        for i1 in range(0, self.rows, blocksize):
            i2 = min(i1 + blocksize, self.rows)
            count = i2 - i1
            # Extract the current block of weights and its corresponding
            # Hessian
            W1 = W[:, i1:i2]
            Q1 = ops.zeros_like(W1)
            Err1 = ops.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Process one column at a time within the block
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if (i1 + i) % group_size == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i) : (i1 + i + group_size)], weight=True
                        )
                else:
                    self.quantizer.find_params(
                        ops.expand_dims(w, 1), weight=True
                    )

                # Quantize the current weight column
                q = dequantize(
                    ops.expand_dims(w, 1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )[:, 0]

                Q1 = ops.slice_update(Q1, (0, i), ops.expand_dims(q, axis=1))
                err = ops.divide(ops.subtract(w, q), d)
                Err1 = ops.slice_update(
                    Err1, (0, i), ops.expand_dims(err, axis=1)
                )

                if i < count - 1:
                    update = ops.matmul(
                        ops.expand_dims(err, 1),
                        ops.expand_dims(Hinv1[i, i + 1 :], 0),
                    )

                    # Efficiently update the remaining part of the W1 tensor.
                    slice_to_update = W1[:, i + 1 :]
                    updated_slice = ops.subtract(slice_to_update, update)
                    W1 = ops.slice_update(W1, (0, i + 1), updated_slice)

            # Update the full quantized matrix Q with the processed block
            Q = ops.concatenate([Q[:, :i1], Q1, Q[:, i2:]], axis=1)

            if i2 < self.rows:
                update_total = ops.matmul(Err1, Hinv[i1:i2, i2:])
                W = ops.concatenate(
                    [W[:, :i2], ops.subtract(W[:, i2:], update_total)], axis=1
                )

        if actorder:
            Q = ops.take(Q, invperm, axis=1)

        Q = ops.transpose(Q)

        if isinstance(self.original_layer, EinsumDense):
            Q = ops.reshape(Q, self.kernel_shape)

        # Set the new quantized weights in the original layer
        new_weights = [ops.convert_to_numpy(Q)]
        if self.original_layer.bias is not None:
            new_weights.append(ops.convert_to_numpy(self.original_layer.bias))

        self.original_layer.set_weights(new_weights)

    def free(self):
        self.H = None
