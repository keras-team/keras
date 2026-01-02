import copy

from keras.src import dtype_policies
from keras.src import layers
from keras.src import ops
from keras.src import quantizers
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.quantizers.quantization_config import QuantizationConfig


@keras_export("keras.layers.ReversibleEmbedding")
class ReversibleEmbedding(layers.Embedding):
    """An embedding layer which can project backwards to the input dim.

    This layer is an extension of `keras.layers.Embedding` for language models.
    This layer can be called "in reverse" with `reverse=True`, in which case the
    layer will linearly project from `output_dim` back to `input_dim`.

    By default, the reverse projection will use the transpose of the
    `embeddings` weights to project to `input_dim` (weights are "tied"). If
    `tie_weights=False`, the model will use a separate, trainable variable for
    reverse projection.

    This layer has no bias terms.

    Args:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        tie_weights: Boolean, whether or not the matrix for embedding and
            the matrix for the `reverse` projection should share the same
            weights.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
        reverse_dtype: The dtype for the reverse projection computation.
            Defaults to the `compute_dtype` of the layer.
        logit_soft_cap: If `logit_soft_cap` is set and `reverse=True`, the
            output logits will be scaled by
            `tanh(logits / logit_soft_cap) * logit_soft_cap`. This narrows the
            range of output logits and can improve training.
        **kwargs: other keyword arguments passed to `keras.layers.Embedding`,
            including `name`, `trainable`, `dtype` etc.

    Call arguments:
        inputs: The tensor inputs to the layer.
        reverse: Boolean. If `True` the layer will perform a linear projection
            from `output_dim` to `input_dim`, instead of a normal embedding
            call. Default to `False`.

    Example:
    ```python
    batch_size = 16
    vocab_size = 100
    hidden_dim = 32
    seq_length = 50

    # Generate random inputs.
    token_ids = np.random.randint(vocab_size, size=(batch_size, seq_length))

    embedding = keras.layers.ReversibleEmbedding(vocab_size, hidden_dim)
    # Embed tokens to shape `(batch_size, seq_length, hidden_dim)`.
    hidden_states = embedding(token_ids)
    # Project hidden states to shape `(batch_size, seq_length, vocab_size)`.
    logits = embedding(hidden_states, reverse=True)
    ```

    References:
    - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    - [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        tie_weights=True,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        reverse_dtype=None,
        logit_soft_cap=None,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            **kwargs,
        )
        self.tie_weights = tie_weights
        self.reverse_dtype = reverse_dtype
        self.logit_soft_cap = logit_soft_cap

    def build(self, inputs_shape=None):
        super().build(inputs_shape)
        if not self.tie_weights and self.quantization_mode not in (
            "int8",
            "int4",
        ):
            self.reverse_embeddings = self.add_weight(
                shape=(self.output_dim, self.input_dim),
                initializer=self.embeddings_initializer,
                name="reverse_embeddings",
                trainable=True,
            )

    def call(self, inputs, reverse=False):
        if not reverse:
            return super().call(inputs)
        else:
            if self.tie_weights:
                kernel = ops.transpose(ops.convert_to_tensor(self.embeddings))
            else:
                kernel = self.reverse_embeddings
            if self.reverse_dtype is not None:
                inputs = ops.cast(inputs, self.reverse_dtype)
                kernel = ops.cast(kernel, self.reverse_dtype)
            logits = ops.matmul(inputs, kernel)
            # Optionally soft-cap logits.
            if self.logit_soft_cap is not None:
                soft_cap = self.logit_soft_cap
                logits = ops.multiply(
                    ops.tanh(ops.divide(logits, soft_cap)), soft_cap
                )
            return logits

    def compute_output_shape(self, input_shape, reverse=False):
        output_shape = list(input_shape)
        if reverse:
            output_shape[-1] = self.input_dim
        else:
            output_shape += [self.output_dim]
        return output_shape

    def compute_output_spec(self, inputs, reverse=False):
        output_shape = list(inputs.shape)
        if reverse:
            output_shape[-1] = self.input_dim
        else:
            output_shape += [self.output_dim]
        return KerasTensor(output_shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tie_weights": self.tie_weights,
                "reverse_dtype": self.reverse_dtype,
                "logit_soft_cap": self.logit_soft_cap,
            }
        )
        return config

    @property
    def variable_serialization_spec(self):
        # Avoid modifying the parent's spec.
        _spec = copy.deepcopy(super().variable_serialization_spec)
        if not self.tie_weights:
            for mode, variable_spec in _spec.items():
                variable_spec.append("reverse_embeddings")
                if mode in ("int4", "int8"):
                    variable_spec.append("reverse_embeddings_scale")
        return _spec

    def quantized_build(self, embeddings_shape, mode, config=None):
        if mode == "int8":
            self._int8_build(embeddings_shape, config)
        elif mode == "int4":
            self._int4_build(embeddings_shape, config)
        else:
            raise self._quantization_mode_error(mode)
        self._is_quantized = True

    def _int8_build(self, embeddings_shape, config=None):
        if embeddings_shape is None:
            embeddings_shape = (self.input_dim, self.output_dim)
        super()._int8_build(embeddings_shape=embeddings_shape)

        self.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(
                config, quantizers.AbsMaxQuantizer(axis=-1)
            )
        )
        if not self.tie_weights:
            self.reverse_embeddings = self.add_weight(
                name="reverse_embeddings",
                shape=(self.output_dim, self.input_dim),
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            self.reverse_embeddings_scale = self.add_weight(
                name="reverse_embeddings_scale",
                shape=(self.input_dim,),
                initializer="ones",
                trainable=False,
            )

    def _int4_build(self, embeddings_shape, config=None):
        if embeddings_shape is None:
            embeddings_shape = (self.input_dim, self.output_dim)
        super()._int4_build(embeddings_shape=embeddings_shape, config=config)

        self.inputs_quantizer = (
            QuantizationConfig.activation_quantizer_or_default(
                config, quantizers.AbsMaxQuantizer(axis=-1)
            )
        )
        if not self.tie_weights:
            packed_rows = (self.output_dim + 1) // 2  # ceil for odd dims
            self.reverse_embeddings = self.add_weight(
                name="reverse_embeddings",
                shape=(packed_rows, self.input_dim),
                initializer="zeros",
                dtype="int8",
                trainable=False,
            )
            self.reverse_embeddings_scale = self.add_weight(
                name="reverse_embeddings_scale",
                shape=(self.input_dim,),
                initializer="ones",
                trainable=False,
            )

    def _int8_call(self, inputs, reverse=False):
        if not reverse:
            return super()._int8_call(inputs)
        else:
            if self.tie_weights:
                kernel = ops.transpose(self._embeddings)
                scale = ops.transpose(self.embeddings_scale)
            else:
                kernel = self.reverse_embeddings
                scale = self.reverse_embeddings_scale
            if self.inputs_quantizer:
                inputs, inputs_scale = self.inputs_quantizer(inputs)
            else:
                inputs_scale = ops.ones((1,), dtype=self.compute_dtype)
            logits = ops.matmul(inputs, kernel)
            # De-scale outputs
            logits = ops.cast(logits, self.compute_dtype)
            logits = ops.divide(logits, ops.multiply(inputs_scale, scale))
            # Optionally soft-cap logits.
            if self.logit_soft_cap is not None:
                soft_cap = self.logit_soft_cap
                logits = ops.multiply(
                    ops.tanh(ops.divide(logits, soft_cap)), soft_cap
                )
            return logits

    def _int4_call(self, inputs, reverse=False):
        if not reverse:
            return super()._int4_call(inputs)
        else:
            if self.tie_weights:
                embeddings = ops.transpose(self._embeddings)
                scale = ops.transpose(self.embeddings_scale)
            else:
                embeddings = self.reverse_embeddings
                scale = self.reverse_embeddings_scale
            unpacked_embeddings = quantizers.unpack_int4(
                embeddings, self.output_dim, axis=0
            )
            if self.inputs_quantizer:
                inputs, inputs_scale = self.inputs_quantizer(inputs)
            else:
                inputs_scale = ops.ones((1,), dtype=self.compute_dtype)
            logits = ops.matmul(inputs, unpacked_embeddings)
            # De-scale outputs
            logits = ops.cast(logits, self.compute_dtype)
            logits = ops.divide(logits, ops.multiply(inputs_scale, scale))
            # Optionally soft-cap logits.
            if self.logit_soft_cap is not None:
                soft_cap = self.logit_soft_cap
                logits = ops.multiply(
                    ops.tanh(ops.divide(logits, soft_cap)), soft_cap
                )
            return logits

    def quantize(self, mode=None, type_check=True, config=None):
        if type_check and type(self) is not ReversibleEmbedding:
            raise self._not_implemented_error(self.quantize)

        self.quantization_config = config

        embeddings_shape = (self.input_dim, self.output_dim)
        if mode == "int8":
            # Quantize `self._embeddings` to int8 and compute corresponding
            # scale.
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                self.quantization_config, quantizers.AbsMaxQuantizer(axis=-1)
            )
            embeddings_value, embeddings_scale = weight_quantizer(
                self._embeddings, to_numpy=True
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            del self._embeddings
            if not self.tie_weights:
                reverse_weight_quantizer = (
                    QuantizationConfig.weight_quantizer_or_default(
                        self.quantization_config,
                        quantizers.AbsMaxQuantizer(axis=0),
                    )
                )
                reverse_embeddings_value, reverse_embeddings_scale = (
                    reverse_weight_quantizer(
                        self.reverse_embeddings, to_numpy=True
                    )
                )
                reverse_embeddings_scale = ops.squeeze(
                    reverse_embeddings_scale, axis=0
                )
                del self.reverse_embeddings
            self.quantized_build(
                embeddings_shape, mode, self.quantization_config
            )
            self._embeddings.assign(embeddings_value)
            self.embeddings_scale.assign(embeddings_scale)
            if not self.tie_weights:
                self.reverse_embeddings.assign(reverse_embeddings_value)
                self.reverse_embeddings_scale.assign(reverse_embeddings_scale)
        elif mode == "int4":
            # Quantize to int4 values (stored in int8 dtype, range [-8, 7]).
            weight_quantizer = QuantizationConfig.weight_quantizer_or_default(
                self.quantization_config,
                quantizers.AbsMaxQuantizer(
                    axis=-1,
                    value_range=(-8, 7),
                    output_dtype="int8",
                ),
            )
            embeddings_value, embeddings_scale = weight_quantizer(
                self._embeddings, to_numpy=True
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            # 2. Pack two int4 values into a single int8 byte.
            packed_embeddings_value, _, _ = quantizers.pack_int4(
                embeddings_value, axis=-1
            )
            del self._embeddings
            if not self.tie_weights:
                reverse_weight_quantizer = (
                    QuantizationConfig.weight_quantizer_or_default(
                        self.quantization_config,
                        quantizers.AbsMaxQuantizer(
                            axis=0,
                            value_range=(-8, 7),
                            output_dtype="int8",
                        ),
                    )
                )
                reverse_embeddings_value, reverse_embeddings_scale = (
                    reverse_weight_quantizer(
                        self.reverse_embeddings, to_numpy=True
                    )
                )
                reverse_embeddings_scale = ops.squeeze(
                    reverse_embeddings_scale, axis=0
                )
                # Pack two int4 values into a single int8 byte.
                packed_reverse_embeddings_value, _, _ = quantizers.pack_int4(
                    reverse_embeddings_value, axis=0
                )
                del self.reverse_embeddings
            self.quantized_build(
                embeddings_shape, mode, self.quantization_config
            )
            self._embeddings.assign(packed_embeddings_value)
            self.embeddings_scale.assign(embeddings_scale)
            if not self.tie_weights:
                self.reverse_embeddings.assign(packed_reverse_embeddings_value)
                self.reverse_embeddings_scale.assign(reverse_embeddings_scale)
        else:
            raise self._quantization_mode_error(mode)

        # Set new dtype policy.
        if self.dtype_policy.quantization_mode is None:
            policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
            self.dtype_policy = policy
