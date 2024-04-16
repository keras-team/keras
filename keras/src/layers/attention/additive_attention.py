from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.attention.attention import Attention


@keras_export("keras.layers.AdditiveAttention")
class AdditiveAttention(Attention):
    """Additive attention layer, a.k.a. Bahdanau-style attention.

    Inputs are a list with 2 or 3 elements:
    1. A `query` tensor of shape `(batch_size, Tq, dim)`.
    2. A `value` tensor of shape `(batch_size, Tv, dim)`.
    3. A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none
        supplied, `value` will be used as `key`.

    The calculation follows the steps:
    1. Calculate attention scores using `query` and `key` with shape
        `(batch_size, Tq, Tv)` as a non-linear sum
        `scores = reduce_sum(tanh(query + key), axis=-1)`.
    2. Use scores to calculate a softmax distribution with shape
        `(batch_size, Tq, Tv)`.
    3. Use the softmax distribution to create a linear combination of `value`
        with shape `(batch_size, Tq, dim)`.

    Args:
        use_scale: If `True`, will create a scalar variable to scale the
            attention scores.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            attention scores. Defaults to `0.0`.

    Call Args:
        inputs: List of the following tensors:
            - `query`: Query tensor of shape `(batch_size, Tq, dim)`.
            - `value`: Value tensor of shape `(batch_size, Tv, dim)`.
            - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If
                not given, will use `value` for both `key` and `value`, which is
                the most common case.
        mask: List of the following tensors:
            - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`.
                If given, the output will be zero at the positions where
                `mask==False`.
            - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`.
                If given, will apply the mask such that values at positions
                 where `mask==False` do not contribute to the result.
        return_attention_scores: bool, it `True`, returns the attention scores
            (after masking and softmax) as an additional output argument.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
        use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds
            a mask such that position `i` cannot attend to positions `j > i`.
            This prevents the flow of information from the future towards the
            past. Defaults to `False`.

    Output:
        Attention outputs of shape `(batch_size, Tq, dim)`.
        (Optional) Attention scores after masking and softmax with shape
            `(batch_size, Tq, Tv)`.
    """

    def __init__(
        self,
        use_scale=True,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(use_scale=use_scale, dropout=dropout, **kwargs)

    def build(self, input_shape):
        self._validate_inputs(input_shape)
        dim = input_shape[0][-1]
        self.scale = None
        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=[dim],
                initializer="glorot_uniform",
                dtype=self.dtype,
                trainable=True,
            )
        self.built = True

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a nonlinear sum of query and key.

        Args:
            query: Query tensor of shape `(batch_size, Tq, dim)`.
            key: Key tensor of shape `(batch_size, Tv, dim)`.

        Returns:
            Tensor of shape `(batch_size, Tq, Tv)`.
        """
        # Reshape tensors to enable broadcasting.
        # Reshape into [batch_size, Tq, 1, dim].
        q_reshaped = ops.expand_dims(query, axis=-2)
        # Reshape into [batch_size, 1, Tv, dim].
        k_reshaped = ops.expand_dims(key, axis=-3)
        scale = self.scale if self.use_scale else 1.0
        return ops.sum(scale * ops.tanh(q_reshaped + k_reshaped), axis=-1)

    def get_config(self):
        base_config = super().get_config()
        del base_config["score_mode"]
        return base_config
