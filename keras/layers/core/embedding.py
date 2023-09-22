from keras import constraints
from keras import initializers
from keras import ops
from keras import regularizers
from keras.api_export import keras_export
from keras.layers.layer import Layer


@keras_export("keras.layers.Embedding")
class Embedding(Layer):
    """Turns positive integers (indexes) into dense vectors of fixed size.

    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

    This layer can only be used on positive integer inputs of a fixed range.

    Example:

    >>> model = keras.Sequential()
    >>> model.add(keras.layers.Embedding(1000, 64, input_length=10))
    >>> # The model will take as input an integer matrix of size (batch,
    >>> # input_length), and the largest integer (i.e. word index) in the input
    >>> # should be no larger than 999 (vocabulary size).
    >>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
    >>> # dimension.
    >>> input_array = np.random.randint(1000, size=(32, 10))
    >>> model.compile('rmsprop', 'mse')
    >>> output_array = model.predict(input_array)
    >>> print(output_array.shape)
    (32, 10, 64)

    Args:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
            This is useful when using recurrent layers which
            may take variable length input. If this is `True`,
            then all subsequent layers in the model need
            to support masking or an exception will be raised.
            If mask_zero is set to True, as a consequence,
            index 0 cannot be used in the vocabulary (input_dim should
            equal size of vocabulary + 1).

    Input shape:
        2D tensor with shape: `(batch_size, input_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, input_length, output_dim)`.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.autocast = False

    def build(self, input_shape=None):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        if inputs.dtype != "int32" and inputs.dtype != "int64":
            inputs = ops.cast(inputs, "int32")
        outputs = ops.take(self.embeddings, inputs, axis=0)
        return ops.cast(outputs, dtype=self.compute_dtype)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return ops.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "embeddings_initializer": initializers.serialize(
                self.embeddings_initializer
            ),
            "embeddings_regularizer": regularizers.serialize(
                self.embeddings_regularizer
            ),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "embeddings_constraint": constraints.serialize(
                self.embeddings_constraint
            ),
            "mask_zero": self.mask_zero,
        }
        return {**base_config, **config}
