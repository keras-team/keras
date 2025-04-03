"""A layer that updates its vocab and embedding matrix during training."""

import tensorflow as tf
from absl import logging
from tensorflow.python.util.tf_export import keras_export

import tf_keras.src as tf_keras
from tf_keras.src.layers import Layer
from tf_keras.src.layers.experimental import dynamic_lookup
from tf_keras.src.utils import warmstart_embedding_matrix


@keras_export("keras.layers.experimental.DynamicEmbedding")
class DynamicEmbedding(Layer):
    """A layer that updates its vocab and embedding matrix during training.

      DynamicEmbedding allows for the continuous updating of the vocabulary
      and embeddings during the training process. In traditional methods, the
      vocabulary and mapping to the embedding vector are set at the beginning of
      the training process and remain fixed throughout the training process.
      However, in many real-world scenarios, the vocabulary and mapping to the
      embeddings need to be updated to reflect the changing nature of the data.

      For instance, in natural language processing tasks, the vocabulary of
      words in a corpus may change over time, and it's important to update the
      embeddings to reflect these changes. Similarly, in recommendation systems,
      the items in the vocabulary may change over time.

      A layer that supports dynamic embeddings addresses this issue by allowing
      for the continuous updating of the vocabulary and embeddings during the
      training process. and it also updates the embedding matrix to reflect the
      new vocabulary.

      This layer maintains a hash table to track the most up-to-date vocabulary
      based on the inputs received by the layer and the eviction policy. When
      this layer is used with an `UpdateEmbeddingCallback`, which is a
      time-based callback, the vocabulary lookup tensor is updated at the time
      interval set in the `UpdateEmbeddingCallback` based on the most up-to-date
      vocabulary hash table maintained by the layer. If this layer is not used
      in conjunction with `UpdateEmbeddingCallback` the behavior of the layer
      would be same as `keras.layers.Embedding`.

    Args:
      input_dim: Size of the vocabulary in the input data. Expects an integer.
      output_dim: The size of the embedding space. Expects an integer.
      initial_vocabulary: The vocabulary to initialize the layer with. If a 1D
        tensor is provided, the vocabulary will be initialized with that tensor.
        If a `tf.DType` object is provided, a random tensor of that dtype and of
        length `input_dim` will be generated as the initial vocabulary.
        Supported `tf.DType` values include `tf.int32`, `tf.int64` and
        `tf.string`.
      eviction_policy: The eviction policy for the vocabulary. Available options
        are "LFU" (Least Frequently Used) and *more to come*. Defaults to "LFU".
        Expects a string.
      input_length:  Length of input sequences, when it is constant. This
        argument is required if you are going to connect `Flatten` then `Dense`
        layers upstream (without it, the shape of the dense outputs cannot be
        computed).Expects an integer.
      embedding_initializer: Initializer for embedding vectors for new input
        vocab tokens to be added to the updated embedding matrix (see
        keras.initializers). Defaults to "uniform".
      num_oov_indices: Number of out of vocabulary token to use. Currently
        supports 1. Expects an integer.
      **kwargs: Additional keyword arguments for the parent class.

    Attributes:
      embedding_layer: Embedding layer of DynamicEmbedding layer.
      dynamic_lookup_layer: DynamicLookup layer of DynamicEmbedding layer.
      embedding_initializer: Initializer for embedding vectors for new input
        vocab tokens to be added to the updated embedding matrix (see
        keras.initializers).
      num_oov_indices: Number of out of vocabulary token to use.

    Example:
    ```
      # Generate dummy data
      train_data = np.array([
          ['a', 'j', 'c', 'd', 'e'],
          ['a', 'h', 'i', 'j', 'b'],
          ['i', 'h', 'c', 'j', 'e'],
      ])
      train_labels = np.array([0, 1, 2])
      vocab = tf.constant(['a', 'b', 'c', 'd', 'e'])
      eviction_policy = 'LFU'
      # Define the model
      model = tf.keras.models.Sequential([
          DynamicEmbedding(
              input_dim=5,
              output_dim=2,
              input_length=5,
              eviction_policy=eviction_policy,
              initial_vocabulary=vocab,
          ),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(3, activation='softmax'),
      ])

      # Compile the model
      model.compile(
          optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'],
      )
      # update the vocabulary every 1 second
      update_embedding_callback = UpdateEmbeddingCallback(
          model.layers[0], interval=1
      )
      with update_embedding_callback:
        result = model.fit(
            train_data,
            train_labels,
            epochs=100,
            batch_size=1,
            callbacks=[update_embedding_callback],
        )
    ```
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        initial_vocabulary,
        eviction_policy="LFU",
        input_length=None,
        embedding_initializer="uniform",
        num_oov_indices=1,
        **kwargs,
    ):
        """Initialize DynamicEmbedding layer."""

        super().__init__(**kwargs)
        # assuming one oov bucket for now
        self.embedding_layer = tf_keras.layers.Embedding(
            input_dim=input_dim + num_oov_indices,
            output_dim=output_dim,
            embeddings_initializer=embedding_initializer,
            input_length=input_length,
            **kwargs,
        )
        self.dynamic_lookup_layer = dynamic_lookup.DynamicLookup(
            vocabulary_size=input_dim,
            eviction_policy=eviction_policy,
            initial_vocabulary=initial_vocabulary,
            **kwargs,
        )
        self.embedding_initializer = embedding_initializer
        self.num_oov_indices = num_oov_indices

    def build(self, input_shape=None):
        self.embedding_layer.build(input_shape)
        self.dynamic_lookup_layer.build(input_shape)

    def call(self, inputs, learn_vocab=True):
        # get vocab to index mapped for dynamic_lookup_layer
        output = self.dynamic_lookup_layer(inputs, learn_vocab=learn_vocab)
        # pass the indices as inputs to embedding_layer
        return self.embedding_layer(output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.embedding_layer.input_dim,
                "output_dim": self.embedding_layer.output_dim,
                "input_length": self.embedding_layer.input_length,
                "eviction_policy": self.dynamic_lookup_layer.eviction_policy,
                "initial_vocabulary": (
                    self.dynamic_lookup_layer.initial_vocabulary.numpy().tolist()  # noqa E501
                ),
                "embedding_initializer": self.embedding_initializer,
                "num_oov_indices": self.num_oov_indices,
            }
        )
        return config

    def get_vocabulary(self):
        return self.dynamic_lookup_layer.get_vocabulary()

    def save_assets(self, dir_path):
        initial_vocabulary = (
            self.dynamic_lookup_layer.initial_vocabulary.numpy().tolist()
        )
        initial_vocabulary_filepath = tf.io.gfile.join(
            dir_path, "initial_vocabulary.txt"
        )
        with open(initial_vocabulary_filepath, "w") as f:
            f.write("\n".join([str(w) for w in initial_vocabulary]))

    def update_embeddings(self, strategy):
        """Update embedding matrix of dynamic embedding layer."""
        try:
            if isinstance(strategy, tf.distribute.ParameterServerStrategy):
                # if using PSS agrregate values
                keys_list = (
                    self.dynamic_lookup_layer.vocabulary_table_keys.read_all()
                )
                values_list = (
                    self.dynamic_lookup_layer.vocabulary_table_values.read_all()
                )
                keys, values = self.aggregate_lookup_table(
                    keys_list, values_list
                )
            else:
                # if using on device strategy, just read values
                keys, values = (
                    self.dynamic_lookup_layer.vocabulary_table_keys,
                    self.dynamic_lookup_layer.vocabulary_table_values,
                )
            old_vocab = self.dynamic_lookup_layer.vocabulary
            new_vocab = self.get_top_vocabulary(
                keys,
                values,
                self.dynamic_lookup_layer.vocabulary_size,
            )
            # remap and update the embedding matrix
            embedding_matrix = self.embedding_layer.embeddings
            oov_token = tf.fill([self.num_oov_indices], "UNK")
            updated_new_vocab = tf.concat([new_vocab, oov_token], axis=0)
            embedding_matrix = warmstart_embedding_matrix(
                base_vocabulary=list(old_vocab.numpy()),
                new_vocabulary=updated_new_vocab,
                base_embeddings=embedding_matrix,
                new_embeddings_initializer=self.embedding_initializer,
            )
            self.dynamic_lookup_layer.vocabulary.assign(new_vocab)
            self.embedding_layer.embeddings.assign(embedding_matrix)
        except AttributeError:
            logging.info(
                "Time interval specified by the UpdateEmbeddingCallback may be"
                " too small, please try increasing the value of `interval`."
            )

    def aggregate_lookup_table(self, keys_list, values_list):
        # Flatten the keys and values matrices
        keys_1d = tf.reshape(keys_list, [-1])
        values_1d = tf.reshape(values_list, [-1])
        # Get unique keys and their corresponding summed values
        unique_keys, idx, _ = tf.unique_with_counts(keys_1d)
        summed_values = tf.math.unsorted_segment_sum(
            values_1d, idx, tf.shape(unique_keys)[0]
        )
        return unique_keys, summed_values

    def get_top_vocabulary(self, keys, values, k):
        """Get Top vocabulary keys and values."""
        values_len = tf.shape(keys)[0]
        if values_len > k:
            _, indices = tf.nn.top_k(values, k=k)
        else:
            _, indices = tf.nn.top_k(values, k=values_len)
        top_k_vocab = tf.gather(keys, indices)
        return top_k_vocab

