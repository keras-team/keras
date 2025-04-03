"""Builds a vocabulary from inputs to the layer."""

import random
import string

import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export

from tf_keras.src.layers import Layer


@keras_export("keras.layers.experimental.DynamicLookup")
class DynamicLookup(Layer):
    """A layer that builds a vocabulary from inputs.

    This layer maintains a vocabulary that is continuously updated based on the
    inputs passed in every forward pass. The frequency of the input is tracked
    and used to maintain a vocabulary. The very last index will be treated as
    the index. If `vocabulary_size=10`, OOV index will be 9.

    Args:
      vocabulary_size: Integer value representing size of the vocabulary to
        build.
      initial_vocabulary: The vocabulary to initialize the layer with. If a 1D
        tensor is provided, the vocabulary will be initialized with that tensor.
        If a `tf.DType` object is provided, a random tensor of that dtype and of
        length `vocabulary_size` will be generated as the initial vocabulary.
        Supported `tf.DType` values include `tf.int32`, `tf.int64` and
        `tf.string`.
      eviction_policy: The eviction policy for the vocabulary. Available options
        are string values like "LFU" (Least Frequently Used) and *more to come*.
        If not specified, the default eviction policy is "LFU". Expects a
        string.
      **kwargs: Arguments for super class.

    Attributes: get_vocabulary(): Returns a tensor representing the current
      vocabulary of the layer. If you want to look up the vocabulary keys given
      a set of indices, you can simply use `tf.gather(vocabulary, indices)`.

    Example:
    Here is an example to demonstrate how to use the DynamicLookup layer
    ```
      vocabulary_size = 3
      eviction_policy = "LFU"
      vocab = tf.constant(["apple", "banana", "cherry"])
      layer = DynamicLookup(
          vocabulary_size,
          vocab,
          eviction_policy=eviction_policy,
      )
      inputs = tf.constant([
          ["apple", "banana"],
      ])

      outputs = layer(inputs)
      tf.print(outputs)

      # you get the following output
      [[0 1]]

      # get top k vocab
      top_k_vocab = layer.get_top_vocabulary(2)
      tf.print(top_k_vocab)

      # you get the following output
      ["apple", "banana"]
      ```
      If you want to checkpoint the vocabulary or vocabulary frequency, see
      the following example

      ```
      checkpoint =
      tf.train.Checkpoint(vocabulary=self.vocabulary)
      checkpoint.write(filepath)
      ```
    """

    def __init__(
        self,
        vocabulary_size,
        initial_vocabulary,
        eviction_policy="LFU",
        **kwargs,
    ):
        """Initializes the DynamicLookup layer."""

        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.eviction_policy = eviction_policy
        if tf.is_tensor(initial_vocabulary):
            self.initial_vocabulary = initial_vocabulary
        elif initial_vocabulary in (
            tf.string,
            tf.int32,
            tf.int64,
        ):
            self.initial_vocabulary = (
                DynamicLookup._generate_random_initial_vocab(
                    vocabulary_size, initial_vocabulary
                )
            )
        else:
            raise ValueError(
                "Either specify the initial vocabulary or provide a "
                "valid dtype. The dtype argument must be one of the "
                "following: tf.string, tf.int32, tf.int64."
            )
        # maintain a 20% bigger hash table
        self.internal_table_size = tf.cast(
            tf.floor(
                tf.multiply(
                    tf.cast(self.vocabulary_size, dtype=tf.float32), 1.2
                )
            ),
            dtype=tf.int32,
        )
        self.vocabulary_dtype = self.initial_vocabulary.dtype
        if self.eviction_policy == "LFU":
            self.vocabulary_table_keys = tf.Variable(
                initial_value=pad_tensor(
                    self.initial_vocabulary, self.internal_table_size
                ),
                shape=tf.TensorShape(self.internal_table_size),
                dtype=self.vocabulary_dtype,
                trainable=False,
                name="vocabulary_table_keys",
                per_worker_variable=True,
            )
            self.vocabulary_table_values = tf.Variable(
                initial_value=pad_tensor(
                    tf.zeros_like(self.initial_vocabulary, dtype=tf.int32),
                    self.internal_table_size,
                ),
                shape=tf.TensorShape(self.internal_table_size),
                dtype=tf.int32,
                trainable=False,
                name="vocabulary_table_values",
                per_worker_variable=True,
            )
        else:
            raise ValueError(
                "{} eviction policy is currently unsupported by DynamicLookup"
                " layer."
                " It currently only supports `LFU`".format(self.eviction_policy)
            )

        # TODO(b/268243335): add more eviction policy
        # TODO(b/268243996): provide multiple OOV

    def build(self, input_shape=None):
        self.vocabulary = self.add_weight(
            shape=self.initial_vocabulary.shape,
            dtype=self.vocabulary_dtype,
            initializer=tf.constant_initializer(
                self.initial_vocabulary.numpy()
            ),
            trainable=False,
            name="vocabulary",
        )
        super().build(input_shape)

    def call(self, inputs, learn_vocab=True):
        """Learn vocabulary from inputs and perform vocabulary lookup.

        Args:
          inputs: Input tensor, or dict/list/tuple of input tensors.
          learn_vocab: A boolean value that specifies whether the vocabulary
            should be learned from the layer inputs or not. Defaults to True.

        Returns:
          A tensor or list/tuple of tensors.
        """
        flattened_inputs = tf.reshape(inputs, [-1])
        # get unique values from inputs
        unique, _ = tf.unique(flattened_inputs)
        unique = tf.cast(unique, dtype=self.vocabulary_dtype)
        # learn vocab form inputs
        if learn_vocab and self.eviction_policy == "LFU":
            self.update_internal_vocabulary(unique)

        # lookup for inputs in self.vocabulary
        top_k_vocab = self.vocabulary
        lookup_values = tf.expand_dims(flattened_inputs, axis=-1)
        condition = tf.reduce_any(
            tf.equal(top_k_vocab, tf.expand_dims(lookup_values, -1)), axis=-1
        )
        # the very last index will be the OOV index
        indices = tf.where(
            condition,
            tf.argmax(
                tf.equal(top_k_vocab, tf.expand_dims(lookup_values, -1)),
                axis=-1,
            ),
            self.vocabulary_size,
        )
        # reshape output to the same shape as input
        out = tf.reshape(tf.squeeze(indices), tf.shape(inputs))
        return out

    def update_internal_vocabulary(self, unique):
        # get new keys
        unpadded_keys = remove_padding(self.vocabulary_table_keys)
        unpadded_values = remove_padding(self.vocabulary_table_values)
        table_expanded = tf.expand_dims(unpadded_keys, axis=0)
        unique_expanded = tf.expand_dims(unique, axis=0)
        new_keys = tf.sets.difference(
            unique_expanded, table_expanded, aminusb=True
        )
        number_of_new_keys = tf.shape(new_keys.values)[0]
        # get number of keys to be removed from vocab_frequency
        number_of_keys_to_remove = (
            tf.shape(unpadded_keys)[0]
            - self.internal_table_size
            + number_of_new_keys
        )
        number_of_keys_to_remove = tf.cast(number_of_keys_to_remove, tf.int32)
        number_of_keys_to_remove = tf.maximum(number_of_keys_to_remove, 0)
        # remove old keys
        updated_keys, updated_values = self._remove_old_keys(
            unpadded_keys,
            unpadded_values,
            number_of_keys_to_remove,
        )
        # add new keys
        self._add_new_keys(
            updated_keys,
            updated_values,
            unique,
            new_keys,
        )
        return unique

    def _remove_old_keys(self, unpadded_keys, unpadded_values, n):
        """remove old keys."""
        updated_keys, updated_values = None, None
        if self.eviction_policy == "LFU":
            # LFU eviction
            # negate the values of counts to find the lower n keys to remove
            negative_count = tf.math.negative(unpadded_values)
            # get index of lower n counts
            _, lower_n_index = tf.nn.top_k(negative_count, k=n)
            # gather keys that needs to be removed
            keys_to_remove = tf.gather(unpadded_keys, lower_n_index)
            # get masks for keys not present in inputs
            mask = tf.reduce_all(
                unpadded_keys[:, tf.newaxis] != keys_to_remove, axis=1
            )
            # updated keys and values with least frequent keys removed
            updated_keys = tf.boolean_mask(
                unpadded_keys,
                mask,
            )
            updated_values = tf.boolean_mask(
                unpadded_values,
                mask,
            )
        return updated_keys, updated_values

    def _add_new_keys(self, updated_keys, updated_values, unique, new_keys):
        """Add new keys and update internal vocabulary table."""
        if self.eviction_policy == "LFU":
            # increment values of old keys when present in current inputs
            matches = tf.where(
                tf.equal(tf.expand_dims(updated_keys, axis=1), unique)
            )[:, 0]
            updates = tf.ones_like(matches, dtype=tf.int32)
            matches = tf.expand_dims(matches, axis=-1)
            values_2 = tf.tensor_scatter_nd_add(
                updated_values, matches, updates
            )
            # add new keys and corresponding values = 1
            values_difference = tf.ones_like(new_keys.values, dtype=tf.int32)
            # concatenate old keys and new keys and pad
            updated_keys = pad_tensor(
                tf.concat([updated_keys, new_keys.values], axis=0),
                self.internal_table_size,
            )
            self.vocabulary_table_keys.assign(updated_keys)
            # concatenate updated old values and new values and pad
            updated_values = pad_tensor(
                tf.concat([values_2, values_difference], axis=0),
                self.internal_table_size,
            )
            self.vocabulary_table_values.assign(updated_values)
        return unique

    def get_top_vocabulary(self, k):
        """Get top k vocabulary keys."""
        top_k_vocab = None
        if self.eviction_policy == "LFU":
            values_len = tf.shape(self.vocabulary_table_keys)[0]
            if values_len > k:
                _, indices = tf.nn.top_k(self.vocabulary_table_values, k=k)
            else:
                _, indices = tf.nn.top_k(
                    self.vocabulary_table_values, k=values_len
                )
            top_k_vocab = tf.gather(self.vocabulary_table_keys, indices)
        return top_k_vocab

    def get_vocabulary(self):
        return self.vocabulary

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "initial_vocabulary": self.initial_vocabulary.numpy().tolist(),
                "eviction_policy": self.eviction_policy,
            }
        )
        return config

    def save_assets(self, dir_path):
        vocabulary = self.vocabulary.numpy().tolist()
        vocabulary_filepath = tf.io.gfile.join(dir_path, "vocabulary.txt")
        with open(vocabulary_filepath, "w") as f:
            f.write("\n".join([str(w) for w in vocabulary]))

    def _generate_random_initial_vocab(
        vocabulary_size, dtype
    ):  # pylint: disable=no-self-argument
        if dtype == tf.string:
            chars = string.ascii_letters
            random_strings = [
                "".join([random.choice(chars) for _ in range(10)])
                for _ in range(vocabulary_size)
            ]
            random_vocab = tf.constant(random_strings, dtype=tf.string)
        elif dtype == tf.int32:
            random_vocab = tf.random.uniform(
                shape=[vocabulary_size],
                minval=0,
                maxval=vocabulary_size,
                dtype=tf.int32,
            )
        elif dtype == tf.int64:
            random_vocab = tf.random.uniform(
                shape=[vocabulary_size],
                minval=0,
                maxval=vocabulary_size,
                dtype=tf.int64,
            )
        else:
            raise ValueError(
                "Supported dtype for initial vocabulary include `tf.int32`,"
                " `tf.int64`, or `tf.string`. But got dtype = {}".format(dtype)
            )

        return random_vocab


def pad_tensor(tensor, n):
    """Pad a tensor to a fixed length."""
    if tensor.dtype == tf.string:
        padding = "unk"
    else:
        padding = -1
    pad_length = tf.maximum(n - tf.shape(tensor)[0], 0)
    padded_tensor = tf.pad(tensor, [[0, pad_length]], constant_values=padding)
    return padded_tensor[:n]


def remove_padding(tensor):
    """Remove padding from a tensor."""
    if tensor.dtype == tf.string:
        padding = "unk"
    else:
        padding = -1
    mask = tf.reshape(tensor != padding, shape=[-1])
    return tf.boolean_mask(tensor, mask)

