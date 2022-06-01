# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras string lookup preprocessing layer."""


import numpy as np
import tensorflow.compat.v2 as tf

from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import index_lookup

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.layers.IntegerLookup",
    "keras.layers.experimental.preprocessing.IntegerLookup",
    v1=[],
)
class IntegerLookup(index_lookup.IndexLookup):
    """A preprocessing layer which maps integer features to contiguous ranges.

    This layer maps a set of arbitrary integer input tokens into indexed integer
    output via a table-based vocabulary lookup. The layer's output indices will
    be contiguously arranged up to the maximum vocab size, even if the input
    tokens are non-continguous or unbounded. The layer supports multiple options
    for encoding the output via `output_mode`, and has optional support for
    out-of-vocabulary (OOV) tokens and masking.

    The vocabulary for the layer must be either supplied on construction or
    learned via `adapt()`. During `adapt()`, the layer will analyze a data set,
    determine the frequency of individual integer tokens, and create a
    vocabulary from them. If the vocabulary is capped in size, the most frequent
    tokens will be used to create the vocabulary and all others will be treated
    as OOV.

    There are two possible output modes for the layer.  When `output_mode` is
    `"int"`, input integers are converted to their index in the vocabulary (an
    integer).  When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`,
    input integers are encoded into an array where each dimension corresponds to
    an element in the vocabulary.

    The vocabulary can optionally contain a mask token as well as an OOV token
    (which can optionally occupy multiple indices in the vocabulary, as set
    by `num_oov_indices`).
    The position of these tokens in the vocabulary is fixed. When `output_mode`
    is `"int"`, the vocabulary will begin with the mask token at index 0,
    followed by OOV indices, followed by the rest of the vocabulary. When
    `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"` the vocabulary will
    begin with OOV indices and instances of the mask token will be dropped.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
      max_tokens: Maximum size of the vocabulary for this layer. This should
        only be specified when adapting the vocabulary or when setting
        `pad_to_max_tokens=True`. If None, there is no cap on the size of the
        vocabulary. Note that this size includes the OOV and mask tokens.
        Defaults to None.
      num_oov_indices: The number of out-of-vocabulary tokens to use. If this
        value is more than 1, OOV inputs are modulated to determine their OOV
        value. If this value is 0, OOV inputs will cause an error when calling
        the layer. Defaults to 1.
      mask_token: An integer token that represents masked inputs. When
        `output_mode` is `"int"`, the token is included in vocabulary and mapped
        to index 0. In other output modes, the token will not appear in the
        vocabulary and instances of the mask token in the input will be dropped.
        If set to None, no mask term will be added. Defaults to None.
      oov_token: Only used when `invert` is True. The token to return for OOV
        indices. Defaults to -1.
      vocabulary: Optional. Either an array of integers or a string path to a
        text file. If passing an array, can pass a tuple, list, 1D numpy array,
        or 1D tensor containing the integer vocbulary terms. If passing a file
        path, the file should contain one line per term in the vocabulary. If
        this argument is set, there is no need to `adapt()` the layer.
      vocabulary_dtype: The dtype of the vocabulary terms, for example
        `"int64"` or `"int32"`. Defaults to `"int64"`.
      idf_weights: Only valid when `output_mode` is `"tf_idf"`. A tuple, list,
        1D numpy array, or 1D tensor or the same length as the vocabulary,
        containing the floating point inverse document frequency weights, which
        will be multiplied by per sample term counts for the final `tf_idf`
        weight. If the `vocabulary` argument is set, and `output_mode` is
        `"tf_idf"`, this argument must be supplied.
      invert: Only valid when `output_mode` is `"int"`. If True, this layer will
        map indices to vocabulary items instead of mapping vocabulary items to
        indices. Default to False.
      output_mode: Specification for the output of the layer. Defaults to
        `"int"`.  Values can be `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`,
        or `"tf_idf"` configuring the layer as follows:
          - `"int"`: Return the vocabulary indices of the input tokens.
          - `"one_hot"`: Encodes each individual element in the input into an
            array the same size as the vocabulary, containing a 1 at the element
            index. If the last dimension is size 1, will encode on that
            dimension.  If the last dimension is not size 1, will append a new
            dimension for the encoded output.
          - `"multi_hot"`: Encodes each sample in the input into a single array
            the same size as the vocabulary, containing a 1 for each vocabulary
            term present in the sample. Treats the last dimension as the sample
            dimension, if input shape is (..., sample_length), output shape will
            be (..., num_tokens).
          - `"count"`: As `"multi_hot"`, but the int array contains a count of
            the number of times the token at that index appeared in the sample.
          - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is applied to
            find the value in each token slot.
        For `"int"` output, any shape of input and output is supported. For all
        other output modes, currently only output up to rank 2 is supported.
      pad_to_max_tokens: Only applicable when `output_mode` is `"multi_hot"`,
        `"count"`, or `"tf_idf"`. If True, the output will have its feature axis
        padded to `max_tokens` even if the number of unique tokens in the
        vocabulary is less than max_tokens, resulting in a tensor of shape
        [batch_size, max_tokens] regardless of vocabulary size. Defaults to
        False.
      sparse: Boolean. Only applicable when `output_mode` is `"multi_hot"`,
        `"count"`, or `"tf_idf"`. If True, returns a `SparseTensor` instead of a
        dense `Tensor`. Defaults to False.

    Examples:

    **Creating a lookup layer with a known vocabulary**

    This example creates a lookup layer with a pre-existing vocabulary.

    >>> vocab = [12, 36, 1138, 42]
    >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])  # Note OOV tokens
    >>> layer = tf.keras.layers.IntegerLookup(vocabulary=vocab)
    >>> layer(data)
    <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
    array([[1, 3, 4],
           [4, 0, 2]])>

    **Creating a lookup layer with an adapted vocabulary**

    This example creates a lookup layer and generates the vocabulary by
    analyzing the dataset.

    >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])
    >>> layer = tf.keras.layers.IntegerLookup()
    >>> layer.adapt(data)
    >>> layer.get_vocabulary()
    [-1, 42, 1138, 1000, 36, 12]

    Note that the OOV token -1 have been added to the vocabulary. The remaining
    tokens are sorted by frequency (42, which has 2 occurrences, is first) then
    by inverse sort order.

    >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])
    >>> layer = tf.keras.layers.IntegerLookup()
    >>> layer.adapt(data)
    >>> layer(data)
    <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
    array([[5, 2, 1],
           [1, 3, 4]])>


    **Lookups with multiple OOV indices**

    This example demonstrates how to use a lookup layer with multiple OOV
    indices.  When a layer is created with more than one OOV index, any OOV
    tokens are hashed into the number of OOV buckets, distributing OOV tokens in
    a deterministic fashion across the set.

    >>> vocab = [12, 36, 1138, 42]
    >>> data = tf.constant([[12, 1138, 42], [37, 1000, 36]])
    >>> layer = tf.keras.layers.IntegerLookup(
    ...     vocabulary=vocab, num_oov_indices=2)
    >>> layer(data)
    <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
    array([[2, 4, 5],
           [1, 0, 3]])>

    Note that the output for OOV token 37 is 1, while the output for OOV token
    1000 is 0. The in-vocab terms have their output index increased by 1 from
    earlier examples (12 maps to 2, etc) in order to make space for the extra
    OOV token.

    **One-hot output**

    Configure the layer with `output_mode='one_hot'`. Note that the first
    `num_oov_indices` dimensions in the ont_hot encoding represent OOV values.

    >>> vocab = [12, 36, 1138, 42]
    >>> data = tf.constant([12, 36, 1138, 42, 7]) # Note OOV tokens
    >>> layer = tf.keras.layers.IntegerLookup(
    ...     vocabulary=vocab, output_mode='one_hot')
    >>> layer(data)
    <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
      array([[0., 1., 0., 0., 0.],
             [0., 0., 1., 0., 0.],
             [0., 0., 0., 1., 0.],
             [0., 0., 0., 0., 1.],
             [1., 0., 0., 0., 0.]], dtype=float32)>

    **Multi-hot output**

    Configure the layer with `output_mode='multi_hot'`. Note that the first
    `num_oov_indices` dimensions in the multi_hot encoding represent OOV tokens

    >>> vocab = [12, 36, 1138, 42]
    >>> data = tf.constant([[12, 1138, 42, 42],
    ...                     [42, 7, 36, 7]]) # Note OOV tokens
    >>> layer = tf.keras.layers.IntegerLookup(
    ...     vocabulary=vocab, output_mode='multi_hot')
    >>> layer(data)
    <tf.Tensor: shape=(2, 5), dtype=float32, numpy=
      array([[0., 1., 0., 1., 1.],
             [1., 0., 1., 0., 1.]], dtype=float32)>

    **Token count output**

    Configure the layer with `output_mode='count'`. As with multi_hot output,
    the first `num_oov_indices` dimensions in the output represent OOV tokens.

    >>> vocab = [12, 36, 1138, 42]
    >>> data = tf.constant([[12, 1138, 42, 42],
    ...                     [42, 7, 36, 7]]) # Note OOV tokens
    >>> layer = tf.keras.layers.IntegerLookup(
    ...     vocabulary=vocab, output_mode='count')
    >>> layer(data)
    <tf.Tensor: shape=(2, 5), dtype=float32, numpy=
      array([[0., 1., 0., 1., 2.],
             [2., 0., 1., 0., 1.]], dtype=float32)>

    **TF-IDF output**

    Configure the layer with `output_mode='tf_idf'`. As with multi_hot output,
    the first `num_oov_indices` dimensions in the output represent OOV tokens.

    Each token bin will output `token_count * idf_weight`, where the idf weights
    are the inverse document frequency weights per token. These should be
    provided along with the vocabulary. Note that the `idf_weight` for OOV
    tokens will default to the average of all idf weights passed in.

    >>> vocab = [12, 36, 1138, 42]
    >>> idf_weights = [0.25, 0.75, 0.6, 0.4]
    >>> data = tf.constant([[12, 1138, 42, 42],
    ...                     [42, 7, 36, 7]]) # Note OOV tokens
    >>> layer = tf.keras.layers.IntegerLookup(
    ...     output_mode='tf_idf', vocabulary=vocab, idf_weights=idf_weights)
    >>> layer(data)
    <tf.Tensor: shape=(2, 5), dtype=float32, numpy=
      array([[0.  , 0.25, 0.  , 0.6 , 0.8 ],
             [1.0 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)>

    To specify the idf weights for oov tokens, you will need to pass the entire
    vocabularly including the leading oov token.

    >>> vocab = [-1, 12, 36, 1138, 42]
    >>> idf_weights = [0.9, 0.25, 0.75, 0.6, 0.4]
    >>> data = tf.constant([[12, 1138, 42, 42],
    ...                     [42, 7, 36, 7]]) # Note OOV tokens
    >>> layer = tf.keras.layers.IntegerLookup(
    ...     output_mode='tf_idf', vocabulary=vocab, idf_weights=idf_weights)
    >>> layer(data)
    <tf.Tensor: shape=(2, 5), dtype=float32, numpy=
      array([[0.  , 0.25, 0.  , 0.6 , 0.8 ],
             [1.8 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)>

    When adapting the layer in tf_idf mode, each input sample will be considered
    a document, and idf weight per token will be calculated as
    `log(1 + num_documents / (1 + token_document_count))`.

    **Inverse lookup**

    This example demonstrates how to map indices to tokens using this layer.
    (You can also use `adapt()` with `inverse=True`, but for simplicity we'll
    pass the vocab in this example.)

    >>> vocab = [12, 36, 1138, 42]
    >>> data = tf.constant([[1, 3, 4], [4, 0, 2]])
    >>> layer = tf.keras.layers.IntegerLookup(vocabulary=vocab, invert=True)
    >>> layer(data)
    <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
    array([[  12, 1138,   42],
           [  42,   -1,   36]])>

    Note that the first index correspond to the oov token by default.


    **Forward and inverse lookup pairs**

    This example demonstrates how to use the vocabulary of a standard lookup
    layer to create an inverse lookup layer.

    >>> vocab = [12, 36, 1138, 42]
    >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])
    >>> layer = tf.keras.layers.IntegerLookup(vocabulary=vocab)
    >>> i_layer = tf.keras.layers.IntegerLookup(
    ...     vocabulary=layer.get_vocabulary(), invert=True)
    >>> int_data = layer(data)
    >>> i_layer(int_data)
    <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
    array([[  12, 1138,   42],
           [  42,   -1,   36]])>

    In this example, the input token 1000 resulted in an output of -1, since
    1000 was not in the vocabulary - it got represented as an OOV, and all OOV
    tokens are returned as -1 in the inverse layer. Also, note that for the
    inverse to work, you must have already set the forward layer vocabulary
    either directly or via `adapt()` before calling `get_vocabulary()`.
    """

    def __init__(
        self,
        max_tokens=None,
        num_oov_indices=1,
        mask_token=None,
        oov_token=-1,
        vocabulary=None,
        vocabulary_dtype="int64",
        idf_weights=None,
        invert=False,
        output_mode="int",
        sparse=False,
        pad_to_max_tokens=False,
        **kwargs,
    ):
        if not tf.dtypes.as_dtype(vocabulary_dtype).is_integer:
            raise ValueError(
                "`vocabulary_dtype` must be an integer dtype. "
                f"Received: {vocabulary_dtype}"
            )

        # Legacy versions of the IntegerLookup layer set layer dtype to int64,
        # instead of the output type. If we see this and output mode is not
        # "int", clear the setting so we don't switch types for old SavedModels.
        if (
            output_mode != "int"
            and "dtype" in kwargs
            and (kwargs["dtype"] == tf.int64 or kwargs["dtype"] == "int64")
        ):
            del kwargs["dtype"]

        # Support deprecated args for this layer.
        if "max_values" in kwargs:
            logging.log_first_n(
                logging.WARN,
                "max_values is deprecated, use max_tokens instead.",
                1,
            )
            max_tokens = kwargs["max_values"]
            del kwargs["max_values"]
        if "mask_value" in kwargs:
            logging.log_first_n(
                logging.WARN,
                "mask_value is deprecated, use mask_token instead.",
                1,
            )
            mask_token = kwargs["mask_value"]
            del kwargs["mask_value"]
        if "oov_value" in kwargs:
            logging.log_first_n(
                logging.WARN,
                "oov_value is deprecated, use oov_token instead.",
                1,
            )
            oov_token = kwargs["oov_value"]
            del kwargs["oov_value"]

        # If max_tokens is set, the token must be greater than 1 - otherwise we
        # are creating a 0-element vocab, which doesn't make sense.
        if max_tokens is not None and max_tokens <= 1:
            raise ValueError(
                f"If `max_tokens` is set for `IntegerLookup`, it must be "
                f"greater than 1. Received: max_tokens={max_tokens}."
            )

        if num_oov_indices < 0:
            raise ValueError(
                f"The value of `num_oov_indices` argument for `IntegerLookup` "
                f"must >= 0. Received num_oov_indices="
                f"{num_oov_indices}."
            )

        # Make sure mask and oov are of the dtype we want.
        mask_token = None if mask_token is None else np.int64(mask_token)
        oov_token = None if oov_token is None else np.int64(oov_token)

        super().__init__(
            max_tokens=max_tokens,
            num_oov_indices=num_oov_indices,
            mask_token=mask_token,
            oov_token=oov_token,
            vocabulary=vocabulary,
            vocabulary_dtype=vocabulary_dtype,
            idf_weights=idf_weights,
            invert=invert,
            output_mode=output_mode,
            sparse=sparse,
            pad_to_max_tokens=pad_to_max_tokens,
            **kwargs,
        )
        base_preprocessing_layer.keras_kpl_gauge.get_cell("IntegerLookup").set(
            True
        )

    # We override this method solely to generate a docstring.
    def adapt(self, data, batch_size=None, steps=None):
        """Computes a vocabulary of interger terms from tokens in a dataset.

        Calling `adapt()` on an `IntegerLookup` layer is an alternative to
        passing in a precomputed vocabulary  on construction via the
        `vocabulary` argument.  An `IntegerLookup` layer should always be either
        adapted over a dataset or supplied with a vocabulary.

        During `adapt()`, the layer will build a vocabulary of all integer
        tokens seen in the dataset, sorted by occurrence count, with ties broken
        by sort order of the tokens (high to low). At the end of `adapt()`, if
        `max_tokens` is set, the vocabulary wil be truncated to `max_tokens`
        size. For example, adapting a layer with `max_tokens=1000` will compute
        the 1000 most frequent tokens occurring in the input dataset. If
        `output_mode='tf-idf'`, `adapt()` will also learn the document
        frequencies of each token in the input dataset.

        In order to make `StringLookup` efficient in any distribution context,
        the vocabulary is kept static with respect to any compiled `tf.Graph`s
        that call the layer. As a consequence, if the layer is adapted a second
        time, any models using the layer should be re-compiled. For more
        information see
        `tf.keras.layers.experimental.preprocessing.PreprocessingLayer.adapt`.

        `adapt()` is meant only as a single machine utility to compute layer
        state.  To analyze a dataset that cannot fit on a single machine, see
        [Tensorflow Transform](
        https://www.tensorflow.org/tfx/transform/get_started) for a
        multi-machine, map-reduce solution.

        Arguments:
          data: The data to train on. It can be passed either as a
              `tf.data.Dataset`, or as a numpy array.
          batch_size: Integer or `None`.
              Number of samples per state update.
              If unspecified, `batch_size` will default to 32.
              Do not specify the `batch_size` if your data is in the
              form of datasets, generators, or `keras.utils.Sequence` instances
              (since they generate batches).
          steps: Integer or `None`.
              Total number of steps (batches of samples)
              When training with input tensors such as
              TensorFlow data tensors, the default `None` is equal to
              the number of samples in your dataset divided by
              the batch size, or 1 if that cannot be determined. If x is a
              `tf.data` dataset, and 'steps' is None, the epoch will run until
              the input dataset is exhausted. When passing an infinitely
              repeating dataset, you must specify the `steps` argument. This
              argument is not supported with array inputs.
        """
        super().adapt(data, batch_size=batch_size, steps=steps)
