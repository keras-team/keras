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
"""Keras CategoryEncoding preprocessing layer."""


import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import layer_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

INT = utils.INT
ONE_HOT = utils.ONE_HOT
MULTI_HOT = utils.MULTI_HOT
COUNT = utils.COUNT


@keras_export(
    "keras.layers.CategoryEncoding",
    "keras.layers.experimental.preprocessing.CategoryEncoding",
)
class CategoryEncoding(base_layer.Layer):
    """A preprocessing layer which encodes integer features.

    This layer provides options for condensing data into a categorical encoding
    when the total number of tokens are known in advance. It accepts integer
    values as inputs, and it outputs a dense or sparse representation of those
    inputs. For integer inputs where the total number of tokens is not known,
    use `tf.keras.layers.IntegerLookup` instead.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Examples:

    **One-hot encoding data**

    >>> layer = tf.keras.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="one_hot")
    >>> layer([3, 2, 0, 1])
    <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
      array([[0., 0., 0., 1.],
             [0., 0., 1., 0.],
             [1., 0., 0., 0.],
             [0., 1., 0., 0.]], dtype=float32)>

    **Multi-hot encoding data**

    >>> layer = tf.keras.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="multi_hot")
    >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]])
    <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
      array([[1., 1., 0., 0.],
             [1., 0., 0., 0.],
             [0., 1., 1., 0.],
             [0., 1., 0., 1.]], dtype=float32)>

    **Using weighted inputs in `"count"` mode**

    >>> layer = tf.keras.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="count")
    >>> count_weights = np.array([[.1, .2], [.1, .1], [.2, .3], [.4, .2]])
    >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]], count_weights=count_weights)
    <tf.Tensor: shape=(4, 4), dtype=float64, numpy=
      array([[0.1, 0.2, 0. , 0. ],
             [0.2, 0. , 0. , 0. ],
             [0. , 0.2, 0.3, 0. ],
             [0. , 0.2, 0. , 0.4]], dtype=float32)>

    Args:
      num_tokens: The total number of tokens the layer should support. All
        inputs to the layer must integers in the range `0 <= value <
        num_tokens`, or an error will be thrown.
      output_mode: Specification for the output of the layer.
        Defaults to `"multi_hot"`. Values can be `"one_hot"`, `"multi_hot"` or
        `"count"`, configuring the layer as follows:
          - `"one_hot"`: Encodes each individual element in the input into an
            array of `num_tokens` size, containing a 1 at the element index. If
            the last dimension is size 1, will encode on that dimension. If the
            last dimension is not size 1, will append a new dimension for the
            encoded output.
          - `"multi_hot"`: Encodes each sample in the input into a single array
            of `num_tokens` size, containing a 1 for each vocabulary term
            present in the sample. Treats the last dimension as the sample
            dimension, if input shape is `(..., sample_length)`, output shape
            will be `(..., num_tokens)`.
          - `"count"`: Like `"multi_hot"`, but the int array contains a count of
            the number of times the token at that index appeared in the sample.
        For all output modes, currently only output up to rank 2 is supported.
      sparse: Boolean. If true, returns a `SparseTensor` instead of a dense
        `Tensor`. Defaults to `False`.

    Call arguments:
      inputs: A 1D or 2D tensor of integer inputs.
      count_weights: A tensor in the same shape as `inputs` indicating the
        weight for each sample value when summing up in `count` mode. Not used
        in `"multi_hot"` or `"one_hot"` modes.
    """

    def __init__(
        self, num_tokens=None, output_mode="multi_hot", sparse=False, **kwargs
    ):
        # max_tokens is an old name for the num_tokens arg we continue to
        # support because of usage.
        if "max_tokens" in kwargs:
            logging.warning(
                "max_tokens is deprecated, please use num_tokens instead."
            )
            num_tokens = kwargs["max_tokens"]
            del kwargs["max_tokens"]

        # By default, output floats. This is already default for TF2, but in TF1
        # dtype is inferred from inputs, and would default to int.
        if "dtype" not in kwargs:
            kwargs["dtype"] = backend.floatx()

        super().__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell(
            "CategoryEncoding"
        ).set(True)

        # Support deprecated names for output_modes.
        if output_mode == "binary":
            output_mode = MULTI_HOT
        # 'output_mode' must be one of (COUNT, ONE_HOT, MULTI_HOT)
        layer_utils.validate_string_arg(
            output_mode,
            allowable_strings=(COUNT, ONE_HOT, MULTI_HOT),
            layer_name="CategoryEncoding",
            arg_name="output_mode",
        )

        if num_tokens is None:
            raise ValueError(
                "num_tokens must be set to use this layer. If the "
                "number of tokens is not known beforehand, use the "
                "IntegerLookup layer instead."
            )
        if num_tokens < 1:
            raise ValueError(
                f"`num_tokens` must be >= 1. Received: num_tokens={num_tokens}."
            )

        self.num_tokens = num_tokens
        self.output_mode = output_mode
        self.sparse = sparse

    def compute_output_shape(self, input_shape):
        if not input_shape:
            return tf.TensorShape([self.num_tokens])
        if self.output_mode == ONE_HOT and input_shape[-1] != 1:
            return tf.TensorShape(input_shape + [self.num_tokens])
        else:
            return tf.TensorShape(input_shape[:-1] + [self.num_tokens])

    def compute_output_signature(self, input_spec):
        output_shape = self.compute_output_shape(input_spec.shape.as_list())
        if self.sparse:
            return tf.SparseTensorSpec(shape=output_shape, dtype=tf.int64)
        else:
            return tf.TensorSpec(shape=output_shape, dtype=tf.int64)

    def get_config(self):
        config = {
            "num_tokens": self.num_tokens,
            "output_mode": self.output_mode,
            "sparse": self.sparse,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, count_weights=None):
        inputs = utils.ensure_tensor(inputs)

        if count_weights is not None:
            if self.output_mode != COUNT:
                raise ValueError(
                    "`count_weights` is not used when `output_mode` is not "
                    "`'count'`. Received `count_weights={count_weights}`."
                )
            count_weights = utils.ensure_tensor(
                count_weights, self.compute_dtype
            )

        depth = self.num_tokens
        if isinstance(inputs, tf.SparseTensor):
            max_value = tf.reduce_max(inputs.values)
            min_value = tf.reduce_min(inputs.values)
        else:
            max_value = tf.reduce_max(inputs)
            min_value = tf.reduce_min(inputs)
        condition = tf.logical_and(
            tf.greater(tf.cast(depth, max_value.dtype), max_value),
            tf.greater_equal(min_value, tf.cast(0, min_value.dtype)),
        )
        assertion = tf.Assert(
            condition,
            [
                "Input values must be in the range 0 <= values < num_tokens"
                " with num_tokens={}".format(depth)
            ],
        )
        with tf.control_dependencies([assertion]):
            return utils.encode_categorical_inputs(
                inputs,
                output_mode=self.output_mode,
                depth=depth,
                dtype=self.compute_dtype,
                sparse=self.sparse,
                count_weights=count_weights,
            )
