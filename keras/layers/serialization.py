# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Layer serialization/deserialization functions."""

import threading

import tensorflow.compat.v2 as tf

from keras.engine import base_layer
from keras.engine import input_layer
from keras.engine import input_spec
from keras.layers import activation
from keras.layers import attention
from keras.layers import convolutional
from keras.layers import core
from keras.layers import locally_connected
from keras.layers import merging
from keras.layers import pooling
from keras.layers import regularization
from keras.layers import reshaping
from keras.layers import rnn
from keras.layers.normalization import batch_normalization
from keras.layers.normalization import batch_normalization_v1
from keras.layers.normalization import group_normalization
from keras.layers.normalization import layer_normalization
from keras.layers.normalization import unit_normalization
from keras.layers.preprocessing import category_encoding
from keras.layers.preprocessing import discretization
from keras.layers.preprocessing import hashed_crossing
from keras.layers.preprocessing import hashing
from keras.layers.preprocessing import image_preprocessing
from keras.layers.preprocessing import integer_lookup
from keras.layers.preprocessing import (
    normalization as preprocessing_normalization,
)
from keras.layers.preprocessing import string_lookup
from keras.layers.preprocessing import text_vectorization
from keras.layers.rnn import cell_wrappers
from keras.layers.rnn import gru
from keras.layers.rnn import lstm
from keras.saving.legacy import serialization
from keras.saving.legacy.saved_model import json_utils
from keras.utils import generic_utils
from keras.utils import tf_inspect as inspect

# isort: off
from tensorflow.python.util.tf_export import keras_export

ALL_MODULES = (
    base_layer,
    input_layer,
    activation,
    attention,
    convolutional,
    core,
    locally_connected,
    merging,
    batch_normalization_v1,
    group_normalization,
    layer_normalization,
    unit_normalization,
    pooling,
    image_preprocessing,
    regularization,
    reshaping,
    rnn,
    hashing,
    hashed_crossing,
    category_encoding,
    discretization,
    integer_lookup,
    preprocessing_normalization,
    string_lookup,
    text_vectorization,
)
ALL_V2_MODULES = (
    batch_normalization,
    layer_normalization,
    cell_wrappers,
    gru,
    lstm,
)
# ALL_OBJECTS is meant to be a global mutable. Hence we need to make it
# thread-local to avoid concurrent mutations.
LOCAL = threading.local()


def populate_deserializable_objects():
    """Populates dict ALL_OBJECTS with every built-in layer."""
    global LOCAL
    if not hasattr(LOCAL, "ALL_OBJECTS"):
        LOCAL.ALL_OBJECTS = {}
        LOCAL.GENERATED_WITH_V2 = None

    if (
        LOCAL.ALL_OBJECTS
        and LOCAL.GENERATED_WITH_V2 == tf.__internal__.tf2.enabled()
    ):
        # Objects dict is already generated for the proper TF version:
        # do nothing.
        return

    LOCAL.ALL_OBJECTS = {}
    LOCAL.GENERATED_WITH_V2 = tf.__internal__.tf2.enabled()

    base_cls = base_layer.Layer
    generic_utils.populate_dict_with_module_objects(
        LOCAL.ALL_OBJECTS,
        ALL_MODULES,
        obj_filter=lambda x: inspect.isclass(x) and issubclass(x, base_cls),
    )

    # Overwrite certain V1 objects with V2 versions
    if tf.__internal__.tf2.enabled():
        generic_utils.populate_dict_with_module_objects(
            LOCAL.ALL_OBJECTS,
            ALL_V2_MODULES,
            obj_filter=lambda x: inspect.isclass(x) and issubclass(x, base_cls),
        )

    # These deserialization aliases are added for backward compatibility,
    # as in TF 1.13, "BatchNormalizationV1" and "BatchNormalizationV2"
    # were used as class name for v1 and v2 version of BatchNormalization,
    # respectively. Here we explicitly convert them to their canonical names.
    LOCAL.ALL_OBJECTS[
        "BatchNormalizationV1"
    ] = batch_normalization_v1.BatchNormalization
    LOCAL.ALL_OBJECTS[
        "BatchNormalizationV2"
    ] = batch_normalization.BatchNormalization

    # Prevent circular dependencies.
    from keras import models
    from keras.feature_column.sequence_feature_column import (
        SequenceFeatures,
    )
    from keras.premade_models.linear import (
        LinearModel,
    )
    from keras.premade_models.wide_deep import (
        WideDeepModel,
    )

    LOCAL.ALL_OBJECTS["Input"] = input_layer.Input
    LOCAL.ALL_OBJECTS["InputSpec"] = input_spec.InputSpec
    LOCAL.ALL_OBJECTS["Functional"] = models.Functional
    LOCAL.ALL_OBJECTS["Model"] = models.Model
    LOCAL.ALL_OBJECTS["SequenceFeatures"] = SequenceFeatures
    LOCAL.ALL_OBJECTS["Sequential"] = models.Sequential
    LOCAL.ALL_OBJECTS["LinearModel"] = LinearModel
    LOCAL.ALL_OBJECTS["WideDeepModel"] = WideDeepModel

    if tf.__internal__.tf2.enabled():
        from keras.feature_column.dense_features_v2 import (
            DenseFeatures,
        )

        LOCAL.ALL_OBJECTS["DenseFeatures"] = DenseFeatures
    else:
        from keras.feature_column.dense_features import (
            DenseFeatures,
        )

        LOCAL.ALL_OBJECTS["DenseFeatures"] = DenseFeatures

    # Merging layers, function versions.
    LOCAL.ALL_OBJECTS["add"] = merging.add
    LOCAL.ALL_OBJECTS["subtract"] = merging.subtract
    LOCAL.ALL_OBJECTS["multiply"] = merging.multiply
    LOCAL.ALL_OBJECTS["average"] = merging.average
    LOCAL.ALL_OBJECTS["maximum"] = merging.maximum
    LOCAL.ALL_OBJECTS["minimum"] = merging.minimum
    LOCAL.ALL_OBJECTS["concatenate"] = merging.concatenate
    LOCAL.ALL_OBJECTS["dot"] = merging.dot


@keras_export("keras.layers.serialize")
def serialize(layer):
    """Serializes a `Layer` object into a JSON-compatible representation.

    Args:
      layer: The `Layer` object to serialize.

    Returns:
      A JSON-serializable dict representing the object's config.

    Example:

    ```python
    from pprint import pprint
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))

    pprint(tf.keras.layers.serialize(model))
    # prints the configuration of the model, as a dict.
    """
    return serialization.serialize_keras_object(layer)


@keras_export("keras.layers.deserialize")
def deserialize(config, custom_objects=None):
    """Instantiates a layer from a config dictionary.

    Args:
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names) of custom
          (non-Keras) objects to class/functions

    Returns:
        Layer instance (may be Model, Sequential, Network, Layer...)

    Example:

    ```python
    # Configuration of Dense(32, activation='relu')
    config = {
      'class_name': 'Dense',
      'config': {
        'activation': 'relu',
        'activity_regularizer': None,
        'bias_constraint': None,
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'bias_regularizer': None,
        'dtype': 'float32',
        'kernel_constraint': None,
        'kernel_initializer': {'class_name': 'GlorotUniform',
                               'config': {'seed': None}},
        'kernel_regularizer': None,
        'name': 'dense',
        'trainable': True,
        'units': 32,
        'use_bias': True
      }
    }
    dense_layer = tf.keras.layers.deserialize(config)
    ```
    """
    populate_deserializable_objects()
    return serialization.deserialize_keras_object(
        config,
        module_objects=LOCAL.ALL_OBJECTS,
        custom_objects=custom_objects,
        printable_module_name="layer",
    )


def get_builtin_layer(class_name):
    """Returns class if `class_name` is registered, else returns None."""
    if not hasattr(LOCAL, "ALL_OBJECTS"):
        populate_deserializable_objects()
    return LOCAL.ALL_OBJECTS.get(class_name)


def deserialize_from_json(json_string, custom_objects=None):
    """Instantiates a layer from a JSON string."""
    populate_deserializable_objects()
    config = json_utils.decode_and_deserialize(
        json_string,
        module_objects=LOCAL.ALL_OBJECTS,
        custom_objects=custom_objects,
    )
    return deserialize(config, custom_objects)
