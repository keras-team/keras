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
"""Keras initializer serialization / deserialization."""

import threading
import warnings

import tensorflow.compat.v2 as tf

from tf_keras.src.initializers import initializers
from tf_keras.src.initializers import initializers_v1
from tf_keras.src.saving import serialization_lib
from tf_keras.src.saving.legacy import serialization as legacy_serialization
from tf_keras.src.utils import generic_utils
from tf_keras.src.utils import tf_inspect as inspect

# isort: off
from tensorflow.python import tf2
from tensorflow.python.ops import init_ops
from tensorflow.python.util.tf_export import keras_export

# LOCAL.ALL_OBJECTS is meant to be a global mutable. Hence we need to make it
# thread-local to avoid concurrent mutations.
LOCAL = threading.local()


def populate_deserializable_objects():
    """Populates dict ALL_OBJECTS with every built-in initializer."""
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

    # Compatibility aliases (need to exist in both V1 and V2).
    LOCAL.ALL_OBJECTS["ConstantV2"] = initializers.Constant
    LOCAL.ALL_OBJECTS["GlorotNormalV2"] = initializers.GlorotNormal
    LOCAL.ALL_OBJECTS["GlorotUniformV2"] = initializers.GlorotUniform
    LOCAL.ALL_OBJECTS["HeNormalV2"] = initializers.HeNormal
    LOCAL.ALL_OBJECTS["HeUniformV2"] = initializers.HeUniform
    LOCAL.ALL_OBJECTS["IdentityV2"] = initializers.Identity
    LOCAL.ALL_OBJECTS["LecunNormalV2"] = initializers.LecunNormal
    LOCAL.ALL_OBJECTS["LecunUniformV2"] = initializers.LecunUniform
    LOCAL.ALL_OBJECTS["OnesV2"] = initializers.Ones
    LOCAL.ALL_OBJECTS["OrthogonalV2"] = initializers.Orthogonal
    LOCAL.ALL_OBJECTS["RandomNormalV2"] = initializers.RandomNormal
    LOCAL.ALL_OBJECTS["RandomUniformV2"] = initializers.RandomUniform
    LOCAL.ALL_OBJECTS["TruncatedNormalV2"] = initializers.TruncatedNormal
    LOCAL.ALL_OBJECTS["VarianceScalingV2"] = initializers.VarianceScaling
    LOCAL.ALL_OBJECTS["ZerosV2"] = initializers.Zeros

    # Out of an abundance of caution we also include these aliases that have
    # a non-zero probability of having been included in saved configs in the
    # past.
    LOCAL.ALL_OBJECTS["glorot_normalV2"] = initializers.GlorotNormal
    LOCAL.ALL_OBJECTS["glorot_uniformV2"] = initializers.GlorotUniform
    LOCAL.ALL_OBJECTS["he_normalV2"] = initializers.HeNormal
    LOCAL.ALL_OBJECTS["he_uniformV2"] = initializers.HeUniform
    LOCAL.ALL_OBJECTS["lecun_normalV2"] = initializers.LecunNormal
    LOCAL.ALL_OBJECTS["lecun_uniformV2"] = initializers.LecunUniform

    if tf.__internal__.tf2.enabled():
        # For V2, entries are generated automatically based on the content of
        # initializers.py.
        v2_objs = {}
        base_cls = initializers.Initializer
        generic_utils.populate_dict_with_module_objects(
            v2_objs,
            [initializers],
            obj_filter=lambda x: inspect.isclass(x) and issubclass(x, base_cls),
        )
        for key, value in v2_objs.items():
            LOCAL.ALL_OBJECTS[key] = value
            # Functional aliases.
            LOCAL.ALL_OBJECTS[generic_utils.to_snake_case(key)] = value
    else:
        # V1 initializers.
        v1_objs = {
            "Constant": tf.compat.v1.constant_initializer,
            "GlorotNormal": tf.compat.v1.glorot_normal_initializer,
            "GlorotUniform": tf.compat.v1.glorot_uniform_initializer,
            "Identity": tf.compat.v1.initializers.identity,
            "Ones": tf.compat.v1.ones_initializer,
            "Orthogonal": tf.compat.v1.orthogonal_initializer,
            "VarianceScaling": tf.compat.v1.variance_scaling_initializer,
            "Zeros": tf.compat.v1.zeros_initializer,
            "HeNormal": initializers_v1.HeNormal,
            "HeUniform": initializers_v1.HeUniform,
            "LecunNormal": initializers_v1.LecunNormal,
            "LecunUniform": initializers_v1.LecunUniform,
            "RandomNormal": initializers_v1.RandomNormal,
            "RandomUniform": initializers_v1.RandomUniform,
            "TruncatedNormal": initializers_v1.TruncatedNormal,
        }
        for key, value in v1_objs.items():
            LOCAL.ALL_OBJECTS[key] = value
            # Functional aliases.
            LOCAL.ALL_OBJECTS[generic_utils.to_snake_case(key)] = value

    # More compatibility aliases.
    LOCAL.ALL_OBJECTS["normal"] = LOCAL.ALL_OBJECTS["random_normal"]
    LOCAL.ALL_OBJECTS["uniform"] = LOCAL.ALL_OBJECTS["random_uniform"]
    LOCAL.ALL_OBJECTS["one"] = LOCAL.ALL_OBJECTS["ones"]
    LOCAL.ALL_OBJECTS["zero"] = LOCAL.ALL_OBJECTS["zeros"]


# For backwards compatibility, we populate this file with the objects
# from ALL_OBJECTS. We make no guarantees as to whether these objects will
# using their correct version.
populate_deserializable_objects()
globals().update(LOCAL.ALL_OBJECTS)

# Utility functions


@keras_export("keras.initializers.serialize")
def serialize(initializer, use_legacy_format=False):
    populate_deserializable_objects()
    if initializer is None:
        return None
    if not isinstance(initializer, tuple(LOCAL.ALL_OBJECTS.values())):
        warnings.warn(
            "The `keras.initializers.serialize()` API should only be used for "
            "objects of type `keras.initializers.Initializer`. Found an "
            f"instance of type {type(initializer)}, which may lead to improper "
            "serialization."
        )
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(initializer)

    return serialization_lib.serialize_keras_object(initializer)


@keras_export("keras.initializers.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False):
    """Return an `Initializer` object from its config."""
    populate_deserializable_objects()
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=LOCAL.ALL_OBJECTS,
            custom_objects=custom_objects,
            printable_module_name="initializer",
        )

    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=LOCAL.ALL_OBJECTS,
        custom_objects=custom_objects,
        printable_module_name="initializer",
    )


@keras_export("keras.initializers.get")
def get(identifier):
    """Retrieve a TF-Keras initializer by the identifier.

    The `identifier` may be the string name of a initializers function or class
    (case-sensitively).

    >>> identifier = 'Ones'
    >>> tf.keras.initializers.deserialize(identifier)
    <...keras.initializers.initializers.Ones...>

    You can also specify `config` of the initializer to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note that
    the `class_name` must map to a `Initializer` class.

    >>> cfg = {'class_name': 'Ones', 'config': {}}
    >>> tf.keras.initializers.deserialize(cfg)
    <...keras.initializers.initializers.Ones...>

    In the case that the `identifier` is a class, this method will return a new
    instance of the class by its constructor.

    Args:
      identifier: String or dict that contains the initializer name or
        configurations.

    Returns:
      Initializer instance base on the input identifier.

    Raises:
      ValueError: If the input identifier is not a supported type or in a bad
        format.
    """

    if identifier is None:
        return None
    if isinstance(identifier, dict):
        use_legacy_format = "module" not in identifier
        return deserialize(identifier, use_legacy_format=use_legacy_format)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return get(config)
    elif callable(identifier):
        if inspect.isclass(identifier):
            identifier = identifier()
        return identifier
    else:
        raise ValueError(
            "Could not interpret initializer identifier: " + str(identifier)
        )

