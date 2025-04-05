# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Functions that save the model's config into different formats."""

# isort: off

import threading
from tensorflow.python.util.tf_export import keras_export
from tf_keras.src.saving.legacy import serialization

MODULE_OBJECTS = threading.local()


@keras_export("keras.models.model_from_config")
def model_from_config(config, custom_objects=None):
    """Instantiates a TF-Keras model from its config.

    Usage:
    ```
    # for a Functional API model
    tf.keras.Model().from_config(model.get_config())

    # for a Sequential model
    tf.keras.Sequential().from_config(model.get_config())
    ```

    Args:
        config: Configuration dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A TF-Keras model instance (uncompiled).

    Raises:
        TypeError: if `config` is not a dictionary.
    """
    if isinstance(config, list):
        raise TypeError(
            "`model_from_config` expects a dictionary, not a list. "
            f"Received: config={config}. Did you meant to use "
            "`Sequential.from_config(config)`?"
        )
    from tf_keras.src import layers

    global MODULE_OBJECTS

    if not hasattr(MODULE_OBJECTS, "ALL_OBJECTS"):
        layers.serialization.populate_deserializable_objects()
        MODULE_OBJECTS.ALL_OBJECTS = layers.serialization.LOCAL.ALL_OBJECTS

    return serialization.deserialize_keras_object(
        config,
        module_objects=MODULE_OBJECTS.ALL_OBJECTS,
        custom_objects=custom_objects,
        printable_module_name="layer",
    )


@keras_export("keras.models.model_from_yaml")
def model_from_yaml(yaml_string, custom_objects=None):
    """Parses a yaml model configuration file and returns a model instance.

    Note: Since TF 2.6, this method is no longer supported and will raise a
    RuntimeError.

    Args:
        yaml_string: YAML string or open file encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A TF-Keras model instance (uncompiled).

    Raises:
        RuntimeError: announces that the method poses a security risk
    """
    raise RuntimeError(
        "Method `model_from_yaml()` has been removed due to security risk of "
        "arbitrary code execution. Please use `Model.to_json()` and "
        "`model_from_json()` instead."
    )


@keras_export("keras.models.model_from_json")
def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration string and returns a model instance.

    Usage:

    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Dense(5, input_shape=(3,)),
    ...     tf.keras.layers.Softmax()])
    >>> config = model.to_json()
    >>> loaded_model = tf.keras.models.model_from_json(config)

    Args:
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A TF-Keras model instance (uncompiled).
    """
    from tf_keras.src.layers import (
        deserialize_from_json,
    )

    return deserialize_from_json(json_string, custom_objects=custom_objects)

