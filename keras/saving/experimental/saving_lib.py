# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Keras python-based idempotent saving functions (experimental)."""
import importlib
import json
import os
import types
from keras.saving.saved_model import json_utils
from keras.utils import generic_utils
import tensorflow.compat.v2 as tf
from tensorflow.python.util import tf_export

_CONFIG_FILE = "config.keras"

# A temporary flag to enable the new idempotent saving framework.
_ENABLED = False


def load(dirpath):
    """Load a saved python model."""
    file_path = os.path.join(dirpath, _CONFIG_FILE)
    with tf.io.gfile.GFile(file_path, "r") as f:
        config_json = f.read()
    config_dict = json_utils.decode(config_json)
    return deserialize_keras_object(config_dict)


def save(model, dirpath):
    """Save a saved python model."""
    if not tf.io.gfile.exists(dirpath):
        tf.io.gfile.mkdir(dirpath)
    file_path = os.path.join(dirpath, _CONFIG_FILE)

    # TODO(rchao): Save the model's metadata (e.g. Keras version) in a separate
    # file in the archive.
    # TODO(rchao): Save the model's state (e.g. layer weights/vocab) in a separate
    # set of files in the archive.
    # TODO(rchao): Write the config into a file in an archive. In this prototype
    # we're temporarily settled on a standalone json file.
    serialized_model_dict = serialize_keras_object(model)
    config_json = json.dumps(serialized_model_dict, cls=json_utils.Encoder)
    with tf.io.gfile.GFile(file_path, "w") as f:
        f.write(config_json)


# TODO(rchao): Replace the current Keras' `deserialize_keras_object` with this
# (as well as the reciprocal function).
def deserialize_keras_object(config_dict):
    """Retrieve the object by deserializing the config dict.

    The config dict is a python dictionary that consists of a set of key-value
    pairs, and represents a Keras object, such as an `Optimizer`, `Layer`,
    `Metrics`, etc. The saving and loading library uses the following keys to
    record information of a Keras object:

    - `class_name`: String. For classes that have an exported Keras namespace,
      this is the full path that starts with "keras", such as
      "keras.optimizers.Adam". For classes that do not have an exported Keras
      namespace, this is the name of the class, as exactly defined in the source
      code, such as "LossesContainer".
    - `config`: Dict. Library-defined or user-defined key-value pairs that store
      the configuration of the object, as obtained by `object.get_config()`.
    - `module`: String. The path of the python module, such as
      "keras.engine.compile_utils". Built-in Keras classes
      expect to have prefix `keras`. For classes that have an exported Keras
      namespace, this is `None` since the class can be fully identified by the
      full Keras path.
    - `registered_name`: String. The key the class is registered under via
      `keras.utils.register_keras_serializable(package, name)` API. The key has
      the format of '{package}>{name}', where `package` and `name` are the
      arguments passed to `register_keras_serializable()`. If `name` is not
      provided, it defaults to the class name. If `registered_name` successfully
      resolves to a class (that was registered), `class_name` and `config` values
      in the dict will not be used. `registered_name` is only used for
      non-built-in classes.

    For example, the following dictionary represents the built-in Adam optimizer
    with the relevant config. Note that for built-in (exported symbols that have
    an exported Keras namespace) classes, the library tracks the class by the
    the import location of the built-in object in the Keras namespace, e.g.
    `"keras.optimizers.Adam"`, and this information is stored in `class_name`:

    ```
    dict_structure = {
        "class_name": "keras.optimizers.Adam",
        "config": {
            "amsgrad": false,
            "beta_1": 0.8999999761581421,
            "beta_2": 0.9990000128746033,
            "decay": 0.0,
            "epsilon": 1e-07,
            "learning_rate": 0.0010000000474974513,
            "name": "Adam"
        },
        "module": null,
        "registered_name": "Adam"
    }
    # Returns an `Adam` instance identical to the original one.
    deserialize_keras_object(dict_structure)
    ```

    If the class does not have an exported Keras namespace, the library tracks it
    by its `module` and `class_name`. For example:

    ```
    dict_structure = {
      "class_name": "LossesContainer",
      "config": {
          "losses": [...],
          "total_loss_mean": {...},
      },
      "module": "keras.engine.compile_utils",
      "registered_name": "LossesContainer"
    }

    # Returns a `LossesContainer` instance identical to the original one.
    deserialize_keras_object(dict_structure)
    ```

    And the following dictionary represents a user-customized `MeanSquaredError`
    loss:

    ```
    @keras.utils.generic_utils.register_keras_serializable(package='my_package')
    class ModifiedMeanSquaredError(keras.losses.MeanSquaredError):
      ...

    dict_structure = {
        "class_name": "ModifiedMeanSquaredError",
        "config": {
            "fn": "mean_squared_error",
            "name": "mean_squared_error",
            "reduction": "auto"
        },
        "registered_name": "my_package>ModifiedMeanSquaredError"
    }
    # Gives `ModifiedMeanSquaredError` object
    deserialize_keras_object(dict_structure)
    ```

    Args:
      config_dict: the python dict structure to deserialize the Keras object from.

    Returns:
      The Keras object that is deserialized from `config_dict`.

    """
    # TODO(rchao): Design a 'version' key for `config_dict` for defining versions
    # for classes.
    class_name = config_dict["class_name"]
    config = config_dict["config"]
    module = config_dict["module"]
    registered_name = config_dict["registered_name"]

    # Strings and functions will have `builtins` as its module.
    if module == "builtins":
        if class_name == "str":
            if not isinstance(config, str):
                raise TypeError(
                    "Config of string is supposed to be a string. "
                    f"Received: {config}."
                )
            return config

        elif class_name == "function":
            custom_function = generic_utils.get_custom_objects_by_name(
                registered_name
            )
            if custom_function is not None:
                # If there is a custom function registered (via
                # `register_keras_serializable` API), that takes precedence.
                return custom_function

            # Otherwise, attempt to import the tracked module, and find the function.
            function_module = config.get("module", None)
            try:
                function_module = importlib.import_module(function_module)
            except ImportError as e:
                raise ImportError(
                    f"The function module {function_module} is not available. The "
                    f"config dictionary provided is {config_dict}."
                ) from e
            return vars(function_module).get(config["function_name"])

        raise TypeError(f"Unrecognized type: {class_name}")

    custom_class = generic_utils.get_custom_objects_by_name(registered_name)
    if custom_class is not None:
        # For others (classes), see if there is a custom class registered (via
        # `register_keras_serializable` API). If so, that takes precedence.
        return custom_class.from_config(config)
    else:
        # Otherwise, attempt to retrieve the class object given the `module`, and
        # `class_name`.
        if module is None:
            # In the case where `module` is not recorded, the `class_name` represents
            # the full exported Keras namespace (used by `keras_export`) such as
            # "keras.optimizers.Adam".
            cls = tf_export.get_symbol_from_name(class_name)
        else:
            # In the case where `module` is available, the class does not have an
            # Keras namespace (which is the case when the symbol is not exported via
            # `keras_export`). Import the tracked module (that is used for the
            # internal path), find the class, and use its config.
            mod = importlib.import_module(module)
            cls = vars(mod).get(class_name, None)
        if not hasattr(cls, "from_config"):
            raise TypeError(f"Unable to reconstruct an instance of {cls}.")
        return cls.from_config(config)


def serialize_keras_object(obj):
    """Retrieve the config dict by serializing the Keras object.

    `serialize_keras_object()` serializes a Keras object to a python dictionary
    that represents the object, and is a reciprocal function of
    `deserialize_keras_object()`. See `deserialize_keras_object()` for more
    information about the config format.

    Args:
      obj: the Keras object to serialize.

    Returns:
      A python dict that represents the object. The python dict can be
      deserialized via `deserialize_keras_object()`.
    """

    # Note that in the case of the `obj` being a function, the module used will be
    # "builtins", and the `class_name` used will be "function"; in the case of the
    # `obj` being a string, the module used will be "builtins", and the
    # `class_name` used will be "str"
    module = None

    # This gets the `keras.*` exported name, such as "keras.optimizers.Adam".
    class_name = tf_export.get_canonical_name_for_symbol(
        obj.__class__, api_name="keras"
    )
    if class_name is None:
        module = obj.__class__.__module__
        class_name = obj.__class__.__name__
    return {
        "module": module,
        "class_name": class_name,
        "config": _get_object_config(obj),
        "registered_name": _get_object_registered_name(obj),
    }


def _get_object_registered_name(obj):
    if isinstance(obj, types.FunctionType):
        return generic_utils.get_registered_name(obj)
    else:
        return generic_utils.get_registered_name(obj.__class__)


def _get_object_config(obj):
    """Return the object's config depending on string, function, or others."""
    if isinstance(obj, str):
        # Use the content of the string as the config for string.
        return obj
    elif isinstance(obj, types.FunctionType):
        # Keep track of the function's module and name in a dict as the config.
        return {
            "module": obj.__module__,
            "function_name": obj.__name__,
        }
    if not hasattr(obj, "get_config"):
        raise TypeError(f"Unable to recognize the config of {obj}.")
    return obj.get_config()
