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
import tempfile
import types
import zipfile

import tensorflow.compat.v2 as tf
from absl import logging

from keras import losses
from keras.engine import base_layer
from keras.optimizers.optimizer_experimental import optimizer
from keras.saving.saved_model import json_utils
from keras.utils import generic_utils
from keras.utils import io_utils

# isort: off
from tensorflow.python.util import tf_export

_ARCHIVE_FILENAME = "archive.keras"
STATE_FILENAME = "states.npz"
_SELF_DIRNAME = "self"
_CONFIG_FILENAME = "config.json"
_STATES_ROOT_DIRNAME = "model"

# A temporary flag to enable the new idempotent saving framework.
_ENABLED = False


def _print_archive(zipfile, action):
    io_utils.print_msg(f"Keras model is being {action} an archive:")
    # Same as `ZipFile.printdir()` except for using Keras' printing utility.
    io_utils.print_msg(
        "%-46s %19s %12s" % ("File Name", "Modified    ", "Size")
    )
    for zinfo in zipfile.filelist:
        date = "%d-%02d-%02d %02d:%02d:%02d" % zinfo.date_time[:6]
        io_utils.print_msg(
            "%-46s %s %12d" % (zinfo.filename, date, zinfo.file_size)
        )


def _is_keras_trackable(object):
    from keras.metrics import base_metric  # To avoid circular import

    return (
        isinstance(object, base_layer.Layer)
        or isinstance(object, optimizer.Optimizer)
        or isinstance(object, base_metric.Metric)
        or isinstance(object, losses.Loss)
    )


def is_container(object):
    return (
        isinstance(object, list)
        or isinstance(object, tuple)
        or isinstance(object, dict)
    )


def _extract_dir(zipfile_to_load, root_system_path, zip_dir):
    for zip_path in zipfile_to_load.namelist():
        if zip_path.startswith(zip_dir):
            created_path = zipfile_to_load.extract(zip_path, root_system_path)
            logging.debug(
                f"Extracting {zip_path} into {root_system_path}. "
                f"Created {created_path}."
            )


def _load_state(trackable, zip_dir_path, temp_path, zipfile_to_load):
    states_dir_path = tf.io.gfile.join(zip_dir_path, _SELF_DIRNAME)
    # Extract the whole directory that represents the states of the trackable
    # into a temporary path.
    _extract_dir(zipfile_to_load, temp_path, states_dir_path)
    dir_path_to_load_state = tf.io.gfile.join(temp_path, states_dir_path)
    # TODO(rchao): Make `.set_state()` and `.load_state()` exported methods
    # and remove the attr check.
    if hasattr(trackable, "_load_state"):
        trackable._load_state(dir_path_to_load_state)
    if tf.io.gfile.exists(dir_path_to_load_state):
        tf.io.gfile.rmtree(dir_path_to_load_state)

    # Recursively load states for Keras trackables such as layers/optimizers.
    for child_attr in dir(trackable):
        if (
            child_attr == "_self_tracked_trackables"
            or child_attr == "_layer_call_argspecs"
            or child_attr == "_output_layers"
        ):
            # Avoid certain attribute names to allow readable state file paths,
            # e.g., `layers`.
            continue
        try:
            child_obj = getattr(trackable, child_attr)
        except Exception:
            # Avoid raising the exception when visiting the attributes.
            continue
        if _is_keras_trackable(child_obj):
            _load_state(
                child_obj,
                tf.io.gfile.join(zip_dir_path, child_attr),
                temp_path,
                zipfile_to_load,
            )
        elif is_container(child_obj):
            _load_container_state(
                child_obj,
                tf.io.gfile.join(zip_dir_path, child_attr),
                temp_path,
                zipfile_to_load,
            )


def _load_container_state(container, zip_dir_path, temp_path, zipfile_to_load):
    for trackable in container:
        if _is_keras_trackable(trackable):
            _load_state(
                trackable,
                tf.io.gfile.join(zip_dir_path, trackable.name),
                temp_path,
                zipfile_to_load,
            )


def load_model(dirpath, custom_objects=None):
    """Load a zip-archive representing a Keras model given the container dir."""
    file_path = tf.io.gfile.join(dirpath, _ARCHIVE_FILENAME)
    temp_path = tempfile.mkdtemp(dir=dirpath)

    with zipfile.ZipFile(file_path, "r") as zipfile_to_load:
        _print_archive(zipfile_to_load, "loaded from")
        with zipfile_to_load.open(_CONFIG_FILENAME, "r") as c:
            config_json = c.read()
        logging.debug(f"Read config: {config_json} from {c}")
        config_dict = json_utils.decode(config_json)
        # Construct the model from the configuration file saved in the archive.
        model = deserialize_keras_object(config_dict, custom_objects)
        _load_state(model, _STATES_ROOT_DIRNAME, temp_path, zipfile_to_load)

    if tf.io.gfile.exists(temp_path):
        tf.io.gfile.rmtree(temp_path)
    return model


def _write_recursively(zipfile_to_save, system_path, zip_path):
    if not tf.io.gfile.isdir(system_path):
        zipfile_to_save.write(system_path, zip_path)
        logging.debug(f"Written {system_path} into {zip_path} in the zip.")
    else:
        for file_name in tf.io.gfile.listdir(system_path):
            system_file_path = tf.io.gfile.join(system_path, file_name)
            zip_file_path = tf.io.gfile.join(zip_path, file_name)
            _write_recursively(zipfile_to_save, system_file_path, zip_file_path)


def _save_state(
    trackable, zip_dir_path, temp_path, zipfile_to_save, saved_trackables
):
    # Check whether this trackable has been saved; if so, do not duplicate the
    # saving.
    if trackable in saved_trackables:
        return

    # TODO(rchao): Make `.get_state()` and `.save_state()` exported methods
    # and remove the attr check.
    if hasattr(trackable, "_save_state"):
        # Designate a `self` directory for the trackable object to save.
        states_dir_path = tf.io.gfile.join(temp_path, _SELF_DIRNAME)
        if not tf.io.gfile.exists(states_dir_path):
            tf.io.gfile.mkdir(states_dir_path)
        trackable._save_state(states_dir_path)
        if states_dir_path is not None:
            # Recursively write the states (represented by files inside the
            # directory) into the zip file.
            _write_recursively(
                zipfile_to_save,
                states_dir_path,
                tf.io.gfile.join(zip_dir_path, _SELF_DIRNAME),
            )
            tf.io.gfile.rmtree(states_dir_path)
        saved_trackables.add(trackable)

    # Recursively ask contained trackable (layers, optimizers,
    # etc.) to save states.
    for child_attr in dir(trackable):
        if (
            child_attr == "_self_tracked_trackables"
            or child_attr == "_layer_call_argspecs"
            or child_attr == "_output_layers"
        ):
            # Avoid certain attribute names to allow readable state file paths,
            # e.g., `layers`.
            continue
        try:
            child_obj = getattr(trackable, child_attr)
        except Exception:
            # Avoid raising the exception when visiting the attributes.
            continue
        if _is_keras_trackable(child_obj):
            _save_state(
                child_obj,
                tf.io.gfile.join(zip_dir_path, child_attr),
                temp_path,
                zipfile_to_save,
                saved_trackables,
            )
        elif is_container(child_obj):
            _save_container_state(
                child_obj,
                tf.io.gfile.join(zip_dir_path, child_attr),
                temp_path,
                zipfile_to_save,
                saved_trackables,
            )


def _save_container_state(
    container, zip_dir_path, temp_path, zipfile_to_save, saved_trackables
):
    for trackable in container:
        if _is_keras_trackable(trackable):
            _save_state(
                trackable,
                tf.io.gfile.join(zip_dir_path, trackable.name),
                temp_path,
                zipfile_to_save,
                saved_trackables,
            )


def save_model(model, dirpath):
    """Save a zip-archive representing a Keras model given the container dir.

    The zip-based archive contains the following structure:

    - JSON-based configuration file (config.json): Records of model, layer, and
        other trackables' configuration.
    - NPZ-based trackable state files, found in respective directories, such as
        model/states.npz, model/dense_layer/states.npz, etc.
    - Metadata file (this is a TODO).

    The states of Keras trackables (layers, optimizers, loss, and metrics) are
    automatically saved as long as they can be discovered through the attributes
    returned by `dir(Model)`. Typically, the state includes the variables
    associated with the trackable, but some specially purposed layers may
    contain more such as the vocabularies stored in the hashmaps. The trackables
    define how their states are saved by exposing `save_state()` and
    `load_state()` APIs.

    For the case of layer states, the variables will be visited as long as
    they are either 1) referenced via layer attributes, or 2) referenced via a
    container (list, tuple, or dict), and the container is referenced via a
    layer attribute. Note that nested containers will not be visited.
    """
    if not tf.io.gfile.exists(dirpath):
        tf.io.gfile.mkdir(dirpath)
    file_path = tf.io.gfile.join(dirpath, _ARCHIVE_FILENAME)

    # TODO(rchao): Save the model's metadata (e.g. Keras version) in a separate
    # file in the archive.
    serialized_model_dict = serialize_keras_object(model)
    config_json = json.dumps(
        serialized_model_dict, cls=json_utils.Encoder
    ).encode()

    # Utilize a temporary directory for the interim npz files.
    temp_path = tempfile.mkdtemp(dir=dirpath)
    if not tf.io.gfile.exists(temp_path):
        tf.io.gfile.mkdir(temp_path)

    # Save the configuration json and state npz's.
    with zipfile.ZipFile(file_path, "x") as zipfile_to_save:
        with zipfile_to_save.open(_CONFIG_FILENAME, "w") as c:
            c.write(config_json)
            logging.debug(f"Written config: {config_json} into {c}.")
        _save_state(
            model, _STATES_ROOT_DIRNAME, temp_path, zipfile_to_save, set()
        )
        _print_archive(zipfile_to_save, "saved in")

    # Remove the directory temporarily used.
    tf.io.gfile.rmtree(temp_path)


# TODO(rchao): Replace the current Keras' `deserialize_keras_object` with this
# (as well as the reciprocal function).
def deserialize_keras_object(config_dict, custom_objects=None):
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
      resolves to a class (that was registered), `class_name` and `config`
      values in the dict will not be used. `registered_name` is only used for
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

    If the class does not have an exported Keras namespace, the library tracks
    it by its `module` and `class_name`. For example:

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
      config_dict: the python dict structure to deserialize the Keras object
        from.

    Returns:
      The Keras object that is deserialized from `config_dict`.

    """
    # TODO(rchao): Design a 'version' key for `config_dict` for defining
    # versions for classes.
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

            # Otherwise, attempt to import the tracked module, and find the
            # function.
            function_module = config.get("module", None)
            try:
                function_module = importlib.import_module(function_module)
            except ImportError as e:
                raise ImportError(
                    f"The function module {function_module} is not available. "
                    f"The config dictionary provided is {config_dict}."
                ) from e
            return vars(function_module).get(config["function_name"])

        raise TypeError(f"Unrecognized type: {class_name}")

    custom_class = generic_utils.get_custom_objects_by_name(registered_name)
    if custom_class is not None:
        # For others (classes), see if there is a custom class registered (via
        # `register_keras_serializable` API). If so, that takes precedence.
        return custom_class.from_config(config)
    else:
        # Otherwise, attempt to retrieve the class object given the `module`,
        # and `class_name`.
        if module is None:
            # In the case where `module` is not recorded, the `class_name`
            # represents the full exported Keras namespace (used by
            # `keras_export`) such as "keras.optimizers.Adam".
            cls = tf_export.get_symbol_from_name(class_name)
        else:
            # In the case where `module` is available, the class does not have
            # an Keras namespace (which is the case when the symbol is not
            # exported via `keras_export`). Import the tracked module (that is
            # used for the internal path), find the class, and use its config.
            mod = importlib.import_module(module)
            cls = vars(mod).get(class_name, None)
        if not hasattr(cls, "from_config"):
            raise TypeError(
                f"Unable to reconstruct an instance of {cls}. "
                "Make sure custom classes are decorated with "
                "`@keras.utils.register_keras_serializable`."
            )
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

    # Note that in the case of the `obj` being a function, the module used will
    # be "builtins", and the `class_name` used will be "function"; in the case
    # of the `obj` being a string, the module used will be "builtins", and the
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
