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
"""Object config serialization and deserialization logic."""

import importlib
import inspect
import threading
import types
import warnings

import numpy as np
import tensorflow.compat.v2 as tf

from keras.saving import object_registration
from keras.saving.legacy import serialization as legacy_serialization
from keras.saving.legacy.saved_model.utils import in_tf_saved_model_scope
from keras.utils import generic_utils

# isort: off
from tensorflow.python.util import tf_export
from tensorflow.python.util.tf_export import keras_export

PLAIN_TYPES = (str, int, float, bool)
SHARED_OBJECTS = threading.local()
SAFE_MODE = threading.local()


class Config:
    def __init__(self, **config):
        self.config = config

    def serialize(self):
        return serialize_keras_object(self.config)


class SafeModeScope:
    """Scope to propagate safe mode flag to nested deserialization calls."""

    def __init__(self, safe_mode=True):
        self.safe_mode = safe_mode

    def __enter__(self):
        self.original_value = in_safe_mode()
        SAFE_MODE.safe_mode = self.safe_mode

    def __exit__(self, *args, **kwargs):
        SAFE_MODE.safe_mode = self.original_value


@keras_export("keras.__internal__.enable_unsafe_deserialization")
def enable_unsafe_deserialization():
    """Disables safe mode globally, allowing deserialization of lambdas."""
    SAFE_MODE.safe_mode = False


def in_safe_mode():
    return getattr(SAFE_MODE, "safe_mode", None)


class ObjectSharingScope:
    """Scope to enable detection and reuse of previously seen objects."""

    def __enter__(self):
        SHARED_OBJECTS.enabled = True
        SHARED_OBJECTS.id_to_obj_map = {}
        SHARED_OBJECTS.id_to_config_map = {}

    def __exit__(self, *args, **kwargs):
        SHARED_OBJECTS.enabled = False
        SHARED_OBJECTS.id_to_obj_map = {}
        SHARED_OBJECTS.id_to_config_map = {}


def get_shared_object(obj_id):
    """Retrieve an object previously seen during deserialization."""
    if getattr(SHARED_OBJECTS, "enabled", False):
        return SHARED_OBJECTS.id_to_obj_map.get(obj_id, None)


def record_object_after_serialization(obj, config):
    """Call after serializing an object, to keep track of its config."""
    if not getattr(SHARED_OBJECTS, "enabled", False):
        return  # Not in a sharing scope
    obj_id = int(id(obj))
    if obj_id not in SHARED_OBJECTS.id_to_config_map:
        SHARED_OBJECTS.id_to_config_map[obj_id] = config
    else:
        config["shared_object_id"] = obj_id
        prev_config = SHARED_OBJECTS.id_to_config_map[obj_id]
        prev_config["shared_object_id"] = obj_id


def record_object_after_deserialization(obj, obj_id):
    """Call after deserializing an object, to keep track of it in the future."""
    if not getattr(SHARED_OBJECTS, "enabled", False):
        return  # Not in a sharing scope
    SHARED_OBJECTS.id_to_obj_map[obj_id] = obj


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
    # Fall back to legacy serialization for all TF1 users or if
    # wrapped by in_tf_saved_model_scope() to explicitly use legacy
    # saved_model logic.
    if not tf.__internal__.tf2.enabled() or in_tf_saved_model_scope():
        return legacy_serialization.serialize_keras_object(obj)

    if obj is None:
        return obj
    if isinstance(obj, PLAIN_TYPES):
        return obj

    if isinstance(obj, (list, tuple)):
        return [serialize_keras_object(x) for x in obj]
    if isinstance(obj, dict):
        return serialize_dict(obj)

    # Special cases:
    if isinstance(obj, bytes):
        return {
            "class_name": "__bytes__",
            "config": {"value": obj.decode("utf-8")},
        }
    if isinstance(obj, tf.TensorShape):
        return obj.as_list()
    if isinstance(obj, tf.Tensor):
        return {
            "class_name": "__tensor__",
            "config": {
                "value": obj.numpy().tolist(),
                "dtype": obj.dtype.name,
            },
        }
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return {
                "class_name": "__numpy__",
                "config": {
                    "value": obj.tolist(),
                    "dtype": obj.dtype.name,
                },
            }
        else:
            # Treat numpy floats / etc as plain types.
            return obj.item()
    if isinstance(obj, tf.DType):
        return obj.name
    if isinstance(obj, types.FunctionType) and obj.__name__ == "<lambda>":
        warnings.warn(
            "The object being serialized includes a `lambda`. This is unsafe. "
            "In order to reload the object, you will have to pass "
            "`safe_mode=False` to the loading function. "
            "Please avoid using `lambda` in the "
            "future, and use named Python functions instead. "
            f"This is the `lambda` being serialized: {inspect.getsource(obj)}",
            stacklevel=2,
        )
        return {
            "class_name": "__lambda__",
            "config": {
                "value": generic_utils.func_dump(obj),
            },
        }
    if isinstance(obj, tf.TypeSpec):
        ts_config = obj._serialize()
        # TensorShape and tf.DType conversion
        ts_config = list(
            map(
                lambda x: x.as_list()
                if isinstance(x, tf.TensorShape)
                else (x.name if isinstance(x, tf.DType) else x),
                ts_config,
            )
        )
        return {
            "class_name": "__typespec__",
            "spec_name": obj.__class__.__name__,
            "module": obj.__class__.__module__,
            "config": ts_config,
            "registered_name": None,
        }

    # This gets the `keras.*` exported name, such as "keras.optimizers.Adam".
    keras_api_name = tf_export.get_canonical_name_for_symbol(
        obj.__class__, api_name="keras"
    )
    if keras_api_name is None:
        # Any custom object or otherwise non-exported object
        if isinstance(obj, types.FunctionType):
            module = obj.__module__
        else:
            module = obj.__class__.__module__
        class_name = obj.__class__.__name__
        if module == "builtins":
            registered_name = None
        else:
            if isinstance(obj, types.FunctionType):
                registered_name = object_registration.get_registered_name(obj)
            else:
                registered_name = object_registration.get_registered_name(
                    obj.__class__
                )
    else:
        # A publicly-exported Keras object
        parts = keras_api_name.split(".")
        module = ".".join(parts[:-1])
        class_name = parts[-1]
        registered_name = None
    config = {
        "module": module,
        "class_name": class_name,
        "config": _get_class_or_fn_config(obj),
        "registered_name": registered_name,
    }
    if hasattr(obj, "get_build_config"):
        build_config = obj.get_build_config()
        if build_config is not None:
            config["build_config"] = serialize_dict(build_config)
    if hasattr(obj, "get_compile_config"):
        compile_config = obj.get_compile_config()
        if compile_config is not None:
            config["compile_config"] = serialize_dict(compile_config)
    record_object_after_serialization(obj, config)
    return config


def _get_class_or_fn_config(obj):
    """Return the object's config depending on its type."""
    # Functions / lambdas:
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    # All classes:
    if hasattr(obj, "get_config"):
        config = obj.get_config()
        if not isinstance(config, dict):
            raise TypeError(
                f"The `get_config()` method of {obj} should return "
                f"a dict. It returned: {config}"
            )
        return serialize_dict(config)
    else:
        raise TypeError(
            f"Cannot serialize object {obj} of type {type(obj)}. "
            "To be serializable, "
            "a class must implement the `get_config()` method."
        )


def serialize_dict(obj):
    return {key: serialize_keras_object(value) for key, value in obj.items()}


def deserialize_keras_object(
    config, custom_objects=None, safe_mode=True, **kwargs
):
    """Retrieve the object by deserializing the config dict.

    The config dict is a Python dictionary that consists of a set of key-value
    pairs, and represents a Keras object, such as an `Optimizer`, `Layer`,
    `Metrics`, etc. The saving and loading library uses the following keys to
    record information of a Keras object:

    - `class_name`: String. This is the name of the class,
      as exactly defined in the source
      code, such as "LossesContainer".
    - `config`: Dict. Library-defined or user-defined key-value pairs that store
      the configuration of the object, as obtained by `object.get_config()`.
    - `module`: String. The path of the python module, such as
      "keras.engine.compile_utils". Built-in Keras classes
      expect to have prefix `keras`.
    - `registered_name`: String. The key the class is registered under via
      `keras.utils.register_keras_serializable(package, name)` API. The key has
      the format of '{package}>{name}', where `package` and `name` are the
      arguments passed to `register_keras_serializable()`. If `name` is not
      provided, it defaults to the class name. If `registered_name` successfully
      resolves to a class (that was registered), the `class_name` and `config`
      values in the dict will not be used. `registered_name` is only used for
      non-built-in classes.

    For example, the following dictionary represents the built-in Adam optimizer
    with the relevant config:

    ```python
    dict_structure = {
        "class_name": "Adam",
        "config": {
            "amsgrad": false,
            "beta_1": 0.8999999761581421,
            "beta_2": 0.9990000128746033,
            "decay": 0.0,
            "epsilon": 1e-07,
            "learning_rate": 0.0010000000474974513,
            "name": "Adam"
        },
        "module": "keras.optimizers",
        "registered_name": None
    }
    # Returns an `Adam` instance identical to the original one.
    deserialize_keras_object(dict_structure)
    ```

    If the class does not have an exported Keras namespace, the library tracks
    it by its `module` and `class_name`. For example:

    ```python
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

    ```python
    @keras.utils.register_keras_serializable(package='my_package')
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
    # Returns the `ModifiedMeanSquaredError` object
    deserialize_keras_object(dict_structure)
    ```

    Args:
        config: Python dict describing the object.
        custom_objects: Python dict containing a mapping between custom
            object names the corresponding classes or functions.
        safe_mode: Boolean, whether to disallow unsafe `lambda` deserialization.
            When `safe_mode=False`, loading an object has the potential to
            trigger arbitrary code execution. This argument is only
            applicable to the Keras v3 model format. Defaults to True.

    Returns:
      The object described by the `config` dictionary.

    """
    safe_scope_arg = in_safe_mode()  # Enforces SafeModeScope
    safe_mode = safe_scope_arg if safe_scope_arg is not None else safe_mode

    module_objects = kwargs.pop("module_objects", None)
    custom_objects = custom_objects or {}

    # Fall back to legacy deserialization for all TF1 users or if
    # wrapped by in_tf_saved_model_scope() to explicitly use legacy
    # saved_model logic.
    if not tf.__internal__.tf2.enabled() or in_tf_saved_model_scope():
        return legacy_serialization.deserialize_keras_object(
            config, module_objects, custom_objects
        )

    if config is None:
        return None
    if isinstance(config, PLAIN_TYPES):
        if isinstance(config, str) and custom_objects.get(config) is not None:
            # This is to deserialize plain functions which are serialized as
            # string names by legacy saving formats.
            return custom_objects[config]
        return config
    if isinstance(config, (list, tuple)):
        return [
            deserialize_keras_object(
                x, custom_objects=custom_objects, safe_mode=safe_mode
            )
            for x in config
        ]
    if not isinstance(config, dict):
        raise TypeError(f"Could not parse config: {config}")

    if "class_name" not in config or "config" not in config:
        return {
            key: deserialize_keras_object(
                value, custom_objects=custom_objects, safe_mode=safe_mode
            )
            for key, value in config.items()
        }

    class_name = config["class_name"]
    inner_config = config["config"]

    # Special cases:
    if class_name == "__tensor__":
        return tf.constant(inner_config["value"], dtype=inner_config["dtype"])
    if class_name == "__numpy__":
        return np.array(inner_config["value"], dtype=inner_config["dtype"])
    if config["class_name"] == "__bytes__":
        return inner_config["value"].encode("utf-8")
    if config["class_name"] == "__lambda__":
        if safe_mode:
            raise ValueError(
                "Requested the deserialization of a `lambda` object. "
                "This carries a potential risk of arbitrary code execution "
                "and thus it is disallowed by default. If you trust the "
                "source of the saved model, you can pass `safe_mode=False` to "
                "the loading function in order to allow `lambda` loading."
            )
        return generic_utils.func_load(inner_config["value"])
    if config["class_name"] == "__typespec__":
        obj = _retrieve_class_or_fn(
            config["spec_name"],
            config["registered_name"],
            config["module"],
            obj_type="class",
            full_config=config,
            custom_objects=custom_objects,
        )
        # Conversion to TensorShape and tf.DType
        inner_config = map(
            lambda x: tf.TensorShape(x)
            if isinstance(x, list)
            else (getattr(tf, x) if hasattr(tf.dtypes, str(x)) else x),
            inner_config,
        )
        return obj._deserialize(tuple(inner_config))
    # TODO(fchollet): support for TypeSpec, CompositeTensor, tf.Dtype
    # TODO(fchollet): consider special-casing tuples (which are currently
    # deserialized as lists).

    # Below: classes and functions.
    module = config.get("module", None)
    registered_name = config.get("registered_name", class_name)

    if class_name == "function":
        fn_name = inner_config
        return _retrieve_class_or_fn(
            fn_name,
            registered_name,
            module,
            obj_type="function",
            full_config=config,
            custom_objects=custom_objects,
        )

    # Below, handling of all classes.
    # First, is it a shared object?
    if "shared_object_id" in config:
        obj = get_shared_object(config["shared_object_id"])
        if obj is not None:
            return obj

    cls = _retrieve_class_or_fn(
        class_name,
        registered_name,
        module,
        obj_type="class",
        full_config=config,
        custom_objects=custom_objects,
    )
    if not hasattr(cls, "from_config"):
        raise TypeError(
            f"Unable to reconstruct an instance of '{class_name}' because "
            f"the class is missing a `from_config()` method. "
            f"Full object config: {config}"
        )

    # Instantiate the class from its config inside a custom object scope
    # so that we can catch any custom objects that the config refers to.
    custom_obj_scope = object_registration.custom_object_scope(custom_objects)
    safe_mode_scope = SafeModeScope(safe_mode)
    with custom_obj_scope, safe_mode_scope:
        instance = cls.from_config(inner_config)
        build_config = config.get("build_config", None)
        if build_config:
            instance.build_from_config(build_config)
        compile_config = config.get("compile_config", None)
        if compile_config:
            instance.compile_from_config(compile_config)

    if "shared_object_id" in config:
        record_object_after_deserialization(
            instance, config["shared_object_id"]
        )
    return instance


def _retrieve_class_or_fn(
    name, registered_name, module, obj_type, full_config, custom_objects=None
):
    # If there is a custom object registered via
    # `register_keras_serializable`, that takes precedence.
    custom_obj = object_registration.get_registered_object(
        registered_name, custom_objects=custom_objects
    )
    if custom_obj is not None:
        return custom_obj

    if module:
        # If it's a Keras built-in object,
        # we cannot always use direct import, because the exported
        # module name might not match the package structure
        # (e.g. experimental symbols).
        if module == "keras" or module.startswith("keras."):
            obj = tf_export.get_symbol_from_name(module + "." + name)
            if obj is not None:
                return obj

        # Otherwise, attempt to retrieve the class object given the `module`
        # and `class_name`. Import the module, find the class.
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            raise TypeError(
                f"Could not deserialize {obj_type} '{name}' because "
                f"its parent module {module} cannot be imported. "
                f"Full object config: {full_config}"
            )
        obj = vars(mod).get(name, None)
        if obj is not None:
            return obj

    raise TypeError(
        f"Could not locate {obj_type} '{name}'. "
        "Make sure custom classes are decorated with "
        "`@keras.utils.register_keras_serializable`. "
        f"Full object config: {full_config}"
    )
