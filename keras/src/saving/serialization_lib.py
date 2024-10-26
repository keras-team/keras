"""Object config serialization and deserialization logic."""

import importlib
import inspect
import types
import warnings

import numpy as np

from keras.src import api_export
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.saving import object_registration
from keras.src.utils import python_utils
from keras.src.utils.module_utils import tensorflow as tf

PLAIN_TYPES = (str, int, float, bool)

# List of Keras modules with built-in string representations for Keras defaults
BUILTIN_MODULES = (
    "activations",
    "constraints",
    "initializers",
    "losses",
    "metrics",
    "optimizers",
    "regularizers",
)


class SerializableDict:
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
        global_state.set_global_attribute("safe_mode_saving", self.safe_mode)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute(
            "safe_mode_saving", self.original_value
        )


@keras_export("keras.config.enable_unsafe_deserialization")
def enable_unsafe_deserialization():
    """Disables safe mode globally, allowing deserialization of lambdas."""
    global_state.set_global_attribute("safe_mode_saving", False)


def in_safe_mode():
    return global_state.get_global_attribute("safe_mode_saving")


class ObjectSharingScope:
    """Scope to enable detection and reuse of previously seen objects."""

    def __enter__(self):
        global_state.set_global_attribute("shared_objects/id_to_obj_map", {})
        global_state.set_global_attribute("shared_objects/id_to_config_map", {})

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("shared_objects/id_to_obj_map", None)
        global_state.set_global_attribute(
            "shared_objects/id_to_config_map", None
        )


def get_shared_object(obj_id):
    """Retrieve an object previously seen during deserialization."""
    id_to_obj_map = global_state.get_global_attribute(
        "shared_objects/id_to_obj_map"
    )
    if id_to_obj_map is not None:
        return id_to_obj_map.get(obj_id, None)


def record_object_after_serialization(obj, config):
    """Call after serializing an object, to keep track of its config."""
    if config["module"] == "__main__":
        config["module"] = None  # Ensures module is None when no module found
    id_to_config_map = global_state.get_global_attribute(
        "shared_objects/id_to_config_map"
    )
    if id_to_config_map is None:
        return  # Not in a sharing scope
    obj_id = int(id(obj))
    if obj_id not in id_to_config_map:
        id_to_config_map[obj_id] = config
    else:
        config["shared_object_id"] = obj_id
        prev_config = id_to_config_map[obj_id]
        prev_config["shared_object_id"] = obj_id


def record_object_after_deserialization(obj, obj_id):
    """Call after deserializing an object, to keep track of it in the future."""
    id_to_obj_map = global_state.get_global_attribute(
        "shared_objects/id_to_obj_map"
    )
    if id_to_obj_map is None:
        return  # Not in a sharing scope
    id_to_obj_map[obj_id] = obj


@keras_export(
    [
        "keras.saving.serialize_keras_object",
        "keras.utils.serialize_keras_object",
    ]
)
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
    if obj is None:
        return obj

    if isinstance(obj, PLAIN_TYPES):
        return obj

    if isinstance(obj, (list, tuple)):
        config_arr = [serialize_keras_object(x) for x in obj]
        return tuple(config_arr) if isinstance(obj, tuple) else config_arr
    if isinstance(obj, dict):
        return serialize_dict(obj)

    # Special cases:
    if isinstance(obj, bytes):
        return {
            "class_name": "__bytes__",
            "config": {"value": obj.decode("utf-8")},
        }
    if isinstance(obj, slice):
        return {
            "class_name": "__slice__",
            "config": {
                "start": serialize_keras_object(obj.start),
                "stop": serialize_keras_object(obj.stop),
                "step": serialize_keras_object(obj.step),
            },
        }
    # Ellipsis is an instance, and ellipsis class is not in global scope.
    # checking equality also fails elsewhere in the library, so we have
    # to dynamically get the type.
    if isinstance(obj, type(Ellipsis)):
        return {"class_name": "__ellipsis__", "config": {}}
    if isinstance(obj, backend.KerasTensor):
        history = getattr(obj, "_keras_history", None)
        if history:
            history = list(history)
            history[0] = history[0].name
        return {
            "class_name": "__keras_tensor__",
            "config": {
                "shape": obj.shape,
                "dtype": obj.dtype,
                "keras_history": history,
            },
        }
    if tf.available and isinstance(obj, tf.TensorShape):
        return obj.as_list() if obj._dims is not None else None
    if backend.is_tensor(obj):
        return {
            "class_name": "__tensor__",
            "config": {
                "value": backend.convert_to_numpy(obj).tolist(),
                "dtype": backend.standardize_dtype(obj.dtype),
            },
        }
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray) and obj.ndim > 0:
            return {
                "class_name": "__numpy__",
                "config": {
                    "value": obj.tolist(),
                    "dtype": backend.standardize_dtype(obj.dtype),
                },
            }
        else:
            # Treat numpy floats / etc as plain types.
            return obj.item()
    if tf.available and isinstance(obj, tf.DType):
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
                "value": python_utils.func_dump(obj),
            },
        }
    if tf.available and isinstance(obj, tf.TypeSpec):
        ts_config = obj._serialize()
        # TensorShape and tf.DType conversion
        ts_config = list(
            map(
                lambda x: (
                    x.as_list()
                    if isinstance(x, tf.TensorShape)
                    else (x.name if isinstance(x, tf.DType) else x)
                ),
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

    inner_config = _get_class_or_fn_config(obj)
    config_with_public_class = serialize_with_public_class(
        obj.__class__, inner_config
    )

    if config_with_public_class is not None:
        get_build_and_compile_config(obj, config_with_public_class)
        record_object_after_serialization(obj, config_with_public_class)
        return config_with_public_class

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

    config = {
        "module": module,
        "class_name": class_name,
        "config": inner_config,
        "registered_name": registered_name,
    }
    get_build_and_compile_config(obj, config)
    record_object_after_serialization(obj, config)
    return config


def get_build_and_compile_config(obj, config):
    if hasattr(obj, "get_build_config"):
        build_config = obj.get_build_config()
        if build_config is not None:
            config["build_config"] = serialize_dict(build_config)
    if hasattr(obj, "get_compile_config"):
        compile_config = obj.get_compile_config()
        if compile_config is not None:
            config["compile_config"] = serialize_dict(compile_config)
    return


def serialize_with_public_class(cls, inner_config=None):
    """Serializes classes from public Keras API or object registration.

    Called to check and retrieve the config of any class that has a public
    Keras API or has been registered as serializable via
    `keras.saving.register_keras_serializable()`.
    """
    # This gets the `keras.*` exported name, such as
    # "keras.optimizers.Adam".
    keras_api_name = api_export.get_name_from_symbol(cls)

    # Case of custom or unknown class object
    if keras_api_name is None:
        registered_name = object_registration.get_registered_name(cls)
        if registered_name is None:
            return None

        # Return custom object config with corresponding registration name
        return {
            "module": cls.__module__,
            "class_name": cls.__name__,
            "config": inner_config,
            "registered_name": registered_name,
        }

    # Split the canonical Keras API name into a Keras module and class name.
    parts = keras_api_name.split(".")
    return {
        "module": ".".join(parts[:-1]),
        "class_name": parts[-1],
        "config": inner_config,
        "registered_name": None,
    }


def serialize_with_public_fn(fn, config, fn_module_name=None):
    """Serializes functions from public Keras API or object registration.

    Called to check and retrieve the config of any function that has a public
    Keras API or has been registered as serializable via
    `keras.saving.register_keras_serializable()`. If function's module name
    is already known, returns corresponding config.
    """
    if fn_module_name:
        return {
            "module": fn_module_name,
            "class_name": "function",
            "config": config,
            "registered_name": config,
        }
    keras_api_name = api_export.get_name_from_symbol(fn)
    if keras_api_name:
        parts = keras_api_name.split(".")
        return {
            "module": ".".join(parts[:-1]),
            "class_name": "function",
            "config": config,
            "registered_name": config,
        }
    else:
        registered_name = object_registration.get_registered_name(fn)
        if not registered_name and not fn.__module__ == "builtins":
            return None
        return {
            "module": fn.__module__,
            "class_name": "function",
            "config": config,
            "registered_name": registered_name,
        }


def _get_class_or_fn_config(obj):
    """Return the object's config depending on its type."""
    # Functions / lambdas:
    if isinstance(obj, types.FunctionType):
        return object_registration.get_registered_name(obj)
    # All classes:
    if hasattr(obj, "get_config"):
        config = obj.get_config()
        if not isinstance(config, dict):
            raise TypeError(
                f"The `get_config()` method of {obj} should return "
                f"a dict. It returned: {config}"
            )
        return serialize_dict(config)
    elif hasattr(obj, "__name__"):
        return object_registration.get_registered_name(obj)
    else:
        raise TypeError(
            f"Cannot serialize object {obj} of type {type(obj)}. "
            "To be serializable, "
            "a class must implement the `get_config()` method."
        )


def serialize_dict(obj):
    return {key: serialize_keras_object(value) for key, value in obj.items()}


@keras_export(
    [
        "keras.saving.deserialize_keras_object",
        "keras.utils.deserialize_keras_object",
    ]
)
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
    - `module`: String. The path of the python module. Built-in Keras classes
      expect to have prefix `keras`.
    - `registered_name`: String. The key the class is registered under via
      `keras.saving.register_keras_serializable(package, name)` API. The
      key has the format of '{package}>{name}', where `package` and `name` are
      the arguments passed to `register_keras_serializable()`. If `name` is not
      provided, it uses the class name. If `registered_name` successfully
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
      "class_name": "MetricsList",
      "config": {
          ...
      },
      "module": "keras.trainers.compile_utils",
      "registered_name": "MetricsList"
    }

    # Returns a `MetricsList` instance identical to the original one.
    deserialize_keras_object(dict_structure)
    ```

    And the following dictionary represents a user-customized `MeanSquaredError`
    loss:

    ```python
    @keras.saving.register_keras_serializable(package='my_package')
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
            applicable to the Keras v3 model format. Defaults to `True`.

    Returns:
        The object described by the `config` dictionary.
    """
    safe_scope_arg = in_safe_mode()  # Enforces SafeModeScope
    safe_mode = safe_scope_arg if safe_scope_arg is not None else safe_mode

    module_objects = kwargs.pop("module_objects", None)
    custom_objects = custom_objects or {}
    tlco = global_state.get_global_attribute("custom_objects_scope_dict", {})
    gco = object_registration.GLOBAL_CUSTOM_OBJECTS
    custom_objects = {**custom_objects, **tlco, **gco}

    if config is None:
        return None

    if (
        isinstance(config, str)
        and custom_objects
        and custom_objects.get(config) is not None
    ):
        # This is to deserialize plain functions which are serialized as
        # string names by legacy saving formats.
        return custom_objects[config]

    if isinstance(config, (list, tuple)):
        return [
            deserialize_keras_object(
                x, custom_objects=custom_objects, safe_mode=safe_mode
            )
            for x in config
        ]

    if module_objects is not None:
        inner_config, fn_module_name, has_custom_object = None, None, False

        if isinstance(config, dict):
            if "config" in config:
                inner_config = config["config"]
            if "class_name" not in config:
                raise ValueError(
                    f"Unknown `config` as a `dict`, config={config}"
                )

            # Check case where config is function or class and in custom objects
            if custom_objects and (
                config["class_name"] in custom_objects
                or config.get("registered_name") in custom_objects
                or (
                    isinstance(inner_config, str)
                    and inner_config in custom_objects
                )
            ):
                has_custom_object = True

            # Case where config is function but not in custom objects
            elif config["class_name"] == "function":
                fn_module_name = config["module"]
                if fn_module_name == "builtins":
                    config = config["config"]
                else:
                    config = config["registered_name"]

            # Case where config is class but not in custom objects
            else:
                if config.get("module", "_") is None:
                    raise TypeError(
                        "Cannot deserialize object of type "
                        f"`{config['class_name']}`. If "
                        f"`{config['class_name']}` is a custom class, please "
                        "register it using the "
                        "`@keras.saving.register_keras_serializable()` "
                        "decorator."
                    )
                config = config["class_name"]

        if not has_custom_object:
            # Return if not found in either module objects or custom objects
            if config not in module_objects:
                # Object has already been deserialized
                return config
            if isinstance(module_objects[config], types.FunctionType):
                return deserialize_keras_object(
                    serialize_with_public_fn(
                        module_objects[config], config, fn_module_name
                    ),
                    custom_objects=custom_objects,
                )
            return deserialize_keras_object(
                serialize_with_public_class(
                    module_objects[config], inner_config=inner_config
                ),
                custom_objects=custom_objects,
            )

    if isinstance(config, PLAIN_TYPES):
        return config
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
    inner_config = config["config"] or {}
    custom_objects = custom_objects or {}

    # Special cases:
    if class_name == "__keras_tensor__":
        obj = backend.KerasTensor(
            inner_config["shape"], dtype=inner_config["dtype"]
        )
        obj._pre_serialization_keras_history = inner_config["keras_history"]
        return obj

    if class_name == "__tensor__":
        return backend.convert_to_tensor(
            inner_config["value"], dtype=inner_config["dtype"]
        )
    if class_name == "__numpy__":
        return np.array(inner_config["value"], dtype=inner_config["dtype"])
    if config["class_name"] == "__bytes__":
        return inner_config["value"].encode("utf-8")
    if config["class_name"] == "__ellipsis__":
        return Ellipsis
    if config["class_name"] == "__slice__":
        return slice(
            deserialize_keras_object(
                inner_config["start"],
                custom_objects=custom_objects,
                safe_mode=safe_mode,
            ),
            deserialize_keras_object(
                inner_config["stop"],
                custom_objects=custom_objects,
                safe_mode=safe_mode,
            ),
            deserialize_keras_object(
                inner_config["step"],
                custom_objects=custom_objects,
                safe_mode=safe_mode,
            ),
        )
    if config["class_name"] == "__lambda__":
        if safe_mode:
            raise ValueError(
                "Requested the deserialization of a `lambda` object. "
                "This carries a potential risk of arbitrary code execution "
                "and thus it is disallowed by default. If you trust the "
                "source of the saved model, you can pass `safe_mode=False` to "
                "the loading function in order to allow `lambda` loading, "
                "or call `keras.config.enable_unsafe_deserialization()`."
            )
        return python_utils.func_load(inner_config["value"])
    if tf is not None and config["class_name"] == "__typespec__":
        obj = _retrieve_class_or_fn(
            config["spec_name"],
            config["registered_name"],
            config["module"],
            obj_type="class",
            full_config=config,
            custom_objects=custom_objects,
        )
        # Conversion to TensorShape and DType
        inner_config = map(
            lambda x: (
                tf.TensorShape(x)
                if isinstance(x, list)
                else (getattr(tf, x) if hasattr(tf.dtypes, str(x)) else x)
            ),
            inner_config,
        )
        return obj._deserialize(tuple(inner_config))

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

    if isinstance(cls, types.FunctionType):
        return cls
    if not hasattr(cls, "from_config"):
        raise TypeError(
            f"Unable to reconstruct an instance of '{class_name}' because "
            f"the class is missing a `from_config()` method. "
            f"Full object config: {config}"
        )

    # Instantiate the class from its config inside a custom object scope
    # so that we can catch any custom objects that the config refers to.
    custom_obj_scope = object_registration.CustomObjectScope(custom_objects)
    safe_mode_scope = SafeModeScope(safe_mode)
    with custom_obj_scope, safe_mode_scope:
        try:
            instance = cls.from_config(inner_config)
        except TypeError as e:
            raise TypeError(
                f"{cls} could not be deserialized properly. Please"
                " ensure that components that are Python object"
                " instances (layers, models, etc.) returned by"
                " `get_config()` are explicitly deserialized in the"
                " model's `from_config()` method."
                f"\n\nconfig={config}.\n\nException encountered: {e}"
            )
        build_config = config.get("build_config", None)
        if build_config and not instance.built:
            instance.build_from_config(build_config)
            instance.built = True
        compile_config = config.get("compile_config", None)
        if compile_config:
            instance.compile_from_config(compile_config)
            instance.compiled = True

    if "shared_object_id" in config:
        record_object_after_deserialization(
            instance, config["shared_object_id"]
        )
    return instance


def _retrieve_class_or_fn(
    name, registered_name, module, obj_type, full_config, custom_objects=None
):
    # If there is a custom object registered via
    # `register_keras_serializable()`, that takes precedence.
    if obj_type == "function":
        custom_obj = object_registration.get_registered_object(
            name, custom_objects=custom_objects
        )
    else:
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
            api_name = module + "." + name

            obj = api_export.get_symbol_from_name(api_name)
            if obj is not None:
                return obj

        # Configs of Keras built-in functions do not contain identifying
        # information other than their name (e.g. 'acc' or 'tanh'). This special
        # case searches the Keras modules that contain built-ins to retrieve
        # the corresponding function from the identifying string.
        if obj_type == "function" and module == "builtins":
            for mod in BUILTIN_MODULES:
                obj = api_export.get_symbol_from_name(
                    "keras." + mod + "." + name
                )
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

        # Special case for keras.metrics.metrics
        if obj is None and registered_name is not None:
            obj = vars(mod).get(registered_name, None)

        if obj is not None:
            return obj

    raise TypeError(
        f"Could not locate {obj_type} '{name}'. "
        "Make sure custom classes are decorated with "
        "`@keras.saving.register_keras_serializable()`. "
        f"Full object config: {full_config}"
    )
