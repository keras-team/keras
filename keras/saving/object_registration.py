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
"""Python utilities required by Keras."""

import inspect
import threading

# isort: off
from tensorflow.python.util.tf_export import keras_export

_GLOBAL_CUSTOM_OBJECTS = {}
_GLOBAL_CUSTOM_NAMES = {}
# Thread-local custom objects set by custom_object_scope.
_THREAD_LOCAL_CUSTOM_OBJECTS = threading.local()


@keras_export(
    "keras.saving.custom_object_scope",
    "keras.utils.custom_object_scope",
    "keras.utils.CustomObjectScope",
)
class CustomObjectScope:
    """Exposes custom classes/functions to Keras deserialization internals.

    Under a scope `with custom_object_scope(objects_dict)`, Keras methods such
    as `tf.keras.models.load_model` or `tf.keras.models.model_from_config`
    will be able to deserialize any custom object referenced by a
    saved config (e.g. a custom layer or metric).

    Example:

    Consider a custom regularizer `my_regularizer`:

    ```python
    layer = Dense(3, kernel_regularizer=my_regularizer)
    # Config contains a reference to `my_regularizer`
    config = layer.get_config()
    ...
    # Later:
    with custom_object_scope({'my_regularizer': my_regularizer}):
      layer = Dense.from_config(config)
    ```

    Args:
        *args: Dictionary or dictionaries of `{name: object}` pairs.
    """

    def __init__(self, *args):
        self.custom_objects = args
        self.backup = None

    def __enter__(self):
        self.backup = _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__.copy()
        for objects in self.custom_objects:
            _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__.update(objects)
        return self

    def __exit__(self, *args, **kwargs):
        _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__.clear()
        _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__.update(self.backup)


@keras_export(
    "keras.saving.get_custom_objects", "keras.utils.get_custom_objects"
)
def get_custom_objects():
    """Retrieves a live reference to the global dictionary of custom objects.

    Custom objects set using using `custom_object_scope` are not added to the
    global dictionary of custom objects, and will not appear in the returned
    dictionary.

    Example:

    ```python
    get_custom_objects().clear()
    get_custom_objects()['MyObject'] = MyObject
    ```

    Returns:
        Global dictionary mapping registered class names to classes.
    """
    return _GLOBAL_CUSTOM_OBJECTS


@keras_export(
    "keras.saving.register_keras_serializable",
    "keras.utils.register_keras_serializable",
)
def register_keras_serializable(package="Custom", name=None):
    """Registers an object with the Keras serialization framework.

    This decorator injects the decorated class or function into the Keras custom
    object dictionary, so that it can be serialized and deserialized without
    needing an entry in the user-provided custom object dict. It also injects a
    function that Keras will call to get the object's serializable string key.

    Note that to be serialized and deserialized, classes must implement the
    `get_config()` method. Functions do not have this requirement.

    The object will be registered under the key 'package>name' where `name`,
    defaults to the object name if not passed.

    Example:

    ```python
    # Note that `'my_package'` is used as the `package` argument here, and since
    # the `name` argument is not provided, `'MyDense'` is used as the `name`.
    @keras.saving.register_keras_serializable('my_package')
    class MyDense(keras.layers.Dense):
      pass

    assert keras.saving.get_registered_object('my_package>MyDense') == MyDense
    assert keras.saving.get_registered_name(MyDense) == 'my_package>MyDense'
    ```

    Args:
      package: The package that this class belongs to. This is used for the
        `key` (which is `"package>name"`) to idenfify the class. Note that this
        is the first argument passed into the decorator.
      name: The name to serialize this class under in this package. If not
        provided or `None`, the class' name will be used (note that this is the
        case when the decorator is used with only one argument, which becomes
        the `package`).

    Returns:
      A decorator that registers the decorated class with the passed names.
    """

    def decorator(arg):
        """Registers a class with the Keras serialization framework."""
        class_name = name if name is not None else arg.__name__
        registered_name = package + ">" + class_name

        if inspect.isclass(arg) and not hasattr(arg, "get_config"):
            raise ValueError(
                "Cannot register a class that does not have a "
                "get_config() method."
            )

        if registered_name in _GLOBAL_CUSTOM_OBJECTS:
            raise ValueError(
                f"{registered_name} has already been registered to "
                f"{_GLOBAL_CUSTOM_OBJECTS[registered_name]}"
            )

        if arg in _GLOBAL_CUSTOM_NAMES:
            raise ValueError(
                f"{arg} has already been registered to "
                f"{_GLOBAL_CUSTOM_NAMES[arg]}"
            )
        _GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
        _GLOBAL_CUSTOM_NAMES[arg] = registered_name

        return arg

    return decorator


@keras_export(
    "keras.saving.get_registered_name", "keras.utils.get_registered_name"
)
def get_registered_name(obj):
    """Returns the name registered to an object within the Keras framework.

    This function is part of the Keras serialization and deserialization
    framework. It maps objects to the string names associated with those objects
    for serialization/deserialization.

    Args:
      obj: The object to look up.

    Returns:
      The name associated with the object, or the default Python name if the
        object is not registered.
    """
    if obj in _GLOBAL_CUSTOM_NAMES:
        return _GLOBAL_CUSTOM_NAMES[obj]
    else:
        return obj.__name__


@keras_export(
    "keras.saving.get_registered_object", "keras.utils.get_registered_object"
)
def get_registered_object(name, custom_objects=None, module_objects=None):
    """Returns the class associated with `name` if it is registered with Keras.

    This function is part of the Keras serialization and deserialization
    framework. It maps strings to the objects associated with them for
    serialization/deserialization.

    Example:

    ```python
    def from_config(cls, config, custom_objects=None):
      if 'my_custom_object_name' in config:
        config['hidden_cls'] = tf.keras.saving.get_registered_object(
            config['my_custom_object_name'], custom_objects=custom_objects)
    ```

    Args:
      name: The name to look up.
      custom_objects: A dictionary of custom objects to look the name up in.
        Generally, custom_objects is provided by the user.
      module_objects: A dictionary of custom objects to look the name up in.
        Generally, module_objects is provided by midlevel library implementers.

    Returns:
      An instantiable class associated with `name`, or `None` if no such class
        exists.
    """
    if name in _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__:
        return _THREAD_LOCAL_CUSTOM_OBJECTS.__dict__[name]
    elif name in _GLOBAL_CUSTOM_OBJECTS:
        return _GLOBAL_CUSTOM_OBJECTS[name]
    elif custom_objects and name in custom_objects:
        return custom_objects[name]
    elif module_objects and name in module_objects:
        return module_objects[name]
    return None


# Aliases
custom_object_scope = CustomObjectScope
