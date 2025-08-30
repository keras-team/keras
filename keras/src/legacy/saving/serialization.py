"""Legacy serialization logic for Keras models."""

import contextlib
import inspect
import threading
import weakref

# isort: off
from keras.src.api_export import keras_export
from keras.src.saving import object_registration

# Flag that determines whether to skip the NotImplementedError when calling
# get_config in custom models and layers. This is only enabled when saving to
# SavedModel, when the config isn't required.
_SKIP_FAILED_SERIALIZATION = False
# If a layer does not have a defined config, then the returned config will be a
# dictionary with the below key.
_LAYER_UNDEFINED_CONFIG_KEY = "layer was saved without config"

# Store a unique, per-object ID for shared objects.
#
# We store a unique ID for each object so that we may, at loading time,
# re-create the network properly.  Without this ID, we would have no way of
# determining whether a config is a description of a new object that
# should be created or is merely a reference to an already-created object.
SHARED_OBJECT_KEY = "shared_object_id"

SHARED_OBJECT_DISABLED = threading.local()
SHARED_OBJECT_LOADING = threading.local()
SHARED_OBJECT_SAVING = threading.local()


# Attributes on the threadlocal variable must be set per-thread, thus we
# cannot initialize these globally. Instead, we have accessor functions with
# default values.
def _shared_object_disabled():
    """Get whether shared object handling is disabled in a threadsafe manner."""
    return getattr(SHARED_OBJECT_DISABLED, "disabled", False)


def _shared_object_loading_scope():
    """Get the current shared object saving scope in a threadsafe manner."""
    return getattr(SHARED_OBJECT_LOADING, "scope", NoopLoadingScope())


def _shared_object_saving_scope():
    """Get the current shared object saving scope in a threadsafe manner."""
    return getattr(SHARED_OBJECT_SAVING, "scope", None)


class DisableSharedObjectScope:
    """A context manager for disabling handling of shared objects.

    Disables shared object handling for both saving and loading.

    Created primarily for use with `clone_model`, which does extra surgery that
    is incompatible with shared objects.
    """

    def __enter__(self):
        SHARED_OBJECT_DISABLED.disabled = True
        self._orig_loading_scope = _shared_object_loading_scope()
        self._orig_saving_scope = _shared_object_saving_scope()

    def __exit__(self, *args, **kwargs):
        SHARED_OBJECT_DISABLED.disabled = False
        SHARED_OBJECT_LOADING.scope = self._orig_loading_scope
        SHARED_OBJECT_SAVING.scope = self._orig_saving_scope


class NoopLoadingScope:
    """The default shared object loading scope. It does nothing.

    Created to simplify serialization code that doesn't care about shared
    objects (e.g. when serializing a single object).
    """

    def get(self, unused_object_id):
        return None

    def set(self, object_id, obj):
        pass


class SharedObjectLoadingScope:
    """A context manager for keeping track of loaded objects.

    During the deserialization process, we may come across objects that are
    shared across multiple layers. In order to accurately restore the network
    structure to its original state, `SharedObjectLoadingScope` allows us to
    re-use shared objects rather than cloning them.
    """

    def __enter__(self):
        if _shared_object_disabled():
            return NoopLoadingScope()

        global SHARED_OBJECT_LOADING
        SHARED_OBJECT_LOADING.scope = self
        self._obj_ids_to_obj = {}
        return self

    def get(self, object_id):
        """Given a shared object ID, returns a previously instantiated object.

        Args:
          object_id: shared object ID to use when attempting to find
            already-loaded object.

        Returns:
          The object, if we've seen this ID before. Else, `None`.
        """
        # Explicitly check for `None` internally to make external calling code a
        # bit cleaner.
        if object_id is None:
            return
        return self._obj_ids_to_obj.get(object_id)

    def set(self, object_id, obj):
        """Stores an instantiated object for future lookup and sharing."""
        if object_id is None:
            return
        self._obj_ids_to_obj[object_id] = obj

    def __exit__(self, *args, **kwargs):
        global SHARED_OBJECT_LOADING
        SHARED_OBJECT_LOADING.scope = NoopLoadingScope()


class SharedObjectConfig(dict):
    """A configuration container that keeps track of references.

    `SharedObjectConfig` will automatically attach a shared object ID to any
    configs which are referenced more than once, allowing for proper shared
    object reconstruction at load time.

    In most cases, it would be more proper to subclass something like
    `collections.UserDict` or `collections.Mapping` rather than `dict` directly.
    Unfortunately, python's json encoder does not support `Mapping`s. This is
    important functionality to retain, since we are dealing with serialization.

    We should be safe to subclass `dict` here, since we aren't actually
    overriding any core methods, only augmenting with a new one for reference
    counting.
    """

    def __init__(self, base_config, object_id, **kwargs):
        self.ref_count = 1
        self.object_id = object_id
        super().__init__(base_config, **kwargs)

    def increment_ref_count(self):
        # As soon as we've seen the object more than once, we want to attach the
        # shared object ID. This allows us to only attach the shared object ID
        # when it's strictly necessary, making backwards compatibility breakage
        # less likely.
        if self.ref_count == 1:
            self[SHARED_OBJECT_KEY] = self.object_id
        self.ref_count += 1


class SharedObjectSavingScope:
    """Keeps track of shared object configs when serializing."""

    def __enter__(self):
        if _shared_object_disabled():
            return None

        global SHARED_OBJECT_SAVING

        # Serialization can happen at a number of layers for a number of
        # reasons.  We may end up with a case where we're opening a saving scope
        # within another saving scope. In that case, we'd like to use the
        # outermost scope available and ignore inner scopes, since there is not
        # (yet) a reasonable use case for having these nested and distinct.
        if _shared_object_saving_scope() is not None:
            self._passthrough = True
            return _shared_object_saving_scope()
        else:
            self._passthrough = False

        SHARED_OBJECT_SAVING.scope = self
        self._shared_objects_config = weakref.WeakKeyDictionary()
        self._next_id = 0
        return self

    def get_config(self, obj):
        """Gets a `SharedObjectConfig` if one has already been seen for `obj`.

        Args:
          obj: The object for which to retrieve the `SharedObjectConfig`.

        Returns:
          The SharedObjectConfig for a given object, if already seen. Else,
            `None`.
        """
        try:
            shared_object_config = self._shared_objects_config[obj]
        except (TypeError, KeyError):
            # If the object is unhashable (e.g. a subclass of
            # `AbstractBaseClass` that has not overridden `__hash__`), a
            # `TypeError` will be thrown.  We'll just continue on without shared
            # object support.
            return None
        shared_object_config.increment_ref_count()
        return shared_object_config

    def create_config(self, base_config, obj):
        """Create a new SharedObjectConfig for a given object."""
        shared_object_config = SharedObjectConfig(base_config, self._next_id)
        self._next_id += 1
        try:
            self._shared_objects_config[obj] = shared_object_config
        except TypeError:
            # If the object is unhashable (e.g. a subclass of
            # `AbstractBaseClass` that has not overridden `__hash__`), a
            # `TypeError` will be thrown.  We'll just continue on without shared
            # object support.
            pass
        return shared_object_config

    def __exit__(self, *args, **kwargs):
        if not getattr(self, "_passthrough", False):
            global SHARED_OBJECT_SAVING
            SHARED_OBJECT_SAVING.scope = None


def serialize_keras_class_and_config(
    cls_name, cls_config, obj=None, shared_object_id=None
):
    """Returns the serialization of the class with the given config."""
    base_config = {"class_name": cls_name, "config": cls_config}

    # We call `serialize_keras_class_and_config` for some branches of the load
    # path. In that case, we may already have a shared object ID we'd like to
    # retain.
    if shared_object_id is not None:
        base_config[SHARED_OBJECT_KEY] = shared_object_id

    # If we have an active `SharedObjectSavingScope`, check whether we've
    # already serialized this config. If so, just use that config. This will
    # store an extra ID field in the config, allowing us to re-create the shared
    # object relationship at load time.
    if _shared_object_saving_scope() is not None and obj is not None:
        shared_object_config = _shared_object_saving_scope().get_config(obj)
        if shared_object_config is None:
            return _shared_object_saving_scope().create_config(base_config, obj)
        return shared_object_config

    return base_config


@contextlib.contextmanager
def skip_failed_serialization():
    global _SKIP_FAILED_SERIALIZATION
    prev = _SKIP_FAILED_SERIALIZATION
    try:
        _SKIP_FAILED_SERIALIZATION = True
        yield
    finally:
        _SKIP_FAILED_SERIALIZATION = prev


@keras_export(
    [
        "keras.legacy.saving.serialize_keras_object",
        "keras.utils.legacy.serialize_keras_object",
    ]
)
def serialize_keras_object(instance):
    """Serialize a Keras object into a JSON-compatible representation.

    Calls to `serialize_keras_object` while underneath the
    `SharedObjectSavingScope` context manager will cause any objects re-used
    across multiple layers to be saved with a special shared object ID. This
    allows the network to be re-created properly during deserialization.

    Args:
      instance: The object to serialize.

    Returns:
      A dict-like, JSON-compatible representation of the object's config.
    """

    # _, instance = tf.__internal__.decorator.unwrap(instance)
    instance = inspect.unwrap(instance)
    if instance is None:
        return None

    if hasattr(instance, "get_config"):
        name = object_registration.get_registered_name(instance.__class__)
        try:
            config = instance.get_config()
        except NotImplementedError as e:
            if _SKIP_FAILED_SERIALIZATION:
                return serialize_keras_class_and_config(
                    name, {_LAYER_UNDEFINED_CONFIG_KEY: True}
                )
            raise e
        serialization_config = {}
        for key, item in config.items():
            if isinstance(item, str):
                serialization_config[key] = item
                continue

            # Any object of a different type needs to be converted to string or
            # dict for serialization (e.g. custom functions, custom classes)
            try:
                serialized_item = serialize_keras_object(item)
                if isinstance(serialized_item, dict) and not isinstance(
                    item, dict
                ):
                    serialized_item["__passive_serialization__"] = True
                serialization_config[key] = serialized_item
            except ValueError:
                serialization_config[key] = item

        name = object_registration.get_registered_name(instance.__class__)
        return serialize_keras_class_and_config(
            name, serialization_config, instance
        )
    if hasattr(instance, "__name__"):
        return object_registration.get_registered_name(instance)
    raise ValueError(
        f"Cannot serialize {instance} because it doesn't implement "
        "`get_config()`."
    )


def class_and_config_for_serialized_keras_object(
    config,
    module_objects=None,
    custom_objects=None,
    printable_module_name="object",
):
    """Returns the class name and config for a serialized keras object."""

    if (
        not isinstance(config, dict)
        or "class_name" not in config
        or "config" not in config
    ):
        raise ValueError(
            f"Improper config format for {config}. "
            "Expecting python dict contains `class_name` and `config` as keys"
        )

    class_name = config["class_name"]
    cls = object_registration.get_registered_object(
        class_name, custom_objects, module_objects
    )
    if cls is None:
        raise ValueError(
            f"Unknown {printable_module_name}: '{class_name}'. "
            "Please ensure you are using a `keras.utils.custom_object_scope` "
            "and that this object is included in the scope. See "
            "https://www.tensorflow.org/guide/keras/save_and_serialize"
            "#registering_the_custom_object for details."
        )

    cls_config = config["config"]
    # Check if `cls_config` is a list. If it is a list, return the class and the
    # associated class configs for recursively deserialization. This case will
    # happen on the old version of sequential model (e.g. `keras_version` ==
    # "2.0.6"), which is serialized in a different structure, for example
    # "{'class_name': 'Sequential',
    #   'config': [{'class_name': 'Embedding', 'config': ...}, {}, ...]}".
    if isinstance(cls_config, list):
        return (cls, cls_config)

    deserialized_objects = {}
    for key, item in cls_config.items():
        if key == "name":
            # Assume that the value of 'name' is a string that should not be
            # deserialized as a function. This avoids the corner case where
            # cls_config['name'] has an identical name to a custom function and
            # gets converted into that function.
            deserialized_objects[key] = item
        elif isinstance(item, dict) and "__passive_serialization__" in item:
            deserialized_objects[key] = deserialize_keras_object(
                item,
                module_objects=module_objects,
                custom_objects=custom_objects,
                printable_module_name="config_item",
            )
        # TODO(momernick): Should this also have 'module_objects'?
        elif isinstance(item, str) and inspect.isfunction(
            object_registration.get_registered_object(item, custom_objects)
        ):
            # Handle custom functions here. When saving functions, we only save
            # the function's name as a string. If we find a matching string in
            # the custom objects during deserialization, we convert the string
            # back to the original function.
            # Note that a potential issue is that a string field could have a
            # naming conflict with a custom function name, but this should be a
            # rare case.  This issue does not occur if a string field has a
            # naming conflict with a custom object, since the config of an
            # object will always be a dict.
            deserialized_objects[key] = (
                object_registration.get_registered_object(item, custom_objects)
            )
    for key, item in deserialized_objects.items():
        cls_config[key] = deserialized_objects[key]

    return (cls, cls_config)


@keras_export(
    [
        "keras.legacy.saving.deserialize_keras_object",
        "keras.utils.legacy.deserialize_keras_object",
    ]
)
def deserialize_keras_object(
    identifier,
    module_objects=None,
    custom_objects=None,
    printable_module_name="object",
):
    """Turns the serialized form of a Keras object back into an actual object.

    This function is for mid-level library implementers rather than end users.

    Importantly, this utility requires you to provide the dict of
    `module_objects` to use for looking up the object config; this is not
    populated by default. If you need a deserialization utility that has
    preexisting knowledge of built-in Keras objects, use e.g.
    `keras.layers.deserialize(config)`, `keras.metrics.deserialize(config)`,
    etc.

    Calling `deserialize_keras_object` while underneath the
    `SharedObjectLoadingScope` context manager will cause any already-seen
    shared objects to be returned as-is rather than creating a new object.

    Args:
      identifier: the serialized form of the object.
      module_objects: A dictionary of built-in objects to look the name up in.
        Generally, `module_objects` is provided by midlevel library
        implementers.
      custom_objects: A dictionary of custom objects to look the name up in.
        Generally, `custom_objects` is provided by the end user.
      printable_module_name: A human-readable string representing the type of
        the object. Printed in case of exception.

    Returns:
      The deserialized object.

    Example:

    A mid-level library implementer might want to implement a utility for
    retrieving an object from its config, as such:

    ```python
    def deserialize(config, custom_objects=None):
       return deserialize_keras_object(
         identifier,
         module_objects=globals(),
         custom_objects=custom_objects,
         name="MyObjectType",
       )
    ```

    This is how e.g. `keras.layers.deserialize()` is implemented.
    """

    if identifier is None:
        return None

    if isinstance(identifier, dict):
        # In this case we are dealing with a Keras config dictionary.
        config = identifier
        (cls, cls_config) = class_and_config_for_serialized_keras_object(
            config, module_objects, custom_objects, printable_module_name
        )

        # If this object has already been loaded (i.e. it's shared between
        # multiple objects), return the already-loaded object.
        shared_object_id = config.get(SHARED_OBJECT_KEY)
        shared_object = _shared_object_loading_scope().get(shared_object_id)
        if shared_object is not None:
            return shared_object

        if hasattr(cls, "from_config"):
            arg_spec = inspect.getfullargspec(cls.from_config)
            custom_objects = custom_objects or {}

            if "custom_objects" in arg_spec.args:
                deserialized_obj = cls.from_config(
                    cls_config,
                    custom_objects={
                        **object_registration.GLOBAL_CUSTOM_OBJECTS,
                        **custom_objects,
                    },
                )
            else:
                with object_registration.CustomObjectScope(custom_objects):
                    deserialized_obj = cls.from_config(cls_config)
        else:
            # Then `cls` may be a function returning a class.
            # in this case by convention `config` holds
            # the kwargs of the function.
            custom_objects = custom_objects or {}
            with object_registration.CustomObjectScope(custom_objects):
                deserialized_obj = cls(**cls_config)

        # Add object to shared objects, in case we find it referenced again.
        _shared_object_loading_scope().set(shared_object_id, deserialized_obj)

        return deserialized_obj

    elif isinstance(identifier, str):
        object_name = identifier
        if custom_objects and object_name in custom_objects:
            obj = custom_objects.get(object_name)
        elif (
            object_name
            in object_registration._THREAD_LOCAL_CUSTOM_OBJECTS.__dict__
        ):
            obj = object_registration._THREAD_LOCAL_CUSTOM_OBJECTS.__dict__[
                object_name
            ]
        elif object_name in object_registration._GLOBAL_CUSTOM_OBJECTS:
            obj = object_registration._GLOBAL_CUSTOM_OBJECTS[object_name]
        else:
            obj = module_objects.get(object_name)
            if obj is None:
                raise ValueError(
                    f"Unknown {printable_module_name}: '{object_name}'. "
                    "Please ensure you are using a "
                    "`keras.utils.custom_object_scope` "
                    "and that this object is included in the scope. See "
                    "https://www.tensorflow.org/guide/keras/save_and_serialize"
                    "#registering_the_custom_object for details."
                )

        # Classes passed by name are instantiated with no args, functions are
        # returned as-is.
        if inspect.isclass(obj):
            return obj()
        return obj
    elif inspect.isfunction(identifier):
        # If a function has already been deserialized, return as is.
        return identifier
    else:
        raise ValueError(
            "Could not interpret serialized "
            f"{printable_module_name}: {identifier}"
        )


def validate_config(config):
    """Determines whether config appears to be a valid layer config."""
    return (
        isinstance(config, dict) and _LAYER_UNDEFINED_CONFIG_KEY not in config
    )


def is_default(method):
    """Check if a method is decorated with the `default` wrapper."""
    return getattr(method, "_is_default", False)
