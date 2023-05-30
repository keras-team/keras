import inspect

from keras_core.api_export import keras_core_export
from keras_core.regularizers.regularizers import L1
from keras_core.regularizers.regularizers import L1L2
from keras_core.regularizers.regularizers import L2
from keras_core.regularizers.regularizers import OrthogonalRegularizer
from keras_core.regularizers.regularizers import Regularizer
from keras_core.saving import serialization_lib
from keras_core.utils.naming import to_snake_case

ALL_OBJECTS = {
    Regularizer,
    L1,
    L2,
    L1L2,
    OrthogonalRegularizer,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)


@keras_core_export("keras_core.regularizers.serialize")
def serialize(initializer):
    return serialization_lib.serialize_keras_object(initializer)


@keras_core_export("keras_core.regularizers.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras regularizer object via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_core_export("keras_core.regularizers.get")
def get(identifier):
    """Retrieve a Keras regularizer object via an identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        obj = deserialize(config)
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(
            f"Could not interpret regularizer identifier: {identifier}"
        )
