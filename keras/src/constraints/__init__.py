import inspect

from keras.src.api_export import keras_export
from keras.src.constraints.constraints import Constraint
from keras.src.constraints.constraints import MaxNorm
from keras.src.constraints.constraints import MinMaxNorm
from keras.src.constraints.constraints import NonNeg
from keras.src.constraints.constraints import UnitNorm
from keras.src.saving import serialization_lib
from keras.src.utils.naming import to_snake_case

ALL_OBJECTS = {
    Constraint,
    MaxNorm,
    MinMaxNorm,
    NonNeg,
    UnitNorm,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)


@keras_export("keras.constraints.serialize")
def serialize(constraint):
    return serialization_lib.serialize_keras_object(constraint)


@keras_export("keras.constraints.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras constraint object via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.constraints.get")
def get(identifier):
    """Retrieve a Keras constraint object via an identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(
            f"Could not interpret constraint identifier: {identifier}"
        )
