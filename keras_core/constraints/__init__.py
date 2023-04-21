import inspect

from keras_core.api_export import keras_core_export
from keras_core.constraints.constraints import Constraint
from keras_core.constraints.constraints import MaxNorm
from keras_core.constraints.constraints import MinMaxNorm
from keras_core.constraints.constraints import NonNeg
from keras_core.constraints.constraints import UnitNorm
from keras_core.saving import serialization_lib
from keras_core.utils.naming import to_snake_case

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


@keras_core_export("keras_core.constraints.serialize")
def serialize(constraint):
    return serialization_lib.serialize_keras_object(constraint)


@keras_core_export("keras_core.initializers.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras constraint object via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_core_export("keras_core.constraints.get")
def get(identifier):
    """Retrieve a Keras constraint object via an identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return deserialize(config)
    elif callable(identifier):
        if inspect.isclass(identifier):
            identifier = identifier()
        return identifier
    else:
        raise ValueError(
            f"Could not interpret constraint object identifier: {identifier}"
        )
