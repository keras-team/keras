import inspect

from keras.src.api_export import keras_export
from keras.src.initializers.constant_initializers import Constant
from keras.src.initializers.constant_initializers import Identity
from keras.src.initializers.constant_initializers import Ones
from keras.src.initializers.constant_initializers import Zeros
from keras.src.initializers.initializer import Initializer
from keras.src.initializers.random_initializers import GlorotNormal
from keras.src.initializers.random_initializers import GlorotUniform
from keras.src.initializers.random_initializers import HeNormal
from keras.src.initializers.random_initializers import HeUniform
from keras.src.initializers.random_initializers import LecunNormal
from keras.src.initializers.random_initializers import LecunUniform
from keras.src.initializers.random_initializers import OrthogonalInitializer
from keras.src.initializers.random_initializers import RandomNormal
from keras.src.initializers.random_initializers import RandomUniform
from keras.src.initializers.random_initializers import TruncatedNormal
from keras.src.initializers.random_initializers import VarianceScaling
from keras.src.saving import serialization_lib
from keras.src.utils.naming import to_snake_case

ALL_OBJECTS = {
    Initializer,
    Constant,
    Identity,
    Ones,
    Zeros,
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
    RandomNormal,
    TruncatedNormal,
    RandomUniform,
    VarianceScaling,
    OrthogonalInitializer,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)
# Aliases
ALL_OBJECTS_DICT.update(
    {
        "uniform": RandomUniform,
        "normal": RandomNormal,
        "orthogonal": OrthogonalInitializer,
        "one": Ones,
        "zero": Zeros,
    }
)


@keras_export("keras.initializers.serialize")
def serialize(initializer):
    """Returns the initializer configuration as a Python dict."""
    return serialization_lib.serialize_keras_object(initializer)


@keras_export("keras.initializers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Keras initializer object via its configuration."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.initializers.get")
def get(identifier):
    """Retrieves a Keras initializer object via an identifier.

    The `identifier` may be the string name of a initializers function or class
    (case-sensitively).

    >>> identifier = 'Ones'
    >>> keras.initializers.deserialize(identifier)
    <...keras.initializers.initializers.Ones...>

    You can also specify `config` of the initializer to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note that
    the `class_name` must map to a `Initializer` class.

    >>> cfg = {'class_name': 'Ones', 'config': {}}
    >>> keras.initializers.deserialize(cfg)
    <...keras.initializers.initializers.Ones...>

    In the case that the `identifier` is a class, this method will return a new
    instance of the class by its constructor.

    Args:
        identifier: String or dict that contains the initializer name or
            configurations.

    Returns:
        Initializer instance base on the input identifier.
    """
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
            f"Could not interpret initializer identifier: {identifier}"
        )
