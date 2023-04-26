from keras_core.api_export import keras_core_export
from keras_core.losses.loss import Loss
from keras_core.losses.losses import CategoricalHinge
from keras_core.losses.losses import Hinge
from keras_core.losses.losses import LossFunctionWrapper
from keras_core.losses.losses import MeanAbsoluteError
from keras_core.losses.losses import MeanAbsolutePercentageError
from keras_core.losses.losses import MeanSquaredError
from keras_core.losses.losses import MeanSquaredLogarithmicError
from keras_core.losses.losses import SquaredHinge
from keras_core.losses.losses import categorical_hinge
from keras_core.losses.losses import hinge
from keras_core.losses.losses import mean_absolute_error
from keras_core.losses.losses import mean_absolute_percentage_error
from keras_core.losses.losses import mean_squared_error
from keras_core.losses.losses import mean_squared_logarithmic_error
from keras_core.losses.losses import squared_hinge
from keras_core.saving import serialization_lib

ALL_OBJECTS = {
    Loss,
    LossFunctionWrapper,
    MeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredLogarithmicError,
    Hinge,
    SquaredHinge,
    CategoricalHinge,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    hinge,
    squared_hinge,
    categorical_hinge,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update({
    "mae": mean_absolute_error,
    "MAE": mean_absolute_error,
    "mse": mean_squared_error,
    "MSE": mean_squared_error,
})


@keras_core_export("keras_core.losses.serialize")
def serialize(loss):
    """Serializes loss function or `Loss` instance.

    Args:
        loss: A Keras `Loss` instance or a loss function.

    Returns:
        Loss configuration dictionary.
    """
    return serialization_lib.serialize_keras_object(loss)


@keras_core_export("keras_core.losses.deserialize")
def deserialize(name, custom_objects=None):
    """Deserializes a serialized loss class/function instance.

    Args:
        name: Loss configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
          objects (classes and functions) to be considered during
          deserialization.

    Returns:
        A Keras `Loss` instance or a loss function.
    """
    return serialization_lib.deserialize_keras_object(
        name,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_core_export("keras_core.losses.get")
def get(identifier):
    """Retrieves a Keras loss as a `function`/`Loss` class instance.

    The `identifier` may be the string name of a loss function or `Loss` class.

    >>> loss = losses.get("categorical_crossentropy")
    >>> type(loss)
    <class 'function'>
    >>> loss = losses.get("CategoricalCrossentropy")
    >>> type(loss)
    <class '...CategoricalCrossentropy'>

    You can also specify `config` of the loss to this function by passing dict
    containing `class_name` and `config` as an identifier. Also note that the
    `class_name` must map to a `Loss` class

    >>> identifier = {"class_name": "CategoricalCrossentropy",
    ...               "config": {"from_logits": True}}
    >>> loss = losses.get(identifier)
    >>> type(loss)
    <class '...CategoricalCrossentropy'>

    Args:
        identifier: A loss identifier. One of None or string name of a loss
            function/class or loss configuration dictionary or a loss function
            or a loss class instance.

    Returns:
        A Keras loss as a `function`/ `Loss` class instance.
    """
    if identifier is None:
        return None
    if isinstance(identifier, str):
        identifier = str(identifier)
        return deserialize(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if callable(identifier):
        return identifier
    raise ValueError(
        f"Could not interpret loss function identifier: {identifier}"
    )
