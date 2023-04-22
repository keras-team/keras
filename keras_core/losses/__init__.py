from keras_core.api_export import keras_core_export
from keras_core.losses.loss import Loss
from keras_core.losses.losses import LossFunctionWrapper
from keras_core.losses.losses import MeanSquaredError


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
            function/class or loss configuration dictionary or a loss function or a
            loss class instance.

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
