from keras.api_export import keras_export
from keras.optimizers.adadelta import Adadelta
from keras.optimizers.adafactor import Adafactor
from keras.optimizers.adagrad import Adagrad
from keras.optimizers.adam import Adam
from keras.optimizers.adamax import Adamax
from keras.optimizers.adamw import AdamW
from keras.optimizers.ftrl import Ftrl
from keras.optimizers.lion import Lion
from keras.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.optimizers.nadam import Nadam
from keras.optimizers.optimizer import Optimizer
from keras.optimizers.rmsprop import RMSprop
from keras.optimizers.sgd import SGD
from keras.saving import serialization_lib

ALL_OBJECTS = {
    Optimizer,
    Adam,
    SGD,
    RMSprop,
    Adadelta,
    AdamW,
    Adagrad,
    Adamax,
    Adafactor,
    Nadam,
    Ftrl,
    Lion,
    LossScaleOptimizer,
}
ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}


@keras_export("keras.optimizers.serialize")
def serialize(optimizer):
    """Returns the optimizer configuration as a Python dict.

    Args:
        optimizer: An `Optimizer` instance to serialize.

    Returns:
        Python dict which contains the configuration of the optimizer.
    """
    return serialization_lib.serialize_keras_object(optimizer)


@keras_export("keras.optimizers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Keras optimizer object via its configuration.

    Args:
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras Optimizer instance.
    """
    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()

    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.optimizers.get")
def get(identifier):
    """Retrieves a Keras Optimizer instance.

    Args:
        identifier: Optimizer identifier, one of:
            - String: name of an optimizer
            - Dictionary: configuration dictionary.
            - Keras Optimizer instance (it will be returned unchanged).

    Returns:
        A Keras Optimizer instance.
    """
    if identifier is None:
        return None
    elif isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"class_name": identifier, "config": {}}
        obj = deserialize(config)
    else:
        obj = identifier

    if isinstance(obj, Optimizer):
        return obj
    raise ValueError(f"Could not interpret optimizer identifier: {identifier}")


# We will add this temporarily so that tensorflow packages that depend on
# estimators will continue to import (there are a large number). Note that
# Keras 3 will not work with the estimators API.
@keras_export(
    [
        "keras.optimizers.legacy.Adagrad",
        "keras.optimizers.legacy.Adam",
        "keras.optimizers.legacy.Ftrl",
        "keras.optimizers.legacy.RMSprop",
        "keras.optimizers.legacy.SGD",
        "keras.optimizers.legacy.Optimizer",
    ]
)
class LegacyOptimizerWarning:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "`keras.optimizers.legacy` is not supported in Keras 3. When using "
            "`tf.keras`, to continue using a `tf.keras.optimizers.legacy` "
            "optimizer, you can install the `tf_keras` package (Keras 2) and "
            "set the environment variable `TF_USE_LEGACY_KERAS=True` to "
            "configure TensorFlow to use `tf_keras` when accessing `tf.keras`."
        )
