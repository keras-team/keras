import types

from keras.activations.activations import elu
from keras.activations.activations import exponential
from keras.activations.activations import gelu
from keras.activations.activations import hard_sigmoid
from keras.activations.activations import hard_silu
from keras.activations.activations import leaky_relu
from keras.activations.activations import linear
from keras.activations.activations import log_softmax
from keras.activations.activations import mish
from keras.activations.activations import relu
from keras.activations.activations import relu6
from keras.activations.activations import selu
from keras.activations.activations import sigmoid
from keras.activations.activations import silu
from keras.activations.activations import softmax
from keras.activations.activations import softplus
from keras.activations.activations import softsign
from keras.activations.activations import tanh
from keras.api_export import keras_export
from keras.saving import object_registration
from keras.saving import serialization_lib

ALL_OBJECTS = {
    relu,
    leaky_relu,
    relu6,
    softmax,
    elu,
    selu,
    softplus,
    softsign,
    silu,
    gelu,
    tanh,
    sigmoid,
    exponential,
    hard_sigmoid,
    hard_silu,
    linear,
    mish,
    log_softmax,
}

ALL_OBJECTS_DICT = {fn.__name__: fn for fn in ALL_OBJECTS}
# Additional aliases
ALL_OBJECTS_DICT["swish"] = silu
ALL_OBJECTS_DICT["hard_swish"] = hard_silu


@keras_export("keras.activations.serialize")
def serialize(activation):
    fn_config = serialization_lib.serialize_keras_object(activation)
    if "config" not in fn_config:
        raise ValueError(
            f"Unknown activation function '{activation}' cannot be "
            "serialized due to invalid function name. Make sure to use "
            "an activation name that matches the references defined in "
            "activations.py or use "
            "`@keras.saving.register_keras_serializable()`"
            "to register any custom activations. "
            f"config={fn_config}"
        )
    if not isinstance(activation, types.FunctionType):
        # Case for additional custom activations represented by objects
        return fn_config
    if (
        isinstance(fn_config["config"], str)
        and fn_config["config"] not in globals()
    ):
        # Case for custom activation functions from external activations modules
        fn_config["config"] = object_registration.get_registered_name(
            activation
        )
        return fn_config
    # Case for keras.activations builtins (simply return name)
    return fn_config["config"]


@keras_export("keras.activations.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras activation function via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_export("keras.activations.get")
def get(identifier):
    """Retrieve a Keras activation function via an identifier."""
    if identifier is None:
        return linear
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier
    if callable(obj):
        return obj
    raise ValueError(
        f"Could not interpret activation function identifier: {identifier}"
    )
