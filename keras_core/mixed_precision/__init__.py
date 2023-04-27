from keras_core.mixed_precision.dtype_policy import DTypePolicy
from keras_core.mixed_precision.dtype_policy import dtype_policy
from keras_core.mixed_precision.dtype_policy import set_dtype_policy
from keras_core.saving import serialization_lib


def resolve_policy(identifier):
    if identifier is None:
        return dtype_policy()
    if isinstance(identifier, DTypePolicy):
        return identifier
    if isinstance(identifier, str):
        return DTypePolicy(identifier)
    if isinstance(identifier, dict):
        return serialization_lib.deserialize_keras_object(identifier)
    raise ValueError(
        "Cannot interpret `dtype` argument. Expected a string "
        f"or an instance of DTypePolicy. Received: dtype={identifier}"
    )
