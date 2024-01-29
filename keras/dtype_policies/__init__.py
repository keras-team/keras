from keras import backend
from keras.dtype_policies import dtype_policy
from keras.saving import serialization_lib


def get(identifier):
    if identifier is None:
        return dtype_policy.dtype_policy()
    if isinstance(identifier, dtype_policy.DTypePolicy):
        return identifier
    if isinstance(identifier, dict):
        return serialization_lib.deserialize_keras_object(identifier)
    if isinstance(identifier, str):
        return dtype_policy.DTypePolicy(identifier)
    try:
        return dtype_policy.DTypePolicy(backend.standardize_dtype(identifier))
    except:
        raise ValueError(
            "Cannot interpret `dtype` argument. Expected a string "
            f"or an instance of DTypePolicy. Received: dtype={identifier}"
        )
