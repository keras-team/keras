from keras.src import backend
from keras.src.dtype_policies import dtype_policy
from keras.src.dtype_policies.dtype_policy import QUANTIZATION_MODES
from keras.src.dtype_policies.dtype_policy import FloatDTypePolicy
from keras.src.dtype_policies.dtype_policy import QuantizedDTypePolicy
from keras.src.dtype_policies.dtype_policy import QuantizedFloat8DTypePolicy


def get(identifier):
    from keras.src.dtype_policies.dtype_policy import (
        _get_quantized_dtype_policy_by_str,
    )
    from keras.src.saving import serialization_lib

    if identifier is None:
        return dtype_policy.dtype_policy()
    if isinstance(identifier, (FloatDTypePolicy, QuantizedDTypePolicy)):
        return identifier
    if isinstance(identifier, dict):
        return serialization_lib.deserialize_keras_object(identifier)
    if isinstance(identifier, str):
        if identifier.startswith(QUANTIZATION_MODES):
            return _get_quantized_dtype_policy_by_str(identifier)
        else:
            return FloatDTypePolicy(identifier)
    try:
        return FloatDTypePolicy(backend.standardize_dtype(identifier))
    except:
        raise ValueError(
            "Cannot interpret `dtype` argument. Expected a string "
            f"or an instance of DTypePolicy. Received: dtype={identifier}"
        )
