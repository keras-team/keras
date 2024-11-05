from keras.src.backend.config import backend

if backend() == "torch":
    # When using the torch backend,
    # torch needs to be imported first, otherwise it will segfault
    # upon import.
    import torch

from keras.src.api_export import keras_export
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.keras_tensor import any_symbolic_tensors
from keras.src.backend.common.keras_tensor import is_keras_tensor
from keras.src.backend.common.masking import get_keras_mask
from keras.src.backend.common.masking import set_keras_mask
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.common.symbolic_scope import in_symbolic_scope
from keras.src.backend.common.variables import AutocastScope
from keras.src.backend.common.variables import get_autocast_scope
from keras.src.backend.common.variables import is_float_dtype
from keras.src.backend.common.variables import is_int_dtype
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.common.variables import standardize_shape
from keras.src.backend.config import epsilon
from keras.src.backend.config import floatx
from keras.src.backend.config import image_data_format
from keras.src.backend.config import set_epsilon
from keras.src.backend.config import set_floatx
from keras.src.backend.config import set_image_data_format
from keras.src.backend.config import standardize_data_format

# Import backend functions.
if backend() == "tensorflow":
    from keras.src.backend.tensorflow import *  # noqa: F403
elif backend() == "jax":
    from keras.src.backend.jax import *  # noqa: F403
elif backend() == "torch":
    from keras.src.backend.torch import *  # noqa: F403

    distribution_lib = None
elif backend() == "numpy":
    from keras.src.backend.numpy import *  # noqa: F403

    distribution_lib = None
else:
    raise ValueError(f"Unable to import backend : {backend()}")


BackendVariable = Variable  # noqa: F405


@keras_export("keras.Variable")
class Variable(BackendVariable):
    pass


backend_name_scope = name_scope  # noqa: F405


@keras_export("keras.name_scope")
class name_scope(backend_name_scope):
    pass


@keras_export("keras.device")
def device(device_name):
    return device_scope(device_name)  # noqa: F405
