from keras_core.backend.config import backend

if backend() == "torch":
    # When using the torch backend,
    # torch needs to be imported first, otherwise it will segfault
    # upon import.
    import torch

from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.keras_tensor import any_symbolic_tensors
from keras_core.backend.common.keras_tensor import is_keras_tensor
from keras_core.backend.common.stateless_scope import StatelessScope
from keras_core.backend.common.stateless_scope import get_stateless_scope
from keras_core.backend.common.stateless_scope import in_stateless_scope
from keras_core.backend.common.variables import AutocastScope
from keras_core.backend.common.variables import get_autocast_scope
from keras_core.backend.common.variables import is_float_dtype
from keras_core.backend.common.variables import is_int_dtype
from keras_core.backend.common.variables import standardize_dtype
from keras_core.backend.common.variables import standardize_shape
from keras_core.backend.config import epsilon
from keras_core.backend.config import floatx
from keras_core.backend.config import image_data_format
from keras_core.backend.config import set_epsilon
from keras_core.backend.config import set_floatx
from keras_core.backend.config import set_image_data_format
from keras_core.backend.config import standardize_data_format
from keras_core.utils.io_utils import print_msg

# Import backend functions.
if backend() == "tensorflow":
    print_msg("Using TensorFlow backend")
    from keras_core.backend.tensorflow import *  # noqa: F403
elif backend() == "jax":
    print_msg("Using JAX backend.")
    from keras_core.backend.jax import *  # noqa: F403
elif backend() == "torch":
    print_msg("Using PyTorch backend.")
    from keras_core.backend.torch import *  # noqa: F403
elif backend() == "numpy":
    print_msg(
        "Using NumPy backend.\nThe NumPy backend does not support "
        "training. It should only be used for inference, evaluation, "
        "and debugging."
    )
    from keras_core.backend.numpy import *  # noqa: F403
else:
    raise ValueError(f"Unable to import backend : {backend()}")
