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
from keras.src.backend.common.variables import Variable
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
    from keras.src.backend.tensorflow.core import Variable as BackendVariable
elif backend() == "jax":
    from keras.src.backend.jax import *  # noqa: F403
    from keras.src.backend.jax.core import Variable as BackendVariable
elif backend() == "torch":
    from keras.src.backend.torch import *  # noqa: F403
    from keras.src.backend.torch.core import Variable as BackendVariable

    distribution_lib = None
elif backend() == "numpy":
    from keras.src.backend.numpy import *  # noqa: F403
    from keras.src.backend.numpy.core import Variable as BackendVariable

    distribution_lib = None
elif backend() == "openvino":
    from keras.src.backend.openvino import *  # noqa: F403
    from keras.src.backend.openvino.core import Variable as BackendVariable

    distribution_lib = None
else:
    raise ValueError(f"Unable to import backend : {backend()}")


@keras_export("keras.Variable")
class Variable(BackendVariable):  # noqa: F811
    pass


backend_name_scope = name_scope  # noqa: F405


@keras_export("keras.name_scope")
class name_scope(backend_name_scope):
    pass


@keras_export("keras.device")
def device(device_name):
    """Context manager for backend-agnostic device placement.

    Use this context manager to control which device tensors are allocated on across
    all backends (TensorFlow, JAX, PyTorch). This is useful for memory management,
    data preprocessing, and multi-device setups.

    Args:
        device_name: String specifying the device in format `"device_type:device_index"`.
            For example: `"cpu:0"`, `"gpu:0"`, `"gpu:1"`.
            For PyTorch backend, `"gpu"` is automatically converted to `"cuda"`.

    Example:

    Basic usage with CPU and GPU:

    ```python
    import keras
    import numpy as np

    # Allocate tensors on CPU
    with keras.device("cpu:0"):
        cpu_tensor = keras.ops.ones((2, 2))

    # Allocate tensors on GPU (if available)
    with keras.device("gpu:0"):
        gpu_tensor = keras.ops.ones((2, 2))
    ```

    Practical example with CPU preprocessing and GPU training:

    ```python
    import keras
    import numpy as np

    # Create dummy data and model
    x_raw = np.random.rand(128, 784)
    y_train = np.random.randint(0, 10, size=(128,))
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    # Preprocess data on CPU
    with keras.device("cpu:0"):
        x_processed = keras.ops.cast(x_raw, "float32")

    # Train on GPU (if available)
    with keras.device("gpu:0"):
        model.fit(x_processed, y_train, epochs=2)
    ```

    Use cases:

    - **Memory management**: Keep large tensors on CPU to save GPU memory
    - **Data preprocessing**: Process data on CPU before moving to GPU for training
    - **Multi-GPU setups**: Explicitly control which GPU receives which tensors

    Device naming conventions:

    - `"cpu:0"` - First CPU
    - `"gpu:0"` - First GPU (works across all backends)
    - `"gpu:1"` - Second GPU

    Note: For distributed training across multiple devices, see the
    [distributed training guides](https://keras.io/guides/distributed_training/).
    """
    return device_scope(device_name)  # noqa: F405
