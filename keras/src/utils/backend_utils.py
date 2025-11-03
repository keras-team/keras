import copy
import importlib
import inspect
import os
import sys
import warnings

from keras.src import backend as backend_module
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state


def in_tf_graph():
    if global_state.get_global_attribute("in_tf_graph_scope", False):
        return True

    if "tensorflow" in sys.modules:
        from keras.src.utils.module_utils import tensorflow as tf

        return not tf.executing_eagerly()
    return False


def convert_tf_tensor(outputs, dtype=None):
    if backend_module.backend() != "tensorflow" and not in_tf_graph():
        outputs = backend_module.convert_to_tensor(outputs, dtype=dtype)
    return outputs


class TFGraphScope:
    def __init__(self):
        self._original_value = global_state.get_global_attribute(
            "in_tf_graph_scope", False
        )

    def __enter__(self):
        global_state.set_global_attribute("in_tf_graph_scope", True)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute(
            "in_tf_graph_scope", self._original_value
        )


def in_grain_data_pipeline():
    if "grain" not in sys.modules:
        # Fast path to check if grain is not imported.
        return False

    # We use a lightweight version of `inspect.stack` to detect execution within
    # grain.
    current_frame = inspect.currentframe()
    while current_frame:
        if (
            os.path.join("grain", "_src", "python", "dataset")
            in current_frame.f_code.co_filename
            or os.path.join("grain", "_src", "python", "data_loader")
            in current_frame.f_code.co_filename
        ):
            return True
        current_frame = current_frame.f_back
    return False


class DynamicBackend:
    """A class that can be used to switch from one backend to another.

    Example:

    ```python
    backend = DynamicBackend("tensorflow")
    y = backend.square(tf.constant(...))
    backend.set_backend("jax")
    y = backend.square(jax.numpy.array(...))
    ```

    Args:
        backend: Initial backend to use (string).
    """

    def __init__(self, backend=None):
        self._backend = backend or backend_module.backend()

    def set_backend(self, backend):
        if backend not in ("tensorflow", "jax", "torch", "numpy", "openvino"):
            raise ValueError(
                "Available backends are ('tensorflow', 'jax', 'torch', "
                f"'numpy' and 'openvino'). Received: backend={backend}"
            )
        self._backend = backend

    def reset(self):
        self._backend = backend_module.backend()

    @property
    def name(self):
        return self._backend

    def __getattr__(self, name):
        if self._backend == "tensorflow":
            module = importlib.import_module("keras.src.backend.tensorflow")
            return getattr(module, name)
        if self._backend == "jax":
            module = importlib.import_module("keras.src.backend.jax")
            return getattr(module, name)
        if self._backend == "torch":
            module = importlib.import_module("keras.src.backend.torch")
            return getattr(module, name)
        if self._backend == "numpy":
            if backend_module.backend() == "numpy":
                return getattr(backend_module, name)
            else:
                raise NotImplementedError(
                    "Currently, we cannot dynamically import the numpy backend "
                    "because it would disrupt the namespace of the import."
                )
        if self._backend == "openvino":
            module = importlib.import_module("keras.src.backend.openvino")
            return getattr(module, name)


@keras_export("keras.config.set_backend")
def set_backend(backend):
    """Reload the backend (and the Keras package).

    Example:

    >>> import os
    >>> os.environ["KERAS_BACKEND"] = "tensorflow"
    >>>
    >>> import keras
    >>> from keras import ops
    >>> type(ops.ones(()))
    <class 'tensorflow.python.framework.ops.EagerTensor'>
    >>>
    >>> keras.config.set_backend("jax")
    UserWarning: Using `keras.config.set_backend` is dangerous...
    >>> del keras, ops
    >>>
    >>> import keras
    >>> from keras import ops
    >>> type(ops.ones(()))
    <class 'jaxlib.xla_extension.ArrayImpl'>

    ⚠️ WARNING ⚠️: Using this function is dangerous and should be done
    carefully. Changing the backend will **NOT** convert
    the type of any already-instantiated objects.
    Thus, any layers / tensors / etc. already created will no
    longer be usable without errors. It is strongly recommended **not**
    to keep around **any** Keras-originated objects instances created
    before calling `set_backend()`.

    This includes any function or class instance that uses any Keras
    functionality. All such code needs to be re-executed after calling
    `set_backend()` and re-importing all imported `keras` modules.
    """
    os.environ["KERAS_BACKEND"] = backend
    # Clear module cache.
    loaded_modules = [
        key for key in sys.modules.keys() if key.startswith("keras")
    ]
    for key in loaded_modules:
        del sys.modules[key]
    # Reimport Keras with the new backend (set via KERAS_BACKEND).
    import keras

    # Finally: refresh all imported Keras submodules.
    globs = copy.copy(globals())
    for key, value in globs.items():
        if value.__class__ == keras.__class__:
            if str(value).startswith("<module 'keras."):
                module_name = str(value)
                module_name = module_name[module_name.find("'") + 1 :]
                module_name = module_name[: module_name.find("'")]
                globals()[key] = importlib.import_module(module_name)

    warnings.warn(
        "Using `keras.config.set_backend` is dangerous and should be done "
        "carefully. Already-instantiated objects will not be converted. Thus, "
        "any layers / tensors / etc. already created will no longer be usable "
        "without errors. It is strongly recommended not to keep around any "
        "Keras-originated objects instances created before calling "
        "`set_backend()`. This includes any function or class instance that "
        "uses any Keras functionality. All such code needs to be re-executed "
        "after calling `set_backend()` and re-importing all imported `keras` "
        "modules.",
        stacklevel=2,
    )
