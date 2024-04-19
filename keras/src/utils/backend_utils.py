import copy
import importlib
import os
import sys

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
        self._backend = backend

    def reset(self):
        self._backend = backend_module.backend()

    def __getattr__(self, name):
        if self._backend == "tensorflow":
            from keras.src.backend import tensorflow as tf_backend

            return getattr(tf_backend, name)
        if self._backend == "jax":
            from keras.src.backend import jax as jax_backend

            return getattr(jax_backend, name)
        if self._backend == "torch":
            from keras.src.backend import torch as torch_backend

            return getattr(torch_backend, name)
        if self._backend == "numpy":
            # TODO (ariG23498):
            # The import `from keras.src.backend import numpy as numpy_backend`
            # is not working. This is a temporary fix.
            # The import is redirected to `keras.backend.numpy.numpy.py`
            from keras.src import backend as numpy_backend

            return getattr(numpy_backend, name)


@keras_export("keras.config.set_backend")
def set_backend(backend):
    """Reload the backend (and the Keras package).

    Example:

    ```python
    keras.config.set_backend("jax")
    ```

    ⚠️ WARNING ⚠️: Using this function is dangerous and should be done
    carefully. Changing the backend will **NOT** convert
    the type of any already-instantiated objects.
    Thus, any layers / tensors / etc. already created will no
    longer be usable without errors. It is strongly recommended **not**
    to keep around **any** Keras-originated objects instances created
    before calling `set_backend()`.

    This includes any function or class instance that uses any Keras
    functionality. All such code needs to be re-executed after calling
    `set_backend()`.
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
