import sys

from keras import backend as backend_module
from keras.backend.common import global_state


def in_tf_graph():
    if global_state.get_global_attribute("in_tf_graph_scope", False):
        return True

    if "tensorflow" in sys.modules:
        from keras.utils.module_utils import tensorflow as tf

        return not tf.executing_eagerly()
    return False


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

    Usage:

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
            from keras.backend import tensorflow as tf_backend

            return getattr(tf_backend, name)
        if self._backend == "jax":
            from keras.backend import jax as jax_backend

            return getattr(jax_backend, name)
        if self._backend == "torch":
            from keras.backend import torch as torch_backend

            return getattr(torch_backend, name)
        if self._backend == "numpy":
            # TODO (ariG23498):
            # The import `from keras.backend import numpy as numpy_backend`
            # is not working. This is a temporary fix.
            # The import is redirected to `keras.backend.numpy.numpy.py`
            from keras import backend as numpy_backend

            return getattr(numpy_backend, name)
