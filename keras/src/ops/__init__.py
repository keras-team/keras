# from keras.src.ops.numpy import Matmul, matmul
# from keras.src.ops.numpy import Add, add
# from keras.src.ops.numpy import Multiply, multiply

from keras.src.ops import image
from keras.src.ops import operation_utils
from keras.src.ops.core import *  # noqa: F403
from keras.src.ops.linalg import *  # noqa: F403
from keras.src.ops.math import *  # noqa: F403
from keras.src.ops.nn import *  # noqa: F403
from keras.src.ops.numpy import *  # noqa: F403

def __dir__():
    return sorted(
        list(globals().keys())
        + ["cast", "cond", "is_tensor", "name_scope", "random"]
    )


def __getattr__(name):
    if name in {"cast", "cond", "is_tensor", "name_scope", "random"}:
        import keras.src.backend as backend

        return getattr(backend, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
