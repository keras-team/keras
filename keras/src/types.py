"""Backend-agnostic type aliases for static type checkers.

These aliases give libraries built on top of Keras a single name to reference
the active backend's tensor type and the shape sequence convention used
throughout the Keras API.

Example:

```python
import keras

def add_one(x: keras.types.Tensor) -> keras.types.Tensor:
    return x + 1
```

Static type checkers (mypy, pyright, PyCharm) see `Tensor` as the union of
every supported backend tensor type, so the annotation type-checks regardless
of which backend the consumer's environment is configured for. At runtime,
`isinstance(x, keras.types.Tensor)` returns `True` for tensors produced by
`keras.ops` on the active backend.
"""

from typing import TYPE_CHECKING
from typing import Sequence
from typing import Union

from keras.src.api_export import keras_export
from keras.src.backend.config import backend as _backend


def _resolve_backend_tensor_type():
    name = _backend()
    if name == "tensorflow":
        import tensorflow as tf

        return tf.Tensor
    if name == "jax":
        import jax

        return jax.Array
    if name == "torch":
        import torch

        return torch.Tensor
    if name == "numpy":
        import numpy as np

        return np.ndarray
    if name == "openvino":
        from keras.src.backend.openvino.core import OpenVINOKerasTensor

        return OpenVINOKerasTensor
    return object


_BACKEND_TENSOR_TYPE = _resolve_backend_tensor_type()


class _TensorMeta(type):
    """Metaclass routing `isinstance` / `issubclass` to the backend type."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, _BACKEND_TENSOR_TYPE)

    def __subclasscheck__(cls, subclass):
        try:
            return issubclass(subclass, _BACKEND_TENSOR_TYPE)
        except TypeError:
            return False


if TYPE_CHECKING:
    import jax
    import numpy as np
    import tensorflow as tf
    import torch

    Tensor = Union[jax.Array, tf.Tensor, torch.Tensor, np.ndarray]
    Shape = Sequence[Union[int, None]]
else:

    @keras_export("keras.types.Tensor")
    class Tensor(metaclass=_TensorMeta):
        """Backend-agnostic tensor type.

        Use this as a type annotation in code that should work across
        backends. Static type checkers see it as the union of every
        supported backend's tensor type
        (`jax.Array | tf.Tensor | torch.Tensor | np.ndarray`). At runtime,
        `isinstance(x, keras.types.Tensor)` is `True` for tensors produced
        by `keras.ops` on the active backend.

        This class cannot be instantiated; it exists only as a type alias.
        """

        def __new__(cls, *args, **kwargs):
            raise TypeError(
                "`keras.types.Tensor` is a backend-agnostic type alias for "
                f"`{_BACKEND_TENSOR_TYPE.__module__}."
                f"{_BACKEND_TENSOR_TYPE.__name__}` on the active backend and "
                "cannot be instantiated directly."
            )

    @keras_export("keras.types.Shape")
    class Shape:
        """Shape of a tensor, as a sequence of dimensions.

        Static dimensions are represented as `int`; dynamic dimensions are
        `None`. Static type checkers see this as `Sequence[int | None]`.

        This class cannot be instantiated; it exists only as a type alias.
        """

        def __new__(cls, *args, **kwargs):
            raise TypeError(
                "`keras.types.Shape` is a type alias and cannot be "
                "instantiated."
            )
