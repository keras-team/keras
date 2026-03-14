"""Backend-agnostic type aliases for Keras.

This module provides type aliases that can be used for type hints
in code that works with any Keras backend. These types are designed
to work with static type checkers (mypy, PyCharm) while not
requiring any specific backend to be installed at runtime.

These are **annotation-only** helpers. Under ``TYPE_CHECKING`` they
resolve to concrete types (e.g. ``Union`` of backend tensors);
at runtime they are lightweight markers used solely in type
annotations and cannot be instantiated.

Example usage:

```python
from keras.types import Tensor, Shape, DType


def my_function(x: Tensor, shape: Shape) -> Tensor:
    ...
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from keras.src.api_export import keras_export

if TYPE_CHECKING:
    from typing import Any
    from typing import Optional
    from typing import Tuple
    from typing import Union

    import numpy as np

    from keras.src.backend.common.keras_tensor import KerasTensor

    try:
        from jax import Array as _JaxArray
    except ImportError:
        _JaxArray = Any

    try:
        from tensorflow import Tensor as _TfTensor
    except ImportError:
        _TfTensor = Any

    try:
        from torch import Tensor as _TorchTensor
    except ImportError:
        _TorchTensor = Any

    Tensor = Union[KerasTensor, np.ndarray, _JaxArray, _TfTensor, _TorchTensor]

    Shape = Tuple[Optional[int], ...]

    DType = str

else:

    @keras_export("keras.types.Tensor")
    class Tensor:
        """Annotation-only type representing any backend tensor.

        Use this in type hints to denote tensor arguments or return
        values that work across all Keras backends. Under static
        type checking, ``Tensor`` resolves to a ``Union`` of:

        - ``keras.KerasTensor`` (symbolic tensor)
        - ``numpy.ndarray``
        - ``jax.Array``
        - ``tensorflow.Tensor``
        - ``torch.Tensor``

        This type exists **only for annotations**. It cannot be
        instantiated or used with ``isinstance()``. Keras
        operations accept native backend tensors directly.

        Example:

        ```python
        from keras.types import Tensor

        def my_function(x: Tensor) -> Tensor:
            return x
        ```
        """

        def __init__(self):
            raise TypeError(
                "keras.types.Tensor is an annotation-only type and "
                "cannot be instantiated. Use it only in type hints."
            )

        def __init_subclass__(cls, **kwargs):
            raise TypeError("keras.types.Tensor cannot be subclassed.")

    @keras_export("keras.types.Shape")
    class Shape:
        """Annotation-only type representing a tensor shape.

        Use this in type hints to denote shape arguments or return
        values. Under static type checking, ``Shape`` resolves to
        ``Tuple[Optional[int], ...]``.

        This type exists **only for annotations**. Keras accepts
        and returns plain ``tuple`` objects for shapes.

        Example:

        ```python
        from keras.types import Shape

        def reshape(x, shape: Shape):
            return keras.ops.reshape(x, shape)
        ```
        """

        def __init__(self):
            raise TypeError(
                "keras.types.Shape is an annotation-only type and "
                "cannot be instantiated. Use a plain tuple instead."
            )

        def __init_subclass__(cls, **kwargs):
            raise TypeError("keras.types.Shape cannot be subclassed.")

    @keras_export("keras.types.DType")
    class DType:
        """Annotation-only type representing a data type.

        Use this in type hints to denote dtype arguments or return
        values. Under static type checking, ``DType`` resolves to
        ``str``.

        This type exists **only for annotations**. Keras uses
        plain strings for dtypes (e.g. ``"float32"``,
        ``"int64"``).

        Example:

        ```python
        from keras.types import DType

        def cast(x, dtype: DType):
            return keras.ops.cast(x, dtype)
        ```
        """

        def __init__(self):
            raise TypeError(
                "keras.types.DType is an annotation-only type and "
                "cannot be instantiated. Use a plain string instead."
            )

        def __init_subclass__(cls, **kwargs):
            raise TypeError("keras.types.DType cannot be subclassed.")
