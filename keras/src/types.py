"""Backend-agnostic type aliases for Keras.

This module provides type aliases that can be used for type hints
in code that works with any Keras backend. These types are designed
to work with static type checkers (mypy, PyCharm) while not
requiring any specific backend to be installed at runtime.

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
        """Backend-agnostic tensor type for use in type annotations.

        Under type checking, this resolves to a ``Union`` of all
        backend tensor types:

        - ``keras.KerasTensor`` (symbolic tensor)
        - ``numpy.ndarray``
        - ``jax.Array`` (if JAX is installed)
        - ``tensorflow.Tensor`` (if TensorFlow is installed)
        - ``torch.Tensor`` (if PyTorch is installed)

        At runtime this is a placeholder class. Do not instantiate
        it or use it with ``isinstance()``.

        Example:

        ```python
        from keras.types import Tensor

        def my_function(x: Tensor) -> Tensor:
            return x
        ```
        """

        pass

    @keras_export("keras.types.Shape")
    class Shape:
        """Backend-agnostic shape type for use in type annotations.

        Under type checking, this resolves to
        ``Tuple[Optional[int], ...]``, a variable-length tuple where
        each dimension is either an ``int`` (known) or ``None``
        (unknown / dynamic).

        At runtime this is a placeholder class. Do not instantiate
        it or use it with ``isinstance()``.

        Example:

        ```python
        from keras.types import Shape

        def reshape(x, shape: Shape):
            return keras.ops.reshape(x, shape)
        ```
        """

        pass

    @keras_export("keras.types.DType")
    class DType:
        """Backend-agnostic dtype type for use in type annotations.

        Under type checking, this resolves to ``str``. Keras
        standardizes all dtype representations to strings
        (e.g. ``"float32"``, ``"int64"``, ``"bool"``).

        At runtime this is a placeholder class. Do not instantiate
        it or use it with ``isinstance()``.

        Example:

        ```python
        from keras.types import DType

        def cast(x, dtype: DType):
            return keras.ops.cast(x, dtype)
        ```
        """

        pass
