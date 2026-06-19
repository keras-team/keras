"""Backend-agnostic type aliases for static type checkers.

These aliases give libraries built on top of Keras a single name to reference
the active backend's tensor type and the shape sequence convention used
throughout the Keras API. The runtime resolution is the active backend's
concrete tensor class, so `isinstance(x, keras.types.Tensor)` returns `True`
for tensors produced by `keras.ops` on that backend.

Example:

```python
import keras

def add_one(x: keras.types.Tensor) -> keras.types.Tensor:
    return x + 1
```
"""

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
from typing import Union

from keras.src.backend.config import backend as _backend

if TYPE_CHECKING:
    # Static type checkers (mypy, pyright, PyCharm) see the union of every
    # supported backend tensor type so annotations stay valid regardless of
    # which backend the consumer's environment is configured for.
    import jax
    import numpy as np
    import tensorflow as tf
    import torch

    Tensor = Union[jax.Array, tf.Tensor, torch.Tensor, np.ndarray]
else:
    _backend_name = _backend()
    if _backend_name == "tensorflow":
        import tensorflow as _tf

        Tensor = _tf.Tensor
    elif _backend_name == "jax":
        import jax as _jax

        Tensor = _jax.Array
    elif _backend_name == "torch":
        import torch as _torch

        Tensor = _torch.Tensor
    elif _backend_name == "numpy":
        import numpy as _np

        Tensor = _np.ndarray
    elif _backend_name == "openvino":
        from keras.src.backend.openvino.core import OpenVINOKerasTensor

        Tensor = OpenVINOKerasTensor
    else:
        Tensor = Any


Shape = Sequence[Union[int, None]]
"""Shape of a tensor, as a sequence of dimensions.

Static dimensions are represented as `int`; dynamic dimensions are `None`.
"""
