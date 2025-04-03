# Copyright 2024 The etils Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numpy utils.

Attributes:
  tau: The circle constant (2 * pi). (https://tauday.com/)
"""

from __future__ import annotations

import sys
import typing
from typing import Any, Optional, TypeVar

from etils import epy
import numpy as np

if typing.TYPE_CHECKING:
  from etils.enp.typing import Array

_T = TypeVar('_T')

# TODO(pytype): Ideally should use `-> Literal[np]:` but Python does not
# support this: https://github.com/python/typing/issues/1039
# Thankfully, pytype correctly auto-infer `np` when returned by `get_xnp`
NpModule = Any

# Mirror math.tau (PEP 628). See https://tauday.com/
tau = 2 * np.pi

# When `strict=False` (in `get_xnp`, `is_array`,...), those types are also
# accepted:
_ARRAY_LIKE_TYPES = (int, bool, float, list, tuple)

# During the class construction, pytype fails because of name conflict between
# the `np` `@property` and the module.
_np = np


class _LazyArrayMeta(type):

  def __instancecheck__(cls, obj) -> bool:
    return lazy.is_array(obj)


class _LazyImporter:
  """Lazy import module.

  Help to write code seamlessly working with np, Jax and TF.
  Because libs are lazily imported, TF and Jax are always optional dependencies.

  """

  @property
  def has_jax(self) -> bool:
    return 'jax' in sys.modules

  @property
  def has_tf(self) -> bool:
    return 'tensorflow' in sys.modules

  @property
  def has_torch(self) -> bool:
    return 'torch' in sys.modules

  @property
  def jax(self):
    import jax  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return jax

  @property
  def jnp(self):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return jnp

  @property
  def tf(self):
    import tensorflow  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return tensorflow

  @property
  def tnp(self):
    import tensorflow.experimental.numpy as tnp  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return tnp

  @property
  def torch(self):
    import torch  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return torch

  @property
  def np(self):
    return np

  def is_np_xnp(self, xnp: NpModule) -> bool:
    return xnp is _np

  def is_tf_xnp(self, xnp: NpModule) -> bool:
    return self.has_tf and xnp is self.tnp

  def is_jax_xnp(self, xnp: NpModule) -> bool:
    return self.has_jax and xnp is self.jnp

  def is_torch_xnp(self, xnp: NpModule) -> bool:
    return self.has_torch and xnp is self.torch

  def is_np(self, x: Array) -> bool:
    return isinstance(x, (np.ndarray, np.generic))

  def is_tf(self, x: Array) -> bool:
    return self.has_tf and isinstance(
        x,
        (
            self.tnp.ndarray,
            self.tf.TensorSpec,
            self.tf.__internal__.types.Tensor,
        ),
    )

  def is_jax(self, x: Array) -> bool:
    return self.has_jax and isinstance(x, self.jnp.ndarray)

  def is_torch(self, x: Array) -> bool:
    return self.has_torch and isinstance(x, self.torch.Tensor)

  def is_array(self, x: Array, *, strict: bool = True) -> bool:
    is_array_like = False if strict else isinstance(x, _ARRAY_LIKE_TYPES)
    return (
        self.is_np(x)
        or self.is_jax(x)
        or self.is_tf(x)
        or self.is_torch(x)
        or is_array_like
    )

  def is_np_dtype(self, dtype) -> bool:
    return isinstance(dtype, np.dtype) or epy.issubclass(dtype, np.generic)

  def is_tf_dtype(self, dtype) -> bool:
    return self.has_tf and isinstance(dtype, self.tf.dtypes.DType)

  def is_jax_dtype(self, dtype) -> bool:
    # `jnp.int64`,... are `jax._src.numpy.lax_numpy._ScalarMeta`, but
    # jnp.ndarray.dtype are numpy dtype
    check_jax = self.has_jax and isinstance(dtype, type(self.jnp.float32))
    return self.is_np_dtype(dtype) or check_jax

  def is_torch_dtype(self, dtype) -> bool:
    return self.has_torch and isinstance(dtype, self.torch.dtype)

  def is_dtype(self, dtype) -> bool:
    return (
        self.is_np_dtype(dtype)
        or self.is_jax_dtype(dtype)
        or self.is_tf_dtype(dtype)
        or self.is_torch_dtype(dtype)
    )

  def as_np_dtype(self, dtype):
    if self.is_tf_dtype(dtype):
      dtype = dtype.as_numpy_dtype
    elif self.is_torch_dtype(dtype):
      from etils.enp import compat  # pylint: disable=g-import-not-at-top

      dtype = compat.dtype_torch_to_np(dtype)
    elif not self.is_jax_dtype(dtype) and not self.is_np_dtype(dtype):
      raise TypeError(f'Invalid dtype: {dtype!r}')
    return np.dtype(dtype)

  def as_tf_dtype(self, dtype):
    return self.tf.dtypes.as_dtype(self.as_np_dtype(dtype))

  def as_jax_dtype(self, dtype):
    return self.as_np_dtype(dtype)  # Jax and numpy types are mostly similar

  def as_torch_dtype(self, dtype):
    from etils.enp import compat  # pylint: disable=g-import-not-at-top

    return compat.dtype_np_to_torch(self.as_np_dtype(dtype))

  def as_dtype(self, dtype, *, xnp: NpModule = _np):
    """Normalize to dtype for the given `xnp`."""
    if self.is_np_xnp(xnp):
      return self.as_np_dtype(dtype)
    elif self.is_tf_xnp(xnp):
      return self.as_tf_dtype(dtype)
    elif self.is_jax_xnp(xnp):
      return self.as_jax_dtype(dtype)
    elif self.is_torch_xnp(xnp):
      return self.as_torch_dtype(dtype)
    else:
      raise TypeError(f'Unknown xnp: {xnp!r}')

  def dtype_from_array(
      self,
      array_like: Array,
      *,
      strict: bool = True,
  ) -> Optional[_np.dtype]:
    """Returns the dtype associated with the array."""
    if self.is_array(array_like):  # Already an ndarray, normalize the dtype
      dtype = array_like.dtype
    elif strict:  # Not an array and strict mode: error
      raise TypeError(
          f'Cannot extract dtype from non-array {type(array_like)}, '
          'when strict=True.'
      )
    elif isinstance(array_like, bool):
      dtype = np.bool_
    elif isinstance(array_like, _ARRAY_LIKE_TYPES):  # list, tuple, int, float
      # TODO(epot): Could have a smarter way of infering the dtype for
      # scalar, int, float,... but difficult to infer list without performance
      # cost (one way would be to call `asarray(array_like, dtype=None)`, then
      # cast again)
      return None
    else:
      raise TypeError(f'Cannot extract dtype from non-array {type(array_like)}')
    return self.as_dtype(dtype)

  def get_xnp(self, x: Array, *, strict: bool = True):  # -> NpModule:
    """Returns the numpy module associated with the given array.

    Args:
      x: Either tf, jax or numpy array.
      strict: If `False`, default to `np.array` if the array can't be infered (
        to support array-like: list, tuple,...)

    Returns:
      The numpy module.
    """
    # This is inspired from NEP 37 but without the `__array_module__` magic:
    # https://numpy.org/neps/nep-0037-array-module.html
    # Note there is also an implementation of NEP 37 from the author, but look
    # overly complicated and not available at google.
    # https://github.com/seberg/numpy-dispatch
    if self.is_jax(x):
      return self.jnp
    elif self.is_tf(x):
      return self.tnp
    elif self.is_np(x):
      return np
    elif self.is_torch(x):
      return self.torch
    elif not strict and isinstance(x, _ARRAY_LIKE_TYPES):
      # `strict=False` support `[0, 0, 0]`, `0`,...
      return np
    else:
      raise TypeError(
          f'Cannot infer the numpy module from array: {type(x).__name__}'
      )

  @property
  def is_tnp_enabled(self) -> bool:
    """Returns `True` if numpy mode is enabled."""
    return self.has_tf and hasattr(self.tf.Tensor, 'reshape')

  class LazyArray(metaclass=_LazyArrayMeta):
    """Represent `tf.Tensor`, `jax.ndarray`, `np.ndarray`, `torch.Tensor`.

    Allow to check isinstance without triggering imports from other modules:

    ```
    assert isinstance(jnp.zeros((2,)), enp.lazy.LazyArray)
    ```
    """


lazy = _LazyImporter()


def get_np_module(array: Array, *, strict: bool = True):  # -> NpModule:
  """Returns the numpy module associated with the given array.

  Args:
    array: Either tf, jax or numpy array.
    strict: If `False`, default to `np.array` if the array can't be infered (
      to support array-like: list, tuple,...)

  Returns:
    The numpy module.
  """
  return lazy.get_xnp(array, strict=strict)


def is_dtype_str(dtype) -> bool:
  """Returns True if the dtype is `str`."""
  # tf.string.as_numpy_dtype is object
  try:
    dtype = np.dtype(dtype)
  except TypeError:  # `jax.random.PRNGKeyArray` fail.
    return False
  return dtype.type in {np.object_, np.str_, np.bytes_}


def is_array_str(x: Any) -> bool:
  """Returns True if the given array is a `str` array.

  Note: Also returns True for scalar `str`, `bytes` values. For compatibility
  with `tensor.numpy()` which returns `bytes`

  Args:
    x: The array to test

  Returns:
    True or False
  """
  # `Tensor(shape=(), dtype=tf.string).numpy()` returns `bytes`.
  if isinstance(x, (bytes, str)):
    return True
  elif is_array(x):
    return is_dtype_str(x.dtype)
  else:
    return False


def is_array(x: Any) -> bool:
  """Returns `True` if array is np or `jnp` array."""
  if isinstance(x, np.ndarray):
    return True
  elif lazy.has_jax and isinstance(x, lazy.jnp.ndarray):
    return True
  else:
    return False


@np.vectorize
def _to_str_array(x):
  """Decodes bytes -> str array."""
  # tf.string tensors are returned as bytes, so need to convert them back to str
  return x.decode('utf8') if isinstance(x, bytes) else x


@typing.overload
def normalize_bytes2str(x: bytes) -> str:
  ...


@typing.overload
def normalize_bytes2str(x: _T) -> _T:
  ...


# Ideally could also add `BytesArray -> StrArray`, but both `bytes` and `str`
# are `StrArray`
def normalize_bytes2str(x):
  """Normalize `bytes` array to `str` (UTF-8).

  Example of usage:

  ```python
  for ex in tfds.as_numpy(ds):  # tf.data returns `tf.string` as `bytes`
    ex = tf.nest.map_structure(enp.normalize_bytes2str, ex)
  ```

  Args:
    x: Any array

  Returns:
    x: `bytes` array are decoded as `str`
  """
  if isinstance(x, str):
    return x
  if isinstance(x, bytes):
    return x.decode('utf8')
  elif is_array_str(x):
    # Note: `np.char.decode` is likely faster but don't work on `object` nor
    # bytes arrays.
    return _to_str_array(x)
  else:
    return x
