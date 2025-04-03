# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Utilities to hold expected dimension sizes."""

from collections.abc import Sized
import math
import re
from typing import Any, Collection, Dict, Optional, Tuple


Shape = Tuple[Optional[int], ...]


class Dimensions:
  """A lightweight utility that maps strings to shape tuples.

  The most basic usage is:

  .. code::

    >>> dims = chex.Dimensions(B=3, T=5, N=7)  # You can specify any letters.
    >>> dims['NBT']
    (7, 3, 5)

  This is useful when dealing with many differently shaped arrays. For instance,
  let's check the shape of this array:

  .. code::

    >>> x = jnp.array([[2, 0, 5, 6, 3],
    ...                [5, 4, 4, 3, 3],
    ...                [0, 0, 5, 2, 0]])
    >>> chex.assert_shape(x, dims['BT'])

  The dimension sizes can be gotten directly, e.g. :code:`dims.N == 7`. This can
  be useful in many applications. For instance, let's one-hot encode our array.

  .. code::

    >>> y = jax.nn.one_hot(x, dims.N)
    >>> chex.assert_shape(y, dims['BTN'])

  You can also store the shape of a given array in :code:`dims`, e.g.

  .. code::

    >>> z = jnp.array([[0, 6, 0, 2],
    ...                [4, 2, 2, 4]])
    >>> dims['XY'] = z.shape
    >>> dims
    Dimensions(B=3, N=7, T=5, X=2, Y=4)

  You can access the flat size of a shape as

  .. code::

    >>> dims.size('BT')  # Same as prod(dims['BT']).
    15

  Similarly, you can flatten axes together by wrapping them in parentheses:

  .. code::

    >>> dims['(BT)N']
    (15, 7)

  You can set a wildcard dimension, cf. :func:`chex.assert_shape`:

  .. code::

    >>> dims.W = None
    >>> dims['BTW']
    (3, 5, None)

  Or you can use the wildcard character `'*'` directly:

  .. code::

    >>> dims['BT*']
    (3, 5, None)

  Single digits are interpreted as literal integers. Note that this notation
  is limited to single-digit literals.

  .. code::

    >>> dims['BT123']
    (3, 5, 1, 2, 3)

  Support for single digits was mainly included to accommodate dummy axes
  introduced for consistent broadcasting. For instance, instead of using
  :func:`jnp.expand_dims <jax.numpy.expand_dims>` you could do the following:

  .. code::

    >>> w = y * x  # Cannot broadcast (3, 5, 7) with (3, 5)
    Traceback (most recent call last):
        ...
    ValueError: Incompatible shapes for broadcasting: ((3, 5, 7), (1, 3, 5))
    >>> w = y * x.reshape(dims['BT1'])
    >>> chex.assert_shape(w, dims['BTN'])

  Sometimes you only care about some array dimensions but not all. You can use
  an underscore to ignore an axis, e.g.

  .. code::

    >>> chex.assert_rank(y, 3)
    >>> dims['__M'] = y.shape  # Skip the first two axes.

  Finally note that a single-character key returns a tuple of length one.

  .. code::

    >>> dims['M']
    (7,)
  """
  # Tell static type checker not to worry about attribute errors.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, **dim_sizes) -> None:
    for dim, size in dim_sizes.items():
      self._setdim(dim, size)

  def size(self, key: str) -> int:
    """Returns the flat size of a given named shape, i.e. prod(shape)."""
    shape = self[key]
    if any(size is None or size <= 0 for size in shape):
      raise ValueError(
          f"cannot take product of shape '{key}' = {shape}, "
          'because it contains non-positive sized dimensions'
      )
    return math.prod(shape)

  def __getitem__(self, key: str) -> Shape:
    self._validate_key(key)
    shape = []
    open_parentheses = False
    dims_to_flatten = ''
    for dim in key:
      # Signal to start accumulating `dims_to_flatten`.
      if dim == '(':
        if open_parentheses:
          raise ValueError(f"nested parentheses are unsupported; got: '{key}'")
        open_parentheses = True

      # Signal to collect accumulated `dims_to_flatten`.
      elif dim == ')':
        if not open_parentheses:
          raise ValueError(f"unmatched parentheses in named shape: '{key}'")
        if not dims_to_flatten:
          raise ValueError(f"found empty parentheses in named shape: '{key}'")
        shape.append(self.size(dims_to_flatten))
        # Reset.
        open_parentheses = False
        dims_to_flatten = ''

      # Accumulate `dims_to_flatten`.
      elif open_parentheses:
        dims_to_flatten += dim

      # The typical (non-flattening) case.
      else:
        shape.append(self._getdim(dim))

    if open_parentheses:
      raise ValueError(f"unmatched parentheses in named shape: '{key}'")
    return tuple(shape)

  def __setitem__(self, key: str, value: Collection[Optional[int]]) -> None:
    self._validate_key(key)
    self._validate_value(value)
    if len(key) != len(value):
      raise ValueError(
          f'key string {repr(key)} and shape {tuple(value)} '
          'have different lengths')
    for dim, size in zip(key, value):
      self._setdim(dim, size)

  def __delitem__(self, key: str) -> None:
    self._validate_key(key)
    for dim in key:
      self._deldim(dim)

  def __repr__(self) -> str:
    args = ', '.join(f'{k}={v}' for k, v in sorted(self._asdict().items()))
    return f'{type(self).__name__}({args})'

  def _asdict(self) -> Dict[str, Optional[int]]:
    return {k: v for k, v in self.__dict__.items()
            if re.fullmatch(r'[a-zA-Z]', k)}

  def _getdim(self, dim: str) -> Optional[int]:
    if dim == '*':
      return None
    if re.fullmatch(r'[0-9]', dim):
      return int(dim)
    try:
      return getattr(self, dim)
    except AttributeError as e:
      raise KeyError(dim) from e

  def _setdim(self, dim: str, size: Optional[int]) -> None:
    if dim == '_':  # Skip.
      return
    self._validate_dim(dim)
    setattr(self, dim, _optional_int(size))

  def _deldim(self, dim: str) -> None:
    if dim == '_':  # Skip.
      return
    self._validate_dim(dim)
    try:
      return delattr(self, dim)
    except AttributeError as e:
      raise KeyError(dim) from e

  def _validate_key(self, key: Any) -> None:
    if not isinstance(key, str):
      raise TypeError(f'key must be a string; got: {type(key).__name__}')

  def _validate_value(self, value: Any) -> None:
    if not isinstance(value, Sized):
      raise TypeError(
          'value must be sized, i.e. an object with a well-defined len(value); '
          f'got object of type: {type(value).__name__}')

  def _validate_dim(self, dim: Any) -> None:
    if not isinstance(dim, str):
      raise TypeError(
          f'dimension name must be a string; got: {type(dim).__name__}')
    if not re.fullmatch(r'[a-zA-Z]', dim):
      raise KeyError(
          'dimension names may only be contain letters (or \'_\' to skip); '
          f'got dimension name: {repr(dim)}')


def _optional_int(x: Any) -> Optional[int]:
  if x is None:
    return None
  try:
    i = int(x)
    if x == i:
      return i
  except ValueError:
    pass
  raise TypeError(f'object cannot be interpreted as a python int: {repr(x)}')
