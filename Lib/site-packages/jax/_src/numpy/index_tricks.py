# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Union

import jax
from jax._src import core
from jax._src.numpy.util import promote_dtypes
from jax._src.numpy.lax_numpy import (
  arange, array, concatenate, expand_dims, linspace, meshgrid, stack, transpose
)
from jax._src.typing import Array, ArrayLike
from jax._src.util import set_module

import numpy as np


export = set_module('jax.numpy')


__all__ = ["c_", "index_exp", "mgrid", "ogrid", "r_", "s_"]


def _make_1d_grid_from_slice(s: slice, op_name: str) -> Array:
  start = core.concrete_or_error(None, s.start,
                                 f"slice start of jnp.{op_name}") or 0
  stop = core.concrete_or_error(None, s.stop,
                                f"slice stop of jnp.{op_name}")
  step = core.concrete_or_error(None, s.step,
                                f"slice step of jnp.{op_name}") or 1
  if np.iscomplex(step):
    newobj = linspace(start, stop, int(abs(step)))
  else:
    newobj = arange(start, stop, step)

  return newobj


class _Mgrid:
  """Return dense multi-dimensional "meshgrid".

  LAX-backend implementation of :obj:`numpy.mgrid`. This is a convenience wrapper for
  functionality provided by :func:`jax.numpy.meshgrid` with ``sparse=False``.

  See Also:
    jnp.ogrid: open/sparse version of jnp.mgrid

  Examples:
    Pass ``[start:stop:step]`` to generate values similar to :func:`jax.numpy.arange`:

    >>> jnp.mgrid[0:4:1]
    Array([0, 1, 2, 3], dtype=int32)

    Passing an imaginary step generates values similar to :func:`jax.numpy.linspace`:

    >>> jnp.mgrid[0:1:4j]
    Array([0.        , 0.33333334, 0.6666667 , 1.        ], dtype=float32)

    Multiple slices can be used to create broadcasted grids of indices:

    >>> jnp.mgrid[:2, :3]
    Array([[[0, 0, 0],
            [1, 1, 1]],
           [[0, 1, 2],
            [0, 1, 2]]], dtype=int32)
  """

  def __getitem__(self, key: slice | tuple[slice, ...]) -> Array:
    if isinstance(key, slice):
      return _make_1d_grid_from_slice(key, op_name="mgrid")
    output: Iterable[Array] = (_make_1d_grid_from_slice(k, op_name="mgrid") for k in key)
    with jax.numpy_dtype_promotion('standard'):
      output = promote_dtypes(*output)
    output_arr = meshgrid(*output, indexing='ij', sparse=False)
    if len(output_arr) == 0:
      return arange(0)
    return stack(output_arr, 0)


mgrid = export(_Mgrid())


class _Ogrid:
  """Return open multi-dimensional "meshgrid".

  LAX-backend implementation of :obj:`numpy.ogrid`. This is a convenience wrapper for
  functionality provided by :func:`jax.numpy.meshgrid` with ``sparse=True``.

  See Also:
    jnp.mgrid: dense version of jnp.ogrid

  Examples:
    Pass ``[start:stop:step]`` to generate values similar to :func:`jax.numpy.arange`:

    >>> jnp.ogrid[0:4:1]
    Array([0, 1, 2, 3], dtype=int32)

    Passing an imaginary step generates values similar to :func:`jax.numpy.linspace`:

    >>> jnp.ogrid[0:1:4j]
    Array([0.        , 0.33333334, 0.6666667 , 1.        ], dtype=float32)

    Multiple slices can be used to create sparse grids of indices:

    >>> jnp.ogrid[:2, :3]
    [Array([[0],
            [1]], dtype=int32),
     Array([[0, 1, 2]], dtype=int32)]
  """

  def __getitem__(
      self, key: slice | tuple[slice, ...]
  ) -> Array | list[Array]:
    if isinstance(key, slice):
      return _make_1d_grid_from_slice(key, op_name="ogrid")
    output: Iterable[Array] = (_make_1d_grid_from_slice(k, op_name="ogrid") for k in key)
    with jax.numpy_dtype_promotion('standard'):
      output = promote_dtypes(*output)
    return meshgrid(*output, indexing='ij', sparse=True)


ogrid = export(_Ogrid())


_IndexType = Union[ArrayLike, str, slice]


class _AxisConcat:
  """Concatenates slices, scalars and array-like objects along a given axis."""
  axis: int
  ndmin: int
  trans1d: int
  op_name: str

  def __getitem__(self, key: _IndexType | tuple[_IndexType, ...]) -> Array:
    key_tup: tuple[_IndexType, ...] = key if isinstance(key, tuple) else (key,)

    params = [self.axis, self.ndmin, self.trans1d, -1]

    directive = key_tup[0]
    if isinstance(directive, str):
      key_tup = key_tup[1:]
      # check two special cases: matrix directives
      if directive == "r":
        params[-1] = 0
      elif directive == "c":
        params[-1] = 1
      else:
        vec: list[Any] = directive.split(",")
        k = len(vec)
        if k < 4:
          vec += params[k:]
        else:
          # ignore everything after the first three comma-separated ints
          vec = vec[:3] + [params[-1]]
        try:
          params = list(map(int, vec))
        except ValueError as err:
          raise ValueError(
            f"could not understand directive {directive!r}"
          ) from err

    axis, ndmin, trans1d, matrix = params

    output = []
    for item in key_tup:
      if isinstance(item, slice):
        newobj = _make_1d_grid_from_slice(item, op_name=self.op_name)
        item_ndim = 0
      elif isinstance(item, str):
        raise ValueError("string directive must be placed at the beginning")
      else:
        newobj = array(item, copy=False)
        item_ndim = newobj.ndim

      newobj = array(newobj, copy=False, ndmin=ndmin)

      if trans1d != -1 and ndmin - item_ndim > 0:
        shape_obj = tuple(range(ndmin))
        # Calculate number of left shifts, with overflow protection by mod
        num_lshifts = ndmin - abs(ndmin + trans1d + 1) % ndmin
        shape_obj = tuple(shape_obj[num_lshifts:] + shape_obj[:num_lshifts])

        newobj = transpose(newobj, shape_obj)

      output.append(newobj)

    res = concatenate(tuple(output), axis=axis)

    if matrix != -1 and res.ndim == 1:
      # insert 2nd dim at axis 0 or 1
      res = expand_dims(res, matrix)

    return res

  def __len__(self) -> int:
    return 0


class RClass(_AxisConcat):
  """Concatenate slices, scalars and array-like objects along the first axis.

  LAX-backend implementation of :obj:`numpy.r_`.

  See Also:
    ``jnp.c_``: Concatenates slices, scalars and array-like objects along the last axis.

  Examples:
    Passing slices in the form ``[start:stop:step]`` generates ``jnp.arange`` objects:

    >>> jnp.r_[-1:5:1, 0, 0, jnp.array([1,2,3])]
    Array([-1,  0,  1,  2,  3,  4,  0,  0,  1,  2,  3], dtype=int32)

    An imaginary value for ``step`` will create a ``jnp.linspace`` object instead,
    which includes the right endpoint:

    >>> jnp.r_[-1:1:6j, 0, jnp.array([1,2,3])]
    Array([-1.        , -0.6       , -0.20000002,  0.20000005,
           0.6       ,  1.        ,  0.        ,  1.        ,
           2.        ,  3.        ], dtype=float32)

    Use a string directive of the form ``"axis,dims,trans1d"`` as the first argument to
    specify concatenation axis, minimum number of dimensions, and the position of the
    upgraded array's original dimensions in the resulting array's shape tuple:

    >>> jnp.r_['0,2', [1,2,3], [4,5,6]] # concatenate along first axis, 2D output
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)

    >>> jnp.r_['0,2,0', [1,2,3], [4,5,6]] # push last input axis to the front
    Array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]], dtype=int32)

    Negative values for ``trans1d`` offset the last axis towards the start
    of the shape tuple:

    >>> jnp.r_['0,2,-2', [1,2,3], [4,5,6]]
    Array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]], dtype=int32)

    Use the special directives ``"r"`` or ``"c"`` as the first argument on flat inputs
    to create an array with an extra row or column axis, respectively:

    >>> jnp.r_['r',[1,2,3], [4,5,6]]
    Array([[1, 2, 3, 4, 5, 6]], dtype=int32)

    >>> jnp.r_['c',[1,2,3], [4,5,6]]
    Array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]], dtype=int32)

    For higher-dimensional inputs (``dim >= 2``), both directives ``"r"`` and ``"c"``
    give the same result.
  """
  axis = 0
  ndmin = 1
  trans1d = -1
  op_name = "r_"


r_ = export(RClass())


class CClass(_AxisConcat):
  """Concatenate slices, scalars and array-like objects along the last axis.

  LAX-backend implementation of :obj:`numpy.c_`.

  See Also:
    ``jnp.r_``: Concatenates slices, scalars and array-like objects along the first axis.

  Examples:

    >>> a = jnp.arange(6).reshape((2,3))
    >>> jnp.c_[a,a]
    Array([[0, 1, 2, 0, 1, 2],
           [3, 4, 5, 3, 4, 5]], dtype=int32)

    Use a string directive of the form ``"axis:dims:trans1d"`` as the first argument to specify
    concatenation axis, minimum number of dimensions, and the position of the upgraded array's
    original dimensions in the resulting array's shape tuple:

    >>> jnp.c_['0,2', [1,2,3], [4,5,6]]
    Array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]], dtype=int32)

    >>> jnp.c_['0,2,-1', [1,2,3], [4,5,6]]
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)

    Use the special directives ``"r"`` or ``"c"`` as the first argument on flat inputs
    to create an array with inputs stacked along the last axis:

    >>> jnp.c_['r',[1,2,3], [4,5,6]]
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
  """
  axis = -1
  ndmin = 2
  trans1d = 0
  op_name = "c_"


c_ = export(CClass())

s_ = np.s_

index_exp = np.index_exp
