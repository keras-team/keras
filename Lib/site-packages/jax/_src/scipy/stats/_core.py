# Copyright 2023 The JAX Authors.
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

from collections import namedtuple
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax import jit
from jax._src import dtypes
from jax._src.api import vmap
from jax._src.numpy.util import check_arraylike, promote_args_inexact
from jax._src.typing import ArrayLike, Array
from jax._src.util import canonicalize_axis


ModeResult = namedtuple('ModeResult', ('mode', 'count'))

@partial(jit, static_argnames=['axis', 'nan_policy', 'keepdims'])
def mode(a: ArrayLike, axis: int | None = 0, nan_policy: str = "propagate", keepdims: bool = False) -> ModeResult:
  """Compute the mode (most common value) along an axis of an array.

  JAX implementation of :func:`scipy.stats.mode`.

  Args:
    a: arraylike
    axis: int, default=0. Axis along which to compute the mode.
    nan_policy: str. JAX only supports ``"propagate"``.
    keepdims: bool, default=False. If true, reduced axes are left in the result
      with size 1.

  Returns:
    A tuple of arrays, ``(mode, count)``. ``mode`` is the array of modal values,
    and ``count`` is the number of times each value appears in the input array.

  Examples:
    >>> x = jnp.array([2, 4, 1, 1, 3, 4, 4, 2, 3])
    >>> mode, count = jax.scipy.stats.mode(x)
    >>> mode, count
    (Array(4, dtype=int32), Array(3, dtype=int32))

    For multi dimensional arrays, ``jax.scipy.stats.mode`` computes the ``mode``
    and the corresponding ``count`` along ``axis=0``:

    >>> x1 = jnp.array([[1, 2, 1, 3, 2, 1],
    ...                 [3, 1, 3, 2, 1, 3],
    ...                 [1, 2, 2, 3, 1, 2]])
    >>> mode, count = jax.scipy.stats.mode(x1)
    >>> mode, count
    (Array([1, 2, 1, 3, 1, 1], dtype=int32), Array([2, 2, 1, 2, 2, 1], dtype=int32))

    If ``axis=1``, ``mode`` and ``count`` will be computed along ``axis 1``.

    >>> mode, count = jax.scipy.stats.mode(x1, axis=1)
    >>> mode, count
    (Array([1, 3, 2], dtype=int32), Array([3, 3, 3], dtype=int32))

    By default, ``jax.scipy.stats.mode`` reduces the dimension of the result.
    To keep the dimensions same as that of the input array, the argument
    ``keepdims`` must be set to ``True``.

    >>> mode, count = jax.scipy.stats.mode(x1, axis=1, keepdims=True)
    >>> mode, count
    (Array([[1],
           [3],
           [2]], dtype=int32), Array([[3],
           [3],
           [3]], dtype=int32))
  """
  check_arraylike("mode", a)
  x = jnp.atleast_1d(a)

  if nan_policy not in ["propagate", "omit", "raise"]:
    raise ValueError(
      f"Illegal nan_policy value {nan_policy!r}; expected one of "
      "{'propagate', 'omit', 'raise'}"
    )
  if nan_policy == "omit":
    # TODO: return answer without nans included.
    raise NotImplementedError(
      f"Logic for `nan_policy` of {nan_policy} is not implemented"
    )
  if nan_policy == "raise":
    raise NotImplementedError(
      "In order to best JIT compile `mode`, we cannot know whether `x` contains nans. "
      "Please check if nans exist in `x` outside of the `mode` function."
    )

  input_shape = x.shape
  if keepdims:
    if axis is None:
      output_shape = tuple(1 for i in input_shape)
    else:
      output_shape = tuple(1 if i == axis else s for i, s in enumerate(input_shape))
  else:
    if axis is None:
      output_shape = ()
    else:
      output_shape = tuple(s for i, s in enumerate(input_shape) if i != axis)

  if axis is None:
    axis = 0
    x = x.ravel()

  def _mode_helper(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Helper function to return mode and count of a given array."""
    if x.size == 0:
      return (jnp.array(jnp.nan, dtype=dtypes.canonicalize_dtype(jnp.float_)),
              jnp.array(0, dtype=dtypes.canonicalize_dtype(jnp.float_)))
    else:
      vals, counts = jnp.unique(x, return_counts=True, size=x.size)
      return vals[jnp.argmax(counts)], counts.max()

  axis = canonicalize_axis(axis, x.ndim)
  x = jnp.moveaxis(x, axis, 0)
  x = x.reshape(x.shape[0], math.prod(x.shape[1:]))
  vals, counts = vmap(_mode_helper, in_axes=1)(x)
  return ModeResult(vals.reshape(output_shape), counts.reshape(output_shape))

def invert_permutation(i: Array) -> Array:
  """Helper function that inverts a permutation array."""
  return jnp.empty_like(i).at[i].set(jnp.arange(i.size, dtype=i.dtype))


@partial(jit, static_argnames=["method", "axis", "nan_policy"])
def rankdata(
  a: ArrayLike,
  method: str = "average",
  *,
  axis: int | None = None,
  nan_policy: str = "propagate",
) -> Array:
  """Compute the rank of data along an array axis.

  JAX implementation of :func:`scipy.stats.rankdata`.

  Ranks begin at 1, and the *method* argument controls how ties are handled.

  Args:
    a: arraylike
    method: str, default="average". Supported methods are
      ``("average", "min", "max", "dense", "ordinal")``
      For details, see the :func:`scipy.stats.rankdata` documentation.
    axis: optional integer. If not specified, the input array is flattened.
    nan_policy: str, JAX's implementation only supports ``"propagate"``.

  Returns:
    array of ranks along the specified axis.

  Examples:

    >>> x = jnp.array([10, 30, 20])
    >>> rankdata(x)
    Array([1., 3., 2.], dtype=float32)

    >>> x = jnp.array([1, 3, 2, 3])
    >>> rankdata(x)
    Array([1. , 3.5, 2. , 3.5], dtype=float32)
  """
  check_arraylike("rankdata", a)

  if nan_policy not in ["propagate", "omit", "raise"]:
    raise ValueError(
      f"Illegal nan_policy value {nan_policy!r}; expected one of "
      "{'propagate', 'omit', 'raise'}"
    )
  if nan_policy == "omit":
    raise NotImplementedError(
      f"Logic for `nan_policy` of {nan_policy} is not implemented"
    )
  if nan_policy == "raise":
    raise NotImplementedError(
      "In order to best JIT compile `mode`, we cannot know whether `x` "
      "contains nans. Please check if nans exist in `x` outside of the "
      "`rankdata` function."
    )

  if method not in ("average", "min", "max", "dense", "ordinal"):
    raise ValueError(f"unknown method '{method}'")

  a = jnp.asarray(a)

  if axis is not None:
    return jnp.apply_along_axis(rankdata, axis, a, method)

  arr = jnp.ravel(a)
  arr, sorter = jax.lax.sort_key_val(arr, jnp.arange(arr.size))
  inv = invert_permutation(sorter)

  if method == "ordinal":
    return inv + 1
  obs = jnp.concatenate([jnp.array([True]), arr[1:] != arr[:-1]])
  dense = obs.cumsum()[inv]
  if method == "dense":
    return dense
  count = jnp.nonzero(obs, size=arr.size + 1, fill_value=obs.size)[0]
  if method == "max":
    return count[dense]
  if method == "min":
    return count[dense - 1] + 1
  if method == "average":
    return .5 * (count[dense] + count[dense - 1] + 1).astype(dtypes.canonicalize_dtype(jnp.float_))
  raise ValueError(f"unknown method '{method}'")


@partial(jit, static_argnames=['axis', 'nan_policy', 'keepdims'])
def sem(a: ArrayLike, axis: int | None = 0, ddof: int = 1, nan_policy: str = "propagate", *, keepdims: bool = False) -> Array:
  """Compute the standard error of the mean.

  JAX implementation of :func:`scipy.stats.sem`.

  Args:
    a: arraylike
    axis: optional integer. If not specified, the input array is flattened.
    ddof: integer, default=1. The degrees of freedom in the SEM computation.
    nan_policy: str, default="propagate". JAX supports only "propagate" and
      "omit".
    keepdims: bool, default=False. If true, reduced axes are left in the result
      with size 1.

  Returns:
    array

  Examples:
    >>> x = jnp.array([2, 4, 1, 1, 3, 4, 4, 2, 3])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x)
    Array(0.41, dtype=float32)

    For multi dimensional arrays, ``sem`` computes standard error of mean along
    ``axis=0``:

    >>> x1 = jnp.array([[1, 2, 1, 3, 2, 1],
    ...                 [3, 1, 3, 2, 1, 3],
    ...                 [1, 2, 2, 3, 1, 2]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x1)
    Array([0.67, 0.33, 0.58, 0.33, 0.33, 0.58], dtype=float32)

    If ``axis=1``, standard error of mean will be computed along ``axis 1``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x1, axis=1)
    Array([0.33, 0.4 , 0.31], dtype=float32)

    If ``axis=None``, standard error of mean will be computed along all the axes.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x1, axis=None)
    Array(0.2, dtype=float32)

    By default, ``sem`` reduces the dimension of the result. To keep the
    dimensions same as that of the input array, the argument ``keepdims`` must
    be set to ``True``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x1, axis=1, keepdims=True)
    Array([[0.33],
           [0.4 ],
           [0.31]], dtype=float32)

    Since, by default, ``nan_policy='propagate'``, ``sem`` propagates the ``nan``
    values in the result.

    >>> nan = jnp.nan
    >>> x2 = jnp.array([[1, 2, 3, nan, 4, 2],
    ...                 [4, 5, 4, 3, nan, 1],
    ...                 [7, nan, 8, 7, 9, nan]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x2)
    Array([1.73,  nan, 1.53,  nan,  nan,  nan], dtype=float32)

    If ``nan_policy='omit```, ``sem`` omits the ``nan`` values and computes the error
    for the remainging values along the specified axis.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x2, nan_policy='omit')
    Array([1.73, 1.5 , 1.53, 2.  , 2.5 , 0.5 ], dtype=float32)
  """
  b, = promote_args_inexact("sem", a)
  if nan_policy == "propagate":
    size = b.size if axis is None else b.shape[axis]
    return b.std(axis, ddof=ddof, keepdims=keepdims) / jnp.sqrt(size).astype(b.dtype)
  elif nan_policy == "omit":
    count = (~jnp.isnan(b)).sum(axis, keepdims=keepdims)
    return jnp.nanstd(b, axis, ddof=ddof, keepdims=keepdims) / jnp.sqrt(count).astype(b.dtype)
  else:
    raise ValueError(f"{nan_policy} is not supported")
