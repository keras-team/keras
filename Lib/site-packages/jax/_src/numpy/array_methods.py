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

# pytype: skip-file
# mypy: disable-error-code=has-type
"""Define methods which are dynamically added to JAX's Arrays and Tracers.

This is done dynamically in order to avoid circular imports.
"""

from __future__ import annotations

__all__ = ['register_jax_array_methods']

import abc
from functools import partial, wraps
import math
from typing import Any, Sequence

import numpy as np
import jax
from jax import lax
from jax.sharding import Sharding
from jax._src import api
from jax._src import core
from jax._src import dtypes
from jax._src.api_util import _ensure_index_tuple
from jax._src.array import ArrayImpl
from jax._src.lax import lax as lax_internal
from jax._src.lib import xla_client as xc
from jax._src.numpy import array_api_metadata
from jax._src.numpy import lax_numpy
from jax._src import mesh as mesh_lib
from jax._src.pjit import hidden_mode, PartitionSpec
from jax._src.sharding_impls import canonicalize_sharding, NamedSharding
from jax._src.numpy import reductions
from jax._src.numpy import ufuncs
from jax._src.ops import scatter
from jax._src.typing import Array, ArrayLike, DimSize, DTypeLike, Shape, StaticScalar
from jax._src.util import safe_zip, safe_map

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


### add method and operator overloads to arraylike classes

# We add operator overloads to Array and ShapedArray. These method and
# operator overloads mainly just forward calls to the corresponding lax_numpy
# functions, which can themselves handle instances from any of these classes.


def _all(self: Array, axis: reductions.Axis = None, out: None = None,
         keepdims: bool = False, *, where: ArrayLike | None = None) -> Array:
  """Test whether all array elements along a given axis evaluate to True.

  Refer to :func:`jax.numpy.all` for the full documentation.
  """
  return reductions.all(self, axis=axis, out=out, keepdims=keepdims, where=where)

def _any(self: Array, axis: reductions.Axis = None, out: None = None,
         keepdims: bool = False, *, where: ArrayLike | None = None) -> Array:
  """Test whether any array elements along a given axis evaluate to True.

  Refer to :func:`jax.numpy.any` for the full documentation.
  """
  return reductions.any(self, axis=axis, out=out, keepdims=keepdims, where=where)

def _argmax(self: Array, axis: int | None = None, out: None = None,
            keepdims: bool | None = None) -> Array:
  """Return the index of the maximum value.

  Refer to :func:`jax.numpy.argmax` for the full documentation.
  """
  return lax_numpy.argmax(self, axis=axis, out=out, keepdims=keepdims)

def _argmin(self: Array, axis: int | None = None, out: None = None,
            keepdims: bool | None = None) -> Array:
  """Return the index of the minimum value.

  Refer to :func:`jax.numpy.argmin` for the full documentation.
  """
  return lax_numpy.argmin(self, axis=axis, out=out, keepdims=keepdims)

def _argpartition(self: Array, kth: int, axis: int = -1) -> Array:
  """Return the indices that partially sort the array.

  Refer to :func:`jax.numpy.argpartition` for the full documentation.
  """
  return lax_numpy.argpartition(self, kth=kth, axis=axis)

def _argsort(self: Array, axis: int | None = -1, *, kind: None = None, order: None = None,
             stable: bool = True, descending: bool = False) -> Array:
  """Return the indices that sort the array.

  Refer to :func:`jax.numpy.argsort` for the full documentation.
  """
  return lax_numpy.argsort(self, axis=axis, kind=kind, order=order,
                           stable=stable, descending=descending)

def _astype(self: Array, dtype: DTypeLike | None, copy: bool = False,
            device: xc.Device | Sharding | None = None) -> Array:
  """Copy the array and cast to a specified dtype.

  This is implemented via :func:`jax.lax.convert_element_type`, which may
  have slightly different behavior than :meth:`numpy.ndarray.astype` in
  some cases. In particular, the details of float-to-int and int-to-float
  casts are implementation dependent.
  """
  return lax_numpy.astype(self, dtype, copy=copy, device=device)

def _choose(self: Array, choices: Sequence[ArrayLike], out: None = None, mode: str = 'raise') -> Array:
  """Construct an array choosing from elements of multiple arrays.

  Refer to :func:`jax.numpy.choose` for the full documentation.
  """
  return lax_numpy.choose(self, choices=choices, out=out, mode=mode)

def _clip(self: Array, min: ArrayLike | None = None, max: ArrayLike | None = None) -> Array:
  """Return an array whose values are limited to a specified range.

  Refer to :func:`jax.numpy.clip` for full documentation.
  """
  return lax_numpy.clip(self, min=min, max=max)

def _compress(self: Array, condition: ArrayLike,
              axis: int | None = None, *, out: None = None,
              size: int | None = None, fill_value: ArrayLike = 0) -> Array:
  """Return selected slices of this array along given axis.

  Refer to :func:`jax.numpy.compress` for full documentation.
  """
  return lax_numpy.compress(condition, self, axis=axis, out=out,
                            size=size, fill_value=fill_value)

def _conj(self: Array) -> Array:
  """Return the complex conjugate of the array.

  Refer to :func:`jax.numpy.conj` for the full documentation.
  """
  return ufuncs.conj(self)

def _conjugate(self: Array) -> Array:
  """Return the complex conjugate of the array.

  Refer to :func:`jax.numpy.conjugate` for the full documentation.
  """
  return ufuncs.conjugate(self)

def _copy(self: Array) -> Array:
  """Return a copy of the array.

  Refer to :func:`jax.numpy.copy` for the full documentation.
  """
  return lax_numpy.copy(self)

def _cumprod(self: Array, axis: reductions.Axis = None, dtype: DTypeLike | None = None,
             out: None = None) -> Array:
  """Return the cumulative product of the array.

  Refer to :func:`jax.numpy.cumprod` for the full documentation.
  """
  return reductions.cumprod(self, axis=axis, dtype=dtype, out=out)

def _cumsum(self: Array, axis: reductions.Axis = None, dtype: DTypeLike | None = None,
            out: None = None) -> Array:
  """Return the cumulative sum of the array.

  Refer to :func:`jax.numpy.cumsum` for the full documentation.
  """
  return reductions.cumsum(self, axis=axis, dtype=dtype, out=out)

def _diagonal(self: Array, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Array:
  """Return the specified diagonal from the array.

  Refer to :func:`jax.numpy.diagonal` for the full documentation.
  """
  return lax_numpy.diagonal(self, offset=offset, axis1=axis1, axis2=axis2)

def _dot(self: Array, b: ArrayLike, *, precision: lax_internal.PrecisionLike = None,
         preferred_element_type: DTypeLike | None = None) -> Array:
  """Compute the dot product of two arrays.

  Refer to :func:`jax.numpy.dot` for the full documentation.
  """
  return lax_numpy.dot(self, b, precision=precision, preferred_element_type=preferred_element_type)

def _flatten(self: Array, order: str = "C") -> Array:
  """Flatten array into a 1-dimensional shape.

  Refer to :func:`jax.numpy.ravel` for the full documentation.
  """
  return lax_numpy.ravel(self, order=order)

def _imag_property(self: Array) -> Array:
  """Return the imaginary part of the array."""
  return ufuncs.imag(self)

def _item(self: Array, *args: int) -> bool | int | float | complex:
  """Copy an element of an array to a standard Python scalar and return it."""
  arr = core.concrete_or_error(np.asarray, self, context="This occurred in the item() method of jax.Array")
  if dtypes.issubdtype(self.dtype, dtypes.extended):
    raise TypeError(f"No Python scalar type for {arr.dtype=}")
  return arr.item(*args)

def _itemsize_property(self: Array) -> int:
  """Length of one array element in bytes."""
  return dtypes.dtype(self, canonicalize=True).itemsize

def _matrix_transpose_property(self: Array):
  """Compute the (batched) matrix transpose.

  Refer to :func:`jax.numpy.matrix_transpose` for details.
  """
  return lax_numpy.matrix_transpose(self)

def _max(self: Array, axis: reductions.Axis = None, out: None = None,
         keepdims: bool = False, initial: ArrayLike | None = None,
         where: ArrayLike | None = None) -> Array:
  """Return the maximum of array elements along a given axis.

  Refer to :func:`jax.numpy.max` for the full documentation.
  """
  return reductions.max(self, axis=axis, out=out, keepdims=keepdims,
                        initial=initial, where=where)


def _mean(self: Array, axis: reductions.Axis = None, dtype: DTypeLike | None = None,
          out: None = None, keepdims: bool = False, *,
          where: ArrayLike | None = None) -> Array:
  """Return the mean of array elements along a given axis.

  Refer to :func:`jax.numpy.mean` for the full documentation.
  """
  return reductions.mean(self, axis=axis, dtype=dtype, out=out,
                         keepdims=keepdims, where=where)

def _min(self: Array, axis: reductions.Axis = None, out: None = None,
         keepdims: bool = False, initial: ArrayLike | None = None,
         where: ArrayLike | None = None) -> Array:
  """Return the minimum of array elements along a given axis.

  Refer to :func:`jax.numpy.min` for the full documentation.
  """
  return reductions.min(self, axis=axis, out=out, keepdims=keepdims,
                        initial=initial, where=where)

def _nbytes_property(self: Array) -> int:
  """Total bytes consumed by the elements of the array."""
  return np.size(self) * dtypes.dtype(self, canonicalize=True).itemsize

def _nonzero(self: Array, *, fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None,
             size: int | None = None) -> tuple[Array, ...]:
  """Return indices of nonzero elements of an array.

  Refer to :func:`jax.numpy.nonzero` for the full documentation.
  """
  return lax_numpy.nonzero(self, size=size, fill_value=fill_value)

def _prod(self: Array, axis: reductions.Axis = None, dtype: DTypeLike | None = None,
          out: None = None, keepdims: bool = False,
          initial: ArrayLike | None = None, where: ArrayLike | None = None,
          promote_integers: bool = True) -> Array:
  """Return product of the array elements over a given axis.

  Refer to :func:`jax.numpy.prod` for the full documentation.
  """
  return reductions.prod(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                         initial=initial, where=where, promote_integers=promote_integers)

def _ptp(self: Array, axis: reductions.Axis = None, out: None = None,
         keepdims: bool = False) -> Array:
  """Return the peak-to-peak range along a given axis.

  Refer to :func:`jax.numpy.ptp` for the full documentation.
  """
  return reductions.ptp(self, axis=axis, out=out, keepdims=keepdims)

def _real_property(self: Array) -> Array:
  """Return the real part of the array."""
  return ufuncs.real(self)

def _repeat(self: Array, repeats: ArrayLike, axis: int | None = None, *,
            total_repeat_length: int | None = None) -> Array:
  """Construct an array from repeated elements.

  Refer to :func:`jax.numpy.repeat` for the full documentation.
  """
  return lax_numpy.repeat(self, repeats=repeats, axis=axis, total_repeat_length=total_repeat_length)

def _reshape(self: Array, *args: Any, order: str = "C") -> Array:
  """Returns an array containing the same data with a new shape.

  Refer to :func:`jax.numpy.reshape` for full documentation.
  """
  __tracebackhide__ = True
  newshape = _compute_newshape(self, args[0] if len(args) == 1 else args)
  if order == "C":
    return lax.reshape(self, newshape, None)
  elif order == "F":
    dims = list(range(self.ndim)[::-1])
    return lax.reshape(self, newshape[::-1], dims).T
  elif order == "A":
    raise NotImplementedError("np.reshape order=A is not implemented.")
  else:
    raise ValueError(f"Unexpected value for 'order' argument: {order}.")

def _round(self: Array, decimals: int = 0, out: None = None) -> Array:
  """Round array elements to a given decimal.

  Refer to :func:`jax.numpy.round` for full documentation.
  """
  return lax_numpy.round(self, decimals=decimals, out=out)

def _searchsorted(self: Array, v: ArrayLike, side: str = 'left',
                  sorter: ArrayLike | None = None, *, method: str = 'scan') -> Array:
  """Perform a binary search within a sorted array.

  Refer to :func:`jax.numpy.searchsorted` for full documentation."""
  return lax_numpy.searchsorted(self, v, side=side, sorter=sorter, method=method)

def _sort(self: Array, axis: int | None = -1, *, kind: None = None,
          order: None = None, stable: bool = True, descending: bool = False) -> Array:
  """Return a sorted copy of an array.

  Refer to :func:`jax.numpy.sort` for full documentation.
  """
  return lax_numpy.sort(self, axis=axis, kind=kind, order=order,
                        stable=stable, descending=descending)

def _squeeze(self: Array, axis: reductions.Axis = None) -> Array:
  """Remove one or more length-1 axes from array.

  Refer to :func:`jax.numpy.squeeze` for full documentation.
  """
  return lax_numpy.squeeze(self, axis=axis)

def _std(self: Array, axis: reductions.Axis = None, dtype: DTypeLike | None = None,
         out: None = None, ddof: int = 0, keepdims: bool = False, *,
         where: ArrayLike | None = None, correction: int | float | None = None) -> Array:
  """Compute the standard deviation along a given axis.

  Refer to :func:`jax.numpy.std` for full documentation.
  """
  return reductions.std(self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims,
                        where=where, correction=correction)

def _sum(self: Array, axis: reductions.Axis = None, dtype: DTypeLike | None = None,
         out: None = None, keepdims: bool = False, initial: ArrayLike | None = None,
         where: ArrayLike | None = None, promote_integers: bool = True) -> Array:
  """Sum of the elements of the array over a given axis.

  Refer to :func:`jax.numpy.sum` for full documentation.
  """
  return reductions.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                        where=where, promote_integers=promote_integers)

def _swapaxes(self: Array, axis1: int, axis2: int) -> Array:
  """Swap two axes of an array.

  Refer to :func:`jax.numpy.swapaxes` for full documentation.
  """
  return lax_numpy.swapaxes(self, axis1=axis1, axis2=axis2)


def _take(self: Array, indices: ArrayLike, axis: int | None = None, out: None = None,
          mode: str | None = None, unique_indices: bool = False, indices_are_sorted: bool = False,
          fill_value: StaticScalar | None = None) -> Array:
  """Take elements from an array.

  Refer to :func:`jax.numpy.take` for full documentation.
  """
  return lax_numpy.take(self, indices, axis=axis, out=out, mode=mode, unique_indices=unique_indices,
                        indices_are_sorted=indices_are_sorted, fill_value=fill_value)

def _to_device(self: Array, device: xc.Device | Sharding, *,
               stream: int | Any | None = None):
  """Return a copy of the array on the specified device

  Args:
    device: :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.
    stream: not implemented, passing a non-None value will lead to an error.
  Returns:
    copy of array placed on the specified device or devices.
  """
  if stream is not None:
    raise NotImplementedError("stream argument of array.to_device()")
  return api.device_put(self, device)


def _trace(self: Array, offset: int | ArrayLike = 0, axis1: int = 0, axis2: int = 1,
           dtype: DTypeLike | None = None, out: None = None) -> Array:
  """Return the sum along the diagonal.

  Refer to :func:`jax.numpy.trace` for full documentation.
  """
  return lax_numpy.trace(self, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)

def _transpose(self: Array, *args: Any) -> Array:
  """Returns a copy of the array with axes transposed.

  Refer to :func:`jax.numpy.transpose` for full documentation.
  """
  if not args:
    axis = None
  elif len(args) == 1:
    axis = args[0] if args[0] is None else _ensure_index_tuple(args[0])
  else:
    axis = _ensure_index_tuple(args)
  return lax_numpy.transpose(self, axis)

def _transpose_property(self: Array):
  """Compute the all-axis array transpose.

  Refer to :func:`jax.numpy.transpose` for details.
  """
  return lax_numpy.transpose(self)

def _var(self: Array, axis: reductions.Axis = None, dtype: DTypeLike | None = None,
         out: None = None, ddof: int = 0, keepdims: bool = False, *,
         where: ArrayLike | None = None, correction: int | float | None = None) -> Array:
  """Compute the variance along a given axis.

  Refer to :func:`jax.numpy.var` for full documentation.
  """
  return reductions.var(self, axis=axis, dtype=dtype, out=out, ddof=ddof,
                        keepdims=keepdims, where=where, correction=correction)


def _compute_newshape(arr: Array, newshape: DimSize | Shape) -> Shape:
  """Fixes a -1 value in newshape, if present."""
  orig_newshape = newshape  # for error messages
  try:
    iter(newshape)  # type: ignore[arg-type]
  except:
    newshape = [newshape]
  newshape = core.canonicalize_shape(newshape)  # type: ignore[arg-type]
  neg1s = [i for i, d in enumerate(newshape) if type(d) is int and d == -1]
  if len(neg1s) > 1:
    raise TypeError("can only specify one unknown axis size with a `-1` value, "
                    f"got {orig_newshape}")
  if neg1s:
    i, = neg1s
    other_sizes = (*newshape[:i], *newshape[i+1:])
    if (all(isinstance(d, int) for d in (*arr.shape, *other_sizes)) and
        arr.size % math.prod(other_sizes) != 0):
      raise TypeError(f"cannot reshape array of shape {arr.shape} (size {arr.size}) "
                      f"into shape {orig_newshape} because the product of "
                      f"specified axis sizes ({math.prod(other_sizes)}) does "
                      f"not evenly divide {arr.size}")
    sz = core.cancel_divide_tracers(arr.shape, other_sizes)
    if sz is not None:
      return (*newshape[:i], sz, *newshape[i+1:])
  else:
    if (all(isinstance(d, int) for d in (*arr.shape, *newshape)) and
        arr.size != math.prod(newshape)):
      raise TypeError(f"cannot reshape array of shape {arr.shape} (size {arr.size}) "
                      f"into shape {orig_newshape} (size {math.prod(newshape)})")
  return tuple(-core.divide_shape_sizes(arr.shape, newshape)
               if core.definitely_equal(d, -1) else d for d in newshape)

def _view(self: Array, dtype: DTypeLike | None = None, type: None = None) -> Array:
  """Return a bitwise copy of the array, viewed as a new dtype.

  This is fuller-featured wrapper around :func:`jax.lax.bitcast_convert_type`.

  If the source and target dtype have the same bitwidth, the result has the same
  shape as the input array. If the bitwidth of the target dtype is different
  from the source, the size of the last axis of the result is adjusted
  accordingly.

  >>> jnp.zeros([1,2,3], dtype=jnp.int16).view(jnp.int8).shape
  (1, 2, 6)
  >>> jnp.zeros([1,2,4], dtype=jnp.int8).view(jnp.int16).shape
  (1, 2, 2)

  Conversions involving booleans are not well-defined in all situations. With
  regards to the shape of result as explained above, booleans are treated as
  having a bitwidth of 8. However, when converting to a boolean array, the input
  should only contain 0 or 1 bytes. Otherwise, results may be unpredictable or
  may change depending on how the result is used.

  This conversion is guaranteed and safe::

    >>> jnp.array([1, 0, 1], dtype=jnp.int8).view(jnp.bool_)
    Array([ True, False,  True], dtype=bool)

  However, there are no guarantees about the results of any expression involving
  a view such as this: `jnp.array([1, 2, 3], dtype=jnp.int8).view(jnp.bool_)`.
  In particular, the results may change between JAX releases and depending on
  the platform. To safely convert such an array to a boolean array, compare it
  with `0`::

    >>> jnp.array([1, 2, 0], dtype=jnp.int8) != 0
    Array([ True,  True, False], dtype=bool)
  """
  if type is not None:
    raise NotImplementedError("`type` argument of array.view() is not supported.")

  dtypes.check_user_dtype_supported(dtype, "view")
  dtype = dtypes.canonicalize_dtype(dtype)

  nbits_in = dtypes.bit_width(self.dtype)
  nbits_out = dtypes.bit_width(dtype)

  if self.ndim == 0:
    if nbits_in != nbits_out:
      raise ValueError("view() of a 0d array is only supported if the itemsize is unchanged.")
    return _view(lax.expand_dims(self, (0,)), dtype).squeeze()

  if (self.shape[-1] * nbits_in) % nbits_out != 0:
    raise ValueError("When changing to a larger dtype, its size must be a divisor "
                     "of the total size in bytes of the last axis of the array.")

  if self.dtype == dtype:
    return self

  # lax.bitcast_convert_type does not support bool or complex; in these cases we
  # cast to a compatible type and recursively call _view for simplicity.
  if self.dtype == bool:
    return _view(self.astype('uint8'), dtype)

  if lax_numpy.issubdtype(self.dtype, np.complexfloating):
    new_shape = (*self.shape[:-1], self.shape[-1] * 2)
    new_dtype = lax_numpy.finfo(self.dtype).dtype
    self = (lax_numpy.zeros(new_shape, new_dtype)
             .at[..., 0::2].set(self.real)
             .at[..., 1::2].set(self.imag))
    return _view(self, dtype)

  if dtype == bool:
    return _view(self, np.uint8).astype(bool)

  if lax_numpy.issubdtype(dtype, np.complexfloating):
    out = _view(self, lax_numpy.finfo(dtype).dtype).astype(dtype)
    return out[..., 0::2] + 1j * out[..., 1::2]

  # lax.bitcast_convert_type adds or subtracts dimensions depending on the
  # relative bitwidths of the dtypes; we account for that with reshapes.
  if nbits_in < nbits_out:
    factor = nbits_out // nbits_in
    out = self.reshape(*self.shape[:-1], self.shape[-1] // factor, factor)
    return lax.bitcast_convert_type(out, dtype)
  elif nbits_in > nbits_out:
    out = lax.bitcast_convert_type(self, dtype)
    return out.reshape(*out.shape[:-2], out.shape[-2] * out.shape[-1])
  else:
    return lax.bitcast_convert_type(self, dtype)


def _notimplemented_flat(self):
  """Not implemented: Use :meth:`~jax.Array.flatten` instead."""
  raise NotImplementedError("JAX Arrays do not implement the arr.flat property: "
                            "consider arr.flatten() instead.")

_accepted_binop_types = (int, float, complex, np.generic, np.ndarray, Array)
_rejected_binop_types = (list, tuple, set, dict)

def _defer_to_unrecognized_arg(opchar, binary_op, swap=False):
  # Ensure that other array types have the chance to override arithmetic.
  def deferring_binary_op(self, other):
    if hasattr(other, '__jax_array__'):
      other = other.__jax_array__()
    args = (other, self) if swap else (self, other)
    if isinstance(other, _accepted_binop_types):
      return binary_op(*args)
    # Note: don't use isinstance here, because we don't want to raise for
    # subclasses, e.g. NamedTuple objects that may override operators.
    if type(other) in _rejected_binop_types:
      raise TypeError(f"unsupported operand type(s) for {opchar}: "
                      f"{type(args[0]).__name__!r} and {type(args[1]).__name__!r}")
    return NotImplemented
  return deferring_binary_op

def _unimplemented_setitem(self, i, x):
  msg = ("JAX arrays are immutable and do not support in-place item assignment."
         " Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method:"
         " https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html")
  raise TypeError(msg.format(type(self)))

def _operator_round(number: ArrayLike, ndigits: int | None = None) -> Array:
  out = lax_numpy.round(number, decimals=ndigits or 0)
  # If `ndigits` is None, for a builtin float round(7.5) returns an integer.
  return out.astype(int) if ndigits is None else out

def _deepcopy(self: Array, memo: Any) -> Array:
  del memo  # unused
  return self.copy()


# Experimental support for NumPy's module dispatch with NEP-37.
# Currently requires https://github.com/seberg/numpy-dispatch
_JAX_ARRAY_TYPES = (core.Tracer, ArrayImpl)
_HANDLED_ARRAY_TYPES = _JAX_ARRAY_TYPES + (np.ndarray,)

def __array_module__(self, types):
  if all(issubclass(t, _HANDLED_ARRAY_TYPES) for t in types):
    return jax.numpy
  else:
    return NotImplemented


@partial(jax.jit, static_argnums=(1,2,3))
def _multi_slice(self: Array,
                 start_indices: tuple[tuple[int, ...]],
                 limit_indices: tuple[tuple[int, ...]],
                 removed_dims: tuple[tuple[int, ...]]) -> list[Array]:
  """Extracts multiple slices from `arr`.

  This is used to shard Array arguments to pmap. It's implemented as a
  Array method here to avoid circular imports.
  """
  results: list[Array] = []
  for starts, limits, removed in zip(start_indices, limit_indices, removed_dims):
    sliced = lax.slice(self, starts, limits)
    if removed:
      sliced = lax.squeeze(sliced, removed)
    results.append(sliced)
  return results

# The next two functions are related to iter(array), implemented here to
# avoid circular imports.
@jax.jit
def _unstack(x: Array) -> list[Array]:
  dims = (0,)
  return [lax.squeeze(t, dims) for t in lax.split(x, (1,) * x.shape[0])]

def _chunk_iter(x, size):
  if size > x.shape[0]:
    yield x
  else:
    num_chunks, tail = ufuncs.divmod(x.shape[0], size)
    for i in range(num_chunks):
      yield lax.dynamic_slice_in_dim(x, i * size, size)
    if tail:
      yield lax.dynamic_slice_in_dim(x, num_chunks * size, tail)

def _getitem(self, item):
  return lax_numpy._rewriting_take(self, item)

# Syntactic sugar for scatter operations.
class _IndexUpdateHelper:
  # Note: this docstring will appear as the docstring for the `at` property.
  """Helper property for index update functionality.

  The ``at`` property provides a functionally pure equivalent of in-place
  array modifications.

  In particular:

  ==============================  ================================
  Alternate syntax                Equivalent In-place expression
  ==============================  ================================
  ``x = x.at[idx].set(y)``        ``x[idx] = y``
  ``x = x.at[idx].add(y)``        ``x[idx] += y``
  ``x = x.at[idx].subtract(y)``   ``x[idx] -= y``
  ``x = x.at[idx].multiply(y)``   ``x[idx] *= y``
  ``x = x.at[idx].divide(y)``     ``x[idx] /= y``
  ``x = x.at[idx].power(y)``      ``x[idx] **= y``
  ``x = x.at[idx].min(y)``        ``x[idx] = minimum(x[idx], y)``
  ``x = x.at[idx].max(y)``        ``x[idx] = maximum(x[idx], y)``
  ``x = x.at[idx].apply(ufunc)``  ``ufunc.at(x, idx)``
  ``x = x.at[idx].get()``         ``x = x[idx]``
  ==============================  ================================

  None of the ``x.at`` expressions modify the original ``x``; instead they return
  a modified copy of ``x``. However, inside a :py:func:`~jax.jit` compiled function,
  expressions like :code:`x = x.at[idx].set(y)` are guaranteed to be applied in-place.

  Unlike NumPy in-place operations such as :code:`x[idx] += y`, if multiple
  indices refer to the same location, all updates will be applied (NumPy would
  only apply the last update, rather than applying all updates.) The order
  in which conflicting updates are applied is implementation-defined and may be
  nondeterministic (e.g., due to concurrency on some hardware platforms).

  By default, JAX assumes that all indices are in-bounds. Alternative out-of-bound
  index semantics can be specified via the ``mode`` parameter (see below).

  Arguments
  ---------
  mode : str
      Specify out-of-bound indexing mode. Options are:

      - ``"promise_in_bounds"``: (default) The user promises that indices are in bounds.
        No additional checking will be performed. In practice, this means that
        out-of-bounds indices in ``get()`` will be clipped, and out-of-bounds indices
        in ``set()``, ``add()``, etc. will be dropped.
      - ``"clip"``: clamp out of bounds indices into valid range.
      - ``"drop"``: ignore out-of-bound indices.
      - ``"fill"``: alias for ``"drop"``.  For `get()`, the optional ``fill_value``
        argument specifies the value that will be returned.

        See :class:`jax.lax.GatherScatterMode` for more details.

  indices_are_sorted : bool
      If True, the implementation will assume that the indices passed to ``at[]``
      are sorted in ascending order, which can lead to more efficient execution
      on some backends.
  unique_indices : bool
      If True, the implementation will assume that the indices passed to ``at[]``
      are unique, which can result in more efficient execution on some backends.
  fill_value : Any
      Only applies to the ``get()`` method: the fill value to return for out-of-bounds
      slices when `mode` is ``'fill'``. Ignored otherwise. Defaults to ``NaN`` for
      inexact types, the largest negative value for signed types, the largest positive
      value for unsigned types, and ``True`` for booleans.

  Examples
  --------
  >>> x = jnp.arange(5.0)
  >>> x
  Array([0., 1., 2., 3., 4.], dtype=float32)
  >>> x.at[2].add(10)
  Array([ 0.,  1., 12.,  3.,  4.], dtype=float32)
  >>> x.at[10].add(10)  # out-of-bounds indices are ignored
  Array([0., 1., 2., 3., 4.], dtype=float32)
  >>> x.at[20].add(10, mode='clip')
  Array([ 0.,  1.,  2.,  3., 14.], dtype=float32)
  >>> x.at[2].get()
  Array(2., dtype=float32)
  >>> x.at[20].get()  # out-of-bounds indices clipped
  Array(4., dtype=float32)
  >>> x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
  Array(nan, dtype=float32)
  >>> x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
  Array(-1., dtype=float32)
  """
  __slots__ = ("array",)

  def __init__(self, array):
    self.array = array

  def __getitem__(self, index):
    return _IndexUpdateRef(self.array, index)

  def __repr__(self):
    return f"_IndexUpdateHelper({self.array!r})"


class _IndexUpdateRef:
  """Helper object to call indexed update functions for an (advanced) index.

  This object references a source array and a specific indexer into that array.
  Methods on this object return copies of the source array that have been
  modified at the positions specified by the indexer.
  """
  __slots__ = ("array", "index")

  def __init__(self, array, index):
    self.array = array
    self.index = index

  def __repr__(self) -> str:
    return f"_IndexUpdateRef({self.array!r}, {self.index!r})"

  def get(self, *, indices_are_sorted=False, unique_indices=False,
          mode=None, fill_value=None, out_sharding=None):
    """Equivalent to ``x[idx]``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexing <numpy.doc.indexing>` ``x[idx]``. This function differs from
    the usual array indexing syntax in that it allows additional keyword
    arguments ``indices_are_sorted`` and ``unique_indices`` to be passed.

    See :mod:`jax.ops` for details.
    """
    take = partial(lax_numpy._rewriting_take,
                   indices_are_sorted=indices_are_sorted,
                   unique_indices=unique_indices, mode=mode,
                   fill_value=fill_value)
    if out_sharding is not None:
      assert isinstance(out_sharding, (NamedSharding, PartitionSpec))
      out_sharding = canonicalize_sharding(out_sharding)
      take = hidden_mode(take, axes=mesh_lib.get_abstract_mesh().axis_names,  # type: ignore
                         out_specs=out_sharding.spec)
    return take(self.array, self.index)

  def set(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:`indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values, lax.scatter,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def apply(self, func, *, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``func.at(x, idx)`` for a unary ufunc ``func``.

    Returns the value of ``x`` that would result from applying the unary
    function ``func`` to ``x`` at the given indices. This is similar to
    ``x.at[idx].set(func(x[idx]))``, but differs in the case of repeated indices:
    in ``x.at[idx].apply(func)``, repeated indices result in the function being
    applied multiple times.

    Note that in the current implementation, ``scatter_apply`` is not compatible
    with automatic differentiation.

    See :mod:`jax.ops` for details.
    """
    def _scatter_apply(x, indices, y, dims, **kwargs):
      return lax.scatter_apply(x, indices, func, dims, update_shape=y.shape, **kwargs)
    return scatter._scatter_update(self.array, self.index,
                                   lax_internal._zero(self.array.dtype),
                                   _scatter_apply,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def add(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] += y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] += y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_add,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def subtract(self, values, *, indices_are_sorted=False, unique_indices=False,
               mode=None):
    """Pure equivalent of ``x[idx] -= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] -= y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_sub,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def multiply(self, values, *, indices_are_sorted=False, unique_indices=False,
               mode=None):
    """Pure equivalent of ``x[idx] *= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_mul,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices,
                                   mode=mode)
  mul = multiply

  def divide(self, values, *, indices_are_sorted=False, unique_indices=False,
             mode=None):
    """Pure equivalent of ``x[idx] /= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] /= y``.

    See :mod:`jax.ops` for details.
    """
    return ufuncs.divide(
      self.array,
      scatter._scatter_update(lax_numpy.ones_like(self.array), self.index, values,
                              lax.scatter_mul,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices, mode=mode))

  def power(self, values, *, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``x[idx] **= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] **= y``.

    See :mod:`jax.ops` for details.
    """
    return ufuncs.power(
      self.array,
      scatter._scatter_update(lax_numpy.ones_like(self.array), self.index, values,
                              lax.scatter_mul,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices, mode=mode))

  def min(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = minimum(x[idx], y)``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = minimum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_min,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def max(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = maximum(x[idx], y)``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = maximum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_max,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

_array_operators = {
  "getitem": _getitem,
  "setitem": _unimplemented_setitem,
  "copy": _copy,
  "deepcopy": _deepcopy,
  "neg": lambda self: ufuncs.negative(self),
  "pos": lambda self: ufuncs.positive(self),
  "eq": _defer_to_unrecognized_arg("==", ufuncs.equal),
  "ne": _defer_to_unrecognized_arg("!=", ufuncs.not_equal),
  "lt": _defer_to_unrecognized_arg("<", ufuncs.less),
  "le": _defer_to_unrecognized_arg("<=", ufuncs.less_equal),
  "gt": _defer_to_unrecognized_arg(">", ufuncs.greater),
  "ge": _defer_to_unrecognized_arg(">=", ufuncs.greater_equal),
  "abs": lambda self: ufuncs.abs(self),
  "add": _defer_to_unrecognized_arg("+", ufuncs.add),
  "radd": _defer_to_unrecognized_arg("+", ufuncs.add, swap=True),
  "sub": _defer_to_unrecognized_arg("-", ufuncs.subtract),
  "rsub": _defer_to_unrecognized_arg("-", ufuncs.subtract, swap=True),
  "mul": _defer_to_unrecognized_arg("*", ufuncs.multiply),
  "rmul": _defer_to_unrecognized_arg("*", ufuncs.multiply, swap=True),
  "div": _defer_to_unrecognized_arg("/", ufuncs.divide),
  "rdiv": _defer_to_unrecognized_arg("/", ufuncs.divide, swap=True),
  "truediv": _defer_to_unrecognized_arg("/", ufuncs.true_divide),
  "rtruediv": _defer_to_unrecognized_arg("/", ufuncs.true_divide, swap=True),
  "floordiv": _defer_to_unrecognized_arg("//", ufuncs.floor_divide),
  "rfloordiv": _defer_to_unrecognized_arg("//", ufuncs.floor_divide, swap=True),
  "divmod": _defer_to_unrecognized_arg("divmod", ufuncs.divmod),
  "rdivmod": _defer_to_unrecognized_arg("divmod", ufuncs.divmod, swap=True),
  "mod": _defer_to_unrecognized_arg("%", ufuncs.mod),
  "rmod": _defer_to_unrecognized_arg("%", ufuncs.mod, swap=True),
  "pow": _defer_to_unrecognized_arg("**", ufuncs.power),
  "rpow": _defer_to_unrecognized_arg("**", ufuncs.power, swap=True),
  "matmul": _defer_to_unrecognized_arg("@", lax_numpy.matmul),
  "rmatmul": _defer_to_unrecognized_arg("@", lax_numpy.matmul, swap=True),
  "and": _defer_to_unrecognized_arg("&", ufuncs.bitwise_and),
  "rand": _defer_to_unrecognized_arg("&", ufuncs.bitwise_and, swap=True),
  "or": _defer_to_unrecognized_arg("|", ufuncs.bitwise_or),
  "ror": _defer_to_unrecognized_arg("|", ufuncs.bitwise_or, swap=True),
  "xor": _defer_to_unrecognized_arg("^", ufuncs.bitwise_xor),
  "rxor": _defer_to_unrecognized_arg("^", ufuncs.bitwise_xor, swap=True),
  "invert": lambda self: ufuncs.bitwise_not(self),
  "lshift": _defer_to_unrecognized_arg("<<", ufuncs.left_shift),
  "rshift": _defer_to_unrecognized_arg(">>", ufuncs.right_shift),
  "rlshift": _defer_to_unrecognized_arg("<<", ufuncs.left_shift, swap=True),
  "rrshift": _defer_to_unrecognized_arg(">>", ufuncs.right_shift, swap=True),
  "round": _operator_round,
}

_array_methods = {
  "__array_namespace__": array_api_metadata.__array_namespace__,
  "all": _all,
  "any": _any,
  "argmax": _argmax,
  "argmin": _argmin,
  "argpartition": _argpartition,
  "argsort": _argsort,
  "astype": _astype,
  "choose": _choose,
  "clip": _clip,
  "compress": _compress,
  "conj": _conj,
  "conjugate": _conjugate,
  "copy": _copy,
  "cumprod": _cumprod,
  "cumsum": _cumsum,
  "diagonal": _diagonal,
  "dot": _dot,
  "flatten": _flatten,
  "item": _item,
  "max": _max,
  "mean": _mean,
  "min": _min,
  "nonzero": _nonzero,
  "prod": _prod,
  "ptp": _ptp,
  "ravel": _flatten,
  "repeat": _repeat,
  "reshape": _reshape,
  "round": _round,
  "searchsorted": _searchsorted,
  "sort": _sort,
  "squeeze": _squeeze,
  "std": _std,
  "sum": _sum,
  "swapaxes": _swapaxes,
  "take": _take,
  "to_device": _to_device,
  "trace": _trace,
  "transpose": _transpose,
  "var": _var,
  "view": _view,

  # Methods exposed in order to avoid circular imports
  "_split": lax_numpy.split,  # used in jacfwd/jacrev
  "_multi_slice": _multi_slice,  # used in pxla for sharding
}

_impl_only_array_methods = {
  "_chunk_iter": _chunk_iter,
  "_unstack": _unstack,
}

_array_properties = {
  "flat": _notimplemented_flat,
  "T": _transpose_property,
  "mT": _matrix_transpose_property,
  "real": _real_property,
  "imag": _imag_property,
  "nbytes": _nbytes_property,
  "itemsize": _itemsize_property,
  "at": _IndexUpdateHelper,
}

def _set_shaped_array_attributes(shaped_array):
  # Set up operator, method, and property forwarding on Tracer instances
  # containing
  # ShapedArray avals by following the forwarding conventions for Tracer.
  # Forward operators using a single-underscore-prefix naming convention:
  for operator_name, function in _array_operators.items():
    setattr(shaped_array, f"_{operator_name}", staticmethod(function))
  # Forward methods and properties using core.{aval_method, aval_property}:
  for method_name, method in _array_methods.items():
    setattr(shaped_array, method_name, core.aval_method(method))
  for prop_name, prop in _array_properties.items():
    setattr(shaped_array, prop_name, core.aval_property(prop))
  setattr(shaped_array, "_array_module", staticmethod(__array_module__))

def _forward_operator_to_aval(name):
  def op(self, *args):
    return getattr(self.aval, f"_{name}")(self, *args)
  return op

def _forward_method_to_aval(name):
  def meth(self, *args, **kwargs):
    __tracebackhide__ = True
    return getattr(self.aval, name).fun(self, *args, **kwargs)
  return meth

def _forward_property_to_aval(name):
  @property
  def prop(self):
    return getattr(self.aval, name).fget(self)
  return prop

def _set_tracer_aval_forwarding(tracer, exclude=()):
  for operator_name in _array_operators:
    if operator_name not in exclude:
      setattr(tracer, f"__{operator_name}__", _forward_operator_to_aval(operator_name))
  for method_name in _array_methods:
    if method_name not in exclude:
      setattr(tracer, method_name, _forward_method_to_aval(method_name))
  for prop_name in _array_properties:
    if prop_name not in exclude:
      setattr(tracer, prop_name, _forward_property_to_aval(prop_name))

def _set_array_base_attributes(array_impl, include=None, exclude=None):
  # Forward operators, methods, and properties on Array to lax_numpy
  # functions (with no Tracers involved; this forwarding is direct)
  def maybe_setattr(attr_name, target):
    if exclude is not None and attr_name in exclude:
      return
    if not include or attr_name in include:
      setattr(array_impl, attr_name, target)

  for operator_name, function in _array_operators.items():
    maybe_setattr(f"__{operator_name}__", function)
  for method_name, method in _array_methods.items():
    maybe_setattr(method_name, method)
  for prop_name, prop in _array_properties.items():
    maybe_setattr(prop_name, property(prop))

  for name, func in _impl_only_array_methods.items():
    setattr(array_impl, name, func)

def _set_array_attributes(array_impl):
  setattr(array_impl, "__array_module__", __array_module__)

def _make_abstract_method(name, func):
  @abc.abstractmethod
  @wraps(func)
  def method(*args, **kwargs):
    raise NotImplementedError(f"Cannot call abstract method {name}")
  return method

def _set_array_abstract_methods(basearray):
  for operator_name, function in _array_operators.items():
    setattr(basearray, f"__{operator_name}__",
            _make_abstract_method(f"__{operator_name}__", function))
  for method_name, method in _array_methods.items():
    setattr(basearray, method_name,
            _make_abstract_method(method_name, method))
  for prop_name, prop in _array_properties.items():
    setattr(basearray, prop_name,
            property(_make_abstract_method(prop_name, prop)))

def register_jax_array_methods():
  """Call this function once to register methods of JAX arrays"""
  _set_shaped_array_attributes(core.ShapedArray)
  _set_shaped_array_attributes(core.DShapedArray)

  _set_array_base_attributes(ArrayImpl, exclude={'__getitem__'})
  _set_tracer_aval_forwarding(core.Tracer, exclude={*_impl_only_array_methods, "at"})
  _set_array_attributes(ArrayImpl)

  _set_array_abstract_methods(Array)
