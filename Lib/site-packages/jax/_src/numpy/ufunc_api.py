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

"""Tools to create numpy-style ufuncs."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
import operator
from typing import Any

import jax
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.lax import lax as lax_internal
import jax._src.numpy.lax_numpy as jnp
from jax._src.numpy.reductions import _moveaxis
from jax._src.numpy.util import check_arraylike, _broadcast_to, _where
from jax._src.numpy.vectorize import vectorize
from jax._src.util import canonicalize_axis, set_module
import numpy as np


export = set_module("jax.numpy")

_AT_INPLACE_WARNING = """\
Because JAX arrays are immutable, jnp.ufunc.at() cannot operate inplace like
np.ufunc.at(). Instead, you can pass inplace=False and capture the result; e.g.
>>> arr = jnp.add.at(arr, ind, val, inplace=False)
"""


@export
class ufunc:
  """Universal functions which operation element-by-element on arrays.

  JAX implementation of :class:`numpy.ufunc`.

  This is a class for JAX-backed implementations of NumPy's ufunc APIs.
  Most users will never need to instantiate :class:`ufunc`, but rather
  will use the pre-defined ufuncs in :mod:`jax.numpy`.

  For constructing your own ufuncs, see :func:`jax.numpy.frompyfunc`.

  Examples:
    Universal functions are functions that apply element-wise to broadcasted
    arrays, but they also come with a number of extra attributes and methods.

    As an example, consider the function :obj:`jax.numpy.add`. The object
    acts as a function that applies addition to broadcasted arrays in an
    element-wise manner:

    >>> x = jnp.array([1, 2, 3, 4, 5])
    >>> jnp.add(x, 1)
    Array([2, 3, 4, 5, 6], dtype=int32)

    Each :class:`ufunc` object includes a number of attributes that describe
    its behavior:

    >>> jnp.add.nin  # number of inputs
    2
    >>> jnp.add.nout  # number of outputs
    1
    >>> jnp.add.identity  # identity value, or None if no identity exists
    0

    Binary ufuncs like :obj:`jax.numpy.add` include  number of methods to
    apply the function to arrays in different manners.

    The :meth:`~ufunc.outer` method applies the function to the
    pair-wise outer-product of the input array values:

    >>> jnp.add.outer(x, x)
    Array([[ 2,  3,  4,  5,  6],
           [ 3,  4,  5,  6,  7],
           [ 4,  5,  6,  7,  8],
           [ 5,  6,  7,  8,  9],
           [ 6,  7,  8,  9, 10]], dtype=int32)

    The :meth:`ufunc.reduce` method perfoms a reduction over the array.
    For example, :meth:`jnp.add.reduce` is equivalent to ``jnp.sum``:

    >>> jnp.add.reduce(x)
    Array(15, dtype=int32)

    The :meth:`ufunc.accumulate` method performs a cumulative reduction
    over the array. For example, :meth:`jnp.add.accumulate` is equivalent
    to :func:`jax.numpy.cumulative_sum`:

    >>> jnp.add.accumulate(x)
    Array([ 1,  3,  6, 10, 15], dtype=int32)

    The :meth:`ufunc.at` method applies the function at particular indices in the
    array; for ``jnp.add`` the computation is similar to :func:`jax.lax.scatter_add`:

    >>> jnp.add.at(x, 0, 100, inplace=False)
    Array([101,   2,   3,   4,   5], dtype=int32)

    And the :meth:`ufunc.reduceat` method performs a number of ``reduce``
    operations bewteen specified indices of an array; for ``jnp.add`` the
    operation is similar to :func:`jax.ops.segment_sum`:

    >>> jnp.add.reduceat(x, jnp.array([0, 2]))
    Array([ 3, 12], dtype=int32)

    In this case, the first element is ``x[0:2].sum()``, and the second element
    is ``x[2:].sum()``.
  """
  def __init__(self, func: Callable[..., Any], /,
               nin: int, nout: int, *,
               name: str | None = None,
               nargs: int | None = None,
               identity: Any = None,
               call: Callable[..., Any] | None = None,
               reduce: Callable[..., Any] | None = None,
               accumulate: Callable[..., Any] | None = None,
               at: Callable[..., Any] | None = None,
               reduceat: Callable[..., Any] | None = None,
               ):
    self.__doc__ = func.__doc__
    self.__name__ = name or func.__name__
    # We want ufunc instances to work properly when marked as static,
    # and for this reason it's important that their properties not be
    # mutated. We prevent this by storing them in a dunder attribute,
    # and accessing them via read-only properties.
    self.__static_props = {
      'func': func,
      'nin': operator.index(nin),
      'nout': operator.index(nout),
      'nargs': operator.index(nargs or nin),
      'identity': identity,
      'call': call,
      'reduce': reduce,
      'accumulate': accumulate,
      'at': at,
      'reduceat': reduceat,
    }

  _func = property(lambda self: self.__static_props['func'])
  nin = property(lambda self: self.__static_props['nin'])
  nout = property(lambda self: self.__static_props['nout'])
  nargs = property(lambda self: self.__static_props['nargs'])
  identity = property(lambda self: self.__static_props['identity'])

  def __hash__(self) -> int:
    # In both __hash__ and __eq__, we do not consider call, reduce, etc.
    # because they are considered implementation details rather than
    # necessary parts of object identity.
    return hash((self._func, self.__name__, self.identity,
                 self.nin, self.nout, self.nargs))

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, ufunc) and (
      (self._func, self.__name__, self.identity, self.nin, self.nout, self.nargs) ==
      (other._func, other.__name__, other.identity, other.nin, other.nout, other.nargs))

  def __repr__(self) -> str:
    return f"<jnp.ufunc '{self.__name__}'>"

  def __call__(self, *args: ArrayLike, out: None = None, where: None = None) -> Any:
    check_arraylike(self.__name__, *args)
    if out is not None:
      raise NotImplementedError(f"out argument of {self}")
    if where is not None:
      raise NotImplementedError(f"where argument of {self}")
    call = self.__static_props['call'] or self._call_vectorized
    return call(*args)

  @partial(jax.jit, static_argnames=['self'])
  def _call_vectorized(self, *args):
    return vectorize(self._func)(*args)

  @partial(jax.jit, static_argnames=['self', 'axis', 'dtype', 'out', 'keepdims'])
  def reduce(self, a: ArrayLike, axis: int = 0,
             dtype: DTypeLike | None = None,
             out: None = None, keepdims: bool = False, initial: ArrayLike | None = None,
             where: ArrayLike | None = None) -> Array:
    """Reduction operation derived from a binary function.

    JAX implementation of :meth:`numpy.ufunc.reduce`.

    Args:
      a: Input array.
      axis: integer specifying the axis over which to reduce. default=0
      dtype: optionally specify the type of the output array.
      out: Unused by JAX
      keepdims: If True, reduced axes are left in the result with size 1.
        If False (default) then reduced axes are squeezed out.
      initial: int or array, Default=None. Initial value for the reduction.
      where: boolean mask, default=None. The elements to be used in the sum. Array
        should be broadcast compatible to the input.

    Returns:
      array containing the result of the reduction operation.

    Examples:
      Consider the following array:

      >>> x = jnp.array([[1, 2, 3],
      ...                [4, 5, 6]])

      :meth:`jax.numpy.add.reduce` is equivalent to :func:`jax.numpy.sum`
      along ``axis=0``:

      >>> jnp.add.reduce(x)
      Array([5, 7, 9], dtype=int32)
      >>> x.sum(0)
      Array([5, 7, 9], dtype=int32)

      Similarly, :meth:`jax.numpy.logical_and.reduce` is equivalent to
      :func:`jax.numpy.all`:

      >>> jnp.logical_and.reduce(x > 2)
      Array([False, False,  True], dtype=bool)
      >>> jnp.all(x > 2, axis=0)
      Array([False, False,  True], dtype=bool)

      Some reductions do not correspond to any built-in aggregation function;
      for example here is the reduction of :func:`jax.numpy.bitwise_or` along
      the first axis of ``x``:

      >>> jnp.bitwise_or.reduce(x, axis=1)
      Array([3, 7], dtype=int32)
    """
    check_arraylike(f"{self.__name__}.reduce", a)
    if self.nin != 2:
      raise ValueError("reduce only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("reduce only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.reduce()")
    if initial is not None:
      check_arraylike(f"{self.__name__}.reduce", initial)
    if where is not None:
      check_arraylike(f"{self.__name__}.reduce", where)
      if self.identity is None and initial is None:
        raise ValueError(f"reduction operation {self.__name__!r} does not have an identity, "
                         "so to use a where mask one has to specify 'initial'.")
      if lax_internal._dtype(where) != bool:
        raise ValueError(f"where argument must have dtype=bool; got dtype={lax_internal._dtype(where)}")
    reduce = self.__static_props['reduce'] or self._reduce_via_scan
    return reduce(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)

  def _reduce_via_scan(self, arr: ArrayLike, axis: int | None = 0, dtype: DTypeLike | None = None,
                       keepdims: bool = False, initial: ArrayLike | None = None,
                       where: ArrayLike | None = None) -> Array:
    assert self.nin == 2 and self.nout == 1
    arr = lax_internal.asarray(arr)
    if initial is None:
      initial = self.identity
    if dtype is None:
      dtype = jax.eval_shape(self._func, lax_internal._one(arr), lax_internal._one(arr)).dtype
    if where is not None:
      where = _broadcast_to(where, arr.shape)
    if isinstance(axis, tuple):
      axis = tuple(canonicalize_axis(a, arr.ndim) for a in axis)
      raise NotImplementedError("tuple of axes")
    elif axis is None:
      if keepdims:
        final_shape = (1,) * arr.ndim
      else:
        final_shape = ()
      arr = arr.ravel()
      if where is not None:
        where = where.ravel()
      axis = 0
    else:
      axis = canonicalize_axis(axis, arr.ndim)
      if keepdims:
        final_shape = (*arr.shape[:axis], 1, *arr.shape[axis + 1:])
      else:
        final_shape = (*arr.shape[:axis], *arr.shape[axis + 1:])

    # TODO: handle without transpose?
    if axis != 0:
      arr = _moveaxis(arr, axis, 0)
      if where is not None:
        where = _moveaxis(where, axis, 0)

    if initial is None and arr.shape[0] == 0:
      raise ValueError("zero-size array to reduction operation {self.__name__} which has no ideneity")

    def body_fun(i, val):
      if where is None:
        return self(val, arr[i].astype(dtype))
      else:
        return _where(where[i], self(val, arr[i].astype(dtype)), val)

    start_value: ArrayLike
    if initial is None:
      start_index = 1
      start_value = arr[0]
    else:
      start_index = 0
      start_value = initial
    start_value = _broadcast_to(lax_internal.asarray(start_value).astype(dtype), arr.shape[1:])

    result = jax.lax.fori_loop(start_index, arr.shape[0], body_fun, start_value)

    if keepdims:
      result = result.reshape(final_shape)
    return result

  @partial(jax.jit, static_argnames=['self', 'axis', 'dtype'])
  def accumulate(self, a: ArrayLike, axis: int = 0, dtype: DTypeLike | None = None,
                 out: None = None) -> Array:
    """Accumulate operation derived from binary ufunc.

    JAX implementation of :func:`numpy.ufunc.accumulate`.

    Args:
      a: N-dimensional array over which to accumulate.
      axis: integer axis over which accumulation will be performed (default = 0)
      dtype: optionally specify the type of the output array.
      out: Unused by JAX

    Returns:
      An array containing the accumulated result.

    Examples:
      Consider the following array:

      >>> x = jnp.array([[1, 2, 3],
      ...                [4, 5, 6]])

      :meth:`jax.numpy.add.accumulate` is equivalent to
      :func:`jax.numpy.cumsum` along the specified axis:
      >>> jnp.add.accumulate(x, axis=1)
      Array([[ 1,  3,  6],
             [ 4,  9, 15]], dtype=int32)
      >>> jnp.cumsum(x, axis=1)
      Array([[ 1,  3,  6],
             [ 4,  9, 15]], dtype=int32)

      Similarly, :meth:`jax.numpy.multiply.accumulate` is equivalent to
      :func:`jax.numpy.cumprod` along the specified axis:

      >>> jnp.multiply.accumulate(x, axis=1)
      Array([[  1,   2,   6],
             [  4,  20, 120]], dtype=int32)
      >>> jnp.cumprod(x, axis=1)
      Array([[  1,   2,   6],
             [  4,  20, 120]], dtype=int32)

      For other binary ufuncs, the accumulation is an operation not available
      via standard APIs. For example, :meth:`jax.numpy.bitwise_or.accumulate`
      is essentially a bitwise cumulative ``any``:

      >>> jnp.bitwise_or.accumulate(x, axis=1)
      Array([[1, 3, 3],
             [4, 5, 7]], dtype=int32)
    """
    if self.nin != 2:
      raise ValueError("accumulate only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("accumulate only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.accumulate()")
    accumulate = self.__static_props['accumulate'] or self._accumulate_via_scan
    return accumulate(a, axis=axis, dtype=dtype)

  def _accumulate_via_scan(self, arr: ArrayLike, axis: int = 0,
                           dtype: DTypeLike | None = None) -> Array:
    assert self.nin == 2 and self.nout == 1
    check_arraylike(f"{self.__name__}.accumulate", arr)
    arr = lax_internal.asarray(arr)

    if dtype is None:
      dtype = jax.eval_shape(self._func, lax_internal._one(arr), lax_internal._one(arr)).dtype

    if axis is None or isinstance(axis, tuple):
      raise ValueError("accumulate does not allow multiple axes")
    axis = canonicalize_axis(axis, np.ndim(arr))

    arr = _moveaxis(arr, axis, 0)
    def scan_fun(carry, _):
      i, x = carry
      y = _where(i == 0, arr[0].astype(dtype), self(x.astype(dtype), arr[i].astype(dtype)))
      return (i + 1, y), y
    _, result = jax.lax.scan(scan_fun, (0, arr[0].astype(dtype)), None, length=arr.shape[0])
    return _moveaxis(result, 0, axis)

  @partial(jax.jit, static_argnums=[0], static_argnames=['inplace'])
  def at(self, a: ArrayLike, indices: Any, b: ArrayLike | None = None, /, *,
         inplace: bool = True) -> Array:
    """Update elements of an array via the specified unary or binary ufunc.

    JAX implementation of :func:`numpy.ufunc.at`.

    Note:
      :meth:`numpy.ufunc.at` mutates arrays in-place. JAX arrays are immutable,
      so :meth:`jax.numpy.ufunc.at` cannot replicate these semantics. Instead, JAX
      will return the updated value, but requires explicitly passing ``inplace=False``
      as a reminder of this difference.

    Args:
      a: N-dimensional array to update
      indices: index, slice, or tuple of indices and slices.
      b: array of values for binary ufunc updates.
      inplace: must be set to False to indicate that an updated copy will be returned.

    Returns:
      an updated copy of the input array.

    Examples:

      Add numbers to specified indices:

      >>> x = jnp.ones(10, dtype=int)
      >>> indices = jnp.array([2, 5, 7])
      >>> values = jnp.array([10, 20, 30])
      >>> jnp.add.at(x, indices, values, inplace=False)
      Array([ 1,  1, 11,  1,  1, 21,  1, 31,  1,  1], dtype=int32)

      This is roughly equivalent to JAX's :meth:`jax.numpy.ndarray.at` method
      called this way:

      >>> x.at[indices].add(values)
      Array([ 1,  1, 11,  1,  1, 21,  1, 31,  1,  1], dtype=int32)
    """
    if inplace:
      raise NotImplementedError(_AT_INPLACE_WARNING)

    at = self.__static_props['at'] or self._at_via_scan
    return at(a, indices) if b is None else at(a, indices, b)

  def _at_via_scan(self, a: ArrayLike, indices: Any, *args: Any) -> Array:
    assert len(args) in {0, 1}
    check_arraylike(f"{self.__name__}.at", a, *args)
    dtype = jax.eval_shape(self._func, lax_internal._one(a), *(lax_internal._one(arg) for arg in args)).dtype
    a = lax_internal.asarray(a).astype(dtype)
    args = tuple(lax_internal.asarray(arg).astype(dtype) for arg in args)
    indices = jnp._eliminate_deprecated_list_indexing(indices)
    if not indices:
      return a

    shapes = [np.shape(i) for i in indices if not isinstance(i, slice)]
    shape = shapes and jax.lax.broadcast_shapes(*shapes)
    if not shape:
      return a.at[indices].set(self(a.at[indices].get(), *args))

    if args:
      arg = _broadcast_to(args[0], (*shape, *args[0].shape[len(shape):]))
      args = (arg.reshape(math.prod(shape), *args[0].shape[len(shape):]),)
    indices = [idx if isinstance(idx, slice) else _broadcast_to(idx, shape).ravel() for idx in indices]

    def scan_fun(carry, x):
      i, a = carry
      idx = tuple(ind if isinstance(ind, slice) else ind[i] for ind in indices)
      a = a.at[idx].set(self(a.at[idx].get(), *(arg[i] for arg in args)))
      return (i + 1, a), x
    carry, _ = jax.lax.scan(scan_fun, (0, a), None, len(indices[0]))
    return carry[1]

  @partial(jax.jit, static_argnames=['self', 'axis', 'dtype'])
  def reduceat(self, a: ArrayLike, indices: Any, axis: int = 0,
               dtype: DTypeLike | None = None, out: None = None) -> Array:
    """Reduce an array between specified indices via a binary ufunc.

    JAX implementation of :meth:`numpy.ufunc.reduceat`

    Args:
      a: N-dimensional array to reduce
      indices: a 1-dimensional array of increasing integer values which encodes
        segments of the array to be reduced.
      axis: integer specifying the axis along which to reduce: default=0.
      dtype: optionally specify the dtype of the output array.
      out: unused by JAX
    Returns:
      An array containing the reduced values.

    Examples:
      The ``reduce`` method lets you efficiently compute reduction operations
      over array segments. For example:

      >>> x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
      >>> indices = jnp.array([0, 2, 5])
      >>> jnp.add.reduce(x, indices)
      Array([ 3, 12, 21], dtype=int32)

      This is more-or-less equivalent to the following:

      >>> jnp.array([x[0:2].sum(), x[2:5].sum(), x[5:].sum()])
      Array([ 3, 12, 21], dtype=int32)

      For some binary ufuncs, JAX provides similar APIs within :mod:`jax.ops`.
      For example, :meth:`jax.add.reduceat` is similar to :func:`jax.ops.segment_sum`,
      although in this case the segments are defined via an array of segment ids:

      >>> segments = jnp.array([0, 0, 1, 1, 1, 2, 2, 2])
      >>> jax.ops.segment_sum(x, segments)
      Array([ 3, 12, 21], dtype=int32)
    """
    if self.nin != 2:
      raise ValueError("reduceat only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("reduceat only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.reduceat()")

    reduceat = self.__static_props['reduceat'] or self._reduceat_via_scan
    return reduceat(a, indices, axis=axis, dtype=dtype)

  def _reduceat_via_scan(self, a: ArrayLike, indices: Any, axis: int = 0,
                         dtype: DTypeLike | None = None) -> Array:
    check_arraylike(f"{self.__name__}.reduceat", a, indices)
    a = lax_internal.asarray(a)
    idx_tuple = jnp._eliminate_deprecated_list_indexing(indices)
    assert len(idx_tuple) == 1
    indices = idx_tuple[0]
    if a.ndim == 0:
      raise ValueError(f"reduceat: a must have 1 or more dimension, got {a.shape=}")
    if indices.ndim != 1:
      raise ValueError(f"reduceat: indices must be one-dimensional, got {indices.shape=}")
    if dtype is None:
      dtype = a.dtype
    if axis is None or isinstance(axis, (tuple, list)):
      raise ValueError("reduceat requires a single integer axis.")
    axis = canonicalize_axis(axis, a.ndim)
    out = jnp.take(a, indices, axis=axis)
    ind = jax.lax.expand_dims(jnp.append(indices, a.shape[axis]),
                              list(np.delete(np.arange(out.ndim), axis)))
    ind_start = jax.lax.slice_in_dim(ind, 0, ind.shape[axis] - 1, axis=axis)
    ind_end = jax.lax.slice_in_dim(ind, 1, ind.shape[axis], axis=axis)
    def loop_body(i, out):
      return _where((i > ind_start) & (i < ind_end),
                    self(out, jnp.take(a, jax.lax.expand_dims(i, (0,)), axis=axis)),
                    out)
    return jax.lax.fori_loop(0, a.shape[axis], loop_body, out)

  @partial(jax.jit, static_argnums=[0])
  def outer(self, A: ArrayLike, B: ArrayLike, /) -> Array:
    """Apply the function to all pairs of values in ``A`` and ``B``.

    JAX implementation of :meth:`numpy.ufunc.outer`.

    Args:
      A: N-dimensional array
      B: N-dimensional array

    Returns:
      An array of shape `tuple(*A.shape, *B.shape)`

    Examples:
      A times-table for integers 1...10 created via
      :meth:`jax.numpy.multiply.outer`:

      >>> x = jnp.arange(1, 11)
      >>> print(jnp.multiply.outer(x, x))
      [[  1   2   3   4   5   6   7   8   9  10]
       [  2   4   6   8  10  12  14  16  18  20]
       [  3   6   9  12  15  18  21  24  27  30]
       [  4   8  12  16  20  24  28  32  36  40]
       [  5  10  15  20  25  30  35  40  45  50]
       [  6  12  18  24  30  36  42  48  54  60]
       [  7  14  21  28  35  42  49  56  63  70]
       [  8  16  24  32  40  48  56  64  72  80]
       [  9  18  27  36  45  54  63  72  81  90]
       [ 10  20  30  40  50  60  70  80  90 100]]

      For input arrays with ``N`` and ``M`` dimensions respectively, the output
      will have dimesion ``N + M``:

      >>> x = jnp.ones((1, 3, 5))
      >>> y = jnp.ones((2, 4))
      >>> jnp.add.outer(x, y).shape
      (1, 3, 5, 2, 4)
    """
    if self.nin != 2:
      raise ValueError("outer only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("outer only supported for functions returning a single value")
    check_arraylike(f"{self.__name__}.outer", A, B)
    _ravel = lambda A: jax.lax.reshape(A, (np.size(A),))
    result = jax.vmap(jax.vmap(self, (None, 0)), (0, None))(_ravel(A), _ravel(B))
    return result.reshape(*np.shape(A), *np.shape(B))


@export
def frompyfunc(func: Callable[..., Any], /, nin: int, nout: int,
               *, identity: Any = None) -> ufunc:
  """Create a JAX ufunc from an arbitrary JAX-compatible scalar function.

  Args:
    func : a callable that takes `nin` scalar arguments and returns `nout` outputs.
    nin: integer specifying the number of scalar inputs
    nout: integer specifying the number of scalar outputs
    identity: (optional) a scalar specifying the identity of the operation, if any.

  Returns:
    wrapped : jax.numpy.ufunc wrapper of func.

  Examples:
    Here is an example of creating a ufunc similar to :obj:`jax.numpy.add`:

    >>> import operator
    >>> add = frompyfunc(operator.add, nin=2, nout=1, identity=0)

    Now all the standard :class:`jax.numpy.ufunc` methods are available:

    >>> x = jnp.arange(4)
    >>> add(x, 10)
    Array([10, 11, 12, 13], dtype=int32)
    >>> add.outer(x, x)
    Array([[0, 1, 2, 3],
           [1, 2, 3, 4],
           [2, 3, 4, 5],
           [3, 4, 5, 6]], dtype=int32)
    >>> add.reduce(x)
    Array(6, dtype=int32)
    >>> add.accumulate(x)
    Array([0, 1, 3, 6], dtype=int32)
    >>> add.at(x, 1, 10, inplace=False)
    Array([ 0, 11,  2,  3], dtype=int32)
  """
  return ufunc(func, nin, nout, identity=identity)
