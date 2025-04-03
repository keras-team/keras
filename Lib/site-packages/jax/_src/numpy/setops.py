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

from functools import partial
import math
import operator
from typing import cast, NamedTuple

import numpy as np

import jax
from jax import jit
from jax import lax

from jax._src import core
from jax._src import dtypes
from jax._src.lax import lax as lax_internal
from jax._src.numpy.lax_numpy import (
    append, arange, array, asarray, concatenate, diff,
    empty, full_like, lexsort, moveaxis, nonzero, ones, ravel,
    sort, where, zeros)
from jax._src.numpy.reductions import any, cumsum
from jax._src.numpy.ufuncs import isnan
from jax._src.numpy.util import check_arraylike, promote_dtypes
from jax._src.util import canonicalize_axis, set_module
from jax._src.typing import Array, ArrayLike


export = set_module('jax.numpy')

_lax_const = lax_internal._const


@partial(jit, static_argnames=('assume_unique', 'invert', 'method'))
def _in1d(ar1: ArrayLike, ar2: ArrayLike, invert: bool,
          method='auto', assume_unique=False) -> Array:
  check_arraylike("in1d", ar1, ar2)
  arr1, arr2 = promote_dtypes(ar1, ar2)
  arr1, arr2 = arr1.ravel(), arr2.ravel()
  if arr1.size == 0 or arr2.size == 0:
    return (ones if invert else zeros)(arr1.shape, dtype=bool)
  if method in ['auto', 'compare_all']:
    if invert:
      return (arr1[:, None] != arr2[None, :]).all(-1)
    else:
      return (arr1[:, None] == arr2[None, :]).any(-1)
  elif method == 'binary_search':
    arr2 = lax.sort(arr2)
    ind = jax.numpy.searchsorted(arr2, arr1)
    if invert:
      return arr1 != arr2[ind]
    else:
      return arr1 == arr2[ind]
  elif method == 'sort':
    if assume_unique:
      ind_out: slice | Array = slice(None)
    else:
      arr1, ind_out = unique(arr1, size=len(arr1), return_inverse=True, fill_value=arr2.max())
    aux, ind = lax.sort_key_val(concatenate([arr1, arr2]), arange(arr1.size + arr2.size))
    if invert:
      return ones(arr1.shape, bool).at[ind[:-1]].set(aux[1:] != aux[:-1], mode='drop')[ind_out]
    else:
      return zeros(arr1.shape, bool).at[ind[:-1]].set(aux[1:] == aux[:-1], mode='drop')[ind_out]
  else:
    raise ValueError(f"{method=} is not implemented; options are "
                     "'compare_all', 'binary_search', 'sort', and 'auto'")


def _concat_unique(arr1: Array, arr2: Array) -> tuple[Array, Array]:
  """Utility to concatenate the unique values from two arrays."""
  arr1, arr2 = ravel(arr1), ravel(arr2)
  arr1, num_unique1 = _unique(arr1, axis=0, size=arr1.size, return_true_size=True)
  arr2, num_unique2 = _unique(arr2, axis=0, size=arr2.size, return_true_size=True)
  arr = zeros(arr1.size + arr2.size, dtype=dtypes.result_type(arr1, arr2))
  arr = lax.dynamic_update_slice(arr, arr1, (0,))
  arr = lax.dynamic_update_slice(arr, arr2, (num_unique1,))
  return arr, num_unique1 + num_unique2


@export
def setdiff1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False,
              *, size: int | None = None, fill_value: ArrayLike | None = None) -> Array:
  """Compute the set difference of two 1D arrays.

  JAX implementation of :func:`numpy.setdiff1d`.

  Because the size of the output of ``setdiff1d`` is data-dependent, the function
  is not typically compatible with :func:`~jax.jit` and other JAX transformations.
  The JAX version adds the optional ``size`` argument which must be specified statically
  for ``jnp.setdiff1d`` to be used in such contexts.

  Args:
    ar1: first array of elements to be differenced.
    ar2: second array of elements to be differenced.
    assume_unique: if True, assume the input arrays contain unique values. This allows
      a more efficient implementation, but if ``assume_unique`` is True and the input
      arrays contain duplicates, the behavior is undefined. default: False.
    size: if specified, return only the first ``size`` sorted elements. If there are fewer
      elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the minimum value.

  Returns:
    an array containing the set difference of elements in the input array: i.e. the elements
    in ``ar1`` that are not contained in ``ar2``.

  See also:
    - :func:`jax.numpy.intersect1d`: the set intersection of two 1D arrays.
    - :func:`jax.numpy.setxor1d`: the set XOR of two 1D arrays.
    - :func:`jax.numpy.union1d`: the set union of two 1D arrays.

  Examples:
    Computing the set difference of two arrays:

    >>> ar1 = jnp.array([1, 2, 3, 4])
    >>> ar2 = jnp.array([3, 4, 5, 6])
    >>> jnp.setdiff1d(ar1, ar2)
    Array([1, 2], dtype=int32)

    Because the output shape is dynamic, this will fail under :func:`~jax.jit` and other
    transformations:

    >>> jax.jit(jnp.setdiff1d)(ar1, ar2)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
       ...
    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[4].
    The error occurred while tracing the function setdiff1d at /Users/vanderplas/github/jax-ml/jax/jax/_src/numpy/setops.py:64 for jit. This concrete value was not available in Python because it depends on the value of the argument ar1.

    In order to ensure statically-known output shapes, you can pass a static ``size``
    argument:

    >>> jit_setdiff1d = jax.jit(jnp.setdiff1d, static_argnames=['size'])
    >>> jit_setdiff1d(ar1, ar2, size=2)
    Array([1, 2], dtype=int32)

    If ``size`` is too small, the difference is truncated:

    >>> jit_setdiff1d(ar1, ar2, size=1)
    Array([1], dtype=int32)

    If ``size`` is too large, then the output is padded with ``fill_value``:

    >>> jit_setdiff1d(ar1, ar2, size=4, fill_value=0)
    Array([1, 2, 0, 0], dtype=int32)
  """
  check_arraylike("setdiff1d", ar1, ar2)
  if size is None:
    ar1 = core.concrete_or_error(None, ar1, "The error arose in setdiff1d()")
  else:
    size = core.concrete_or_error(operator.index, size, "The error arose in setdiff1d()")
  arr1 = asarray(ar1)
  fill_value = asarray(0 if fill_value is None else fill_value, dtype=arr1.dtype)
  if arr1.size == 0:
    return full_like(arr1, fill_value, shape=size or 0)
  if not assume_unique:
    arr1 = cast(Array, unique(arr1, size=size and arr1.size))
  mask = _in1d(arr1, ar2, invert=True, assume_unique=assume_unique)
  if size is None:
    return arr1[mask]
  else:
    if not (assume_unique or size is None):
      # Set mask to zero at locations corresponding to unique() padding.
      n_unique = arr1.size + 1 - (arr1 == arr1[0]).sum()
      mask = where(arange(arr1.size) < n_unique, mask, False)
    return where(arange(size) < mask.sum(), arr1[where(mask, size=size)], fill_value)


@export
def union1d(ar1: ArrayLike, ar2: ArrayLike,
            *, size: int | None = None, fill_value: ArrayLike | None = None) -> Array:
  """Compute the set union of two 1D arrays.

  JAX implementation of :func:`numpy.union1d`.

  Because the size of the output of ``union1d`` is data-dependent, the function
  is not typically compatible with :func:`~jax.jit` and other JAX transformations.
  The JAX version adds the optional ``size`` argument which must be specified
  statically for ``jnp.union1d`` to be used in such contexts.

  Args:
    ar1: first array of elements to be unioned.
    ar2: second array of elements to be unioned
    size: if specified, return only the first ``size`` sorted elements. If there are fewer
      elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the minimum value.

  Returns:
    an array containing the union of elements in the input array.

  See also:
    - :func:`jax.numpy.intersect1d`: the set intersection of two 1D arrays.
    - :func:`jax.numpy.setxor1d`: the set XOR of two 1D arrays.
    - :func:`jax.numpy.setdiff1d`: the set difference of two 1D arrays.

  Examples:
    Computing the union of two arrays:

    >>> ar1 = jnp.array([1, 2, 3, 4])
    >>> ar2 = jnp.array([3, 4, 5, 6])
    >>> jnp.union1d(ar1, ar2)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

    Because the output shape is dynamic, this will fail under :func:`~jax.jit` and other
    transformations:

    >>> jax.jit(jnp.union1d)(ar1, ar2)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
       ...
    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[4].
    The error occurred while tracing the function union1d at /Users/vanderplas/github/jax-ml/jax/jax/_src/numpy/setops.py:101 for jit. This concrete value was not available in Python because it depends on the value of the argument ar1.

    In order to ensure statically-known output shapes, you can pass a static ``size``
    argument:

    >>> jit_union1d = jax.jit(jnp.union1d, static_argnames=['size'])
    >>> jit_union1d(ar1, ar2, size=6)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

    If ``size`` is too small, the union is truncated:

    >>> jit_union1d(ar1, ar2, size=4)
    Array([1, 2, 3, 4], dtype=int32)

    If ``size`` is too large, then the output is padded with ``fill_value``:

    >>> jit_union1d(ar1, ar2, size=8, fill_value=0)
    Array([1, 2, 3, 4, 5, 6, 0, 0], dtype=int32)
  """
  check_arraylike("union1d", ar1, ar2)
  if size is None:
    ar1 = core.concrete_or_error(None, ar1, "The error arose in union1d()")
    ar2 = core.concrete_or_error(None, ar2, "The error arose in union1d()")
  else:
    size = core.concrete_or_error(operator.index, size, "The error arose in union1d()")
  out = unique(concatenate((ar1, ar2), axis=None), size=size,
               fill_value=fill_value)
  return cast(Array, out)


@partial(jit, static_argnames=['assume_unique', 'size'])
def _setxor1d_size(arr1: Array, arr2: Array, fill_value: ArrayLike | None, *,
                   assume_unique: bool, size: int, ) -> Array:
  # Ensured by caller
  assert arr1.ndim == arr2.ndim == 1
  assert arr1.dtype == arr2.dtype

  if assume_unique:
    arr = concatenate([arr1, arr2])
    aux = sort(concatenate([arr1, arr2]))
    flag = concatenate((bool(aux.size), aux[1:] != aux[:-1], True), axis=None)
  else:
    arr, num_unique = _concat_unique(arr1, arr2)
    mask = arange(arr.size + 1) < num_unique + 1
    _, aux = lax.sort([~mask[1:], arr], is_stable=True, num_keys=2)
    flag = mask & concatenate((bool(aux.size), aux[1:] != aux[:-1], False),
                              axis=None).at[num_unique].set(True)
  aux_mask = flag[1:] & flag[:-1]
  num_results = aux_mask.sum()
  if aux.size:
    indices = nonzero(aux_mask, size=size, fill_value=len(aux))[0]
    vals = aux.at[indices].get(mode='fill', fill_value=0)
  else:
    vals = zeros(size, aux.dtype)
  if fill_value is None:
    vals = where(arange(len(vals)) < num_results, vals, vals.max())
    return where(arange(len(vals)) < num_results, vals, vals.min())
  else:
    return where(arange(len(vals)) < num_results, vals, fill_value)


@export
def setxor1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False, *,
             size: int | None = None, fill_value: ArrayLike | None = None) -> Array:
  """Compute the set-wise xor of elements in two arrays.

  JAX implementation of :func:`numpy.setxor1d`.

  Because the size of the output of ``setxor1d`` is data-dependent, the function is not
  compatible with JIT or other JAX transformations.

  Args:
    ar1: first array of values to intersect.
    ar2: second array of values to intersect.
    assume_unique: if True, assume the input arrays contain unique values. This allows
      a more efficient implementation, but if ``assume_unique`` is True and the input
      arrays contain duplicates, the behavior is undefined. default: False.
    size: if specified, return only the first ``size`` sorted elements. If there are fewer
      elements than ``size`` indicates, the return value will be padded with ``fill_value``,
      and returned indices will be padded with an out-of-bound index.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the smallest value
      in the xor result.

  Returns:
    An array of values that are found in exactly one of the input arrays.

  See also:
    - :func:`jax.numpy.intersect1d`: the set intersection of two 1D arrays.
    - :func:`jax.numpy.union1d`: the set union of two 1D arrays.
    - :func:`jax.numpy.setdiff1d`: the set difference of two 1D arrays.

  Examples:
    >>> ar1 = jnp.array([1, 2, 3, 4])
    >>> ar2 = jnp.array([3, 4, 5, 6])
    >>> jnp.setxor1d(ar1, ar2)
    Array([1, 2, 5, 6], dtype=int32)
  """
  check_arraylike("setxor1d", ar1, ar2)
  arr1, arr2 = promote_dtypes(ravel(ar1), ravel(ar2))
  del ar1, ar2

  if size is not None:
    return _setxor1d_size(arr1, arr2, fill_value=fill_value,
                          assume_unique=assume_unique, size=size)

  if not assume_unique:
    arr1 = unique(arr1)
    arr2 = unique(arr2)
  aux = concatenate((arr1, arr2))
  if aux.size == 0:
    return aux
  aux = sort(aux)
  flag = concatenate((True, aux[1:] != aux[:-1], True), axis=None)
  return aux[flag[1:] & flag[:-1]]


@partial(jit, static_argnames=['return_indices'])
def _intersect1d_sorted_mask(arr1: Array, arr2: Array,
                             return_indices: bool) -> tuple[Array, Array, Array | None]:
  """JIT-compatible helper function for intersect1d"""
  assert arr1.ndim == arr2.ndim == 1
  arr = concatenate((arr1, arr2))
  if return_indices:
    iota = lax.broadcasted_iota(np.int64, np.shape(arr), dimension=0)
    aux, indices = lax.sort_key_val(arr, iota)
  else:
    aux = sort(arr)
    indices = None
  mask = aux[1:] == aux[:-1]
  return aux, mask, indices


@partial(jit, static_argnames=['fill_value', 'assume_unique', 'size', 'return_indices'])
def _intersect1d_size(arr1: Array, arr2: Array, fill_value: ArrayLike | None, assume_unique: bool,
                      size: int, return_indices: bool) -> Array | tuple[Array, Array, Array]:
  """Jit-compatible helper function for intersect1d with size specified."""
  # Ensured by caller
  assert arr1.ndim == arr2.ndim == 1
  assert arr1.dtype == arr2.dtype

  # First step: we concatenate the unique values of arr1 and arr2.
  # The resulting values are:
  #   num_unique1/num_unique2: number of unique values in arr1/arr2
  #   aux[:num_unique1 + num_unique2] contains the sorted concatenated
  #     unique values drawn from arr1 and arr2.
  #   aux_sorted_indices: indices mapping aux to concatenation of arr1 and arr2
  #   ind1[:num_unique1], ind2[:num_unique2]: indices of sorted unique
  #     values in arr1/arr2
  #   mask: boolean mask of relevant values in aux & aux_sorted_indices
  if assume_unique:
    ind1, num_unique1 = arange(arr1.size), asarray(arr1.size)
    ind2, num_unique2 = arange(arr2.size), asarray(arr2.size)
    arr = concatenate([arr1, arr2])
    aux, aux_sort_indices = lax.sort([arr, arange(arr.size)], is_stable=True, num_keys=1)
    mask = ones(arr.size, dtype=bool)
  else:
    arr1, ind1, num_unique1 = _unique(arr1, 0, size=arr1.size, return_index=True, return_true_size=True, fill_value=0)
    arr2, ind2, num_unique2 = _unique(arr2, 0, size=arr2.size, return_index=True, return_true_size=True, fill_value=0)
    arr = zeros(arr1.size + arr2.size, dtype=dtypes.result_type(arr1, arr2))
    arr = lax.dynamic_update_slice(arr, arr1, (0,))
    arr = lax.dynamic_update_slice(arr, arr2, (num_unique1,))
    mask = arange(arr.size) < num_unique1 + num_unique2
    _, aux, aux_sort_indices = lax.sort([~mask, arr, arange(arr.size)], is_stable=True, num_keys=2)

  # Second step: extract the intersection values from aux
  # Since we've sorted the unique entries in arr1 and arr2, any place where
  # adjacent entries are equal is a value of the intersection.
  # relevant results here:
  #   num_results: number of values in the intersection of arr1 and arr2
  #   vals: array where vals[:num_results] contains the intersection of arr1 and arr2,
  #         and vals[num_results:] contains the appropriate fill_value.
  aux_mask = (aux[1:] == aux[:-1]) & mask[1:]
  num_results = aux_mask.sum()
  if aux.size:
    val_indices = nonzero(aux_mask, size=size, fill_value=aux.size)[0]
    vals = aux.at[val_indices].get(mode='fill', fill_value=0)
  else:
    vals = zeros(size, aux.dtype)
  if fill_value is None:
    vals = where(arange(len(vals)) < num_results, vals, vals.max())
    vals = where(arange(len(vals)) < num_results, vals, vals.min())
  else:
    vals = where(arange(len(vals)) < num_results, vals, fill_value)

  # Third step: extract the indices of the intersection values.
  # This requires essentially unwinding aux_sort_indices and ind1/ind2 to find
  # the appropriate list of indices from the original arrays.
  if return_indices:
    arr1_indices = aux_sort_indices.at[val_indices].get(mode='fill', fill_value=arr1.size)
    arr1_indices = where(arange(len(arr1_indices)) < num_results, arr1_indices, arr1.size)
    arr2_indices = aux_sort_indices.at[val_indices + 1].get(mode='fill', fill_value=arr2.size) - num_unique1
    arr2_indices = where(arange(len(arr2_indices)) < num_results, arr2_indices, arr2.size)
    if not assume_unique:
      arr1_indices = ind1.at[arr1_indices].get(mode='fill', fill_value=ind1.size)
      arr2_indices = ind2.at[arr2_indices].get(mode='fill', fill_value=ind2.size)
    return vals, arr1_indices, arr2_indices
  else:
    return vals


@export
def intersect1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False,
                return_indices: bool = False, *, size: int | None = None,
                fill_value: ArrayLike | None = None) -> Array | tuple[Array, Array, Array]:
  """Compute the set intersection of two 1D arrays.

  JAX implementation of :func:`numpy.intersect1d`.

  Because the size of the output of ``intersect1d`` is data-dependent, the function
  is not typically compatible with :func:`~jax.jit` and other JAX transformations.
  The JAX version adds the optional ``size`` argument which must be specified
  statically for ``jnp.intersect1d`` to be used in such contexts.

  Args:
    ar1: first array of values to intersect.
    ar2: second array of values to intersect.
    assume_unique: if True, assume the input arrays contain unique values. This allows
      a more efficient implementation, but if ``assume_unique`` is True and the input
      arrays contain duplicates, the behavior is undefined. default: False.
    return_indices: If True, return arrays of indices specifying where the intersected
      values first appear in the input arrays.
    size: if specified, return only the first ``size`` sorted elements. If there are fewer
      elements than ``size`` indicates, the return value will be padded with ``fill_value``,
      and returned indices will be padded with an out-of-bound index.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the smallest value
      in the intersection.

  Returns:
    An array ``intersection``, or if ``return_indices=True``, a tuple of arrays
    ``(intersection, ar1_indices, ar2_indices)``. Returned values are

    - ``intersection``:
      A 1D array containing each value that appears in both ``ar1`` and ``ar2``.
    - ``ar1_indices``:
      *(returned if return_indices=True)* an array of shape ``intersection.shape`` containing
      the indices in flattened ``ar1`` of values in ``intersection``. For 1D inputs,
      ``intersection`` is equivalent to ``ar1[ar1_indices]``.
    - ``ar2_indices``:
      *(returned if return_indices=True)* an array of shape ``intersection.shape`` containing
      the indices in flattened ``ar2`` of values in ``intersection``. For 1D inputs,
      ``intersection`` is equivalent to ``ar2[ar2_indices]``.

  See also:
    - :func:`jax.numpy.union1d`: the set union of two 1D arrays.
    - :func:`jax.numpy.setxor1d`: the set XOR of two 1D arrays.
    - :func:`jax.numpy.setdiff1d`: the set difference of two 1D arrays.

  Examples:
    >>> ar1 = jnp.array([1, 2, 3, 4])
    >>> ar2 = jnp.array([3, 4, 5, 6])
    >>> jnp.intersect1d(ar1, ar2)
    Array([3, 4], dtype=int32)

    Computing intersection with indices:

    >>> intersection, ar1_indices, ar2_indices = jnp.intersect1d(ar1, ar2, return_indices=True)
    >>> intersection
    Array([3, 4], dtype=int32)

    ``ar1_indices`` gives the indices of the intersected values within ``ar1``:

     >>> ar1_indices
     Array([2, 3], dtype=int32)
     >>> jnp.all(intersection == ar1[ar1_indices])
     Array(True, dtype=bool)

    ``ar2_indices`` gives the indices of the intersected values within ``ar2``:

     >>> ar2_indices
     Array([0, 1], dtype=int32)
     >>> jnp.all(intersection == ar2[ar2_indices])
     Array(True, dtype=bool)
  """
  check_arraylike("intersect1d", ar1, ar2)
  arr1, arr2 = promote_dtypes(ar1, ar2)
  del ar1, ar2
  arr1 = ravel(arr1)
  arr2 = ravel(arr2)

  if size is not None:
    return _intersect1d_size(arr1, arr2, return_indices=return_indices,
                             size=size, fill_value=fill_value, assume_unique=assume_unique)

  if not assume_unique:
    if return_indices:
      arr1, ind1 = unique(arr1, return_index=True)
      arr2, ind2 = unique(arr2, return_index=True)
    else:
      arr1 = unique(arr1)
      arr2 = unique(arr2)

  aux, mask, aux_sort_indices = _intersect1d_sorted_mask(arr1, arr2, return_indices)

  int1d = aux[:-1][mask]

  if return_indices:
    assert aux_sort_indices is not None
    arr1_indices = aux_sort_indices[:-1][mask]
    arr2_indices = aux_sort_indices[1:][mask] - np.size(arr1)
    if not assume_unique:
      arr1_indices = ind1[arr1_indices]
      arr2_indices = ind2[arr2_indices]
    return int1d, arr1_indices, arr2_indices
  else:
    return int1d


@export
def isin(element: ArrayLike, test_elements: ArrayLike,
         assume_unique: bool = False, invert: bool = False, *,
         method='auto') -> Array:
  """Determine whether elements in ``element`` appear in ``test_elements``.

  JAX implementation of :func:`numpy.isin`.

  Args:
    element: input array of elements for which membership will be checked.
    test_elements: N-dimensional array of test values to check for the presence of
      each element.
    invert: If True, return ``~isin(element, test_elements)``. Default is False.
    assume_unique: if true, input arrays are assumed to be unique, which can
      lead to more efficient computation. If the input arrays are not unique
      and assume_unique is set to True, the results are undefined.
    method: string specifying the method used to compute the result. Supported
      options are 'compare_all', 'binary_search', 'sort', and 'auto' (default).

  Returns:
    A boolean array of shape ``element.shape`` that specifies whether each element
    appears in ``test_elements``.

  Examples:
    >>> elements = jnp.array([1, 2, 3, 4])
    >>> test_elements = jnp.array([[1, 5, 6, 3, 7, 1]])
    >>> jnp.isin(elements, test_elements)
    Array([ True, False,  True, False], dtype=bool)
  """
  check_arraylike("isin", element, test_elements)
  result = _in1d(element, test_elements, invert=invert,
                 method=method, assume_unique=assume_unique)
  return result.reshape(np.shape(element))


### SetOps

UNIQUE_SIZE_HINT = (
  "To make jnp.unique() compatible with JIT and other transforms, you can specify "
  "a concrete value for the size argument, which will determine the output size.")

@partial(jit, static_argnames=['axis', 'equal_nan'])
def _unique_sorted_mask(ar: Array, axis: int, equal_nan: bool) -> tuple[Array, Array, Array]:
  aux = moveaxis(ar, axis, 0)
  if np.issubdtype(aux.dtype, np.complexfloating):
    # Work around issue in sorting of complex numbers with Nan only in the
    # imaginary component. This can be removed if sorting in this situation
    # is fixed to match numpy.
    aux = where(isnan(aux), _lax_const(aux, np.nan), aux)
  size, *out_shape = aux.shape
  if math.prod(out_shape) == 0:
    size = 1
    perm = zeros(1, dtype=int)
  else:
    perm = lexsort(aux.reshape(size, math.prod(out_shape)).T[::-1])
  aux = aux[perm]
  if aux.size:
    if dtypes.issubdtype(aux.dtype, np.inexact) and equal_nan:
      # This is appropriate for both float and complex due to the documented behavior of np.unique:
      # See https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/arraysetops.py#L212-L220
      neq = lambda x, y: lax.ne(x, y) & ~(isnan(x) & isnan(y))
    else:
      neq = lax.ne
    mask = ones(size, dtype=bool).at[1:].set(any(neq(aux[1:], aux[:-1]), tuple(range(1, aux.ndim))))
  else:
    mask = zeros(size, dtype=bool)
  return aux, mask, perm

def _unique(ar: Array, axis: int, return_index: bool = False, return_inverse: bool = False,
            return_counts: bool = False, equal_nan: bool = True, size: int | None = None,
            fill_value: ArrayLike | None = None, return_true_size: bool = False
            ) -> Array | tuple[Array, ...]:
  """
  Find the unique elements of an array along a particular axis.
  """
  axis = canonicalize_axis(axis, ar.ndim)

  if ar.shape[axis] == 0 and size and fill_value is None:
    raise ValueError(
      "jnp.unique: for zero-sized input with nonzero size argument, fill_value must be specified")

  aux, mask, perm = _unique_sorted_mask(ar, axis, equal_nan)
  if size is None:
    ind = core.concrete_or_error(None, mask,
        "The error arose in jnp.unique(). " + UNIQUE_SIZE_HINT)
  else:
    ind = nonzero(mask, size=size)[0]
  result = aux[ind] if aux.size else aux
  if size is not None and fill_value is not None:
    fill_value = asarray(fill_value, dtype=result.dtype)
    if result.shape[0]:
      valid = lax.expand_dims(arange(size) < mask.sum(), tuple(range(1, result.ndim)))
      result = where(valid, result, fill_value)
    else:
      result = full_like(result, fill_value, shape=(size, *result.shape[1:]))
  result = moveaxis(result, 0, axis)

  ret: tuple[Array, ...] = (result,)
  if return_index:
    if aux.size:
      ret += (perm[ind],)
    else:
      ret += (perm,)
  if return_inverse:
    if aux.size:
      imask = cumsum(mask) - 1
      inv_idx = zeros(mask.shape, dtype=dtypes.canonicalize_dtype(dtypes.int_))
      inv_idx = inv_idx.at[perm].set(imask)
    else:
      inv_idx = zeros(ar.shape[axis], dtype=int)
    ret += (inv_idx,)
  if return_counts:
    if aux.size:
      if size is None:
        idx = append(nonzero(mask)[0], mask.size)
      else:
        idx = nonzero(mask, size=size + 1)[0]
        idx = idx.at[1:].set(where(idx[1:], idx[1:], mask.size))
      ret += (diff(idx),)
    elif ar.shape[axis]:
      ret += (array([ar.shape[axis]], dtype=dtypes.canonicalize_dtype(dtypes.int_)),)
    else:
      ret += (empty(0, dtype=int),)
  if return_true_size:
    # Useful for internal uses of unique().
    ret += (mask.sum(),)
  return ret[0] if len(ret) == 1 else ret


@export
def unique(ar: ArrayLike, return_index: bool = False, return_inverse: bool = False,
           return_counts: bool = False, axis: int | None = None,
           *, equal_nan: bool = True, size: int | None = None, fill_value: ArrayLike | None = None):
  """Return the unique values from an array.

  JAX implementation of :func:`numpy.unique`.

  Because the size of the output of ``unique`` is data-dependent, the function
  is not typically compatible with :func:`~jax.jit` and other JAX transformations.
  The JAX version adds the optional ``size`` argument which must be specified
  statically for ``jnp.unique`` to be used in such contexts.

  Args:
    ar: N-dimensional array from which unique values will be extracted.
    return_index: if True, also return the indices in ``ar`` where each value occurs
    return_inverse: if True, also return the indices that can be used to reconstruct
      ``ar`` from the unique values.
    return_counts: if True, also return the number of occurrences of each unique value.
    axis: if specified, compute unique values along the specified axis. If None (default),
      then flatten ``ar`` before computing the unique values.
    equal_nan: if True, consider NaN values equivalent when determining uniqueness.
    size: if specified, return only the first ``size`` sorted unique elements. If there are fewer
      unique elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the minimum unique value.

  Returns:
    An array or tuple of arrays, depending on the values of ``return_index``, ``return_inverse``,
    and ``return_counts``. Returned values are

    - ``unique_values``:
        if ``axis`` is None, a 1D array of length ``n_unique``, If ``axis`` is
        specified, shape is ``(*ar.shape[:axis], n_unique, *ar.shape[axis + 1:])``.
    - ``unique_index``:
        *(returned only if return_index is True)* An array of shape ``(n_unique,)``. Contains
        the indices of the first occurrence of each unique value in ``ar``. For 1D inputs,
        ``ar[unique_index]`` is equivalent to ``unique_values``.
    - ``unique_inverse``:
        *(returned only if return_inverse is True)* An array of shape ``(ar.size,)`` if ``axis``
        is None, or of shape ``(ar.shape[axis],)`` if ``axis`` is specified.
        Contains the indices within ``unique_values`` of each value in ``ar``. For 1D inputs,
        ``unique_values[unique_inverse]`` is equivalent to ``ar``.
    - ``unique_counts``:
        *(returned only if return_counts is True)* An array of shape ``(n_unique,)``.
        Contains the number of occurrences of each unique value in ``ar``.

  See also:
    - :func:`jax.numpy.unique_counts`: shortcut to ``unique(arr, return_counts=True)``.
    - :func:`jax.numpy.unique_inverse`: shortcut to ``unique(arr, return_inverse=True)``.
    - :func:`jax.numpy.unique_all`: shortcut to ``unique`` with all return values.
    - :func:`jax.numpy.unique_values`: like ``unique``, but no optional return values.

  Examples:
    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> jnp.unique(x)
    Array([1, 3, 4], dtype=int32)

    **JIT compilation & the size argument**

    If you try this under :func:`~jax.jit` or another transformation, you will get an
    error because the output shape is dynamic:

    >>> jax.jit(jnp.unique)(x)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
       ...
    jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[5].
    The error arose for the first argument of jnp.unique(). To make jnp.unique() compatible with JIT and other transforms, you can specify a concrete value for the size argument, which will determine the output size.

    The issue is that the output of transformed functions must have static shapes.
    In order to make this work, you can pass a static ``size`` parameter:

    >>> jit_unique = jax.jit(jnp.unique, static_argnames=['size'])
    >>> jit_unique(x, size=3)
    Array([1, 3, 4], dtype=int32)

    If your static size is smaller than the true number of unique values, they will be truncated.

    >>> jit_unique(x, size=2)
    Array([1, 3], dtype=int32)

    If the static size is larger than the true number of unique values, they will be padded with
    ``fill_value``, which defaults to the minimum unique value:

    >>> jit_unique(x, size=5)
    Array([1, 3, 4, 1, 1], dtype=int32)
    >>> jit_unique(x, size=5, fill_value=0)
    Array([1, 3, 4, 0, 0], dtype=int32)

    **Multi-dimensional unique values**

    If you pass a multi-dimensional array to ``unique``, it will be flattened by default:

    >>> M = jnp.array([[1, 2],
    ...                [2, 3],
    ...                [1, 2]])
    >>> jnp.unique(M)
    Array([1, 2, 3], dtype=int32)

    If you pass an ``axis`` keyword, you can find unique *slices* of the array along
    that axis:

    >>> jnp.unique(M, axis=0)
    Array([[1, 2],
           [2, 3]], dtype=int32)

    **Returning indices**

    If you set ``return_index=True``, then ``unique`` returns the indices of the
    first occurrence of each unique value:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> values, indices = jnp.unique(x, return_index=True)
    >>> print(values)
    [1 3 4]
    >>> print(indices)
    [2 0 1]
    >>> jnp.all(values == x[indices])
    Array(True, dtype=bool)

    In multiple dimensions, the unique values can be extracted with :func:`jax.numpy.take`
    evaluated along the specified axis:

    >>> values, indices = jnp.unique(M, axis=0, return_index=True)
    >>> jnp.all(values == jnp.take(M, indices, axis=0))
    Array(True, dtype=bool)

    **Returning inverse**

    If you set ``return_inverse=True``, then ``unique`` returns the indices within the
    unique values for every entry in the input array:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> values, inverse = jnp.unique(x, return_inverse=True)
    >>> print(values)
    [1 3 4]
    >>> print(inverse)
    [1 2 0 1 0]
    >>> jnp.all(values[inverse] == x)
    Array(True, dtype=bool)

    In multiple dimensions, the input can be reconstructed using
    :func:`jax.numpy.take`:

    >>> values, inverse = jnp.unique(M, axis=0, return_inverse=True)
    >>> jnp.all(jnp.take(values, inverse, axis=0) == M)
    Array(True, dtype=bool)

    **Returning counts**

    If you set ``return_counts=True``, then ``unique`` returns the number of occurrences
    within the input for every unique value:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> values, counts = jnp.unique(x, return_counts=True)
    >>> print(values)
    [1 3 4]
    >>> print(counts)
    [2 2 1]

    For multi-dimensional arrays, this also returns a 1D array of counts
    indicating number of occurrences along the specified axis:

    >>> values, counts = jnp.unique(M, axis=0, return_counts=True)
    >>> print(values)
    [[1 2]
     [2 3]]
    >>> print(counts)
    [2 1]
  """
  check_arraylike("unique", ar)
  if size is None:
    ar = core.concrete_or_error(None, ar,
        "The error arose for the first argument of jnp.unique(). " + UNIQUE_SIZE_HINT)
  else:
    size = core.concrete_or_error(operator.index, size,
         "The error arose for the size argument of jnp.unique(). " + UNIQUE_SIZE_HINT)
  arr = asarray(ar)
  arr_shape = arr.shape
  if axis is None:
    axis_int: int = 0
    arr = arr.flatten()
  else:
    axis_int = canonicalize_axis(axis, arr.ndim)
  result = _unique(arr, axis_int, return_index, return_inverse, return_counts,
                   equal_nan=equal_nan, size=size, fill_value=fill_value)
  if return_inverse and axis is None:
    idx = 2 if return_index else 1
    result = (*result[:idx], result[idx].reshape(arr_shape), *result[idx + 1:])
  return result


class _UniqueAllResult(NamedTuple):
  """Struct returned by :func:`jax.numpy.unique_all`."""
  values: Array
  indices: Array
  inverse_indices: Array
  counts: Array


class _UniqueCountsResult(NamedTuple):
    """Struct returned by :func:`jax.numpy.unique_counts`."""
    values: Array
    counts: Array


class _UniqueInverseResult(NamedTuple):
    """Struct returned by :func:`jax.numpy.unique_inverse`."""
    values: Array
    inverse_indices: Array


@export
def unique_all(x: ArrayLike, /, *, size: int | None = None,
               fill_value: ArrayLike | None = None) -> _UniqueAllResult:
  """Return unique values from x, along with indices, inverse indices, and counts.

  JAX implementation of :func:`numpy.unique_all`; this is equivalent to calling
  :func:`jax.numpy.unique` with `return_index`, `return_inverse`, `return_counts`,
  and `equal_nan` set to True.

  Because the size of the output of ``unique_all`` is data-dependent, the function
  is not typically compatible with :func:`~jax.jit` and other JAX transformations.
  The JAX version adds the optional ``size`` argument which must be specified
  statically for ``jnp.unique`` to be used in such contexts.

  Args:
    x: N-dimensional array from which unique values will be extracted.
    size: if specified, return only the first ``size`` sorted unique elements. If there are fewer
      unique elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the minimum unique value.

  Returns:
    A tuple ``(values, indices, inverse_indices, counts)``, with the following properties:

    - ``values``:
        an array of shape ``(n_unique,)`` containing the unique values from ``x``.
    - ``indices``:
        An array of shape ``(n_unique,)``. Contains the indices of the first occurrence of
        each unique value in ``x``. For 1D inputs, ``x[indices]`` is equivalent to ``values``.
    - ``inverse_indices``:
        An array of shape ``x.shape``. Contains the indices within ``values`` of each value
        in ``x``. For 1D inputs, ``values[inverse_indices]`` is equivalent to ``x``.
    - ``counts``:
        An array of shape ``(n_unique,)``. Contains the number of occurrences of each unique
        value in ``x``.

  See also:
    - :func:`jax.numpy.unique`: general function for computing unique values.
    - :func:`jax.numpy.unique_values`: compute only ``values``.
    - :func:`jax.numpy.unique_counts`: compute only ``values`` and ``counts``.
    - :func:`jax.numpy.unique_inverse`: compute only ``values`` and ``inverse``.

  Examples:
    Here we compute the unique values in a 1D array:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> result = jnp.unique_all(x)

    The result is a :class:`~typing.NamedTuple` with four named attributes.
    The ``values`` attribute contains the unique values from the array:

    >>> result.values
    Array([1, 3, 4], dtype=int32)

    The ``indices`` attribute contains the indices of the unique ``values`` within
    the input array:

    >>> result.indices
    Array([2, 0, 1], dtype=int32)
    >>> jnp.all(result.values == x[result.indices])
    Array(True, dtype=bool)

    The ``inverse_indices`` attribute contains the indices of the input within ``values``:

    >>> result.inverse_indices
    Array([1, 2, 0, 1, 0], dtype=int32)
    >>> jnp.all(x == result.values[result.inverse_indices])
    Array(True, dtype=bool)

    The ``counts`` attribute contains the counts of each unique value in the input:

    >>> result.counts
    Array([2, 2, 1], dtype=int32)

    For examples of the ``size`` and ``fill_value`` arguments, see :func:`jax.numpy.unique`.
  """
  check_arraylike("unique_all", x)
  values, indices, inverse_indices, counts = unique(
    x, return_index=True, return_inverse=True, return_counts=True, equal_nan=False,
    size=size, fill_value=fill_value)
  return _UniqueAllResult(values=values, indices=indices, inverse_indices=inverse_indices, counts=counts)


@export
def unique_counts(x: ArrayLike, /, *, size: int | None = None,
                  fill_value: ArrayLike | None = None) -> _UniqueCountsResult:
  """Return unique values from x, along with counts.

  JAX implementation of :func:`numpy.unique_counts`; this is equivalent to calling
  :func:`jax.numpy.unique` with `return_counts` and `equal_nan` set to True.

  Because the size of the output of ``unique_counts`` is data-dependent, the function
  is not typically compatible with :func:`~jax.jit` and other JAX transformations.
  The JAX version adds the optional ``size`` argument which must be specified
  statically for ``jnp.unique`` to be used in such contexts.

  Args:
    x: N-dimensional array from which unique values will be extracted.
    size: if specified, return only the first ``size`` sorted unique elements. If there are fewer
      unique elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the minimum unique value.

  Returns:
    A tuple ``(values, counts)``, with the following properties:

    - ``values``:
        an array of shape ``(n_unique,)`` containing the unique values from ``x``.
    - ``counts``:
        An array of shape ``(n_unique,)``. Contains the number of occurrences of each unique
        value in ``x``.

  See also:
    - :func:`jax.numpy.unique`: general function for computing unique values.
    - :func:`jax.numpy.unique_values`: compute only ``values``.
    - :func:`jax.numpy.unique_inverse`: compute only ``values`` and ``inverse``.
    - :func:`jax.numpy.unique_all`: compute ``values``, ``indices``, ``inverse_indices``,
      and ``counts``.

  Examples:
    Here we compute the unique values in a 1D array:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> result = jnp.unique_counts(x)

    The result is a :class:`~typing.NamedTuple` with two named attributes.
    The ``values`` attribute contains the unique values from the array:

    >>> result.values
    Array([1, 3, 4], dtype=int32)

    The ``counts`` attribute contains the counts of each unique value in the input:

    >>> result.counts
    Array([2, 2, 1], dtype=int32)

    For examples of the ``size`` and ``fill_value`` arguments, see :func:`jax.numpy.unique`.
  """
  check_arraylike("unique_counts", x)
  values, counts = unique(x, return_counts=True, equal_nan=False,
                          size=size, fill_value=fill_value)
  return _UniqueCountsResult(values=values, counts=counts)


@export
def unique_inverse(x: ArrayLike, /, *, size: int | None = None,
                   fill_value: ArrayLike | None = None) -> _UniqueInverseResult:
  """Return unique values from x, along with indices, inverse indices, and counts.

  JAX implementation of :func:`numpy.unique_inverse`; this is equivalent to calling
  :func:`jax.numpy.unique` with `return_inverse` and `equal_nan` set to True.

  Because the size of the output of ``unique_inverse`` is data-dependent, the function
  is not typically compatible with :func:`~jax.jit` and other JAX transformations.
  The JAX version adds the optional ``size`` argument which must be specified
  statically for ``jnp.unique`` to be used in such contexts.

  Args:
    x: N-dimensional array from which unique values will be extracted.
    size: if specified, return only the first ``size`` sorted unique elements. If there are fewer
      unique elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the minimum unique value.

  Returns:
    A tuple ``(values, indices, inverse_indices, counts)``, with the following properties:

    - ``values``:
        an array of shape ``(n_unique,)`` containing the unique values from ``x``.
    - ``inverse_indices``:
        An array of shape ``x.shape``. Contains the indices within ``values`` of each value
        in ``x``. For 1D inputs, ``values[inverse_indices]`` is equivalent to ``x``.

  See also:
    - :func:`jax.numpy.unique`: general function for computing unique values.
    - :func:`jax.numpy.unique_values`: compute only ``values``.
    - :func:`jax.numpy.unique_counts`: compute only ``values`` and ``counts``.
    - :func:`jax.numpy.unique_all`: compute ``values``, ``indices``, ``inverse_indices``,
      and ``counts``.

  Examples:
    Here we compute the unique values in a 1D array:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> result = jnp.unique_inverse(x)

    The result is a :class:`~typing.NamedTuple` with two named attributes.
    The ``values`` attribute contains the unique values from the array:

    >>> result.values
    Array([1, 3, 4], dtype=int32)

    The ``indices`` attribute contains the indices of the unique ``values`` within
    the input array:

    The ``inverse_indices`` attribute contains the indices of the input within ``values``:

    >>> result.inverse_indices
    Array([1, 2, 0, 1, 0], dtype=int32)
    >>> jnp.all(x == result.values[result.inverse_indices])
    Array(True, dtype=bool)

    For examples of the ``size`` and ``fill_value`` arguments, see :func:`jax.numpy.unique`.
  """
  check_arraylike("unique_inverse", x)
  values, inverse_indices = unique(x, return_inverse=True, equal_nan=False,
                                   size=size, fill_value=fill_value)
  return _UniqueInverseResult(values=values, inverse_indices=inverse_indices)


@export
def unique_values(x: ArrayLike, /, *, size: int | None = None,
                  fill_value: ArrayLike | None = None) -> Array:
  """Return unique values from x, along with indices, inverse indices, and counts.

  JAX implementation of :func:`numpy.unique_values`; this is equivalent to calling
  :func:`jax.numpy.unique` with `equal_nan` set to True.

  Because the size of the output of ``unique_values`` is data-dependent, the function
  is not typically compatible with :func:`~jax.jit` and other JAX transformations.
  The JAX version adds the optional ``size`` argument which must be specified statically
  for ``jnp.unique`` to be used in such contexts.

  Args:
    x: N-dimensional array from which unique values will be extracted.
    size: if specified, return only the first ``size`` sorted unique elements. If there are fewer
      unique elements than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value: when ``size`` is specified and there are fewer than the indicated number of
      elements, fill the remaining entries ``fill_value``. Defaults to the minimum unique value.

  Returns:
    An array ``values`` of shape ``(n_unique,)`` containing the unique values from ``x``.

  See also:
    - :func:`jax.numpy.unique`: general function for computing unique values.
    - :func:`jax.numpy.unique_values`: compute only ``values``.
    - :func:`jax.numpy.unique_counts`: compute only ``values`` and ``counts``.
    - :func:`jax.numpy.unique_inverse`: compute only ``values`` and ``inverse``.

  Examples:
    Here we compute the unique values in a 1D array:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> jnp.unique_values(x)
    Array([1, 3, 4], dtype=int32)

    For examples of the ``size`` and ``fill_value`` arguments, see :func:`jax.numpy.unique`.
  """
  check_arraylike("unique_values", x)
  return cast(Array, unique(x, equal_nan=False, size=size, fill_value=fill_value))
