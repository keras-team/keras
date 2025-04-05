# Copyright 2020 The JAX Authors.
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

from collections.abc import Sequence
from functools import partial
from typing import Any

import warnings

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src.lax import lax
from jax._src.util import safe_zip, safe_map
from jax._src.typing import Array, ArrayLike, DimSize, DType, DTypeLike, Shape

import numpy as np

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map

_dtype = partial(dtypes.dtype, canonicalize=True)

def promote_shapes(fun_name: str, *args: ArrayLike) -> list[Array]:
  """Apply NumPy-style broadcasting, making args shape-compatible for lax.py."""
  if len(args) < 2:
    return [lax.asarray(arg) for arg in args]
  else:
    shapes = [np.shape(arg) for arg in args]
    if config.dynamic_shapes.value:
      # With dynamic shapes we don't support singleton-dimension broadcasting;
      # we instead broadcast out to the full shape as a temporary workaround.
      # TODO(mattjj): revise this workaround
      res_shape = lax.broadcast_shapes(*shapes)  # Can raise an error!
      return [_broadcast_to(arg, res_shape) for arg, shp in zip(args, shapes)]
    else:
      if all(len(shapes[0]) == len(s) for s in shapes[1:]):
        return [lax.asarray(arg) for arg in args]  # no need for rank promotion, so rely on lax promotion
      nonscalar_ranks = {len(shp) for shp in shapes if shp}
      if len(nonscalar_ranks) < 2:
        return [lax.asarray(arg) for arg in args]  # rely on lax scalar promotion
      else:
        if config.numpy_rank_promotion.value != "allow":
          _rank_promotion_warning_or_error(fun_name, shapes)
        result_rank = len(lax.broadcast_shapes(*shapes))
        return [lax.broadcast_to_rank(arg, result_rank) for arg in args]


def _rank_promotion_warning_or_error(fun_name: str, shapes: Sequence[Shape]):
  if config.numpy_rank_promotion.value == "warn":
    msg = ("Following NumPy automatic rank promotion for {} on shapes {}. "
           "Set the jax_numpy_rank_promotion config option to 'allow' to "
           "disable this warning; for more information, see "
           "https://jax.readthedocs.io/en/latest/rank_promotion_warning.html.")
    warnings.warn(msg.format(fun_name, ' '.join(map(str, shapes))))
  elif config.numpy_rank_promotion.value == "raise":
    msg = ("Operands could not be broadcast together for {} on shapes {} "
           "and with the config option jax_numpy_rank_promotion='raise'. "
           "For more information, see "
           "https://jax.readthedocs.io/en/latest/rank_promotion_warning.html.")
    raise ValueError(msg.format(fun_name, ' '.join(map(str, shapes))))


def promote_dtypes(*args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument dtype promotion."""
  # TODO(dougalm,mattjj): This is a performance bottleneck. Consider memoizing.
  if len(args) < 2:
    return [lax.asarray(arg) for arg in args]
  else:
    to_dtype, weak_type = dtypes._lattice_result_type(*args)
    to_dtype = dtypes.canonicalize_dtype(to_dtype, allow_extended_dtype=True)  # type: ignore[assignment]
    if config.sharding_in_types.value:
      return [lax._convert_element_type(x, to_dtype, weak_type,
                                        getattr(x, "sharding", None))
              for x in args]
    else:
      return [lax._convert_element_type(x, to_dtype, weak_type) for x in args]


def promote_dtypes_inexact(*args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument dtype promotion.

  Promotes arguments to an inexact type."""
  to_dtype, weak_type = dtypes._lattice_result_type(*args)
  to_dtype = dtypes.canonicalize_dtype(to_dtype, allow_extended_dtype=True)  # type: ignore[assignment]
  to_dtype_inexact = dtypes.to_inexact_dtype(to_dtype)
  return [lax._convert_element_type(x, to_dtype_inexact, weak_type)
          for x in args]


def promote_dtypes_numeric(*args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument dtype promotion.

  Promotes arguments to a numeric (non-bool) type."""
  to_dtype, weak_type = dtypes._lattice_result_type(*args)
  to_dtype = dtypes.canonicalize_dtype(to_dtype)
  to_dtype_numeric = dtypes.to_numeric_dtype(to_dtype)
  return [lax._convert_element_type(x, to_dtype_numeric, weak_type)
          for x in args]


def promote_dtypes_complex(*args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument dtype promotion.

  Promotes arguments to a complex type."""
  to_dtype, weak_type = dtypes._lattice_result_type(*args)
  to_dtype = dtypes.canonicalize_dtype(to_dtype)
  to_dtype_complex = dtypes.to_complex_dtype(to_dtype)
  return [lax._convert_element_type(x, to_dtype_complex, weak_type)
          for x in args]


def _complex_elem_type(dtype: DTypeLike) -> DType:
  """Returns the float type of the real/imaginary parts of a complex dtype."""
  return np.abs(np.zeros((), dtype)).dtype


def _arraylike(x: ArrayLike) -> bool:
  return (isinstance(x, np.ndarray) or isinstance(x, Array) or
          hasattr(x, '__jax_array__') or np.isscalar(x))


def check_arraylike(fun_name: str, *args: Any, emit_warning=False, stacklevel=3):
  """Check if all args fit JAX's definition of arraylike."""
  assert isinstance(fun_name, str), f"fun_name must be a string. Got {fun_name}"
  if any(not _arraylike(arg) for arg in args):
    pos, arg = next((i, arg) for i, arg in enumerate(args)
                    if not _arraylike(arg))
    msg = f"{fun_name} requires ndarray or scalar arguments, got {type(arg)} at position {pos}."
    if emit_warning:
      warnings.warn(msg + " In a future JAX release this will be an error.",
                    category=DeprecationWarning, stacklevel=stacklevel)
    else:
      raise TypeError(msg.format(fun_name, type(arg), pos))


def check_arraylike_or_none(fun_name: str, *args: Any):
  assert isinstance(fun_name, str), f"fun_name must be a string. Got {fun_name}"
  if any(not (_arraylike(arg) or arg is None) for arg in args):
    pos, arg = next((i, arg) for i, arg in enumerate(args)
                    if not (_arraylike(arg) or arg is None))
    msg = "{} requires ndarray, scalar, or None arguments, got {} at position {}."
    raise TypeError(msg.format(fun_name, type(arg), pos))


def check_no_float0s(fun_name: str, *args: Any):
  """Check if none of the args have dtype float0."""
  if any(dtypes.dtype(arg) == dtypes.float0 for arg in args):
    raise TypeError(
        f"Called {fun_name} with a float0 array. "
        "float0s do not support any operations by design because they "
        "are not compatible with non-trivial vector spaces. No implicit dtype "
        "conversion is done. You can use np.zeros_like(arr, dtype=np.float) "
        "to cast a float0 array to a regular zeros array. \n"
        "If you didn't expect to get a float0 you might have accidentally "
        "taken a gradient with respect to an integer argument.")
_check_no_float0s = check_no_float0s


def check_for_prngkeys(fun_name: str, *args: Any):
  """Check if args don't match and none of the args have typed prng dtype"""
  arg_dtypes = [dtypes.dtype(arg) for arg in args]
  if len(set(arg_dtypes)) < 2:
    return  # Will be caught by extended dtype impl rules.
  if any(dtypes.issubdtype(dt, dtypes.prng_key) for dt in arg_dtypes):
    if len(arg_dtypes) == 1:
      raise TypeError(
        f"{fun_name} does not accept dtype {str(arg_dtypes[0])}.")
    else:
      raise TypeError(
        f"{fun_name} does not accept dtypes {', '.join(map(str, arg_dtypes))}."
      )


def promote_args(fun_name: str, *args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument shape and dtype promotion."""
  check_arraylike(fun_name, *args)
  _check_no_float0s(fun_name, *args)
  check_for_prngkeys(fun_name, *args)
  return promote_shapes(fun_name, *promote_dtypes(*args))


def promote_args_numeric(fun_name: str, *args: ArrayLike) -> list[Array]:
  check_arraylike(fun_name, *args)
  _check_no_float0s(fun_name, *args)
  check_for_prngkeys(fun_name, *args)
  return promote_shapes(fun_name, *promote_dtypes_numeric(*args))


def promote_args_inexact(fun_name: str, *args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument shape and dtype promotion.

  Promotes non-inexact types to an inexact type."""
  check_arraylike(fun_name, *args)
  _check_no_float0s(fun_name, *args)
  check_for_prngkeys(fun_name, *args)
  return promote_shapes(fun_name, *promote_dtypes_inexact(*args))


@partial(api.jit, inline=True)
def _broadcast_arrays(*args: ArrayLike) -> list[Array]:
  """Like Numpy's broadcast_arrays but doesn't return views."""
  avals = [core.shaped_abstractify(arg) for arg in args]
  shapes = [a.shape for a in avals]
  if not shapes or all(core.definitely_equal_shape(shapes[0], s) for s in shapes):
    return [lax.asarray(arg) for arg in args]
  result_shape = lax.broadcast_shapes(*shapes)
  result_sharding = (lax.broadcast_shardings(*avals)  # type: ignore
                     if config.sharding_in_types.value else None)
  return [_broadcast_to(arg, result_shape, result_sharding) for arg in args]


def _broadcast_to(arr: ArrayLike, shape: DimSize | Shape, sharding=None
                  ) -> Array:
  check_arraylike("broadcast_to", arr)
  arr = arr if isinstance(arr, Array) else lax.asarray(arr)
  if not isinstance(shape, tuple) and np.ndim(shape) == 0:
    shape = (shape,)
  # check that shape is concrete
  shape = core.canonicalize_shape(shape)  # type: ignore[arg-type]
  arr_shape = np.shape(arr)
  if core.definitely_equal_shape(arr_shape, shape):
    return arr
  elif len(shape) < len(arr_shape):
    raise ValueError(f"Cannot broadcast to shape with fewer dimensions: {arr_shape=} {shape=}")
  else:
    nlead = len(shape) - len(arr_shape)
    shape_tail = shape[nlead:]
    compatible = all(core.definitely_equal_one_of_dim(arr_d, [1, shape_d])
                     for arr_d, shape_d in safe_zip(arr_shape, shape_tail))
    if nlead < 0 or not compatible:
      msg = "Incompatible shapes for broadcasting: {} and requested shape {}"
      raise ValueError(msg.format(arr_shape, shape))
    return lax.broadcast_in_dim(arr, shape, tuple(range(nlead, len(shape))),
                                sharding=sharding)


# The `jit` on `where` exists to avoid materializing constants in cases like
# `np.where(np.zeros(1000), 7, 4)`. In op-by-op mode, we don't want to
# materialize the broadcast forms of scalar arguments.
@api.jit
def _where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> Array:
  if x is None or y is None:
    raise ValueError("Either both or neither of the x and y arguments should "
                     "be provided to jax.numpy.where, got {} and {}."
                     .format(x, y))
  if not np.issubdtype(_dtype(condition), np.bool_):
    condition = lax.ne(condition, lax._zero(condition))
  x, y = promote_dtypes(x, y)
  if np.ndim(condition) == 0:
    # lax.select() handles scalar conditions without broadcasting.
    x_arr, y_arr = _broadcast_arrays(x, y)
  else:
    condition, x_arr, y_arr = _broadcast_arrays(condition, x, y)
  try:
    is_always_empty = core.is_empty_shape(x_arr.shape)
  except:
    is_always_empty = False  # can fail with dynamic shapes
  return lax.select(condition, x_arr, y_arr) if not is_always_empty else x_arr
