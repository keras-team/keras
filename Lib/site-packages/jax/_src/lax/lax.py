# Copyright 2018 The JAX Authors.
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

import builtins
from collections.abc import Callable, Sequence
import enum
import functools
from functools import partial
import itertools
import math
import operator
from typing import Any, NamedTuple, TypeVar, Union, cast as type_cast, overload
import warnings

import numpy as np

from jax import tree_util
from jax.sharding import Sharding
from jax.tree_util import tree_map

from jax._src import ad_util
from jax._src import api
from jax._src import api_util
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import pretty_printer as pp
from jax._src import source_info_util
from jax._src import state
from jax._src import util
from jax._src.abstract_arrays import array_types
from jax._src.core import (Primitive, UnshapedArray, ShapedArray,
                           abstract_token, canonicalize_shape)
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import pxla
from jax._src.interpreters import xla
from jax._src.interpreters.batching import RaggedAxis
from jax._src.lax import slicing
from jax._src.lax.utils import (
  _input_dtype, dtype_to_string, standard_abstract_eval,
  standard_multi_result_abstract_eval, standard_primitive)
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import hlo
from jax._src.sharding_impls import (PmapSharding, NamedSharding,
                                     PartitionSpec as P, canonicalize_sharding)
from jax._src.typing import Array, ArrayLike, DimSize, DuckTypedArray, DTypeLike, Shape
from jax._src.util import (NumpyComplexWarning, cache, canonicalize_axis,
                           safe_map, safe_zip, split_list, weakref_lru_cache)

_max = builtins.max
_min = builtins.min
_reduce = functools.reduce

T = TypeVar("T")

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def _matrix_transpose(x: Array) -> Array:
  assert x.ndim >= 2
  return transpose(x, [*range(x.ndim - 2), x.ndim - 1, x.ndim - 2])

def _clip_int_to_valid_range(val: DimSize, dtype, where: str) -> int:
  info = np.iinfo(dtype)
  val = core.concrete_dim_or_error(val, where)
  return core.max_dim(info.min, core.min_dim(val, info.max))

def _validate_shapes(shapes: Sequence[Shape]):
  def _check_static_shape(shape: Shape):
    checked = canonicalize_shape(shape)
    if not all(idx >= 0 for idx in checked):
      msg = f"Only non-negative indices are allowed when broadcasting" \
            f" static shapes, but got shape {shape!r}."
      raise TypeError(msg)

  assert shapes
  if config.dynamic_shapes.value:
    # pass dynamic shapes through unchecked
    return
  else:
    map(_check_static_shape, shapes)

def _try_broadcast_shapes(*shapes: tuple[int, ...], name: str) -> tuple[int, ...]:
  """
  Attempt to broadcast shapes, raising a TypeError if broadcasting fails.
  """
  if not shapes:
    raise TypeError(f"{name}: At least one shape is required.")
  ranks = {len(shape) for shape in shapes}
  if len(ranks) != 1:
    raise TypeError(f'{name}: arrays must have the same number of dimensions,'
                    f' got {ranks}')
  result_shape = []
  for ds in zip(*shapes):
    if all(core.same_referent(d, ds[0]) for d in ds[1:]):
      # if all axes are identical objects, the resulting size is the object
      result_shape.append(ds[0])
    else:
      # if all dims are equal (or 1), the result is the non-1 size
      non_1s = [d for d in ds if not core.definitely_equal(d, 1)]
      if not non_1s:
        result_shape.append(1)
      elif all(core.definitely_equal(non_1s[0], d) for d in non_1s[1:]):
        result_shape.append(non_1s[0])
      else:
        raise TypeError(f'{name} got incompatible shapes for broadcasting: '
                        f'{", ".join(map(str, map(tuple, shapes)))}.')
  return tuple(result_shape)

def asarray(x: ArrayLike) -> Array:
  """Lightweight conversion of ArrayLike input to Array output."""
  if isinstance(x, Array):
    return x
  if isinstance(x, (np.ndarray, np.generic, bool, int, float, builtins.complex)):
    return _convert_element_type(x, weak_type=dtypes.is_weakly_typed(x))  # type: ignore[unused-ignore,bad-return-type]
  else:
    raise TypeError(f"asarray: expected ArrayLike, got {x} of type {type(x)}.")

@overload
def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]: ...

@overload
def broadcast_shapes(*shapes: tuple[int | core.Tracer, ...]
                     ) -> tuple[int | core.Tracer, ...]: ...

def broadcast_shapes(*shapes):
  """Returns the shape that results from NumPy broadcasting of `shapes`."""
  # NOTE: We have both cached and uncached versions to handle Tracers in shapes.
  try:
    return _broadcast_shapes_cached(*shapes)
  except:
    return _broadcast_shapes_uncached(*shapes)

@cache()
def _broadcast_shapes_cached(*shapes: tuple[int, ...]) -> tuple[int, ...]:
  return _broadcast_shapes_uncached(*shapes)

def _broadcast_shapes_uncached(*shapes):
  _validate_shapes(shapes)
  fst, *rst = shapes
  if not rst: return fst

  # First check if we need only rank promotion (and not singleton-broadcasting).
  result_shape = _max(shapes, key=len)
  ndim = len(result_shape)
  if ndim == 0 or all(core.definitely_equal_shape(result_shape[ndim - len(s):], s) for s in shapes):
    return result_shape

  # Next try singleton-broadcasting, padding out ranks using singletons.
  rank_promoted_shapes = tuple((*((1,) * (ndim - len(shape))), *shape) for shape in shapes)
  try:
    return _try_broadcast_shapes(*rank_promoted_shapes, name='broadcast_shapes')
  except TypeError as err:
    # Raise ValueError here for backward compatibility.
    raise ValueError(f"Incompatible shapes for broadcasting: shapes={list(shapes)}") from err

def broadcast_shardings(*avals) -> NamedSharding:
  fst, *rst = avals
  if not rst:
    return fst.sharding

  # First check if we need only rank promotion (and not singleton-broadcasting).
  res_aval = _max(avals, key=lambda a: a.ndim)
  ndim = res_aval.ndim
  if ndim == 0 or all(
      res_aval.sharding.spec[ndim - a.ndim:] == a.sharding.spec for a in avals):
    return res_aval.sharding

  # Next try singleton-broadcasting, padding out ranks using singletons.
  aval_list = []
  for a in avals:
    new_spec = P(*(None,) * (ndim - a.ndim) + a.sharding.spec)
    new_shape = (1,) * (ndim - a.ndim) + a.shape
    aval_list.append(a.update(shape=new_shape,
                              sharding=a.sharding.with_spec(new_spec)))
  return broadcasting_sharding_rule('broadcast_shardings', *aval_list)

def _identity(x): return x

def _extract_tracers_dyn_shape(
    shape: Sequence[int | core.Tracer]
  ) -> tuple[list[core.Tracer], list[int | None]]:
  # Given a sequence representing a shape, pull out Tracers, replacing with None
  if config.dynamic_shapes.value:
    # We must gate this behavior under a flag because otherwise the errors
    # raised are different (and have worse source provenance information).
    dyn_shape = [d for d in shape if isinstance(d, core.Tracer)]
    static_shape = [None if isinstance(d, core.Tracer) else d for d in shape]
    return dyn_shape, static_shape
  else:
    return [], list(shape)  # type: ignore

def _merge_dyn_shape(
    static_shape: Sequence[int | None],
    dyn_shape: Sequence[Any],
  ) -> tuple[int | mlir.Value | core.Tracer, ...]:
  # Replace Nones in static_shape with elements of dyn_shape, in order
  dyn_shape_it = iter(dyn_shape)
  shape = tuple(next(dyn_shape_it) if d is None else d for d in static_shape)
  assert next(dyn_shape_it, None) is None
  return shape

def _dyn_shape_staging_rule(trace, prim, out_aval, *args, **params):
  source_info = source_info_util.current()
  out_tracer = pe.DynamicJaxprTracer(trace, out_aval, source_info)
  eqn = pe.new_jaxpr_eqn([trace.getvar(x) for x in args],
                         [trace.makevar(out_tracer)],
                         prim, params, core.no_effects, source_info)
  trace.frame.add_eqn(eqn)
  return out_tracer


### traceables

def neg(x: ArrayLike) -> Array:
  r"""Elementwise negation: :math:`-x`."""
  return neg_p.bind(x)

def sign(x: ArrayLike) -> Array:
  r"""Elementwise sign.

  For floating-point inputs, returns
  :math:`\mathrm{sign}(x) = \begin{cases}
  -1 & x < 0\\
  -0 & x = -0\\
  \mathit{NaN} & x = \mathit{NaN}\\
  +0 & x = +0\\
  1 & x > 0
  \end{cases}`

  For signed integer inputs, returns
  :math:`\mathrm{sign}(x) = \begin{cases}
  -1 & x < 0\\
  0 & x = 0\\
  1 & x > 0
  \end{cases}`

  For complex inputs, returns the complex phase, i.e.
  :math:`\mathrm{sign}(x) = \frac{x}{|x|}`.
  """
  return sign_p.bind(x)

def nextafter(x1: ArrayLike, x2: ArrayLike) -> Array:
  r"""Returns the next representable value after `x1` in the direction of `x2`.

  Note that in some environments flush-denormal-to-zero semantics is used.
  This means that, around zero, this function returns strictly non-zero
  values which appear as zero in any operations. Consider this example::

    >>> jnp.nextafter(0, 1)  # denormal numbers are representable
    Array(1.e-45, dtype=float32, weak_type=True)
    >>> jnp.nextafter(0, 1) * 1  # but are flushed to zero
    Array(0., dtype=float32, weak_type=True)

  For the smallest usable (i.e. normal) float, use ``tiny`` of ``jnp.finfo``.
  """
  return nextafter_p.bind(x1, x2)

def floor(x: ArrayLike) -> Array:
  r"""Elementwise floor: :math:`\left\lfloor x \right\rfloor`."""
  return floor_p.bind(x)

def ceil(x: ArrayLike) -> Array:
  r"""Elementwise ceiling: :math:`\left\lceil x \right\rceil`."""
  return ceil_p.bind(x)

class RoundingMethod(enum.IntEnum):
  """Rounding strategies for handling halfway values (e.g., 0.5) in
  :func:`jax.lax.round`.
  """

  AWAY_FROM_ZERO = 0
  """Rounds halfway values away from zero (e.g., 0.5 -> 1, -0.5 -> -1)."""

  TO_NEAREST_EVEN = 1
  """Rounds halfway values to the nearest even integer. This is also known
  as “banker’s rounding” (e.g., 0.5 -> 0, 1.5 -> 2).
  """

def round(x: ArrayLike,
          rounding_method: RoundingMethod = RoundingMethod.AWAY_FROM_ZERO
          ) -> Array:
  r"""Elementwise round.

  Rounds values to the nearest integer.

  Args:
    x: an array or scalar value to round.
    rounding_method: the method to use when rounding halfway values
      (e.g., `0.5`). See :class:`jax.lax.RoundingMethod` for possible values.

  Returns:
    An array containing the elementwise rounding of x.
  """
  rounding_method = RoundingMethod(rounding_method)
  return round_p.bind(x, rounding_method=rounding_method)

def is_finite(x: ArrayLike) -> Array:
  r"""Elementwise :math:`\mathrm{isfinite}`.

  For each element x returns `True` if and only if x is not :math:`\pm\infty` or
  :math:`\mathit{NaN}`.
  """
  return is_finite_p.bind(x)

def exp(x: ArrayLike) -> Array:
  r"""Elementwise exponential: :math:`e^x`."""
  return exp_p.bind(x)

def exp2(x: ArrayLike) -> Array:
  r"""Elementwise base-2 exponential: :math:`2^x`."""
  return exp2_p.bind(x)

def expm1(x: ArrayLike) -> Array:
  r"""Elementwise :math:`e^{x} - 1`."""
  return expm1_p.bind(x)

def log(x: ArrayLike) -> Array:
  r"""Elementwise natural logarithm: :math:`\mathrm{log}(x)`."""
  return log_p.bind(x)

def log1p(x: ArrayLike) -> Array:
  r"""Elementwise :math:`\mathrm{log}(1 + x)`."""
  return log1p_p.bind(x)

def tanh(x: ArrayLike) -> Array:
  r"""Elementwise hyperbolic tangent: :math:`\mathrm{tanh}(x)`."""
  return tanh_p.bind(x)

def logistic(x: ArrayLike) -> Array:
  r"""Elementwise logistic (sigmoid) function: :math:`\frac{1}{1 + e^{-x}}`."""
  return logistic_p.bind(x)

def sin(x: ArrayLike) -> Array:
  r"""Elementwise sine: :math:`\mathrm{sin}(x)`."""
  return sin_p.bind(x)

def cos(x: ArrayLike) -> Array:
  r"""Elementwise cosine: :math:`\mathrm{cos}(x)`."""
  return cos_p.bind(x)

def atan2(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise arc tangent of two variables:
    :math:`\mathrm{atan}({x \over y})`."""
  return atan2_p.bind(x, y)

def real(x: ArrayLike) -> Array:
  r"""Elementwise extract real part: :math:`\mathrm{Re}(x)`.

  Returns the real part of a complex number.
  """
  return real_p.bind(x)

def imag(x: ArrayLike) -> Array:
  r"""Elementwise extract imaginary part: :math:`\mathrm{Im}(x)`.

  Returns the imaginary part of a complex number.
  """
  return imag_p.bind(x)

def complex(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise make complex number: :math:`x + jy`.

  Builds a complex number from real and imaginary parts.
  """
  return complex_p.bind(x, y)

def conj(x: ArrayLike) -> Array:
  r"""Elementwise complex conjugate function: :math:`\overline{x}`."""
  # TODO(mattjj): remove input_dtype, not needed anymore
  return conj_p.bind(x, input_dtype=_dtype(x))

def abs(x: ArrayLike) -> Array:
  r"""Elementwise absolute value: :math:`|x|`."""
  return abs_p.bind(x)

def pow(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise power: :math:`x^y`."""
  return pow_p.bind(x, y)

def integer_pow(x: ArrayLike, y: int) -> Array:
  r"""Elementwise power: :math:`x^y`, where :math:`y` is a fixed integer."""
  return integer_pow_p.bind(x, y=y)

def sqrt(x: ArrayLike) -> Array:
  r"""Elementwise square root: :math:`\sqrt{x}`."""
  return sqrt_p.bind(x)

def rsqrt(x: ArrayLike) -> Array:
  r"""Elementwise reciprocal square root:  :math:`1 \over \sqrt{x}`."""
  return rsqrt_p.bind(x)

def cbrt(x: ArrayLike) -> Array:
  r"""Elementwise cube root: :math:`\sqrt[3]{x}`."""
  return cbrt_p.bind(x)

def bitwise_not(x: ArrayLike) -> Array:
  r"""Elementwise NOT: :math:`\neg x`."""
  return not_p.bind(x)

def bitwise_and(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise AND: :math:`x \wedge y`."""
  return and_p.bind(x, y)

def bitwise_or(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise OR: :math:`x \vee y`."""
  return or_p.bind(x, y)

def bitwise_xor(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise exclusive OR: :math:`x \oplus y`."""
  return xor_p.bind(x, y)

def population_count(x: ArrayLike) -> Array:
  r"""Elementwise popcount, count the number of set bits in each element."""
  return population_count_p.bind(x)

def clz(x: ArrayLike) -> Array:
  r"""Elementwise count-leading-zeros."""
  return clz_p.bind(x)

def add(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise addition: :math:`x + y`."""
  return add_p.bind(x, y)

def sub(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise subtraction: :math:`x - y`."""
  return sub_p.bind(x, y)

def mul(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise multiplication: :math:`x \times y`."""
  return mul_p.bind(x, y)

def div(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise division: :math:`x \over y`.

  Integer division overflow
  (division by zero or signed division of INT_SMIN with -1)
  produces an implementation defined value.
  """
  return div_p.bind(x, y)

def rem(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise remainder: :math:`x \bmod y`.

  The sign of the result is taken from the dividend,
  and the absolute value of the result is always
  less than the divisor's absolute value.

  Integer division overflow
  (remainder by zero or remainder of INT_SMIN with -1)
  produces an implementation defined value.
  """
  return rem_p.bind(x, y)

def max(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise maximum: :math:`\mathrm{max}(x, y)`

  For complex numbers, uses a lexicographic comparison on the
  `(real, imaginary)` pairs."""
  return max_p.bind(x, y)

def min(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise minimum:  :math:`\mathrm{min}(x, y)`

  For complex numbers, uses a lexicographic comparison on the
  `(real, imaginary)` pairs."""
  return min_p.bind(x, y)

def shift_left(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise left shift: :math:`x \ll y`."""
  return shift_left_p.bind(x, y)

def shift_right_arithmetic(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise arithmetic right shift: :math:`x \gg y`."""
  return shift_right_arithmetic_p.bind(x, y)

def shift_right_logical(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise logical right shift: :math:`x \gg y`."""
  return shift_right_logical_p.bind(x, y)

def eq(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise equals: :math:`x = y`."""
  return eq_p.bind(x, y)

def ne(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise not-equals: :math:`x \neq y`."""
  return ne_p.bind(x, y)

def ge(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise greater-than-or-equals: :math:`x \geq y`."""
  return ge_p.bind(x, y)

def gt(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise greater-than: :math:`x > y`."""
  return gt_p.bind(x, y)

def le(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise less-than-or-equals: :math:`x \leq y`."""
  return le_p.bind(x, y)

def lt(x: ArrayLike, y: ArrayLike) -> Array:
  r"""Elementwise less-than: :math:`x < y`."""
  return lt_p.bind(x, y)

def convert_element_type(operand: ArrayLike,
                         new_dtype: DTypeLike | dtypes.ExtendedDType) -> Array:
  """Elementwise cast.

  Wraps XLA's `ConvertElementType
  <https://www.tensorflow.org/xla/operation_semantics#convertelementtype>`_
  operator, which performs an elementwise conversion from one type to another.
  Similar to a C++ `static_cast`.

  Args:
    operand: an array or scalar value to be cast.
    new_dtype: a NumPy dtype representing the target type.

  Returns:
    An array with the same shape as `operand`, cast elementwise to `new_dtype`.
  """
  return _convert_element_type(operand, new_dtype, weak_type=False)  # type: ignore[unused-ignore,bad-return-type]

def _convert_element_type(
    operand: ArrayLike,
    new_dtype: DTypeLike | dtypes.ExtendedDType | None = None,
    weak_type: bool = False,
    sharding: Sharding | None = None,
    warn_on_complex_to_real_cast: bool = True):
  if hasattr(operand, '__jax_array__'):
    operand = operand.__jax_array__()

  # Don't canonicalize old_dtype because x64 context might cause
  # un-canonicalized operands to be passed in.
  old_dtype = dtypes.dtype(operand, canonicalize=False)

  if (isinstance(new_dtype, dtypes.ExtendedDType) or
      isinstance(old_dtype, dtypes.ExtendedDType)):
    if sharding is not None or weak_type: raise NotImplementedError
    if new_dtype == old_dtype: return operand
    if (isinstance(new_dtype, dtypes.ExtendedDType) and
        isinstance(old_dtype, dtypes.ExtendedDType)):
      old_rep_dtype = core.physical_element_aval(old_dtype).dtype
      new_rep_dtype = core.physical_element_aval(new_dtype).dtype
      raise ValueError(
          "cannot directly convert between extended dtypes: from "
          f"{dtype_to_string(old_dtype)} to {dtype_to_string(new_dtype)}. "
          "Instead, convert to and from their representation dtypes, e.g.:\n"
          f"{dtype_to_string(old_dtype)} -> {dtype_to_string(old_rep_dtype)} "
          f"-> {dtype_to_string(new_rep_dtype)} -> {dtype_to_string(new_dtype)}")
    if isinstance(new_dtype, dtypes.ExtendedDType):
      return to_edtype_p.bind(operand, edtype=new_dtype)
    return from_edtype_p.bind(operand, dtype=np.dtype(new_dtype))

  new_dtype = type_cast(DTypeLike | None, new_dtype)

  old_weak_type = dtypes.is_weakly_typed(operand)
  if new_dtype is None:
    new_dtype = old_dtype
  else:
    new_dtype = np.dtype(new_dtype)
  new_dtype = dtypes.dtype(new_dtype, canonicalize=True)

  if (config.sharding_in_types.value and sharding is None and
      isinstance(operand, Array)):
    sharding = operand.sharding

  sharding = canonicalize_sharding(sharding, check_mesh_consistency=False)  # type: ignore

  if (warn_on_complex_to_real_cast and
      dtypes.issubdtype(old_dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    msg = "Casting complex values to real discards the imaginary part"
    warnings.warn(msg, NumpyComplexWarning, stacklevel=2)

  # Python has big integers, but convert_element_type(2 ** 100, np.float32) need
  # not be an error since the target dtype fits the value. Handle this case by
  # converting to a NumPy array before calling bind. Without this step, we'd
  # first canonicalize the input to a value of dtype int32 or int64, leading to
  # an overflow error.
  if type(operand) is int:
    operand = np.asarray(operand).astype(new_dtype)
    old_weak_type = False

  if ((old_dtype, old_weak_type) == (new_dtype, weak_type) and
      isinstance(operand, Array) and
      not (isinstance(operand, core.Tracer) and core.is_concrete(operand)) and
      (sharding is None or getattr(operand, 'sharding', None) == sharding)):
    return operand
  else:
    return convert_element_type_p.bind(
        operand, new_dtype=new_dtype, weak_type=bool(weak_type),
        sharding=sharding)

def bitcast_convert_type(operand: ArrayLike, new_dtype: DTypeLike) -> Array:
  """Elementwise bitcast.

  Wraps XLA's `BitcastConvertType
  <https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype>`_
  operator, which performs a bit cast from one type to another.

  The output shape depends on the size of the input and output dtypes with
  the following logic::

    if new_dtype.itemsize == operand.dtype.itemsize:
      output_shape = operand.shape
    if new_dtype.itemsize < operand.dtype.itemsize:
      output_shape = (*operand.shape, operand.dtype.itemsize // new_dtype.itemsize)
    if new_dtype.itemsize > operand.dtype.itemsize:
      assert operand.shape[-1] * operand.dtype.itemsize == new_dtype.itemsize
      output_shape = operand.shape[:-1]

  Args:
    operand: an array or scalar value to be cast
    new_dtype: the new type. Should be a NumPy type.

  Returns:
    An array of shape `output_shape` (see above) and type `new_dtype`,
    constructed from the same bits as operand.
  """
  new_dtype = dtypes.canonicalize_dtype(new_dtype)
  return bitcast_convert_type_p.bind(operand, new_dtype=new_dtype)

def clamp(min: ArrayLike, x: ArrayLike, max: ArrayLike) -> Array:
  r"""Elementwise clamp.

  Returns :math:`\mathrm{clamp}(x) = \begin{cases}
  \mathit{min} & \text{if } x < \mathit{min},\\
  \mathit{max} & \text{if } x > \mathit{max},\\
  x & \text{otherwise}
  \end{cases}`.
  """
  return clamp_p.bind(min, x, max)


@weakref_lru_cache
def _trace_composite_to_jaxpr(fun, in_tree, in_avals):
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  debug_info = pe.tracing_debug_info(fun, in_tree, out_tree, False, "composite")
  jaxpr, _, consts, _ = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug_info)
  # TODO(danfm): support const inputs to composite.
  assert not consts
  closed_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
  return closed_jaxpr, out_tree


def composite(
    decomposition: Callable,
    name: str,
    version: int = 0,
):
  """Composite with semantics defined by the decomposition function.

  A composite is a higher-order JAX function that encapsulates an operation mad
  up (composed) of other JAX functions. The semantics of the op are implemented
  by the ``decomposition`` function. In other words, the defined composite
  function can be replaced with its decomposed implementation without changing
  the semantics of the encapsulated operation.

  The compiler can recognize specific composite operations by their ``name``,
  ``version``, ``kawargs``, and dtypes to emit more efficient code, potentially
  leveraging hardware-specific instructions or optimizations. If the compiler
  doesn't recognize the composite, it falls back to compiling the
  ``decomposition`` function.

  Consider a "tangent" composite operation. Its ``decomposition`` function could
  be implemented as ``sin(x) / cos(x)``. A hardware-aware compiler could
  recognize the "tangent" composite and emit a single ``tangent`` instruction
  instead of three separate instructions (``sin``, ``divide``, and ``cos``).
  With compilers for hardwares without dedicated tangent support, it would fall
  back to compiling the decomposition.

  This is useful for preserving high level abstraction that would otherwise be
  lost while lowering which allows for easier pattern-matching in low-level IR.

  Args:
    decomposition: function that implements the semantics of the composite op.
    name: name of the encapsulated operation.
    version: optional int to indicate semantic changes to the composite.

  Returns:
    out: callable composite function. Note that positional arguments to this
      function should be interpreted as inputs and keyword arguments should be
      interpreted as attributes of the op.

  Examples:
    Tangent kernel:
    >>> def my_tangent_composite(x):
    ...   return lax.composite(
    ...     lambda x: lax.sin(x) / lax.cos(x), name='my.tangent'
    ...   )(x)
    ...
    >>> pi = jnp.pi
    >>> x = jnp.array([0.0, pi / 4, 3 * pi / 4, pi])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(my_tangent_composite(x))
    ...   print(lax.tan(x))
    [ 0.  1. -1.  0.]
    [ 0.  1. -1.  0.]

    The recommended way to create composites is via a decorator. Use `/` and `*`
    in the function signature to be explicit about positional and keyword
    arguments respectively:
    >>> @partial(lax.composite, name="my.softmax")
    ... def my_softmax_composite(x, /, *, axis):
    ...   return jax.nn.softmax(x, axis)
  """
  @functools.wraps(decomposition)
  def _decorator(*args, **kwargs):
    flat_args, in_tree = tree_util.tree_flatten(args)
    in_avals = tuple(core.get_aval(x) for x in flat_args)
    closed_jaxpr, out_tree = _trace_composite_to_jaxpr(
        partial(decomposition, **kwargs), in_tree, in_avals
    )
    out_flat = composite_p.bind(
        *flat_args,
        name=name,
        attributes=tuple((k, v) for k, v in kwargs.items()),
        version=version,
        jaxpr=closed_jaxpr,
    )
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return _decorator


def _composite_lowering(
    ctx: mlir.LoweringRuleContext,
    *args: Any,
    name: str,
    attributes: Sequence[tuple[str, Any]],
    version: int,
    jaxpr: core.ClosedJaxpr,
):
  """Makes composite which calls the implementation function.

  Lowering a composite primitive to a ``stablehlo.composite`` op.

  Args:
    ctx: The MLIR context.
    *args: The arguments to the composite.
    name: The name of the composite.
    attributes: The attributes of the composite.
    version: The version of the composite.
    jaxpr: The jaxpr of the underlying composite.

  Returns:
    The results of the composite.
  """
  func_op, _, _ = mlir.lower_called_computation(
      name,
      ctx.name_stack,
      jaxpr,
      ctx.module_context,
      ctx.avals_out,
      ctx.tokens_in,
  )
  composite_attrs = {k : mlir.ir_attribute(v) for k, v in attributes}
  symbol_name = func_op.name.value
  composite = hlo.CompositeOp(
      func_op.type.results,
      mlir.flatten_ir_values(args),
      name=ir.StringAttr.get(name),
      decomposition=ir.FlatSymbolRefAttr.get(symbol_name),
      composite_attributes=ir.DictAttr.get(composite_attrs),
      version=mlir.i32_attr(version),
  )
  return composite.results


def _composite_impl(*args, jaxpr, **_):
  return core.jaxpr_as_fun(jaxpr)(*args)


def _composite_abstract_eval(*args, jaxpr, **_):
  del args
  return jaxpr.out_avals


def composite_jvp(*args, **_):
  del args
  raise ValueError(
      "JVP rule for composite not implemented. You can use `jax.custom_jvp` to "
      "add support. See "
      "https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html"
  )


def composite_transpose(*args, **_):
  del args
  raise ValueError(
      "Transpose rule for composite not implemented. You can use"
      "`jax.custom_jvp` or `jax.custom_vjp` to add support. See "
      "https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html"
  )


composite_p = core.Primitive("composite")
composite_p.def_impl(_composite_impl)
composite_p.def_abstract_eval(_composite_abstract_eval)
composite_p.multiple_results = True
ad.primitive_jvps[composite_p] = composite_jvp
ad.primitive_transposes[composite_p] = composite_transpose
mlir.register_lowering(composite_p, _composite_lowering)


def concatenate(operands: Array | Sequence[ArrayLike], dimension: int) -> Array:
  """Concatenates a sequence of arrays along `dimension`.

  Wraps XLA's `Concatenate
  <https://www.tensorflow.org/xla/operation_semantics#concatenate>`_
  operator.

  Args:
    operands: a sequence of arrays to concatenate. The arrays must have equal
      shapes, except in the `dimension` axis.
    dimension: the dimension along which to concatenate the arrays.

  Returns:
    An array containing the concatenation.
  """
  if len(operands) == 0:
    raise ValueError("concatenate requires a non-empty sequences of arrays")
  if len(operands) == 1:
    op, = operands
    if isinstance(op, Array):
      return op
  return concatenate_p.bind(*operands, dimension=dimension)


def split(operand: ArrayLike, sizes: Sequence[int],
          axis: int = 0) -> Sequence[Array]:
  """Splits an array along ``axis``.

  Args:
    operand: an array to split
    sizes: the sizes of the split arrays. The sum of the sizes must be equal
      to the size of the ``axis`` dimension of ``operand``.
    axis: the axis along which to split the array.

  Returns:
    A sequence of ``len(sizes)`` arrays. If ``sizes`` is
    ``[s1, s2, ...]``, this function returns chunks of sizes ``s1``, ``s2``,
    taken along ``axis``.
  """
  operand = asarray(operand)
  return split_p.bind(operand, sizes=tuple(sizes),
                      axis=canonicalize_axis(axis, operand.ndim))


_precision_strings: dict[Any, Precision] = {}

class Precision(enum.Enum):
  """Precision enum for lax matrix multiply related functions.

  The device-dependent `precision` argument to JAX functions generally
  controls the tradeoff between speed and accuracy for array computations on
  accelerator backends, (i.e. TPU and GPU). Has no impact on CPU backends.
  This only has an effect on float32 computations, and does not affect the
  input/output datatypes. Members are:

  DEFAULT:
    Fastest mode, but least accurate. On TPU: performs float32 computations in
    bfloat16. On GPU: uses tensorfloat32 if available (e.g. on A100 and H100
    GPUs), otherwise standard float32 (e.g. on V100 GPUs). Aliases:
    ``'default'``, ``'fastest'``.
  HIGH:
    Slower but more accurate. On TPU: performs float32 computations in 3
    bfloat16 passes. On GPU: uses tensorfloat32 where available, otherwise
    float32. Aliases: ``'high'``..
  HIGHEST:
    Slowest but most accurate. On TPU: performs float32 computations in 6
    bfloat16. Aliases: ``'highest'``. On GPU: uses float32.
  """

  DEFAULT = 0
  HIGH = 1
  HIGHEST = 2

  @classmethod
  def _missing_(cls, value: object) -> Precision | None:
    return _precision_strings.get(value)

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}.{self.name}'

  def __str__(self) -> str:
    return self.name


_precision_strings['highest'] = Precision.HIGHEST
_precision_strings['float32'] = Precision.HIGHEST
_precision_strings['high'] = Precision.HIGH
_precision_strings['bfloat16_3x'] = Precision.HIGH
_precision_strings['tensorfloat32'] = Precision.HIGH
_precision_strings['default'] = Precision.DEFAULT
_precision_strings['bfloat16'] = Precision.DEFAULT
_precision_strings['fastest'] = Precision.DEFAULT
_precision_strings[None] = Precision.DEFAULT


class DotAlgorithm(NamedTuple):
  """Specify the algorithm used for computing dot products.

  When used to specify the ``precision`` input to :func:`~jax.lax.dot`,
  :func:`~jax.lax.dot_general`, and other dot product functions, this data
  structure is used for controlling the properties of the algorithm used for
  computing the dot product. This API controls the precision used for the
  computation, and allows users to access hardware-specific accelerations.

  Support for these algorithms is platform dependent, and using an unsupported
  algorithm will raise a Python exception when the computation is compiled. The
  algorithms that are known to be supported on at least some platforms are
  listed in the :class:`~jax.lax.DotAlgorithmPreset` enum, and these are a
  good starting point for experimenting with this API.

  A "dot algorithm" is specified by the following parameters:

  * ``lhs_precision_type`` and ``rhs_precision_type``, the data types that the
    LHS and RHS of the operation are rounded to.
  * ``accumulation_type`` the data type used for accumulation.
  * ``lhs_component_count``, ``rhs_component_count``, and
    ``num_primitive_operations`` apply to algorithms that decompose the LHS
    and/or RHS into multiple components and execute multiple operations on
    those values, usually to emulate a higher precision. For algorithms with no
    decomposition, these values should be set to ``1``.
  * ``allow_imprecise_accumulation`` to specify if accumulation in lower
    precision is permitted for some steps (e.g.
    ``CUBLASLT_MATMUL_DESC_FAST_ACCUM``).

  The `StableHLO spec <https://openxla.org/stablehlo/spec#dot_general>`_ for
  the dot operation doesn't require that the precision types be the same as the
  storage types for the inputs or outputs, but some plaforms may require that
  these types match. Furthermore, the return type of
  :func:`~jax.lax.dot_general` is always defined by the ``accumulation_type``
  parameter of the input algorithm, if specified.

  Examples:

    Accumulate two 16-bit floats using a 32-bit float accumulator:

    >>> algorithm = DotAlgorithm(
    ...     lhs_precision_type=np.float16,
    ...     rhs_precision_type=np.float16,
    ...     accumulation_type=np.float32,
    ... )
    >>> lhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    >>> rhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    >>> dot(lhs, rhs, precision=algorithm)  # doctest: +SKIP
    array([ 1.,  4.,  9., 16.], dtype=float16)

    Or, equivalently, using a preset:

    >>> algorithm = DotAlgorithmPreset.F16_F16_F32
    >>> dot(lhs, rhs, precision=algorithm)  # doctest: +SKIP
    array([ 1.,  4.,  9., 16.], dtype=float16)

    Presets can also be specified by name:

    >>> dot(lhs, rhs, precision="F16_F16_F32")  # doctest: +SKIP
    array([ 1.,  4.,  9., 16.], dtype=float16)

    The ``preferred_element_type`` parameter can be used to return the output
    without downcasting the accumulation type:

    >>> dot(lhs, rhs, precision="F16_F16_F32", preferred_element_type=np.float32)  # doctest: +SKIP
    array([ 1.,  4.,  9., 16.], dtype=float32)
  """

  lhs_precision_type: DTypeLike
  rhs_precision_type: DTypeLike
  accumulation_type: DTypeLike
  lhs_component_count: int = 1
  rhs_component_count: int = 1
  num_primitive_operations: int = 1
  allow_imprecise_accumulation: bool = False

  def _convert_to_hlo_attr(self, lhs_dtype: DTypeLike,
                           rhs_dtype: DTypeLike) -> hlo.DotAlgorithm:
    del lhs_dtype, rhs_dtype  # unused
    return hlo.DotAlgorithm.get(
        mlir.dtype_to_ir_type(dtypes.dtype(self.lhs_precision_type)),
        mlir.dtype_to_ir_type(dtypes.dtype(self.rhs_precision_type)),
        mlir.dtype_to_ir_type(dtypes.dtype(self.accumulation_type)),
        self.lhs_component_count,
        self.rhs_component_count,
        self.num_primitive_operations,
        self.allow_imprecise_accumulation,
    )


class DotAlgorithmPreset(enum.Enum):
  """An enum of known algorithms for computing dot products.

  This ``Enum`` provides a named set of :class:`~jax.lax.DotAlgorithm` objects
  that are known to be supported on at least platform. See the
  :class:`~jax.lax.DotAlgorithm` documentation for more details about the
  behavior of these algorithms.

  An algorithm can be selected from this list when calling :func:`~jax.lax.dot`,
  :func:`~jax.lax.dot_general`, or most other JAX dot product functions, by
  passing either a member of this ``Enum`` or it's name as a string using the
  ``precision`` argument.

  For example, users can specify the preset using this ``Enum`` directly:

  >>> lhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
  >>> rhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
  >>> algorithm = DotAlgorithmPreset.F16_F16_F32
  >>> dot(lhs, rhs, precision=algorithm)  # doctest: +SKIP
  array([ 1.,  4.,  9., 16.], dtype=float16)

  or, equivalently, they can be specified by name:

  >>> dot(lhs, rhs, precision="F16_F16_F32")  # doctest: +SKIP
  array([ 1.,  4.,  9., 16.], dtype=float16)

  The names of the presets are typically ``LHS_RHS_ACCUM`` where ``LHS`` and
  ``RHS`` are the element types of the ``lhs`` and ``rhs`` inputs
  respectively, and ``ACCUM`` is the element type of the accumulator. Some
  presets have an extra suffix, and the meaning of each of these is
  documented below. The supported presets are:
  """
  DEFAULT = enum.auto()
  """An algorithm will be selected based on input and output types."""

  ANY_F8_ANY_F8_F32 = enum.auto()
  """Accepts any float8 input types and accumulates into float32."""

  ANY_F8_ANY_F8_F32_FAST_ACCUM = enum.auto()
  """Like ``ANY_F8_ANY_F8_F32``, but using faster accumulation with the cost
  of lower accuracy.
  """

  ANY_F8_ANY_F8_ANY = enum.auto()
  """Like ``ANY_F8_ANY_F8_F32``, but the accumulation type is controlled by
  ``preferred_element_type``.
  """

  ANY_F8_ANY_F8_ANY_FAST_ACCUM = enum.auto()
  """Like ``ANY_F8_ANY_F8_F32_FAST_ACCUM``, but the accumulation type is
  controlled by ``preferred_element_type``.
  """

  F16_F16_F16 = enum.auto()
  F16_F16_F32 = enum.auto()
  BF16_BF16_BF16 = enum.auto()
  BF16_BF16_F32 = enum.auto()
  BF16_BF16_F32_X3 = enum.auto()
  """The ``_X3`` suffix indicates that the algorithm uses 3 operations to
  emulate higher precision.
  """

  BF16_BF16_F32_X6 = enum.auto()
  """Like ``BF16_BF16_F32_X3``, but using 6 operations instead of 3."""

  TF32_TF32_F32 = enum.auto()
  TF32_TF32_F32_X3 = enum.auto()
  """The ``_X3`` suffix indicates that the algorithm uses 3 operations to
  emulate higher precision.
  """

  F32_F32_F32 = enum.auto()
  F64_F64_F64 = enum.auto()

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}.{self.name}'

  def __str__(self) -> str:
    return self.name

  @property
  def supported_lhs_types(self) -> tuple[DTypeLike, ...] | None:
    match self:
      case (
          DotAlgorithmPreset.DEFAULT
          | DotAlgorithmPreset.ANY_F8_ANY_F8_F32
          | DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM
          | DotAlgorithmPreset.ANY_F8_ANY_F8_ANY
          | DotAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM
      ):
        return None
      case DotAlgorithmPreset.F16_F16_F16 | DotAlgorithmPreset.F16_F16_F32:
        return (np.float16,)
      case (
          DotAlgorithmPreset.BF16_BF16_BF16 |
          DotAlgorithmPreset.BF16_BF16_F32
      ):
        # These algorithms support either f32 or bf32 input storage types.
        # If either of those types are provided as input, we use the provided
        # type. If not, we explicitly cast to bfloat16.
        return (dtypes.bfloat16, np.float32)
      case DotAlgorithmPreset.F64_F64_F64:
        return (np.float64,)
      case _:
        return (np.float32,)

  @property
  def supported_rhs_types(self) -> tuple[DTypeLike, ...] | None:
    return self.supported_lhs_types

  @property
  def accumulation_type(self) -> DTypeLike | None:
    match self:
      case (
          DotAlgorithmPreset.DEFAULT
          | DotAlgorithmPreset.ANY_F8_ANY_F8_ANY
          | DotAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM
      ):
        return None
      case DotAlgorithmPreset.F16_F16_F16:
        return np.float16
      case DotAlgorithmPreset.BF16_BF16_BF16:
        return dtypes.bfloat16
      case DotAlgorithmPreset.F64_F64_F64:
        return np.float64
      case _:
        return np.float32

  def supported_output_types(
      self, lhs_dtype: DTypeLike, rhs_dtype: DTypeLike
  ) -> tuple[DTypeLike, ...] | None:
    match self:
      case (
          DotAlgorithmPreset.ANY_F8_ANY_F8_F32
          | DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM
      ):
        return (
            np.float32,
            np.float16,
            dtypes.bfloat16,
            dtypes.float8_e4m3fn,
            dtypes.float8_e5m2,
            dtypes.float8_e5m2fnuz,
            dtypes.float8_e4m3fnuz,
            dtypes.float8_e4m3b11fnuz,
        )
      case DotAlgorithmPreset.F16_F16_F32:
        # F16 output is only supported with F16 inputs.
        if dtypes.promote_types(lhs_dtype, rhs_dtype) == np.float16:
          return (np.float32, np.float16)
        else:
          return (np.float32,)
      case DotAlgorithmPreset.BF16_BF16_F32:
        # BF16 output is only supported with BF16 inputs.
        if dtypes.promote_types(lhs_dtype, rhs_dtype) == dtypes.bfloat16:
          return (np.float32, dtypes.bfloat16)
        else:
          return (np.float32,)
      case _:
        accumulation_type = self.accumulation_type
        return None if accumulation_type is None else (accumulation_type,)

  def _convert_to_hlo_attr(self, lhs_dtype: DTypeLike,
                            rhs_dtype: DTypeLike) -> hlo.DotAlgorithm | None:
    f16 = ir.F16Type.get()
    f32 = ir.F32Type.get()
    f64 = ir.F64Type.get()
    bf16 = ir.BF16Type.get()
    tf32 = ir.FloatTF32Type.get()
    match self:
      case (
          DotAlgorithmPreset.ANY_F8_ANY_F8_F32
          | DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM
          | DotAlgorithmPreset.ANY_F8_ANY_F8_ANY
          | DotAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM
      ):
        fp8_dtypes = [
            np.dtype(dtypes.float8_e4m3b11fnuz),
            np.dtype(dtypes.float8_e4m3fn),
            np.dtype(dtypes.float8_e4m3fnuz),
            np.dtype(dtypes.float8_e5m2),
            np.dtype(dtypes.float8_e5m2fnuz),
        ]
        if dtypes.float8_e3m4 is not None:
          fp8_dtypes += [np.dtype(dtypes.float8_e3m4)]
        if dtypes.float8_e4m3 is not None:
          fp8_dtypes += [np.dtype(dtypes.float8_e4m3)]
        if lhs_dtype not in fp8_dtypes or rhs_dtype not in fp8_dtypes:
          raise ValueError(
              f"The dot algorithm '{self}' requires both inputs to have float8 "
              f'dtypes. Got {lhs_dtype} and {rhs_dtype} instead.'
          )
        lhs = mlir.dtype_to_ir_type(dtypes.dtype(lhs_dtype))
        rhs = mlir.dtype_to_ir_type(dtypes.dtype(rhs_dtype))
        acc = ir.F32Type.get()
        return hlo.DotAlgorithm.get(
            lhs,
            rhs,
            acc,
            1,
            1,
            1,
            self == DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM,
        )
      case DotAlgorithmPreset.F16_F16_F16:
        return hlo.DotAlgorithm.get(f16, f16, f16, 1, 1, 1, False)
      case DotAlgorithmPreset.F16_F16_F32:
        return hlo.DotAlgorithm.get(f16, f16, f32, 1, 1, 1, False)
      case DotAlgorithmPreset.BF16_BF16_BF16:
        return hlo.DotAlgorithm.get(bf16, bf16, bf16, 1, 1, 1, False)
      case DotAlgorithmPreset.BF16_BF16_F32:
        return hlo.DotAlgorithm.get(bf16, bf16, f32, 1, 1, 1, False)
      case DotAlgorithmPreset.BF16_BF16_F32_X3:
        return hlo.DotAlgorithm.get(bf16, bf16, f32, 1, 1, 3, False)
      case DotAlgorithmPreset.BF16_BF16_F32_X6:
        return hlo.DotAlgorithm.get(bf16, bf16, f32, 1, 1, 6, False)
      case DotAlgorithmPreset.TF32_TF32_F32:
        return hlo.DotAlgorithm.get(tf32, tf32, f32, 1, 1, 1, False)
      case DotAlgorithmPreset.TF32_TF32_F32_X3:
        return hlo.DotAlgorithm.get(tf32, tf32, f32, 1, 1, 3, False)
      case DotAlgorithmPreset.F32_F32_F32:
        return hlo.DotAlgorithm.get(f32, f32, f32, 1, 1, 1, False)
      case DotAlgorithmPreset.F64_F64_F64:
        return hlo.DotAlgorithm.get(f64, f64, f64, 1, 1, 1, False)
      case _:
        return None


PrecisionLike = Union[
    None,
    str,
    Precision,
    tuple[str, str],
    tuple[Precision, Precision],
    DotAlgorithm,
    DotAlgorithmPreset,
]
CanonicalPrecision = Union[
    None,
    tuple[Precision, Precision],
    DotAlgorithm,
    DotAlgorithmPreset,
]


def dot(lhs: Array, rhs: Array, precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None) -> Array:
  """Vector/vector, matrix/vector, and matrix/matrix multiplication.

  Wraps XLA's `Dot <https://www.tensorflow.org/xla/operation_semantics#dot>`_
  operator.

  For more general contraction, see the :func:`jax.lax.dot_general` operator.

  Args:
    lhs: an array of dimension 1 or 2.
    rhs: an array of dimension 1 or 2.
    precision: Optional. This parameter controls the numerics of the
      computation, and it can be one of the following:

      - ``None``, which means the default precision for the current backend,
      - a :class:`~jax.lax.Precision` enum value or a tuple of two
        :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and
        ``rhs``, or
      - a :class:`~jax.lax.DotAlgorithm` or a
        :class:`~jax.lax.DotAlgorithmPreset` indicating the algorithm that
        must be used to accumulate the dot product.

    preferred_element_type: Optional. This parameter controls the data type
      output by the dot product. By default, the output element type of this
      operation will match the ``lhs`` and ``rhs`` input element types under
      the usual type promotion rules. Setting ``preferred_element_type`` to a
      specific ``dtype`` will mean that the operation returns that element type.
      When ``precision`` is not a :class:`~jax.lax.DotAlgorithm` or
      :class:`~jax.lax.DotAlgorithmPreset`, ``preferred_element_type`` provides
      a hint to the compiler to accumulate the dot product using this data type.

  Returns:
    An array containing the product.
  """
  if 1 <= lhs.ndim <= 2 and 1 <= rhs.ndim <= 2 and core.definitely_equal(lhs.shape[-1], rhs.shape[0]):
    return dot_general(lhs, rhs, (((lhs.ndim - 1,), (0,)), ((), ())),
                       precision=precision,
                       preferred_element_type=preferred_element_type)
  else:
    raise TypeError("Incompatible shapes for dot: got {} and {}.".format(
        lhs.shape, rhs.shape))


DotDimensionNumbers = tuple[tuple[Sequence[int], Sequence[int]],
                            tuple[Sequence[int], Sequence[int]]]

def dot_general(lhs: ArrayLike, rhs: ArrayLike, dimension_numbers: DotDimensionNumbers,
                precision: PrecisionLike = None,
                preferred_element_type: DTypeLike | None = None,
                out_sharding=None) -> Array:
  """General dot product/contraction operator.

  Wraps XLA's `DotGeneral
  <https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`_
  operator.

  The semantics of ``dot_general`` are complicated, but most users should not have to
  use it directly. Instead, you can use higher-level functions like :func:`jax.numpy.dot`,
  :func:`jax.numpy.matmul`, :func:`jax.numpy.tensordot`, :func:`jax.numpy.einsum`,
  and others which will construct appropriate calls to ``dot_general`` under the hood.
  If you really want to understand ``dot_general`` itself, we recommend reading XLA's
  `DotGeneral  <https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`_
  operator documentation.

  Args:
    lhs: an array
    rhs: an array
    dimension_numbers: a tuple of tuples of sequences of ints of the form
      ``((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
      rhs_batch_dims))``
    precision: Optional. This parameter controls the numerics of the
      computation, and it can be one of the following:

      - ``None``, which means the default precision for the current backend,
      - a :class:`~jax.lax.Precision` enum value or a tuple of two
        :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and
        ``rhs``, or
      - a :class:`~jax.lax.DotAlgorithm` or a
        :class:`~jax.lax.DotAlgorithmPreset` indicating the algorithm that
        must be used to accumulate the dot product.

    preferred_element_type: Optional. This parameter controls the data type
      output by the dot product. By default, the output element type of this
      operation will match the ``lhs`` and ``rhs`` input element types under
      the usual type promotion rules. Setting ``preferred_element_type`` to a
      specific ``dtype`` will mean that the operation returns that element type.
      When ``precision`` is not a :class:`~jax.lax.DotAlgorithm` or
      :class:`~jax.lax.DotAlgorithmPreset`, ``preferred_element_type`` provides
      a hint to the compiler to accumulate the dot product using this data type.

  Returns:
    An array whose first dimensions are the (shared) batch dimensions, followed
    by the ``lhs`` non-contracting/non-batch dimensions, and finally the ``rhs``
    non-contracting/non-batch dimensions.
  """
  if out_sharding is not None and not config.sharding_in_types.value:
    raise NotImplementedError("out_sharding only works when sharding_in_types "
                              "config is True.")
  if out_sharding is not None and not isinstance(out_sharding, NamedSharding):
    raise NotImplementedError(
        '`out_sharding` argument of `dot_general` only supports NamedSharding '
        'instances. Please file a bug if this is not enough for your use case.')
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  preferred_element_type = (
      None if preferred_element_type is None else
      dtypes.canonicalize_dtype(np.dtype(preferred_element_type)))
  return dot_general_p.bind(lhs, rhs,
                            dimension_numbers=(cdims, bdims),
                            precision=canonicalize_precision(precision),
                            preferred_element_type=preferred_element_type,
                            out_sharding=out_sharding)


def ragged_dot(
    lhs: Array,
    rhs: Array,
    group_sizes: Array,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Array | None = None,
    ) -> Array:
  """Ragged matrix multiplication.

  Args:
    lhs: (m, k) shaped array.
    rhs: (g, k, n) shaped array.
    group_sizes: (g,) shaped array with integer element type, where g denotes   number of groups. The ith element indicates the size of ith group.
    precision: Optional. Consistent with precision argument for :func:`jax.lax.dot`.
    preferred_element_type: Optional. Consistent with precision argument for :func:`jax.lax.dot`.
    group_offset: Optional. (1,) shaped array that indicates the group in group_sizes to start computing from. If not specified, defaults to [0].

  Results:
    (m, n) shaped array with preferred_element_type element type.
  """
  return ragged_dot_p.bind(lhs, rhs, group_sizes,
                            precision=canonicalize_precision(precision),
                            preferred_element_type=preferred_element_type,
                            group_offset=group_offset)


def broadcast(operand: ArrayLike, sizes: Sequence[int], sharding=None) -> Array:
  """Broadcasts an array, adding new leading dimensions

  Args:
    operand: an array
    sizes: a sequence of integers, giving the sizes of new leading dimensions
      to add to the front of the array.

  Returns:
    An array containing the result.

  See Also:
    jax.lax.broadcast_in_dim : add new dimensions at any location in the array shape.
  """
  if len(sizes) == 0 and sharding is None:
    return asarray(operand)
  dims = tuple(range(len(sizes), len(sizes) + np.ndim(operand)))
  return broadcast_in_dim(operand, tuple(sizes) + np.shape(operand), dims,
                          sharding=sharding)

def broadcast_in_dim(operand: ArrayLike, shape: Shape,
                     broadcast_dimensions: Sequence[int], sharding=None) -> Array:
  """Wraps XLA's `BroadcastInDim
  <https://www.tensorflow.org/xla/operation_semantics#broadcastindim>`_
  operator.

  Args:
    operand: an array
    shape: the shape of the target array
    broadcast_dimensions: to which dimension in the target shape each dimension
      of the operand shape corresponds to.  That is, dimension i of the operand
      becomes dimension broadcast_dimensions[i] of the result.

  Returns:
    An array containing the result.

  See Also:
    jax.lax.broadcast : simpler interface to add new leading dimensions.
  """
  if not config.sharding_in_types.value and sharding is not None:
    raise NotImplementedError("sharding argument to broadcast_in_dim is only "
                              "allowed when sharding_in_types config is on.")
  sharding = canonicalize_sharding(sharding)
  if (np.ndim(operand) == len(shape) and not len(broadcast_dimensions) and
      isinstance(operand, Array) and sharding is None):
    return operand
  if config.dynamic_shapes.value:
    # We must gate this behavior under a flag because otherwise the errors
    # raised are different (and have worse source provenance information).
    dyn_shape, static_shape = _extract_tracers_dyn_shape(shape)
  else:
    dyn_shape, static_shape = [], shape  # type: ignore
  return broadcast_in_dim_p.bind(
      operand, *dyn_shape, shape=tuple(static_shape),
      broadcast_dimensions=tuple(broadcast_dimensions),
      sharding=sharding)

def broadcast_to_rank(x: ArrayLike, rank: int) -> Array:
  """Adds leading dimensions of ``1`` to give ``x`` rank ``rank``."""
  ndim = np.ndim(x)
  if ndim == rank:
    return asarray(x)
  return broadcast(x, (1,) * (rank - ndim))

def reshape(operand: ArrayLike, new_sizes: Shape,
            dimensions: Sequence[int] | None = None,
            sharding: NamedSharding | P | None = None) -> Array:
  """Wraps XLA's `Reshape
  <https://www.tensorflow.org/xla/operation_semantics#reshape>`_
  operator.

  For inserting/removing dimensions of size 1, prefer using ``lax.squeeze`` /
  ``lax.expand_dims``. These preserve information about axis identity that may
  be useful for advanced transformation rules.

  Args:
    operand: array to be reshaped.
    new_sizes: sequence of integers specifying the resulting shape. The size
      of the final array must match the size of the input.
    dimensions: optional sequence of integers specifying the permutation order of
      the input shape. If specified, the length must match ``operand.shape``.

  Returns:
    out: reshaped array.

  Examples:
    Simple reshaping from one to two dimensions:

    >>> x = jnp.arange(6)
    >>> y = reshape(x, (2, 3))
    >>> y
    Array([[0, 1, 2],
                 [3, 4, 5]], dtype=int32)

    Reshaping back to one dimension:

    >>> reshape(y, (6,))
    Array([0, 1, 2, 3, 4, 5], dtype=int32)

    Reshaping to one dimension with permutation of dimensions:

    >>> reshape(y, (6,), (1, 0))
    Array([0, 3, 1, 4, 2, 5], dtype=int32)
  """
  new_sizes = canonicalize_shape(new_sizes)  # TODO
  new_sizes = tuple(new_sizes)
  same_shape = core.definitely_equal_shape(np.shape(operand), new_sizes)
  if dimensions is None:
    same_dims = True
    dims = None
  else:
    dims = api_util._ensure_index_tuple(dimensions)
    same_dims = tuple(dims) == tuple(range(np.ndim(operand)))
  if np.shape(operand) and same_shape and same_dims and isinstance(operand, Array):
    return operand
  else:
    dyn_shape, static_new_sizes = _extract_tracers_dyn_shape(new_sizes)
    sharding = canonicalize_sharding(sharding)
    return reshape_p.bind(
      operand, *dyn_shape, new_sizes=tuple(static_new_sizes),
      dimensions=None if dims is None or same_dims else dims,
      sharding=sharding)

def pad(operand: ArrayLike, padding_value: ArrayLike,
        padding_config: Sequence[tuple[int, int, int]]) -> Array:
  """Applies low, high, and/or interior padding to an array.

  Wraps XLA's `Pad
  <https://www.tensorflow.org/xla/operation_semantics#pad>`_
  operator.

  Args:
    operand: an array to be padded.
    padding_value: the value to be inserted as padding. Must have the same dtype
      as ``operand``.
    padding_config: a sequence of ``(low, high, interior)`` tuples of integers,
      giving the amount of low, high, and interior (dilation) padding to insert
      in each dimension.

  Returns:
    The ``operand`` array with padding value ``padding_value`` inserted in each
    dimension according to the ``padding_config``.

  Examples:
    >>> from jax import lax
    >>> import jax.numpy as jnp

    Pad a 1-dimensional array with zeros, We'll specify two zeros in front and
    three at the end:

    >>> x = jnp.array([1, 2, 3, 4])
    >>> lax.pad(x, 0, [(2, 3, 0)])
    Array([0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=int32)

    Pad a 1-dimensional array with *interior* zeros; i.e. insert a single zero
    between each value:

    >>> lax.pad(x, 0, [(0, 0, 1)])
    Array([1, 0, 2, 0, 3, 0, 4], dtype=int32)

    Pad a 2-dimensional array with the value ``-1`` at front and end, with a pad
    size of 2 in each dimension:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> lax.pad(x, -1, [(2, 2, 0), (2, 2, 0)])
    Array([[-1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1],
           [-1, -1,  1,  2,  3, -1, -1],
           [-1, -1,  4,  5,  6, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1]], dtype=int32)
  """
  return pad_p.bind(operand, padding_value, padding_config=tuple(padding_config))

def rev(operand: ArrayLike, dimensions: Sequence[int]) -> Array:
  """Wraps XLA's `Rev
  <https://www.tensorflow.org/xla/operation_semantics#rev_reverse>`_
  operator.
  """
  return rev_p.bind(operand, dimensions=tuple(dimensions))

def select(pred: ArrayLike, on_true: ArrayLike, on_false: ArrayLike) -> Array:
  """Selects between two branches based on a boolean predicate.

  Wraps XLA's `Select
  <https://www.tensorflow.org/xla/operation_semantics#select>`_
  operator.

  In general :func:`~jax.lax.select` leads to evaluation of both branches, although
  the compiler may elide computations if possible. For a similar function that
  usually evaluates only a single branch, see :func:`~jax.lax.cond`.

  Args:
    pred: boolean array
    on_true: array containing entries to return where ``pred`` is True. Must have
      the same shape as ``pred``, and the same shape and dtype as ``on_false``.
    on_false: array containing entries to return where ``pred`` is False. Must have
      the same shape as ``pred``, and the same shape and dtype as ``on_true``.

  Returns:
    result: array with same shape and dtype as ``on_true`` and ``on_false``.
  """
  # Caution! The select_n_p primitive has the *opposite* order of arguments to
  # select(). This is because it implements `select_n`.
  return select_n_p.bind(pred, on_false, on_true)

def select_n(which: ArrayLike, *cases: ArrayLike) -> Array:
  """Selects array values from multiple cases.

  Generalizes XLA's `Select
  <https://www.tensorflow.org/xla/operation_semantics#select>`_
  operator. Unlike XLA's version, the operator is variadic and can select
  from many cases using an integer `pred`.

  Args:
    which: determines which case should be returned. Must be an array containing
      either a boolean or integer values. May either be a scalar or have
      shape matching ``cases``. For each array element, the value of ``which``
      determines which of ``cases`` is taken. ``which`` must be in the range
      ``[0 .. len(cases))``; for values outside that range the behavior is
      implementation-defined.
    *cases: a non-empty list of array cases. All must have equal dtypes and
      equal shapes.
  Returns:
    An array with shape and dtype equal to the cases, whose values are chosen
    according to ``which``.
  """
  if len(cases) == 0:
    raise ValueError("select_n() must have at least one case")
  return select_n_p.bind(which, *cases)


def transpose(operand: ArrayLike,
              permutation: Sequence[int] | np.ndarray) -> Array:
  """Wraps XLA's `Transpose
  <https://www.tensorflow.org/xla/operation_semantics#transpose>`_
  operator.
  """
  permutation = tuple(operator.index(d) for d in permutation)
  if permutation == tuple(range(np.ndim(operand))) and isinstance(operand, Array):
    return operand
  else:
    return transpose_p.bind(operand, permutation=permutation)

def argmin(operand: ArrayLike, axis: int,
           index_dtype: DTypeLike) -> Array:
  """Computes the index of the minimum element along ``axis``."""
  return argmin_p.bind(operand, axes=(axis,),
                       index_dtype=dtypes.canonicalize_dtype(index_dtype))

def argmax(operand: ArrayLike, axis: int,
           index_dtype: DTypeLike) -> Array:
  """Computes the index of the maximum element along ``axis``."""
  return argmax_p.bind(operand, axes=(axis,),
                       index_dtype=dtypes.canonicalize_dtype(index_dtype))

def reduce(operands: Any,
           init_values: Any,
           computation: Callable[[Any, Any], Any],
           dimensions: Sequence[int]) -> Any:
  """Wraps XLA's `Reduce
  <https://www.tensorflow.org/xla/operation_semantics#reduce>`_
  operator.

  ``init_values`` and ``computation`` together must form a `monoid
  <https://en.wikipedia.org/wiki/Monoid>`_
  for correctness. That is ``init_values`` must be an identity of
  ``computation``, and ``computation`` must be associative. XLA may exploit both
  of these properties during code generation; if either is violated the result
  is undefined.
  """
  flat_operands, operand_tree = tree_util.tree_flatten(operands)
  flat_init_values, init_value_tree = tree_util.tree_flatten(init_values)
  if operand_tree != init_value_tree:
    raise ValueError('Operands must have the same tree structure as init_values:'
                     f' {operand_tree} vs. {init_value_tree}')
  if len(flat_operands) != len(flat_init_values):
    raise ValueError('Must have same total number of operands as init_values: '
                     f' {len(flat_operands)} vs. {len(flat_init_values)}')
  monoid_reducer = _get_monoid_reducer(computation, flat_init_values)
  if monoid_reducer:
    # monoid reducers bypass the weak_type_rule, so we set it explicitly.
    weak_type = dtypes.is_weakly_typed(*flat_operands) and dtypes.is_weakly_typed(*flat_init_values)
    return _convert_element_type(monoid_reducer(*flat_operands, dimensions),
                                 weak_type=weak_type)
  else:
    flat_init_avals = safe_map(core.get_aval, flat_init_values)
    closed_jaxpr, out_tree = _variadic_reduction_jaxpr(
        computation, tuple(flat_init_avals), init_value_tree)
    out = reduce_p.bind(*flat_operands, *flat_init_values, computation=computation,
                        jaxpr=closed_jaxpr, dimensions=tuple(dimensions))
    return tree_util.tree_unflatten(out_tree, out)

@cache()
def _reduction_jaxpr(computation, aval):
  @lu.wrap_init
  def comp(x, y):
    result = computation(x, y)
    if not (isinstance(result, core.Tracer) or core.valid_jaxtype(result)):
      raise ValueError(
          f"Invalid return type from reduction function: {type(result)}\n"
          f"Reduction functions should only return an array.\n"
          f"Full return value: {result}")
    return (result,)
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(comp, (aval, aval))
  if any(isinstance(c, core.Tracer) for c in consts):
    raise NotImplementedError(
        "Reduction computations can't close over Tracers. Please open an issue "
        "at https://github.com/jax-ml/jax.")
  return jaxpr, tuple(consts)

@cache()
def _variadic_reduction_jaxpr(computation, flat_avals, aval_tree):
  avals = tree_util.tree_unflatten(aval_tree, flat_avals)
  flat_in_avals, in_tree = tree_util.tree_flatten((avals, avals))
  comp = lu.wrap_init(computation)
  flat_comp, out_tree = api_util.flatten_fun_nokwargs(comp, in_tree)
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_comp, tuple(flat_in_avals))
  if any(isinstance(c, core.Tracer) for c in consts):
    raise NotImplementedError(
        "Reduction computations can't close over Tracers. Please open an issue "
        "at https://github.com/jax-ml/jax.")
  return core.ClosedJaxpr(jaxpr, consts), out_tree()

def _get_monoid_reducer(monoid_op: Callable,
                        xs: Sequence[Array]) -> Callable | None:
  if len(xs) != 1:
    return None
  x, = xs
  aval = core.get_aval(x)
  dtype = _dtype(x)
  if core.is_concrete(x) and aval.shape == ():
    val = core.to_concrete_value(x)
    # allow bitwise reductions for boolean and integer types
    _is_intlike = dtype == np.bool_ or dtypes.issubdtype(dtype, np.integer)
    if monoid_op is add:
      return _reduce_sum if np.equal(val, 0) else None
    elif monoid_op is mul:
      return _reduce_prod if np.equal(val, 1) else None
    elif monoid_op is bitwise_or and _is_intlike:
      return _reduce_or if np.equal(val, _get_bitwise_or_identity(dtype)) else None
    elif monoid_op is bitwise_and and _is_intlike:
      return _reduce_and if np.equal(val, _get_bitwise_and_identity(dtype)) else None
    elif monoid_op is bitwise_xor and _is_intlike:
      return _reduce_xor if np.equal(val, _get_bitwise_or_identity(dtype)) else None
    elif monoid_op is max:
      return _reduce_max if np.equal(val, _get_max_identity(dtype)) else None
    elif monoid_op is min:
      return _reduce_min if np.equal(val, _get_min_identity(dtype)) else None
  return None

def _get_bitwise_and_identity(dtype: DTypeLike) -> np.ndarray:
  return np.array(-1).astype(dtype)

def _get_bitwise_or_identity(dtype: DTypeLike) -> np.ndarray:
  return np.array(0, dtype)

def _get_sum_identity(dtype: DTypeLike) -> np.ndarray:
  return np.array(0, dtype)

def _get_prod_identity(dtype: DTypeLike) -> np.ndarray:
  return np.array(1, dtype)

def _get_max_identity(dtype: DTypeLike) -> np.ndarray:
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(-np.inf if dtypes.supports_inf(dtype) else dtypes.finfo(dtype).min,
                    dtype=dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).min, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(False, np.bool_)
  else:
    raise ValueError(f"Unsupported dtype for max: {dtype}")

def _get_min_identity(dtype: DTypeLike) -> np.ndarray:
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(np.inf if dtypes.supports_inf(dtype) else dtypes.finfo(dtype).max,
                    dtype=dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).max, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(True, np.bool_)
  else:
    raise ValueError(f"Unsupported dtype for min: {dtype}")

def _reduce_sum(operand: ArrayLike, axes: Sequence[int]) -> Array:
  return reduce_sum_p.bind(operand, axes=tuple(axes))

def _reduce_prod(operand: ArrayLike, axes: Sequence[int]) -> Array:
  return reduce_prod_p.bind(operand, axes=tuple(axes))

def _reduce_max(operand: ArrayLike, axes: Sequence[int]) -> Array:
  return reduce_max_p.bind(operand, axes=tuple(axes))

def _reduce_min(operand: ArrayLike, axes: Sequence[int]) -> Array:
  return reduce_min_p.bind(operand, axes=tuple(axes))

def _reduce_or(operand: ArrayLike, axes: Sequence[int]) -> Array:
  return reduce_or_p.bind(operand, axes=tuple(axes))

def _reduce_and(operand: ArrayLike, axes: Sequence[int]) -> Array:
  return reduce_and_p.bind(operand, axes=tuple(axes))

def _reduce_xor(operand: ArrayLike, axes: Sequence[int]) -> Array:
  return reduce_xor_p.bind(operand, axes=tuple(axes))

@overload
def sort(operand: Array, dimension: int = -1,
         is_stable: bool = True, num_keys: int = 1) -> Array: ...

@overload
def sort(operand: Sequence[Array], dimension: int = -1,
         is_stable: bool = True, num_keys: int = 1) -> tuple[Array, ...]: ...

def sort(operand: Array | Sequence[Array], dimension: int = -1,
         is_stable: bool = True, num_keys: int = 1) -> Array | tuple[Array, ...]:
  """Wraps XLA's `Sort
  <https://www.tensorflow.org/xla/operation_semantics#sort>`_ operator.

  For floating point inputs, -0.0 and 0.0 are treated as equivalent, and NaN values
  are sorted to the end of the array. For complex inputs, the sort order is
  lexicographic over the real and imaginary parts, with the real part primary.

  Args:
    operand : Array or sequence of arrays
    dimension : integer dimension along which to sort. Default: -1.
    is_stable : boolean specifying whether to use a stable sort. Default: True.
    num_keys : number of operands to treat as sort keys. Default: 1.
      For num_keys > 1, the sort order will be determined lexicographically using
      the first `num_keys` arrays, with the first key being primary.
      The remaining operands will be returned with the same permutation.

  Returns:
    operand : sorted version of the input or inputs.
  """
  if isinstance(operand, Sequence):
    if len(operand) == 0:
      raise TypeError("Sort requires at least one operand")
    if not (1 <= num_keys <= len(operand)):
      raise ValueError(f"{num_keys=} must be between 1 and {len(operand)=}")
    dimension = canonicalize_axis(dimension, len(operand[0].shape))
    return tuple(sort_p.bind(*operand, dimension=dimension,
                             is_stable=is_stable,
                             num_keys=num_keys))
  else:
    if num_keys != 1:
      raise ValueError(f"{num_keys=} must equal 1 for a single operand.")
    dimension = canonicalize_axis(dimension, len(operand.shape))
    return sort_p.bind(operand, dimension=dimension, is_stable=is_stable, num_keys=1)[0]

def sort_key_val(keys: Array, values: ArrayLike, dimension: int = -1,
                 is_stable: bool = True) -> tuple[Array, Array]:
  """Sorts ``keys`` along ``dimension`` and applies the same permutation to ``values``."""
  dimension = canonicalize_axis(dimension, len(keys.shape))
  k, v = sort_p.bind(keys, values, dimension=dimension, is_stable=is_stable, num_keys=1)
  return k, v

def top_k(operand: ArrayLike, k: int) -> tuple[Array, Array]:
  """Returns top ``k`` values and their indices along the last axis of ``operand``.

  Args:
    operand: N-dimensional array of non-complex type.
    k: integer specifying the number of top entries.

  Returns:
    A tuple ``(values, indices)`` where

    - ``values`` is an array containing the top k values along the last axis.
    - ``indices`` is an array containing the indices corresponding to values.

  See also:
    - :func:`jax.lax.approx_max_k`
    - :func:`jax.lax.approx_min_k`

  Examples:
    Find the largest three values, and their indices, within an array:

    >>> x = jnp.array([9., 3., 6., 4., 10.])
    >>> values, indices = jax.lax.top_k(x, 3)
    >>> values
    Array([10.,  9.,  6.], dtype=float32)
    >>> indices
    Array([4, 0, 2], dtype=int32)
  """
  if core.is_constant_dim(k):
    k = int(k)
  if k < 0:
    raise ValueError(f"k argument to top_k must be nonnegative, got {k}")
  return top_k_p.bind(operand, k=k)

def tie_in(x: Any, y: T) -> T:
  """Deprecated. Ignores ``x`` and returns ``y``."""
  return y

def full(shape: Shape, fill_value: ArrayLike, dtype: DTypeLike | None = None, *,
         sharding: Sharding | None = None) -> Array:
  """Returns an array of `shape` filled with `fill_value`.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    fill_value: the value to fill the new array with.
    dtype: the type of the output array, or `None`. If not `None`, `fill_value`
      will be cast to `dtype`.
    sharding: an optional sharding specification for the resulting array,
      note, sharding will currently be ignored in jitted mode, this might change
      in the future.
  """
  shape = canonicalize_shape(shape)
  if np.shape(fill_value):
    msg = "full must be called with scalar fill_value, got fill_value.shape {}."
    raise TypeError(msg.format(np.shape(fill_value)))
  if dtypes.issubdtype(dtype, dtypes.extended):
    return dtype._rules.full(shape, fill_value, dtype)  # type: ignore[union-attr]
  weak_type = dtype is None and dtypes.is_weakly_typed(fill_value)
  dtype = dtypes.canonicalize_dtype(dtype or _dtype(fill_value))
  fill_value = _convert_element_type(fill_value, dtype, weak_type)
  if (sharding is not None and not isinstance(sharding, PmapSharding) and
      isinstance(fill_value, array.ArrayImpl)):
    broadcast_shape = sharding.shard_shape(shape)
    shard = broadcast(fill_value, broadcast_shape)
    return array.make_array_from_callback(shape, sharding, lambda _: shard)

  if config.sharding_in_types.value and sharding is not None:
    return broadcast(fill_value, shape, sharding=sharding)
  else:
    return broadcast(fill_value, shape)

def zeros_like_shaped_array(aval: ShapedArray) -> Array:
  assert isinstance(aval, ShapedArray)
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    scalar_zero = aval.dtype._rules.zero(aval.dtype)
  elif aval.dtype == dtypes.float0:
    scalar_zero = np.zeros((), dtype=aval.dtype)
  else:
    scalar_zero = _convert_element_type(0, aval.dtype, aval.weak_type)
  if config.sharding_in_types.value:
    return broadcast(scalar_zero, aval.shape, sharding=aval.sharding)
  return broadcast(scalar_zero, aval.shape)

ad_util.aval_zeros_likers[ShapedArray] = zeros_like_shaped_array

def zeros_like_abstract_ref(aval: state.AbstractRef) -> core.MutableArray:
  val = ad_util.zeros_like_aval(aval.inner_aval)
  return core.mutable_array(val)

# TODO(dougalm): this is nonsense but it's here because in places like
# custom_vjp we assume that all arguments have tangent spaces. We could have
# a distinct NotATangentType value instead.
ad_util.aval_zeros_likers[state.AbstractRef] = zeros_like_abstract_ref  # type: ignore

def iota(dtype: DTypeLike, size: int) -> Array:
  """Wraps XLA's `Iota
  <https://www.tensorflow.org/xla/operation_semantics#iota>`_
  operator.
  """
  return broadcasted_iota(dtype, (size,), 0)

def broadcasted_iota(dtype: DTypeLike, shape: Shape, dimension: int,
                     sharding=None) -> Array:
  """Convenience wrapper around ``iota``."""
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = canonicalize_shape(shape)
  dynamic_shape = [d for d in shape if isinstance(d, core.Tracer)]
  static_shape = [None if isinstance(d, core.Tracer) else d for d in shape]
  dimension = core.concrete_or_error(
      int, dimension, "dimension argument of lax.broadcasted_iota")
  if not config.sharding_in_types.value and sharding is not None:
    raise NotImplementedError('sharding support for broadcasted_iota is not '
                              'implemented outside of sharding_in_types mode.')
  sharding = canonicalize_sharding(sharding)
  return iota_p.bind(*dynamic_shape, dtype=dtype, shape=tuple(static_shape),
                     dimension=dimension, sharding=sharding)

def _eye(dtype: DTypeLike, shape: Shape, offset: DimSize = 0) -> Array:
  """Like numpy.eye, create a 2D array with ones on a diagonal."""
  offset = _clip_int_to_valid_range(offset, np.int32,
                                    "argument `offset` of jax.numpy.eye")
  dtype = dtypes.canonicalize_dtype(dtype)
  bool_eye = eq(add(broadcasted_iota(np.int32, shape, 0), np.int32(offset)),
                broadcasted_iota(np.int32, shape, 1))
  return convert_element_type_p.bind(bool_eye, new_dtype=dtype, weak_type=False,
                                     sharding=None)

def _delta(dtype: DTypeLike, shape: Shape, axes: Sequence[int]) -> Array:
  """This utility function exists for creating Kronecker delta arrays."""
  axes = map(int, axes)
  dtype = dtypes.canonicalize_dtype(dtype)
  base_shape = tuple(np.take(shape, axes))
  iotas = [broadcasted_iota(np.uint32, base_shape, i)
           for i in range(len(base_shape))]
  eyes = [eq(i1, i2) for i1, i2 in zip(iotas[:-1], iotas[1:])]
  result = convert_element_type_p.bind(
      _reduce(operator.and_, eyes), new_dtype=dtype, weak_type=False,
      sharding=None)
  return broadcast_in_dim(result, shape, axes)

def _tri(dtype: DTypeLike, shape: Shape, offset: DimSize) -> Array:
  """Like numpy.tri, create a 2D array with ones below a diagonal."""
  offset = _clip_int_to_valid_range(offset, np.int32,
                                    "argument `offset` of jax.numpy.tri")
  dtype = dtypes.canonicalize_dtype(dtype)
  bool_tri = ge(add(broadcasted_iota(np.int32, shape, 0),
                    asarray(core.dimension_as_value(offset)).astype(np.int32)),
                broadcasted_iota(np.int32, shape, 1))
  return convert_element_type_p.bind(bool_tri, new_dtype=dtype, weak_type=False,
                                     sharding=None)

def stop_gradient(x: T) -> T:
  """Stops gradient computation.

  Operationally ``stop_gradient`` is the identity function, that is, it returns
  argument `x` unchanged. However, ``stop_gradient`` prevents the flow of
  gradients during forward or reverse-mode automatic differentiation. If there
  are multiple nested gradient computations, ``stop_gradient`` stops gradients
  for all of them. For some discussion of where this is useful, refer to
  :ref:`stopping-gradients`.

  Args:
    x: array or pytree of arrays

  Returns:
    input value is returned unchanged, but within autodiff will be treated as
    a constant.

  Examples:
    Consider a simple function that returns the square of the input value:

    >>> def f1(x):
    ...   return x ** 2
    >>> x = jnp.float32(3.0)
    >>> f1(x)
    Array(9.0, dtype=float32)
    >>> jax.grad(f1)(x)
    Array(6.0, dtype=float32)

    The same function with ``stop_gradient`` around ``x`` will be equivalent
    under normal evaluation, but return a zero gradient because ``x`` is
    effectively treated as a constant:

    >>> def f2(x):
    ...   return jax.lax.stop_gradient(x) ** 2
    >>> f2(x)
    Array(9.0, dtype=float32)
    >>> jax.grad(f2)(x)
    Array(0.0, dtype=float32)

    This is used in a number of places within the JAX codebase; for example
    :func:`jax.nn.softmax` internally normalizes the input by its maximum
    value, and this maximum value is wrapped in ``stop_gradient`` for
    efficiency. Refer to :ref:`stopping-gradients` for more discussion of
    the applicability of ``stop_gradient``.
  """
  def stop(x):
    # only bind primitive on inexact dtypes, to avoid some staging
    if dtypes.issubdtype(core.get_aval(x).dtype, dtypes.extended):
      return x
    elif (dtypes.issubdtype(_dtype(x), np.floating) or
        dtypes.issubdtype(_dtype(x), np.complexfloating)):
      # break abstractions to support legacy leaked tracer use cases
      if isinstance(x, ad.JVPTracer):
        return stop(x.primal)
      return ad_util.stop_gradient_p.bind(x)
    else:
      return x
  return tree_map(stop, x)

def reduce_precision(operand: float | ArrayLike,
                     exponent_bits: int,
                     mantissa_bits: int) -> Array:
  """Wraps XLA's `ReducePrecision
  <https://www.tensorflow.org/xla/operation_semantics#reduceprecision>`_
  operator.
  """
  exponent_bits = core.concrete_or_error(
    operator.index, exponent_bits, "exponent_bits argument of lax.reduce_precision")
  mantissa_bits = core.concrete_or_error(
    operator.index, mantissa_bits, "mantissa_bits argument of lax.reduce_precision")
  return reduce_precision_p.bind(operand, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)

def squeeze(array: ArrayLike, dimensions: Sequence[int]) -> Array:
  """Squeeze any number of size 1 dimensions from an array."""
  ndim = np.ndim(array)
  dimensions = tuple(sorted(canonicalize_axis(i, ndim) for i in dimensions))
  if not dimensions and isinstance(array, Array):
    return array
  return squeeze_p.bind(array, dimensions=dimensions)

def expand_dims(array: ArrayLike, dimensions: Sequence[int]) -> Array:
  """Insert any number of size 1 dimensions into an array."""
  if len(set(dimensions)) != len(dimensions):
    raise ValueError(f'repeated axis in lax.expand_dims: {dimensions}')
  ndim_out = np.ndim(array) + len(dimensions)
  dims = [canonicalize_axis(i, ndim_out) for i in dimensions]
  if len(set(dims)) != len(dims):  # check again after canonicalizing
    raise ValueError(f'repeated axis in lax.expand_dims: {dims}')
  dims_set = frozenset(dims)
  result_shape = list(np.shape(array))
  for i in sorted(dims_set):
    result_shape.insert(i, 1)
  broadcast_dims = [i for i in range(ndim_out) if i not in dims_set]
  return broadcast_in_dim(array, result_shape, broadcast_dims)


### convenience wrappers around traceables

def full_like(x: ArrayLike | DuckTypedArray,
              fill_value: ArrayLike, dtype: DTypeLike | None = None,
              shape: Shape | None = None, sharding: Sharding | None = None) -> Array:
  """Create a full array like np.full based on the example array `x`.

  Args:
    x: example array-like, used for shape and dtype information.
    fill_value: a scalar value to fill the entries of the output array.
    dtype: optional, a dtype parameter for the output ndarray.
    shape: optional, a shape parameter for the output ndarray.
    sharding: an optional sharding specification for the resulting array.
      If not specified, the output will have the same sharding as the input,
      with a few exceptions/limitations in particular:
      1. Sharding is not available during tracing, thus this will rely on jit.
      2. If x is weakly typed or uncommitted, will use default sharding.
      3. Shape is not None and is different from x.shape, default will be used.

  Returns:
    An ndarray with the same shape as `x` with its entries set equal to
    `fill_value`, similar to the output of np.full.
  """
  fill_shape = np.shape(x) if shape is None else canonicalize_shape(shape)  # type: ignore[arg-type]
  weak_type = dtype is None and dtypes.is_weakly_typed(x)
  dtype = dtype or _dtype(x)
  if dtypes.issubdtype(dtype, dtypes.extended):
    return dtype._rules.full(fill_shape, fill_value, dtype)  # type: ignore[union-attr]

  if (config.sharding_in_types.value and sharding is None and
      isinstance(x, Array)):
    sharding = x.sharding
  else:
    # If `x` has a sharding but no `_committed` attribute
    # (in case of ShapeDtypeStruct), default it to True.
    use_x_sharding = (
        sharding is None
        # Tracer have special logic in handling sharding and even
        # though hasattr(x, 'sharding') returns False, it is very slow.
        # This bypasses the check.
        and not isinstance(x, core.Tracer)
        and hasattr(x, 'sharding')
        and getattr(x, '_committed', True)
        and not weak_type
        and fill_shape == np.shape(x)  # type: ignore[arg-type]
    )
    if use_x_sharding:
      # TODO(yashkatariya): Use shard_alike in tracing_mode once it is supported.
      sharding = x.sharding  # type: ignore
  val = full(fill_shape, _convert_element_type(fill_value, dtype, weak_type),
             sharding=sharding)
  return val


def collapse(operand: Array, start_dimension: int,
             stop_dimension: int | None = None) -> Array:
  """Collapses dimensions of an array into a single dimension.

  For example, if ``operand`` is an array with shape ``[2, 3, 4]``,
  ``collapse(operand, 0, 2).shape == [6, 4]``. The elements of the collapsed
  dimension are laid out major-to-minor, i.e., with the lowest-numbered
  dimension as the slowest varying dimension.

  Args:
    operand: an input array.
    start_dimension: the start of the dimensions to collapse (inclusive).
    stop_dimension: the end of the dimensions to collapse (exclusive). Pass None
      to collapse all the dimensions after start.

  Returns:
    An array where dimensions ``[start_dimension, stop_dimension)`` have been
    collapsed (raveled) into a single dimension.
  """
  lo, hi, _ = slice(start_dimension, stop_dimension).indices(len(operand.shape))
  if hi < lo:
    raise ValueError(f"Invalid dimension range passed to collapse: {operand.shape}"
                     f"[{start_dimension}:{stop_dimension}]")
  size = math.prod(operand.shape[lo:hi])
  new_shape = operand.shape[:lo] + (size,) + operand.shape[hi:]
  return reshape(operand, new_shape)


def batch_matmul(lhs: Array, rhs: Array,
                 precision: PrecisionLike = None) -> Array:
  """Batch matrix multiplication."""
  if _min(lhs.ndim, rhs.ndim) < 2:
    raise ValueError('Arguments to batch_matmul must be at least 2D, got {}, {}'
                     .format(lhs.ndim, rhs.ndim))
  if lhs.ndim != rhs.ndim:
    raise ValueError('Arguments to batch_matmul must have same ndim, got {}, {}'
                     .format(lhs.ndim, rhs.ndim))
  lhs_contract = (lhs.ndim - 1,)
  rhs_contract = (rhs.ndim - 2,)
  batch = tuple(range(lhs.ndim - 2))
  return dot_general(lhs, rhs, ((lhs_contract, rhs_contract), (batch, batch)),
                     precision=precision)


# These functions also exist in the XLA client library, but we treat them
# as non-primitive to maintain a smaller set of autodiff primitives.

def square(x: ArrayLike) -> Array:
  r"""Elementwise square: :math:`x^2`."""
  return square_p.bind(x)

def reciprocal(x: ArrayLike) -> Array:
  r"""Elementwise reciprocal: :math:`1 \over x`."""
  return integer_pow(x, -1)

def tan(x: ArrayLike) -> Array:
  r"""Elementwise tangent: :math:`\mathrm{tan}(x)`."""
  return tan_p.bind(x)

def asin(x: ArrayLike) -> Array:
  r"""Elementwise arc sine: :math:`\mathrm{asin}(x)`."""
  return asin_p.bind(x)

def acos(x: ArrayLike) -> Array:
  r"""Elementwise arc cosine: :math:`\mathrm{acos}(x)`."""
  return acos_p.bind(x)

def atan(x: ArrayLike) -> Array:
  r"""Elementwise arc tangent: :math:`\mathrm{atan}(x)`."""
  return atan_p.bind(x)

def sinh(x: ArrayLike) -> Array:
  r"""Elementwise hyperbolic sine: :math:`\mathrm{sinh}(x)`."""
  return sinh_p.bind(x)

def cosh(x: ArrayLike) -> Array:
  r"""Elementwise hyperbolic cosine: :math:`\mathrm{cosh}(x)`."""
  return cosh_p.bind(x)

def asinh(x: ArrayLike) -> Array:
  r"""Elementwise inverse hyperbolic sine: :math:`\mathrm{asinh}(x)`."""
  return asinh_p.bind(x)

def acosh(x: ArrayLike) -> Array:
  r"""Elementwise inverse hyperbolic cosine: :math:`\mathrm{acosh}(x)`."""
  return acosh_p.bind(x)

def atanh(x: ArrayLike) -> Array:
  r"""Elementwise inverse hyperbolic tangent: :math:`\mathrm{atanh}(x)`."""
  return atanh_p.bind(x)


# Add some methods to ShapedArray that rely on lax primitives

ShapedArray.broadcast = core.aval_method(broadcast)
ShapedArray.transpose = core.aval_method(transpose)  # clobbered by lax_numpy
ShapedArray.reshape = core.aval_method(reshape)      # clobbered by lax_numpy

def _iter(tracer):
  if tracer.ndim == 0:
    raise TypeError("iteration over a 0-d array")  # same as numpy error
  else:
    n = int(tracer.shape[0])
    if any(isinstance(d, core.Tracer) for d in tracer.shape):
      return (slicing.dynamic_index_in_dim(tracer, i, keepdims=False)
              for i in range(n))
    else:
      return (slicing.index_in_dim(tracer, i, keepdims=False) for i in range(n))
ShapedArray._iter = staticmethod(_iter)
core.DShapedArray._iter = staticmethod(_iter)

def zeros_like_array(x: ArrayLike) -> Array:
  return full_like(x, 0)


def _add_arrays(x, y):
  if (isinstance(a := core.get_aval(x), ShapedArray) and
      dtypes.issubdtype(a.dtype, dtypes.extended)):
    return dtype._rules.add(dtype, x, y)  # pytype: disable=attribute-error
  return add(x, y)

for t in itertools.chain(
    dtypes.python_scalar_dtypes.keys(), array_types, [array.ArrayImpl]):
  ad_util.raw_jaxval_adders[t] = _add_arrays


### primitives


_fixed_dtype = \
    lambda dtype: lambda *args, **kwargs: dtypes.canonicalize_dtype(dtype)
_complex_basetype = lambda dtype: np.abs(np.zeros((), dtype)).dtype

_strip_weak_type = lambda *args, **_: False


def unop_dtype_rule(result_dtype, accepted_dtypes, name, aval, **kwargs):
  if aval.dtype == dtypes.float0:
    raise TypeError(
        f"Called {name} with a float0 array. "
        "float0s do not support any operations by design, because they "
        "are not compatible with non-trivial vector spaces. No implicit dtype "
        "conversion is done. You can use np.zeros_like(arr, dtype=np.float) "
        "to cast a float0 array to a regular zeros array. \n"
        "If you didn't expect to get a float0 you might have accidentally "
        "taken a gradient with respect to an integer argument.")
  if not any(dtypes.issubdtype(aval.dtype, t) for t in accepted_dtypes):
    msg = '{} does not accept dtype {}. Accepted dtypes are subtypes of {}.'
    typename = dtype_to_string(aval.dtype)
    accepted_typenames = (t.__name__ for t in accepted_dtypes)
    raise TypeError(msg.format(name, typename, ', '.join(accepted_typenames)))
  return result_dtype(aval.dtype)


def unop(result_dtype, accepted_dtypes, name):
  dtype_rule = partial(unop_dtype_rule, result_dtype, accepted_dtypes, name)
  prim = standard_primitive(_attrgetter('shape'), dtype_rule, name,
                            sharding_rule=_attrgetter('sharding'))
  batching.defvectorized(prim)
  pe.def_trivial_padding(prim)
  return prim

standard_unop = partial(unop, _identity)

_attrgetter = lambda name: lambda x, **kwargs: getattr(x, name)


def naryop_dtype_rule(result_dtype, accepted_dtypes, name, *avals,
                      require_same=True, allow_extended_dtype=False, **kwargs):
  del kwargs
  assert len(avals) == len(accepted_dtypes), (avals, accepted_dtypes)
  for i, aval in enumerate(avals):
    if allow_extended_dtype and isinstance(aval.dtype, dtypes.ExtendedDType):
      continue
    types = accepted_dtypes[i]
    if not any(dtypes.issubdtype(aval.dtype, t) for t in types):
      if aval.dtype == dtypes.float0:
        raise TypeError(
            f"Called {name} with a float0 at position {i}. "
            "float0s do not support any operations by design, because they "
            "are not compatible with non-trivial vector spaces. No implicit dtype "
            "conversion is done. You can use np.zeros_like(arr, dtype=np.float) "
            "to cast a float0 array to a regular zeros array. \n"
            "If you didn't expect to get a float0 you might have accidentally "
            "taken a gradient with respect to an integer argument.")
      else:
        msg = ('{} does not accept dtype {} at position {}. '
               'Accepted dtypes at position {} are subtypes of {}.')
        typename = dtype_to_string(aval.dtype)
        typenames = ', '.join(t.__name__ for t in types)
        raise TypeError(msg.format(name, typename, i, i, typenames))
  if require_same: check_same_dtypes(name, *avals)
  return result_dtype(*avals)


def broadcasting_shape_rule(name, *avals):
  shapes = [aval.shape for aval in avals if aval.shape]
  if not shapes:
    return ()
  return _try_broadcast_shapes(*shapes, name=name)


def broadcasting_sharding_rule(name, *avals):
  mesh = None
  for a in avals:
    if a.sharding is not None:
      if mesh is not None and mesh != a.sharding.mesh:
        raise ValueError(
            f'Mesh for all inputs should be equal. Got one mesh: {mesh} and'
            f' another mesh: {a.sharding.mesh}')
      mesh = a.sharding.mesh
  assert mesh is not None

  shapes = [aval.shape for aval in avals if aval.shape]
  if not shapes:
    return NamedSharding(mesh, P())
  if len({len(shape) for shape in shapes}) != 1:
    msg = '{}: arrays must have same number of dimensions, got {}.'
    raise TypeError(msg.format(name, ', '.join(map(str, map(tuple, shapes)))))

  specs = [a.sharding.spec for a in avals if a.shape]

  result_specs = [None] * len(shapes[0])
  for i, (ss, ds) in enumerate(zip(zip(*specs), zip(*shapes))):
    if all(s == ss[0] for s in ss[1:]):
      # if all dimension shardings are same, the resulting dimension sharding is
      # the same.
      result_specs[i] = ss[0]
    else:
      non_trivial_s = [s for s, d in zip(ss, ds)
                       if not (core.definitely_equal(d, 1) and s is None)]
      if not non_trivial_s:
        result_specs[i] = None
      elif all(non_trivial_s[0] == s for s in non_trivial_s[1:]):
        result_specs[i] = non_trivial_s[0]
      else:
        for s in ss:
          if result_specs[i] is None and s is not None:
            result_specs[i] = s
          elif (result_specs[i] is not None and s is not None and
                result_specs[i] != s):
            raise TypeError(
                f'{name} got incompatible shardings for broadcasting: '
                f'{", ".join(map(str, map(tuple, specs)))}.')
  return NamedSharding(mesh, P(*result_specs))


def naryop(result_dtype, accepted_dtypes, name, allow_extended_dtype=False,
           require_same_dtypes=True):
  dtype_rule = partial(naryop_dtype_rule, result_dtype, accepted_dtypes, name,
                       allow_extended_dtype=allow_extended_dtype,
                       require_same=require_same_dtypes)
  shape_rule = partial(broadcasting_shape_rule, name)
  sharding_rule = partial(broadcasting_sharding_rule, name)
  prim = standard_primitive(shape_rule, dtype_rule, name,
                            sharding_rule=sharding_rule)
  batching.defbroadcasting(prim)
  pe.def_trivial_padding(prim)
  return prim
standard_naryop = partial(naryop, _input_dtype)


# Like autograd.numpy.numpy_vjps.unbroadcast, this utility handles transposition
# involving linear primitives with implicit broadcasting.
def _unbroadcast(aval, x):
  if not isinstance(aval, (core.DShapedArray, ShapedArray)):
    raise TypeError("transpose with implicit broadcasting of unshaped values")
  x_shape = np.shape(x)
  if core.definitely_equal_shape(aval.shape, x_shape):
    return x
  assert not aval.shape or len(x_shape) == len(aval.shape)
  if not aval.shape:
    return _reduce_sum(x, list(range(len(x_shape))))
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, aval.shape)) if not core.definitely_equal(a, b)]
    if config.enable_checks.value: assert all(aval.shape[i] == 1 for i in dims)
    return reshape(_reduce_sum(x, dims), aval.shape)

def _maybe_broadcast(target_shape, x):
  x_shape = np.shape(x)
  if core.definitely_equal_shape(x_shape, target_shape):
    return x
  elif not x_shape:
    return broadcast_in_dim(x, target_shape, ())
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, target_shape))
            if core.definitely_equal(a, b)]
    squeeze_shape = [x_shape[i] for i in dims]
    return broadcast_in_dim(reshape(x, squeeze_shape), target_shape, dims)

def broadcast_hlo(
    aval_out: core.ShapedArray, avals: Sequence[core.ShapedArray],
    args: Sequence[ir.Value]) -> Sequence[ir.Value]:
  """Broadcasts HLO values with broadcast-compatible shapes to the same shape.
  """
  out = []
  for aval, arg in zip(avals, args):
    if aval.shape != aval_out.shape:
      assert len(aval.shape) <= len(aval_out.shape), (aval, aval_out)
      dims = mlir.dense_int_array(
          list(range(len(aval_out.shape) - len(aval.shape), len(aval_out.shape))))
      if any(isinstance(d, ir.Value) for d in aval_out.shape):
        arg = hlo.dynamic_broadcast_in_dim(
            mlir.aval_to_ir_type(aval_out), arg,
            mlir.shape_tensor(aval_out.shape), dims)
      else:
        arg = hlo.broadcast_in_dim(
            mlir.aval_to_ir_type(aval.update(shape=aval_out.shape)), arg,
            dims)
    out.append(arg)
  return out

def multi_sharding_in_dim(ctx, ops, in_avals, out_aval):
  out = []
  for op, in_aval in zip(ops, in_avals):
    if in_aval.sharding == out_aval.sharding or in_aval.sharding is None:
      out.append(op)
    else:
      out.append(mlir.lower_sharding_under_shit(ctx, op, out_aval))
  return out


def _nary_lower_hlo(op: Callable, ctx,
                    *args: ir.Value, **params) -> Sequence[ir.Value]:
  """Lowers an elementwise operator to its MLIR equivalent.
  """
  del params
  avals_in, (aval_out,) = ctx.avals_in, ctx.avals_out
  args = mlir.multi_broadcast_in_dim(ctx, args, avals_in, aval_out.shape)  # type: ignore
  if config.sharding_in_types.value:
    args = multi_sharding_in_dim(ctx, args, avals_in, aval_out)

  out = op(*args)
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  else:
    return [out]


_float = {np.floating}
_complex = {np.complexfloating}
_complex_elem_types = {np.float32, np.float64}
_int = {np.integer}
_bool = {np.bool_}
_signedint = {np.signedinteger}

_num = _int | _float | _complex
_any = _int | _float | _complex | _bool
_bool_or_int = _int | _bool
_ordered = _int | _float | _bool

neg_p = standard_unop(_num, 'neg')
ad.deflinear2(neg_p, lambda t, operand: [neg(t)])
mlir.register_lowering(neg_p, partial(_nary_lower_hlo, hlo.negate))

sign_p = standard_unop(_num, 'sign')
ad.defjvp_zero(sign_p)

def _sign_lower_hlo(ctx, x):
  x_aval, = ctx.avals_in
  if dtypes.issubdtype(x_aval.dtype, np.unsignedinteger):
    return [hlo.select(
        mlir.compare_hlo(x, mlir.full_like_aval(ctx, 0, x_aval), 'EQ',
                         'UNSIGNED'),
        mlir.full_like_aval(ctx, 0, x_aval),
        mlir.full_like_aval(ctx, 1, x_aval))]
  return [hlo.sign(x)]

mlir.register_lowering(sign_p, _sign_lower_hlo)

nextafter_p = standard_naryop([_float, _float], 'nextafter')
mlir.register_lowering(nextafter_p, partial(_nary_lower_hlo, chlo.next_after))

floor_p = standard_unop(_float, 'floor')
ad.defjvp_zero(floor_p)
mlir.register_lowering(floor_p, partial(_nary_lower_hlo, hlo.floor))

ceil_p = standard_unop(_float, 'ceil')
ad.defjvp_zero(ceil_p)
mlir.register_lowering(ceil_p, partial(_nary_lower_hlo, hlo.ceil))

round_p = standard_unop(_float, 'round')
ad.defjvp_zero(round_p)

def _round_lower(ctx, x, *, rounding_method):
  if rounding_method is RoundingMethod.AWAY_FROM_ZERO:
    return [hlo.round_nearest_afz(x)]
  else:
    assert rounding_method is RoundingMethod.TO_NEAREST_EVEN
    return [hlo.round_nearest_even(x)]
mlir.register_lowering(round_p, _round_lower)

is_finite_p = unop(_fixed_dtype(np.bool_), _float, 'is_finite')
ad.defjvp_zero(is_finite_p)
mlir.register_lowering(is_finite_p, partial(_nary_lower_hlo, hlo.is_finite))

exp_p = standard_unop(_float | _complex, 'exp')
ad.defjvp2(exp_p, lambda g, ans, x: mul(g, ans))
mlir.register_lowering(exp_p, partial(_nary_lower_hlo, hlo.exponential))
batching.ragged_prop_rules[exp_p] = batching.ragged_mask_elementwise_rule

exp2_p = standard_unop(_float | _complex, 'exp2')
ad.defjvp2(exp2_p, lambda g, ans, x: mul(log(_const(x, 2)), mul(g, ans)))
def _exp2_lower(ctx, x):
  x_aval, = ctx.avals_in
  log2 = mlir.ir_constant(np.array(np.log(2), x_aval.dtype))
  log2 = mlir.broadcast_in_dim(ctx, log2, x_aval, broadcast_dimensions=())
  return [hlo.exponential(hlo.multiply(log2, x))]
mlir.register_lowering(exp2_p, _exp2_lower)

log_p = standard_unop(_float | _complex, 'log')
ad.defjvp(log_p, lambda g, x: div(g, x))
mlir.register_lowering(log_p, partial(_nary_lower_hlo, hlo.log))

expm1_p = standard_unop(_float | _complex, 'expm1')
ad.defjvp2(expm1_p, lambda g, ans, x: mul(g, add(ans, _one(ans))))
mlir.register_lowering(expm1_p,
                       partial(_nary_lower_hlo, hlo.exponential_minus_one))

log1p_p = standard_unop(_float | _complex, 'log1p')
ad.defjvp(log1p_p, lambda g, x: div(g, add(x, _one(x))))
mlir.register_lowering(log1p_p, partial(_nary_lower_hlo, hlo.log_plus_one))

tanh_p = standard_unop(_float | _complex, 'tanh')
ad.defjvp2(tanh_p, lambda g, ans, x: mul(add(g, mul(g, ans)),
                                         sub(_one(x), ans)))
mlir.register_lowering(tanh_p, partial(_nary_lower_hlo, hlo.tanh))

logistic_p = standard_unop(_float | _complex, 'logistic')
ad.defjvp2(logistic_p, lambda g, ans, x: mul(g, mul(ans, sub(_one(ans), ans))))
# TODO(phawkins): switch to LogisticOp lowering; debug numerical problems.
# mlir.register_lowering(logistic_p, partial(_nary_lower_hlo, hlo.logistic))

def logistic_impl(x):
  one = _const(x, 1)
  return div(one, add(one, exp(neg(x))))

mlir.register_lowering(logistic_p,
                       mlir.lower_fun(logistic_impl, multiple_results=False))

def _sin_complex(x):
  # use expm1 instead of exp to avoid cancellation when abs(x) is small
  # relies on the quality of real-valued expm1, sin, cos
  # sin(x) = complex(sin(real(x)) * cosh(imag(x)), cos(real(x)) * sinh(imag(x)))
  # 2 * sinh(x) = exp(x) - 1 - (exp(-x) - 1) = expm1(x) - expm1(-x)
  # 2 * cosh(x) = exp(x) - 1 + (exp(-x) - 1) + 2 = expm1(x) + expm1(-x) + 2
  a, b = real(x), imag(x)
  a_is_zero = eq(a, _const(a, 0))
  sn, cs = sin(a), cos(a)
  e1m, e2m = expm1(b), expm1(-b)
  snh, csh = (e1m - e2m) / 2, (e1m + e2m + 2) / 2
  re, im = sn * csh, cs * snh
  # avoid nan value when real(x) is zero and abs(x) is so large that abs(expm1(x)) is inf
  return select(a_is_zero, complex(_const(a, 0), im), complex(re, im))

def _sin_lowering(ctx, x):
  if dtypes.issubdtype(ctx.avals_in[0].dtype, np.complexfloating):
    sine = mlir.lower_fun(_sin_complex, multiple_results=False)
    return sine(ctx, x)
  return _nary_lower_hlo(hlo.sine, ctx, x)

def _sin_p_lin(nzs, x):
  nz, = nzs
  cos_x = cos(x) # TODO: allow this to happen in the linearized computation (need to fix backward_pass)
  return (sin_p.bind(x), nz, cos_x, lambda cos_x_, t: mul(t, cos_x_))

sin_p = standard_unop(_float | _complex, 'sin')
ad.defjvp(sin_p, lambda g, x: mul(g, cos(x)))
ad.primitive_linearizations[sin_p] = _sin_p_lin
mlir.register_lowering(sin_p, _sin_lowering)
batching.ragged_prop_rules[sin_p] = batching.ragged_mask_elementwise_rule

def _cos_complex(x):
  # cos(x) = complex(cos(real(x)) * cosh(imag(x)), -sin(real(x)) * sinh(imag(x)))
  # see also _sin_complex
  a, b = real(x), imag(x)
  a_is_zero = eq(a, _const(a, 0))
  sn, cs = sin(a), cos(a)
  e1m, e2m = expm1(b), expm1(-b)
  snh, csh = (e1m - e2m) / 2, (e1m + e2m + 2) / 2
  re, im = cs * csh, -sn * snh
  return select(a_is_zero, complex(re, _const(a, 0)), complex(re, im))

def _cos_lowering(ctx, x):
  if dtypes.issubdtype(ctx.avals_in[0].dtype, np.complexfloating):
    cosine = mlir.lower_fun(_cos_complex, multiple_results=False)
    return cosine(ctx, x)
  return _nary_lower_hlo(hlo.cosine, ctx, x)

cos_p = standard_unop(_float | _complex, 'cos')
ad.defjvp(cos_p, lambda g, x: neg(mul(g, sin(x))))
mlir.register_lowering(cos_p, _cos_lowering)

tan_p = standard_unop(_float | _complex, 'tan')
ad.defjvp2(tan_p, lambda g, ans, x: mul(g, _const(x, 1) + square(ans)))
mlir.register_lowering(tan_p, partial(_nary_lower_hlo, hlo.tan))

asin_p = standard_unop(_float | _complex, 'asin')
ad.defjvp(asin_p, lambda g, x: mul(g, rsqrt(_const(x, 1) - square(x))))
mlir.register_lowering(asin_p, partial(_nary_lower_hlo, chlo.asin))

acos_p = standard_unop(_float | _complex, 'acos')
ad.defjvp(acos_p, lambda g, x: mul(g, -rsqrt(_const(x, 1) - square(x))))
mlir.register_lowering(acos_p, partial(_nary_lower_hlo, chlo.acos))

def atan_impl(x):
  return atan2(x, _const(x, 1))

atan_p = standard_unop(_float | _complex, 'atan')
ad.defjvp(atan_p, lambda g, x: div(g, _const(x, 1) + square(x)))
mlir.register_lowering(atan_p, partial(_nary_lower_hlo, chlo.atan))

atan2_p = standard_naryop([_float | _complex, _float | _complex], 'atan2')
ad.defjvp(atan2_p,
          lambda g, x, y: g * (y / (square(x) + square(y))),
          lambda g, x, y: g * -x / (square(x) + square(y)))
mlir.register_lowering(atan2_p, partial(_nary_lower_hlo, hlo.atan2))

sinh_p = standard_unop(_float | _complex, 'sinh')
ad.defjvp(sinh_p, lambda g, x: mul(g, cosh(x)))
mlir.register_lowering(sinh_p, partial(_nary_lower_hlo, chlo.sinh))

cosh_p = standard_unop(_float | _complex, 'cosh')
ad.defjvp(cosh_p, lambda g, x: mul(g, sinh(x)))
mlir.register_lowering(cosh_p, partial(_nary_lower_hlo, chlo.cosh))

asinh_p = standard_unop(_float | _complex, 'asinh')
ad.defjvp(asinh_p, lambda g, x: mul(g, rsqrt(square(x) + _one(x))))
mlir.register_lowering(asinh_p, partial(_nary_lower_hlo, chlo.asinh))

acosh_p = standard_unop(_float | _complex, 'acosh')
ad.defjvp(acosh_p,
          lambda g, x: mul(g, rsqrt((x - _one(x)) * (x + _one(x)))))
mlir.register_lowering(acosh_p, partial(_nary_lower_hlo, chlo.acosh))

atanh_p = standard_unop(_float | _complex, 'atanh')
ad.defjvp(atanh_p,
          lambda g, x: mul(reciprocal(_one(x) + x), div(g, (_one(x) - x))))
mlir.register_lowering(atanh_p, partial(_nary_lower_hlo, chlo.atanh))

real_p = unop(_complex_basetype, _complex, 'real')
ad.deflinear2(real_p, lambda t, _: [complex(t, np.zeros((), _dtype(t)))])
mlir.register_lowering(real_p, partial(_nary_lower_hlo, hlo.real))

imag_p = unop(_complex_basetype, _complex, 'imag')
ad.deflinear2(imag_p, lambda t, _: [complex(np.zeros((), _dtype(t)), neg(t))])
mlir.register_lowering(imag_p, partial(_nary_lower_hlo, hlo.imag))


def _complex_transpose_rule(t, x, y):
  assert ad.is_undefined_primal(x) or ad.is_undefined_primal(y)
  if ad.is_undefined_primal(x) and ad.is_undefined_primal(y):
    if type(t) is ad_util.Zero:
      return [ad_util.Zero(x.aval), ad_util.Zero(y.aval)]
    else:
      return [_unbroadcast(x.aval, real(t)), _unbroadcast(y.aval, imag(neg(t)))]
  elif ad.is_undefined_primal(x):
    if type(t) is ad_util.Zero:
      return [ad_util.Zero(x.aval), None]
    else:
      return [_unbroadcast(x.aval, real(t)), None]
  else:
    if type(t) is ad_util.Zero:
      return [None, ad_util.Zero(y.aval)]
    else:
      return [None, _unbroadcast(y.aval, imag(neg(t)))]

_complex_dtype = lambda dtype, *args: (np.zeros((), dtype) + np.zeros((), np.complex64)).dtype
complex_p = naryop(_complex_dtype, [_complex_elem_types, _complex_elem_types],
                  'complex')
ad.deflinear2(complex_p, _complex_transpose_rule)
mlir.register_lowering(complex_p, partial(_nary_lower_hlo, hlo.complex))

conj_p = unop(_complex_dtype, _complex_elem_types | _complex, 'conj')

def _conj_impl(x, **kw):
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return complex(real(x), -imag(x))
  else:
    return complex(x, _zeros(x))

mlir.register_lowering(conj_p,
                       mlir.lower_fun(_conj_impl, multiple_results=False))


def _conj_transpose_rule(t, x, *, input_dtype):
  assert ad.is_undefined_primal(x)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(x.aval)]
  elif dtypes.issubdtype(input_dtype, np.complexfloating):
    return [conj(t)]
  else:
    return [real(t)]

ad.primitive_jvps[conj_p] = partial(ad.linear_jvp, conj_p)
ad.primitive_transposes[conj_p] = _conj_transpose_rule

abs_p = unop(_complex_basetype, _signedint | _float | _complex, 'abs')
mlir.register_lowering(abs_p, partial(_nary_lower_hlo, hlo.abs))

def _abs_jvp_rule(g, ans, x):
  if _iscomplex(x):
    return _maybe_real(mul(g, div(_maybe_conj(x),
           _replace_zero(convert_element_type(ans, _dtype(x))))))
  else:
    return select(ge(x, _zero(x)), g, neg(g))
ad.defjvp2(abs_p, _abs_jvp_rule)
_maybe_conj = lambda x: conj(x) if _iscomplex(x) else x
_maybe_real = lambda x: real(x) if _iscomplex(x) else x

sqrt_p = standard_unop(_float | _complex, 'sqrt')
ad.defjvp2(sqrt_p, lambda g, ans, x: mul(g, div(_const(x, 0.5), ans)))
mlir.register_lowering(sqrt_p, partial(_nary_lower_hlo, hlo.sqrt))

rsqrt_p = standard_unop(_float | _complex, 'rsqrt')
ad.defjvp2(rsqrt_p,
           lambda g, ans, x:
           mul(g, mul(_const(x, -0.5), div(ans, x))))
mlir.register_lowering(rsqrt_p, partial(_nary_lower_hlo, hlo.rsqrt))

cbrt_p = standard_unop(_float, 'cbrt')
ad.defjvp2(cbrt_p,
           lambda g, ans, x: mul(g, mul(_const(x, 1/3), integer_pow(ans, -2))))
mlir.register_lowering(cbrt_p, partial(_nary_lower_hlo, hlo.cbrt))

square_p = standard_unop(_int | _float | _complex, 'square')

def _square_complex(x):
  a, b = real(x), imag(x)
  # zero square(x).real is handled explicitly for abs(a)==abs(b) cases
  # where for finite a, 2 * a is non-finite:
  zero_re = is_finite(a) & (eq(a, b) | eq(a, -b))
  # equivalent to a**2 - b**2 but avoids overflow errors for large a
  # and large b cases:
  re = (a - b) * (a + b)
  im = a * b * 2
  return select(zero_re, complex(_const(a, 0), im), complex(re, im))

def _square_lower_hlo(ctx, x):
  if dtypes.issubdtype(ctx.avals_in[0].dtype, np.complexfloating):
    return mlir.lower_fun(_square_complex, multiple_results=False)(ctx, x)
  return [hlo.multiply(x, x)]

ad.defjvp2(square_p, lambda g, ans, x: mul(g, mul(_const(x, 2), x)))
mlir.register_lowering(square_p, _square_lower_hlo)  # TODO(pearu): use chlo.square

def _pow_dtype_rule(x, y):
  if (dtypes.issubdtype(x.dtype, np.inexact) and
      dtypes.issubdtype(y.dtype, np.integer)):
    return x.dtype
  if x.dtype == y.dtype:
    return x.dtype
  raise TypeError("the first argument to pow must have an inexact dtype (float "
                  "or complex), and the second argument must have an inexact or"
                  " integer dtype, and two inexact dtypes must match, but got "
                  f"{x.dtype} and {y.dtype} respectively.")
pow_p = naryop(_pow_dtype_rule, [_float | _complex, _int | _float | _complex],
               'pow', require_same_dtypes=False)

def _pow_jvp_lhs(g, ans, x, y):
  y_dtype = dtypes.dtype(y)
  result_dtype = dtypes.result_type(x, y)
  if result_dtype == bool:
    result_dtype = 'int32'
  x = convert_element_type(x, result_dtype)
  y = convert_element_type(y, result_dtype)
  if dtypes.issubdtype(y_dtype, np.integer):
    if x.shape != y.shape:
      shape = broadcast_shapes(x.shape, y.shape)
      x = _maybe_broadcast(shape, x)
      y = _maybe_broadcast(shape, y)
    jac = select(eq(y, _const(y, 0)), _zeros(y),
                 mul(_replace_zero(y), pow(x, sub(y, _ones(y)))))
  else:
    jac = mul(y, pow(x, sub(y, _ones(y))))
  return mul(g, jac)

def _pow_jvp_rhs(g, ans, x, y):
  y_dtype = dtypes.dtype(y)
  assert dtypes.issubdtype(y_dtype, np.inexact)
  return convert_element_type(mul(g, mul(log(_replace_zero(x)), ans)), y_dtype)
ad.defjvp2(pow_p, _pow_jvp_lhs, _pow_jvp_rhs)

def _pow_lower(ctx, x, y):
  x_aval, y_aval = ctx.avals_in
  if x_aval.dtype != y_aval.dtype:
    out_aval, = ctx.avals_out
    y_aval = y_aval.update(dtype=out_aval.dtype)
    y = hlo.convert(mlir.aval_to_ir_type(y_aval), y)
    ctx = ctx.replace(avals_in=[x_aval, y_aval])
  return _nary_lower_hlo(hlo.power, ctx, x, y)
mlir.register_lowering(pow_p, _pow_lower)

def _integer_pow_dtype_rule(x, *, y):
  dtype = unop_dtype_rule(_identity, _int | _float | _complex, 'integer_pow', x)
  if y < 0 and dtypes.issubdtype(dtype, np.integer):
    raise TypeError("Integers cannot be raised to negative powers, got "
                    f"integer_pow({x}, {y})")
  return dtype

def _integer_pow_jvp(g, x, *, y):
  return _zeros(g) if y == 0 else mul(g, mul(_const(x, y), integer_pow(x, y - 1)))

integer_pow_p = standard_primitive(
  _attrgetter('shape'), _integer_pow_dtype_rule, 'integer_pow',
  sharding_rule=_attrgetter('sharding'))
batching.defvectorized(integer_pow_p)
ad.defjvp(integer_pow_p, _integer_pow_jvp)
pe.def_trivial_padding(integer_pow_p)

def _integer_pow(x, *, y):
  # This should be kept in sync with the jax2tf translation rule.
  if y == 0:
    return full_like(x, 1)
  is_reciprocal = y < 0
  if is_reciprocal:
    y = -y
  acc = None
  while y > 0:
    if y & 1:
      acc = x if acc is None else mul(acc, x)
    y >>= 1
    if y > 0:
      # We don't call square because it calls integer_pow.
      x = mul(x, x)
  return div(full_like(acc, 1), acc) if is_reciprocal else acc


def _integer_pow_lowering(ctx, x, *, y):
  # These cases are subsumed by the general case, but it's faster to emit these
  # common cases directly.
  if y == 1:
    out = x
  elif y == 2:
    out = hlo.multiply(x, x)
  elif y == 3:
    out = hlo.multiply(hlo.multiply(x, x), x)
  elif y == -1:
    out = hlo.divide(mlir.full_like_aval(ctx, 1, ctx.avals_in[0]), x)
  else:
    lowering = mlir.lower_fun(_integer_pow, multiple_results=False)
    if builtins.abs(y) >= 3:
      lowering = mlir.cache_lowering(lowering)
    out, = lowering(ctx, x, y=y)
  if config.sharding_in_types.value:
    aval_out, = ctx.avals_out
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

mlir.register_lowering(integer_pow_p, _integer_pow_lowering)

_replace_zero = lambda x: select(eq(x, _const(x, 0)), _ones(x), x)

not_p = standard_unop(_bool_or_int, 'not')
ad.defjvp_zero(not_p)
mlir.register_lowering(not_p, partial(_nary_lower_hlo, hlo.not_))

and_p = standard_naryop([_bool_or_int, _bool_or_int], 'and')
ad.defjvp_zero(and_p)
mlir.register_lowering(and_p, partial(_nary_lower_hlo, hlo.and_))

or_p = standard_naryop([_bool_or_int, _bool_or_int], 'or')
ad.defjvp_zero(or_p)
mlir.register_lowering(or_p, partial(_nary_lower_hlo, hlo.or_))

xor_p = standard_naryop([_bool_or_int, _bool_or_int], 'xor')
ad.defjvp_zero(xor_p)
mlir.register_lowering(xor_p, partial(_nary_lower_hlo, hlo.xor))

population_count_p = standard_unop(_int, 'population_count')
mlir.register_lowering(population_count_p, partial(_nary_lower_hlo, hlo.popcnt))

clz_p = standard_unop(_int, 'clz')
mlir.register_lowering(clz_p, partial(_nary_lower_hlo, hlo.count_leading_zeros))

def _add_jvp(primals, tangents):
  x, y = primals
  xdot, ydot = tangents
  primal_out = add(x, y)
  if type(xdot) is type(ydot) is ad_util.Zero:
    return primal_out, ad_util.Zero.from_primal_value(primal_out)
  if type(xdot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, ydot)
  elif type(ydot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, xdot)
  else:
    return primal_out, add(xdot, ydot)

def _add_transpose(t, x, y):
  # Morally the following assertion is true, but because we instantiate zeros in
  # some places (e.g. in custom_jvp) it may not always hold. For example, see
  # api_test.py's CustomJVPTest.test_jaxpr_zeros.
  # assert ad.is_undefined_primal(x) and ad.is_undefined_primal(y)
  x_aval = x.aval if ad.is_undefined_primal(x) else core.get_aval(x)
  y_aval = y.aval if ad.is_undefined_primal(y) else core.get_aval(y)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(x_aval), ad_util.Zero(y_aval)]
  else:
    return [_unbroadcast(x_aval, t), _unbroadcast(y_aval, t)]

# TODO(slebedev): Why does mypy fail to infer the type here?
add_p: Primitive = standard_naryop([_num, _num], 'add')
ad.primitive_jvps[add_p] = _add_jvp
ad.primitive_transposes[add_p] = _add_transpose
mlir.register_lowering(add_p, partial(_nary_lower_hlo, hlo.add))
batching.ragged_prop_rules[add_p] = batching.ragged_mask_elementwise_rule

def _sub_jvp(primals, tangents):
  x, y = primals
  xdot, ydot = tangents
  primal_out = sub(x, y)
  if type(xdot) is type(ydot) is ad_util.Zero:
    return primal_out, ad_util.Zero.from_primal_value(primal_out)
  if type(xdot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, neg(ydot))
  elif type(ydot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, xdot)
  else:
    return primal_out, sub(xdot, ydot)

def _sub_transpose(t, x, y):
  # Morally the following assertion is true, but see the comment in add_p's
  # transpose rule.
  # assert ad.is_undefined_primal(x) and ad.is_undefined_primal(y)
  x_aval = x.aval if ad.is_undefined_primal(x) else core.get_aval(x)
  y_aval = y.aval if ad.is_undefined_primal(y) else core.get_aval(y)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(x_aval), ad_util.Zero(y_aval)]
  else:
    return [_unbroadcast(x_aval, t), _unbroadcast(y_aval, neg(t))]

sub_p = standard_naryop([_num, _num], 'sub')
ad.primitive_jvps[sub_p] = _sub_jvp
ad.primitive_transposes[sub_p] = _sub_transpose
mlir.register_lowering(sub_p, partial(_nary_lower_hlo, hlo.subtract))
batching.ragged_prop_rules[sub_p] = batching.ragged_mask_elementwise_rule


def _mul_transpose(ct, x, y):
  assert ad.is_undefined_primal(x) ^ ad.is_undefined_primal(y)
  if ad.is_undefined_primal(x):
    if type(ct) is ad_util.Zero:
      return [ad_util.Zero(x.aval), None]
    else:
      return [_unbroadcast(x.aval, mul(ct, y)), None]
  else:
    if type(ct) is ad_util.Zero:
      return [None, ad_util.Zero(y.aval)]
    else:
      return [None, _unbroadcast(y.aval, mul(x, ct))]

mul_p = standard_naryop([_num, _num], 'mul')
ad.defjvp(mul_p,
          lambda xdot, x, y: mul(xdot, y),
          lambda ydot, x, y: mul(x, ydot))
ad.primitive_transposes[mul_p] = _mul_transpose
mlir.register_lowering(mul_p, partial(_nary_lower_hlo, hlo.multiply))
batching.ragged_prop_rules[mul_p] = batching.ragged_mask_elementwise_rule

def _div_transpose_rule(cotangent, x, y):
  assert ad.is_undefined_primal(x) and not ad.is_undefined_primal(y)
  if type(cotangent) is ad_util.Zero:
    return [ad_util.Zero(x.aval), None]
  else:
    return [_unbroadcast(x.aval, div(cotangent, y)), None]
div_p = standard_naryop([_num, _num], 'div')
ad.defjvp(div_p,
          lambda g, x, y: div(g, y),
          lambda g, x, y: mul(mul(neg(g), x), integer_pow(y, -2)))
ad.primitive_transposes[div_p] = _div_transpose_rule
mlir.register_lowering(div_p, partial(_nary_lower_hlo, hlo.divide))
batching.ragged_prop_rules[div_p] = batching.ragged_mask_elementwise_rule

rem_p = standard_naryop([_int | _float, _int | _float], 'rem')
ad.defjvp(
    rem_p,
    lambda g, x, y: _maybe_broadcast(broadcast_shapes(np.shape(x), np.shape(y)), g),
    lambda g, x, y: mul(neg(g), mul(sign(div(x, y)), floor(abs(div(x, y))))))
mlir.register_lowering(rem_p, partial(_nary_lower_hlo, hlo.remainder))

def _minmax_complex_lowering(x, y, *, lax_cmp_pick_x):
  result_shape = broadcast_shapes(np.shape(x), np.shape(y))
  x = _maybe_broadcast(result_shape, x)
  y = _maybe_broadcast(result_shape, y)
  rx = real(x)
  ry = real(y)
  pick_x = select(eq(rx, ry), lax_cmp_pick_x(imag(x), imag(y)),
                  lax_cmp_pick_x(rx, ry))
  return select(pick_x, x, y)

max_p: core.Primitive = standard_naryop([_any, _any], 'max')
ad.defjvp2(max_p,
           lambda g, ans, x, y: mul(g, _balanced_eq(x, ans, y)),
           lambda g, ans, x, y: mul(g, _balanced_eq(y, ans, x)))
mlir.register_lowering(max_p, partial(_nary_lower_hlo, mlir.max_hlo))
batching.ragged_prop_rules[max_p] = batching.ragged_mask_elementwise_rule

min_p: core.Primitive = standard_naryop([_any, _any], 'min')
ad.defjvp2(min_p,
           lambda g, ans, x, y: mul(g, _balanced_eq(x, ans, y)),
           lambda g, ans, x, y: mul(g, _balanced_eq(y, ans, x)))
mlir.register_lowering(min_p, partial(_nary_lower_hlo, mlir.min_hlo))
batching.ragged_prop_rules[min_p] = batching.ragged_mask_elementwise_rule

shift_left_p = standard_naryop([_int, _int], 'shift_left')
ad.defjvp_zero(shift_left_p)
mlir.register_lowering(shift_left_p, partial(_nary_lower_hlo, hlo.shift_left))

shift_right_arithmetic_p = standard_naryop([_int, _int], 'shift_right_arithmetic')
ad.defjvp_zero(shift_right_arithmetic_p)
mlir.register_lowering(shift_right_arithmetic_p,
                       partial(_nary_lower_hlo, hlo.shift_right_arithmetic))

shift_right_logical_p = standard_naryop([_int, _int], 'shift_right_logical')
ad.defjvp_zero(shift_right_logical_p)
mlir.register_lowering(shift_right_logical_p,
                       partial(_nary_lower_hlo, hlo.shift_right_logical))

def _opaque_comparison_hlo(direction, reduction_op, identity, ctx,
                           avals_in, aval_out, x, y):
  aval_x, aval_y = avals_in
  base_aval_x = core.physical_aval(aval_x)
  base_aval_y = core.physical_aval(aval_y)
  base_aval_out = core.ShapedArray(base_aval_x.shape, aval_out.dtype)
  reduce_axes = tuple(range(aval_out.ndim, base_aval_out.ndim))
  res, = mlir.delegate_lowering(
      ctx, partial(_compare_lower_hlo, direction, False),
      x, y, avals_in=[base_aval_x, base_aval_y], avals_out=[base_aval_out])
  return mlir.delegate_lowering(
      ctx, partial(_unary_reduce_lower, reduction_op, identity,
                   axes=reduce_axes),
      res, avals_in=[base_aval_out], avals_out=[aval_out])

_opaque_eq_hlo = partial(
    _opaque_comparison_hlo, 'EQ', hlo.AndOp, _get_bitwise_and_identity)
_opaque_ne_hlo = partial(
    _opaque_comparison_hlo, 'NE', hlo.OrOp, _get_bitwise_or_identity)

def _compare_lower_hlo_opaque(direction: str, ctx, avals_in, aval_out, x, y):
  broadcast_avals_in = tuple(
      core.ShapedArray(aval_out.shape, aval.dtype) for aval in avals_in)
  if direction == 'EQ':
    return _opaque_eq_hlo(ctx, broadcast_avals_in, aval_out, x, y)
  elif direction == 'NE':
    return _opaque_ne_hlo(ctx, broadcast_avals_in, aval_out, x, y)
  else:
    raise NotImplementedError(
        f"HLO comparison {direction} for extended dtype {avals_in[0].dtype}")


def _compare_lower_hlo(direction: str, total_order: bool, ctx, x, y):
  avals_in, (aval_out,) = ctx.avals_in, ctx.avals_out
  x_dtype = avals_in[0].dtype
  x, y = mlir.multi_broadcast_in_dim(ctx, (x, y), avals_in, aval_out.shape)
  if dtypes.issubdtype(x_dtype, dtypes.extended):
    assert not total_order
    return _compare_lower_hlo_opaque(direction, ctx, avals_in, aval_out, x, y)
  if dtypes.issubdtype(x_dtype, np.inexact):
    compare_type = "TOTALORDER" if total_order else "FLOAT"
  elif dtypes.issubdtype(x_dtype, np.signedinteger):
    compare_type = "SIGNED"
  else:
    compare_type = "UNSIGNED"
  return [mlir.compare_hlo(x, y, direction, compare_type)]

eq_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'eq', allow_extended_dtype=True)
ad.defjvp_zero(eq_p)
mlir.register_lowering(eq_p, partial(_compare_lower_hlo, "EQ", False))
batching.ragged_prop_rules[eq_p] = batching.ragged_mask_elementwise_rule

ne_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'ne', allow_extended_dtype=True)
ad.defjvp_zero(ne_p)
mlir.register_lowering(ne_p, partial(_compare_lower_hlo, "NE", False))

ge_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'ge')
ad.defjvp_zero(ge_p)
mlir.register_lowering(ge_p, partial(_compare_lower_hlo, "GE", False))

gt_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'gt')
ad.defjvp_zero(gt_p)
mlir.register_lowering(gt_p, partial(_compare_lower_hlo, "GT", False))

le_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'le')
ad.defjvp_zero(le_p)
mlir.register_lowering(le_p, partial(_compare_lower_hlo, "LE", False))

lt_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'lt')
ad.defjvp_zero(lt_p)
mlir.register_lowering(lt_p, partial(_compare_lower_hlo, "LT", False))
batching.ragged_prop_rules[lt_p] = batching.ragged_mask_elementwise_rule

eq_to_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'eq_to')
ad.defjvp_zero(eq_to_p)
mlir.register_lowering(eq_to_p, partial(_compare_lower_hlo, "EQ", True))

le_to_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'le_to')
ad.defjvp_zero(le_to_p)
mlir.register_lowering(le_to_p, partial(_compare_lower_hlo, "LE", True))

lt_to_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'lt_to')
ad.defjvp_zero(lt_to_p)
mlir.register_lowering(lt_to_p, partial(_compare_lower_hlo, "LT", True))


def _convert_element_type_shape_rule(operand, *, new_dtype, weak_type,
                                     sharding):
  return operand.shape

def _convert_element_type_sharding_rule(operand, *, new_dtype, weak_type,
                                        sharding):
  return sharding

def _convert_element_type_dtype_rule(operand, *, new_dtype, weak_type,
                                     sharding):
  return new_dtype

def _convert_element_type_weak_type_rule(operand, *, new_dtype, weak_type,
                                         sharding):
  return weak_type

def _convert_element_type_transpose_rule(ct, operand, *, new_dtype, weak_type,
                                         sharding):
  assert ad.is_undefined_primal(operand)
  old_dtype = operand.aval.dtype
  old_weak_type = dtypes.is_weakly_typed(operand)
  if type(ct) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)]
  elif core.primal_dtype_to_tangent_dtype(old_dtype) == dtypes.float0:
    return [ad_util.Zero(operand.aval.update(dtype=dtypes.float0, weak_type=False))]
  else:
    return [convert_element_type_p.bind(
        ct, new_dtype=old_dtype, weak_type=old_weak_type, sharding=sharding)]

def _convert_element_type_jvp_rule(tangent, primal_result, operand, *,
                                   new_dtype, weak_type, sharding):
  new_tangent_dtype = core.primal_dtype_to_tangent_dtype(new_dtype)
  if new_tangent_dtype == dtypes.float0:
    return ad_util.Zero.from_primal_value(primal_result)
  else:
    return convert_element_type_p.bind(tangent, new_dtype=new_tangent_dtype,
                                       weak_type=weak_type, sharding=sharding)

def _convert_elt_type_folding_rule(consts, eqn):
  # We constant-fold convert_element_types applied to constants if those
  # constants are Python builtin numeric types or numpy.ndarrays (so as not
  # to perform any device operations when constant-folding) and if the output
  # type can be faithfully represented by a Python builtin numeric type or
  # numpy.ndarray. If those conditions are met, we output a numpy.ndarray
  # constant if the output type is not weak, and if the output type is weak then
  # we output a Python builtin numeric type.
  # TODO(mattjj): allow constant-folding CPU-backed JAX arrays
  c, = consts
  o, = eqn.outvars
  new_dtype = eqn.params['new_dtype']
  if (type(c) in {np.ndarray, *dtypes.python_scalar_dtypes} and
      isinstance(o.aval, core.UnshapedArray) and not np.shape(c) and
      not dtypes.issubdtype(new_dtype, dtypes.extended)):
    out = np.array(c)
    if (dtypes.issubdtype(out.dtype, np.complexfloating) and
        not dtypes.issubdtype(new_dtype, np.complexfloating)):
      out = out.real
    out = out.astype(new_dtype)
    if not o.aval.weak_type:
      return [out], None
    out = out.item()
    if core.get_aval(out).dtype is o.aval.dtype:
      return [out], None
  return [None], eqn

def _convert_elt_type_fwd_rule(eqn):
  v, = eqn.invars
  if (not dtypes.issubdtype(eqn.params['new_dtype'], dtypes.extended) and
      not dtypes.issubdtype(v.aval.dtype, dtypes.extended) and
      v.aval.dtype == eqn.params['new_dtype'] and
      v.aval.weak_type == eqn.params['weak_type']):
    return [v], None
  else:
    return [None], eqn

def _convert_elt_type_pp_rule(eqn, context, settings):
  # don't print new_dtype because the output binder shows it, don't print
  # weak_type when false
  params = dict(eqn.params)
  if params['sharding'] is None:
    del params['sharding']  # don't show trivial case
  return core._pp_eqn(eqn.replace(params=params), context, settings)

convert_element_type_p = Primitive('convert_element_type')

# TODO(dougalm): I'm overriding bind_with_trace here because that's the closest thing to
# the old "custom bind" but it might not be the best way to do this.
def _convert_element_type_bind_with_trace(trace, args, params):
  sharding = params['sharding']
  operand = core.Primitive.bind_with_trace(convert_element_type_p, trace, args, params)
  if sharding is not None and not config.sharding_in_types.value:
    with core.set_current_trace(trace):
      operand = pjit.with_sharding_constraint(operand, sharding)
  return operand
convert_element_type_p.def_bind_with_trace(_convert_element_type_bind_with_trace)

convert_element_type_p.def_impl(partial(dispatch.apply_primitive, convert_element_type_p))
convert_element_type_p.def_abstract_eval(
    partial(standard_abstract_eval, convert_element_type_p,
            _convert_element_type_shape_rule, _convert_element_type_dtype_rule,
            _convert_element_type_weak_type_rule,
            _convert_element_type_sharding_rule))
ad.defjvp2(convert_element_type_p, _convert_element_type_jvp_rule)
ad.primitive_transposes[convert_element_type_p] = _convert_element_type_transpose_rule
batching.defvectorized(convert_element_type_p)
pe.const_fold_rules[convert_element_type_p] = _convert_elt_type_folding_rule
pe.forwarding_rules[convert_element_type_p] = _convert_elt_type_fwd_rule
pe.def_trivial_padding(convert_element_type_p)
core.pp_eqn_rules[convert_element_type_p] = _convert_elt_type_pp_rule
batching.ragged_prop_rules[convert_element_type_p] = (
    batching.ragged_mask_elementwise_rule
)

def _real_dtype(dtype): return np.finfo(dtype).dtype

def _convert_element_type_lower(ctx, operand, *, new_dtype, weak_type,
                                sharding):
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  if (dtypes.issubdtype(aval_in.dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    operand = hlo.real(operand)
    aval_in = aval_in.update(dtype=_real_dtype(aval_in.dtype))
  out = mlir.convert_hlo(ctx, operand, aval_in, aval_out)
  if config.sharding_in_types.value:
    if sharding is not None:
      assert aval_out.sharding == sharding
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

mlir.register_lowering(convert_element_type_p, _convert_element_type_lower)


def _to_edtype_abstract_eval(x, *, edtype):
  assert (isinstance(edtype, dtypes.ExtendedDType) and
          not isinstance(x.dtype, dtypes.ExtendedDType))
  # For backward compatibility, if the edtype rules have a `convert_to` method,
  # use that rather than looking for an `allow_conversion: bool` attribute.
  if convert_to := getattr(edtype._rules, 'convert_to', None):
    allow_conversion = convert_to(x.dtype, edtype)
  else:
    allow_conversion = edtype._rules.allow_conversion
  if not allow_conversion:
    raise ValueError(
        f"Cannot convert_element_type from {dtype_to_string(x.dtype)} "
        f"to {dtype_to_string(edtype)}")
  rep_aval = core.physical_element_aval(edtype)
  if x.dtype != rep_aval.dtype:
    raise ValueError(
        "can only convert to extended dtype from its representation dtype, "
        f"but tried to convert from {dtype_to_string(x.dtype)} to "
        f"{dtype_to_string(edtype)} which doesn't match the representation type "
        f"{dtype_to_string(rep_aval.dtype)}.")
  if x.ndim < rep_aval.ndim:
    raise ValueError(
        "can only convert to extended dtype from an array of its "
        f"representation type, but the extended dtype {dtype_to_string(edtype)}"
        f" has a representation shape {rep_aval.shape} (rank {rep_aval.ndim}) "
        f"while the given representation array has shape {x.shape} (rank "
        f"{x.ndim} < {rep_aval.ndim}).")
  n = x.ndim - rep_aval.ndim
  shape_prefix, shape_suffix = x.shape[:n], x.shape[n:]
  if shape_suffix != rep_aval.shape:
    raise ValueError(
        "can only convert to extended dtype from an array of its "
        f"representation type, but the extended dtype {dtype_to_string(edtype)}"
        f" has a representation shape {rep_aval.shape} while the given "
        f"representation array has shape {x.shape}, so the shape suffix "
        f"does not match: given {shape_suffix} but required {rep_aval.shape}.")
  return x.update(shape=shape_prefix, dtype=edtype)

to_edtype_p = Primitive('to_edtype')
to_edtype_p.def_impl(partial(dispatch.apply_primitive, to_edtype_p))
to_edtype_p.def_abstract_eval(_to_edtype_abstract_eval)
ad.defjvp(to_edtype_p,
          lambda t, x, edtype:
          convert_element_type(t, core.primal_dtype_to_tangent_dtype(edtype)))
ad.primitive_transposes[to_edtype_p] = \
    lambda ct, x, edtype: [from_edtype_p.bind(ct, dtype=x.aval.dtype)]  # type: ignore
batching.defvectorized(to_edtype_p)
mlir.register_lowering(to_edtype_p, lambda _, x, **__: [x])


def _from_edtype_abstract_eval(x, *, dtype):
  assert (isinstance(x.dtype, dtypes.ExtendedDType) and
          not isinstance(dtype, dtypes.ExtendedDType))
  if convert_from := getattr(x.dtype._rules, 'convert_from', None):
    allow_conversion = convert_from(x.dtype, dtype)
  else:
    allow_conversion = x.dtype._rules.allow_conversion
  if not allow_conversion:
    raise ValueError(
        f"Cannot convert_element_type from {dtype_to_string(x.dtype)} "
        f"to {dtype_to_string(dtype)}")
  rep_aval = core.physical_element_aval(x.dtype)
  if rep_aval.dtype != dtype:
    raise ValueError(
        "can only convert from extended dtype to its representation dtype, "
        f"but tried to convert from {dtype_to_string(x.dtype)} to "
        f"{dtype_to_string(dtype)} which doesn't match the representation type "
        f"{dtype_to_string(rep_aval.dtype)}.")
  if all(isinstance(d, int) for d in x.shape):
    return core.ShapedArray(shape=(*x.shape, *rep_aval.shape), dtype=dtype)
  else:
    raise NotImplementedError

from_edtype_p = Primitive('from_edtype')
from_edtype_p.def_impl(partial(dispatch.apply_primitive, from_edtype_p))
from_edtype_p.def_abstract_eval(_from_edtype_abstract_eval)
ad.defjvp(from_edtype_p,
          lambda t, x, dtype:
          convert_element_type(t, core.primal_dtype_to_tangent_dtype(dtype)))
ad.primitive_transposes[from_edtype_p] = \
    lambda ct, x, dtype: [to_edtype_p.bind(ct, edtype=x.dtype)]
batching.defvectorized(from_edtype_p)
mlir.register_lowering(from_edtype_p, lambda _, x, **__: [x])


def _bitcast_convert_type_shape_rule(operand, *, new_dtype):
  old_dtype = dtypes.canonicalize_dtype(operand.dtype)
  new_dtype = dtypes.canonicalize_dtype(new_dtype)

  old_nbits = dtypes.bit_width(old_dtype)
  new_nbits = dtypes.bit_width(new_dtype)

  if old_nbits == new_nbits:
    return operand.shape
  elif old_nbits > new_nbits:
    return (*operand.shape, old_nbits // new_nbits)
  else:
    dim_size = operand.shape[-1] if operand.shape else 1
    if dim_size * old_nbits != new_nbits:
      raise ValueError(
        f"Attempting to convert array of shape {operand.shape} "
        f"from {old_dtype} of size {old_nbits} bits "
        f"to {new_dtype} of size {new_nbits}, bits "
        f"but {dim_size} * {old_nbits} != {new_nbits}")
    return operand.shape[:-1]

def _bitcast_convert_type_dtype_rule(operand, *, new_dtype):
  old_dtype = dtypes.canonicalize_dtype(operand.dtype)
  new_dtype = dtypes.canonicalize_dtype(new_dtype)
  if (dtypes.issubdtype(old_dtype, np.bool_) or
      dtypes.issubdtype(old_dtype, np.complexfloating) or
      dtypes.issubdtype(new_dtype, np.bool_) or
      dtypes.issubdtype(new_dtype, np.complexfloating)):
    if old_dtype != new_dtype:
      raise TypeError("lax.bitcast_convert_type does not support bool or complex values "
                      "unless the operand and destination types match. "
                      f"Got operand dtype={old_dtype}, {new_dtype=}. "
                      "Consider using the arr.view() method instead.")
  return new_dtype

bitcast_convert_type_p = standard_primitive(
    _bitcast_convert_type_shape_rule, _bitcast_convert_type_dtype_rule,
    'bitcast_convert_type', weak_type_rule=_strip_weak_type)
ad.defjvp_zero(bitcast_convert_type_p)
batching.defvectorized(bitcast_convert_type_p)

def _bitcast_convert_type_lower(ctx, operand, *, new_dtype):
  aval_out, = ctx.avals_out
  return [hlo.bitcast_convert(mlir.aval_to_ir_type(aval_out), operand)]

mlir.register_lowering(bitcast_convert_type_p, _bitcast_convert_type_lower)


def _validate_preferred_element_type(input_dtype, preferred_element_type):
  if (dtypes.issubdtype(input_dtype, np.integer) and
      dtypes.issubdtype(preferred_element_type, np.floating)):
    # Special-case integer->float multiply. This is allowed, and also allows
    # different signedness between input and output.
    pass
  else:
    allowed_types = (np.integer, np.floating, np.complexfloating)
    if any(dtypes.issubdtype(input_dtype, t) and not
           dtypes.issubdtype(preferred_element_type, t) for t in allowed_types):
      raise TypeError("Input type is incompatible with "
                      "`preferred_element_type`. The compatible combinations "
                      "of (input_type, preferred_element_type) are "
                      "(integral, integral), (integral, floating), "
                      "(floating, floating), (complex, complex.")
    if (dtypes.issubdtype(input_dtype, np.signedinteger) and
        not dtypes.issubdtype(preferred_element_type, np.signedinteger)):
      raise TypeError("`preferred_element_type` must have the same signedness "
                      "as the original type.")
  input_bitwidth = np.dtype(input_dtype).itemsize
  preferred_bitwidth = np.dtype(preferred_element_type).itemsize
  if preferred_bitwidth < input_bitwidth:
    raise TypeError("`preferred_element_type` must not be narrower than the "
                    "original type.")


def _dot_general_shape_rule(lhs, rhs, *, dimension_numbers, precision,
                            preferred_element_type: DTypeLike | None,
                            out_sharding):
  if out_sharding is not None and not isinstance(out_sharding, NamedSharding):
    raise NotImplementedError
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  if not all(np.all(np.greater_equal(d, 0)) and np.all(np.less(d, lhs.ndim))
             for d in (lhs_contracting, lhs_batch)):
    msg = ("dot_general requires lhs dimension numbers to be nonnegative and "
           "less than the number of axes of the lhs value, got "
           f"lhs_batch of {lhs_batch} and lhs_contracting of {lhs_contracting} "
           f"for lhs of rank {lhs.ndim}")
    raise TypeError(msg)
  if not all(np.all(np.greater_equal(d, 0)) and np.all(np.less(d, rhs.ndim))
             for d in (rhs_contracting, rhs_batch)):
    msg = ("dot_general requires rhs dimension numbers to be nonnegative and "
           "less than the number of axes of the rhs value, got "
           f"rhs_batch of {rhs_batch} and rhs_contracting of {rhs_contracting} "
           f"for rhs of rank {rhs.ndim}")
    raise TypeError(msg)
  if len(lhs_batch) != len(rhs_batch):
    msg = ("dot_general requires equal numbers of lhs_batch and rhs_batch "
           "dimensions, got lhs_batch {} and rhs_batch {}.")
    raise TypeError(msg.format(lhs_batch, rhs_batch))
  lhs_contracting_set, lhs_batch_set = set(lhs_contracting), set(lhs_batch)
  rhs_contracting_set, rhs_batch_set = set(rhs_contracting), set(rhs_batch)
  if len(lhs_batch_set) != len(lhs_batch):
    msg = ("dot_general requires lhs batch dimensions to be distinct, got "
           f"lhs_batch {lhs_batch}.")
    raise TypeError(msg)
  if len(rhs_batch_set) != len(rhs_batch):
    msg = ("dot_general requires rhs batch dimensions to be distinct, got "
           f"rhs_batch {rhs_batch}.")
    raise TypeError(msg)
  if len(lhs_contracting_set) != len(lhs_contracting):
    msg = ("dot_general requires lhs contracting dimensions to be distinct, "
           f"got lhs_contracting {lhs_contracting}.")
    raise TypeError(msg)
  if len(rhs_contracting_set) != len(rhs_contracting):
    msg = ("dot_general requires rhs contracting dimensions to be distinct, "
           f"got rhs_contracting {rhs_contracting}.")
    raise TypeError(msg)
  if lhs_contracting_set & lhs_batch_set:
    msg = ("dot_general requires lhs batch dimensions to be disjoint from "
           "contracting dimensions, got lhs_batch {} and lhs_contracting {}.")
    raise TypeError(msg.format(lhs_batch, lhs_contracting))
  if rhs_contracting_set & rhs_batch_set:
    msg = ("dot_general requires rhs batch dimensions to be disjoint from "
           "contracting dimensions, got rhs_batch {} and rhs_contracting {}.")
    raise TypeError(msg.format(rhs_batch, rhs_contracting))
  lhs_batch_shape = tuple(lhs.shape[i] for i in lhs_batch)
  rhs_batch_shape = tuple(rhs.shape[i] for i in rhs_batch)
  if not core.definitely_equal_shape(lhs_batch_shape, rhs_batch_shape):
    msg = ("dot_general requires lhs batch dimensions and rhs batch dimensions "
           "to have the same shape, got {} and {}.")
    raise TypeError(msg.format(lhs_batch_shape, rhs_batch_shape))
  lhs_contracting_shape = tuple(lhs.shape[i] for i in lhs_contracting)
  rhs_contracting_shape = tuple(rhs.shape[i] for i in rhs_contracting)
  if not core.definitely_equal_shape(lhs_contracting_shape, rhs_contracting_shape):
    msg = ("dot_general requires contracting dimensions to have the same "
           "shape, got {} and {}.")
    raise TypeError(msg.format(lhs_contracting_shape, rhs_contracting_shape))

  return _dot_general_shape_computation(lhs.shape, rhs.shape, dimension_numbers)

def _dot_general_shape_computation(lhs_shape, rhs_shape, dimension_numbers):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  batch_shape = tuple(lhs_shape[i] for i in lhs_batch)
  lhs_contract_or_batch = tuple(sorted(tuple(lhs_contracting) + tuple(lhs_batch)))
  lhs_tensored_shape = tuple_delete(lhs_shape, lhs_contract_or_batch)
  rhs_contract_or_batch = tuple(sorted(tuple(rhs_contracting) + tuple(rhs_batch)))
  rhs_tensored_shape = tuple_delete(rhs_shape, rhs_contract_or_batch)
  return batch_shape + lhs_tensored_shape + rhs_tensored_shape


def _check_specs_match(lhs_spec, rhs_spec, msg):
  for l, r in zip(lhs_spec, rhs_spec):
    if l is not None and r is not None and l != r:
      raise TypeError(msg)

def _dot_general_sharding_rule(lhs, rhs, *, dimension_numbers, precision,
                               preferred_element_type: DTypeLike | None,
                               out_sharding):
  if lhs.sharding.mesh != rhs.sharding.mesh:
    raise ValueError(
        'Mesh of both lhs and rhs should match. Got lhs:'
        f' {lhs.sharding.mesh} and rhs: {rhs.sharding.mesh}')

  if out_sharding is not None:
    assert isinstance(out_sharding, NamedSharding)
    return out_sharding

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_batch_spec = tuple(lhs.sharding.spec[i] for i in lhs_batch)
  rhs_batch_spec = tuple(rhs.sharding.spec[i] for i in rhs_batch)
  msg = ("dot_general requires lhs batch dimensions and rhs batch dimensions "
        f"to have the consistent sharding, got {lhs_batch_spec} and "
        f"{rhs_batch_spec}.")
  _check_specs_match(lhs_batch_spec, rhs_batch_spec, msg)

  lhs_contracting_spec = tuple(lhs.sharding.spec[i] for i in lhs_contracting)
  rhs_contracting_spec = tuple(rhs.sharding.spec[i] for i in rhs_contracting)
  msg = ("dot_general requires contracting dimensions to have consistent "
        f"sharding, got {lhs_contracting_spec} and {rhs_contracting_spec}.")
  _check_specs_match(lhs_contracting_spec, rhs_contracting_spec, msg)

  return _dot_general_sharding_computation(
      lhs.sharding.spec, rhs.sharding.spec, dimension_numbers, lhs.sharding.mesh)

def _dot_general_sharding_computation(lhs_spec, rhs_spec,
                                      dimension_numbers, mesh):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  batch_spec = tuple(lhs_spec[i] for i in lhs_batch)
  lhs_contract_or_batch = tuple(sorted(tuple(lhs_contracting) + tuple(lhs_batch)))
  lhs_tensored_spec = tuple_delete(lhs_spec, lhs_contract_or_batch)
  rhs_contract_or_batch = tuple(sorted(tuple(rhs_contracting) + tuple(rhs_batch)))
  rhs_tensored_spec = tuple_delete(rhs_spec, rhs_contract_or_batch)
  return NamedSharding(mesh, P(*(batch_spec + lhs_tensored_spec + rhs_tensored_spec)))

def tuple_delete(tup, idx):
  idx_ = set(idx)
  return tuple(tup[i] for i in range(len(tup)) if i not in idx_)


def _dot_general_dtype_rule(lhs, rhs, *, dimension_numbers, precision,
                            preferred_element_type: DTypeLike | None,
                            out_sharding):
  if out_sharding is not None and not isinstance(out_sharding, NamedSharding):
    raise NotImplementedError
  del dimension_numbers  # unused
  # We're mostly matching XLA's logic here, namely in shape_inference.cc and
  # primitive_util.h's HigherPrecisionType, e.g.
  # https://github.com/openxla/xla/blob/ea3a841768d0dcf192e5820c9b25c34c73f2226a/xla/primitive_util.h#L329
  def type_properties(dt):
    c = _real_dtype(dt) if dtypes.issubdtype(dt, np.complexfloating) else dt
    return (dtypes.issubdtype(dt, np.complexfloating),
            dtypes.finfo(c).maxexp if dtypes.issubdtype(c, np.floating) else -1,
            dtypes.finfo(c).nmant  if dtypes.issubdtype(c, np.floating) else -1,
            _bit_width(c),
            not dtypes.issubdtype(c, np.unsignedinteger))
  lhs_prop, rhs_prop = type_properties(lhs.dtype), type_properties(rhs.dtype)
  if lhs_prop > rhs_prop:
    result_dtype = lhs.dtype
  elif rhs_prop > lhs_prop:
    result_dtype = rhs.dtype
  else:
    if lhs.dtype != rhs.dtype:
      raise TypeError(
          f"lax.dot_general argument type error: {lhs.dtype}, {rhs.dtype}")
    result_dtype = lhs.dtype
  has_algorithm = isinstance(precision, (DotAlgorithm, DotAlgorithmPreset))
  return _maybe_upcast(result_dtype, preferred_element_type,
                       check_bit_width=not has_algorithm)

def _bit_width(d):
  if dtypes.issubdtype(d, np.inexact): return dtypes.finfo(d).bits
  elif dtypes.issubdtype(d, np.integer): return dtypes.iinfo(d).bits
  elif d == np.dtype('bool'): return 1
  else: assert False, d  # should be unreachable, open an issue!

def _maybe_upcast(result_dtype, preferred_element_type, check_bit_width):
  # replicates the logic in shape_inference.cc's MaybeUpcast
  if (preferred_element_type is None or
      result_dtype == preferred_element_type):
    return result_dtype
  if (check_bit_width and not dtypes.issubdtype(result_dtype, np.floating) and
      _bit_width(preferred_element_type) < _bit_width(result_dtype)):
    raise TypeError("`preferred_element_type` must not be narrower than the "
                    "original type, got preferred_element_type of "
                    f"{preferred_element_type} for result type of "
                    f"{result_dtype}.")
  return preferred_element_type

def _dot_general_transpose_lhs(g, x, y, *, dimension_numbers, precision,
                               preferred_element_type: DTypeLike | None,
                               out_sharding, swap_ans=False):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  x_ndim = x.aval.ndim
  x_kept = remaining(range(x_ndim), x_contract, x_batch)
  y_kept = remaining(range(np.ndim(y)), y_contract, y_batch)
  if swap_ans:
    ans_batch, ans_y, _ = ranges_like(x_batch, y_kept, x_kept)
  else:
    ans_batch, _, ans_y = ranges_like(x_batch, x_kept, y_kept)
  dims = ((ans_y, y_kept), (ans_batch, y_batch))
  x_contract_sorted_by_y = list(np.take(x_contract, np.argsort(y_contract)))
  unsorted_axes = list(x_batch) + x_kept + x_contract_sorted_by_y
  out_axes = np.argsort(unsorted_axes)
  if config.sharding_in_types.value:
    xs = x.aval.sharding
    inverse_spec = tuple(xs.spec[o] for o in unsorted_axes)
    ds = xs.with_spec(inverse_spec)
  else:
    ds = None
  dot_general_out = dot_general(g, y, dims, precision=precision,
                                preferred_element_type=preferred_element_type,
                                out_sharding=ds)
  x_bar = transpose(dot_general_out, tuple(out_axes))
  if x_bar.dtype != x.aval.dtype:
    x_bar = _convert_element_type(x_bar, x.aval.dtype, x.aval.weak_type)
  return x_bar

def _dot_general_transpose_rhs(g, x, y, *, dimension_numbers, precision,
                               preferred_element_type: DTypeLike | None,
                               out_sharding):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
  y_bar = _dot_general_transpose_lhs(
    g, y, x, dimension_numbers=swapped_dimension_numbers, precision=precision,
    preferred_element_type=preferred_element_type, out_sharding=out_sharding,
    swap_ans=True)
  if y_bar.dtype != y.aval.dtype:
    y_bar = _convert_element_type(y_bar, y.aval.dtype, y.aval.weak_type)
  return y_bar


def _dot_batch_rule(
    unpack_args,
    unpack_dims,
    invoke_prim,
    batched_args,
    batch_dims,
    *,
    dimension_numbers,
    out_sharding,
    precision,
    preferred_element_type: DTypeLike | None,
    **_,
):

  lhs, rhs = unpack_args(batched_args)
  lbd, rbd = unpack_dims(batch_dims)

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  left_stack_dim = lbd.stacked_axis if type(lbd) is RaggedAxis else lbd
  right_stack_dim = rbd.stacked_axis if type(rbd) is RaggedAxis else rbd
  new_dimension_numbers, result_stack_dim = _dot_general_batch_dim_nums(
      (np.ndim(lhs), np.ndim(rhs)), (left_stack_dim, right_stack_dim),
      dimension_numbers)
  # TODO Should probably check that any ragged dimensions have corresponding
  # sizes, because otherwise the dot product is technically undefined.
  #
  # This masking is not strictly necessary for non-contraction dimensions;
  # we could micro-optimize here by avoiding computing that mask.
  if type(lbd) is RaggedAxis:
    lhs = batching.mask_ragged_axes(lhs, _get_sum_identity, lbd)
    lhs_shape = batching.bdim_as_shape(lbd, lhs.shape)
  else:
    lhs_shape = np.shape(lhs)
  if type(rbd) is RaggedAxis:
    rhs = batching.mask_ragged_axes(rhs, _get_sum_identity, rbd)
    rhs_shape = batching.bdim_as_shape(rbd, rhs.shape)
  else:
    rhs_shape = np.shape(rhs)
  if out_sharding is not None:
    raise NotImplementedError("vmap with out_sharding is not supported. "
                              "Please open an issue.")
  batched_out = invoke_prim(
      lhs,
      rhs,
      new_dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
      out_sharding=out_sharding,
  )
  result_batch_dim = batching.shape_as_bdim(
      result_stack_dim,
      _dot_general_shape_computation(lhs_shape, rhs_shape, new_dimension_numbers))
  return batched_out, result_batch_dim


def _dot_general_batch_dim_nums(ndims, batch_dims, dimension_numbers):
  # There are three kinds of dimensions in a dot_general:
  # - contraction dimensions appear in lhs and rhs but not the result
  # - batch dimensions appear in lhs, rhs, and result
  # - tensor product dimensions appear in the result and one of lhs or rhs
  # The dimensions of the result are ordered as
  # - Batch dimensions
  #   - Q: In what order?  The order of appearance in lhs, rhs, or
  #     dimension_numbers?
  # - Tensor dimensions from the LHS
  # - Tensor dimensions from the RHS
  lhs_ndim, rhs_ndim = ndims
  # lbd and rbd are "batch" dimensions in the sense of dimensions being
  # vmapped, not to be confused with "batch" dimensions in the sense of
  # explicitly present dimensions that this dot_general is zipping together.
  lbd, rbd = batch_dims
  assert lbd is not None or rbd is not None
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  def bump_dims(dims, b):
    return tuple(np.add(dims, np.greater_equal(dims, b)))

  if type(lbd) is type(rbd) is int:
    # The vmapped dimensions become an additional batch dimension in the
    # batched dot_general, which we arbitrarily put first.
    lhs_batch = (lbd,) + bump_dims(lhs_batch, lbd)
    rhs_batch = (rbd,) + bump_dims(rhs_batch, rbd)
    lhs_contract = bump_dims(lhs_contract, lbd)
    rhs_contract = bump_dims(rhs_contract, rbd)
    result_batch_dim = 0
  elif (type(lbd) is int and rbd is None):
    # The left vmapped dimension becomes an additional tensor dimension in the
    # batched dot_general.
    lhs_tensor = [d for d in range(lhs_ndim)
                  if d not in lhs_batch and d not in lhs_contract]
    result_batch_dim = len(lhs_batch) + int(sum(np.less(lhs_tensor, lbd)))
    lhs_batch = bump_dims(lhs_batch, lbd)
    lhs_contract = bump_dims(lhs_contract, lbd)
  elif (type(rbd) is int and lbd is None):
    # The right vmapped dimension becomes an additional tensor dimension in the
    # batched dot_general.
    rhs_tensor = [d for d in range(rhs_ndim)
                  if d not in rhs_batch and d not in rhs_contract]
    result_batch_dim = (lhs_ndim - len(lhs_contract) +
                        int(sum(np.less(rhs_tensor, rbd))))
    rhs_batch = bump_dims(rhs_batch, rbd)
    rhs_contract = bump_dims(rhs_contract, rbd)
  else:
    # We wouldn't be here if we didn't have at least one vmapped dimension.
    assert False

  new_dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
  return new_dimension_numbers, result_batch_dim

def _dot_general_padding_rule(in_avals, out_avals, lhs, rhs, *,
                              dimension_numbers, **params):
  lhs_aval, _ = in_avals
  (lhs_contract, _), _ = dimension_numbers
  padded_axes = [(i, lhs_aval.shape[i].val) for i in lhs_contract
                 if isinstance(lhs_aval.shape[i], pe.BoundedAxisSize)]
  lhs_ = _replace_masked_values(lhs, 0, padded_axes)
  return [dot_general(lhs_, rhs, dimension_numbers=dimension_numbers, **params)]

def _dot_general_pp_rule(eqn, context, settings) -> pp.Doc:
  # * suppress printing precision or preferred_element_type when None.
  # * print dimension_numbers as list-of-lists to be shorter.
  printed_params = {k: v for k, v in eqn.params.items() if v is not None}
  (lhs_cont, rhs_cont), (lhs_batch, rhs_batch) = eqn.params['dimension_numbers']
  printed_params['dimension_numbers'] = (
      (list(lhs_cont), list(rhs_cont)), (list(lhs_batch), list(rhs_batch)))
  return core._pp_eqn(eqn.replace(params=printed_params), context, settings)


def _dot_general_ragged_prop_rule(eqn_params, invar_raggedness, outvars):
  assert len(invar_raggedness) == 2
  assert len(outvars) == 1
  invar_raggedness_lhs = invar_raggedness[0]
  invar_raggedness_rhs = invar_raggedness[1]

  dimension_numbers = eqn_params['dimension_numbers']
  (lhs_contracting, rhs_contracting), (_, _) = dimension_numbers

  if not invar_raggedness_lhs and not invar_raggedness_rhs:
    # Both are dense - it is valid to reach here, because dense operations
    # are legal in code running under ragged prop.
    return invar_raggedness, [None]

  if not invar_raggedness_lhs or not invar_raggedness_rhs:
    # One ragged, one dense
    if not invar_raggedness_lhs:
      # left is dense, right is ragged
      _, ragged_axis_dim_rhs, _, _ = invar_raggedness_rhs
      if rhs_contracting != ragged_axis_dim_rhs:
        # Contraction is on a dense dimension, this is valid!
        return invar_raggedness, [None]
    if not invar_raggedness_rhs:
      # left is ragged, right is dense
      _, ragged_axis_dim_lhs, _, _ = invar_raggedness_lhs
      if lhs_contracting != ragged_axis_dim_lhs:
        # Contraction is on a dense dimension, this is valid!
        return invar_raggedness, [None]

    raise NotImplementedError('NYI - dense and ragged dim contraction')

  stacked_axis_lhs, ragged_axis_dim_lhs, _, _ = invar_raggedness_lhs
  stacked_axis_rhs, ragged_axis_dim_rhs, _, _ = invar_raggedness_rhs

  if stacked_axis_rhs != 0 or stacked_axis_lhs != 0:
    raise NotImplementedError(
        'Dot general ragged prop for non 0 stacked axis, NYI'
    )

  # We only support ragged k atm, that is, lhs is (m, ragged_k) and rhs is
  # (ragged_k, n), meaning the output is dense.
  if ragged_axis_dim_lhs != 2 or ragged_axis_dim_rhs != 1:
    raise NotImplementedError(
        'Dot general ragged prop for non contraction raggedness, NYI'
    )

  assert len(outvars) == 1

  # TODO(mvoz): A constant on batching.* ?
  # Dense (m, n) - no jumble only atm
  return invar_raggedness, [None]


dot_general_p = standard_primitive(
    _dot_general_shape_rule,
    _dot_general_dtype_rule,
    'dot_general',
    sharding_rule=_dot_general_sharding_rule,
)


def _dot_general_batch_unpack_args(batch_args):
  lhs, rhs = batch_args
  return (lhs, rhs)


def _dot_general_batch_unpack_dims(batch_dims):
  lbd, rbd = batch_dims
  return (lbd, rbd)

# DotDimensionNumbers used in the dot_general call for ragged_dot().
_RAGGED_DOT_DOT_DIMENSION_NUMBERS: DotDimensionNumbers = (
    ([2, 0], [1, 0]),
    ([], []),
)
_RAGGED_DOT_BATCH_DOT_DIMENSION_NUMBERS: DotDimensionNumbers = (
    ([3, 1], [2, 1]),
    ([0], [0]),
)

ad.defbilinear(dot_general_p,
               _dot_general_transpose_lhs, _dot_general_transpose_rhs)
_dot_general_batch_rule = functools.partial(
    _dot_batch_rule,
    _dot_general_batch_unpack_args,
    _dot_general_batch_unpack_dims,
    dot_general,
)
batching.primitive_batchers[dot_general_p] = _dot_general_batch_rule
pe.padding_rules[dot_general_p] = _dot_general_padding_rule
core.pp_eqn_rules[dot_general_p] = _dot_general_pp_rule
batching.ragged_prop_rules[dot_general_p] = _dot_general_ragged_prop_rule

def precision_attr(precision: Precision) -> ir.ArrayAttr:
  if precision is None or isinstance(precision, (DotAlgorithm, DotAlgorithmPreset)):
    full_precision = (Precision.DEFAULT, Precision.DEFAULT)
  elif not isinstance(precision, tuple):
    full_precision = (precision, precision)
  else:
    full_precision = precision
  return ir.ArrayAttr.get(
      [hlo.PrecisionAttr.get(str(p)) for p in full_precision])


def dot_algorithm_attr(precision: CanonicalPrecision, lhs_dtype: DTypeLike,
                       rhs_dtype: DTypeLike) -> hlo.DotAlgorithm | None:
  if not isinstance(precision, (DotAlgorithm, DotAlgorithmPreset)):
    return None
  return precision._convert_to_hlo_attr(lhs_dtype, rhs_dtype)


def get_algorithm_compute_types(
    algorithm: DotAlgorithm | DotAlgorithmPreset,
    lhs_dtype: DTypeLike,
    rhs_dtype: DTypeLike,
    out_dtype: DTypeLike | None = None,
) -> tuple[DTypeLike | None, DTypeLike | None, DTypeLike | None]:
  if isinstance(algorithm, DotAlgorithm):
    return (
        algorithm.lhs_precision_type,
        algorithm.rhs_precision_type,
        algorithm.accumulation_type,
    )

  def maybe_convert_dtype(input_dtype, target_dtypes):
    if target_dtypes is None:
      return input_dtype
    if np.dtype(input_dtype) in map(np.dtype, target_dtypes):
      return input_dtype
    return target_dtypes[0]

  lhs_dtype = maybe_convert_dtype(lhs_dtype, algorithm.supported_lhs_types)
  rhs_dtype = maybe_convert_dtype(rhs_dtype, algorithm.supported_rhs_types)
  out_type = maybe_convert_dtype(
      out_dtype, algorithm.supported_output_types(lhs_dtype, rhs_dtype)
  )
  return lhs_dtype, rhs_dtype, out_type


def _dot_general_lower(ctx, lhs, rhs, *, dimension_numbers,
                       precision, preferred_element_type: np.dtype | None,
                       out_sharding, platform: str = "default"):
  def _is_fp8_mixed_precision_matmul(_lhs_dtypes, _rhs_dtypes):
    fp8_dtypes = (dtypes.float8_e4m3fn, dtypes.float8_e5m2,
                  dtypes.float8_e5m2fnuz, dtypes.float8_e4m3fnuz)
    if dtypes.float8_e3m4 is not None:
      fp8_dtypes += (dtypes.float8_e3m4,)
    if dtypes.float8_e4m3 is not None:
      fp8_dtypes += (dtypes.float8_e4m3,)
    return _lhs_dtypes in fp8_dtypes and _rhs_dtypes in fp8_dtypes
  del preferred_element_type  # Implied by the output aval
  lhs_aval, rhs_aval = ctx.avals_in
  lhs_dtype, rhs_dtype = lhs_aval.dtype, rhs_aval.dtype
  aval_out, = ctx.avals_out
  accumulation_aval = aval_out
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

  dot_dnums = hlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=list(lhs_batch),
      rhs_batching_dimensions=list(rhs_batch),
      lhs_contracting_dimensions=list(lhs_contracting),
      rhs_contracting_dimensions=list(rhs_contracting))

  algorithm_kwarg = {}
  if isinstance(precision, (DotAlgorithm, DotAlgorithmPreset)):
    # The CPU backend silently ignores the algorithm spec, so we check here to
    # make sure that the selected algorithm is supported. We could be a little
    # bit more liberal here (any algorithm where the input and output types
    # match and all the other parameters have default values should work), but
    # it's probably sufficient to just check the presets here.
    if platform == "cpu" and precision not in {
        DotAlgorithmPreset.DEFAULT, DotAlgorithmPreset.F16_F16_F16,
        DotAlgorithmPreset.F32_F32_F32, DotAlgorithmPreset.F64_F64_F64,
        DotAlgorithmPreset.BF16_BF16_F32, DotAlgorithmPreset.BF16_BF16_F32_X3,
        DotAlgorithmPreset.BF16_BF16_F32_X6,
    }:
      raise ValueError(
          f"The precision '{precision}' is not supported by dot_general on CPU")

    # If an explicit algorithm was specified, we always cast the input types to
    # the correct types.
    def maybe_convert_dtype(operand, operand_aval, target_dtype):
      if target_dtype is None or operand_aval.dtype == target_dtype:
        return operand
      aval = core.ShapedArray(operand_aval.shape, target_dtype)
      return mlir.convert_hlo(ctx, operand, operand_aval, aval)

    lhs_dtype, rhs_dtype, accumulation_dtype = get_algorithm_compute_types(
        precision, lhs_dtype, rhs_dtype, aval_out.dtype)
    lhs = maybe_convert_dtype(lhs, lhs_aval, lhs_dtype)
    rhs = maybe_convert_dtype(rhs, rhs_aval, rhs_dtype)
    if accumulation_dtype is not None:
      accumulation_aval = core.ShapedArray(aval_out.shape, accumulation_dtype)

    if precision != DotAlgorithmPreset.DEFAULT:
      algorithm_kwarg = {
          "algorithm": dot_algorithm_attr(precision, lhs_dtype, rhs_dtype)
      }
  else:
    # TODO(b/...): JAX's dot_general primitive accepts the same input dtype
    # combinations that are accepted in XLA's shape_inference.cc (the canonical
    # reference for the HLO type system), but actually different XLA platforms
    # fail on codegen for different accepted cases. To handle those cases, we
    # insert ConvertOps on the input, in a platform-dependent way.
    if lhs_dtype != rhs_dtype:
      if platform == "tpu":
        handled = lambda dt: (dtypes.issubdtype(dt, np.floating) or
                              dtypes.issubdtype(dt, np.integer))
        if not (handled(lhs_dtype) and handled(rhs_dtype)):
          lhs = mlir.convert_hlo(ctx, lhs, lhs_aval,
                                 core.ShapedArray(lhs_aval.shape, aval_out.dtype))
          rhs = mlir.convert_hlo(ctx, rhs, rhs_aval,
                                 core.ShapedArray(rhs_aval.shape, aval_out.dtype))
      else:  # cpu and gpu
        # Do not convert mixed fp8 types to output type.
        if not _is_fp8_mixed_precision_matmul(lhs_dtype, rhs_dtype):
          lhs = mlir.convert_hlo(ctx, lhs, lhs_aval,
                                 core.ShapedArray(lhs_aval.shape, aval_out.dtype))
          rhs = mlir.convert_hlo(ctx, rhs, rhs_aval,
                                 core.ShapedArray(rhs_aval.shape, aval_out.dtype))

  result = hlo.dot_general(
      mlir.aval_to_ir_type(accumulation_aval),
      lhs,
      rhs,
      dot_dnums,
      precision_config=precision_attr(precision),
      **algorithm_kwarg,
  )
  if config.sharding_in_types.value:
    if out_sharding is not None:
      assert aval_out.sharding == out_sharding
    result = mlir.lower_sharding_under_shit(ctx, result, aval_out)
  if accumulation_aval.dtype != aval_out.dtype:
    result = mlir.convert_hlo(ctx, result, accumulation_aval, aval_out)
  return [result]

mlir.register_lowering(dot_general_p, _dot_general_lower)

for platform in ["cpu", "tpu"]:
  mlir.register_lowering(dot_general_p,
                         partial(_dot_general_lower, platform=platform),
                         platform=platform)


def _ragged_dot_shape_rule(lhs: Array, rhs: Array, group_sizes: Array, **_) -> Shape:
  if len(lhs.shape) == 3:
    # Batched case
    b, m, k = lhs.shape
    b2, group_count, rk, n = rhs.shape
    b3 = group_sizes.shape[0]
    if b != b2:
      raise TypeError(
          f'ragged_dot requires that lhs.shape[0] == rhs.shape[0]: got {b} and'
          f' {b2}.'
      )
    if b3 != b:
      raise TypeError(
          'ragged_dot requires that group_sizes.shape[0] == lhs.shape[0]: got'
          f' {b3} and {b}.'
      )
    if k != rk:
      raise TypeError(
          f'ragged_dot requires that lhs.shape[1] == rhs.shape[1]: got {k} and'
          f' {rk}.'
      )
    num_groups = group_sizes.shape[1]
    if group_count != num_groups:
      raise TypeError(
          'ragged_dot requires that rhs.shape[1] == group_sizes.shape[1]: got'
          f' {group_count} and {num_groups}.'
      )
    return (b, m, n)

  m, k = lhs.shape
  group_count, rk, n = rhs.shape
  if k != rk:
    raise TypeError(f"ragged_dot requires that lhs.shape[1] == rhs.shape[1]: got {k} and {rk}.")
  num_groups = group_sizes.shape[0]
  if group_count != num_groups:
    raise TypeError(f"ragged_dot requires that rhs.shape[0] == group_sizes.shape[0]: got {group_count} and {num_groups}.")
  return (m, n)

def _ragged_dot_dtype_rule(lhs: Array, rhs: Array, group_sizes: Array,
                           precision, preferred_element_type: DTypeLike | None,
                           **_) -> np.dtype:
  if not dtypes.issubdtype(group_sizes.dtype, np.integer):
    raise TypeError("ragged_dot requires that group_sizes.dtype is subtype of np.integer.")
  # defer the output dtype to dot_general, which is part of the _ragged_dot_impl.
  return _dot_general_dtype_rule(
      lhs, rhs, dimension_numbers=_RAGGED_DOT_DOT_DIMENSION_NUMBERS,
      precision=precision, preferred_element_type=preferred_element_type,
      out_sharding=None)


def _ragged_dot_jvp_rule(
    primals, tangents, precision, preferred_element_type, group_offset
):
  # note - we could ostensibly just get this by passing on the
  # value to ragged_dot below, but, this feels cleaner.
  if group_offset is not None:
    raise NotImplementedError('Unimplemented group_offset support.')
  x, y, gs = primals
  dx, dy, _ = tangents  # no tan on the gs

  # primal
  primal_out = ragged_dot(
      x,
      y,
      gs,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )

  # tangent
  dx_out = (
      ragged_dot(
          dx,
          y,
          gs,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
      if type(dx) is not ad_util.Zero
      else _zeros(primal_out)
  )
  dy_out = (
      ragged_dot(
          x,
          dy,
          gs,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
      if type(dy) is not ad_util.Zero
      else _zeros(primal_out)
  )
  tangent_out = dx_out + dy_out

  return primal_out, tangent_out


def _ragged_to_dense(x, y, group_sizes):
  from jax._src.lax import control_flow  # avoid circular imports
  shape = (y.shape[0], x.shape[0], x.shape[1])
  x = broadcast_in_dim(x, shape, [1, 2])
  iota = broadcasted_iota(group_sizes.dtype, shape, 1)
  group_ends = control_flow.cumsum(group_sizes)
  group_starts = concatenate(
      [_zeros(group_sizes)[:1], group_ends[:-1]],
      dimension=0,
  )
  group_ends = broadcast_in_dim(group_ends, shape, (0,))
  group_starts = broadcast_in_dim(group_starts, shape, (0,))
  mask = bitwise_and(group_starts <= iota, iota < group_ends)
  x = select(mask, x, _zeros(x))
  return x


def _ragged_dot_transpose_rule(
    ct, *operands, precision, preferred_element_type, group_offset
):
  x, y, gs = operands
  if group_offset is not None:
    raise NotImplementedError('Unimplemented group_offset support.')

  if ad.is_undefined_primal(y):
    grad_x = None
  else:
    y_t = _matrix_transpose(y)
    grad_x = ragged_dot(
        ct,
        y_t,
        gs,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  if ad.is_undefined_primal(x):
    grad_y = None
  else:
    y = y.aval if ad.is_undefined_primal(y) else y
    x_dense = _ragged_to_dense(x, y, group_sizes=gs)
    ct_dense = _ragged_to_dense(ct, y, group_sizes=gs)
    dimension_numbers = (([1], [1]), ([0], [0]))
    grad_y = dot_general(
        x_dense,
        ct_dense,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  return grad_x, grad_y, None


def _ragged_dot_batch_unpack_args(batched_args):
  lhs, rhs, _ = batched_args
  return (lhs, rhs)


def _ragged_dot_batch_unpack_dims(batch_dims):
  if not all(dim == 0 for dim in batch_dims):
    raise NotImplementedError('ragged_dot vmap over any dim but 0 - NYI')
  lbd, rbd, _ = batch_dims
  return (lbd, rbd)


def _ragged_dot_invoke_prim(
    group_sizes,
    lhs,
    rhs,
    new_dimension_numbers,
    precision,
    preferred_element_type,
    out_sharding,
):
  del out_sharding
  return ragged_dot(
      lhs,
      rhs,
      group_sizes,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )


def _ragged_dot_batch_rule(
    batched_args,
    batch_dims,
    *,
    precision,
    preferred_element_type: DTypeLike | None,
    **_,
):
  invoke = functools.partial(_ragged_dot_invoke_prim, batched_args[2])

  return _dot_batch_rule(
      _ragged_dot_batch_unpack_args,
      _ragged_dot_batch_unpack_dims,
      invoke,
      batched_args,
      batch_dims,
      dimension_numbers=_RAGGED_DOT_DOT_DIMENSION_NUMBERS,
      precision=precision,
      preferred_element_type=preferred_element_type,
      out_sharding=None,
  )


ragged_dot_p = standard_primitive(_ragged_dot_shape_rule,
                                  _ragged_dot_dtype_rule, 'ragged_dot')
ragged_dot_p.def_impl(partial(dispatch.apply_primitive, ragged_dot_p))
ad.primitive_jvps[ragged_dot_p] = _ragged_dot_jvp_rule
ad.primitive_transposes[ragged_dot_p] = _ragged_dot_transpose_rule
batching.primitive_batchers[ragged_dot_p] = _ragged_dot_batch_rule

def _ragged_dot_impl(
    lhs: Array,
    rhs: Array,
    group_sizes: Array,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    group_offset: Array | None = None,
    ) -> Array:
  if group_offset is not None:
    raise NotImplementedError("Unimplemented group_offset support.")

  if len(lhs.shape) == 3:
    ragged_dot_dims = _RAGGED_DOT_BATCH_DOT_DIMENSION_NUMBERS
    ragged_to_dense = api.vmap(_ragged_to_dense, in_axes=(0, 0, 0))
  else:
    ragged_dot_dims = _RAGGED_DOT_DOT_DIMENSION_NUMBERS
    ragged_to_dense = _ragged_to_dense

  lhs = ragged_to_dense(lhs, rhs, group_sizes)

  return dot_general(
      lhs,
      rhs,
      dimension_numbers=ragged_dot_dims,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )

mlir.register_lowering(ragged_dot_p, mlir.lower_fun(_ragged_dot_impl, multiple_results=False))


def _broadcast_in_dim_shape_rule(operand, *, shape, broadcast_dimensions,
                                 sharding):
  _check_shapelike('broadcast_in_dim', 'shape', shape)
  _check_shapelike('broadcast_in_dim', 'broadcast_dimensions',
                   broadcast_dimensions)
  operand_ndim = np.ndim(operand)
  if operand_ndim != len(broadcast_dimensions):
    msg = ('broadcast_in_dim broadcast_dimensions must have length equal to '
           'operand ndim; got broadcast_dimensions {} for operand ndim {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand_ndim))
  if len(shape) < operand_ndim:
    msg = ('broadcast_in_dim target broadcast shape must have equal or higher rank '
           'to the operand shape; got operand ndim {} and target broadcast ndim {}.')
    raise TypeError(msg.format(operand_ndim, len(shape)))
  if not set(broadcast_dimensions).issubset(set(range(len(shape)))):
    msg = ('broadcast_in_dim broadcast_dimensions must be a subset of output '
           'dimensions, got {} for operand ndim {} and shape {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand_ndim, shape))
  if not all(core.definitely_equal_one_of_dim(operand.shape[i],
                                              [1, shape[broadcast_dimensions[i]]])
             for i in range(operand_ndim)):
    msg = (
        "broadcast_in_dim operand dimension sizes must either be 1, or be "
        "equal to their corresponding dimensions in the target broadcast "
        "shape; got operand of shape {}, target broadcast shape {}, "
        "broadcast_dimensions {} ")
    raise TypeError(msg.format(
        tuple(core.replace_tracer_for_error_message(d) for d in operand.shape),
        shape, broadcast_dimensions))
  if (len(broadcast_dimensions) != len(set(broadcast_dimensions)) or
      tuple(broadcast_dimensions) != tuple(sorted(broadcast_dimensions))):
    msg = ("broadcast_in_dim broadcast_dimensions must be strictly increasing; "
           "got broadcast_dimensions {}")
    raise TypeError(msg.format(broadcast_dimensions))
  return shape

def _broadcast_in_dim_sharding_rule(operand, *, shape, broadcast_dimensions,
                                    sharding):
  if sharding is not None:
    return sharding
  bds = set(broadcast_dimensions)
  orig_spec = iter(operand.sharding.spec)
  new_spec = [next(orig_spec) if i in bds else None for i in range(len(shape))]
  assert next(orig_spec, None) is None
  return operand.sharding.with_spec(new_spec)

def _broadcast_in_dim_typecheck_rule(
    _, operand, *dyn_shape, shape, broadcast_dimensions, sharding):
  if not dyn_shape:
    out_aval, effects = broadcast_in_dim_p.abstract_eval(
        operand.aval, shape=shape, broadcast_dimensions=broadcast_dimensions,
        sharding=sharding)
    return [out_aval], effects
  else:
    # TODO(mattjj): perform more checks like _broadcast_in_dim_shape_rule
    out_shape = _merge_dyn_shape(shape, dyn_shape)
    out_shape = [x.val if type(x) is core.Literal else x for x in out_shape]  # pytype: disable=attribute-error
    out_aval = core.DShapedArray(tuple(out_shape), operand.aval.dtype,
                                 operand.aval.weak_type)
    return [out_aval], core.no_effects

def _broadcast_in_dim_transpose_rule(ct, operand, *dyn_shape,
                                     shape, broadcast_dimensions, sharding):
  if type(ct) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)]
  unit_dims = [i for i, s in enumerate(operand.aval.shape)
               if core.definitely_equal(s, 1)]
  bdims = tuple(np.delete(broadcast_dimensions, unit_dims))
  axes = tuple(np.delete(range(len(shape)), bdims))
  return ([expand_dims(_reduce_sum(ct, axes), unit_dims)] +
          [None] * len(dyn_shape))

def _broadcast_in_dim_batch_rule(batched_args, batch_dims, shape,
                                 broadcast_dimensions, sharding):
  # `dyn_shape` is the dynamic portion of the target shape.  `shape`
  # is the target shape, with `None` for dynamic sections.
  # broadcast_dimensions gives indices where dimensions of the input
  # have to go: dimension i of the input becomes dimension
  # broadcast_dimensions[i] of the output.
  operand, *dyn_shape = batched_args
  operand_bdim, *dyn_shape_bdims = batch_dims

  stacked_size = None
  if operand_bdim is not None:
    if isinstance(operand_bdim, RaggedAxis):
      stacked_axis = operand_bdim.stacked_axis
    else:
      stacked_axis = operand_bdim
    new_operand = batching.moveaxis(operand, stacked_axis, 0)
    if isinstance(operand_bdim, RaggedAxis):
      stacked_size = operand_bdim.size
    else:
      stacked_size = operand.shape[stacked_axis]
    new_broadcast_dimensions = (0,) + tuple(np.add(1, broadcast_dimensions))
  else:
    new_operand = operand
    new_broadcast_dimensions = tuple(np.add(1, broadcast_dimensions))

  # TODO(mattjj,axch) This section assumes that the shape of the operand is
  # broadcast-compatible with the requested shape.  We should tweak vmap to run
  # the abstract_eval rule so this can be checked while the raggedness
  # information is available.
  dyn_limits = []
  out_ragged_sizes = []
  for sizes, bdim in zip(dyn_shape, dyn_shape_bdims):
    if bdim is None:
      # TODO(mattjj,axch) Is this what bdim == None means?
      assert isinstance(sizes, int)
      bound = sizes
    else:
      bound = sizes.dtype.bound
      out_ragged_sizes.append(sizes)
      if stacked_size is None:
        stacked_size = len(sizes)
      else:
        msg = "All segments lengths arrays must be the same length"
        assert len(sizes) == stacked_size, msg
    dyn_limits.append(bound)
  new_shape = (stacked_size,) + _merge_dyn_shape(shape, dyn_limits)
  if sharding is not None:
    raise NotImplementedError('Implement broadcast_in_dim_batch_rule')
  result = broadcast_in_dim(new_operand, new_shape, new_broadcast_dimensions)
  out_ragged_axes = [idx+1 for idx, s in enumerate(shape) if s is None]
  out_bdim = batching.make_batch_axis(
      result.ndim, 0, zip(out_ragged_axes, out_ragged_sizes))
  return result, out_bdim

def _broadcast_in_dim_fwd_rule(eqn):
  v, *dyn = eqn.invars
  if not dyn and core.definitely_equal_shape(eqn.params['shape'], v.aval.shape):
    return [v], None
  else:
    return [None], eqn

def _broadcast_in_dim_staging_rule(
    trace, x, *dyn, shape, broadcast_dimensions, sharding):
  params = dict(shape=shape, broadcast_dimensions=broadcast_dimensions,
                sharding=sharding)
  if not dyn:
    return trace.default_process_primitive(broadcast_in_dim_p, (x,), params)
  aval = core.DShapedArray(_merge_dyn_shape(shape, dyn), x.dtype, x.weak_type)
  return _dyn_shape_staging_rule(trace, broadcast_in_dim_p, aval, x, *dyn,
                                 **params)

def _broadcast_in_dim_padding_rule(in_avals, out_avals, x, *dyn_shape,
                                   shape, broadcast_dimensions):
  del in_avals, dyn_shape
  out_aval, = out_avals
  new_shape = []
  new_dyn_shape = []
  for d in out_aval.shape:
    if type(d) is pe.BoundedAxisSize:
      new_shape.append(d.bound)
    elif type(d) is int:
      new_shape.append(d)
    else:
      assert isinstance(d, core.Tracer)
      new_shape.append(None)
      new_dyn_shape.append(d)
  return [broadcast_in_dim_p.bind(x, *new_dyn_shape, shape=tuple(new_shape),
                                  broadcast_dimensions=broadcast_dimensions)]

def _broadcast_in_dim_jvp_rule(primals, tangents, *, shape, broadcast_dimensions,
                               sharding):
  operand, *dyn_shape = primals
  operand_dot, *_ = tangents
  y = broadcast_in_dim_p.bind(operand, *dyn_shape, shape=shape,
                              broadcast_dimensions=broadcast_dimensions,
                              sharding=sharding)
  if type(operand_dot) is ad_util.Zero:
    y_dot = ad_util.Zero.from_primal_value(y)
  else:
    y_dot = broadcast_in_dim_p.bind(operand_dot, *dyn_shape, shape=shape,
                                    broadcast_dimensions=broadcast_dimensions,
                                    sharding=sharding)
  return y, y_dot

def _broadcast_in_dim_partial_eval(
    trace, operand, *dyn_shape, shape, broadcast_dimensions, sharding):
  if not dyn_shape:
    return trace.default_process_primitive(
        broadcast_in_dim_p, (operand, *dyn_shape),
        dict(shape=shape, broadcast_dimensions=broadcast_dimensions,
             sharding=sharding))
  assert all(t.pval.is_known() for t in dyn_shape)
  operand_tracer = trace.instantiate_const(operand)
  dyn_shape_tracers = map(trace.instantiate_const, dyn_shape)
  dyn_shape_tracers_ = iter(dyn_shape_tracers)
  shape_ = [next(dyn_shape_tracers_) if d is None else d for d in shape]
  out_aval = core.DShapedArray(tuple(shape_), operand.dtype, operand.weak_type)
  out_tracer = pe.JaxprTracer(trace, pe.PartialVal.unknown(out_aval), None)
  eqn = pe.new_eqn_recipe(
      [operand_tracer, *dyn_shape_tracers], [out_tracer], broadcast_in_dim_p,
      dict(shape=shape, broadcast_dimensions=broadcast_dimensions,
           sharding=None),
      core.no_effects, source_info_util.current())
  out_tracer.recipe = eqn
  return out_tracer

def _broadcast_in_dim_lower(ctx, x, *dyn_shape, shape, broadcast_dimensions,
                            sharding) -> Sequence[ir.Value]:
  aval_out, = ctx.avals_out
  if dyn_shape:
    aval_out = aval_out.update(shape=_merge_dyn_shape(shape, dyn_shape))
  out = mlir.broadcast_in_dim(ctx, x, aval_out,
                              broadcast_dimensions=broadcast_dimensions)
  if config.sharding_in_types.value:
    if sharding is not None:
      assert sharding == aval_out.sharding
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

def _broadcast_in_dim_abstract_eval(x, *dyn_shape, shape, broadcast_dimensions,
                                    sharding):
  if (not dyn_shape and
      not any(isinstance(d, core.DArray) and
              type(core.get_aval(d).dtype) is core.bint for d in shape)):
    shape = _broadcast_in_dim_shape_rule(  # error checking
        x, shape=shape, broadcast_dimensions=broadcast_dimensions, sharding=None)
    if config.sharding_in_types.value:
      new_sharding = _broadcast_in_dim_sharding_rule(
          x, shape=shape, broadcast_dimensions=broadcast_dimensions,
          sharding=sharding)
    else:
      new_sharding = None
    return core.ShapedArray(shape, x.dtype, x.weak_type, sharding=new_sharding)
  # If any BInts in shape, or Tracers in dyn_shape, produce a DShapedArray
  # (even if x is a ShapedArray)
  # TODO(mattjj): unify DShapedArray with ShapedArray, and remove this code
  return core.DShapedArray(_merge_dyn_shape(shape, dyn_shape), x.dtype, x.weak_type)


def _broadcast_in_dim_ragged_prop_rule(eqn_params, invar_raggedness, outvars):
  assert len(invar_raggedness) == 1
  assert not isinstance(invar_raggedness[0], core.Var)
  return invar_raggedness, [None] * len(outvars)


broadcast_in_dim_p = standard_primitive(
    _broadcast_in_dim_shape_rule, _input_dtype, 'broadcast_in_dim')
broadcast_in_dim_p.def_abstract_eval(_broadcast_in_dim_abstract_eval)
ad.primitive_jvps[broadcast_in_dim_p] = _broadcast_in_dim_jvp_rule
ad.primitive_transposes[broadcast_in_dim_p] = _broadcast_in_dim_transpose_rule
batching.primitive_batchers[broadcast_in_dim_p] = _broadcast_in_dim_batch_rule
pe.forwarding_rules[broadcast_in_dim_p] = _broadcast_in_dim_fwd_rule
pe.custom_partial_eval_rules[broadcast_in_dim_p] = _broadcast_in_dim_partial_eval
pe.custom_staging_rules[broadcast_in_dim_p] = _broadcast_in_dim_staging_rule
pe.padding_rules[broadcast_in_dim_p] = _broadcast_in_dim_padding_rule
core.custom_typechecks[broadcast_in_dim_p] = _broadcast_in_dim_typecheck_rule
mlir.register_lowering(broadcast_in_dim_p, _broadcast_in_dim_lower)
batching.ragged_prop_rules[broadcast_in_dim_p] = (
    _broadcast_in_dim_ragged_prop_rule
)


def _clamp_shape_rule(min, operand, max):
  if min.shape and min.shape != operand.shape:
    raise TypeError("clamp requires min.shape == operand.shape or min.shape == "
                    f"(), got min.shape={min.shape}, {operand.shape=}.")
  if max.shape and max.shape != operand.shape:
    raise TypeError("clamp requires max.shape == operand.shape or max.shape == "
                    f"(), got max.shape={max.shape}, {operand.shape=}.")
  return operand.shape

_clamp_dtype_rule = partial(naryop_dtype_rule, _input_dtype, [_any, _any, _any],
                            'clamp')

def _clamp_batch_rule(batched_args, batch_dims, **params):
  min, x, max = batched_args
  min_bdim, x_bdim, max_bdim = batch_dims
  size = next(x.shape[i] for x, i in zip(batched_args, batch_dims)
              if i is not None)

  # avoid transposes and some broadcasts in special cases
  if min_bdim == x_bdim == max_bdim:
    if np.shape(min) == np.shape(x) == np.shape(max):
      return clamp_p.bind(min, x, max), x_bdim
    elif np.ndim(min) == np.ndim(max) == 0:
      return clamp_p.bind(min, x, max), x_bdim
    elif np.ndim(min) == np.ndim(max) == 1:
      min = broadcast_in_dim(min, x.shape, [min_bdim])
      max = broadcast_in_dim(max, x.shape, [max_bdim])
      return clamp_p.bind(min, x, max), x_bdim
  elif np.ndim(min) == 0 and np.ndim(max) == 0 and x_bdim is not None:
    return clamp_p.bind(min, x, max), x_bdim

  min = batching.bdim_at_front(min, min_bdim, size) if np.shape(min) else min
  max = batching.bdim_at_front(max, max_bdim, size) if np.shape(max) else max
  x = batching.bdim_at_front(x, x_bdim, size) if np.shape(x) else x
  if np.ndim(min) == 0 and np.ndim(x) > 0:
    min = broadcast(min, x.shape)
  if np.ndim(max) == 0 and np.ndim(x) > 0:
    max = broadcast(max, x.shape)
  if 0 < np.ndim(min) < np.ndim(x):
    assert np.ndim(min) == 1, np.ndim(min)
    min = broadcast_in_dim(min, x.shape, [0])
  if 0 < np.ndim(max) < np.ndim(x):
    assert np.ndim(max) == 1, np.ndim(max)
    max = broadcast_in_dim(max, x.shape, [0])
  if np.ndim(min) > np.ndim(x):
    assert np.ndim(x) == 0, np.ndim(x)
    x = broadcast(x, min.shape)
  return clamp_p.bind(min, x, max), 0

clamp_p = standard_primitive(_clamp_shape_rule, _clamp_dtype_rule, 'clamp')
ad.defjvp(clamp_p,
          lambda g, min, operand, max:
          select(bitwise_and(gt(min, operand), lt(min, max)),
                 g, _zeros(operand)),
          lambda g, min, operand, max:
          select(bitwise_and(gt(operand, min), lt(operand, max)),
                 g, _zeros(operand)),
          lambda g, min, operand, max:
          select(lt(max, operand), g, _zeros(operand)))
batching.primitive_batchers[clamp_p] = _clamp_batch_rule
mlir.register_lowering(clamp_p, partial(_nary_lower_hlo, hlo.clamp))
pe.def_trivial_padding(clamp_p)

def _concatenate_shape_rule(*operands, **kwargs):
  dimension = kwargs.pop('dimension')
  if not operands:
    msg = "concatenate expects at least one operand, got 0."
    raise TypeError(msg)
  if not all(isinstance(operand, UnshapedArray) for operand in operands):
    msg = "All objects to concatenate must be arrays, got {}."
    op = next(op for op in operands if not isinstance(op, UnshapedArray))
    raise TypeError(msg.format(type(op)))
  if len({operand.ndim for operand in operands}) != 1:
    msg = "Cannot concatenate arrays with different numbers of dimensions: got {}."
    raise TypeError(msg.format(", ".join(str(o.shape) for o in operands)))
  if not 0 <= dimension < operands[0].ndim:
    msg = "concatenate dimension out of bounds: dimension {} for shapes {}."
    raise TypeError(msg.format(dimension, ", ".join([str(o.shape) for o in operands])))
  shapes = [operand.shape[:dimension] + operand.shape[dimension+1:]
            for operand in operands]
  if shapes[:-1] != shapes[1:]:
    msg = ("Cannot concatenate arrays with shapes that differ in dimensions "
           "other than the one being concatenated: concatenating along "
           "dimension {} for shapes {}.")
    shapes = [operand.shape for operand in operands]
    raise TypeError(msg.format(dimension, ", ".join(map(str, shapes))))

  concat_size = sum(o.shape[dimension] for o in operands)
  ex_shape = operands[0].shape
  return ex_shape[:dimension] + (concat_size,) + ex_shape[dimension+1:]

def _concatenate_sharding_rule(*operands, **kwargs):
  if not all(o.sharding == operands[0].sharding for o in operands):
    ss = ", ".join(str(o.sharding) for o in operands)
    raise TypeError(
        f"All operands should have the same sharding. Got shardings {ss}")
  return operands[0].sharding

def _concatenate_dtype_rule(*operands, **kwargs):
  check_same_dtypes('concatenate', *operands)
  return operands[0].dtype

def _concatenate_transpose_rule(t, *operands, dimension):
  operand_shapes = [o.aval.shape if ad.is_undefined_primal(o) else o.shape
                    for o in operands]
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(o.aval) if ad.is_undefined_primal(o) else None
            for o in operands]
  else:
    return split(t, tuple(shape[dimension] for shape in operand_shapes),
                 axis=dimension)

def _concatenate_batch_rule(batched_args, batch_dims, *, dimension):
  size = next(op.shape[bdim] for op, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  operands = [batching.moveaxis(op, bdim, 0) if bdim is not None
              else broadcast(op, (size,))
              for op, bdim in zip(batched_args, batch_dims)]
  return concatenate(operands, dimension + 1), 0

def _concatenate_pad_rule(in_avals, out_avals, *operands, dimension):
  if all(isinstance(a.shape[dimension], (int, np.integer))
         for a in in_avals):
    return [concatenate(operands, dimension)]
  else:
    raise NotImplementedError  # TODO(mattjj)

concatenate_p = standard_primitive(
    _concatenate_shape_rule, _concatenate_dtype_rule, 'concatenate',
    sharding_rule=_concatenate_sharding_rule)
ad.deflinear2(concatenate_p, _concatenate_transpose_rule)
ad.primitive_transposes[concatenate_p] = _concatenate_transpose_rule
batching.primitive_batchers[concatenate_p] = _concatenate_batch_rule
pe.padding_rules[concatenate_p] = _concatenate_pad_rule

def _concatenate_lower(ctx, *xs, dimension):
  aval_out, = ctx.avals_out
  out = hlo.concatenate(xs, mlir.i64_attr(dimension))
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]
mlir.register_lowering(concatenate_p, _concatenate_lower)


def _split_shape_rule(operand, *, sizes, axis):
  shapes = []
  shape = list(operand.shape)
  if any(s < 0 for s in sizes):
    raise ValueError(
      f"Sizes passed to split must be nonnegative, got {list(sizes)}")
  if operand.shape[axis] != np.sum(sizes):
    raise ValueError(
      f"Sum of sizes {np.sum(sizes)} must be equal to dimension {axis} of the "
      f"operand shape {list(operand.shape)}")
  for size in sizes:
    shape[axis] = size
    shapes.append(tuple(shape))
  return shapes

def _split_dtype_rule(operand, *, sizes, axis):
  return (operand.dtype,) * len(sizes)

def _split_weak_type_rule(operand, *, sizes, axis):
  return (operand.weak_type,) * len(sizes)

def _split_transpose_rule(cotangents, operand, *, sizes, axis):
  assert ad.is_undefined_primal(operand)
  if all(type(t) is ad_util.Zero for t in cotangents):
    return ad_util.Zero(operand.aval),
  cotangents = [
    _zeros(t.aval) if type(t) is ad_util.Zero else t
    for t in cotangents
  ]
  return concatenate(cotangents, dimension=axis),

def _split_batch_rule(batched_args, batch_dims, *, sizes, axis):
  operand, = batched_args
  bdim, = batch_dims
  new_bdims = (bdim,) * len(sizes)
  out = split(operand, sizes=sizes, axis=axis + 1 if axis >= bdim else axis)
  return out, new_bdims

def _split_lower(ctx, x, *, sizes, axis):
  x_aval, = ctx.avals_in
  start_indices = [0] * x_aval.ndim
  limit_indices = list(x_aval.shape)
  strides = (1,) * x_aval.ndim
  outs = []
  for aval_out in ctx.avals_out:
    limit_indices[axis] = start_indices[axis] + aval_out.shape[axis]
    out = mlir.slice_op(ctx, x, aval_out, start_indices=start_indices,
                        limit_indices=limit_indices, strides=strides)
    outs.append(mlir.lower_sharding_under_shit(ctx, out, aval_out)
                if config.sharding_in_types.value else out)
    start_indices[axis] = limit_indices[axis]
  return outs

def _split_sharding_rule(operand, *, sizes, axis):
  # TODO(yashkatariya): Once JAX supports uneven sharding at the top level,
  # change this logic to `return operand.sharding` directly.
  out_shapes = _split_shape_rule(operand, sizes=sizes, axis=axis)
  return [slicing._get_sharding_for_varying_out_shape(out_sh, operand, 'split')
          for out_sh in out_shapes]

split_p = core.Primitive('split')
split_p.multiple_results = True
split_p.def_abstract_eval(
    partial(standard_multi_result_abstract_eval, split_p, _split_shape_rule,
            _split_dtype_rule, _split_weak_type_rule, _split_sharding_rule))
split_p.def_impl(partial(dispatch.apply_primitive, split_p))
ad.deflinear2(split_p, _split_transpose_rule)
batching.primitive_batchers[split_p] = _split_batch_rule
mlir.register_lowering(split_p, _split_lower)

def _pad_dtype_rule(operand, padding_value, *, padding_config):
  if operand.dtype != padding_value.dtype:
    msg = "pad operand and padding_value must be same dtype: got {} and {}."
    raise TypeError(msg.format(operand.dtype, padding_value.dtype))

  return _input_dtype(operand, padding_value)

def _pad_shape_rule(operand, padding_value, *, padding_config):
  if np.ndim(padding_value) != 0:
    raise ValueError(f"padding_value must be a scalar; got {np.shape(padding_value)=}")
  op_shape = np.shape(operand)
  if not len(padding_config) == np.ndim(operand):
    raise ValueError("length of padding_config must equal the number of axes "
                     f"of operand, got padding_config {padding_config} "
                     f"for operand shape {op_shape}")
  if not all(i >= 0 for _, _, i in padding_config):
    raise ValueError("interior padding in padding_config must be nonnegative, "
                     f"got padding_config {padding_config}")
  result = tuple(l + h + core.dilate_dim(d, i + 1)
                 for (l, h, i), d in zip(padding_config, op_shape))
  if not all(d >= 0 for d in result):
    msg = (f"Dimension size after padding is not at least 0, "
           f"got result shape {result}, for padding_config {padding_config}"
           f" and operand shape {op_shape}")
    raise ValueError(msg)
  return result

def _pad_sharding_rule(operand, padding_value, *, padding_config):
  # TODO(yashkatariya): Once JAX supports uneven sharding at the top level,
  # change this logic to `return operand.sharding` directly.
  out_shape = _pad_shape_rule(operand, padding_value,
                              padding_config=padding_config)
  return slicing._get_sharding_for_varying_out_shape(
      out_shape, operand, 'padding')


def _pad_transpose(t, operand, padding_value, *, padding_config):
  if type(t) is ad_util.Zero:
    t_operand = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    t_padv = ad_util.Zero(padding_value.aval) if ad.is_undefined_primal(padding_value) else None
  else:
    lo, hi, interior = util.unzip3(padding_config)
    total = lambda x: _reduce_sum(x, list(range(t.ndim)))

    def t_op():
      unpad_config = safe_zip(np.negative(lo), np.negative(hi),
                              np.zeros_like(interior))
      unpadded = pad(t, np.array(0., t.dtype), unpad_config)
      return slicing.slice(unpadded, np.zeros_like(lo), unpadded.shape,
                           np.add(interior, 1))

    t_operand = t_op() if ad.is_undefined_primal(operand) else None
    t_padv = sub(total(t), total(t_operand)) if ad.is_undefined_primal(padding_value) else None
  return [t_operand, t_padv]

def _pad_batch_rule(batched_args, batch_dims, *, padding_config):
  operand, padding_value = batched_args
  operand_bdim, padding_value_bdim = batch_dims
  if operand_bdim is None:
    operand_bdim = 0
    operand = broadcast(operand, (padding_value.shape[padding_value_bdim],))

  padding_config = list(padding_config)
  padding_config.insert(operand_bdim, (0, 0, 0))
  if padding_value_bdim is None:
    return pad(operand, padding_value, padding_config), operand_bdim

  assert padding_value_bdim == 0, padding_value_bdim

  x = pad(operand, _zero(operand), padding_config)
  mask = pad(full_like(operand, True, np.bool_), False, padding_config)
  broadcasted_padding = broadcast_in_dim(padding_value, x.shape,
                                         (operand_bdim,))
  return select(mask, x, broadcasted_padding), operand_bdim

pad_p = standard_primitive(_pad_shape_rule, _pad_dtype_rule, 'pad',
                           sharding_rule=_pad_sharding_rule)
ad.deflinear2(pad_p, _pad_transpose)
batching.primitive_batchers[pad_p] = _pad_batch_rule

def _pad_lower(ctx, x, padding_value, *, padding_config):
  aval_out, = ctx.avals_out
  low, high, interior = util.unzip3(padding_config)
  out = mlir.pad(ctx, aval_out, x, padding_value, low, high, interior)
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

mlir.register_lowering(pad_p, _pad_lower)


# The squeeze primitive exists for the benefit of masking and other
# transformations that need to keep track of axis identity.
# For example, consider reshaping a 2D array with shape (1, N) into a 1D array
# with shape (N,). This results in the following JAXpr:
#   reshape[ dimension=None new_sizes=(N,) ]
# For N > 1, we can match up the output array axis with the second axis of the
# input. But for N = 1, it is not clear how axes match up: all we know from the
# JAXpr is that we are reshaping from (1, 1) to (1,).
# In contrast, squeeze[ dimensions=(0,) ] is unambiguous.


def _squeeze_dtype_rule(operand, *, dimensions):
  return operand.dtype

def _squeeze_shape_rule(operand, *, dimensions):
  return _compute_squeeze_shape(np.shape(operand), dimensions)

def _squeeze_sharding_rule(operand, *, dimensions):
  dims_set = set(dimensions)
  new_spec = tuple(s for i, s in enumerate(operand.sharding.spec)
                   if i not in dims_set)
  return operand.sharding.with_spec(new_spec)

def _compute_squeeze_shape(shape, dimensions):
  dims_set = set(dimensions)
  if len(dims_set) != len(dimensions):
    raise ValueError(f"dimensions are not unique: {dimensions}")
  if not all(0 <= d < len(shape) for d in dims_set):
    raise ValueError(f"dimensions outside range [0, ndim): {dimensions}")
  if any(not core.definitely_equal(shape[d], 1) for d in dimensions):
    raise ValueError(
        "cannot select an axis to squeeze out which has size not equal to "
        f"one, got {shape=} and {dimensions=}")
  return tuple(s for i, s in enumerate(shape) if i not in dims_set)

def _squeeze_transpose_rule(t, operand, *, dimensions):
  assert ad.is_undefined_primal(operand)
  return [expand_dims(t, dimensions)]

def _squeeze_batch_rule(batched_args, batch_dims, *, dimensions):
  operand, = batched_args
  bdim, = batch_dims
  operand, bdim = batching.move_stacked_axis(operand, bdim, 0)
  dimensions = tuple(np.add(1, dimensions))
  out_stack_dim = bdim.stacked_axis if isinstance(bdim, RaggedAxis) else bdim
  bdim_out = batching.shape_as_bdim(
      out_stack_dim,
      _compute_squeeze_shape(batching.bdim_as_shape(bdim, operand.shape), dimensions))
  return squeeze(operand, dimensions=dimensions), bdim_out

squeeze_p = standard_primitive(_squeeze_shape_rule, _squeeze_dtype_rule,
                               'squeeze', sharding_rule=_squeeze_sharding_rule)
ad.deflinear2(squeeze_p, _squeeze_transpose_rule)
batching.primitive_batchers[squeeze_p] = _squeeze_batch_rule
pe.def_trivial_padding(squeeze_p)
batching.ragged_prop_rules[squeeze_p] = batching.ragged_mask_no_op_rule

def _squeeze_lower(ctx, operand, *, dimensions):
  del dimensions  # Implied by the output aval.
  aval_out, = ctx.avals_out
  out = mlir.reshape(ctx, operand, aval_out)
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

mlir.register_lowering(squeeze_p, _squeeze_lower)


def shape_as_value(shape: core.Shape):
  """Converts a shape that may contain Poly values into a JAX value."""
  if len(shape) == 0:
    return full((0,), np.array(0, np.int64))
  if core.is_constant_shape(shape):
    return np.asarray(shape, dtype=np.int64)
  dims = [
      expand_dims(convert_element_type(core.dimension_as_value(d), np.int64),
                  (0,))
      for d in shape
  ]
  return concatenate(dims, dimension=0)

def _reshape_shape_rule(operand, *, new_sizes, dimensions, sharding):
  if not all(d >= 0 for d in new_sizes):
    msg = 'reshape new_sizes must all be positive, got {}.'
    raise TypeError(msg.format(new_sizes))
  # TODO(necula): re-enable this check
  operand_size = math.prod(np.shape(operand))
  new_size = math.prod(new_sizes)
  if (not config.dynamic_shapes.value and
      not operand_size == new_size):
    msg = (f"reshape total size must be unchanged, got new_sizes {new_sizes} "
           f"(of total size {new_size}) for shape {np.shape(operand)} "
           f"(of total size {operand_size}).")
    raise TypeError(msg)
  if dimensions is not None:
    if set(dimensions) != set(range(np.ndim(operand))):
      msg = ('reshape dimensions must be a permutation of operand dimensions, '
             'got dimensions {} for shape {}.')
      raise TypeError(msg.format(dimensions, np.shape(operand)))
  return tuple(new_sizes)

def _split_on_one_axis(op_shape, new_sizes, name):
  if len(new_sizes) <= len(op_shape):
    return False, []
  i, j, count, out = 0, 0, 0, []
  while j < len(new_sizes):
    if op_shape[i] == new_sizes[j]:
      out.append(op_shape[i])
    else:
      count += 1
      if count > 1:
        raise ValueError(
            f'{name} on more than 1 axis is not supported. Please specify'
            ' the sharding of the output via the `sharding` argument of'
            f' jax.lax.reshape. Got operand.shape={op_shape} and {new_sizes=}')
      temp = [new_sizes[j]]
      while math.prod(temp) != op_shape[i]:
        if math.prod(temp) > op_shape[i]:
          return False, []
        j += 1
        temp.append(new_sizes[j])
      out.append(temp)
    i += 1
    j += 1
  assert len(op_shape) == len(out)
  return True, out


def _merge_on_one_axis(operand, new_sizes):
  if len(new_sizes) >= len(operand.shape):
    return False, []
  return _split_on_one_axis(new_sizes, operand.shape, 'Merging')


def _reshape_sharding_rule(operand, *, new_sizes, dimensions, sharding):
  if sharding is not None:
    return sharding
  non_1s_op_shape = [s for s in operand.shape if s != 1]
  non_1s_new_shape = [s for s in new_sizes if s != 1]
  if non_1s_op_shape == non_1s_new_shape:
    return _split_merge_singleton_dim_sharding_rule(operand, new_sizes)

  is_split, out_split = _split_on_one_axis(operand.shape, new_sizes, 'Splitting')
  if is_split:
    return _split_an_axis_sharding_rule(operand, out_split, new_sizes)

  is_merge, operand_merge = _merge_on_one_axis(operand, new_sizes)
  if is_merge:
    return _merge_an_axis_sharding_rule(operand, operand_merge, new_sizes)

  raise ValueError(
      'This reshape is not supported. Only 4 out of the box reshapes are'
      ' supported.Adding/removing singleton dims and splitting/merging without'
      ' sharded split/merged axes are supported. Please specify the sharding of'
      ' the output via the `sharding` argument of jax.lax.reshape.')

def _split_merge_singleton_dim_sharding_rule(operand, new_sizes):
  filtered_spec = [sp for sh, sp in zip(operand.shape, operand.sharding.spec)
                   if sh != 1]
  fs = iter(filtered_spec)
  new_spec = []
  for n in new_sizes:
    if n == 1:
      new_spec.append(None)
    else:
      sp = next(fs)
      new_spec.append(sp)
  return operand.sharding.with_spec(new_spec)

def _split_an_axis_sharding_rule(operand, out_split, new_sizes):
  new_spec = []
  for sh, out, sp in safe_zip(operand.shape, out_split, operand.sharding.spec):
    if isinstance(out, list):
      if sp is not None:
        raise ValueError(
            f'Split axis cannot be sharded. Got operand dim {sh} with spec'
            f' {sp}. Please specify the sharding of the output via the'
            ' `sharding` argument of jax.lax.reshape.')
      new_spec.extend([None] * len(out))
    else:
      new_spec.append(sp)
  assert len(new_spec) == len(new_sizes)
  return operand.sharding.with_spec(new_spec)


def _merge_an_axis_sharding_rule(operand, operand_merge, new_sizes):
  new_spec = []
  op_spec = iter(operand.sharding.spec)
  for op_merge in operand_merge:
    if isinstance(op_merge, list):
      sp = [next(op_spec) for _ in op_merge]
      if not all(s is None for s in sp):
        raise ValueError(
            f'Merged axis cannot be sharded. Got {sp}. Please specify the'
            ' sharding of the output via the `sharding` argument of'
            ' jax.lax.reshape.')
      new_spec.append(None)
    else:
      new_spec.append(next(op_spec))
  assert next(op_spec, None) is None
  assert len(new_spec) == len(new_sizes)
  return operand.sharding.with_spec(new_spec)


def _reshape_typecheck_rule(_, operand, *dyn_shape, new_sizes, dimensions,
                            sharding):
  if not dyn_shape:
    out_aval, effects = reshape_p.abstract_eval(
        operand.aval, new_sizes=new_sizes, dimensions=dimensions,
        sharding=sharding)
    return [out_aval], effects
  else:
    # TODO(mattjj, necula): perform more checks like _reshape_shape_rule
    out_shape = _merge_dyn_shape(new_sizes, dyn_shape)
    out_shape = [x.val if type(x) is core.Literal else x for x in out_shape]  # pytype: disable=attribute-error
    out_aval = core.DShapedArray(tuple(out_shape), operand.aval.dtype,
                                 operand.aval.weak_type)
    return [out_aval], core.no_effects


def _reshape_dtype_rule(operand, *, new_sizes, dimensions, sharding):
  return operand.dtype

def _reshape_transpose_rule(t, operand, *, new_sizes, dimensions, sharding):
  assert ad.is_undefined_primal(operand)
  if dimensions is None:
    if config.sharding_in_types.value:
      return [reshape(t, operand.aval.shape, sharding=operand.aval.sharding)]
    return [reshape(t, operand.aval.shape)]
  else:
    if config.sharding_in_types.value:
      t_s = operand.sharding.with_spec(
          tuple(map(str, np.take(operand.aval.sharding.spec, dimensions))))
    else:
      t_s = None
    return [transpose(reshape(t, np.take(operand.aval.shape, dimensions),
                              sharding=t_s),
                      np.argsort(dimensions))]

def _reshape_batch_rule(batched_args, batch_dims, *, new_sizes, dimensions,
                        sharding):
  if sharding is not None:
    raise NotImplementedError
  operand, = batched_args
  bdim, = batch_dims
  operand = batching.moveaxis(operand, bdim, 0)
  if dimensions is not None:
    dimensions = (0,) + tuple(np.add(1, dimensions))
  return reshape(operand, operand.shape[:1] + new_sizes, dimensions), 0


def _reshape_lower(ctx, x, *dyn_shape, new_sizes, dimensions, sharding):
  aval_out, = ctx.avals_out
  if dimensions is not None:
    x = hlo.transpose(x, mlir.dense_int_array(dimensions))
  if dyn_shape:
    aval_out = aval_out.update(shape=_merge_dyn_shape(new_sizes, dyn_shape))
  out = mlir.reshape(ctx, x, aval_out)
  if config.sharding_in_types.value:
    if sharding is not None:
      assert sharding == aval_out.sharding
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

def _reshape_staging_rule(
    trace, x, *dyn, new_sizes, dimensions, sharding):
  params = dict(new_sizes=new_sizes, dimensions=dimensions, sharding=sharding)
  if not dyn:
    return trace.default_process_primitive(reshape_p, (x,), params)
  av = core.DShapedArray(_merge_dyn_shape(new_sizes, dyn), x.dtype, x.weak_type)
  return _dyn_shape_staging_rule(trace, reshape_p, av, x, *dyn, **params)

reshape_p = standard_primitive(_reshape_shape_rule, _reshape_dtype_rule,
                               'reshape', sharding_rule=_reshape_sharding_rule)
ad.deflinear2(reshape_p, _reshape_transpose_rule)
batching.primitive_batchers[reshape_p] = _reshape_batch_rule
mlir.register_lowering(reshape_p, _reshape_lower)
core.custom_typechecks[reshape_p] = _reshape_typecheck_rule
pe.custom_staging_rules[reshape_p] = _reshape_staging_rule


def _rev_shape_rule(operand, *, dimensions):
  _check_shapelike('rev', 'dimensions', dimensions)
  if len(set(dimensions)) != len(dimensions):
    msg = 'rev dimensions must be unique, got {}.'
    raise TypeError(msg.format(dimensions))
  if dimensions and not _max(dimensions) < operand.ndim:
    msg = ('rev dimensions must all be less than operand ndim, got dimensions '
           '{} for operand ndim {}.')
    raise TypeError(msg.format(dimensions, operand.ndim))
  return operand.shape

def _rev_batch_rule(batched_args, batch_dims, *, dimensions):
  operand, = batched_args
  bdim, = batch_dims
  new_dimensions = [i + 1 if i >= bdim else i for i in dimensions]
  return rev(operand, new_dimensions), bdim

rev_p = standard_primitive(_rev_shape_rule, _input_dtype, 'rev')
ad.deflinear2(rev_p, lambda t, _, dimensions: [rev(t, dimensions)])
batching.primitive_batchers[rev_p] = _rev_batch_rule

def _rev_lower(ctx, x, *, dimensions):
  return [hlo.reverse(x, mlir.dense_int_array(dimensions))]
mlir.register_lowering(rev_p, _rev_lower)


def _transpose_shape_rule(operand, *, permutation):
  if not isinstance(permutation, (tuple, list, np.ndarray)):
    msg = "transpose permutation must be a tuple/list/ndarray, got {}."
    raise TypeError(msg.format(type(permutation)))
  if tuple(sorted(permutation)) != tuple(range(operand.ndim)):
    msg = ("transpose permutation isn't a permutation of operand dimensions, "
           "got permutation {} for operand shape {}.")
    raise TypeError(msg.format(permutation, operand.shape))
  return tuple(operand.shape[old_idx] for old_idx in permutation)

def _transpose_sharding_rule(operand, *, permutation):
  o_spec = operand.sharding.spec
  new_spec = [o_spec[old_idx] for old_idx in permutation]
  return operand.sharding.with_spec(new_spec)

def _transpose_batch_rule(batched_args, batch_dims, *, permutation):
  operand, = batched_args
  bdim, = batch_dims
  stack_dim = bdim.stacked_axis if isinstance(bdim, RaggedAxis) else bdim
  perm = (stack_dim,) + tuple(i if i < stack_dim else i+1 for i in permutation)
  if isinstance(bdim, RaggedAxis):
    res_bdim = batching.transpose_ragged_axes(bdim.move_stacked_axis(0), perm)
  else:
    res_bdim = 0
  return transpose(operand, perm), res_bdim

def _transpose_lower(ctx, x, *, permutation):
  aval_out, = ctx.avals_out
  if dtypes.issubdtype(aval_out.dtype, dtypes.extended):
    elt_shape = core.physical_element_aval(aval_out.dtype).shape
    trailing_dims = [aval_out.ndim + i for i in range(len(elt_shape))]
    permutation = [*permutation, *trailing_dims]
  out = hlo.transpose(x, mlir.dense_int_array(permutation))
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

transpose_p = standard_primitive(
    _transpose_shape_rule, _input_dtype, 'transpose',
    sharding_rule=_transpose_sharding_rule)
ad.deflinear2(transpose_p,
              lambda t, _, permutation: [transpose(t, np.argsort(permutation))])
batching.primitive_batchers[transpose_p] = _transpose_batch_rule
mlir.register_lowering(transpose_p, _transpose_lower)
pe.def_trivial_padding(transpose_p)


def _select_shape_rule(which, *cases):
  if len(cases) == 0:
    raise TypeError("select must have at least one case")
  if any(case.shape != cases[0].shape for case in cases[1:]):
    msg = "select cases must have the same shapes, got [{}]."
    raise TypeError(msg.format(", ".join([str(c.shape) for c in cases])))
  if which.shape and which.shape != cases[0].shape:
    msg = ("select `which` must be scalar or have the same shape as cases, "
           "got `which` shape {} but case shape {}.")
    raise TypeError(msg.format(which.shape, cases[0].shape))
  return cases[0].shape

def _select_sharding_rule(which, *cases):
  if any(case.sharding != cases[0].sharding for case in cases[1:]):
    msg = "select cases must have the same shardings, got [{}]."
    raise TypeError(msg.format(", ".join([str(c.sharding) for c in cases])))
  if which.shape and which.sharding != cases[0].sharding:
    raise TypeError(
        'select `which` must be scalar or have the same sharding as cases, got'
        f' `which` sharding {which.sharding} but case sharding'
        f' {cases[0].sharding}.')
  return cases[0].sharding


def _select_dtype_rule(which, *cases):
  check_same_dtypes("select", *cases)
  if (not dtypes.issubdtype(which.dtype, np.bool_) and
      not dtypes.issubdtype(which.dtype, np.integer)):
    raise TypeError("select `which` must be boolean or integer type, got "
                    f"{which.dtype}.")
  if dtypes.issubdtype(which.dtype, np.bool_) and len(cases) > 2:
    raise TypeError("select with boolean `which` cannot have > 2 cases.")
  return cases[0].dtype

def _select_weak_type_rule(which, *cases):
  return all(c.weak_type for c in cases)

def _select_transpose_rule(t, which, *cases):
  assert not ad.is_undefined_primal(which)
  if type(t) is ad_util.Zero:
    return [None] + [ad_util.Zero(c.aval) if ad.is_undefined_primal(c) else None
                     for c in cases]
  else:
    zeros = full_like(t, 0)
    if dtypes.dtype(which) == np.dtype(np.bool_):
      ct0 = select(which, zeros, t) if ad.is_undefined_primal(cases[0]) else None
      ct1 = select(which, t, zeros) if ad.is_undefined_primal(cases[1]) else None
      return (None, ct0, ct1)
    else:
      return [None] + [
          select(eq(which, _const(which, i)), t, zeros)
          if ad.is_undefined_primal(case) else None for i, case in enumerate(cases)
      ]

def _select_batch_rule(batched_args, batch_dims, **unused_kwargs):
  which, *cases = batched_args
  which_bdim, *case_bdims = batch_dims
  size = next(x.shape[i] for x, i in zip(batched_args, batch_dims)
              if i is not None)

  # avoid transposes and some broadcasts in special cases
  if all(which_bdim == bdim for bdim in case_bdims):
    if np.shape(which) == np.shape(cases[0]):
      return select_n(which, *cases), which_bdim
    else:
      # vmapped function had a scalar which with nonscalar args
      assert np.ndim(which) == 1
      which = broadcast_in_dim(which, cases[0].shape, [which_bdim])
      return select_n(which, *cases), which_bdim
  elif np.ndim(which) == 0 and all(bdim is not None for bdim in case_bdims):
    if all(case_bdims[0] == bdim for bdim in case_bdims[1:]):
      return select_n(which, *cases), case_bdims[0]
    elif all(np.shape(cases[0]) == np.shape(c) for c in cases):
      bdim = case_bdims[0]
      other_cases = [batching.moveaxis(c, c_bdim, bdim)
                     for c, c_bdim in zip(cases[1:], case_bdims[1:])]
      return select_n(which, cases[0], *other_cases), bdim

  which = (batching.bdim_at_front(which, which_bdim, size) if np.shape(which)
           else which)
  if not all(() == np.shape(c) for c in cases):
    cases = [batching.bdim_at_front(c, bdim, size)
             for c, bdim in zip(cases, case_bdims)]
  assert all(np.shape(cases[0]) == np.shape(c) for c in cases[1:])
  if 0 < np.ndim(which) < np.ndim(cases[0]):
    # vmapped function had a scalar which with nonscalar args
    assert np.ndim(which) == 1
    which = broadcast_in_dim(which, cases[0].shape, [0])
  if np.ndim(which) > np.ndim(cases[0]):
    assert np.ndim(cases[0]) == 0
    cases = [broadcast(c, which.shape) for c in cases]
  return select_n(which, *cases), 0

def _select_jvp(primals, tangents):
  which, *case_primals = primals
  case_tangents = tangents[1:]
  out = select_n(which, *case_primals)
  if all(type(t) is ad_util.Zero for t in case_tangents):
    out_dot = ad_util.Zero(case_tangents[0].aval)
  else:
    z = _zeros(next(t for t in case_tangents if type(t) is not ad_util.Zero))
    case_tangents = [z if type(t) is ad_util.Zero else t for t in case_tangents]
    out_dot = select_n(which, *case_tangents)
  return out, out_dot

def _select_hlo_lowering_opaque(ctx, which, *cases):
  avals_in = ctx.avals_in
  aval_out, = ctx.avals_out
  assert all(aval_case == aval_out for aval_case in avals_in[1:])
  select_lower = _select_hlo_lowering

  physical_aval_out = core.physical_aval(aval_out)
  physical_avals_cases = [physical_aval_out] * (len(avals_in) - 1)
  aval_which = avals_in[0]
  aval_which_bcast = physical_aval_out.update(dtype=aval_which.dtype)
  assert aval_which_bcast.shape[:aval_which.ndim] == aval_which.shape

  bcast_dims = list(range(aval_which.ndim))
  which_bcast = mlir.broadcast_in_dim(
      ctx, which, aval_which_bcast, broadcast_dimensions=bcast_dims)

  return mlir.delegate_lowering(
      ctx, select_lower, which_bcast, *cases,
      avals_in=[aval_which_bcast, *physical_avals_cases],
      avals_out=[physical_aval_out])[0]

def _add_shit_to_select(ctx, op, aval_out):
  if config.sharding_in_types.value:
    return mlir.lower_sharding_under_shit(ctx, op, aval_out)
  return op

def _select_hlo_lowering(ctx, which, *cases):
  which_aval = ctx.avals_in[0]
  aval_out, = ctx.avals_out

  if dtypes.issubdtype(aval_out.dtype, dtypes.extended):
    op = _select_hlo_lowering_opaque(ctx, which, *cases)
    return [_add_shit_to_select(ctx, op, aval_out)]

  if which_aval.dtype == np.dtype(np.bool_):
    assert len(cases) <= 2
    if len(cases) == 1: return cases
    op = hlo.select(which, cases[1], cases[0])
    return [_add_shit_to_select(ctx, op, aval_out)]

  if dtypes.issubdtype(which_aval.dtype, np.signedinteger):
    compare_type = 'SIGNED'
  else:
    compare_type = 'UNSIGNED'
  lt = 'LT'

  def _select(offset, cases):
    assert len(cases) > 0
    if len(cases) == 1:
      return cases[0]
    mid = len(cases) // 2
    pred = mlir.compare_hlo(which,
                            mlir.full_like_aval(ctx, offset + mid, which_aval),
                            lt, compare_type)
    return hlo.select(pred, _select(offset, cases[:mid]),
                      _select(offset + mid, cases[mid:]))

  op = _select(0, cases)
  return [_add_shit_to_select(ctx, op, aval_out)]

select_n_p = standard_primitive(
    _select_shape_rule, _select_dtype_rule, 'select_n',
    weak_type_rule=_select_weak_type_rule, sharding_rule=_select_sharding_rule)
ad.primitive_jvps[select_n_p] = _select_jvp
ad.primitive_transposes[select_n_p] = _select_transpose_rule
batching.primitive_batchers[select_n_p] = _select_batch_rule
mlir.register_lowering(select_n_p, _select_hlo_lowering)
pe.def_trivial_padding(select_n_p)


def _reduce_shape_rule(*avals, computation, jaxpr, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  if any(arg.shape != () for arg in init_val_avals):
    init_val_shapes = [a.shape for a in init_val_avals]
    raise ValueError(f'reduce found non-scalar initial value: {init_val_shapes}')
  return [tuple(np.delete(op.shape, dimensions)) for op in operand_avals]

def _reduce_sharding_rule(*avals, computation, jaxpr, dimensions):
  operand_avals, _ = split_list(avals, [len(avals) // 2])
  return [op.sharding.with_spec(tuple_delete(op.sharding.spec, dimensions))
          for op in operand_avals]

def _reduce_dtype_rule(*avals, computation, jaxpr, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  operand_dtypes = [dtypes.canonicalize_dtype(op.dtype) for op in operand_avals]
  init_val_dtypes = [dtypes.canonicalize_dtype(init.dtype) for init in init_val_avals]
  if operand_dtypes != init_val_dtypes:
    raise TypeError(
        "reduce operand dtypes should match corresponding initial value dtypes, "
        f"got operands={operand_avals} and initial_values={init_val_avals}")
  return operand_dtypes

def _reduce_weak_type_rule(*avals, computation, jaxpr, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  return [op.weak_type and init_val.weak_type
          for op, init_val in safe_zip(operand_avals, init_val_avals)]

def _reduce_batch_rule(batched_args, batch_dims, *, computation, jaxpr,
                       dimensions):
  # TODO(mattjj,frostig): use batch_jaxpr, delete computation (assumes poly??)
  num_operands = len(batched_args) // 2
  operands, init_values = split_list(batched_args, [num_operands])
  operand_bdims, init_value_bdims = split_list(batch_dims, [num_operands])
  if all(init_value_bdim is batching.not_mapped
         for init_value_bdim in init_value_bdims):
    size = next(x.shape[ax] for x, ax in zip(batched_args, batch_dims)
                if ax is not None)
    operands = [batching.bdim_at_front(arg, bdim, size)
                for arg, bdim in zip(operands, operand_bdims)]
    new_dimensions = [d + 1 for d in dimensions]
    new_operand_bdims = [0] * num_operands
    return reduce_p.bind(*(operands + init_values),
                         computation=computation,
                         dimensions=tuple(new_dimensions),
                         jaxpr=jaxpr), new_operand_bdims
  else:
    raise NotImplementedError  # loop and stack

def _reduce_jvp(reducer, init_values, primals, tangents, axes):
  input_shape = np.array(primals[0].shape, dtype=int)

  n = np.prod(input_shape[list(axes)])
  non_axes = np.delete(np.arange(len(input_shape)), axes)

  # Move the reduced axes to the front, and flatten them to 1D.
  permutation = axes + tuple(non_axes)
  new_shape = (n,) + tuple(input_shape[non_axes])
  primals = tuple(reshape(x, new_shape, permutation) for x in primals)
  tangents = tuple(reshape(t, new_shape, permutation) for t in tangents)

  for d in range(len(non_axes) + 1):
    reducer = api.vmap(reducer)
  def _reduce_tree(*xs, axis=0):
    """Reduce by repeatedly splitting the array and multiplying."""
    while xs[0].shape[axis] > 1:
      n = xs[0].shape[axis]
      n1 = (n + 1) // 2
      n2 = n - n1
      xs1 = [slicing.slice_in_dim(x, 0, n1) for x in xs]
      xs2 = [slicing.slice_in_dim(x, n1, None) for x in xs]
      if n2 != n1:
        paddings = [(0, 0, 0)] * len(xs[0].shape)
        paddings[axis] = (0, 1, 0)
        xs2 = [pad(x2, i, paddings) for x2, i in zip(xs2, init_values)]
      xs = reducer(*(xs1 + xs2))
    if xs[0].shape[axis] == 0:
      return [full(input_shape[non_axes], i) for i in init_values]
    return tuple(squeeze(x, (axis,)) for x in xs)

  return api.jvp(_reduce_tree, primals, tangents)

def _reduce_jvp_rule(primals, tangents, *, computation, jaxpr, dimensions):
  primal_xs, init_values = split_list(primals, [len(primals) // 2])
  tangent_xs, tangent_init = split_list(tangents, [len(tangents) // 2])
  # This test may be too strict, if a value is actually zero but we cannot prove
  # it is symbolically zero.
  if any(type(t) is not ad_util.Zero for t in tangent_init):
    raise NotImplementedError(
      "Gradient of general lax.reduce with non-zero tangents for "
      "initial values to reduction not implemented")
  reducer = core.jaxpr_as_fun(jaxpr)
  return _reduce_jvp(reducer, init_values, primal_xs, tangent_xs, dimensions)

reduce_p = core.Primitive('reduce')
reduce_p.multiple_results = True
reduce_p.def_impl(partial(dispatch.apply_primitive, reduce_p))
reduce_p.def_abstract_eval(
    partial(standard_multi_result_abstract_eval, reduce_p, _reduce_shape_rule,
            _reduce_dtype_rule, _reduce_weak_type_rule, _reduce_sharding_rule))
batching.primitive_batchers[reduce_p] = _reduce_batch_rule
ad.primitive_jvps[reduce_p] = _reduce_jvp_rule

def _reduce_lower(ctx, *values, computation, jaxpr, dimensions):
  assert all(isinstance(x, core.ShapedArray) for x in ctx.avals_in), ctx.avals_in
  operands, init_values = util.split_list(values, [len(values) // 2])
  init_value_avals = ctx.avals_in[len(values) // 2:]
  op = hlo.ReduceOp([mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
                    operands, init_values, mlir.dense_int_array(dimensions))
  ir_types = [mlir.aval_to_ir_type(aval) for aval in init_value_avals]
  reducer = op.regions[0].blocks.append(*(ir_types + ir_types))
  with ir.InsertionPoint(reducer):
    name_stack = source_info_util.new_name_stack()
    if jaxpr.effects:
      raise NotImplementedError('Cannot lower effectful `reduce`.')
    out_nodes, _ = mlir.jaxpr_subcomp(ctx.module_context, jaxpr.jaxpr,
                                      name_stack, mlir.TokenSet(),
                                      jaxpr.consts,
                                      *reducer.arguments,
                                      dim_var_values=ctx.dim_var_values)
    hlo.return_(mlir.flatten_ir_values(out_nodes))
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, r, aval)
            for r, aval in safe_zip(op.results, ctx.avals_out)]
  return op.results

mlir.register_lowering(reduce_p, _reduce_lower)


def _reduce_number_dtype_rule(name, operand, *args, **kw):
  if not dtypes.issubdtype(operand.dtype, np.number):
    raise TypeError("{} does not accept dtype {}. Accepted dtypes are subtypes "
                    "of number.".format(name, dtype_to_string(operand.dtype)))
  return dtypes.canonicalize_dtype(operand.dtype)

def _reduce_sum_transpose_rule(cotangent, operand, *, axes):
  assert ad.is_undefined_primal(operand)
  input_shape = operand.aval.shape
  broadcast_dimensions = tuple(np.delete(np.arange(len(input_shape)), axes))
  if config.sharding_in_types.value:
    result = broadcast_in_dim(cotangent, input_shape, broadcast_dimensions,
                              sharding=operand.aval.sharding)
  else:
    result = broadcast_in_dim(cotangent, input_shape, broadcast_dimensions)
  assert result.shape == input_shape
  return [result]

def _reducer_padding(traceable, ident, in_avals, out_avals, operand, *, axes):
  del out_avals
  aval, = in_avals
  padded_axes = [(i, d.val) for i, d in enumerate(aval.shape)
                 if isinstance(d, pe.BoundedAxisSize)]
  operand_ = _replace_masked_values(operand, ident(aval.dtype), padded_axes)
  return [traceable(operand_, axes)]

def _replace_masked_values(x, val, padded_axes):
  if not padded_axes: return x
  dtype = dtypes._scalar_type_to_dtype(int)
  masks = [broadcasted_iota(dtype, x.shape, i) < d for i, d in padded_axes]
  return select(_reduce(operator.and_, masks), x, full_like(x, val))

def _reduce_op_shape_rule(operand, *, axes, input_shape=None):
  del input_shape  # Unused.
  if len(axes) != len(set(axes)):
    raise ValueError(f"duplicate value in 'axes' of reduction: {axes}")
  if not all(0 <= a < operand.ndim for a in axes):
    raise ValueError(f"reduction axes {axes} contains out-of-bounds indices for {operand}.")
  axes = frozenset(axes)
  return tuple(d for i, d in enumerate(operand.shape) if i not in axes)

def _reduce_op_sharding_rule(operand, *, axes):
  axes = frozenset(axes)
  new_spec = P(*tuple(s for i, s in enumerate(operand.sharding.spec)
                      if i not in axes))
  return operand.sharding.with_spec(new_spec)

reduce_sum_p = standard_primitive(
  _reduce_op_shape_rule, partial(_reduce_number_dtype_rule, 'reduce_sum'),
  'reduce_sum', sharding_rule=_reduce_op_sharding_rule)
ad.deflinear2(reduce_sum_p, _reduce_sum_transpose_rule)
batching.defreducer(reduce_sum_p, _get_sum_identity)
pe.padding_rules[reduce_sum_p] = partial(_reducer_padding, _reduce_sum,
                                         _get_sum_identity)
batching.ragged_prop_rules[reduce_sum_p] = batching.ragged_mask_elementwise_rule

def _reduce_prod_jvp_rule(primals, tangents, *, axes):
  reducer = lambda x, y: [mul(x, y)]
  primals_out, tangents_out = _reduce_jvp(reducer, [_const(primals[0], 1)],
                                          primals, tangents, axes)
  return primals_out[0], tangents_out[0]

reduce_prod_p = standard_primitive(
  _reduce_op_shape_rule, partial(_reduce_number_dtype_rule, 'reduce_prod'),
  'reduce_prod')
ad.primitive_jvps[reduce_prod_p] = _reduce_prod_jvp_rule
batching.defreducer(reduce_prod_p, _get_prod_identity)
pe.padding_rules[reduce_prod_p] = partial(_reducer_padding, _reduce_prod,
                                          _get_prod_identity)


def _reduce_chooser_jvp_rule(g, ans, operand, *, axes):
  # TODO(mattjj): an alternative is to use variadic reduce to compute the chosen
  # locations in a single pass (rather than comparing equality) and use a
  # gather, and/or even push along the chosen elements of g (b/112040122)
  shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
  location_indicators = convert_element_type(
      _eq_meet(operand, reshape(ans, shape)), g.dtype)
  counts = _reduce_sum(location_indicators, axes)
  return div(_reduce_sum(mul(g, location_indicators), axes), counts)


reduce_max_p = standard_primitive(
    _reduce_op_shape_rule, _input_dtype, 'reduce_max',
    sharding_rule=_reduce_op_sharding_rule)
ad.defjvp2(reduce_max_p, _reduce_chooser_jvp_rule)
batching.defreducer(reduce_max_p, _get_max_identity)
pe.padding_rules[reduce_max_p] = partial(_reducer_padding, _reduce_max,
                                         _get_max_identity)
batching.ragged_prop_rules[reduce_max_p] = batching.ragged_mask_elementwise_rule


reduce_min_p = standard_primitive(_reduce_op_shape_rule, _input_dtype,
                                  'reduce_min')
ad.defjvp2(reduce_min_p, _reduce_chooser_jvp_rule)
batching.defreducer(reduce_min_p, _get_min_identity)
pe.padding_rules[reduce_min_p] = partial(_reducer_padding, _reduce_min,
                                         _get_min_identity)


def _argminmax_shape_rule(operand, *, axes, index_dtype):
  axis, = axes
  if not (0 <= axis < len(operand.shape)):
    raise ValueError(f"Invalid axis {axis} for operand shape {operand.shape}")
  if operand.shape[axis] < 1:
    raise ValueError("argmin and argmax require non-empty reduced dimension. "
                     f"operand.shape={operand.shape} {axis=}")
  return util.tuple_delete(operand.shape, axis)

def _argminmax_sharding_rule(operand, *, axes, index_dtype):
  axis, = axes
  return operand.sharding.with_spec(
      util.tuple_delete(operand.sharding.spec, axis))

def _argminmax_dtype_rule(operand, *, axes, index_dtype):
  if not dtypes.issubdtype(index_dtype, np.integer):
    raise TypeError("index_dtype must be an integer type, but got {}"
                    .format(dtype_to_string(index_dtype)))
  return index_dtype

class _ArgMinMaxReducer:

  def __init__(self, value_comparator):
    self._value_comparator = value_comparator

  def __repr__(self):
    # Override the repr so that the metadata attached to the lowered op does not
    # contain unstable function ids. This plays more nicely with computation
    # fingerprint calculation in the compilation cache.
    return f'_ArgMinMaxReducer({self._value_comparator.__name__})'

  def __call__(self, op_val_index, acc_val_index):
    op_val, op_index = op_val_index
    acc_val, acc_index = acc_val_index
    # Pick op_val if Lt (for argmin) or if NaN
    pick_op_val = bitwise_or(self._value_comparator(op_val, acc_val),
                             ne(op_val, op_val))
    # If x and y are not NaN and x = y, then pick the first
    pick_op_index = bitwise_or(pick_op_val,
                               bitwise_and(eq(op_val, acc_val),
                                           lt(op_index, acc_index)))
    return (select(pick_op_val, op_val, acc_val),
            select(pick_op_index, op_index, acc_index))

def _compute_argminmax(value_comparator, get_identity,
                       operand, *, index_dtype, axes):
  # value_comparator is either lax.lt (for argmin) or lax.gt
  # get_identity(operand.dtype) is inf for argmin or -inf for argmax
  axis, = axes
  indices = broadcasted_iota(
      index_dtype, np.shape(operand), axis,
      sharding=operand.sharding if config.sharding_in_types.value else None)
  res = reduce([operand, indices],
               [get_identity(operand.dtype), np.array(0, index_dtype)],
               _ArgMinMaxReducer(value_comparator),
               axes)
  return res[1]

argmin_p = standard_primitive(_argminmax_shape_rule, _argminmax_dtype_rule,
                              'argmin', weak_type_rule=_strip_weak_type,
                              sharding_rule=_argminmax_sharding_rule)
batching.defreducer(argmin_p, _get_min_identity)
ad.defjvp_zero(argmin_p)

argmax_p = standard_primitive(_argminmax_shape_rule, _argminmax_dtype_rule,
                              'argmax', weak_type_rule=_strip_weak_type,
                              sharding_rule=_argminmax_sharding_rule)
batching.defreducer(argmax_p, _get_max_identity)
ad.defjvp_zero(argmax_p)

mlir.register_lowering(argmin_p, mlir.cache_lowering(
    mlir.lower_fun(partial(_compute_argminmax, lt, _get_min_identity),
                   multiple_results=False)))

mlir.register_lowering(argmax_p, mlir.cache_lowering(
    mlir.lower_fun(partial(_compute_argminmax, gt, _get_max_identity),
                   multiple_results=False)))


def _reduce_logical_shape_rule(operand, *, axes):
  if operand.dtype != np.bool_ and not np.issubdtype(operand.dtype, np.integer):
    raise TypeError(f"logical reduction requires operand dtype bool or int, got {operand.dtype}.")
  return tuple(np.delete(operand.shape, axes))

reduce_or_p = standard_primitive(
    _reduce_logical_shape_rule, _input_dtype, 'reduce_or',
    weak_type_rule=_strip_weak_type)
batching.defreducer(reduce_or_p, _get_bitwise_or_identity)


reduce_and_p = standard_primitive(
    _reduce_logical_shape_rule, _input_dtype, 'reduce_and',
    weak_type_rule=_strip_weak_type)
batching.defreducer(reduce_and_p, _get_bitwise_and_identity)
batching.ragged_prop_rules[reduce_and_p] = batching.ragged_mask_elementwise_rule


reduce_xor_p = standard_primitive(
    _reduce_logical_shape_rule, _input_dtype, 'reduce_xor',
    weak_type_rule=_strip_weak_type)
batching.defreducer(reduce_xor_p, _get_bitwise_or_identity)


def _unary_reduce_lower(reducer, unit_factory, ctx, x, *, axes):
  aval_out, = ctx.avals_out
  dtype = aval_out.dtype
  op = hlo.ReduceOp([mlir.aval_to_ir_type(aval_out)], [x],
                    [mlir.ir_constant(unit_factory(aval_out.dtype))],
                    mlir.dense_int_array(axes))
  scalar_type = mlir.aval_to_ir_type(core.ShapedArray((), dtype))
  reducer_region = op.regions[0].blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(reducer_region):
    hlo.return_([reducer(*reducer_region.arguments)])
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, op.result, aval_out)]
  return op.results

mlir.register_lowering(reduce_sum_p, partial(_unary_reduce_lower, hlo.AddOp,
                                             _get_sum_identity))
mlir.register_lowering(reduce_prod_p, partial(_unary_reduce_lower, hlo.MulOp,
                                              _get_prod_identity))
mlir.register_lowering(reduce_or_p, partial(_unary_reduce_lower, hlo.OrOp,
                                            _get_bitwise_or_identity))
mlir.register_lowering(reduce_and_p, partial(_unary_reduce_lower, hlo.AndOp,
                                             _get_bitwise_and_identity))
mlir.register_lowering(reduce_xor_p, partial(_unary_reduce_lower, hlo.XorOp,
                                             _get_bitwise_or_identity))
mlir.register_lowering(reduce_min_p, partial(_unary_reduce_lower, mlir.min_hlo,
                                             _get_min_identity))
mlir.register_lowering(reduce_max_p, partial(_unary_reduce_lower, mlir.max_hlo,
                                             _get_max_identity))


def _reduce_precision_shape_rule(operand, *, exponent_bits, mantissa_bits):
  exponent_bits = operator.index(exponent_bits)
  mantissa_bits = operator.index(mantissa_bits)
  if exponent_bits < 1:
    raise ValueError(f"reduce_precision: exponent_bits must be positive; got {exponent_bits}")
  if mantissa_bits < 0:
    raise ValueError(f"reduce_precision: mantissa_bits must be non-negative; got {mantissa_bits}")
  return operand.shape


reduce_precision_p = standard_primitive(
    _reduce_precision_shape_rule,
    partial(unop_dtype_rule, _identity, _float, 'reduce_precision'),
    name='reduce_precision')
ad.deflinear(reduce_precision_p, lambda t, **kwargs: [reduce_precision_p.bind(t, **kwargs)])
batching.defvectorized(reduce_precision_p)

def _reduce_precision_lower(ctx, operand, *, exponent_bits, mantissa_bits):
  aval_out, = ctx.avals_out
  return [hlo.reduce_precision(operand, mlir.i32_attr(exponent_bits),
                               mlir.i32_attr(mantissa_bits))]

mlir.register_lowering(reduce_precision_p, _reduce_precision_lower)


_UINT_DTYPES = {
  16: np.dtype(np.uint16),
  32: np.dtype(np.uint32),
  64: np.dtype(np.uint64),
}

_INT_DTYPES = {
  16: np.dtype(np.int16),
  32: np.dtype(np.int32),
  64: np.dtype(np.int64),
}


def _sort_abstract_eval(*args, **kwargs):
  args = tuple(args)
  if any(arg.shape != args[0].shape for arg in args[1:]):
    shapes = " ".join(str(a.shape) for a in args)
    raise TypeError(f"Arguments to sort must have equal shapes, got: {shapes}")
  return args


def _canonicalize_float_for_sort(x):
  # In the sort comparator, we are going to use a comparision operator where -0
  # would be before 0, and -NaN and NaN appear at the beginning and end of the
  # ordering. In this scheme, -0 would be before 0, and -NaN and NaN appear at
  # the beginning and end of the ordering. This causes issues for stable
  # sorts, so we avoid this by standardizing the representation of zeros
  # and NaNs in the output.

  result = select(eq(x, _zero(x)), _zeros(x), x)
  with config.debug_nans(False):
    result = select(_isnan(x), full_like(result, np.nan), result)

  return result


# Default comparator that sorts the operands lexicographically on the
# first `num_keys` arguments.
# For floating point types, a total order is created where
# -infinity < ... < 0 < ... < infinity < NaN.
# 0.0 and -0.0 are treated as equivalent, as are all NaN representations.
# For complex types, the (real, imag) pairs are sorted lexicographically
# (following NumPy's semantics).
# This code adds complex-number support and lexicographic ordering to the algorithm from:
# https://github.com/tensorflow/tensorflow/blob/ba43780830f09da72081fe5061c436f1c6203a92/tensorflow/compiler/xla/client/lib/comparators.h#L33
def _sort_lt_comparator(*operands, num_keys=1):
  x_keys, y_keys = _operands_to_keys(*operands, num_keys=num_keys)
  p = None
  for xk, yk in zip(x_keys[::-1], y_keys[::-1]):
    p = (bitwise_or(lt_to_p.bind(xk, yk), bitwise_and(eq_to_p.bind(xk, yk), p)) if p is not None
         else lt_to_p.bind(xk, yk))
  return p

# Similar to sort_lt_comparator, but implements less than or equal. Used by
# the searchsorted() implementation.
def _sort_le_comparator(*operands, num_keys=1):
  x_keys, y_keys = _operands_to_keys(*operands, num_keys=num_keys)
  p = None
  for xk, yk in zip(x_keys[::-1], y_keys[::-1]):
    p = (bitwise_or(lt_to_p.bind(xk, yk), bitwise_and(eq_to_p.bind(xk, yk), p)) if p is not None
         else le_to_p.bind(xk, yk))
  return p

def _operands_to_keys(*operands, num_keys=1):
  assert len(operands) >= 2 and len(operands) % 2 == 0, operands
  assert len(operands) // 2 >= num_keys, (operands, num_keys)
  x_keys, y_keys = [], []
  for x, y in zip(operands[:2*num_keys:2], operands[1:2*num_keys:2]):
    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    if dtypes.issubdtype(x.dtype, np.complexfloating):
      x_keys.extend([_canonicalize_float_for_sort(real(x)), _canonicalize_float_for_sort(imag(x))])
      y_keys.extend([_canonicalize_float_for_sort(real(y)), _canonicalize_float_for_sort(imag(y))])
    elif dtypes.issubdtype(x.dtype, np.floating):
      x_keys.append(_canonicalize_float_for_sort(x))
      y_keys.append(_canonicalize_float_for_sort(y))
    else:
      x_keys.append(x)
      y_keys.append(y)
  return x_keys, y_keys


def _sort_jvp(primals, tangents, *, dimension, is_stable, num_keys):
  shape = primals[0].shape
  sorted_primals_and_idx = sort_p.bind(
      *primals, broadcasted_iota(np.uint64, shape, dimension),
      dimension=dimension, is_stable=is_stable, num_keys=num_keys)
  batch_dims = tuple(np.delete(np.arange(len(shape), dtype=np.int64),
                               dimension))
  dnums = slicing.GatherDimensionNumbers(
    offset_dims=(),
    collapsed_slice_dims=(dimension,),
    start_index_map=(dimension,),
    operand_batching_dims=batch_dims,
    start_indices_batching_dims=batch_dims,
  )
  idx = expand_dims(sorted_primals_and_idx[-1], (len(shape),))
  gather_idx = partial(
    slicing.gather,
    start_indices=idx, dimension_numbers=dnums, slice_sizes=(1,) * len(shape),
    mode=slicing.GatherScatterMode.PROMISE_IN_BOUNDS
  )
  tangents_out = [t if type(t) is ad_util.Zero else gather_idx(t)
                  for t in tangents]
  return tuple(sorted_primals_and_idx[:-1]), tangents_out

def _sort_batch_rule(batched_args, batch_dims, *, dimension, is_stable, num_keys):
  prototype_arg, new_bdim = next(
    (a, b) for a, b in zip(batched_args, batch_dims) if b is not None)
  new_args = []
  for arg, bdim in zip(batched_args, batch_dims):
    if bdim is None:
      dims = np.delete(np.arange(prototype_arg.ndim), new_bdim)
      new_args.append(broadcast_in_dim(arg, prototype_arg.shape, dims))
    else:
      new_args.append(batching.moveaxis(arg, bdim, new_bdim))
  new_dimension = dimension + (new_bdim <= dimension)
  bdims = (new_bdim,) * len(new_args)
  return (sort_p.bind(*new_args, dimension=new_dimension, is_stable=is_stable, num_keys=num_keys),
          bdims)


sort_p = Primitive('sort')
sort_p.multiple_results = True
sort_p.def_impl(partial(dispatch.apply_primitive, sort_p))
sort_p.def_abstract_eval(_sort_abstract_eval)
ad.primitive_jvps[sort_p] = _sort_jvp
batching.primitive_batchers[sort_p] = _sort_batch_rule


def _sort_lower(ctx, *operands, dimension, is_stable, num_keys):
  assert all(isinstance(x, core.ShapedArray) for x in ctx.avals_in), ctx.avals_in
  sort = hlo.SortOp([mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
                    mlir.flatten_ir_values(operands),
                    dimension=mlir.i64_attr(dimension),
                    is_stable=ir.BoolAttr.get(is_stable))
  scalar_avals = [aval.update(shape=()) for aval in ctx.avals_in]
  scalar_types = safe_map(mlir.aval_to_ir_type, scalar_avals)
  comparator = sort.comparator.blocks.append(
      *util.flatten(zip(scalar_types, scalar_types)))
  with ir.InsertionPoint(comparator):
    lower_comparator = mlir.lower_fun(partial(_sort_lt_comparator),
                                      multiple_results=False)
    sub_ctx = ctx.replace(primitive=None,
                          avals_in=util.flatten(zip(scalar_avals, scalar_avals)),
                          avals_out=[core.ShapedArray((), np.bool_)])

    out = lower_comparator(sub_ctx, *comparator.arguments, num_keys=num_keys)
    hlo.return_(mlir.flatten_ir_values(out))
  return sort.results

mlir.register_lowering(sort_p, _sort_lower)


def _top_k_abstract_eval(operand, *, k):
  if dtypes.issubdtype(operand.dtype, np.complexfloating):
    raise ValueError("top_k is not compatible with complex inputs.")
  if k < 0:
    raise ValueError(f"k argument to top_k must be nonnegative, got {k}")
  if len(operand.shape) == 0:
    raise TypeError("top_k operand must have >= 1 dimension, got {}"
                    .format(operand.shape))
  shape = list(operand.shape)
  if shape[-1] < k:
    msg = "k argument to top_k must be no larger than minor dimension; {} vs {}"
    raise ValueError(msg.format(k, shape))
  shape[-1] = k
  return (operand.update(shape=shape, dtype=operand.dtype,
                         weak_type=operand.weak_type),
          operand.update(shape=shape, dtype=np.dtype(np.int32)))

def _top_k_jvp(primals, tangents, *, k):
  operand, = primals
  tangent, = tangents
  primals_out = top_k(operand, k)
  if type(tangent) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_primal_value(primals_out[0])
  else:
    _, k_idxs = primals_out
    idx_shape = k_idxs.shape
    rank = len(idx_shape)
    gather_index_shape = idx_shape + (1,)
    gather_indices = reshape(k_idxs, gather_index_shape)
    slice_sizes = (1,) * rank
    dnums = slicing.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(rank - 1,),
        operand_batching_dims=tuple(range(rank - 1)),
        start_indices_batching_dims=tuple(range(rank - 1)),
        start_index_map=(rank - 1,),
    )
    tangent_out = slicing.gather(tangent, gather_indices, dnums, slice_sizes)
  return primals_out, (tangent_out, ad_util.Zero.from_primal_value(primals_out[1]))

def _top_k_batch_rule(batched_args, batch_dims, *, k):
  operand, = batched_args
  bdim, = batch_dims
  if bdim == operand.ndim-1:
    perm = np.arange(operand.ndim)
    perm[bdim-1], perm[bdim] = perm[bdim], perm[bdim-1]
    top_k_v, top_k_i = top_k(transpose(operand, perm), k=k)
    return (transpose(top_k_v, perm),
            transpose(top_k_i, perm)), (bdim, bdim)
  else:
    return top_k(operand, k=k), (bdim, bdim)

top_k_p = Primitive('top_k')
top_k_p.multiple_results = True
top_k_p.def_impl(partial(dispatch.apply_primitive, top_k_p))
top_k_p.def_abstract_eval(_top_k_abstract_eval)
def _top_k_lower(ctx, operand, k):
  if core.is_constant_dim(k):
    return chlo.TopKOp(operand, mlir.i64_attr(k)).results
  k_value, = mlir.eval_dynamic_shape_as_vals(ctx, (k,))
  out_values_aval, out_indices_aval, = ctx.avals_out
  return mlir.custom_call(
      "stablehlo.dynamic_top_k",
      result_types=[mlir.aval_to_ir_type(out_values_aval),
       mlir.aval_to_ir_type(out_indices_aval)],
      operands=[operand, k_value]).results

mlir.register_lowering(top_k_p, _top_k_lower)
ad.primitive_jvps[top_k_p] = _top_k_jvp
batching.primitive_batchers[top_k_p] = _top_k_batch_rule

def _stop_gradient_jvp_rule(primals, tangents):
  # if we don't call stop_gradient here, we'd only peel off one autodiff tracer
  x, = primals
  return stop_gradient(x), ad_util.Zero.from_primal_value(x)

def _stop_gradient_batch_rule(batched_args, batch_dims):
  x, = batched_args
  dim, = batch_dims
  return stop_gradient(x), dim

ad.primitive_jvps[ad_util.stop_gradient_p] = _stop_gradient_jvp_rule
batching.primitive_batchers[ad_util.stop_gradient_p] = _stop_gradient_batch_rule
pe.def_trivial_padding(ad_util.stop_gradient_p)


def create_token(_=None):
  """Creates an XLA token value with no preconditions for sequencing effects.

  Experimental.

  The argument is ignored. It exists for backward compatibility.
  """
  return create_token_p.bind()

create_token_p = Primitive("create_token")
create_token_p.def_impl(partial(dispatch.apply_primitive, create_token_p))
create_token_p.def_abstract_eval(lambda *_: abstract_token)

def _create_token_lowering(ctx, *operands):
  aval_out, = ctx.avals_out
  return [hlo.create_token()]
mlir.register_lowering(create_token_p, _create_token_lowering)


def after_all(*operands):
  """Merges one or more XLA token values. Experimental.

  Wraps the XLA AfterAll operator."""
  return after_all_p.bind(*operands)

def _after_all_abstract_eval(*operands):
  if any(x is not abstract_token for x in operands):
    raise TypeError("Arguments to after_all must be tokens")
  return abstract_token


after_all_p = Primitive("after_all")
after_all_p.def_impl(partial(dispatch.apply_primitive, after_all_p))
after_all_p.def_abstract_eval(_after_all_abstract_eval)

def _after_all_lowering(ctx, *operands):
  aval_out, = ctx.avals_out
  return [hlo.after_all(operands)]
mlir.register_lowering(after_all_p, _after_all_lowering)


class InOutFeedEffect(effects.Effect):
  pass
infeed_effect = InOutFeedEffect()
outfeed_effect = InOutFeedEffect()


def infeed(token, shape=None, partitions=None):
  """Consumes an infeed value of `shape` from the host. Experimental.

  `token` is used to sequence infeed and outfeed effects.
  `partitions` may be specified inside a `sharded_jit` function.
  """
  flat_shapes, treedef = tree_util.tree_flatten(shape)
  for shape in flat_shapes:
    if not isinstance(shape, ShapedArray):
      raise TypeError("shape argument to infeed must be a pytree of "
                      "ShapedArray values, got {}".format(shape))
  if partitions is not None:
    # Always replicate token.
    # We specifically use type() to raise an error for PartitionSpecs.
    if type(partitions) != tuple:  # pylint: disable=unidiomatic-typecheck
      raise ValueError(f"'partitions' argument to infeed should be a tuple, "
                       f"got {partitions}")
    partitions = partitions + (None,)
  xs_and_token = infeed_p.bind(token, shapes=tuple(flat_shapes),
                               partitions=partitions)
  return (treedef.unflatten(xs_and_token[:-1]), xs_and_token[-1])

def _infeed_abstract_eval(token, *, shapes, partitions):
  if token is not abstract_token:
    raise TypeError("First argument to infeed must be a token")
  return (*shapes, abstract_token), {infeed_effect}


infeed_p = Primitive("infeed")
infeed_p.multiple_results = True
infeed_p.def_impl(partial(dispatch.apply_primitive, infeed_p))
infeed_p.def_effectful_abstract_eval(_infeed_abstract_eval)
mlir.lowerable_effects.add_type(InOutFeedEffect)


def _infeed_lowering(ctx, token, *, shapes, partitions):
  output_types = safe_map(mlir.aval_to_ir_type, ctx.avals_out[:-1])
  flat_output_types = mlir.flatten_ir_types(output_types)
  # TODO(phawkins): verify `shapes` have a major-to-minor layout.
  layouts = ir.ArrayAttr.get([
      ir.ArrayAttr.get(
          [mlir.i64_attr(i)
           for i in range(len(aval.shape) - 1, -1, -1)])
      for aval in shapes
  ])
  infeed = hlo.InfeedOp(
      flat_output_types + [hlo.TokenType.get()],
      token,
      infeed_config=ir.StringAttr.get(''),
      layout=layouts)
  if partitions is not None:
    mlir.set_sharding(infeed, xla.sharding_to_proto(partitions))
  token = infeed.results[-1]
  outs = infeed.results[:-1]
  return mlir.unflatten_ir_values_like_types(outs, output_types) + [
      token,
  ]

mlir.register_lowering(infeed_p, _infeed_lowering)


def outfeed(token, xs, partitions = None):
  """Outfeeds value `xs` to the host. Experimental.

  `token` is used to sequence infeed and outfeed effects.
  `partitions` may be specified inside a `sharded_jit` or `pjit` function.
  """
  if partitions is not None:
    # We specifically use type() to raise an error for PartitionSpecs.
    if type(partitions) != tuple:  # pylint: disable=unidiomatic-typecheck
      raise ValueError(f"'partitions' argument to outfeed should be a tuple, "
                       f"got {partitions}")
  flat_xs, _ = tree_util.tree_flatten(xs)
  return outfeed_p.bind(token, *flat_xs, partitions=partitions)

def _outfeed_abstract_eval(token, *xs, partitions):
  if token is not abstract_token:
    raise TypeError("First argument to outfeed must be a token")
  return abstract_token, {outfeed_effect}

outfeed_p = Primitive("outfeed")
outfeed_p.def_impl(partial(dispatch.apply_primitive, outfeed_p))
outfeed_p.def_effectful_abstract_eval(_outfeed_abstract_eval)
mlir.lowerable_effects.add_type(InOutFeedEffect)


def _outfeed_lowering(ctx, token, *xs, partitions):
  outfeed = hlo.OutfeedOp(
      mlir.flatten_ir_values(xs),
      token,
      outfeed_config=ir.StringAttr.get(''))
  if partitions is not None:
    mlir.set_sharding(outfeed, xla.sharding_to_proto(partitions))
  return outfeed.results

mlir.register_lowering(outfeed_p, _outfeed_lowering)


def rng_uniform(a, b, shape):
  """Stateful PRNG generator. Experimental and its use is discouraged.

  Returns uniformly distributed random numbers in the range [a, b). If
  b <= a, then the result is undefined, and different implementations may
  return different results.

  You should use jax.random for most purposes; this function exists only for
  niche use cases with special performance requirements.

  This API may be removed at any time.
  """
  return rng_uniform_p.bind(a, b, shape=tuple(shape))

def _rng_uniform_abstract_eval(a, b, *, shape):
  if a.dtype != b.dtype:
    raise ValueError(
      "Arguments to rng_uniform must have identical dtypes, got {} "
      "and {}.".format(a.dtype, b.dtype))
  if a.shape != () or b.shape != ():
    raise ValueError(
      "Arguments to rng_uniform must be scalars; got shapes {} and {}."
      .format(a.shape, b.shape))
  return a.update(shape=shape, dtype=a.dtype,
                  weak_type=(a.weak_type and b.weak_type))

rng_uniform_p = Primitive("rng_uniform")
rng_uniform_p.def_impl(partial(dispatch.apply_primitive, rng_uniform_p))
rng_uniform_p.def_abstract_eval(_rng_uniform_abstract_eval)

def _rng_uniform_lowering(ctx, a, b, *, shape):
  aval_out, = ctx.avals_out
  shape = mlir.ir_constant(np.array(aval_out.shape, np.int64))
  return [hlo.rng(a, b, shape, hlo.RngDistributionAttr.get('UNIFORM'))]

mlir.register_lowering(rng_uniform_p, _rng_uniform_lowering)


def _rng_bit_generator_shape_rule(key, *, shape, dtype, algorithm):
  del dtype, algorithm
  return (key.shape, tuple(shape))

def _rng_bit_generator_dtype_rule(key, *, shape, dtype, algorithm):
  del shape, algorithm
  return (key.dtype, dtype)

def _rng_bit_generator_weak_type_rule(key, *, shape, dtype, algorithm):
  del shape, dtype, algorithm
  return (key.weak_type, False)


class RandomAlgorithm(enum.IntEnum):
  """Describes which PRNG algorithm to use for rng_bit_generator."""

  RNG_DEFAULT = 0
  "The platform's default algorithm."

  RNG_THREE_FRY = 1
  "The Threefry-2x32 PRNG algorithm."

  RNG_PHILOX = 2
  "The Philox-4x32 PRNG algorithm."


RandomAlgorithm.__str__ = lambda algorithm: algorithm.name  # type: ignore[method-assign]

def _rng_algorithm(algorithm: RandomAlgorithm):
  if algorithm == RandomAlgorithm.RNG_THREE_FRY:
    return hlo.RngAlgorithmAttr.get("THREE_FRY")
  elif algorithm == RandomAlgorithm.RNG_PHILOX:
    return hlo.RngAlgorithmAttr.get("PHILOX")
  elif algorithm == RandomAlgorithm.RNG_DEFAULT:
    return hlo.RngAlgorithmAttr.get("DEFAULT")
  else:
    assert False

def _rng_bit_generator_lowering(
    ctx, key, *, shape, dtype, algorithm):
  key_type = ir.RankedTensorType(key.type)
  key_shape, key_etype = key_type.shape, key_type.element_type
  # While the RngBitGenerator HLO accepts a u64[2] key on all backends, we
  # typically represent the key argument to this primitive as a u32[4] so as to
  # sidestep issues with the jax_enable_x64=False configuration. As a result, we
  # need to convert u32[4] -> u64[2] here in the translation rule. However, we
  # also polymorphically allow a u64[2] for backward compatibility.
  #
  # Separately, RngBitGenerator doesn't support generating u8 or
  # u16, so we request u32 and truncate in that case.
  u32_type = ir.IntegerType.get_unsigned(32)
  u64_type = ir.IntegerType.get_unsigned(64)
  assert ((key_shape == [4] and key_etype == u32_type) or
          (key_shape == [2] and key_etype == u64_type)), (key_shape, key_etype)
  dtype = np.dtype(dtype)
  etype = mlir.dtype_to_ir_type(dtype)
  if dtype in (np.dtype('uint8'), np.dtype('uint16'), np.dtype('uint32'),
               np.dtype('uint64')):
    rbg_etype = etype
    rbg_dtype = dtype
  else:
    rbg_etype = u32_type
    rbg_dtype = np.uint32
  if key_etype == u32_type:
    key = hlo.bitcast_convert(
        ir.RankedTensorType.get([2], u64_type),
        hlo.reshape(ir.RankedTensorType.get([2, 2], u32_type), key))
  algorithm_attr = _rng_algorithm(algorithm)
  _, out_vals_aval = ctx.avals_out
  if any(not core.is_constant_shape(a.shape) for a in ctx.avals_out):
    output_shape = mlir.shape_tensor(
      mlir.eval_dynamic_shape(ctx, out_vals_aval.shape))
    out_key, out_vals = mlir.custom_call(
        "stablehlo.dynamic_rng_bit_generator",
        result_types=[key.type,
                      mlir.aval_to_ir_type(core.ShapedArray(shape, rbg_dtype))],
        operands=[key, output_shape],
        extra_attributes=dict(rng_algorithm=algorithm_attr)).results
  else:
    out_key, out_vals = hlo.RngBitGeneratorOp(
        key.type,
        ir.RankedTensorType.get(shape, rbg_etype),
        algorithm_attr, key).results
  if key_etype == u32_type:
    out_key = hlo.reshape(
        ir.RankedTensorType.get([4], u32_type),
        hlo.bitcast_convert(
            ir.RankedTensorType.get([2, 2], u32_type), out_key))
  if rbg_etype != etype:
    out_vals = hlo.convert(
      ir.RankedTensorType.get(ir.RankedTensorType(out_vals.type).shape, etype),
      out_vals)
  return [out_key, out_vals]


rng_bit_generator_p = Primitive("rng_bit_generator")
rng_bit_generator_p.multiple_results = True
rng_bit_generator_p.def_impl(
    partial(dispatch.apply_primitive, rng_bit_generator_p))
rng_bit_generator_p.def_abstract_eval(
    partial(standard_multi_result_abstract_eval, rng_bit_generator_p,
            _rng_bit_generator_shape_rule, _rng_bit_generator_dtype_rule,
            _rng_bit_generator_weak_type_rule, None))
mlir.register_lowering(rng_bit_generator_p,
                       _rng_bit_generator_lowering)


def _array_copy(arr: ArrayLike) -> Array:
  return copy_p.bind(arr)


def _which_dim_sharded(s: PmapSharding) -> int | None:
  sharded_dim = None
  for i, s in enumerate(s.sharding_spec.sharding):
    if isinstance(s, (pxla.Unstacked, pxla.Chunked)):
      sharded_dim = i
      break
  return sharded_dim


def _identity_fn(x): return x


def _copy_impl_pmap_sharding(sharded_dim, *args, **kwargs):
  axis_name, static_broadcasted_tuple, donate_tuple = api._shared_code_pmap(
    _identity_fn, None, (), (), sharded_dim, sharded_dim)
  p = api._prepare_pmap(
      _identity_fn, sharded_dim, sharded_dim, static_broadcasted_tuple,
      donate_tuple, None, None, None, args, kwargs)
  out_flat =  pxla.xla_pmap_impl(
      p.flat_fun, *p.flat_args, backend=None, axis_name=axis_name,
      axis_size=p.local_axis_size, global_axis_size=p.global_axis_size,
      devices=p.devices, in_axes=p.in_axes_flat,
      out_axes_thunk=p.out_axes_thunk, name=p.flat_fun.__name__,
      donated_invars=p.donated_invars,
      is_explicit_global_axis_size=p.is_explicit_global_axis_size,
  )
  return tree_util.tree_unflatten(p.out_tree(), out_flat)


# TODO(https://github.com/jax-ml/jax/issues/13552): Look into making this a
# method on jax.Array so that we can bypass the XLA compilation here.
def _copy_impl(prim, *args, **kwargs):
  a, = args
  if isinstance(a, Array) and isinstance(a.sharding, PmapSharding):
    sharded_dim = _which_dim_sharded(a.sharding)
    if sharded_dim is None:
      return dispatch.apply_primitive(prim, *args, **kwargs)
    return _copy_impl_pmap_sharding(sharded_dim, *args, **kwargs)
  return dispatch.apply_primitive(prim, *args, **kwargs)

# The copy_p primitive exists for expressing making copies of runtime arrays.
# For that reason we don't simplify it out of jaxprs (e.g. for jit invariance).
# It's used in jnp.array(x, copy=True), which is the user-facing API.
copy_p = core.Primitive('copy')
copy_p.def_impl(partial(_copy_impl, copy_p))
copy_p.def_abstract_eval(lambda x: x)
mlir.register_lowering(copy_p, lambda ctx, x: [x])
ad.deflinear(copy_p, lambda t: [copy_p.bind(t)])
pe.def_trivial_padding(copy_p)
batching.defvectorized(copy_p)
def _propagate_mem_kind_copy(in_mem_kind):
  return in_mem_kind
pxla.memory_kind_propagate_rule[copy_p] = _propagate_mem_kind_copy

def rng_bit_generator(key, shape, dtype=np.uint32,
                      algorithm=RandomAlgorithm.RNG_DEFAULT):
  """Stateless PRNG bit generator. Experimental and its use is discouraged.

  Returns uniformly distributed random bits with the specified shape and dtype
  (what is required to be an integer type) using the platform specific
  default algorithm or the one specified.

  It provides direct access to the RngBitGenerator primitive exposed by XLA
  (https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator) for low
  level API access.

  Most users should use `jax.random` instead for a stable and more user
  friendly API.
  """
  shape = core.canonicalize_shape(shape)
  dtype = dtypes.canonicalize_dtype(dtype)
  if np.dtype(dtype) not in {np.dtype('uint8'), np.dtype('uint16'),
                             np.dtype('uint32'), np.dtype('uint64')}:
    raise TypeError(f'rng_bit_generator: unsupported dtype {dtype}')
  return tuple(
      rng_bit_generator_p.bind(
          key, shape=shape, dtype=dtype, algorithm=algorithm))


def _iota_abstract_eval(*dyn_shape, dtype, shape, dimension, sharding):
  if not dyn_shape:
    # TODO(mattjj) Generalize shape_like checking to permit dynamic shapes
    _check_shapelike("iota", "shape", shape)
  if not any(dtypes.issubdtype(dtype, t) for t in _num):
    msg = 'iota does not accept dtype {}. Accepted dtypes are subtypes of {}.'
    typename = dtype_to_string(dtype)
    accepted_typenames = (t.__name__ for t in _num)
    raise TypeError(msg.format(typename, ', '.join(accepted_typenames)))
  if not 0 <= dimension < len(shape):
    raise ValueError("iota dimension must be between 0 and len(shape), got "
                     f"{dimension=} for {shape=}")
  if (not dyn_shape and
      not any(isinstance(d, core.DArray) and
              type(core.get_aval(d).dtype) is core.bint for d in shape)):
    return ShapedArray(shape, dtype, sharding=sharding)
  # TODO(mattjj): unify DShapedArray with ShapedArray, and remove this code
  return core.DShapedArray(_merge_dyn_shape(shape, dyn_shape), dtype, False)


iota_p = Primitive('iota')
iota_p.def_impl(partial(dispatch.apply_primitive, iota_p))
iota_p.def_abstract_eval(_iota_abstract_eval)
batching.ragged_prop_rules[iota_p] = batching.ragged_mask_no_op_rule

def _iota_staging_rule(trace, *dyn_shape, dtype, shape, dimension, sharding):
  params = dict(dtype=dtype, shape=shape, dimension=dimension,
                sharding=sharding)
  if not dyn_shape:
    return trace.default_process_primitive(iota_p, (), params)
  aval = core.DShapedArray(_merge_dyn_shape(shape, dyn_shape), dtype, False)
  return _dyn_shape_staging_rule(trace, iota_p, aval, *dyn_shape, **params)
pe.custom_staging_rules[iota_p] = _iota_staging_rule

def _iota_typecheck_rule(_, *dyn_shape, dtype, shape, dimension, sharding):
  if not dyn_shape:
    out_aval, effects = iota_p.abstract_eval(
        dtype=dtype, shape=shape, dimension=dimension, sharding=sharding)
    return [out_aval], effects
  else:
    out_shape = _merge_dyn_shape(shape, dyn_shape)
    out_shape = [x.val if type(x) is core.Literal else x for x in out_shape]  # pytype: disable=attribute-error
    out_aval = core.DShapedArray(tuple(out_shape), dtype, False)
    return [out_aval], core.no_effects
core.custom_typechecks[iota_p] = _iota_typecheck_rule

def _iota_lower(ctx, *dyn_shape, dtype, shape, dimension, sharding):
  del dtype
  aval_out, = ctx.avals_out
  if dyn_shape:
    aval_out = aval_out.update(shape=_merge_dyn_shape(shape, dyn_shape))
  out = mlir.iota(ctx, aval_out, dimension=dimension)
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]
mlir.register_lowering(iota_p, _iota_lower)

def _iota_batching_rule(in_vals, in_dims, *, dtype, shape, dimension,
                        sharding):
  (segment_lengths,), (ax,) = in_vals, in_dims
  assert ax == 0
  bound = segment_lengths.dtype.bound
  ragged_axis, = (i for i, dim in enumerate(shape) if dim is None)
  shape = (len(segment_lengths),) + _merge_dyn_shape(shape, (bound,))
  if sharding is not None:
    raise NotImplementedError('Please file an issue if you want this support')
  iota = broadcasted_iota(dtype, shape, dimension+1)
  return iota, batching.RaggedAxis(ax, ((ragged_axis+1, segment_lengths),))
batching.primitive_batchers[iota_p] = _iota_batching_rule

def _iota_padding_rule(in_avals, out_avals, *dyn_shape, dtype, shape, dimension,
                       sharding):
  out_aval, = out_avals
  new_shape = []
  new_dyn_shape = []
  for d in out_aval.shape:
    if type(d) is pe.BoundedAxisSize:
      new_shape.append(d.bound)
    elif type(d) is int:
      new_shape.append(d)
    else:
      assert isinstance(d, core.Tracer)
      new_shape.append(None)
      new_dyn_shape.append(d)
  if sharding is not None:
    raise NotImplementedError('Please file an issue if you want this support')
  return [iota_p.bind(*new_dyn_shape, shape=tuple(new_shape),
                      dtype=dtype, dimension=dimension, sharding=sharding)]
pe.padding_rules[iota_p] = _iota_padding_rule


### util

_ndim = np.ndim


def _dilate_shape(shape, dilation):
  """Utility function for computing the shape resulting from a dilation."""
  if not np.all(np.greater(dilation, 0)):
    msg = "All dilations must be positive, got {}."
    raise TypeError(msg.format(dilation))
  dilation = (1,) * (len(shape) - len(dilation)) + tuple(dilation)
  return tuple(map(core.dilate_dim, shape, dilation))

def _ceil_divide(x1, x2):
  return -np.floor_divide(np.negative(x1), x2)


class PaddingType(enum.Enum):
  VALID = 1
  SAME = 2
  SAME_LOWER = 3


def padtype_to_pads(in_shape, window_shape, window_strides, padding):
  """Convert padding string to list of pairs of pad values."""

  if isinstance(padding, str):
    mapping = {
        'VALID': PaddingType.VALID,
        'SAME': PaddingType.SAME,
        'SAME_LOWER': PaddingType.SAME_LOWER,
    }
    try:
      padding = mapping[padding.upper()]
    except KeyError as err:
      msg = "Unrecognized padding type: expected 'VALID' or 'SAME', got {}."
      raise RuntimeError(msg.format(padding)) from err

  if padding == PaddingType.SAME or padding == PaddingType.SAME_LOWER:
    out_shape = _ceil_divide(in_shape, window_strides)
    pad_sizes = (core.max_dim(d, 0)
                 for d in (out_shape - 1) * window_strides +
                          window_shape - in_shape)
    if padding == PaddingType.SAME:
      pads = [
          (pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes
      ]
    else:
      pads = [
          (pad_size - pad_size // 2, pad_size // 2) for pad_size in pad_sizes
      ]
    # Avoids verbose numpy scalars in jaxprs.
    return [p.item() if isinstance(p, np.generic) else p for p in pads]
  elif padding == PaddingType.VALID:
    return [(0, 0)] * len(in_shape)
  else:
    msg = "Unknown padding type: {}."
    raise TypeError(msg.format(padding))


# Map of lax function to equivalent jax.numpy function for use in error string below.
_JNP_FUNCTION_EQUIVALENTS = {
  'abs': 'fabs',
  'acos': 'arccos',
  'acosh': 'arccosh',
  'add': 'add',
  'asin': 'arcsin',
  'asinh': 'arcsinh',
  'atan': 'arctan',
  'atan2': 'arctan2',
  'atanh': 'arctanh',
  'bitwise_and': 'bitwise_and',
  'bitwise_not': 'bitwise_not',
  'bitwise_or': 'bitwise_or',
  'bitwise_xor': 'bitwise_xor',
  'cbrt': 'cbrt',
  'ceil': 'ceil',
  'concatenate': 'concatenate',
  'cos': 'cos',
  'cosh': 'cosh',
  'div': 'divide',
  'eq': 'equal',
  'exp': 'exp',
  'expm1': 'expm1',
  'floor': 'floor',
  'greater': 'greater',
  'greater_equal': 'greater_equal',
  'less': 'less',
  'less_equal': 'less_equal',
  'log': 'log',
  'logical_and': 'logical_and',
  'logical_not': 'logical_not',
  'logical_or': 'logical_or',
  'logical_xor': 'logical_xor',
  'log1p': 'log1p',
  'max': 'maximum',
  'min': 'minimum',
  'mul': 'multiply',
  'ne': 'not_equal',
  'neg': 'negative',
  'nextafter': 'nextafter',
  'pow': 'float_power',
  'round': 'round',
  'select': 'where',
  'shift_left': 'left_shift',
  'shift_right_logical': 'right_shift',
  'shift_right_arithmetic': 'right_shift',
  'sign': 'sign',
  'sin': 'sin',
  'sinh': 'sinh',
  'sqrt': 'sqrt',
  'sub': 'subtract',
  'tan': 'tan',
  'tanh': 'tanh'
}

def check_same_dtypes(name: str, *avals: core.UnshapedArray) -> None:
  """Check that dtypes agree, possibly ignoring float precision."""
  # the `ignore_fp_precision` flag exists because the XLA shape inference logic
  # allows mixed floating point precision, but the HLO verifier often rejects it
  if any(dtypes.issubdtype(aval.dtype, dtypes.extended) for aval in avals):
    return  # TODO(mattjj,frostig): do some checking, friend
  if len(avals) < 2:
    return

  dtype = dtypes.canonicalize_dtype(avals[0].dtype)
  if any(dtypes.canonicalize_dtype(aval.dtype) != dtype for aval in avals[1:]):
    msg = "lax.{} requires arguments to have the same dtypes, got {}."
    if name in _JNP_FUNCTION_EQUIVALENTS:
      equiv = _JNP_FUNCTION_EQUIVALENTS[name]
      msg += f" (Tip: jnp.{equiv} is a similar function that does automatic type promotion on inputs)."
    raise TypeError(msg.format(name, ", ".join(str(a.dtype) for a in avals)))


def _check_shapelike(fun_name, arg_name, obj, non_zero_shape=False):
  """Check that `obj` is a shape-like value (e.g. tuple of nonnegative ints)."""
  if not isinstance(obj, (tuple, list, np.ndarray)):
    msg = "{} {} must be of type tuple/list/ndarray, got {}."
    raise TypeError(msg.format(fun_name, arg_name, type(obj)))
  # bool(obj) for an ndarray raises an error, so we check len
  if not len(obj):  # pylint: disable=g-explicit-length-test
    return
  if (config.dynamic_shapes.value and isinstance(obj, (tuple, list)) and
      any(isinstance(d, (core.Tracer, core.DArray)) for d in obj)):
    return  # TODO(mattjj): handle more checks in the dynamic shape case
  obj_arr = np.array(obj)
  if obj_arr.ndim != 1:
    msg = "{} {} must be 1-dimensional, got {}."
    raise TypeError(msg.format(obj_arr.ndim))
  try:
    canonicalize_shape(obj_arr)
  except TypeError as err:
    msg = "{} {} must have every element be an integer type, got {}."
    raise TypeError(msg.format(fun_name, arg_name, tuple(map(type, obj)))) from err
  lower_bound, bound_error = (
      (1, "strictly positive") if non_zero_shape else (0, "nonnegative"))
  if not all(d >= lower_bound for d in obj_arr):
    msg = "{} {} must have every element be {}, got {}."
    raise TypeError(msg.format(fun_name, arg_name, bound_error, obj))


def _const(example, val):
  dtype = _dtype(example)
  if dtypes.is_python_scalar(example):
    val = dtypes.scalar_type_of(example)(val)
    return val if dtype == _dtype(val) else np.array(val, dtype)
  return np.array(val, dtype)

_zeros: Callable = partial(full_like, fill_value=0)

def _zero(x):
  if config.sharding_in_types.value:
    return full_like(x, shape=(), fill_value=0,
                     sharding=x.sharding.with_spec(P()))  # type: ignore
  return full_like(x, shape=(), fill_value=0)

_ones: Callable = partial(full_like, fill_value=1)

def _one(x):
  if config.sharding_in_types.value:
    return full_like(x, shape=(), fill_value=1,
                     sharding=x.sharding.with_spec(P()))
  return full_like(x, shape=(), fill_value=1)

_twos: Callable = partial(full_like, fill_value=2)
_two: Callable = partial(full_like, shape=(), fill_value=2)

dtype: Callable = partial(dtypes.dtype, canonicalize=True)
_dtype: Callable = partial(dtypes.dtype, canonicalize=True)

def _isnan(x: ArrayLike) -> Array:
  return ne(x, x)

def _iscomplex(x) -> bool:
  return dtypes.issubdtype(_dtype(x), np.complexfloating)


def ranges_like(*xs):
  start = 0
  for x in xs:
    x_len = len(x)
    yield range(start, start + x_len)
    start += x_len


def remaining(original, *removed_lists):
  removed = set(itertools.chain(*removed_lists))
  return [i for i in original if i not in removed]


def canonicalize_precision(precision: PrecisionLike) -> CanonicalPrecision:
  """Turns an API precision specification into a pair of enumeration values.

  The API can take the precision as a string, or int, and either as a single
  value to apply to both operands, or as a sequence of two values.
  """
  if precision is None:
    if config.default_matmul_precision.value is None:
      return None
    try:
      return canonicalize_precision(config.default_matmul_precision.value)
    except ValueError:
      raise ValueError(
          "jax_default_matmul_precision flag must be set to None, a value in "
          f"{list(_precision_strings)}, or the name of a lax.DotAlgorithmPreset, "
          f"but got {config.default_matmul_precision.value}"
      ) from None
  elif isinstance(precision, str):
    if precision in _precision_strings:
      return Precision(precision), Precision(precision)
    else:
      try:
        return DotAlgorithmPreset[precision]
      except KeyError:
        pass
  elif isinstance(precision, Precision):
    return precision, precision
  elif isinstance(precision, (DotAlgorithm, DotAlgorithmPreset)):
    return precision
  elif (isinstance(precision, (list, tuple)) and len(precision) == 2 and
        all(isinstance(p, Precision) for p in precision)):
    return type_cast(tuple[Precision, Precision], precision)
  elif (isinstance(precision, (list, tuple)) and len(precision) == 2 and
        all(isinstance(s, str) for s in precision)):
    s1, s2 = type_cast(tuple[str, str], precision)
    p1 = type_cast(tuple[Precision, Precision], canonicalize_precision(s1))[0]
    p2 = type_cast(tuple[Precision, Precision], canonicalize_precision(s2))[0]
    return (p1, p2)
  raise ValueError(
      "Precision argument must be one of:\n"
      "- None,\n"
      f"- a string in {list(_precision_strings)},\n"
      "- a lax.Precision value,\n"
      "- a tuple of two lax.Precision values or strings,\n"
      "- a lax.DotAlgorithmPreset or the name of one of these presets, or\n"
      "- a lax.DotAlgorithm value;\n"
      f"but got {precision}.")


def _balanced_eq(x, z, y):
  return div(select(_eq_meet(x, z), _ones(z), _zeros(z)),
             select(_eq_meet(y, z), _twos(z), _ones(z)))


def _eq_meet(a, b):
  a_dtype, b_dtype = _dtype(a), _dtype(b)
  if a_dtype != b_dtype:
    higher_dtype = dtypes.promote_types(a_dtype, b_dtype)
    if higher_dtype == a_dtype:
      a = convert_element_type(a, b_dtype)
    else:
      b = convert_element_type(b, a_dtype)
  return eq(a, b)


def empty(dtype):
  return empty_p.bind(dtype=dtype)
empty_p = core.Primitive('empty')
empty_p.def_abstract_eval(lambda *, dtype: core.ShapedArray((), dtype))
def _empty_lower(ctx, *, dtype):
  dtype = dtype if dtypes.issubdtype(dtype, dtypes.extended) else np.dtype(dtype)
  phys_aval = core.physical_aval(core.ShapedArray((), dtype))
  return mlir.ir_constant(np.zeros(phys_aval.shape, phys_aval.dtype)),
mlir.register_lowering(empty_p, _empty_lower)


tie_p = core.Primitive('tie')
tie_p.def_impl(lambda x, y: y)
tie_p.def_abstract_eval(lambda x, y: y)
mlir.register_lowering(tie_p, lambda ctx, x, y: [y])
ad.primitive_jvps[tie_p] = \
    lambda primals, tangents: (tie_p.bind(*primals), tangents[-1])
ad.primitive_transposes[tie_p] = lambda ct, x, _: [None, ct]
pe.def_trivial_padding(tie_p)
batching.defvectorized(tie_p)


class BIntRules:
  allow_conversion: bool = True

  @staticmethod
  def physical_element_aval(dtype) -> core.ShapedArray:
    return core.ShapedArray((), np.dtype('int32'))

  @staticmethod
  def result_handler(sticky_device, aval):
    def handler(_, buf):
      buf.aval = core.ShapedArray(buf.shape, buf.dtype)
      return core.DArray(aval, buf)
    return handler

  @staticmethod
  def global_sharded_result_handler(aval, out_sharding, committed):
    phys_aval = core.physical_aval(aval)
    phys_handler_maker = pxla.global_result_handlers[core.ShapedArray]

    if not dispatch.is_single_device_sharding(out_sharding):
      raise NotImplementedError  # TODO(mattjj)
    else:
      phys_sharding = out_sharding
    phys_handler = phys_handler_maker(phys_aval, phys_sharding, committed)

    def handler(bufs):
      return core.DArray(aval, phys_handler(bufs))
    return handler


core.bint._rules = BIntRules


def optimization_barrier(operand, /):
  """Prevents the compiler from moving operations across the barrier.

  Optimization barriers have a number of possible uses:

  * An optimization barrier ensures that all inputs are evaluated before any
    operators that depend on the barrier's outputs. This can be used to enforce
    a particular order of operations.
  * An optimization barrier prevents common subexpression elimination. This is
    used by JAX to implement rematerialization.
  * Optimization barriers prevent compiler fusions. That is, operations before
    the barrier may not be fused into the same kernel as operations after the
    barrier by the compiler.

  JAX does not define derivative or batching rules for an optimization barrier.

  Optimization barriers have no effect outside a compiled function.

  Args:
    operand: a pytree of JAX values.

  Returns:
    A pytree of JAX values, with the same structure and contents as ``operand``.

  Examples:
    Prevents common-subexpression elimination between the two calls to `sin`:

    >>> def f(x):
    ...   return jax.lax.optimization_barrier(jax.lax.sin(x)) + jax.lax.sin(x)
    >>> jax.jit(f)(0.)
    Array(0., dtype=float32, weak_type=True)
  """
  flat_args, treedef = tree_util.tree_flatten(operand)
  return tree_util.tree_unflatten(
    treedef, optimization_barrier_p.bind(*flat_args))


def _optimization_barrier_abstract_eval(*args):
  return args

def _optimization_barrier_lowering_rule(ctx, *args):
  barrier_types = map(mlir.aval_to_ir_type, ctx.avals_in)
  flat_args = mlir.flatten_ir_values(args)
  barrier_op = hlo.OptimizationBarrierOp(flat_args)
  return mlir.unflatten_ir_values_like_types(barrier_op.results, barrier_types)


optimization_barrier_p = core.Primitive('optimization_barrier')
optimization_barrier_p.multiple_results = True
optimization_barrier_p.def_impl(
    partial(dispatch.apply_primitive, optimization_barrier_p))
optimization_barrier_p.def_abstract_eval(_optimization_barrier_abstract_eval)
mlir.register_lowering(optimization_barrier_p,
                       _optimization_barrier_lowering_rule)

def _optimization_barrier_batcher(batched_args, batch_dims, **params):
  return optimization_barrier_p.bind(*batched_args, **params), batch_dims
batching.primitive_batchers[optimization_barrier_p] = _optimization_barrier_batcher
