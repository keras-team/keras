# Copyright 2019 The JAX Authors.
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

# Array type functions.
#
# JAX dtypes differ from NumPy in both:
# a) their type promotion rules, and
# b) the set of supported types (e.g., bfloat16),
# so we need our own implementation that deviates from NumPy in places.

from __future__ import annotations

import abc
import builtins
import dataclasses
import functools
import types
from typing import cast, overload, Any, Literal, Union
import warnings

import ml_dtypes
import numpy as np

from jax._src import config
from jax._src.typing import Array, DType, DTypeLike
from jax._src.util import set_module, StrictABC

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

try:
  _ml_dtypes_version = tuple(map(int, ml_dtypes.__version__.split('.')[:3]))
except:
  pass
else:
  if _ml_dtypes_version < (0, 2, 0):
    raise ValueError("JAX requires ml_dtypes version 0.2.0 or newer; "
                     f"installed version is {ml_dtypes.__version__}.")

export = set_module('jax.dtypes')

@export
class extended(np.generic):
  """Scalar class for extended dtypes.

  This is an abstract class that should never be instantiated, but rather
  exists for the sake of `jnp.issubdtype`.

  Examples:
    >>> from jax import random
    >>> from jax import dtypes
    >>> key = random.key(0)
    >>> jnp.issubdtype(key.dtype, dtypes.extended)
    True
  """


@export
class prng_key(extended):
  """Scalar class for PRNG Key dtypes.

  This is an abstract class that should never be instantiated, but rather
  exists for the sake of `jnp.issubdtype`.

  Examples:
    >>> from jax import random
    >>> from jax import dtypes
    >>> key = random.key(0)
    >>> jnp.issubdtype(key.dtype, dtypes.prng_key)
    True
  """


class ExtendedDType(StrictABC):
  """Abstract Base Class for extended dtypes"""
  @property
  @abc.abstractmethod
  def type(self) -> type: ...


# fp8 support
# TODO: remove Optional when minimum ml_dtypes version >= 0.5.0
float8_e3m4: type[np.generic] | None = None
float8_e4m3: type[np.generic] | None = None
float8_e4m3b11fnuz: type[np.generic] = ml_dtypes.float8_e4m3b11fnuz
float8_e4m3fn: type[np.generic] = ml_dtypes.float8_e4m3fn
float8_e4m3fnuz: type[np.generic] = ml_dtypes.float8_e4m3fnuz
float8_e5m2: type[np.generic] = ml_dtypes.float8_e5m2
float8_e5m2fnuz: type[np.generic] = ml_dtypes.float8_e5m2fnuz

_float8_e3m4_dtype: np.dtype | None = None
_float8_e4m3_dtype: np.dtype | None = None
_float8_e4m3b11fnuz_dtype: np.dtype = np.dtype(float8_e4m3b11fnuz)
_float8_e4m3fn_dtype: np.dtype = np.dtype(float8_e4m3fn)
_float8_e4m3fnuz_dtype: np.dtype = np.dtype(float8_e4m3fnuz)
_float8_e5m2_dtype: np.dtype = np.dtype(float8_e5m2)
_float8_e5m2fnuz_dtype: np.dtype = np.dtype(float8_e5m2fnuz)

def supports_inf(dtype: DTypeLike) -> bool:
  """Return true if the dtype supports infinity, else return False."""
  typ = np.dtype(dtype).type
  if typ in {float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz, float8_e5m2fnuz}:
    return False
  return issubdtype(dtype, np.inexact)

# bfloat16 support
bfloat16: type[np.generic] = ml_dtypes.bfloat16
_bfloat16_dtype: np.dtype = np.dtype(bfloat16)

_custom_float_scalar_types = [
    float8_e4m3b11fnuz,
    float8_e4m3fn,
    float8_e4m3fnuz,
    float8_e5m2,
    float8_e5m2fnuz,
    bfloat16,
]
_custom_float_dtypes = [
    _float8_e4m3b11fnuz_dtype,
    _float8_e4m3fn_dtype,
    _float8_e4m3fnuz_dtype,
    _float8_e5m2_dtype,
    _float8_e5m2fnuz_dtype,
    _bfloat16_dtype,
]
_float8_dtypes = [
    _float8_e4m3b11fnuz_dtype,
    _float8_e4m3fn_dtype,
    _float8_e4m3fnuz_dtype,
    _float8_e5m2_dtype,
    _float8_e5m2fnuz_dtype,
]

# TODO: remove the if statements below when minimum ml_dtypes version >= 0.5.0
if hasattr(ml_dtypes, "float8_e4m3"):
  float8_e4m3 = ml_dtypes.float8_e4m3
  _float8_e4m3_dtype = np.dtype(float8_e4m3)
  _custom_float_scalar_types.insert(0, float8_e4m3)  # type: ignore[arg-type]
  _custom_float_dtypes.insert(0, _float8_e4m3_dtype)
  _float8_dtypes.insert(0, _float8_e4m3_dtype)
if hasattr(ml_dtypes, "float8_e3m4"):
  float8_e3m4 = ml_dtypes.float8_e3m4
  _float8_e3m4_dtype = np.dtype(float8_e3m4)
  _custom_float_scalar_types.insert(0, float8_e3m4)  # type: ignore[arg-type]
  _custom_float_dtypes.insert(0, _float8_e3m4_dtype)
  _float8_dtypes.insert(0, _float8_e3m4_dtype)

# 2-bit integer support
int2: type[np.generic] | None = None
uint2: type[np.generic] | None = None

_int2_dtype: np.dtype | None = None
_uint2_dtype: np.dtype | None = None

_intn_dtypes = []

# Remove the condition once the minimum ml_dtypes version required by JAX
# contains https://github.com/jax-ml/ml_dtypes/pull/154.
if hasattr(ml_dtypes, 'int2'):
  int2 = ml_dtypes.int2
  uint2 = ml_dtypes.uint2
  _int2_dtype = np.dtype(int2)
  _uint2_dtype = np.dtype(uint2)
  _intn_dtypes.extend([_int2_dtype, _uint2_dtype])

# 4-bit integer support
int4: type[np.generic] = ml_dtypes.int4
uint4: type[np.generic] = ml_dtypes.uint4
_int4_dtype = np.dtype(int4)
_uint4_dtype = np.dtype(uint4)
_intn_dtypes.extend([_int4_dtype, _uint4_dtype])

# Default types.
bool_ = np.bool_
int_: type[Any]
uint: type[Any]
float_: type[Any]
complex_: type[Any]
if config.default_dtype_bits.value == '32':
  int_ = np.int32
  uint = np.uint32
  float_ = np.float32
  complex_ = np.complex64
else:
  int_ = np.int64
  uint = np.uint64
  float_ = np.float64
  complex_ = np.complex128
_default_types: dict[str, type[Any]] = {
    'b': bool_,
    'i': int_,
    'u': uint,
    'f': float_,
    'c': complex_,
}

def bit_width(dtype: DTypeLike) -> int:
  """Number of bits per element for the dtype."""
  # Note: we cannot use dtype.itemsize here because this is
  # incorrect for sub-byte integer types.
  if dtype == np.dtype(bool):
    return 8  # physical bit layout for boolean dtype
  elif issubdtype(dtype, np.integer):
    return iinfo(dtype).bits
  elif issubdtype(dtype, np.floating):
    return finfo(dtype).bits
  elif issubdtype(dtype, np.complexfloating):
    return 2 * finfo(dtype).bits
  else:
    raise ValueError(f"unexpected input: {dtype=}")

# Trivial vectorspace datatype needed for tangent values of int/bool primals
float0: np.dtype = np.dtype([('float0', np.void, 0)])

_dtype_to_32bit_dtype: dict[DType, DType] = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}

# Note: we promote narrow types to float32 here for backward compatibility
# with earlier approaches. We might consider revisiting this, or perhaps
# tying the logic more closely to the type promotion lattice.
_dtype_to_inexact: dict[DType, DType] = {
    np.dtype(k): np.dtype(v) for k, v in [
        ('bool', 'float32'),
        ('uint8', 'float32'), ('int8', 'float32'),
        ('uint16', 'float32'), ('int16', 'float32'),
        ('uint32', 'float32'), ('int32', 'float32'),
        ('uint64', 'float64'), ('int64', 'float64')
    ]
}

def to_numeric_dtype(dtype: DTypeLike) -> DType:
  """Promotes a dtype into an numeric dtype, if it is not already one."""
  dtype_ = np.dtype(dtype)
  return np.dtype('int32') if dtype_ == np.dtype('bool') else dtype_


def to_inexact_dtype(dtype: DTypeLike) -> DType:
  """Promotes a dtype into an inexact dtype, if it is not already one."""
  dtype_ = np.dtype(dtype)
  return _dtype_to_inexact.get(dtype_, dtype_)


def to_complex_dtype(dtype: DTypeLike) -> DType:
  ftype = to_inexact_dtype(dtype)
  if ftype in [np.dtype('float64'), np.dtype('complex128')]:
    return np.dtype('complex128')
  return np.dtype('complex64')


@functools.cache
def _canonicalize_dtype(x64_enabled: bool, allow_extended_dtype: bool, dtype: Any) -> DType | ExtendedDType:
  if issubdtype(dtype, extended):
    if not allow_extended_dtype:
      raise ValueError(f"Internal: canonicalize_dtype called on extended dtype {dtype} "
                       "with allow_extended_dtype=False")
    return dtype
  try:
    dtype_ = np.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if x64_enabled:
    return dtype_
  else:
    return _dtype_to_32bit_dtype.get(dtype_, dtype_)

@overload
def canonicalize_dtype(dtype: Any, allow_extended_dtype: Literal[False] = False) -> DType: ...

@overload
def canonicalize_dtype(dtype: Any, allow_extended_dtype: bool = False) -> DType | ExtendedDType: ...

@export
def canonicalize_dtype(dtype: Any, allow_extended_dtype: bool = False) -> DType | ExtendedDType:
  """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
  return _canonicalize_dtype(config.enable_x64.value, allow_extended_dtype, dtype)  # pytype: disable=bad-return-type

# Default dtypes corresponding to Python scalars.
python_scalar_dtypes : dict[type, DType] = {
  bool: np.dtype('bool'),
  int: np.dtype('int64'),
  float: np.dtype('float64'),
  complex: np.dtype('complex128'),
}

@export
def scalar_type_of(x: Any) -> type:
  """Return the scalar type associated with a JAX value."""
  typ = dtype(x)
  if typ in _custom_float_dtypes:
    return float
  elif typ in _intn_dtypes:
    return int
  elif np.issubdtype(typ, np.bool_):
    return bool
  elif np.issubdtype(typ, np.integer):
    return int
  elif np.issubdtype(typ, np.floating):
    return float
  elif np.issubdtype(typ, np.complexfloating):
    return complex
  else:
    raise TypeError(f"Invalid scalar value {x}")


def _scalar_type_to_dtype(typ: type, value: Any = None) -> DType:
  """Return the numpy dtype for the given scalar type.

  Raises
  ------
  OverflowError: if `typ` is `int` and the value is too large for int64.

  Examples
  --------
  >>> _scalar_type_to_dtype(int)
  dtype('int32')
  >>> _scalar_type_to_dtype(float)
  dtype('float32')
  >>> _scalar_type_to_dtype(complex)
  dtype('complex64')
  >>> _scalar_type_to_dtype(int)
  dtype('int32')
  >>> _scalar_type_to_dtype(int, 0)
  dtype('int32')
  >>> _scalar_type_to_dtype(int, 1 << 63)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  OverflowError: Python int 9223372036854775808 too large to convert to int32
  """
  dtype = canonicalize_dtype(python_scalar_dtypes[typ])
  if typ is int and value is not None:
    if value < np.iinfo(dtype).min or value > np.iinfo(dtype).max:
      raise OverflowError(f"Python int {value} too large to convert to {dtype}")
  return dtype


def coerce_to_array(x: Any, dtype: DTypeLike | None = None) -> np.ndarray:
  """Coerces a scalar or NumPy array to an np.array.

  Handles Python scalar type promotion according to JAX's rules, not NumPy's
  rules.
  """
  if dtype is None and type(x) in python_scalar_dtypes:
    dtype = _scalar_type_to_dtype(type(x), x)
  return np.asarray(x, dtype)

iinfo = ml_dtypes.iinfo
finfo = ml_dtypes.finfo

def _issubclass(a: Any, b: Any) -> bool:
  """Determines if ``a`` is a subclass of ``b``.

  Similar to issubclass, but returns False instead of an exception if `a` is not
  a class.
  """
  try:
    return issubclass(a, b)
  except TypeError:
    return False


_types_for_issubdtype = (type, np.dtype, ExtendedDType)

# TODO(jakevdp): consider whether to disallow None here. We allow it
# because np.issubdtype allows it (and treats it as equivalent to float64).
@set_module('jax.numpy')
def issubdtype(a: DTypeLike | ExtendedDType | None,
               b: DTypeLike | ExtendedDType | None) -> bool:
  """Returns True if first argument is a typecode lower/equal in type hierarchy.

  This is like :func:`numpy.issubdtype`, but can handle dtype extensions such as
  :obj:`jax.dtypes.bfloat16` and `jax.dtypes.prng_key`.
  """
  # Main departures from np.issubdtype are:
  # - "extended" dtypes (like prng key types) are not normal numpy dtypes, so we
  #   need to handle them specifically. However, their scalar types do conform to
  #   the numpy scalar type hierarchy.
  # - custom dtypes (like bfloat16, int4, etc.) are normal numpy dtypes, but they
  #   don't conform to the standard numpy type hierarchy (e.g. the bfloat16 scalar
  #   type is not a subclass of np.floating) so we must also handle these specially.

  # We cannot use the cached version directly for all inputs, because some may be
  # unhashable (e.g. custom objects with a dtype attribute). The following check is
  # fast and covers the majority of calls to this function within JAX library code.
  return _issubdtype_cached(
    a if isinstance(a, _types_for_issubdtype) else np.dtype(a),  # type: ignore[arg-type]
    b if isinstance(b, _types_for_issubdtype) else np.dtype(b),  # type: ignore[arg-type]
  )


@functools.lru_cache(512)  # don't use util.memoize because there is no X64 dependence.
def _issubdtype_cached(a: type | np.dtype | ExtendedDType,
                       b: type | np.dtype | ExtendedDType) -> bool:
  # First handle extended dtypes, which require their own logic.
  a_is_type = isinstance(a, type)
  b_is_type = isinstance(b, type)
  if b_is_type and _issubclass(b, extended):
    if isinstance(a, ExtendedDType):
      return _issubclass(a.type, b)
    if a_is_type and _issubclass(a, np.generic):
      return _issubclass(a, b)
    return _issubclass(np.dtype(a).type, b)
  if isinstance(b, ExtendedDType):
    return isinstance(a, ExtendedDType) and a == b
  if isinstance(a, ExtendedDType):
    a = a.type
    a_is_type = isinstance(a, type)

  # For all others, normalize inputs to scalar types.
  a_sctype = a if a_is_type and _issubclass(a, np.generic) else np.dtype(a).type
  b_sctype = b if b_is_type and _issubclass(b, np.generic) else np.dtype(b).type

  # Now do special handling of custom float and int types, as they don't conform
  # to the normal scalar type hierarchy.
  if a_sctype in _custom_float_scalar_types:
    return b_sctype in {a_sctype, np.floating, np.inexact, np.number, np.generic}
  if (int2 is not None and a_sctype == int2) or a_sctype == int4:
    return b_sctype in {a_sctype, np.signedinteger, np.integer, np.number, np.generic}
  if (uint2 is not None and a_sctype == uint2) or a_sctype == uint4:
    return b_sctype in {a_sctype, np.unsignedinteger, np.integer, np.number, np.generic}

  # Otherwise, fall back to numpy.issubdtype
  return bool(np.issubdtype(a_sctype, b_sctype))

can_cast = np.can_cast

JAXType = Union[type, DType]

# Enumeration of all valid JAX types in order.
_weak_types: list[JAXType] = [int, float, complex]
_bool_types: list[JAXType] = [np.dtype(bool)]
_signed_types: list[JAXType]
_unsigned_types: list[JAXType]
_int_types: list[JAXType]
_unsigned_types = [
    np.dtype(uint4),
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
]
_signed_types = [
    np.dtype(int4),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
]

if _int2_dtype is not None:
  _signed_types.insert(0, _int2_dtype)
if _uint2_dtype is not None:
  _unsigned_types.insert(0, _uint2_dtype)

_int_types = _unsigned_types + _signed_types

_float_types: list[JAXType] = [
    *_custom_float_dtypes,
    np.dtype('float16'),
    np.dtype('float32'),
    np.dtype('float64'),
]
_complex_types: list[JAXType] = [
    np.dtype('complex64'),
    np.dtype('complex128'),
]
_jax_types = _bool_types + _int_types + _float_types + _complex_types
_jax_dtype_set = {float0, *_bool_types, *_int_types, *_float_types, *_complex_types}


_dtype_kinds: dict[str, set] = {
  'bool': {*_bool_types},
  'signed integer': {*_signed_types},
  'unsigned integer': {*_unsigned_types},
  'integral': {*_signed_types, *_unsigned_types},
  'real floating': {*_float_types},
  'complex floating': {*_complex_types},
  'numeric': {*_signed_types, *_unsigned_types, *_float_types, *_complex_types},
}


@set_module('jax.numpy')
def isdtype(dtype: DTypeLike, kind: str | DTypeLike | tuple[str | DTypeLike, ...]) -> bool:
  """Returns a boolean indicating whether a provided dtype is of a specified kind.

  Args:
    dtype : the input dtype
    kind : the data type kind.
      If ``kind`` is dtype-like, return ``dtype = kind``.
      If ``kind`` is a string, then return True if the dtype is in the specified category:

      - ``'bool'``: ``{bool}``
      - ``'signed integer'``: ``{int4, int8, int16, int32, int64}``
      - ``'unsigned integer'``: ``{uint4, uint8, uint16, uint32, uint64}``
      - ``'integral'``: shorthand for ``('signed integer', 'unsigned integer')``
      - ``'real floating'``: ``{float8_*, float16, bfloat16, float32, float64}``
      - ``'complex floating'``: ``{complex64, complex128}``
      - ``'numeric'``: shorthand for ``('integral', 'real floating', 'complex floating')``

      If ``kind`` is a tuple, then return True if dtype matches any entry of the tuple.

  Returns:
    True or False
  """
  the_dtype = np.dtype(dtype)
  kind_tuple: tuple[str | DTypeLike, ...] = (
    kind if isinstance(kind, tuple) else (kind,)
  )
  options: set[DType] = set()
  for kind in kind_tuple:
    if isinstance(kind, str) and kind in _dtype_kinds:
      options.update(_dtype_kinds[kind])
      continue
    try:
      _dtype = np.dtype(kind)
    except TypeError as e:
      if isinstance(kind, str):
        raise ValueError(
          f"Unrecognized {kind=} expected one of {list(_dtype_kinds.keys())}, "
          "or a compatible input for jnp.dtype()")
      raise TypeError(
        f"Expected kind to be a dtype, string, or tuple; got {kind=}"
      ) from e
    options.add(_dtype)
  return the_dtype in options


def _jax_type(dtype: DType, weak_type: bool) -> JAXType:
  """Return the jax type for a dtype and weak type."""
  if weak_type:
    if dtype == bool:
      return dtype
    if dtype in _custom_float_dtypes:
      return float
    return type(dtype.type(0).item())
  return dtype

def _dtype_and_weaktype(value: Any) -> tuple[DType, bool]:
  """Return a (dtype, weak_type) tuple for the given input."""
  return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)

def _type_promotion_lattice(jax_numpy_dtype_promotion: str) -> dict[JAXType, list[JAXType]]:
  """
  Return the type promotion lattice in the form of a DAG.
  This DAG maps each type to its immediately higher type on the lattice.
  """
  b1, = _bool_types
  if _int2_dtype is not None:
    assert _uint2_dtype is not None
    _uint2, uint4, u1, u2, u4, u8, _int2, int4, i1, i2, i4, i8 = _int_types
  else:
    uint4, u1, u2, u4, u8, int4, i1, i2, i4, i8 = _int_types
  *f1_types, bf, f2, f4, f8 = _float_types
  c4, c8 = _complex_types
  i_, f_, c_ = _weak_types
  if jax_numpy_dtype_promotion == 'standard':
    out: dict[JAXType, list[JAXType]]
    out = {
      b1: [i_],
      i_: [u1, uint4, i1, int4],
      uint4: [], u1: [i2, u2], u2: [i4, u4], u4: [i8, u8], u8: [f_],
      int4: [], i1: [i2], i2: [i4], i4: [i8], i8: [f_],
      f_: [*f1_types, bf, f2, c_],
      **{t: [] for t in f1_types}, bf: [f4], f2: [f4], f4: [f8, c4], f8: [c8],
      c_: [c4], c4: [c8], c8: [],
    }
    if _int2_dtype is not None:
      out[i_].append(_int2_dtype)
      out[_int2_dtype] = []
    if _uint2_dtype is not None:
      out[i_].append(_uint2_dtype)
      out[_uint2_dtype] = []
    return out
  elif jax_numpy_dtype_promotion == 'strict':
    return {
      i_: [f_] + _int_types,
      f_: [c_] + _float_types,
      c_: _complex_types,
      **{t: [] for t in _jax_types}
    }
  else:
    raise ValueError(
      f"Unexpected value of jax_numpy_dtype_promotion={jax_numpy_dtype_promotion!r}")

def _make_lattice_upper_bounds(jax_numpy_dtype_promotion: str) -> dict[JAXType, set[JAXType]]:
  lattice = _type_promotion_lattice(jax_numpy_dtype_promotion)
  upper_bounds = {node: {node} for node in lattice}
  for n in lattice:
    while True:
      new_upper_bounds = set().union(*(lattice[b] for b in upper_bounds[n]))
      if n in new_upper_bounds:
        raise ValueError(f"cycle detected in type promotion lattice for node {n}")
      if new_upper_bounds.issubset(upper_bounds[n]):
        break
      upper_bounds[n] |= new_upper_bounds
  return upper_bounds

_lattice_upper_bounds: dict[str, dict[JAXType, set[JAXType]]] = {
  'standard': _make_lattice_upper_bounds('standard'),
  'strict': _make_lattice_upper_bounds('strict'),
}

class TypePromotionError(ValueError):
  pass

@functools.lru_cache(512)  # don't use util.memoize because there is no X64 dependence.
def _least_upper_bound(jax_numpy_dtype_promotion: str, *nodes: JAXType) -> JAXType:
  """Compute the least upper bound of a set of nodes.

  Args:
    nodes: sequence of entries from _jax_types + _weak_types
  Returns:
    the _jax_type representing the least upper bound of the input nodes
      on the promotion lattice.
  """
  # This function computes the least upper bound of a set of nodes N within a partially
  # ordered set defined by the lattice generated above.
  # Given a partially ordered set S, let the set of upper bounds of n ∈ S be
  #   UB(n) ≡ {m ∈ S | n ≤ m}
  # Further, for a set of nodes N ⊆ S, let the set of common upper bounds be given by
  #   CUB(N) ≡ {a ∈ S | ∀ b ∈ N: a ∈ UB(b)}
  # Then the least upper bound of N is defined as
  #   LUB(N) ≡ {c ∈ CUB(N) | ∀ d ∈ CUB(N), c ≤ d}
  # The definition of an upper bound implies that c ≤ d if and only if d ∈ UB(c),
  # so the LUB can be expressed:
  #   LUB(N) = {c ∈ CUB(N) | ∀ d ∈ CUB(N): d ∈ UB(c)}
  # or, equivalently:
  #   LUB(N) = {c ∈ CUB(N) | CUB(N) ⊆ UB(c)}
  # By definition, LUB(N) has a cardinality of 1 for a partially ordered set.
  # Note a potential algorithmic shortcut: from the definition of CUB(N), we have
  #   ∀ c ∈ N: CUB(N) ⊆ UB(c)
  # So if N ∩ CUB(N) is nonempty, if follows that LUB(N) = N ∩ CUB(N).
  N = set(nodes)
  UB = _lattice_upper_bounds[jax_numpy_dtype_promotion]
  try:
    bounds = [UB[n] for n in N]
  except KeyError:
    dtype = next(n for n in N if n not in UB)
    raise ValueError(f"{dtype=} is not a valid dtype for JAX type promotion.")
  CUB = set.intersection(*bounds)
  LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
  if len(LUB) == 1:
    return LUB.pop()
  elif len(LUB) == 0:
    if config.numpy_dtype_promotion.value == 'strict':
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path when jax_numpy_dtype_promotion=strict. Try explicitly casting "
        "inputs to the desired output type, or set jax_numpy_dtype_promotion=standard.")
    elif any(n in _float8_dtypes for n in nodes):
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. To avoid unintended promotion, 8-bit floats do not support "
        "implicit promotion. If you'd like your inputs to be promoted to another type, "
        "you can do so explicitly using e.g. x.astype('float32')")
    elif any(n in _intn_dtypes for n in nodes):
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. To avoid unintended promotion, 2-bit and 4-bit integers do not "
        "support implicit promotion. If you'd like your inputs to be promoted to another "
        "type, you can do so explicitly using e.g. x.astype('int32')")
    else:
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. Try explicitly casting inputs to the desired output type.")
    raise TypePromotionError(msg)
  else:
    # If we get here, it means the lattice is ill-formed.
    raise TypePromotionError(
      f"Internal Type Promotion error: {nodes} do not have a unique least upper bound "
      f"on the specified lattice; options are {LUB}. This is an unexpected error in "
      "JAX's internal logic; please report it to the JAX maintainers."
    )

@set_module('jax.numpy')
def promote_types(a: DTypeLike, b: DTypeLike) -> DType:
  """Returns the type to which a binary operation should cast its arguments.

  JAX implementation of :func:`numpy.promote_types`. For details of JAX's
  type promotion semantics, see :ref:`type-promotion`.

  Args:
    a: a :class:`numpy.dtype` or a dtype specifier.
    b: a :class:`numpy.dtype` or a dtype specifier.

  Returns:
    A :class:`numpy.dtype` object.

  Examples:
    Type specifiers may be strings, dtypes, or scalar types, and the return
    value is always a dtype:

    >>> jnp.promote_types('int32', 'float32')  # strings
    dtype('float32')
    >>> jnp.promote_types(jnp.dtype('int32'), jnp.dtype('float32'))  # dtypes
    dtype('float32')
    >>> jnp.promote_types(jnp.int32, jnp.float32)  # scalar types
    dtype('float32')

    Built-in scalar types (:type:`int`, :type:`float`, or :type:`complex`) are
    treated as weakly-typed and will not change the bit width of a strongly-typed
    counterpart (see discussion in :ref:`type-promotion`):

    >>> jnp.promote_types('uint8', int)
    dtype('uint8')
    >>> jnp.promote_types('float16', float)
    dtype('float16')

    This differs from the NumPy version of this function, which treats built-in scalar
    types as equivalent to 64-bit types:

    >>> import numpy
    >>> numpy.promote_types('uint8', int)
    dtype('int64')
    >>> numpy.promote_types('float16', float)
    dtype('float64')
  """
  # Note: we deliberately avoid `if a in _weak_types` here because we want to check
  # object identity, not object equality, due to the behavior of np.dtype.__eq__
  a_tp = cast(JAXType, a if any(a is t for t in _weak_types) else np.dtype(a))
  b_tp = cast(JAXType, b if any(b is t for t in _weak_types) else np.dtype(b))
  return np.dtype(_least_upper_bound(config.numpy_dtype_promotion.value, a_tp, b_tp))

def is_weakly_typed(x: Any) -> bool:
  if type(x) in _weak_types:
    return True
  try:
    return x.aval.weak_type
  except AttributeError:
    return False

def is_python_scalar(x: Any) -> bool:
  try:
    return x.aval.weak_type and np.ndim(x) == 0
  except AttributeError:
    return type(x) in python_scalar_dtypes

def check_valid_dtype(dtype: DType) -> None:
  if dtype not in _jax_dtype_set:
    raise TypeError(f"Dtype {dtype} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")

def dtype(x: Any, *, canonicalize: bool = False) -> DType:
  """Return the dtype object for a value or type, optionally canonicalized based on X64 mode."""
  if x is None:
    raise ValueError(f"Invalid argument to dtype: {x}.")
  is_type = isinstance(x, type)
  if is_type and x in python_scalar_dtypes:
    dt = python_scalar_dtypes[x]
  elif type(x) in python_scalar_dtypes:
    dt = python_scalar_dtypes[type(x)]
  elif is_type and _issubclass(x, np.generic):
    return np.dtype(x)
  elif issubdtype(getattr(x, 'dtype', None), extended):
    dt = x.dtype
  else:
    try:
      dt = np.result_type(x)
    except TypeError as err:
      raise TypeError(f"Cannot determine dtype of {x}") from err
  if dt not in _jax_dtype_set and not issubdtype(dt, extended):
    raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")
  # TODO(jakevdp): fix return type annotation and remove this ignore.
  return canonicalize_dtype(dt, allow_extended_dtype=True) if canonicalize else dt  # type: ignore[return-value]

def _lattice_result_type(*args: Any) -> tuple[DType, bool]:
  dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  if len(dtypes) == 1:
    out_dtype = dtypes[0]
    out_weak_type = weak_types[0]
  elif len(set(dtypes)) == 1 and not all(weak_types):
    # Trivial promotion case. This allows extended dtypes through.
    out_dtype = dtypes[0]
    out_weak_type = False
  elif all(weak_types) and config.numpy_dtype_promotion.value != 'strict':
    # If all inputs are weakly typed, we compute the bound of the strongly-typed
    # counterparts and apply the weak type at the end. This avoids returning the
    # incorrect result with non-canonical weak types (e.g. weak int16).
    # TODO(jakevdp): explore removing this special case.
    result_type = _least_upper_bound(config.numpy_dtype_promotion.value,
                                     *{_jax_type(dtype, False) for dtype in dtypes})
    out_dtype = dtype(result_type)
    out_weak_type = True
  else:
    result_type = _least_upper_bound(config.numpy_dtype_promotion.value,
                                     *{_jax_type(d, w) for d, w in zip(dtypes, weak_types)})
    out_dtype = dtype(result_type)
    out_weak_type = any(result_type is t for t in _weak_types)
  return out_dtype, (out_dtype != bool_) and out_weak_type

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[True]) -> tuple[DType, bool]: ...

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[False] = False) -> DType: ...

@overload
def result_type(*args: Any, return_weak_type_flag: bool = False) -> DType | tuple[DType, bool]: ...

@export
def result_type(*args: Any, return_weak_type_flag: bool = False) -> DType | tuple[DType, bool]:
  """Convenience function to apply JAX argument dtype promotion.

  Args:
    return_weak_type_flag : if True, then return a ``(dtype, weak_type)`` tuple.
      If False, just return `dtype`

  Returns:
    dtype or (dtype, weak_type) depending on the value of the ``return_weak_type`` argument.
  """
  if len(args) == 0:
    raise ValueError("at least one array or dtype is required")
  dtype: DType | ExtendedDType
  dtype, weak_type = _lattice_result_type(*(float_ if arg is None else arg for arg in args))
  if weak_type:
    dtype = canonicalize_dtype(
      _default_types['f' if dtype in _custom_float_dtypes else dtype.kind])
  else:
    dtype = canonicalize_dtype(dtype, allow_extended_dtype=True)
  # TODO(jakevdp): fix return type annotation and remove this ignore.
  return (dtype, weak_type) if return_weak_type_flag else dtype  # type: ignore[return-value]

def check_user_dtype_supported(dtype, fun_name=None):
  if isinstance(dtype, Array):
    # Deprecation warning added 2024 June 13.
    warnings.warn("Passing an array as a dtype argument is deprecated; "
                  "instead of dtype=arr use dtype=arr.dtype.",
                  category=DeprecationWarning, stacklevel=3)
    return  # no further check needed, as array dtypes have already been validated.
  if issubdtype(dtype, extended):
    return
  # Avoid using `dtype in [...]` because of numpy dtype equality overloading.
  if isinstance(dtype, type) and dtype in {bool, int, float, builtins.complex}:
    return
  np_dtype = np.dtype(dtype)
  is_custom_dtype = np_dtype.type in [
      *_custom_float_scalar_types,
      int2,
      int4,
      uint2,
      uint4
  ]
  if np_dtype.kind not in "biufc" and not is_custom_dtype and not dtype == float0:
    msg = f"JAX only supports number and bool dtypes, got dtype {dtype}"
    msg += f" in {fun_name}" if fun_name else ""
    raise TypeError(msg)
  if dtype is not None and np_dtype != canonicalize_dtype(np_dtype):
    msg = ("Explicitly requested dtype {} {} is not available, "
           "and will be truncated to dtype {}. To enable more dtypes, set the "
           "jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell "
           "environment variable. "
           "See https://github.com/jax-ml/jax#current-gotchas for more.")
    fun_name = f"requested in {fun_name}" if fun_name else ""
    truncated_dtype = canonicalize_dtype(np_dtype).name
    warnings.warn(msg.format(dtype, fun_name, truncated_dtype), stacklevel=3)

def safe_to_cast(input_dtype_or_value: Any,
                 output_dtype_or_value: Any) -> bool:
  """Check if a dtype/value is safe to cast to another dtype/value

  Args:
    input_dtype_or_value: a dtype or value (to be passed to result_type)
      representing the source dtype.
    output_dtype_or_value: a dtype or value (to be passed to result_type)
      representing the target dtype.

  Returns:
    boolean representing whether the values are safe to cast according to
    default type promotion semantics.

  Raises:
    TypePromotionError: if the inputs have differing types and no type promotion
    path under the current jax_numpy_dtype_promotion setting.

  Examples:

    >>> safe_to_cast('int32', 'float64')
    True
    >>> safe_to_cast('float64', 'int32')
    False
    >>> safe_to_cast('float32', 'complex64')
    True
    >>> safe_to_cast('complex64', 'float64')
    False
  """
  input_dtype = dtype(input_dtype_or_value, canonicalize=True)
  output_dtype = dtype(output_dtype_or_value, canonicalize=True)
  if input_dtype == output_dtype:
    return True
  # We deliberately use output_dtype rather than output_dtype_or_value here:
  # this effectively treats the output dtype as always strongly-typed.
  return result_type(input_dtype_or_value, output_dtype) == output_dtype

def primal_tangent_dtype(primal_dtype, tangent_dtype,
                         name: str | None = None) -> ExtendedDType:
  primal_dtype, tangent_dtype = map(dtype, (primal_dtype, tangent_dtype))
  name_ = name or (f'PrimalTangentDType{{{short_dtype_name(primal_dtype)}'
                   f'/{short_dtype_name(tangent_dtype)}}}')
  rules = types.SimpleNamespace(
      physical_element_aval=
      lambda dtype: types.SimpleNamespace(shape=(), dtype=primal_dtype),
      tangent_dtype=lambda dtype: tangent_dtype,
      allow_conversion=True)

  class primal_tangent_dtype_scalar(extended): ...

  @dataclasses.dataclass(frozen=True)
  class PrimalTangentDType(ExtendedDType):
    name = name_
    _rules = rules
    type = primal_tangent_dtype_scalar
    __repr__ = lambda _: name_

  return PrimalTangentDType()

def short_dtype_name(dtype) -> str:
  if isinstance(dtype, ExtendedDType):
    return str(dtype)
  else:
    return (dtype.name.replace('float', 'f').replace('uint'   , 'u')
                      .replace('int'  , 'i').replace('complex', 'c'))
