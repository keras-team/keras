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

# This submodule includes lax-specific private test utilities that are not
# exported to jax.test_util. Functionality appearing here is for internal use
# only, and may be changed or removed at any time and without any deprecation
# cycle.

from __future__ import annotations

import collections
import itertools
from typing import Union, cast

import jax
from jax import lax
from jax._src import dtypes
from jax._src import test_util
from jax._src.util import safe_map, safe_zip

import numpy as np

jax.config.parse_flags_with_absl()

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


# For standard unops and binops, we can generate a large number of tests on
# arguments of appropriate shapes and dtypes using the following table.

float_dtypes = test_util.dtypes.all_floating
complex_elem_dtypes = test_util.dtypes.floating
complex_dtypes = test_util.dtypes.complex
inexact_dtypes = test_util.dtypes.all_inexact
int_dtypes = test_util.dtypes.all_integer
uint_dtypes = test_util.dtypes.all_unsigned
bool_dtypes = test_util.dtypes.boolean

default_dtypes = float_dtypes + int_dtypes
all_dtypes = (
    float_dtypes + complex_dtypes + int_dtypes + uint_dtypes + bool_dtypes
)
python_scalar_types = [bool, int, float, complex]

compatible_shapes = [[(3,)], [(3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]

OpRecord = collections.namedtuple(
    "OpRecord", ["op", "nargs", "dtypes", "rng_factory", "tol"]
)


def op_record(op, nargs, dtypes, rng_factory, tol=None):
  return OpRecord(op, nargs, dtypes, rng_factory, tol)


ReducerOpRecord = collections.namedtuple(
    "ReducerOpRecord", ["op", "reference_op", "init_val", "dtypes", "primitive"]
)


def lax_reduce_ops():
  return [
      ReducerOpRecord(lax.add, np.add, 0, default_dtypes, lax.reduce_sum_p),
      ReducerOpRecord(
          lax.mul, np.multiply, 1, default_dtypes, lax.reduce_prod_p
      ),
      ReducerOpRecord(
          lax.max, np.maximum, 0, uint_dtypes + bool_dtypes, lax.reduce_max_p
      ),
      ReducerOpRecord(
          lax.max, np.maximum, -np.inf, float_dtypes, lax.reduce_max_p
      ),
      ReducerOpRecord(
          lax.max,
          np.maximum,
          dtypes.iinfo(np.int32).min,
          [np.int32],
          lax.reduce_max_p,
      ),
      ReducerOpRecord(
          lax.max,
          np.maximum,
          dtypes.iinfo(np.int64).min,
          [np.int64],
          lax.reduce_max_p,
      ),
      ReducerOpRecord(
          lax.min, np.minimum, np.inf, float_dtypes, lax.reduce_min_p
      ),
      ReducerOpRecord(
          lax.min,
          np.minimum,
          dtypes.iinfo(np.int32).max,
          [np.int32],
          lax.reduce_min_p,
      ),
      ReducerOpRecord(
          lax.min,
          np.minimum,
          dtypes.iinfo(np.int64).max,
          [np.int64],
          lax.reduce_min_p,
      ),
      ReducerOpRecord(
          lax.min,
          np.minimum,
          dtypes.iinfo(np.uint32).max,
          [np.uint32],
          lax.reduce_min_p,
      ),
      ReducerOpRecord(
          lax.min,
          np.minimum,
          dtypes.iinfo(np.uint64).max,
          [np.uint64],
          lax.reduce_min_p,
      ),
      ReducerOpRecord(
          lax.bitwise_and,
          np.bitwise_and,
          -1,
          int_dtypes + uint_dtypes + bool_dtypes,
          lax.reduce_and_p,
      ),
      ReducerOpRecord(
          lax.bitwise_or,
          np.bitwise_or,
          0,
          int_dtypes + uint_dtypes + bool_dtypes,
          lax.reduce_or_p,
      ),
      ReducerOpRecord(
          lax.bitwise_xor,
          np.bitwise_xor,
          0,
          int_dtypes + uint_dtypes + bool_dtypes,
          lax.reduce_xor_p,
      ),
  ]


def lax_ops():
  return [
      op_record(
          "neg", 1, default_dtypes + complex_dtypes, test_util.rand_small
      ),
      op_record("sign", 1, default_dtypes + uint_dtypes, test_util.rand_small),
      op_record("floor", 1, float_dtypes, test_util.rand_small),
      op_record("ceil", 1, float_dtypes, test_util.rand_small),
      op_record("round", 1, float_dtypes, test_util.rand_default),
      op_record(
          "nextafter",
          2,
          [f for f in float_dtypes if f != dtypes.bfloat16],
          test_util.rand_default,
          tol=0,
      ),
      op_record("is_finite", 1, float_dtypes, test_util.rand_small),
      op_record("exp", 1, float_dtypes + complex_dtypes, test_util.rand_small),
      op_record("exp2", 1, float_dtypes + complex_dtypes, test_util.rand_small),
      # TODO(b/142975473): on CPU, expm1 for float64 is only accurate to ~float32
      # precision.
      op_record(
          "expm1",
          1,
          float_dtypes + complex_dtypes,
          test_util.rand_small,
          {np.float64: 1e-8},
      ),
      op_record(
          "log", 1, float_dtypes + complex_dtypes, test_util.rand_positive
      ),
      op_record(
          "log1p", 1, float_dtypes + complex_dtypes, test_util.rand_positive
      ),
      # TODO(b/142975473): on CPU, tanh for complex128 is only accurate to
      # ~float32 precision.
      # TODO(b/143135720): on GPU, tanh has only ~float32 precision.
      op_record(
          "tanh",
          1,
          float_dtypes + complex_dtypes,
          test_util.rand_small,
          {np.float64: 1e-9, np.complex128: 1e-7},
      ),
      op_record(
          "logistic", 1, float_dtypes + complex_dtypes, test_util.rand_default
      ),
      op_record(
          "sin", 1, float_dtypes + complex_dtypes, test_util.rand_default
      ),
      op_record(
          "cos", 1, float_dtypes + complex_dtypes, test_util.rand_default
      ),
      op_record("atan2", 2, float_dtypes, test_util.rand_default),
      op_record("sqrt", 1, float_dtypes, test_util.rand_positive),
      op_record("sqrt", 1, complex_dtypes, test_util.rand_default),
      op_record("rsqrt", 1, float_dtypes, test_util.rand_positive),
      op_record("rsqrt", 1, complex_dtypes, test_util.rand_default),
      op_record("cbrt", 1, float_dtypes, test_util.rand_default),
      op_record(
          "square", 1, float_dtypes + complex_dtypes, test_util.rand_default
      ),
      op_record(
          "reciprocal",
          1,
          float_dtypes + complex_dtypes,
          test_util.rand_positive,
      ),
      op_record(
          "tan",
          1,
          float_dtypes + complex_dtypes,
          test_util.rand_default,
          {np.float32: 3e-5},
      ),
      op_record(
          "asin",
          1,
          float_dtypes + complex_dtypes,
          test_util.rand_small,
          {np.complex128: 5e-12},
      ),
      op_record("acos", 1, float_dtypes + complex_dtypes, test_util.rand_small),
      op_record("atan", 1, float_dtypes + complex_dtypes, test_util.rand_small),
      op_record(
          "asinh",
          1,
          float_dtypes + complex_dtypes,
          test_util.rand_default,
          tol={np.complex64: 1e-4, np.complex128: 1e-5},
      ),
      op_record(
          "acosh", 1, float_dtypes + complex_dtypes, test_util.rand_positive
      ),
      # TODO(b/155331781): atanh has only ~float precision
      op_record(
          "atanh",
          1,
          float_dtypes + complex_dtypes,
          test_util.rand_small,
          {np.float64: 1e-9},
      ),
      op_record(
          "sinh", 1, float_dtypes + complex_dtypes, test_util.rand_default
      ),
      op_record(
          "cosh", 1, float_dtypes + complex_dtypes, test_util.rand_default
      ),
      op_record(
          "lgamma",
          1,
          float_dtypes,
          test_util.rand_positive,
          {
              np.float32: 1e-5,
              np.float64: 1e-14,
          },
      ),
      op_record(
          "digamma",
          1,
          float_dtypes,
          test_util.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record(
          "betainc",
          3,
          float_dtypes,
          test_util.rand_uniform,
          {
              np.float32: 1e-5,
              np.float64: 1e-12,
          },
      ),
      op_record(
          "igamma",
          2,
          [f for f in float_dtypes if f not in [dtypes.bfloat16, np.float16]],
          test_util.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record(
          "igammac",
          2,
          [f for f in float_dtypes if f not in [dtypes.bfloat16, np.float16]],
          test_util.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record("erf", 1, float_dtypes, test_util.rand_small),
      op_record("erfc", 1, float_dtypes, test_util.rand_small),
      # TODO(b/142976030): the approximation of erfinf used by XLA is only
      # accurate to float32 precision.
      op_record(
          "erf_inv", 1, float_dtypes, test_util.rand_small, {np.float64: 1e-9}
      ),
      op_record("bessel_i0e", 1, float_dtypes, test_util.rand_default),
      op_record("bessel_i1e", 1, float_dtypes, test_util.rand_default),
      op_record("real", 1, complex_dtypes, test_util.rand_default),
      op_record("imag", 1, complex_dtypes, test_util.rand_default),
      op_record("complex", 2, complex_elem_dtypes, test_util.rand_default),
      op_record(
          "conj",
          1,
          complex_elem_dtypes + complex_dtypes,
          test_util.rand_default,
      ),
      op_record(
          "abs", 1, default_dtypes + complex_dtypes, test_util.rand_default
      ),
      op_record(
          "pow", 2, float_dtypes + complex_dtypes, test_util.rand_positive
      ),
      op_record("bitwise_and", 2, bool_dtypes, test_util.rand_small),
      op_record("bitwise_not", 1, bool_dtypes, test_util.rand_small),
      op_record("bitwise_or", 2, bool_dtypes, test_util.rand_small),
      op_record("bitwise_xor", 2, bool_dtypes, test_util.rand_small),
      op_record(
          "population_count", 1, int_dtypes + uint_dtypes, test_util.rand_int
      ),
      op_record("clz", 1, int_dtypes + uint_dtypes, test_util.rand_int),
      op_record(
          "add", 2, default_dtypes + complex_dtypes, test_util.rand_small
      ),
      op_record(
          "sub", 2, default_dtypes + complex_dtypes, test_util.rand_small
      ),
      op_record(
          "mul", 2, default_dtypes + complex_dtypes, test_util.rand_small
      ),
      op_record(
          "div", 2, default_dtypes + complex_dtypes, test_util.rand_nonzero
      ),
      op_record("rem", 2, default_dtypes, test_util.rand_nonzero),
      op_record("max", 2, all_dtypes, test_util.rand_small),
      op_record("min", 2, all_dtypes, test_util.rand_small),
      op_record("eq", 2, all_dtypes, test_util.rand_some_equal),
      op_record("ne", 2, all_dtypes, test_util.rand_small),
      op_record("ge", 2, default_dtypes, test_util.rand_small),
      op_record("gt", 2, default_dtypes, test_util.rand_small),
      op_record("le", 2, default_dtypes, test_util.rand_small),
      op_record("lt", 2, default_dtypes, test_util.rand_small),
  ]


def all_bdims(*shapes):
  bdims = (itertools.chain([cast(Union[int, None], None)],
                           range(len(shape) + 1)) for shape in shapes)
  return (t for t in itertools.product(*bdims) if not all(e is None for e in t))


def add_bdim(bdim_size, bdim, shape):
  shape = list(shape)
  if bdim is not None:
    shape.insert(bdim, bdim_size)
  return tuple(shape)


def slicer(x, bdim):
  if bdim is None:
    return lambda _: x
  else:
    return lambda i: lax.index_in_dim(x, i, bdim, keepdims=False)


def args_slicer(args, bdims):
  slicers = map(slicer, args, bdims)
  return lambda i: [sl(i) for sl in slicers]
