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
"""Defines test callables and inputs to cover the usage of JAX primitives.

A test harness encodes one use case for one JAX numeric primitives. It
describes how to generate the inputs and parameters and a JAX callable to invoke
on those inputs. A test harness can be used in multiple kinds of tests.
For example, we can use the harnesses to check
that each primitive is compiled correctly, or that we can apply a certain
transformation, e.g., `vmap`.

See the `Harness` class below for how to define a harness, describing one
use case of one primitive.

Some use cases are known to be partially implemented
in JAX, e.g., because of an implementation limitation. We do have harnesses
for those cases too, but there is a mechanism to filter them out.
Instead of writing this information as conditions inside one
particular test, we write them as `Limitation` objects that can be reused in
multiple tests and can also be used to generate documentation, e.g.,
the report of [unsupported and partially-implemented JAX
primitives](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/g3doc/jax_primitives_coverage.md)

The limitations are used to filter out from tests the harnesses that are known
to fail. A Limitation is specific to a harness.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import operator
import os
from functools import partial
from typing import Any, NamedTuple, Union

from absl import testing
import numpy as np

import jax
from jax import dtypes
from jax import lax
from jax import numpy as jnp

from jax._src import ad_util
from jax._src import config
from jax._src import dispatch
from jax._src import prng
from jax._src import test_util as jtu
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import windowed_reductions as lax_windowed_reductions
from jax._src import random as jax_random

# mypy generates a lot of false positive due to re-assigned variables.
# mypy: disable-error-code="assignment, no-redef"

# The code in this file relies on the values of some flags that are defined by
# jtu. Note that the following can not always be moved to a test file since
# then the test file has to import jtu first (to define the flags) which is not
# desired if the test file is outside of this project (we don't want a
# dependency on jtu outside of jax repo).
config.parse_flags_with_absl()

Rng = Any  # A random number generator
DType = Any


class RandArg(NamedTuple):
  """Descriptor for a randomly generated argument.

  See description of `Harness`.
  """
  shape: tuple[int, ...]
  dtype: DType


class StaticArg(NamedTuple):
  """Descriptor for a static argument.

  See description of `Harness`.
  """
  value: Any


class CustomArg(NamedTuple):
  """Descriptor for a dynamic argument generated from a PRNG factory.

  See description of `Harness`.
  """
  make: Callable[[Rng], Any]  # Called with a Rng to make a tensor


ArgDescriptor = Union[RandArg, StaticArg, CustomArg, Any]


class Harness:
  """Specifies inputs and callable for a test harness.

  See the module docstring for an introduction to harnesses.

  A harness is conceptually a callable and a list of arguments, that together
  exercise a use case. The harness can optionally have additional parameters
  that can be used by the test.

  The arguments are specified through argument descriptors. An argument
  descriptor can be:
    * a numeric value or ndarray, or
    * an instance of ``RandArg(shape, dtype)`` to be used with a PRNG to
    generate random tensor of the given shape and type, or
    * an instance of ``CustomArg(fun)`` to be used with a PRNG, or
    * an instance of ``StaticArg(value)``. Often these are the non-array
      arguments, e.g., a shape.

  The given callable will be passed one argument corresponding to each
  argument descriptor, e.g., `harness.fun(* harness.args_maker(rng))`.
  However, in many applications we only care about the non-static arguments.
  For that purpose, you can use `harness.dyn_fun(*
  harness.dyn_args_maked(rng))`,
  where `harness.dyn_fun` is `harness.fun` specialized to the static arguments.

  For example, a harness for ``lax.take(arr, indices, axis=None)`` may want
  to expose as external (non-static) argument the array and the indices, and
  keep the axis as a static argument (technically specializing the `take` to
  a axis):

    Harness(lax.slice_p,
            f"take_axis={axis}",
            lax.take,
            [RandArg((2, 4), np.float32), np.array([-1, 0, 1]),
            StaticArg(axis)],
            axis=axis)

  Each harness can have a list of Limitations that describe the cases when
  the harness may not be fully implemented.
  """
  # The group name most often is the primitive name.
  group_name: str
  # Descriptive name of the harness, used as a testcase_name. Unique in a group.
  # Will be sanitized to work with -k test filtering.
  name: str
  # The function taking all arguments (static and dynamic).
  fun: Callable
  # Describes how to construct arguments, see the class docstring.
  arg_descriptors: Sequence[ArgDescriptor]
  dtype: DType
  # A set of limitations describing the cases that are not supported or
  # partially implemented in JAX for this harness.
  jax_unimplemented: Sequence[Limitation]
  rng_factory: Callable
  # Carry some arbitrary parameters that the test can access.
  params: dict[str, Any]

  def __init__(self,
               group_name,
               name,
               fun,
               arg_descriptors,
               *,
               dtype,
               rng_factory=jtu.rand_default,
               jax_unimplemented: Sequence[Limitation] = (),
               **params):
    """See class docstring."""
    self.group_name = jtu.sanitize_test_name(group_name)
    self.name = jtu.sanitize_test_name(name)
    self.fullname = self.name if self.group_name is None else f"{self.group_name}_{self.name}"
    self.fun = fun
    self.arg_descriptors = arg_descriptors
    self.rng_factory = rng_factory
    self.jax_unimplemented = jax_unimplemented
    self.dtype = dtype
    self.params = params

  def __str__(self):
    return self.fullname


  def _arg_maker(self, arg_descriptor, rng: Rng):
    if isinstance(arg_descriptor, StaticArg):
      return arg_descriptor.value
    if isinstance(arg_descriptor, RandArg):
      return self.rng_factory(rng)(arg_descriptor.shape, arg_descriptor.dtype)
    if isinstance(arg_descriptor, CustomArg):
      return arg_descriptor.make(rng)

    return arg_descriptor

  def args_maker(self, rng: Rng) -> Sequence:
    """All-argument maker, including the static ones."""
    return [self._arg_maker(ad, rng) for ad in self.arg_descriptors]

  def dyn_args_maker(self, rng: Rng) -> Sequence:
    """A dynamic-argument maker, for use with `dyn_fun`."""
    return [
        self._arg_maker(ad, rng)
        for ad in self.arg_descriptors
        if not isinstance(ad, StaticArg)
    ]

  def dyn_fun(self, *dyn_args):
    """Invokes `fun` given just the dynamic arguments."""
    all_args = self._args_from_dynargs(dyn_args)
    return self.fun(*all_args)

  def _args_from_dynargs(self, dyn_args: Sequence) -> Sequence:
    """All arguments, including the static ones."""
    next_dynamic_argnum = 0
    all_args = []
    for ad in self.arg_descriptors:
      if isinstance(ad, StaticArg):
        all_args.append(ad.value)
      else:
        all_args.append(dyn_args[next_dynamic_argnum])
        next_dynamic_argnum += 1
    return all_args

  def filter(self,
             device_under_test: str,
             *,
             include_jax_unimpl: bool = False,
             one_containing: str | None = None) -> bool:
    if not include_jax_unimpl:
      if any(
          device_under_test in l.devices
          for l in self.jax_unimplemented
          if l.filter(device=device_under_test, dtype=self.dtype)
      ):
        return False

    if one_containing is not None and one_containing not in self.fullname:
      return False
    return True


def dtypes_to_str(dtype_list: Sequence[DType], empty_means_all=False) -> str:
  """User-friendly description of a set of dtypes"""
  if not dtype_list and empty_means_all:
    return "all"

  names = {np.dtype(dt).name for dt in dtype_list}
  signed = {"int8", "int16", "int32", "int64"}
  if signed <= names:
    names = (names - signed) | {"signed"}
  integers = {"uint8", "uint16", "uint32", "uint64"}
  if integers <= names:
    names = (names - integers) | {"unsigned"}
  integer = {"signed", "unsigned"}
  if integer <= names:
    names = (names - integer) | {"integer"}

  floating = {"bfloat16", "float16", "float32", "float64"}
  if floating <= names:
    names = (names - floating) | {"floating"}

  complex = {"complex64", "complex128"}
  if complex <= names:
    names = (names - complex) | {"complex"}

  inexact = {"floating", "complex"}
  if inexact <= names:
    names = (names - inexact) | {"inexact"}

  all_types = {"integer", "inexact", "bool"}
  if all_types <= names:
    names = (names - all_types) | {"all"}

  return ", ".join(sorted(names))


##### All harnesses in this file.
all_harnesses: list[Harness] = []


def define(
    group_name,
    name,
    fun,
    arg_descriptors,
    *,
    dtype,
    rng_factory=jtu.rand_default,
    jax_unimplemented: Sequence[Limitation] = (),
    **params):
  """Defines a harness and stores it in `all_harnesses`. See Harness."""
  group_name = str(group_name)
  h = Harness(
      group_name,
      name,
      fun,
      arg_descriptors,
      rng_factory=rng_factory,
      jax_unimplemented=jax_unimplemented,
      dtype=dtype,
      **params)
  all_harnesses.append(h)


class Limitation:
  """Encodes conditions under which a harness is limited, e.g., not runnable in JAX.

  See the module docstring for an introduction to harnesses and limitations.
  """

  def __init__(
      self,
      description: str,
      *,
      enabled: bool = True,
      devices: str | Sequence[str] = ("cpu", "gpu", "tpu"),
      dtypes: Sequence[DType] = (),
      skip_run: bool = False,
  ):
    """Args:

      description: text to augment the harness group name with the description
      of the limitation. Used for reports.
      enabled: whether this limitation is enabled for the harness in which
        it appears. This is only used during testing to know whether to ignore
        harness errors. Use this sparingly, prefer `devices` and
        `dtypes` for enabled conditions that are included in reports.
      devices: a device type (string) or a sequence of device types
        for which this applies. By default, it applies to all devices types.
        Used for filtering during harness execution, and for reports.
      dtypes: the sequence of dtypes for which this applies. An empty sequence
        denotes all dtypes. Used for filtering during harness execution, and
        for reports.
      skip_run: this harness should not even be invoked (typically because it
        results in a crash). This should be rare.
    """
    assert isinstance(description, str), f"{description}"
    self.description = description
    self.skip_run = skip_run
    if isinstance(devices, str):
      devices = (devices,)
    else:
      devices = tuple(devices)
    self.devices = devices
    assert isinstance(dtypes, Iterable)
    dtypes = tuple(dtypes)
    self.dtypes = dtypes
    self.enabled = enabled  # Does it apply to the current harness?

  def __str__(self):
    return (f"\"{self.description}\" devices={self.devices} "
            f"dtypes={[np.dtype(dt).name for dt in self.dtypes]}" +
            (" (skip_run) " if self.skip_run else ""))

  __repr__ = __str__

  def filter(self,
             device: str | None = None,
             dtype: DType | None = None) -> bool:
    """Check that a limitation is enabled for the given dtype and device."""
    return (self.enabled and
            (not self.dtypes or dtype is None or dtype in self.dtypes) and
            (device is None or device in self.devices))


def parameterized(harnesses: Iterable[Harness],
                  *,
                  one_containing: str | None = None,
                  include_jax_unimpl: bool = False):
  """Decorator for tests.

  The tests receive a `harness` argument.

  The `JAX_TEST_HARNESS_ONE_CONTAINING` environment variable is useful for
  debugging. If given, then picks only one harness whose name contains the
  string. The whole set of parameterized tests is reduced to one test,
  whose name is not decorated to make it easier to pick with an IDE for
  running.
  """
  one_containing = one_containing or os.environ.get(
      "JAX_TEST_HARNESS_ONE_CONTAINING")
  cases = tuple(
      # Change the testcase name to include the harness name.
      dict(
          testcase_name=harness.fullname if one_containing is None else "",
          harness=harness) for harness in harnesses if harness.filter(
              jtu.device_under_test(),
              one_containing=one_containing,
              include_jax_unimpl=include_jax_unimpl))
  if one_containing is not None:
    if not cases:
      raise ValueError(
          f"Cannot find test case with name containing {one_containing}."
          "Names are:"
          "\n".join([harness.fullname for harness in harnesses]))
    cases = cases[0:1]
  if not cases:
    # We filtered out all the harnesses.
    return jtu.skip_on_devices(jtu.device_under_test())
  return testing.parameterized.named_parameters(*cases)


###############################################################################
###                    Harness definitions                                  ###
###############################################################################


def _make_unary_elementwise_harness(*, prim, shape=(20, 20), dtype):
  define(
      str(prim),
      f"shape={jtu.format_shape_dtype_string(shape, dtype)}",
      prim.bind, [RandArg(shape, dtype)],
      prim=prim,
      dtype=dtype,
      shape=shape)


for dtype in (set(jtu.dtypes.all) -
              set(jtu.dtypes.all_unsigned + jtu.dtypes.boolean)):
  _make_unary_elementwise_harness(prim=lax.abs_p, dtype=dtype)

for dtype in jtu.dtypes.all_floating + jtu.dtypes.complex:
  _make_unary_elementwise_harness(prim=lax.acosh_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.asinh_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.atanh_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.acos_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.atan_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.asin_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.cos_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.cosh_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.exp_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.expm1_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.log_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.log1p_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.rsqrt_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.sin_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.sinh_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.sqrt_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.tan_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.tanh_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.logistic_p, dtype=dtype)

for dtype in jtu.dtypes.all_floating:
  _make_unary_elementwise_harness(prim=lax.bessel_i0e_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.bessel_i1e_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.ceil_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.erf_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.erf_inv_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.erfc_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.floor_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.is_finite_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.lgamma_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.digamma_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.cbrt_p, dtype=dtype)

for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.boolean):
  _make_unary_elementwise_harness(prim=lax.neg_p, dtype=dtype)
  _make_unary_elementwise_harness(prim=lax.sign_p, dtype=dtype)
  define(
      str(lax.sign_p),
      f"special_0_dtype={jtu.dtype_str(dtype)}",
      lax.sign_p.bind, [StaticArg(np.zeros((2, 2), dtype=dtype))],
      prim=lax.sign_p,
      dtype=dtype)


def _make_round_harness(name,
                        *,
                        shape=(100, 100),
                        dtype=np.float32,
                        rounding_method=lax.RoundingMethod.AWAY_FROM_ZERO,
                        operand=None):
  operand = operand if operand is not None else RandArg(shape, dtype)
  define(
      "round",
      f"{name}_shape={jtu.format_shape_dtype_string(operand.shape, operand.dtype)}_roundingmethod={rounding_method}",
      lax.round, [operand, StaticArg(rounding_method)],
      dtype=dtype,
      operand=operand,
      rounding_method=rounding_method)


# Validate dtypes
for dtype in jtu.dtypes.all_floating:
  _make_round_harness("dtypes", dtype=dtype)

for rounding_method in [
    lax.RoundingMethod.AWAY_FROM_ZERO, lax.RoundingMethod.TO_NEAREST_EVEN
]:
  operand = np.array([[0.5, 1.2, 1.5, 1.7, 2.5], [-0.5, -1.2, -1.5, -1.7, -2.5]], dtype=np.float32)
  _make_round_harness(
      "rounding_methods", operand=operand, rounding_method=rounding_method)


def _make_convert_element_type_harness(name,
                                       *,
                                       shape=(100, 100),
                                       dtype=np.float32,
                                       new_dtype=np.float32):
  define(
      "convert_element_type",
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_olddtype={jtu.dtype_str(dtype)}_newdtype={jtu.dtype_str(new_dtype)}",
      lambda arg: (lax.convert_element_type_p.bind(
          arg, new_dtype=np.dtype(new_dtype), weak_type=False, sharding=None)),
      [RandArg(shape, dtype)],
      shape=shape,
      dtype=dtype,
      new_dtype=new_dtype)


for old_dtype in jtu.dtypes.all:
  # TODO(bchetioui): JAX behaves weirdly when old_dtype corresponds to floating
  # point numbers and new_dtype is an unsigned integer. See issue
  # https://github.com/jax-ml/jax/issues/5082 for details.
  for new_dtype in (jtu.dtypes.all
                    if not (dtypes.issubdtype(old_dtype, np.floating) or
                            dtypes.issubdtype(old_dtype, np.complexfloating))
                    else set(jtu.dtypes.all) - set(jtu.dtypes.all_unsigned)):
    _make_convert_element_type_harness(
        "dtypes_to_dtypes", dtype=old_dtype, new_dtype=new_dtype)


def _make_integer_pow_harness(name, *, shape=(20, 30), dtype=np.int32, y=3):
  define(
      "integer_pow",
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{y=}",
      lax.integer_pow,
      [RandArg(shape, dtype), StaticArg(y)],
      shape=shape,
      dtype=dtype,
      y=y)


for dtype in [d for d in jtu.dtypes.all if d not in jtu.dtypes.boolean]:
  # Validate dtypes and y values for some special cases.
  for y in range(-3, 5):
    if np.issubdtype(dtype, np.integer) and y < 0:
      continue  # No negative powers for integers
    _make_integer_pow_harness("dtypes", dtype=dtype, y=y)
  # Validate overflow behavior by dtype. y=127
  # TODO(necula): this results in RuntimeWarning np.abs overflow.
  # In export_harnesses_multi_platform
  # _make_integer_pow_harness("overflow", y=127, dtype=dtype)

for dtype in jtu.dtypes.all_inexact:
  # Validate negative y by dtype
  _make_integer_pow_harness("negative_overflow", y=-3, dtype=dtype)


def _make_pow_harness(name,
                      *,
                      shapes=((20, 30), (20, 30)),
                      dtype=np.float32,
                      lhs=None,
                      rhs=None):
  lhs = RandArg(shapes[0], dtype) if lhs is None else lhs
  rhs = RandArg(shapes[1], dtype) if rhs is None else rhs
  define(
      "pow",
      f"{name}_lhs={jtu.format_shape_dtype_string(lhs.shape, dtype)}_rhs={jtu.format_shape_dtype_string(rhs.shape, dtype)}",
      lax.pow, [lhs, rhs],
      lhs=lhs,
      rhs=rhs,
      dtype=dtype)


for dtype in jtu.dtypes.all_inexact:
  # Validate dtypes
  _make_pow_harness("dtypes", dtype=dtype)

# Validate broadcasting behavior
for shapes in [
    ((), (4, 5, 6)),  # broadcasting lhs
    ((4, 5, 6), ()),  # broadcasting rhs
    ((4, 1, 6), (4, 5, 6)),  # broadcasting lhs on a specific axis
    ((4, 5, 6), (4, 1, 6)),  # broadcasting rhs on a specific axis
]:
  _make_pow_harness("broadcast", shapes=shapes)


def _make_reshape_harness(name,
                          *,
                          shape=(2, 3),
                          new_sizes=(3, 2),
                          dimensions=(0, 1),
                          dtype=np.float32):
  define(
      "reshape",
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_newsizes={new_sizes}_{dimensions=}",
      lax.reshape,
      [RandArg(shape, dtype),
       StaticArg(new_sizes),
       StaticArg(dimensions)],
      shape=shape,
      dtype=dtype,
      new_sizes=new_sizes,
      dimensions=dimensions)


# Validate dtypes
for dtype in jtu.dtypes.all:
  _make_reshape_harness("dtypes", dtype=dtype)

# Validate new_sizes
for shape, new_sizes, dimensions in [
    ((3, 4, 5), (3, 20), (0, 1, 2)),  # merging two dimensions
    ((3, 4, 5), (4, 15), (0, 1, 2)),  # changing leading dimension
]:
  _make_reshape_harness(
      "new_sizes", shape=shape, new_sizes=new_sizes, dimensions=dimensions)
# Validate dimensions collapsing order
for shape, new_sizes, dimensions in [
    ((3, 4, 5), (3, 20), (2, 1, 0)),  # transpose shape (0, 1, 2) into (2, 1, 0)
    ((3, 4, 5), (3, 20), (2, 0, 1)),  # transpose shape (0, 1, 2) into (2, 0, 1)
]:
  _make_reshape_harness(
      "dimensions", shape=shape, new_sizes=new_sizes, dimensions=dimensions)


def _make_rev_harness(name, *, shape=(4, 5), dtype=np.float32, dimensions=(0,)):
  define(
      "rev",
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{dimensions=}",
      lax.rev,
      [RandArg(shape, dtype), StaticArg(dimensions)],
      shape=shape,
      dtype=dtype,
      dimensions=dimensions)


# Validate dtypes
for dtype in jtu.dtypes.all:
  _make_rev_harness("dtypes", dtype=dtype)
# Validate dimensions
for shape, dimensions in [
    ((3, 4, 5), ()),  # empty dimensions
    ((3, 4, 5), (0, 2)),  # some dimensions
    ((3, 4, 5), (0, 1, 2)),  # all dimensions (ordered)
    ((3, 4, 5), (2, 0, 1)),  # all dimensions (unordered)
]:
  _make_rev_harness("dimensions", shape=shape, dimensions=dimensions)


def _make_device_put_harness(name,
                             *,
                             shape=(3, 4),
                             dtype=np.float32,
                             device=None):
  _device_fn = lambda: jax.devices(device)[0] if device is not None else None
  define(
      "device_put",
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{device=}",
      lambda x: dispatch.device_put_p.bind(
          x, devices=[_device_fn()], srcs=[None],
          copy_semantics=[dispatch.CopySemantics.ALIAS])[0],
      [RandArg(shape, dtype)],
      shape=shape,
      dtype=dtype,
      device=device)


# Validate dtypes
for dtype in jtu.dtypes.all:
  _make_device_put_harness("dtypes", dtype=dtype)
# Validate devices
_make_device_put_harness("devices", device="cpu")


def _make_bitcast_convert_type_harness(name,
                                       *,
                                       shape=(2, 3),
                                       dtype=np.float32,
                                       new_dtype=np.float32):
  define(
      "bitcast_convert_type",
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_newdtype={np.dtype(new_dtype).name}",
      lambda x: lax.bitcast_convert_type_p.bind(x,
                                                new_dtype=np.dtype(new_dtype)),
      [RandArg(shape, dtype)],
      shape=shape,
      dtype=dtype,
      new_dtype=new_dtype)


def _can_bitcast(dtype, target_dtype):

  def _to_equivalence_class(dtype):
    if dtypes.issubdtype(dtype, np.integer):
      return dtypes.iinfo(dtype).bits
    elif dtypes.issubdtype(dtype, np.floating):
      return dtypes.finfo(dtype).bits
    else:
      assert dtype == np.bool_ or dtypes.issubdtype(dtype, np.complexfloating)
      # Complex and boolean types can only be cast to themselves
      return np.dtype(dtype).name

  return _to_equivalence_class(dtype) == _to_equivalence_class(target_dtype)


# Validate dtypes combinations
for dtype in jtu.dtypes.all:
  for new_dtype in filter(partial(_can_bitcast, dtype), jtu.dtypes.all):
    _make_bitcast_convert_type_harness(
        "dtypes_to_new_dtypes", dtype=dtype, new_dtype=new_dtype)

def _make_add_any_harness(name, *, shapes=((2,), (2,)), dtype=np.float32):
  define(
      ad_util.add_any_p,
      f"{name}_lhs={jtu.format_shape_dtype_string(shapes[0], dtype)}_rhs={jtu.format_shape_dtype_string(shapes[1], dtype)}",
      ad_util.add_jaxvals_p.bind,
      list(map(lambda s: RandArg(s, dtype), shapes)),
      dtype=dtype,
      shapes=shapes)

for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.boolean):
  _make_add_any_harness("dtypes", dtype=dtype)


for dtype in jtu.dtypes.all:
  shape: tuple[int, ...] = (20, 20)
  define(
      ad_util.stop_gradient_p,
      f"{jtu.format_shape_dtype_string(shape, dtype)}",
      ad_util.stop_gradient_p.bind, [RandArg(shape, dtype)],
      shape=shape,
      dtype=dtype)

_LAX_COMPARATORS = dict(eq=jnp.equal, ne=jnp.not_equal,
                        ge=jnp.greater_equal, gt=jnp.greater,
                        le=jnp.less_equal, lt=jnp.less)


def _make_comparator_harness(name,
                             *,
                             dtype=np.float32,
                             op=lax.eq_p,
                             op_name="eq",
                             lhs_shape=(),
                             rhs_shape=()):
  define(
      op_name,
      f"{name}_lhs={jtu.format_shape_dtype_string(lhs_shape, dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, dtype)}",
      lambda *args: op(*args),
      [RandArg(lhs_shape, dtype),
       RandArg(rhs_shape, dtype)],
      op=op,
      op_name=op_name,
      lhs_shape=lhs_shape,
      rhs_shape=rhs_shape,
      dtype=dtype)


for op_name, op in _LAX_COMPARATORS.items():
  for dtype in (jtu.dtypes.all if op in [lax.eq_p, lax.ne_p] else
                set(jtu.dtypes.all)):
    # Validate dtypes
    _make_comparator_harness("dtypes", dtype=dtype, op=op, op_name=op_name)

  # Validate broadcasting behavior
  for lhs_shape, rhs_shape in [
      ((), (2, 3)),  # broadcast scalar
      ((1, 2), (3, 2)),  # broadcast along specific axis
  ]:
    _make_comparator_harness(
        "broadcasting", lhs_shape=lhs_shape, rhs_shape=rhs_shape,
        op=op, op_name=op_name)

for dtype in jtu.dtypes.all_integer + jtu.dtypes.all_unsigned:
  if np.issubdtype(dtype, np.unsignedinteger):
    arg = np.array([0, 1, 2], dtype=dtype)
  else:
    arg = np.array([-1, -2, 0, 1], dtype=dtype)
  define(
      "population_count",
      f"{jtu.dtype_str(dtype)}",
      lax.population_count, [arg],
      dtype=dtype)


def _get_max_identity(dtype):
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(-np.inf, dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).min, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(False, np.bool_)


def _get_min_identity(dtype):
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(np.inf, dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).max, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(True, np.bool_)


def _make_argminmax_harness(prim,
                            name,
                            *,
                            shape=(15,),
                            dtype=jnp.float32,
                            axes=(0,),
                            index_dtype=np.int32,
                            arr=None,
                            works_without_xla=True):
  arr = arr if arr is not None else RandArg(shape, dtype)
  dtype, shape = arr.dtype, arr.shape
  index_dtype = dtypes.canonicalize_dtype(index_dtype)
  for enable_xla in [True, False]:
    define(
        prim,
        f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{axes=}_indexdtype={index_dtype}_enable_xla={enable_xla}",
        lambda arg: prim.bind(arg, axes=axes, index_dtype=index_dtype), [arr],
        shape=shape,
        dtype=dtype,
        axes=axes,
        index_dtype=index_dtype,
        prim=prim,
        enable_xla=enable_xla)


for prim in [lax.argmin_p, lax.argmax_p]:
  for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.complex):
    # Validate dtypes for each primitive
    _make_argminmax_harness(prim, "dtypes", dtype=dtype)

  # Validate axes for each primitive; non major axis
  _make_argminmax_harness(prim, "axes", shape=(18, 12), axes=(1,))

  # Validate index dtype for each primitive
  for index_dtype in jtu.dtypes.all_integer + jtu.dtypes.all_unsigned:
    _make_argminmax_harness(prim, "index_dtype", index_dtype=index_dtype)

  # Some special cases, with equal elements and NaN
  for name, operand in [
      ("nan_0", np.array([np.nan, np.nan, 2., -2., -np.nan, -np.nan], np.float32)),
      ("nan_1", np.array([np.nan, -np.nan, 2., -2.], np.float32)),
      ("inf_0", np.array([2., np.inf, np.inf, -2.], np.float32)),
      ("inf_1", np.array([2., np.inf, -np.inf, -2.], np.float32)),
      ("inf_2", np.array([2., -np.inf, np.inf, -2.], np.float32)),
      ("inf_3", np.array([2., -np.inf, -np.inf, -2.], np.float32)),
      ("nan_inf_0", np.array([2., np.nan, np.inf, -2.], np.float32)),
      ("nan_inf_1", np.array([2., np.nan, -np.inf, -2.], np.float32)),
      ("equal", np.array([2., 2., 2.], np.int32)),
      ("singleton", np.array([1.], np.float32)),
      ]:
    _make_argminmax_harness(prim, f"special_{name}", shape=operand.shape,
                            arr=operand)

# TODO(bchetioui): the below documents a limitation of argmin and argmax when a
# dimension of the input is too large. However, it is not categorizable as it
# seems that the converter fails before reaching the actual primitive call. This
# suggests that we may need to harden the converter to handle inputs this big.
# + tuple( # Document limitation in case of too large axis
#  _make_argminmax_harness("overflow_axis", prim=prim,
#                          arr=np.ones((2**31,), dtype=np.uint8))
#  for prim in [lax.argmin_p, lax.argmax_p]
# )


def _make_iota_harness(name, *, shape=(2, 3), dtype=np.float32, dimension=0):
  define(
      lax.iota_p,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{dimension=}",
      lambda dtype, shape, dim:
      (lax.iota_p.bind(dtype=np.dtype(dtype), shape=shape, dimension=dim,
                       sharding=None)),
      [StaticArg(dtype),
       StaticArg(shape),
       StaticArg(dimension)],
      shape=shape,
      dtype=dtype,
      dimension=dimension,
      sharding=None)


for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.boolean):
  _make_iota_harness("dtypes", dtype=dtype)

# Validate broadcasting
for shape, dimension in [
    ((4, 8, 1, 1), 1),  # broadcasting along non-major dimension
    ((4, 8, 1, 1), 2),  # broadcasting along dimension == 1
]:
  _make_iota_harness("broadcasting", shape=shape, dimension=dimension)


def _make_div_rem_harness(prim,
                          name,
                          *,
                          shapes=((2,), (2,)),
                          dtype=np.float32,
                          arrs=(None, None)):
  lhs, rhs = arrs

  def _make_non_zero(rng):
    return jtu.rand_nonzero(rng)(shapes[1], dtype)

  lhs = RandArg(shapes[0], dtype) if lhs is None else lhs
  rhs_shape = rhs.shape if rhs is not None else shapes[1]
  rhs_dtype = rhs.dtype if rhs is not None else dtype
  rhs = CustomArg(_make_non_zero) if rhs is None else rhs

  define(
      prim,
      f"{name}_lhs={jtu.format_shape_dtype_string(lhs.shape, lhs.dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)}",
      prim.bind, [lhs, rhs],
      dtype=dtype,
      lhs=lhs,
      rhs=rhs,
      prim=prim)


for prim in [lax.div_p, lax.rem_p]:
  for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.boolean) - (
      set() if prim is lax.div_p else set(jtu.dtypes.complex)):
    _make_div_rem_harness(prim, "dtypes", dtype=dtype)

  # Validate broadcasting
  for shapes in [
      ((2, 1, 3), (2, 4, 3)),  # broadcast dividend
      ((2, 4, 3), (2, 1, 3)),  # broadcast divisor
  ]:
    _make_div_rem_harness(prim, "broadcast", shapes=shapes)

  # Validate singularity points
  for name, arrs in [
      ("positive_by_0", (np.ones(
          (2,), dtype=np.float32), np.zeros((2,), dtype=np.float32))),
      # TODO: this fails on CPU, different result
      # ("positive_by_0_int32", (np.ones((2,), dtype=np.int32),
      #                         np.zeros((2,), dtype=np.int32))),
      ("negative_by_0", (-np.ones(
          (2,), dtype=np.float32), np.zeros((2,), dtype=np.float32))),
      ("0_by_0", (np.zeros(
          (2,), dtype=np.float32), np.zeros((2,), dtype=np.float32))),
      ("inf_by_inf", (np.array([np.inf], dtype=np.float32),
                      np.array([np.inf], dtype=np.float32))),
  ]:
    _make_div_rem_harness(prim, f"singularity_{name}", arrs=arrs)


def _make_binary_elementwise_harnesses(prim,
                                       dtypes,
                                       default_dtype=np.float32,
                                       broadcasting_dtypes=None,
                                       jax_unimplemented=lambda **kwargs: []):

  def _make(name, *, shapes=((20, 20), (20, 20)), dtype):
    lhs_shape, rhs_shape = shapes
    define(
        prim,
        f"{name}_lhs={jtu.format_shape_dtype_string(lhs_shape, dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, dtype)}",
        prim.bind, [RandArg(lhs_shape, dtype),
                    RandArg(rhs_shape, dtype)],
        jax_unimplemented=jax_unimplemented(
            dtype=dtype, prim=prim, shapes=shapes),
        prim=prim,
        dtype=dtype,
        shapes=shapes)
  broadcasting_dtypes = broadcasting_dtypes or (default_dtype,)
  return (
      # Validate dtypes
      tuple(_make("dtypes", dtype=dtype) for dtype in dtypes) +
      # Validate broadcasting
      tuple(_make("broadcasting", dtype=dtype, shapes=shapes)
            for shapes in [
              ((20, 20), (1, 20)),  # broadcasting rhs
              ((1, 20), (20, 20)),  # broadcasting lhs
            ]
            for dtype in broadcasting_dtypes)
  )


_make_binary_elementwise_harnesses(
    prim=lax.add_p, dtypes=set(jtu.dtypes.all) - set(jtu.dtypes.boolean))

_make_binary_elementwise_harnesses(
    prim=lax.mul_p, dtypes=set(jtu.dtypes.all) - set(jtu.dtypes.boolean))

_make_binary_elementwise_harnesses(
    prim=lax.atan2_p, dtypes=jtu.dtypes.all_floating)

_make_binary_elementwise_harnesses(
    prim=lax.igamma_p, dtypes=jtu.dtypes.all_floating)

_make_binary_elementwise_harnesses(
    prim=lax.igammac_p, dtypes=jtu.dtypes.all_floating)

_make_binary_elementwise_harnesses(
    prim=lax.nextafter_p, dtypes=jtu.dtypes.all_floating)

_make_binary_elementwise_harnesses(
    prim=lax.and_p,
    default_dtype=np.int32,
    dtypes=jtu.dtypes.all_integer + jtu.dtypes.all_unsigned +
    jtu.dtypes.boolean)

_make_binary_elementwise_harnesses(
    prim=lax.or_p,
    default_dtype=np.int32,
    dtypes=jtu.dtypes.all_integer + jtu.dtypes.all_unsigned +
    jtu.dtypes.boolean)

_make_binary_elementwise_harnesses(
    prim=lax.xor_p,
    default_dtype=np.int32,
    dtypes=jtu.dtypes.all_integer + jtu.dtypes.all_unsigned +
    jtu.dtypes.boolean)

_make_binary_elementwise_harnesses(
    prim=lax.shift_left_p,
    default_dtype=np.int32,
    dtypes=jtu.dtypes.all_integer + jtu.dtypes.all_unsigned)

_make_binary_elementwise_harnesses(
    prim=lax.shift_right_logical_p,
    default_dtype=np.int32,
    dtypes=jtu.dtypes.all_integer + jtu.dtypes.all_unsigned)

_make_binary_elementwise_harnesses(
    prim=lax.shift_right_arithmetic_p,
    default_dtype=np.int32,
    dtypes=jtu.dtypes.all_integer + jtu.dtypes.all_unsigned)

_make_binary_elementwise_harnesses(
    prim=lax.sub_p, dtypes=set(jtu.dtypes.all) - set(jtu.dtypes.boolean))

for minmax_p in [lax.min_p, lax.max_p]:
  _make_binary_elementwise_harnesses(
      prim=minmax_p, dtypes=jtu.dtypes.all,
      broadcasting_dtypes=(np.float32, np.complex64, np.complex128))
  # Validate special cases nan/inf/-inf
  for dtype in jtu.dtypes.all_floating + jtu.dtypes.complex:
    define(
        minmax_p,
        f"inf_nan_{jtu.dtype_str(dtype)}",
        minmax_p.bind, [np.array([[np.nan, np.nan, np.nan],
                                  [np.inf, np.inf, np.inf],
                                  [-np.inf, -np.inf, -np.inf]], dtype=dtype),
                        np.array([[np.nan, np.inf, -np.inf],
                                  [np.nan, np.inf, -np.inf],
                                  [np.nan, np.inf, -np.inf]], dtype=dtype)],
        prim=minmax_p,
        dtype=dtype)

def _make_broadcast_in_dim_harness(name,
                                   *,
                                   dtype=np.float32,
                                   shape=(2,),
                                   outshape=(2,),
                                   broadcast_dimensions=(0,)):
  define(
      lax.broadcast_in_dim_p,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{outshape=}_broadcastdimensions={broadcast_dimensions}",
      lambda operand: lax.broadcast_in_dim_p.bind(
          operand, shape=outshape, broadcast_dimensions=broadcast_dimensions,
          sharding=None),
      [RandArg(shape, dtype)],
      shape=shape,
      dtype=dtype,
      outshape=outshape,
      broadcast_dimensions=broadcast_dimensions)


for dtype in jtu.dtypes.all:
  _make_broadcast_in_dim_harness("dtypes", dtype=dtype)

# Validate parameter combinations
for shape, outshape, broadcast_dimensions in [
    [(2,), (3, 2), (1,)],  # add major dimension
    [(2,), (2, 3), (0,)],  # add inner dimension
    [(), (2, 3), ()],  # use scalar shape
    [(1, 2), (4, 3, 2), (0, 2)],  # map size 1 dim to different output dim value
]:
  _make_broadcast_in_dim_harness(
      "parameter_combinations",
      shape=shape,
      outshape=outshape,
      broadcast_dimensions=broadcast_dimensions)

for dtype in jtu.dtypes.all_floating:
  for arg1, arg2, arg3 in [
      (np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.3, 1, 1.4, 1.6], dtype=dtype),
       np.array([-1.6, 1.4, 1.0, 0.0, 0.2, 0.1, 1, 1.4, -1.6], dtype=dtype),
       np.array([1.0, -1.0, 2.0, 1.0, 0.3, 0.3, -1.0, 2.4, 1.6], dtype=dtype))
  ]:
    define(
        lax.regularized_incomplete_beta_p,
        f"_{jtu.dtype_str(dtype)}",
        lax.betainc, [arg1, arg2, arg3],
        dtype=dtype)

## GATHER
# Validate dtypes
for dtype in set(jtu.dtypes.all):
  indices = np.array(2, dtype=np.int32)
  shape = (10,)
  axis = 0
  define(
      lax.gather_p,
      f"dtypes_shape={jtu.format_shape_dtype_string(shape, dtype)}_{axis=}_enable_xla=True",
      lambda a, i, axis: jnp.take(a, i, axis=axis),
      [RandArg(shape, dtype), indices,
       StaticArg(axis)],
      dtype=dtype,
      enable_xla=True)

# Construct gather harnesses using take
_gather_input = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
for indices, index_oob, indices_name in [
    # Ensure each set of indices has a distinct name
    (np.array(2, dtype=np.int32), False, "1"),
    (np.array([2], dtype=np.int32), False, "2"),
    (np.array([2, 4], dtype=np.int32), False, "3"),
    (np.array([2, 4], dtype=np.uint32), False, "3_uint32"),  # uint32 indices
    (np.array([[2, 4], [5, 6]], dtype=np.int32), False, "4"),
    (np.array([[0], [1], [10]], dtype=np.int32), True, "5_oob"), # Index out of bounds too high
    (np.array([[0, 1], [2, -1]], dtype=np.int32), False, "6_neg"), # Negative index is from the end
    (np.array([0, 1, 2, 3, -10], dtype=np.int32), False, "7_neg"), # Index out of bounds, but works
    (np.array([[[0], [1]], [[3], [-11]]], dtype=np.int32), True, "8_neg_oob")  # Index out of bounds, too low
]:
  for axis in [0, 1, 2]:
    for enable_xla in [True, False]:
      for mode in ["clip", "fill"]:
        define(
            lax.gather_p,
            f"from_take_{indices_name=}_{axis=}_{enable_xla=}_{mode=!s}",
            lambda a, i, axis, mode: jnp.take(a, i, axis=axis, mode=mode),
            [_gather_input, indices, StaticArg(axis), StaticArg(mode)],
            dtype=_gather_input.dtype,
            enable_xla=enable_xla,
            index_oob=index_oob,
            mode=mode)

# Construct gather harnesses using array indexing and slicing.
for slices, name in [
    ((0,), "[0]"),
    ((0, 1), "[0,1]"),
    ((slice(0, 10), 2, slice(0, 10)), "[:,:2,:]"),
    ((slice(2, 5), 5), "[2:5,5]"),
    ((-1, -5, -200), "[-1,-5,-200]"),
    ((slice(5, -2), 300), "[5:-2,300]"),
]:
  for enable_xla in [False, True]:
    define(
        lax.gather_p,
        f"from_slicing_{name=}_{enable_xla=}",
        lambda arr, *s: jnp.array(arr).__getitem__(*s),
        [_gather_input, StaticArg(slices)],
        dtype=_gather_input.dtype,
        enable_xla=enable_xla)

# Directly from lax.gather in lax_test.py.
for shape, idxs, dnums, slice_sizes, needs_xla in [
    ((5,), np.array([[0], [2]]),
     lax.GatherDimensionNumbers(
         offset_dims=(), collapsed_slice_dims=(0,),
         start_index_map=(0,)), (1,), False),
    ((10,), np.array([[0], [0], [0]]),
     lax.GatherDimensionNumbers(
         offset_dims=(1,), collapsed_slice_dims=(),
         start_index_map=(0,)), (2,), True),
    ((10, 5,), np.array([[0], [2], [1]]),
     lax.GatherDimensionNumbers(
         offset_dims=(1,), collapsed_slice_dims=(0,),
         start_index_map=(0,)), (1, 3), True),
    ((10, 6), np.array([[0, 2], [1, 0]]),
     lax.GatherDimensionNumbers(
         offset_dims=(1,), collapsed_slice_dims=(0,),
         start_index_map=(0, 1)), (1, 3), True),
    ((2, 5), np.array([[[0], [2]], [[1], [1]]]),
     lax.GatherDimensionNumbers(
         offset_dims=(), collapsed_slice_dims=(1,),
         start_index_map=(1,), operand_batching_dims=(0,),
         start_indices_batching_dims=(0,)),
     (1, 1), True),
    ((2, 3, 10), np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
     lax.GatherDimensionNumbers(
         offset_dims=(2,), collapsed_slice_dims=(),
         start_index_map=(2,), operand_batching_dims=(0, 1),
         start_indices_batching_dims=(1, 0)),
     (1, 1, 3), True)
]:
  dtype = np.float32
  for enable_xla in ([True] if needs_xla else [True, False]):
    define(
        lax.gather_p,
        f"{shape=}_idxs_shape={idxs.shape}_{dnums=}_{slice_sizes=}_{enable_xla=}",
        lambda op, idxs, dnums, slice_sizes: lax.gather(
            op, idxs, dimension_numbers=dnums, slice_sizes=slice_sizes),
        [RandArg(shape, dtype), idxs,
         StaticArg(dnums),
         StaticArg(slice_sizes)],
        dtype=dtype,
        enable_xla=enable_xla)


# Test cases lax.gather with non-empty batch_dims. This is for instance
# triggered when executing `jax.vmap(lax.dynamic_slice)`.
# We currently only support the case where we have a single batch dimension.
dnums_2d = lax.GatherDimensionNumbers(
    offset_dims=(1,),
    collapsed_slice_dims=(0,),  # Batch dimension.
    start_index_map=(0, 1))
dnums_2d_2 = lax.GatherDimensionNumbers(
    offset_dims=(),
    collapsed_slice_dims=(0, 1),  # Only first dim is batch, collapse both.
    start_index_map=(0, 1))
dnums_3d = lax.GatherDimensionNumbers(
    offset_dims=(1, 2),
    collapsed_slice_dims=(0,),  # Batch dimension.
    start_index_map=(0, 1, 2))

for op_shape, start_indices, slice_sizes, dnums in [
    ((4, 6),    [[0, 1], [1, 2], [2, 3], [3, 2]], (1, 3),    dnums_2d),
    ((1, 2),    [[0, 1]], (1, 1),                            dnums_2d_2),
    ((2, 6, 3), [[0, 1, 0], [1, 2, 0]],           (1, 3, 3), dnums_3d),
    # Test out of bounds behavior.
    ((3, 10),   [[0, 0], [1, 8], [2, 0]],         (1, 5),    dnums_2d),
    ((2, 3, 3), [[0, 1, 0], [1, 2, 1]],           (1, 3, 2), dnums_3d)]:
  start_indices = np.array(start_indices)
  for enable_xla in [True, False]:
    define(
        lax.gather_p,
        f"batchdims_shape={op_shape}_start_indices_shape={start_indices.shape}_{slice_sizes=}_{enable_xla=}",
        lambda op, idxs, dnums, slice_sizes: lax.gather(
                  op, idxs, dimension_numbers=dnums, slice_sizes=slice_sizes),
        [RandArg(op_shape, np.float32), start_indices,
        StaticArg(dnums),
        StaticArg(slice_sizes)],
        dtype=np.float32,
        enable_xla=enable_xla)


# Test cases lax.gather with non-empty 2D batch_dims. This is for instance
# triggered when executing `jax.vmap(jax.vmap(lax.dynamic_slice))`.
gather_2d_bd = lax.GatherDimensionNumbers(
    offset_dims=(2, 3, 4), collapsed_slice_dims=(), start_index_map=(1, 2)
)

gather_2d_bd_nid = lax.GatherDimensionNumbers(
    offset_dims=(2, 3, 4), collapsed_slice_dims=(), start_index_map=(2, 1)
)

gather_3d_bd = lax.GatherDimensionNumbers(
    offset_dims=(1, 2, 3, 4), collapsed_slice_dims=(), start_index_map=(1, 2, 3)
)

gather_2d_bd2 = lax.GatherDimensionNumbers(
    offset_dims=(2, 3, 4), collapsed_slice_dims=(), start_index_map=(1, 2)
)

for op_shape, start_indices, slice_sizes, dnums in [
    ((10, 10, 10), [[[0, 1], [1, 0], [0, 1], [1, 0]]], (2, 2, 2), gather_2d_bd,),  # non-contigous 2d batch dims
    ((10, 10, 10), [[[-1, 1], [1, -1], [0, -1], [-1, 0]]], (2, 2, 3), gather_2d_bd,),  # negative indices
    ((10, 10, 10), [[[1, 1], [1, 1], [0, 1], [2147483647, 0]]], (2, 3, 3), gather_2d_bd,),  # oob indices
    ((10, 10, 10), [[[0, 1], [1, 0], [0, 1], [1, 0]]], (2, 2, 2), gather_2d_bd_nid,),  # start_index_map not identity
    ((10, 10, 10), [[[0, 1], [1, 0], [0, 1], [1, 0]]], (10, 10, 10), gather_2d_bd,),  # oob behavior coming from slice_sizes
    ((10,10,10,10),    [[[[0, 1,0], [1, 0,0], [0, 1,0], [1, 0,0]]]], (10,2,2,2),    gather_3d_bd),  # test 3d batch dims
    ((10,10,10),    [[[0, 1], [1, 0], [0, 1], [1, 0]]], (10,2,2),    gather_2d_bd2),  # test contiguous 2d batch dims
]:
  start_indices = np.array(start_indices)
  for enable_xla in [True, False]:
    define(
        "gather",
        f"op_shape={op_shape}_offset_dims={dnums.offset_dims}_start_index_map={dnums.start_index_map}_start_indices_shape={start_indices.shape}_{slice_sizes=}_{enable_xla=}",
        lambda op, idxs, dnums, slice_sizes: lax.gather(
            op, idxs, dimension_numbers=dnums, slice_sizes=slice_sizes
        ),
        [
            RandArg(op_shape, np.float32),
            start_indices,
            StaticArg(dnums),
            StaticArg(slice_sizes),
        ],
        dtype=np.float32,
        enable_xla=enable_xla,
    )

def _make_scatter_harness(name,
                          *,
                          shape=(5,),
                          f_lax=lax.scatter_min,
                          indices_are_sorted=False,
                          unique_indices=False,
                          scatter_indices=np.array([[0], [2]]),
                          update_shape=(2,),
                          mode=lax.GatherScatterMode.FILL_OR_DROP,
                          dtype=np.float32,
                          dimension_numbers=lax.ScatterDimensionNumbers(
                              update_window_dims=(), inserted_window_dims=(0,),
                              scatter_dims_to_operand_dims=(0,)),
                          enable_and_disable_xla=False):
  xla_options = [True, False] if enable_and_disable_xla else [True]

  for enable_xla in xla_options:
    define(
        f_lax.__name__,
        f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_scatterindices={scatter_indices.tolist()}_updateshape={update_shape}_{dimension_numbers=}_indicesaresorted={indices_are_sorted}_uniqueindices={unique_indices}_{mode=!s}_enablexla={enable_xla}"
        .replace(" ", ""),
        partial(
            f_lax,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode), [
                RandArg(shape, dtype),
                StaticArg(scatter_indices),
                RandArg(update_shape, dtype),
                StaticArg(dimension_numbers)
            ],
        jax_unimplemented=[
            Limitation(
                "unimplemented",
                dtypes=[np.bool_],
                enabled=(f_lax in [lax.scatter_add, lax.scatter_mul])),
        ],
        f_lax=f_lax,
        shape=shape,
        dtype=dtype,
        scatter_indices=scatter_indices,
        update_shape=update_shape,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
        enable_xla=enable_xla)


# Validate dtypes
for dtype in jtu.dtypes.all:
  for f_lax in [
      lax.scatter, lax.scatter_add, lax.scatter_mul, lax.scatter_max, lax.scatter_min
  ]:
    _make_scatter_harness("dtypes", dtype=dtype, f_lax=f_lax)

# Validate f_lax/update_jaxpr
# We explicitly decide against testing lax.scatter, as its reduction function
# is lambda x, y: y, which is not commutative and thus makes results
# non-deterministic when an index into the operand is updated several times.

# Validate shapes, dimension numbers and scatter indices. All are in bounds.
for shape, scatter_indices, update_shape, dimension_numbers in [
    ((10,), [[0], [0], [0]], (3, 2),
     lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,))),
    ((10, 5), [[0], [2], [1]], (3, 3),
     lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))),
    ((2, 3, 10), [[[0], [1]], [[2], [3]], [[4], [5]]], (3, 2, 3),
     lax.ScatterDimensionNumbers(
         update_window_dims=(2,), inserted_window_dims=(),
         scatter_dims_to_operand_dims=(2,), operand_batching_dims=(0, 1),
         scatter_indices_batching_dims=(1, 0)))
]:
  _make_scatter_harness(
      "shapes_and_dimension_numbers",
      shape=shape,
      update_shape=update_shape,
      scatter_indices=np.array(scatter_indices),
      dimension_numbers=dimension_numbers)

# Validate sorted indices
_make_scatter_harness("indices_are_sorted", indices_are_sorted=True)
# Validate unique_indices
# `unique_indices` does not affect correctness, only performance, and thus
# does not need to be tested here. If/when it will make sense to add a test
# with `unique_indices` = True, particular care will have to be taken with
# regards to the choice of parameters, as the results are only predictable
# when all the indices to be updated are pairwise non-overlapping. Identifying
# such cases is non-trivial.
# _make_scatter_harness("unique_indices", unique_indices=False)

# Validate different out-of-bounds modes
for mode in (lax.GatherScatterMode.PROMISE_IN_BOUNDS,
             lax.GatherScatterMode.FILL_OR_DROP):
  for f_lax in [
      lax.scatter_add, lax.scatter_mul, lax.scatter_max, lax.scatter_min, lax.scatter
  ]:
    _make_scatter_harness("modes_in_bounds",
                          f_lax=f_lax,
                          mode=mode)
    _make_scatter_harness(
        "modes_out_of_bounds",
        mode=mode,
        shape=(1, 5),
        f_lax=f_lax,
        scatter_indices=np.array([10]),
        update_shape=(1,),
        dimension_numbers=lax.ScatterDimensionNumbers((0,), (1,), (1,)),
        enable_and_disable_xla=True,
    )

# Validate no XLA scatters
for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.complex) - set(jtu.dtypes.boolean):
  for f_lax in [
      lax.scatter_add, lax.scatter_mul, lax.scatter_max, lax.scatter_min, lax.scatter
  ]:
    for shape, scatter_indices, update_shape, dimension_numbers in [
        ((1,), [0], (),
         lax.ScatterDimensionNumbers((), (0,), (0,))),  # zero case
        ((1, 1), [0], (1,),
         lax.ScatterDimensionNumbers((0,), (0,), (0,))),
        ((1, 1, 1), [0], (1, 1),
         lax.ScatterDimensionNumbers((0, 1), (0,), (0,))),
        ((1, 50, 3), [32], (1, 3),
         lax.ScatterDimensionNumbers((0, 1), (1,), (1,))),
        ((1, 2, 3), [1], (1, 3),
         lax.ScatterDimensionNumbers((0, 1), (1,), (1,))),  # slice 2nd dim
        ((1, 2, 3), [0], (2, 3),
         lax.ScatterDimensionNumbers((0, 1), (0,), (0,))),  # slice 1st dim
        ((1, 2, 3), [1, 2], (1,),
         lax.ScatterDimensionNumbers((0,), (1, 2), (1, 2))),  # 2nd and 3rd
        ((4, 2, 3), [3, 2], (2,),
         lax.ScatterDimensionNumbers((0,), (0, 2), (0, 2))),  # 1st and 3rd
        ((4, 2, 3, 5), [0, 4], (4, 3),
         lax.ScatterDimensionNumbers((0, 1), (1, 3), (1, 3))),  # 2nd and 4th
        ((5, 6, 7), [[0, 1], [2, 3]], (2, 7),
         lax.ScatterDimensionNumbers((1,), (0, 1), (0, 1))),
         # .at[((3,4),(5,5))] shapes
        ((5, 6, 7), [[[0], [1]], [[2], [3]]], (5, 2, 2, 7),
         lax.ScatterDimensionNumbers((0, 3), (1,), (1,))),
         # .at[:, ((3,4),(5,5))] shapes
        ((5, 6, 7), [[[0, 1], [2, 3]], [[4, 0], [1, 2]]], (5, 2, 2),
         lax.ScatterDimensionNumbers((0,), (1, 2), (1, 2))),
         # .at[:, ((3,4),(5,5)), 3] shapes
        ((1, 125), [0], (1,), lax.ScatterDimensionNumbers((0,), (1,), (1,))),
    ]:
      for mode in (lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                   lax.GatherScatterMode.FILL_OR_DROP):
        _make_scatter_harness(
            "no_xla_unique_indices",
            shape=shape,
            f_lax=f_lax,
            unique_indices=True,
            scatter_indices=np.array(scatter_indices),
            update_shape=update_shape,
            mode=mode,
            dtype=dtype,
            dimension_numbers=dimension_numbers,
            enable_and_disable_xla=True)

# Validate no XLA scatters with non-unique indices with an indexing depth of 1.
for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.complex) - set(jtu.dtypes.boolean):
  # Note that lax.scatter currently does not work here.
  for f_lax in [
      lax.scatter_add, lax.scatter_mul, lax.scatter_max, lax.scatter_min
  ]:
    for shape, scatter_indices, update_shape, dimension_numbers in [
        ((1,), [[0],[0]], (2,),
         lax.ScatterDimensionNumbers((), (0,), (0,))),  # .at[((0,0),)]
        ((3,), [[1],[0],[1]], (3,),
         lax.ScatterDimensionNumbers((), (0,), (0,))),  # .at[((1,0,1),)]
        ((2, 3), [[[2],[2],[2]]], (2, 1, 3),
         lax.ScatterDimensionNumbers((0,), (1,), (1,))),  # 2nd dim, .at[:, ((2,2,2),)]
        ((3, 5, 40), [[1],[1]], (3, 5, 2),
         lax.ScatterDimensionNumbers((0, 1), (2,), (2,))),
        ((3, 5, 4), [[1],[1]], (3, 2, 4),
         lax.ScatterDimensionNumbers((0, 2), (1,), (1,))),
    ]:
      for mode in (lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                   lax.GatherScatterMode.FILL_OR_DROP):
        _make_scatter_harness(
            "no_xla_non_unique_indices",
            shape=shape,
            f_lax=f_lax,
            unique_indices=False,
            scatter_indices=np.array(scatter_indices),
            update_shape=update_shape,
            mode=mode,
            dtype=dtype,
            dimension_numbers=dimension_numbers,
            enable_and_disable_xla=True)


for dtype in jtu.dtypes.all:
  arg_shape = (2, 3)
  for pads in [
      [(0, 0, 0), (0, 0, 0)],  # no padding
      [(1, 1, 0), (2, 2, 0)],  # only positive edge padding
      [(1, 2, 1), (0, 1, 0)],  # edge padding and interior padding
      [(0, 0, 0), (-1, -1, 0)],  # negative padding
      [(0, 0, 0), (-2, -2, 4)],  # add big dilation then remove from edges
      [(0, 0, 0), (-2, -3, 1)],  # remove everything in one dimension
  ]:
    for enable_xla in [True, False]:
      define(
        lax.pad_p,
        f"inshape={jtu.format_shape_dtype_string(arg_shape, dtype)}_{pads=}_{enable_xla=}",
        lax.pad,
        [RandArg(arg_shape, dtype),
         np.array(0, dtype),
         StaticArg(pads)],
        rng_factory=jtu.rand_small,
        arg_shape=arg_shape,
        dtype=dtype,
        pads=pads,
        enable_xla=enable_xla)


def _make_select_n_harness(name,
                         *,
                         shape_pred=(2, 3),
                         shape_args=(2, 3),
                         dtype=np.float32):
  define(
      lax.select_n_p,
      f"{name}_shapepred={jtu.format_shape_dtype_string(shape_pred, np.bool_)}_shapeargs={jtu.format_shape_dtype_string(shape_args, dtype)}",
      lax.select_n, [
          RandArg(shape_pred, np.bool_),
          RandArg(shape_args, dtype),
          RandArg(shape_args, dtype)
      ],
      shape_pred=shape_pred,
      shape_args=shape_args,
      dtype=dtype)
  define(
      lax.select_n_p,
      f"{name}_shapepred={jtu.format_shape_dtype_string(shape_pred, np.int32)}_shapeargs={jtu.format_shape_dtype_string(shape_args, dtype)}",
      lax.select_n, [
          CustomArg(lambda rng: jtu.rand_int(rng, high=3)(shape_args, np.int32)),
          RandArg(shape_args, dtype),
          RandArg(shape_args, dtype),
          RandArg(shape_args, dtype),
      ],
      shape_pred=shape_pred,
      shape_args=shape_args,
      dtype=dtype)

for dtype in jtu.dtypes.all:
  _make_select_n_harness("dtypes", dtype=dtype)

# Validate shapes
_make_select_n_harness("shapes", shape_pred=(), shape_args=(18,))


def _make_transpose_harness(name,
                            *,
                            shape=(2, 3),
                            permutation=(1, 0),
                            dtype=np.float32):
  define(
      lax.transpose_p,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{permutation=}"
      .replace(" ", ""),
      lambda x: lax.transpose_p.bind(x, permutation=permutation),
      [RandArg(shape, dtype)],
      shape=shape,
      dtype=dtype,
      permutation=permutation)


for dtype in jtu.dtypes.all:
  _make_transpose_harness("dtypes", dtype=dtype)

# Validate permutations
for shape, permutation in [
    ((2, 3, 4), (0, 1, 2)),  # identity
    ((2, 3, 4), (1, 2, 0)),  # transposition
]:
  _make_transpose_harness("permutations", shape=shape, permutation=permutation)


## CUMREDUCE
def _make_cumreduce_harness(name,
                            *,
                            f_jax=lax_control_flow.cummin,
                            shape=(8, 9),
                            dtype=np.float32,
                            axis=0,
                            reverse=False):
  limitations = []
  define(
      f_jax.__name__,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{axis=}_{reverse=}",
      f_jax, [RandArg(shape, dtype),
              StaticArg(axis),
              StaticArg(reverse)],
      jax_unimplemented=limitations,
      f_jax=f_jax,
      shape=shape,
      dtype=dtype,
      axis=axis,
      reverse=reverse,
      associative_scan_reductions=False
  )
  define(
      f_jax.__name__,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_associative_scan_reductions_{axis=}_{reverse=}",
      f_jax, [RandArg(shape, dtype),
              StaticArg(axis),
              StaticArg(reverse)],
      jax_unimplemented=limitations,
      f_jax=f_jax,
      shape=shape,
      dtype=dtype,
      axis=axis,
      reverse=reverse,
      associative_scan_reductions=True
  )

# Validate dtypes for each function
for f_jax in [
    lax_control_flow.cummin,
    lax_control_flow.cummax,
    lax_control_flow.cumlogsumexp,
    lax_control_flow.cumsum,
    lax_control_flow.cumprod,
]:
  for dtype in jtu.dtypes.all:
    # cumlogsumexp is only defined for floating point types.
    if (f_jax == lax_control_flow.cumlogsumexp and
        not np.issubdtype(dtype, np.floating)):
      continue
    if dtype == np.bool_:
      continue
    _make_cumreduce_harness("dtype_by_fun", dtype=dtype, f_jax=f_jax)

  # Validate axis for each function
  shape = (8, 9)
  for axis in range(len(shape)):
    _make_cumreduce_harness("axis_by_fun", axis=axis, f_jax=f_jax, shape=shape)

  # Validate reverse for each function
  _make_cumreduce_harness("reverse", reverse=True, f_jax=f_jax)


### TOP_K
def _make_top_k_harness(name,
                        *,
                        operand=None,
                        shape=(5, 3),
                        dtype=np.float32,
                        k=2):
  if operand is None:
    operand = RandArg(shape, dtype)
  define(
      lax.top_k_p,
      f"{name}_inshape={jtu.format_shape_dtype_string(operand.shape, operand.dtype)}_{k=}",
      lax.top_k, [operand, StaticArg(k)],
      shape=operand.shape,
      dtype=operand.dtype,
      k=k)


for dtype in set(jtu.dtypes.all) - set(jtu.dtypes.complex):
  # Validate dtypes
  _make_top_k_harness("dtypes", dtype=dtype)

# Validate implicit properties of the sort
for name, operand, k in [("stability",
                          np.array([5, 7, 5, 8, 8, 5], dtype=np.int32), 3),
                         ("sort_inf_nan",
                          np.array([+np.inf, np.nan, -np.nan, -np.inf, 3],
                                   dtype=np.float32), 5)]:
  _make_top_k_harness(name, operand=operand, k=k)


### SORT
def _make_sort_harness(name,
                       *,
                       operands=None,
                       shape=(5, 7),
                       dtype=np.float32,
                       dimension=0,
                       is_stable=False,
                       num_keys=1):
  if operands is None:
    operands = [RandArg(shape, dtype)]
  define(
      lax.sort_p,
      f"{name}_num_arrays={len(operands)}_shape={jtu.format_shape_dtype_string(operands[0].shape, operands[0].dtype)}_axis={dimension}_isstable={is_stable}_{num_keys=}",
      lambda *args: lax.sort_p.bind(
          *args[:-3], dimension=args[-3], is_stable=args[-2], num_keys=args[-1]
      ), [
          *operands,
          StaticArg(dimension),
          StaticArg(is_stable),
          StaticArg(num_keys)
      ],
      shape=operands[0].shape,
      dimension=dimension,
      dtype=operands[0].dtype,
      is_stable=is_stable,
      num_keys=num_keys,
      num_arrays=len(operands))


_lax_sort_multiple_array_shape = (100,)
# In order to test lexicographic ordering and sorting stability, the first
# array contains only integers 0 and 1
_lax_sort_multiple_array_first_arg = (
    np.random.uniform(0, 2, _lax_sort_multiple_array_shape).astype(np.int32))

# Validate dtypes
for dtype in jtu.dtypes.all:
  _make_sort_harness("dtypes", dtype=dtype)

# Validate dimensions
for dimension in [0, 1]:
  _make_sort_harness("dimensions", dimension=dimension)
# Validate stable sort
_make_sort_harness("is_stable", is_stable=True)
# Potential edge cases
for operands, dimension in [
    ([np.array([+np.inf, np.nan, -np.nan, -np.inf, 2], dtype=np.float32)], 0)
]:
  _make_sort_harness("edge_cases", operands=operands, dimension=dimension)

# Validate multiple arrays, num_keys, and is_stable
for is_stable in [False, True]:
  for operands in (
      [
          _lax_sort_multiple_array_first_arg,
          RandArg(_lax_sort_multiple_array_shape, np.int32)
      ],
      [
          _lax_sort_multiple_array_first_arg,
          RandArg(_lax_sort_multiple_array_shape, np.int32),
          RandArg(_lax_sort_multiple_array_shape, np.float32)
      ],
  ):
    for num_keys in range(1, len(operands) + 1):
      _make_sort_harness(
          "multiple_arrays",
          operands=operands,
          num_keys=num_keys,
          is_stable=is_stable,
          shape=_lax_sort_multiple_array_first_arg.shape,
          dtype=_lax_sort_multiple_array_first_arg.dtype)


def _make_cholesky_arg(shape, dtype, rng):
  a = jtu.rand_default(rng)(shape, dtype)
  return np.matmul(a, jnp.conj(np.swapaxes(a, -1, -2)))


for dtype in jtu.dtypes.all_inexact:
  for shape in [(1, 1), (4, 4), (2, 5, 5), (200, 200), (1000, 0, 0)]:
    define(
        lax.linalg.cholesky_p,
        f"shape={jtu.format_shape_dtype_string(shape, dtype)}",
        lambda *args: lax.linalg.cholesky_p.bind(*args),
        [CustomArg(partial(_make_cholesky_arg, shape, dtype))],
        jax_unimplemented=[
            Limitation(
                "unimplemented", dtypes=[np.float16], devices=("cpu", "gpu"))
        ],
        shape=shape,
        dtype=dtype)

for dtype in jtu.dtypes.all_floating + jtu.dtypes.complex:
  for shape in [(1, 1), (3, 3), (3, 4), (2, 10, 5), (2, 200, 100)]:
    for full_matrices in [False, True]:
      define(
          lax.linalg.qr_p,
          f"multi_array_shape={jtu.format_shape_dtype_string(shape, dtype)}_fullmatrices={full_matrices}",
          partial(lax.linalg.qr, full_matrices=full_matrices),
          [RandArg(shape, dtype)],
          # See jax._src.lib.lapack.geqrf for the list of compatible types
          jax_unimplemented=[
              Limitation(
                  "unimplemented",
                  devices=("cpu", "gpu"),
                  dtypes=[np.float16, dtypes.bfloat16]),
          ],
          shape=shape,
          dtype=dtype,
          full_matrices=full_matrices)


def _make_fft_harness(name,
                      *,
                      shape=(14, 15, 16, 17),
                      dtype=np.float32,
                      fft_type=lax.FftType.FFT,
                      fft_lengths=(17,)):

  def _fft_rng_factory(dtype):
    _all_integers = (
        jtu.dtypes.all_integer + jtu.dtypes.all_unsigned + jtu.dtypes.boolean)
    # For integer types, use small values to keep the errors small
    if dtype in _all_integers:
      return jtu.rand_small
    else:
      return jtu.rand_default

  define(
      lax.fft_p,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_ffttype={fft_type}_fftlengths={fft_lengths}",
      lambda *args: lax.fft_p.bind(
          args[0], fft_type=args[1], fft_lengths=args[2]),
      [RandArg(shape, dtype),
       StaticArg(fft_type),
       StaticArg(fft_lengths)],
      rng_factory=_fft_rng_factory(dtype),
      shape=shape,
      dtype=dtype,
      fft_type=fft_type,
      fft_lengths=fft_lengths)


# FFT, IFFT, RFFT, IRFFT
for fft_type in list(map(lax.FftType, [0, 1, 2, 3])):
  # Validate dtypes per FFT type
  for dtype in (jtu.dtypes.floating
                if fft_type == lax.FftType.RFFT else jtu.dtypes.complex):
    shape = (14, 15, 16, 17)
    if fft_type != lax.FftType.IRFFT:
      fft_lengths_list = [ (shape[-1],) ]
    else:
      fft_lengths_list = [ ((shape[-1] - 1) * 2,), (shape[-1] * 2 - 1,) ]
    for fft_lengths in fft_lengths_list:
      _make_fft_harness(
          "dtypes",
          shape=shape,
          dtype=dtype,
          fft_type=fft_type,
          fft_lengths=fft_lengths)
      # And with a 0 shape
      _make_fft_harness(
          "dtypes_zero",
          shape=(14, 15, 0, 17),
          dtype=dtype,
          fft_type=fft_type,
          fft_lengths=fft_lengths)

  # Validate dimensions per FFT type
  for dtype in [
      np.float32 if fft_type == lax.FftType.RFFT else np.complex64
  ]:
    for dims in [1, 2, 3]:
      for fft_lengths in [
          shape[-dims:] if fft_type != lax.FftType.IRFFT else
          shape[-dims:-1] + ((shape[-1] - 1) * 2,)
      ]:
        _make_fft_harness(
            "dims",
            shape=shape,
            fft_type=fft_type,
            fft_lengths=fft_lengths,
            dtype=dtype)

for dtype in jtu.dtypes.all_floating + jtu.dtypes.complex:
  for shape in [(2, 2), (2, 7), (29, 29), (2, 3, 53), (2, 3, 29, 7)]:
    for full_matrices in [False, True]:
      for compute_uv in [False, True]:
        subset_by_index = None
        define(
            lax.linalg.svd_p,
            f"shape={jtu.format_shape_dtype_string(shape, dtype)}_fullmatrices={full_matrices}_computeuv={compute_uv}",
            lambda *args: lax.linalg.svd_p.bind(
                args[0],
                full_matrices=args[1],
                compute_uv=args[2],
                subset_by_index=args[3],
            ),
            [
                RandArg(shape, dtype),
                StaticArg(full_matrices),
                StaticArg(compute_uv),
                StaticArg(subset_by_index),
            ],
            jax_unimplemented=[
                Limitation(
                    "unimplemented",
                    dtypes=[np.float16, dtypes.bfloat16],
                ),
            ],
            shape=shape,
            dtype=dtype,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            subset_by_index=subset_by_index,
        )

for dtype in jtu.dtypes.all_inexact:
  for shape in [(0, 0), (5, 5), (2, 6, 6)]:
    for compute_left_eigenvectors in [False, True]:
      for compute_right_eigenvectors in [False, True]:
        define(
            lax.linalg.eig_p,
            f"shape={jtu.format_shape_dtype_string(shape, dtype)}_computelefteigenvectors={compute_left_eigenvectors}_computerighteigenvectors={compute_right_eigenvectors}",
            partial(lax.linalg.eig, compute_left_eigenvectors=compute_left_eigenvectors, compute_right_eigenvectors=compute_right_eigenvectors),
            [
                RandArg(shape, dtype),
            ],
            jax_unimplemented=[
                Limitation(
                    "only supported on CPU in JAX", devices=("tpu", "gpu")),
                Limitation(
                    "unimplemented",
                    devices="cpu",
                    dtypes=[np.float16, dtypes.bfloat16])
            ],
            shape=shape,
            dtype=dtype,
            compute_left_eigenvectors=compute_left_eigenvectors,
            compute_right_eigenvectors=compute_right_eigenvectors)


def _make_triangular_eigh_operand(shape, dtype, lower: bool, rng: Rng):
  # For testing eigh we use triangular matrices
  operand = jtu.rand_default(rng)(shape, dtype)
  # Make operand self-adjoint
  operand = (operand + np.conj(np.swapaxes(operand, -1, -2))) / 2
  # Make operand lower/upper triangular
  return operand  # np.tril(operand) if lower else np.triu(operand)


for dtype in jtu.dtypes.all_inexact:
  # The eigh implementation for TPU uses different lowering for n >= 256
  # TODO: add a test case for n=300. First attempt resulted in significant
  # numerical differences.
  # And the implementation on GPU uses different lowering for n >= 32
  for shape in [(0, 0), (50, 50), (2, 20, 20)]:
    for lower in [False, True]:
      define(
          lax.linalg.eigh_p,
          f"shape={jtu.format_shape_dtype_string(shape, dtype)}_{lower=}",
          # Make operand lower/upper triangular
          lambda operand, lower, symmetrize_input: (lax.linalg.eigh(
              jnp.tril(operand)
              if lower else jnp.triu(operand), lower=lower, symmetrize_input=symmetrize_input)),
          [
              CustomArg(
                  partial(_make_triangular_eigh_operand, shape, dtype, lower)),
              StaticArg(lower),
              StaticArg(False)
          ],
          jax_unimplemented=[
              Limitation(
                  "unimplemented",
                  devices=("cpu", "gpu"),
                  dtypes=[np.float16, dtypes.bfloat16]),
          ],
          shape=shape,
          dtype=dtype,
          lower=lower)

for dtype in jtu.dtypes.all_inexact:
  for shape in [
      (5, 5),  # square
      (3, 5, 5),  # batched
      (3, 5),  # non-square
  ]:
    define(
        lax.linalg.lu_p,
        f"shape={jtu.format_shape_dtype_string(shape, dtype)}",
        lax.linalg.lu, [RandArg(shape, dtype)],
        jax_unimplemented=[
            Limitation("unimplemented", dtypes=[np.float16, dtypes.bfloat16])
        ],
        shape=shape,
        dtype=dtype)


def _make_triangular_solve_harness(name,
                                   *,
                                   left_side=True,
                                   lower=False,
                                   ab_shapes=((4, 4), (4, 1)),
                                   dtype=np.float32,
                                   transpose_a=False,
                                   conjugate_a=False,
                                   unit_diagonal=False):
  a_shape, b_shape = ab_shapes
  f_lax = lambda a, b: (
      lax.linalg.triangular_solve_p.bind(
          a,
          b,
          left_side=left_side,
          lower=lower,
          transpose_a=transpose_a,
          conjugate_a=conjugate_a,
          unit_diagonal=unit_diagonal))

  define(
      lax.linalg.triangular_solve_p,
      f"{name}_a={jtu.format_shape_dtype_string(a_shape, dtype)}_b={jtu.format_shape_dtype_string(b_shape, dtype)}_leftside={left_side}_{lower=}_transposea={transpose_a}_conjugatea={conjugate_a}_unitdiagonal={unit_diagonal}",
      f_lax, [RandArg(a_shape, dtype),
              RandArg(b_shape, dtype)],
      jax_unimplemented=[
          Limitation("unimplemented", devices="gpu", dtypes=[np.float16]),
      ],
      dtype=dtype,
      a_shape=a_shape,
      b_shape=b_shape,
      left_side=left_side,
      lower=lower,
      tranpose_a=transpose_a,
      conjugate_a=conjugate_a,
      unit_diagonal=unit_diagonal)


# Validate dtypes
# This first harness runs the tests for all dtypes using default values for
# all the other parameters, except unit_diagonal (to ensure that
# tf.linalg.set_diag works reliably for all dtypes). Variations of other
# parameters can thus safely skip testing their corresponding default value.
# Note that this validates solving on the left.
for dtype in jtu.dtypes.all_inexact:
  for unit_diagonal in [False, True]:
    _make_triangular_solve_harness(
        "dtypes", dtype=dtype, unit_diagonal=unit_diagonal)

# Validate shapes when solving on the right
for ab_shapes in [
    ((4, 4), (1, 4)),  # standard
    ((2, 8, 8), (2, 10, 8)),  # batched
]:
  _make_triangular_solve_harness(
      "shapes_right", ab_shapes=ab_shapes, left_side=False)
# Validate transformations of a complex matrix
for lower in [False, True]:
  for transpose_a in [False, True]:
    for conjugate_a in [False, True]:
      _make_triangular_solve_harness(
          "complex_transformations",
          dtype=np.complex64,
          lower=lower,
          transpose_a=transpose_a,
          conjugate_a=conjugate_a)

# Validate transformations of a real matrix
for lower in [False, True]:
  for transpose_a in [False, True]:
    # conjugate_a is irrelevant for real dtypes, and is thus omitted
    _make_triangular_solve_harness(
        "real_transformations",
        dtype=np.float32,
        lower=lower,
        transpose_a=transpose_a)


def _make_linear_solve_harnesses():

  def linear_solve(a, b, solve, transpose_solve=None, symmetric=False):
    matvec = partial(lax.dot, a, precision=lax.Precision.HIGHEST)
    return lax.custom_linear_solve(matvec, b, solve, transpose_solve, symmetric)

  def explicit_jacobian_solve(matvec, b):
    return lax.stop_gradient(jnp.linalg.solve(jax.jacobian(matvec)(b), b))

  def _make_harness(name,
                    *,
                    shape=(4, 4),
                    dtype=np.float32,
                    symmetric=False,
                    solvers=(explicit_jacobian_solve, explicit_jacobian_solve)):
    solve, transpose_solve = solvers
    transpose_solve_name = transpose_solve.__name__ if transpose_solve else None

    def _make_first_argument(rng):
      a = jtu.rand_default(rng)(shape, dtype)
      if symmetric:
        a = a + a.T
      return a

    define(
        lax.linear_solve_p,
        f"{name}_a={jtu.format_shape_dtype_string(shape, dtype)}_b={jtu.format_shape_dtype_string(shape[:-1], dtype)}_solve={solve.__name__}_transposesolve={transpose_solve_name}_{symmetric=}",
        linear_solve, [
            CustomArg(_make_first_argument),
            RandArg(shape[:-1], dtype),
            StaticArg(solve),
            StaticArg(transpose_solve),
            StaticArg(symmetric)
        ],
        shape=shape,
        dtype=dtype,
        solve=solve,
        transpose_solve=transpose_solve,
        symmetric=symmetric)

  for dtype in jtu.dtypes.all_floating:
    if not dtype in [np.float16, dtypes.bfloat16]:
      _make_harness("dtypes", dtype=dtype)
  # Validate symmetricity
  _make_harness("symmetric", symmetric=True)
  # Validate removing transpose_solve
  _make_harness("transpose_solve", solvers=(explicit_jacobian_solve, None))


_make_linear_solve_harnesses()

# tridiagonal_solve_p
for dtype in [np.float32, np.float64]:
  define(
      lax.linalg.tridiagonal_solve_p,
      f"shape={jtu.format_shape_dtype_string((3,), dtype)}",
      lax.linalg.tridiagonal_solve,
      [ np.array([0.0, 2.0, 3.0], dtype=dtype),
        np.ones(3, dtype=dtype),
        np.array([1.0, 2.0, 0.0], dtype=dtype),
        np.ones([3, 1], dtype=dtype)],
      dtype=dtype)

def _make_slice_harness(name,
                        shape=(3,),
                        start_indices=(1,),
                        limit_indices=(2,),
                        strides=None,
                        dtype=np.float32):
  define(
      lax.slice_p,
      f"{name}_a={jtu.format_shape_dtype_string(shape, dtype)}_{start_indices=}_{limit_indices=}_{strides=}",
      lax.slice,
      [
          RandArg(shape, dtype),
          StaticArg(start_indices),
          StaticArg(limit_indices),
          StaticArg(strides)
      ],
      dtype=dtype,
      shape=shape,
      start_indices=start_indices,
      limit_indices=limit_indices)


# Test first all dtypes
for dtype in jtu.dtypes.all:
  _make_slice_harness("dtypes", dtype=dtype)
# Now test many shapes
for shape, start_indices, limit_indices, strides in [
    ((3,), (1,), (2,), None),
    ((7,), (4,), (7,), None),
    ((5,), (1,), (5,), (2,)),
    ((8,), (1,), (6,), (2,)),
    ((5, 3), (1, 1), (3, 2), None),
    ((5, 3), (1, 1), (3, 1), None),
    ((7, 5, 3), (4, 0, 1), (7, 1, 3), None),
    ((5, 3), (1, 1), (2, 1), (1, 1)),
    ((5, 3), (1, 1), (5, 3), (2, 1)),
]:
  _make_slice_harness(
      "shapes",
      shape=shape,
      start_indices=start_indices,
      limit_indices=limit_indices,
      strides=strides)


def _make_complex_harness(name, *, shapes=((3, 4), (3, 4)), dtype=np.float32):
  define(
      lax.complex_p,
      f"{name}_lhs={jtu.format_shape_dtype_string(shapes[0], dtype)}_rhs={jtu.format_shape_dtype_string(shapes[1], dtype)}",
      lax.complex_p.bind,
      [RandArg(shapes[0], dtype),
       RandArg(shapes[1], dtype)],
      shapes=shapes,
      dtype=dtype)


for dtype in jtu.dtypes.floating:
  _make_complex_harness("dtypes", dtype=dtype)

# Validate broadcasting
for shapes in [
    ((3, 2), (3, 1)),  # broadcast imaginary part
    ((3, 1), (3, 2)),  # broadcast real part
]:
  _make_complex_harness("broadcast", shapes=shapes)


def _make_conj_harness(name, *, shape=(3, 4), dtype=np.float32, **kwargs):
  define(
      lax.conj_p,
      f"{name}_operand={jtu.format_shape_dtype_string(shape, dtype)}_{kwargs=}"
      .replace(" ", ""),
      lambda x: lax.conj_p.bind(x, **kwargs), [RandArg(shape, dtype)],
      shape=shape,
      dtype=dtype,
      **kwargs)


for dtype in jtu.dtypes.floating + jtu.dtypes.complex:
  _make_conj_harness("dtypes", dtype=dtype)

# Validate kwargs
_make_conj_harness("kwargs", _input_dtype=np.float32)


def _make_real_imag_harness(prim, name, *, shape=(2, 3), dtype=np.float32):
  define(
      prim,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}",
      prim.bind, [RandArg(shape, dtype)],
      shape=shape,
      dtype=dtype,
      prim=prim)


for dtype in jtu.dtypes.complex:
  for prim in [lax.real_p, lax.imag_p]:
    _make_real_imag_harness(prim, "dtypes", dtype=dtype)


def _make_dynamic_slice_harness(name,
                                shape=(3,),
                                start_indices=(1,),
                                limit_indices=(2,),
                                dtype=np.float32):
  for enable_xla in [False, True]:
    define(
        lax.dynamic_slice_p,
        f"{name}_a={jtu.format_shape_dtype_string(shape, dtype)}_{start_indices=}_{limit_indices=}_enablexla={enable_xla}",
        lax.dynamic_slice,
        [
            RandArg(shape, dtype),
            np.array(list(start_indices)),
            StaticArg(tuple(map(operator.sub, limit_indices, start_indices)))
        ],
        dtype=dtype,
        shape=shape,
        start_indices=start_indices,
        limit_indices=limit_indices,
        enable_xla=enable_xla)


# Test first all dtypes
for dtype in jtu.dtypes.all:
  _make_dynamic_slice_harness("dtypes", dtype=dtype)
# Now test many shapes
for shape, start_indices, limit_indices in [
    ((3,), (1,), (2,)),
    ((7,), (4,), (7,)),
    ((5,), (1,), (5,)),
    ((8,), (1,), (6,)),
    ((5, 3), (1, 1), (3, 2)),
    ((7, 5, 3), (4, 0, 1), (7, 1, 3)),
    ((5, 3), (1, 1), (2, 1)),
    ((3,), (np.array(1, np.uint32),), (np.array(2,
                                                np.uint32),)),  # uint32 indices
    ((3,), (np.array(1, np.uint8),), (np.array(2, np.uint8),)),  # uint8 indices
    # out-of-bounds cases, allowed for dynamic_slice
    ((5,), (-1,), (0,)),
    ((5,), (-1,), (1,)),
    ((5,), (-4,), (-2,)),
    ((5,), (-5,), (-2,)),
    ((5,), (-6,), (-5,)),
    ((5,), (-10,), (-9,)),
    ((5,), (-100,), (-99,)),
    ((5,), (5,), (6,)),
    ((5,), (10,), (11,)),
    ((5,), (3,), (6,))
]:
  _make_dynamic_slice_harness(
      "shapes",
      shape=shape,
      start_indices=start_indices,
      limit_indices=limit_indices)


def _make_dynamic_update_slice_harness(name,
                                       shape=(3,),
                                       start_indices=(1,),
                                       dtype=np.float32,
                                       update_shape=(1,)):
  for enable_xla in [False, True]:
    define(
        lax.dynamic_update_slice_p,
        (
            f"{name}_operand={jtu.format_shape_dtype_string(shape, dtype)}"
            f"_update={jtu.format_shape_dtype_string(update_shape, dtype)}"
            f"_{start_indices=}_{enable_xla=}"),
        lax.dynamic_update_slice,
        [
            RandArg(shape, dtype),
            RandArg(update_shape, dtype),
            np.array(start_indices)
        ],
        dtype=dtype,
        shape=shape,
        start_indices=start_indices,
        update_shape=update_shape,
        enable_xla=enable_xla)


# Test first all dtypes
for dtype in jtu.dtypes.all:
  _make_dynamic_update_slice_harness("dtypes", dtype=dtype)
# Now test many shapes
for shape, start_indices, update_shape in [
    ((3,), (1,), (1,)),
    ((5, 3), (1, 1), (3, 1)),
    ((7, 5, 3), (4, 1, 0), (2, 0, 1)),
    ((3,), (np.array(1, np.uint32),), (1,)),  # uint32 indices
    ((3,), (np.array(1, np.uint8),), (1,)),  # uint8 indices
    ((3,), (-1,), (1,)),  # out-of-bounds
    ((3,), (10,), (1,)),  # out-of-bounds
    ((3,), (10,), (2,)),  # out-of-bounds
]:
  _make_dynamic_update_slice_harness(
      "shapes",
      shape=shape,
      start_indices=start_indices,
      update_shape=update_shape)


def _make_squeeze_harness(name,
                          shape=(1, 2),
                          dimensions=(0,),
                          dtype=np.float32):
  define(
      lax.squeeze_p,
      f"{name}_inshape={jtu.format_shape_dtype_string(shape, dtype)}_{dimensions=}",
      lax.squeeze,
      [RandArg(shape, dtype), StaticArg(dimensions)],
      dtype=dtype,
      arg_shape=shape,
      dimensions=dimensions)


# Test first all dtypes
for dtype in set(jtu.dtypes.all):
  _make_squeeze_harness("dtypes", dtype=dtype)
# Now test many shapes
for shape, dimensions in [
    ((1,), (0,)),
    ((1,), (-1,)),
    ((2, 1, 4), (1,)),
    ((2, 1, 4), (-2,)),
    ((2, 1, 3, 1), (1,)),
    ((2, 1, 3, 1), (1, 3)),
    ((2, 1, 3, 1), (3,)),
    ((2, 1, 3, 1), (1, -1)),
]:
  _make_squeeze_harness("shapes", shape=shape, dimensions=dimensions)


def _make_select_and_scatter_add_harness(name,
                                         *,
                                         shape=(2, 4, 6),
                                         dtype=np.float32,
                                         select_prim=lax.ge_p,
                                         window_dimensions=(2, 2, 2),
                                         window_strides=(1, 1, 1),
                                         padding=((0, 0), (0, 0), (0, 0)),
                                         nb_inactive_dims=0):
  ones = (1,) * len(shape)
  cotangent_shape = jax.eval_shape(
      lambda x: lax_windowed_reductions._select_and_gather_add(
          x, x, lax.ge_p, window_dimensions, window_strides, padding,
          ones, ones),
      np.ones(shape, dtype)).shape
  define(
      lax.select_and_scatter_add_p,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_selectprim={select_prim}_windowdimensions={window_dimensions}_windowstrides={window_strides}_{padding=!s}",
      lax_windowed_reductions._select_and_scatter_add, [
          RandArg(cotangent_shape, dtype),
          RandArg(shape, dtype),
          StaticArg(select_prim),
          StaticArg(window_dimensions),
          StaticArg(window_strides),
          StaticArg(padding)
      ],
      jax_unimplemented=[
          Limitation(
              "works only for 2 or more inactive dimensions",
              devices="tpu",
              enabled=(nb_inactive_dims < 2))
      ],
      shape=shape,
      dtype=dtype,
      select_prim=select_prim,
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      padding=padding)


for dtype in set(jtu.dtypes.all) - {np.complex64, np.complex128}:
  _make_select_and_scatter_add_harness("dtypes", dtype=dtype)

# Validate different reduction primitives
_make_select_and_scatter_add_harness("select_prim", select_prim=lax.le_p)

# Validate padding
for padding in [
    # TODO(bchetioui): commented out the test based on
    # https://github.com/jax-ml/jax/issues/4690
    # ((1, 2), (2, 3), (3, 4)) # non-zero padding
    ((1, 1), (1, 1), (1, 1))  # non-zero padding
]:
  _make_select_and_scatter_add_harness("padding", padding=padding)

# Validate window_dimensions; uneven dimensions
_make_select_and_scatter_add_harness(
    "window_dimensions", window_dimensions=(1, 2, 3))

# Validate window_strides
# smaller than/same as/bigger than corresponding window dimension
_make_select_and_scatter_add_harness("window_strides", window_strides=(1, 2, 3))

# Validate dtypes on TPU
for dtype in set(jtu.dtypes.all) - {
    np.bool_, np.complex64, np.complex128, np.int8, np.uint8}:
  for window_strides, window_dimensions, nb_inactive_dims in [((1, 2, 1),
                                                               (1, 3, 1), 2)]:
    _make_select_and_scatter_add_harness(
        "tpu_dtypes",
        dtype=dtype,
        nb_inactive_dims=nb_inactive_dims,
        window_strides=window_strides,
        window_dimensions=window_dimensions)


def _make_select_and_gather_add_harness(name,
                                        *,
                                        shape=(4, 6),
                                        dtype=np.float32,
                                        select_prim=lax.le_p,
                                        padding="VALID",
                                        window_dimensions=(2, 2),
                                        window_strides=(1, 1),
                                        base_dilation=(1, 1),
                                        window_dilation=(1, 1)):
  if isinstance(padding, str):
    padding = tuple(
        lax.padtype_to_pads(shape, window_dimensions, window_strides, padding))
  define(
      lax.select_and_gather_add_p,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_selectprim={select_prim}_windowdimensions={window_dimensions}_windowstrides={window_strides}_{padding=!s}_basedilation={base_dilation}_windowdilation={window_dilation}",
      lax_windowed_reductions._select_and_gather_add, [
          RandArg(shape, dtype),
          RandArg(shape, dtype),
          StaticArg(select_prim),
          StaticArg(window_dimensions),
          StaticArg(window_strides),
          StaticArg(padding),
          StaticArg(base_dilation),
          StaticArg(window_dilation)
      ],
      shape=shape,
      dtype=dtype,
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      padding=padding,
      base_dilation=base_dilation,
      window_dilation=window_dilation)


for dtype in jtu.dtypes.all_floating:
  for select_prim in [lax.ge_p, lax.le_p]:
    _make_select_and_gather_add_harness(
        "dtypes", dtype=dtype, select_prim=select_prim)

# Validate selection primitives
_make_select_and_gather_add_harness("select_prim", select_prim=lax.ge_p)
# Validate window dimensions
_make_select_and_gather_add_harness(
    "window_dimensions", window_dimensions=(2, 3))

# Validate window strides
_make_select_and_gather_add_harness("window_strides", window_strides=(2, 3))
# Validate padding
_make_select_and_gather_add_harness("padding", padding="SAME")

# Validate dilations
for base_dilation, window_dilation in [
    ((2, 3), (1, 1)),  # base dilation, no window dilation
    ((1, 1), (2, 3)),  # no base dilation, window dilation
    ((2, 3), (3, 2))  # base dilation, window dilation
]:
  _make_select_and_gather_add_harness(
      "dilations", base_dilation=base_dilation, window_dilation=window_dilation)

def _make_reduce_harness(name, *,
                         shape=(4, 6),  # The shape of all operands
                         nr_operands=1,  # How many operands
                         computation=lax.add,  # Takes Tuple(op1, [op2,]) and Tuple(init_val1, [init_val2]). Returns Tuple(out_val1, [out_val2]).
                         dimensions: Sequence[int] = (0,),
                         init_value=0,  # The init value for first operand
                         dtype=np.float32):  # The dtype of first operand
  def reducer(*args):
    init_val = np.array(init_value, dtype=dtype)
    init_values = [init_val]
    if nr_operands == 2:
      init_values.append(np.int32(0.))
    return lax.reduce(args[0:nr_operands], tuple(init_values),
                      computation, dimensions)
  define(
      lax.reduce_p,
      f"gen_{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_initvalue={init_value}_{nr_operands=}_{dimensions=}".replace(" ", ""),
      reducer,
      [
          RandArg(shape, dtype),
          # Second operand (optional, always i32). We cannot mix multiple float
          # types in XLA.
          RandArg(shape, np.int32),
      ],
      shape=shape,
      dtype=dtype,
      init_value=init_value,
      computation=computation,
      dimensions=dimensions)

for dtype in jtu.dtypes.all:
  for name, nr_operands, computation, init_value in [
      ("add_scalar", 1,
       lambda ops, inits: (lax.add(ops[0], inits[0]),), 3),
      # Compute the max (starting with 3) and the min (from 0), in parallel
      ("max_min", 2,
       lambda ops, inits: (lax.max(ops[0], inits[0]),
                           lax.min(ops[1], inits[1])), 3),
  ]:
    if not (dtype == np.bool_ and name == "add_scalar"):
      _make_reduce_harness(name, nr_operands=nr_operands,
                           computation=computation, init_value=init_value,
                           dtype=dtype)
    # Test the dimensions, but only for int32 (to keep the # of tests small)
    if dtype == np.int32:
      _make_reduce_harness(name, nr_operands=nr_operands,
                           computation=computation, init_value=init_value,
                           dimensions=(1,),
                           dtype=dtype)
      _make_reduce_harness(name, nr_operands=nr_operands,
                           computation=computation, init_value=init_value,
                           dimensions=(0, 1),
                           dtype=dtype)

def _make_reduce_window_harness(name,
                                *,
                                shape=(4, 6),
                                base_dilation=(1, 1),
                                computation=lax.add,
                                window_dimensions=(2, 2),
                                window_dilation=(1, 1),
                                init_value=0,
                                window_strides=(1, 1),
                                dtype=np.float32,
                                padding=((0, 0), (0, 0)),
                                requires_xla=False):
  prim_name = f"reduce_window_{computation.__name__}"
  limitations = []
  xla_opts = [True] if requires_xla else [True, False]

  for enable_xla in xla_opts:
    define(
        prim_name,
        f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}_initvalue={init_value}_windowdimensions={window_dimensions}_windowstrides={window_strides}_{padding=!s}_basedilation={base_dilation}_windowdilation={window_dilation}_enablexla={enable_xla}"
        .replace(" ", ""),
        lax.reduce_window,
        [
            RandArg(shape, dtype),
            # Must be static to trigger the picking of the reducers
            StaticArg(np.array(init_value, dtype=dtype)),
            StaticArg(computation),
            StaticArg(window_dimensions),
            StaticArg(window_strides),
            StaticArg(padding),
            StaticArg(base_dilation),
            StaticArg(window_dilation)
        ],
        jax_unimplemented=limitations,
        shape=shape,
        dtype=dtype,
        init_value=np.array(init_value, dtype=dtype),
        computation=computation,
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding,
        base_dilation=base_dilation,
        window_dilation=window_dilation,
        enable_xla=enable_xla)

def requires_xla_for_reduce(name, dtype):
  if name not in ["min", "max", "add"]:
    return True
  if name in ["min", "max"] and dtype in [
      np.bool_, np.uint32, np.uint64, np.complex64, np.complex128
  ]:
    return True
  if name == "min" and dtype in [np.uint8, np.uint16]:
    return True
  if name == "add" and dtype not in [
      dtypes.bfloat16, np.float16, np.float32, np.float64
  ]:
    return True
  return False

# Validate dtypes across all execution paths
# This first harness runs the tests for all dtypes using default values for
# the other parameters (outside of computation and its init_value), through
# several execution paths. Variations of other parameters can thus safely
# skip testing their corresponding default value.
for dtype in jtu.dtypes.all:
  for computation, init_value in [
      (lax.min, _get_min_identity(dtype)),  # path through reduce_window_min
      (lax.max, _get_max_identity(dtype)),  # path through TF reduce_window_max
      (lax.max, 1),  # path through reduce_window
      (lax.add, 0),  # path_through reduce_window_sum
      (lax.add, 1),  # path through reduce_window
      (lax.mul, 0),  # path through reduce_window
      (lax.mul, 1),  # path through reduce_window
      (lax.mul, 2),  # path through reduce_window
  ]:
    if dtype == np.bool_ and computation in [lax.add, lax.mul]:
      continue
    _make_reduce_window_harness(
        "dtypes",
        dtype=dtype,
        computation=computation,
        init_value=init_value,
        requires_xla=requires_xla_for_reduce(computation.__name__, dtype))

# Validate window_dimensions
_make_reduce_window_harness("window_dimensions", window_dimensions=(1, 1))
# Validate window_strides
_make_reduce_window_harness("window_strides", window_strides=(1, 2))
# Validate padding
_make_reduce_window_harness("padding", padding=((1, 2), (0, 3)),
                            requires_xla=True)
# Validate base_dilation
_make_reduce_window_harness("base_dilation", base_dilation=(1, 2),
                            requires_xla=True)
# Validate window_dilation
_make_reduce_window_harness("window_dilation", window_dilation=(1, 2))
# Validate batch and channel dimensions behavior. lax.reduce_window accepts
# inputs that either have or do not have batch and channel dimensions.
# N=batch, DHW=spatial, C=channel.
# Without XLA only supports 1D/2D reductions.
for shape, window_dimensions, requires_xla in [
    ((2,), (2,), False),  # W
    ((2, 1), (2, 1), False),  # WC
    ((1, 2), (1, 2), False),  # NW
    ((1, 2, 1), (1, 2, 1), False),  # NWC
    ((2, 4), (2, 2), False),  # HW
    ((1, 2, 4, 1), (1, 2, 2, 1), False),  # NHWC
    ((2, 4, 3), (2, 2, 2), True),  # DHW
    ((1, 4, 3, 2, 1), (1, 2, 2, 2, 1), True)  # NDHWC
]:
  _make_reduce_window_harness(
      "batch_channel_dims",
      computation=lax.max,
      shape=shape,
      dtype=np.float32,
      init_value=-np.inf,
      base_dilation=tuple([1] * len(shape)),
      window_dilation=tuple([1] * len(shape)),
      padding=tuple([(0, 0)] * len(shape)),
      window_strides=tuple([1] * len(shape)),
      window_dimensions=window_dimensions,
      requires_xla=requires_xla)

for computation, id_value in [(lax.max, _get_max_identity(np.float32)),
                              (lax.min, _get_min_identity(np.float32)),
                              (lax.add, 0.)]:
  _make_reduce_window_harness(
      "same_padding",
      shape=(112, 112),
      init_value=id_value,
      computation=computation,
      window_dimensions=(3, 3),
      window_strides=(2, 2),
      padding="SAME")

# A few additional test cases for manual padding, which is applied when calling
# reduce_window with lax.add, SAME padding and window_dimensions != (1, 1, ...).
for window_dimensions, window_strides in [((2, 2), (1, 1)), ((3, 3), (2, 2)),
                                          ((13, 13), (5, 6))]:
  _make_reduce_window_harness(
      "manual_padding",
      shape=(12, 12),
      init_value=0.,
      computation=lax.add,
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      padding="SAME")

_make_reduce_window_harness(
    "init_value_1d",
    shape=(1, 1600),
    init_value=1.0,
    computation=lax.min,
    window_dimensions=[1, 401],
    window_strides=[1, 160],
    padding="VALID",
    requires_xla=False)

def _make_reducer_harness(prim,
                          name,
                          *,
                          shape=(2, 3),
                          axes=(0,),
                          dtype=np.int32):
  define(
      prim,
      f"{name}_shape={jtu.format_shape_dtype_string(shape, dtype)}",
      lambda arg: prim.bind(arg, axes=axes), [RandArg(shape, dtype)],
      prim=prim,
      shape=shape,
      dtype=dtype,
      axes=axes)


for prim in [
    lax.reduce_sum_p, lax.reduce_prod_p, lax.reduce_max_p, lax.reduce_min_p,
    lax.reduce_or_p, lax.reduce_and_p
]:
  for dtype in {
      lax.reduce_sum_p: set(jtu.dtypes.all) - set(jtu.dtypes.boolean),
      lax.reduce_prod_p: set(jtu.dtypes.all) - set(jtu.dtypes.boolean),
      lax.reduce_max_p: jtu.dtypes.all,
      lax.reduce_min_p: jtu.dtypes.all,
      lax.reduce_or_p: jtu.dtypes.boolean,
      lax.reduce_and_p: jtu.dtypes.boolean
  }[prim]:
    _make_reducer_harness(prim, "dtypes", dtype=dtype)

for dtype in (np.float32, np.float64):
  for shape in ((), (3,)):
    define(
        "random_gamma",
        f"shape={jtu.format_shape_dtype_string(shape, dtype)}",
        jax.jit(lambda x: jax_random.gamma(jax.random.key(42), x)),
        [RandArg(shape, dtype)],
        dtype=dtype)


def wrap_and_split():
  key = jax.random.key(42)
  result = jax.random.split(key, 2)
  return jax.random.key_data(result)

define(
    "random_split",
    "",
    jax.jit(wrap_and_split),
    [],
    dtype=np.uint32)

# A few library functions from jax.random
for dtype in jtu.dtypes.all_floating:
  for shape in ((8,), (5, 4)):
    for axis in range(len(shape)):
      define(
          "random_categorical",
          f"shape={jtu.format_shape_dtype_string(shape, dtype)}_{axis=}",
          lambda x, axis: jax.random.categorical(
            jax.random.key(42), x, axis),
          [RandArg(shape, dtype),
           StaticArg(axis)],
          dtype=dtype,
          axis=axis)

for dtype in jtu.dtypes.all_floating:
  for shape in ((), (5, 4), (32,)):
    define(
        "random_uniform",
        f"shape={jtu.format_shape_dtype_string(shape, dtype)}",
        lambda shape, dtype: jax.random.uniform(
          jax.random.key(42), shape, dtype),
        [StaticArg(shape), StaticArg(dtype)],
        dtype=dtype)

for dtype in jtu.dtypes.all_integer:
  for shape in ((), (5, 4), (32,)):
    maxval = {
        np.uint8: 256,   # Borderline
    }.get(dtype, 5)
    define(
        "random_randint",
        f"shape={jtu.format_shape_dtype_string(shape, dtype)}",
        lambda shape, minval, maxval, dtype: jax.random.randint(
          jax.random.key(42), shape, minval, maxval, dtype),
        [StaticArg(shape),
         StaticArg(-5),  # minval
         StaticArg(maxval),
         StaticArg(dtype)],
        dtype=dtype)

def _make_clamp_harness(name,
                        *,
                        min_shape=(),
                        operand_shape=(2, 3),
                        max_shape=(),
                        dtype=np.float32,
                        min_max=None):
  min_arr, max_arr = (
      min_max if min_max is not None else
      [RandArg(min_shape, dtype),
       RandArg(max_shape, dtype)])
  define(
      lax.clamp_p,
      f"{name}_min={jtu.format_shape_dtype_string(min_arr.shape, min_arr.dtype)}_operand={jtu.format_shape_dtype_string(operand_shape, dtype)}_max={jtu.format_shape_dtype_string(max_arr.shape, max_arr.dtype)}",
      lax.clamp, [min_arr, RandArg(operand_shape, dtype), max_arr],
      min_shape=min_arr.shape,
      operand_shape=operand_shape,
      max_shape=max_arr.shape,
      dtype=dtype,
      jax_unimplemented=[
          Limitation(
              "unimplemented",
              dtypes=[np.bool_, np.complex64, np.complex128])],
  )


for dtype in set(jtu.dtypes.all):
  _make_clamp_harness("dtypes", dtype=dtype)

# Validate broadcasting of min/max arrays
for min_shape, operand_shape, max_shape in [
    ((), (2, 3), (2, 3)),  # no broadcasting for max
    ((2, 3), (2, 3), ()),  # no broadcasting for min
    ((2, 3), (2, 3), (2, 3)),  # no broadcasting
]:
  _make_clamp_harness(
      "broadcasting",
      min_shape=min_shape,
      max_shape=max_shape,
      operand_shape=operand_shape)

# Validate clamping when minval > maxval, and when minval < maxval
for is_ordered, min_arr, max_arr in [
    (False, np.array(4., dtype=np.float32), np.array(1., dtype=np.float32)),
    (True, np.array(1., dtype=np.float32), np.array(4., dtype=np.float32))
]:
  _make_clamp_harness(
      f"order={is_ordered}", min_max=(min_arr, max_arr), dtype=np.float32)


def _make_dot_general_harness(name,
                              *,
                              lhs_shape=(3, 4),
                              rhs_shape=(4, 2),
                              lhs_dtype=np.float32,
                              rhs_dtype=np.float32,
                              precision=None,
                              dimension_numbers=(((1,), (0,)), ((), ())),
                              preferred_element_type=None,
                              enable_xla=True):
  suffix = ""
  if precision is not None:
    suffix += f"_{precision=}"
  if preferred_element_type is not None:
    suffix += f"_preferred={jtu.dtype_str(preferred_element_type)}"

  define(
      lax.dot_general_p,
      f"{name}_lhs={jtu.format_shape_dtype_string(lhs_shape, lhs_dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)}_dimensionnumbers={dimension_numbers}{suffix}_enable_xla={enable_xla}"
      .replace(" ", ""),
      lax.dot_general,
      [
          RandArg(lhs_shape, lhs_dtype),
          RandArg(rhs_shape, rhs_dtype),
          StaticArg(dimension_numbers),
          StaticArg(precision),
          StaticArg(preferred_element_type)
      ],
      dtype=lhs_dtype,
      rhs_dtype=rhs_dtype,
      lhs_shape=lhs_shape,
      rhs_shape=rhs_shape,
      dimension_numbers=dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
      enable_xla=enable_xla,
      jax_unimplemented=[
          Limitation("preferred_element_type must match dtype for floating point",
                     devices="gpu",
                     dtypes=[np.float16, dtypes.bfloat16, np.float32, np.float64, np.complex64, np.complex128],
                     enabled=(preferred_element_type is not None and preferred_element_type != lhs_dtype)),
          Limitation("preferred_element_type must be floating for integer dtype",
                     devices="gpu",
                     dtypes=[np.int8, np.uint8, np.int16, np.uint16,
                             np.int32, np.uint32, np.int64, np.uint64],
                     enabled=(preferred_element_type is not None
                              and preferred_element_type in [
                                np.float16, dtypes.bfloat16, np.float32,
                                np.float64, np.complex64, np.complex128]),
                     skip_run=True),  # skip run because we get internal XLA error
     ])


# There are two execution paths in the conversion of dot_general. The main path
# uses tf.einsum, while special cases use tf.linalg.matmul. For that reason,
# the below tests are designed to perform the same checks on both execution
# paths.
# Validate dtypes and precision
# This first harness runs the tests for all dtypes and precisions using
# default values for all the other parameters. Variations of other parameters
# can thus safely skip testing their corresponding default value.

for dtype in jtu.dtypes.all:
  for precision in [
      None, lax.Precision.DEFAULT, lax.Precision.HIGH, lax.Precision.HIGHEST
  ]:
    for lhs_shape, rhs_shape, dimension_numbers in [
        ((3, 4), (4, 2), (((1,), (0,)), ((), ()))),
        ((1, 3, 4), (1, 4, 3), (((2, 1), (1, 2)), ((0,), (0,)))),
        # Some batch dimensions
        ((7, 3, 4), (7, 4), (((2,), (1,)), ((0,), (0,)))),
    ]:
      _make_dot_general_harness(
          "dtypes_and_precision",
          precision=precision,
          lhs_shape=lhs_shape,
          rhs_shape=rhs_shape,
          dimension_numbers=dimension_numbers,
          lhs_dtype=dtype,
          rhs_dtype=dtype)

# The other tests are only for float32.
# Validate batch dimensions
for lhs_shape, rhs_shape, dimension_numbers in [
    # Unique pattern that can go through tf.linalg.matmul
    ((4, 4, 3, 3, 4), (4, 4, 3, 4, 2), (((4,), (3,)), ((0, 1, 2), (0, 1, 2)))),
    # Main path with out of order batch dimensions
    ((8, 4, 3, 3, 4), (4, 8, 3, 4, 2), (((4, 3), (3, 2)), ((0, 1), (1, 0))))
]:
  _make_dot_general_harness(
      "batch_dimensions",
      lhs_shape=lhs_shape,
      rhs_shape=rhs_shape,
      dimension_numbers=dimension_numbers)

# Validate squeezing behavior for matmul path
for lhs_shape, rhs_shape, dimension_numbers in [
    ((4,), (4, 4), (((0,), (0,)), ((), ()))),  # (1, 4) -> (4,)
    ((4, 4), (4,), (((1,), (0,)), ((), ()))),  # (4, 1) -> (4,)
    ((4,), (4,), (((0,), (0,)), ((), ()))),  # (1, 1) -> ()
]:
  _make_dot_general_harness(
      "squeeze",
      lhs_shape=lhs_shape,
      rhs_shape=rhs_shape,
      dimension_numbers=dimension_numbers)

# Validate preferred element type
# From lax_test.py
preferred_type_combinations = [
  (np.float16, np.float16), (np.float16, np.float32), (np.float16, np.float64),
  (dtypes.bfloat16, dtypes.bfloat16), (dtypes.bfloat16, np.float32),
  (dtypes.bfloat16, np.float64), (np.float32, np.float32),
  (np.float32, np.float64),
  (np.float64, np.float64), (np.int8, np.int8), (np.int8, np.int16),
  (np.int8, np.int32),
  (np.int8, np.int64), (np.int16, np.int16), (np.int16, np.int32),
  (np.int16, np.int64),
  (np.int32, np.int32), (np.int32, np.int64), (np.int64, np.int64),
  (np.complex64, np.complex64), (np.complex64, np.complex128),
  (np.complex128, np.complex128),
  (np.int8, np.float16), (np.int8, dtypes.bfloat16), (np.int8, np.float32),
  (np.int8, np.float64),
  (np.int16, np.float16), (np.int16, dtypes.bfloat16), (np.int16, np.float32),
  (np.int16, np.float64),
  (np.int32, np.float32), (np.int32, np.float64), (np.int64, np.float64)]

for lhs_shape in [(4, 3)]:
  for rhs_shape in [(3, 6)]:
    for dtype, preferred_element_type in preferred_type_combinations:
      _make_dot_general_harness(
          "preferred",
          lhs_dtype=dtype,
          rhs_dtype=dtype,
          lhs_shape=lhs_shape,
          rhs_shape=rhs_shape,
          dimension_numbers=(((len(lhs_shape) - 1,), (0,)), ((), ())),
          preferred_element_type=preferred_element_type)

    # Validate lhs_dtype different than rhs_dtype
    all_types = (jtu.dtypes.all_integer + jtu.dtypes.all_unsigned + jtu.dtypes.all_inexact)
    for i_lhs in range(len(all_types)):
      for i_rhs in range(len(all_types)):
        if i_rhs <= i_lhs: continue
        lhs_dtype = all_types[i_lhs]
        rhs_dtype = all_types[i_rhs]

        for enable_xla in [True, False]:
          _make_dot_general_harness(
              "different_dtypes",
              lhs_dtype=lhs_dtype,
              rhs_dtype=rhs_dtype,
              lhs_shape=lhs_shape,
              rhs_shape=rhs_shape,
              dimension_numbers=(((len(lhs_shape) - 1,), (0,)), ((), ())),
              enable_xla=enable_xla)


def _make_concatenate_harness(name,
                              *,
                              shapes=[(2, 3), (2, 3)],
                              dimension=0,
                              dtype=np.float32):
  shapes_str = "_".join(jtu.format_shape_dtype_string(s, dtype) for s in shapes)
  define(
      lax.concatenate_p,
      f"{name}_shapes={shapes_str}_{dimension=}",
      lambda *args: lax.concatenate_p.bind(*args, dimension=dimension),
      [RandArg(shape, dtype) for shape in shapes],
      shapes=shapes,
      dtype=dtype,
      dimension=dimension)


for dtype in jtu.dtypes.all:
  _make_concatenate_harness("dtypes", dtype=dtype)

# Validate dimension; non-major axis
_make_concatenate_harness("dimension", dimension=1)

# Validate > 2 operands
for shapes in [
    [(2, 3, 4), (3, 3, 4), (4, 3, 4)],  # 3 operands
]:
  _make_concatenate_harness("nb_operands", shapes=shapes)


def _make_conv_harness(name,
                       *,
                       lhs_shape=(2, 3, 9, 10),
                       rhs_shape=(3, 3, 4, 5),
                       dtype=np.float32,
                       window_strides=(1, 1),
                       precision=None,
                       padding=((0, 0), (0, 0)),
                       lhs_dilation=(1, 1),
                       rhs_dilation=(1, 1),
                       feature_group_count=1,
                       dimension_numbers=("NCHW", "OIHW", "NCHW"),
                       batch_group_count=1,
                       preferred_element_type=None,
                       works_without_xla=False):
  enable_xla_cases = [True, False] if works_without_xla else [True]

  for enable_xla in enable_xla_cases:
    define(
        lax.conv_general_dilated_p,
        f"{name}_lhs={jtu.format_shape_dtype_string(lhs_shape, dtype)}_rhs={jtu.format_shape_dtype_string(rhs_shape, dtype)}_windowstrides={window_strides}_{padding=!s}_lhsdilation={lhs_dilation}_rhsdilation={rhs_dilation}_dimensionnumbers={dimension_numbers}_featuregroupcount={feature_group_count}_batchgroupcount={batch_group_count}_{precision=}_preferred={jtu.dtype_str(preferred_element_type)}_enablexla={enable_xla}"
        .replace(" ", ""),
        lax.conv_general_dilated,
        [
            RandArg(lhs_shape, dtype),
            RandArg(rhs_shape, dtype),
            StaticArg(window_strides),
            StaticArg(padding),
            StaticArg(lhs_dilation),
            StaticArg(rhs_dilation),
            StaticArg(dimension_numbers),
            StaticArg(feature_group_count),
            StaticArg(batch_group_count),
            StaticArg(precision),
            StaticArg(preferred_element_type),
        ],
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        dtype=dtype,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision=precision,
        preferred_element_type=preferred_element_type,
        enable_xla=enable_xla,
        jax_unimplemented=[
            # b/183565702 - no integer convolutions for GPU
            Limitation(
                "preferred_element_type not implemented for integers",
                devices="gpu",
                dtypes=(np.int8, np.int16, np.int32, np.int64),
                enabled=(preferred_element_type in [np.int16, np.int32,
                                                    np.int64])),
        ],
    )


# Validate dtypes and precision
for dtype in jtu.dtypes.all_inexact:
  for precision in [
      None, lax.Precision.DEFAULT, lax.Precision.HIGH, lax.Precision.HIGHEST
  ]:
    # This first harness runs the tests for all dtypes and precisions using
    # default values for all the other parameters. Variations of other parameters
    # can thus safely skip testing their corresponding default value.
    _make_conv_harness(
        "dtype_precision",
        dtype=dtype,
        precision=precision)


# Validate preferred_element_type
for dtype, preferred_element_type in preferred_type_combinations:
  works_without_xla = dtype == np.float32 and preferred_element_type == np.float32
  _make_conv_harness(
      "preferred", dtype=dtype,
      preferred_element_type=preferred_element_type,
      works_without_xla=works_without_xla)

# Validate variations of feature_group_count and batch_group_count
for batch_group_count, feature_group_count in [
    (1, 2),  # feature_group_count != 1
    (2, 1),  # batch_group_count != 1
]:
  for lhs_shape, rhs_shape in [
      ((2 * batch_group_count, 3 * feature_group_count, 9, 10),
       (3 * feature_group_count * batch_group_count, 3, 4, 5))
  ]:
    _make_conv_harness(
        "group_counts",
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count)

# --- BEGIN Tests for conv_general_dilated with works_without_xla=True ---

# Validate Conv1D.
_make_conv_harness(
    "conv1d",
    lhs_shape=(2, 3, 10),
    rhs_shape=(3, 3, 5),
    window_strides=(1,),
    padding=((0, 0),),
    lhs_dilation=(1,),
    rhs_dilation=(1,),
    dimension_numbers=("NCH", "OIH", "NCH"),
    works_without_xla=True)


# feature_group_count is supported for enable_xla=False only if we are doing a
# depthwise convolution, i.e.: in_channels == feature_group_count.
# See explanation of depthwise convolution at
# https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
_make_conv_harness(
    "depthwise2d",
    lhs_shape=(2, 3, 9, 9),  # "NCHW": in_channels == 3
    rhs_shape=(12, 1, 3, 3),  # "OIHW": channel_multiplier = 12/3 = 4
    feature_group_count=3,
    works_without_xla=True)

_make_conv_harness(
    "depthwise2d_dilated",
    lhs_shape=(2, 3, 9, 9),  # "NCHW": in_channels == 3
    rhs_shape=(12, 1, 3, 3),  # "OIHW": channel_multiplier = 12/3 = 4
    feature_group_count=3,
    lhs_dilation=(1, 1),
    rhs_dilation=(2, 1),
    works_without_xla=True)

_make_conv_harness(
    "depthwise1d",
    lhs_shape=(2, 3, 9),  # "NCH": in_channels == 3
    rhs_shape=(12, 1, 3),  # "OIH": channel_multiplier = 12/3 = 4
    feature_group_count=3,
    lhs_dilation=(1,),
    rhs_dilation=(1,),
    window_strides=(1, ),
    padding=((0, 0),),
    dimension_numbers=("NCH", "OIH", "NCH"),
    works_without_xla=True)

_make_conv_harness(
    "depthwise1d_dilated",
    lhs_shape=(2, 3, 9),  # "NCH": in_channels == 3
    rhs_shape=(12, 1, 3),  # "OIH": channel_multiplier = 12/3 = 4
    feature_group_count=3,
    lhs_dilation=(1,),
    rhs_dilation=(2,),
    window_strides=(1,),
    padding=((0, 0),),
    dimension_numbers=("NCH", "OIH", "NCH"),
    works_without_xla=True)

# Validate variations of window_strides
for window_strides in [(2, 3)]:
  _make_conv_harness(
      "window_strides",
      window_strides=window_strides,
      works_without_xla=True)

# Validate variations of padding
for padding in [
    ((1, 2), (0, 0)),  # padding only one spatial axis
    ((1, 2), (2, 1))  # padding on both spatial axes
]:
  _make_conv_harness("padding", padding=padding, works_without_xla=True)

# Validate variations of dilations
for lhs_dilation, rhs_dilation in [
     ((1, 1), (2, 2)),  # dilation only on RHS (atrous)
]:
  _make_conv_harness(
      "dilations",
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      works_without_xla=True)

# Simulate a call from lax.conv_transpose.
_make_conv_harness(
    "conv_tranpose2d_valid_padding",
    lhs_shape=(1, 16, 16, 2),
    rhs_shape=(2, 3, 2, 2),
    window_strides=(1, 1),
    lhs_dilation=(2, 2),
    padding=((1, 1), (2, 2)),
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
    works_without_xla=True)

# Simulate a call from lax.conv_transpose.
_make_conv_harness(
    "conv_tranpose1d_valid_padding",
    lhs_shape=(1, 16, 2),
    rhs_shape=(3, 2, 2),
    window_strides=(1,),
    lhs_dilation=(2,),
    rhs_dilation=(1,),
    padding=((2, 2), ),
    dimension_numbers=("NHC", "HIO", "NHC"),
    works_without_xla=True)

_make_conv_harness(
    "conv_tranpose1d_same_padding",
    lhs_shape=(1, 16, 2),
    rhs_shape=(3, 2, 2),
    window_strides=(1,),
    lhs_dilation=(2,),
    rhs_dilation=(1,),
    padding=((2, 1), ),
    dimension_numbers=("NHC", "HIO", "NHC"),
    works_without_xla=True)

_make_conv_harness(
    "conv_tranpose2d_same_padding",
    lhs_shape=(1, 16, 16, 2),
    rhs_shape=(2, 3, 2, 2),
    window_strides=(1, 1),
    lhs_dilation=(2, 2),
    padding=((1, 1), (2, 1)),
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
    works_without_xla=True)

# Validate rhs > lhs.
# One dimension of rhs is bigger than lhs.
_make_conv_harness(
    "rhs_oob",
    lhs_shape=(2, 3, 9, 10),
    rhs_shape=(3, 3, 10, 5),
    works_without_xla=True)

# Effective rhs size is too big after applying rhs_dilation.
_make_conv_harness(
    "rhs_oob_after_dilation",
    lhs_shape=(2, 3, 9, 10),
    rhs_shape=(3, 3, 4, 5),
    rhs_dilation=(2, 3),
    works_without_xla=True)

# Effective rhs is too big after applying input padding.
_make_conv_harness(
    "rhs_oob_after_pading",
    lhs_shape=(1, 3, 2, 2),
    rhs_shape=(64, 3, 7, 7),
    window_strides=(2, 2),
    padding=((3, 3), (3, 3)),
    works_without_xla=True)

# rhs out of bounds with "SAME" padding.
_make_conv_harness(
    "rhs_oob_same_padding",
    lhs_shape=(1, 1, 16, 1),
    padding="SAME",
    rhs_shape=(4, 1, 1, 2),
    dimension_numbers=("NHWC", "HWIO", "NHWC"),  # TF default
    works_without_xla=True)


# Dimension numbers and corresponding permutation
for dimension_numbers, lhs_shape, rhs_shape in [
    (("NHWC", "HWIO", "NHWC"), (2, 9, 10, 3), (4, 5, 3, 3)),  # TF default
    (("NCHW", "HWIO", "NHWC"), (2, 3, 9, 10), (4, 5, 3, 3)),  # custom
]:
  _make_conv_harness(
      "dimension_numbers",
      lhs_shape=lhs_shape,
      rhs_shape=rhs_shape,
      dimension_numbers=dimension_numbers,
      works_without_xla=True)

for padding, lhs_dilation, rhs_dilation in [
    ("VALID", (1, 1), (1, 1)),  # no dilation with "VALID" padding
    ("SAME", (1, 1), (1, 1)),  # no dilation with "SAME" padding
    ("VALID", (1, 1), (1, 2)),  # dilation only on RHS with "VALID" padding
    ("SAME", (1, 1), (1, 2)),  # dilation only on RHS with "SAME" padding
    ([(1, 2), (0, 1)], (1, 1), (1, 2))
]:
  for dimension_numbers, lhs_shape, rhs_shape in [
      (("NHWC", "HWIO", "NHWC"), (1, 28, 28, 1), (3, 3, 1, 16)),  # TF default
      (("NCHW", "HWIO", "NCHW"), (1, 1, 28, 28), (3, 3, 1, 16)),
  ]:
    _make_conv_harness(
        "tf_conversion_path_2d",
        lhs_shape=lhs_shape,
        padding=padding,
        rhs_shape=rhs_shape,
        dimension_numbers=dimension_numbers,
        window_strides=(1, 1),
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        works_without_xla=True)

for padding, lhs_dilation, rhs_dilation in [
    ("VALID", (1,), (1,)),  # no dilation with "VALID" padding
    ("SAME", (1,), (1,)),  # no dilation with "SAME" padding
    ("VALID", (1,), (2,)),  # dilation only on RHS with "VALID" padding
    ("SAME", (1,), (2,)),  # dilation only on RHS with "SAME" padding
]:
  for dimension_numbers, lhs_shape, rhs_shape in [
      (("NWC", "WIO", "NWC"), (1, 28, 1), (3, 1, 16)),  # TF default
  ]:
    _make_conv_harness(
        "tf_conversion_path_1d",
        lhs_shape=lhs_shape,
        padding=padding,
        rhs_shape=rhs_shape,
        dimension_numbers=dimension_numbers,
        window_strides=(1,),
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        works_without_xla=True)

# --- END Tests for conv_general_dilated with works_without_xla=True ---

for lhs_dilation, rhs_dilation in [
    # Note: LHS dilation does work for enable_xla=False, but only if
    # padding=='VALID' (see test above for conv_transpose2d_valid_padding).
    ((2, 2), (1, 1)),  # dilation only on LHS (transposed)
    ((2, 3), (3, 2))   # dilation on both LHS and RHS (transposed & atrous)
]:
  _make_conv_harness(
      "dilations", lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation)


for padding, lhs_dilation, rhs_dilation in [
    ("VALID", (1, 1, 1), (1, 1, 1)),  # no dilation with "VALID" padding
    ("SAME", (1, 1, 1), (1, 1, 1)),  # no dilation with "SAME" padding
    ("VALID", (1, 1, 1), (1, 1,
                          2)),  # dilation only on RHS with "VALID" padding
    ("SAME", (1, 1, 1), (1, 1, 2)),  # dilation only on RHS with "SAME" padding
]:
  for dimension_numbers, lhs_shape, rhs_shape in [
      # TF default
      (("NDHWC", "DHWIO", "NDHWC"), (1, 4, 28, 28, 1), (2, 3, 3, 1, 16)),
  ]:
    _make_conv_harness(
        "tf_conversion_path_3d",
        lhs_shape=lhs_shape,
        padding=padding,
        rhs_shape=rhs_shape,
        dimension_numbers=dimension_numbers,
        window_strides=(1, 1, 1),
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation)

key_types: list[tuple[tuple[int, ...], jax.typing.DTypeLike]]
key_types = [((4,), np.uint32)]
if config.enable_x64.value:
  key_types.append(((2,), np.uint64))

for algorithm in [lax.RandomAlgorithm.RNG_THREE_FRY,
                  lax.RandomAlgorithm.RNG_PHILOX,
                  lax.RandomAlgorithm.RNG_DEFAULT]:
  for dtype in [np.uint32, np.uint64]:
    for shape in [(), (5, 7), (100, 100)]:
      for key_shape, key_dtype in key_types:
        define(
            lax.rng_bit_generator_p,
            f"{key_dtype=}_shape={jtu.format_shape_dtype_string(shape, dtype)}_{algorithm=}",
            lambda key, shape, dtype, algorithm: lax.rng_bit_generator(key, shape, dtype=dtype,
                                                                       algorithm=algorithm),
            [RandArg(key_shape, key_dtype),
             StaticArg(shape), StaticArg(dtype), StaticArg(algorithm)],
            shape=shape,
            dtype=dtype,
            algorithm=algorithm)

def _make_iota_2x32_shape_harness(shape):
  shapestr = ','.join(str(dim) for dim in shape)
  define(
      prng.iota_2x32_shape_p,
      f"shape=({shapestr})",
      lambda shape: prng.iota_2x32_shape_p.bind(shape=shape),
      [StaticArg(shape)],
      dtype=jnp.uint32,
      shape=shape)

for shape in [(3,), (5, 7, 4), (100, 100)]:
  _make_iota_2x32_shape_harness(shape)


for in_dtype in jtu.dtypes.all_floating:
  for out_dtype in jtu.dtypes.all_floating:
    out_iinfo = dtypes.finfo(out_dtype)
    for shape in [(), (5, 7)]:
        define(
            lax.reduce_precision_p,
            f"in={jtu.format_shape_dtype_string(shape, in_dtype)}_out={jtu.format_shape_dtype_string(shape, out_dtype)}",
            lambda x, exp_bits, mant_bits: lax.reduce_precision(x,
                                                                exponent_bits=exp_bits,
                                                                mantissa_bits=mant_bits),
            [RandArg(shape, in_dtype),
             StaticArg(out_iinfo.nexp), StaticArg(out_iinfo.nmant)],
            shape=shape,
            dtype=in_dtype,
            out_dtype=out_dtype)

# approx_top_k
for is_max in [True, False]:
  for dtype in jtu.dtypes.all_floating:
    # There are different lowerings for sizes < 1024 for rank-1 and 128 for higher
    # rank.
    for shape in [(32,), (2048,), (16, 256)]:
      define(
          lax.approx_top_k_p,
          f"large={np.prod(shape) >= 1024}_max={is_max}_op={jtu.format_shape_dtype_string(shape, dtype)}",
          lambda operand, is_max: lax.approx_top_k_p.bind(
              operand, k=4, reduction_dimension=-1, recall_target=0.95,
              is_max_k=is_max,
              reduction_input_size_override=-1,
              aggregate_to_topk=True),
          [RandArg(shape, dtype), StaticArg(is_max)],
          dtype=dtype,
          is_max=is_max,
          shape=shape)
