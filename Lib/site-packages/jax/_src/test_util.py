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

# pyformat: disable
from __future__ import annotations

import collections
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import datetime
import functools
from functools import partial
import inspect
import logging
import math
import os
import platform
import re
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any, TextIO
import unittest
import zlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import deprecations
from jax._src import dispatch
from jax._src import dtypes as _dtypes
from jax._src import lib as _jaxlib
from jax._src import monitoring
from jax._src import test_warning_util
from jax._src import xla_bridge
from jax._src import util
from jax._src import mesh as mesh_lib
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm
from jax._src.interpreters import mlir
from jax._src.numpy.util import promote_dtypes, promote_dtypes_inexact
from jax._src.public_test_util import (  # noqa: F401
    _assert_numpy_allclose, _check_dtypes_match, _default_tolerance, _dtype, check_close, check_grads,
    check_jvp, check_vjp, default_gradient_tolerance, default_tolerance, rand_like, tolerance)
from jax._src.util import unzip2
from jax.experimental.compilation_cache import compilation_cache
from jax.tree_util import tree_all, tree_flatten, tree_map, tree_unflatten
import numpy as np
import numpy.random as npr


# This submodule includes private test utilities that are not exported to
# jax.test_util. Functionality appearing here is for internal use only, and
# may be changed or removed at any time and without any deprecation cycle.

_TEST_DUT = config.string_flag(
    'jax_test_dut', '',
    help=
    'Describes the device under test in case special consideration is required.'
)

NUM_GENERATED_CASES = config.int_flag(
  'jax_num_generated_cases',
  int(os.getenv('JAX_NUM_GENERATED_CASES', '10')),
  help='Number of generated cases to test')

_MAX_CASES_SAMPLING_RETRIES = config.int_flag(
  'max_cases_sampling_retries',
  int(os.getenv('JAX_MAX_CASES_SAMPLING_RETRIES', '100')),
  'Number of times a failed test sample should be retried. '
  'When an unseen case cannot be generated in this many trials, the '
  'sampling process is terminated.'
)

_SKIP_SLOW_TESTS = config.bool_flag(
    'jax_skip_slow_tests',
    config.bool_env('JAX_SKIP_SLOW_TESTS', False),
    help='Skip tests marked as slow (> 5 sec).'
)

_TEST_TARGETS = config.string_flag(
  'test_targets', os.getenv('JAX_TEST_TARGETS', ''),
  'Regular expression specifying which tests to run, called via re.search on '
  'the test name. If empty or unspecified, run all tests.'
)
_EXCLUDE_TEST_TARGETS = config.string_flag(
  'exclude_test_targets', os.getenv('JAX_EXCLUDE_TEST_TARGETS', ''),
  'Regular expression specifying which tests NOT to run, called via re.search '
  'on the test name. If empty or unspecified, run all tests.'
)
TEST_WITH_PERSISTENT_COMPILATION_CACHE = config.bool_flag(
    'jax_test_with_persistent_compilation_cache',
    config.bool_env('JAX_TEST_WITH_PERSISTENT_COMPILATION_CACHE', False),
    help='If enabled, the persistent compilation cache will be enabled for all '
    'test cases. This can be used to increase compilation cache coverage.')

HYPOTHESIS_PROFILE = config.string_flag(
    'hypothesis_profile',
    os.getenv('JAX_HYPOTHESIS_PROFILE', 'deterministic'),
    help=('Select the hypothesis profile to use for testing. Available values: '
          'deterministic, interactive'),
)

TEST_NUM_THREADS = config.int_flag(
    'jax_test_num_threads', int(os.getenv('JAX_TEST_NUM_THREADS', '0')),
    help='Number of threads to use for running tests. 0 means run everything '
    'in the main thread. Using > 1 thread is experimental.'
)

# We sanitize test names to ensure they work with "unitttest -k" and
# "pytest -k" test filtering. pytest accepts '[' and ']' but unittest -k
# does not. We replace sequences of problematic characters with a single '_'.
kSanitizeNameRE = re.compile(r"[ \"'\[\](){}<>=,._]+")
def sanitize_test_name(s: str) -> str:
  return kSanitizeNameRE.sub("_", s)

def num_float_bits(dtype):
  return _dtypes.finfo(_dtypes.canonicalize_dtype(dtype)).bits

def to_default_dtype(arr):
  """Convert a value to an array with JAX's default dtype.

  This is generally used for type conversions of values returned by numpy functions,
  to make their dtypes take into account the state of the ``jax_enable_x64`` and
  ``jax_default_dtype_bits`` flags.
  """
  arr = np.asarray(arr)
  dtype = _dtypes._default_types.get(arr.dtype.kind)
  return arr.astype(_dtypes.canonicalize_dtype(dtype)) if dtype else arr

def with_jax_dtype_defaults(func, use_defaults=True):
  """Return a version of a function with outputs that match JAX's default dtypes.

  This is generally used to wrap numpy functions within tests, in order to make
  their default output dtypes match those of corresponding JAX functions, taking
  into account the state of the ``jax_enable_x64`` and ``jax_default_dtype_bits``
  flags.

  Args:
    use_defaults : whether to convert any given output to the default dtype. May be
      a single boolean, in which case it specifies the conversion for all outputs,
      or may be a pytree with the same structure as the function output.
  """
  @functools.wraps(func)
  def wrapped(*args, **kwargs):
    result = func(*args, **kwargs)
    if isinstance(use_defaults, bool):
      return tree_map(to_default_dtype, result) if use_defaults else result
    else:
      f = lambda arr, use_default: to_default_dtype(arr) if use_default else arr
      return tree_map(f, result, use_defaults)
  return wrapped

def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True

def _normalize_tolerance(tol):
  tol = tol or 0
  if isinstance(tol, dict):
    return {np.dtype(k): v for k, v in tol.items()}
  else:
    return dict.fromkeys(_default_tolerance, tol)

def join_tolerance(tol1, tol2):
  tol1 = _normalize_tolerance(tol1)
  tol2 = _normalize_tolerance(tol2)
  out = tol1
  for k, v in tol2.items():
    out[k] = max(v, tol1.get(k, 0))
  return out


def check_eq(xs, ys, err_msg=''):
  assert_close = partial(_assert_numpy_allclose, err_msg=err_msg)
  tree_all(tree_map(assert_close, xs, ys))


@contextmanager
def _capture_output(fp: TextIO) -> Generator[Callable[[], str], None, None]:
  """Context manager to capture all output written to a given file object.

  Unlike ``contextlib.redirect_stdout``, this context manager works for
  any file object and also for both pure Python and native code.

  Example::

    with capture_output(sys.stdout) as get_output:
      print(42)
    print("Captured": get_output())

  Yields:
    A function returning the captured output. The function must be called
    *after* the context is no longer active.
  """
  # ``None`` means nothing has not been captured yet.
  captured = None

  def get_output() -> str:
    if captured is None:
      raise ValueError("get_output() called while the context is active.")
    return captured

  with tempfile.NamedTemporaryFile(mode="w+", encoding='utf-8') as f:
    original_fd = os.dup(fp.fileno())
    os.dup2(f.fileno(), fp.fileno())
    try:
      yield get_output
    finally:
      # Python also has its own buffers, make sure everything is flushed.
      fp.flush()
      os.fsync(fp.fileno())
      f.seek(0)
      captured = f.read()
      os.dup2(original_fd, fp.fileno())


capture_stdout = partial(_capture_output, sys.stdout)
capture_stderr = partial(_capture_output, sys.stderr)


class EventThreadLocalState(threading.local):
  def __init__(self):
    self.counts = {}  # Mapping from string name to count.
    self.nested_device_put_count = 0  # Number of recursive calls to device_put

    # Per-function counts
    self.infer_params_fun_counts = None
    self.lower_jaxpr_to_fun_counts = None

thread_local_state = EventThreadLocalState()


def event_listener(name, *args):
  counts = thread_local_state.counts
  counts[name] = counts.get(name, 0) + 1

  # device_put handlers might call `dispatch.device_put` (e.g. on an
  # underlying payload or several). We only want to count these
  # recursive puts once, so we skip counting more than the outermost
  # one in such a call stack.
  if name == "batched_device_put_start":
    if thread_local_state.nested_device_put_count == 0:
      counts["batched_device_put"] = counts.get("batched_device_put", 0) + 1
    thread_local_state.nested_device_put_count += 1
  elif name == "batched_device_put_end":
    thread_local_state.nested_device_put_count -= 1

  elif name == "pjit._infer_params_impl":
    # For infer_params, we collect per-function data, but only while a context
    # manager is active.
    infer_counts = thread_local_state.infer_params_fun_counts
    if infer_counts is not None:
      (fun,) = args
      infer_counts[fun] += 1
  elif name == "lower_jaxpr_to_fun":
    # For infer_params, we collect per-function data, but only while a context
    # manager is active.
    lower_counts = thread_local_state.lower_jaxpr_to_fun_counts
    if lower_counts is not None:
      (fun,) = args
      lower_counts[fun] += 1


util.test_event_listener = event_listener


def count_events(event):
  "Returns a context-manager that yields a function that counts a test event."
  @contextmanager
  def count_event():
    before = thread_local_state.counts.get(event, 0)
    yield lambda: thread_local_state.counts.get(event, 0) - before
  return count_event

count_device_put = count_events("batched_device_put")
count_device_put_fast_path_hit = count_events("batched_copy_array")
count_pjit_cpp_cache_miss = count_events("pjit_lower")
count_jit_tracing_cache_miss = count_events("create_pjit_jaxpr")
count_aot_jit_cpp_cache_miss = count_events("stages_compiled_call")
count_jit_and_pmap_lowerings = count_events("lower_jaxpr_to_module")
count_jit_compilation_cache_miss = count_events("pxla_cached_compilation")
count_jax_array_shard_arg_calls = count_events("_array_shard_arg")


@contextmanager
def count_primitive_compiles():
  dispatch.xla_primitive_callable.cache_clear()

  count = [-1]
  try:
    yield lambda: count[0]
  finally:
    count[0] = dispatch.xla_primitive_callable.cache_info().misses

@contextmanager
def count_jit_infer_params_cache_miss():
  assert thread_local_state.infer_params_fun_counts is None
  counts = collections.Counter()
  thread_local_state.infer_params_fun_counts = counts
  try:
    yield counts
  finally:
    thread_local_state.infer_params_fun_counts = None

@contextmanager
def count_subjaxpr_to_hlo_conversion(fun_name):
  assert thread_local_state.lower_jaxpr_to_fun_counts is None
  counts = collections.Counter()
  thread_local_state.lower_jaxpr_to_fun_counts = counts
  try:
    yield lambda: counts[fun_name]
  finally:
    thread_local_state.lower_jaxpr_to_fun_counts = None



@contextmanager
def assert_num_jit_and_pmap_compilations(times):
  with count_jit_and_pmap_lowerings() as count:
    yield
  if count() != times:
    raise AssertionError(f"Expected exactly {times} XLA compilations, "
                         f"but executed {count()}")


def jaxlib_version() -> tuple[int, ...]:
  return _jaxlib.version


def device_under_test():
  return _TEST_DUT.value or xla_bridge.get_backend().platform

def supported_dtypes():
  if device_under_test() == "tpu":
    types = {np.bool_, np.int8, np.int16, np.int32, np.uint8, np.uint16,
             np.uint32, _dtypes.bfloat16, np.float16, np.float32, np.complex64,
             _dtypes.float8_e4m3fn, _dtypes.float8_e4m3b11fnuz,
             _dtypes.float8_e5m2}
  elif device_under_test() == "gpu":
    types = {np.bool_, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64,
             _dtypes.bfloat16, np.float16, np.float32, np.float64,
             np.complex64, np.complex128, _dtypes.float8_e4m3fn,
             _dtypes.float8_e5m2}
  elif device_under_test() == "METAL":
    types = {np.int32, np.uint32, np.float32}
  else:
    types = {np.bool_, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64,
             _dtypes.bfloat16, np.float16, np.float32, np.float64,
             np.complex64, np.complex128}
  if not config.enable_x64.value:
    types -= {np.uint64, np.int64, np.float64, np.complex128}
  return types

def is_device_rocm():
  return 'rocm' in xla_bridge.get_backend().platform_version

def is_device_cuda():
  return 'cuda' in xla_bridge.get_backend().platform_version

def is_cloud_tpu():
  return running_in_cloud_tpu_vm

# Returns True if it is not cloud TPU. If it is cloud TPU, returns True if it is
# built at least `date``.
# TODO(b/327203806): after libtpu adds a XLA version and the oldest support
# libtpu contains the XLA version, remove using built time to skip tests.
def if_cloud_tpu_at_least(year: int, month: int, day: int):
  date = datetime.date(year, month, day)
  if not is_cloud_tpu():
    return True
  # The format of Cloud TPU platform_version is like:
  # PJRT C API
  # TFRT TPU v2
  # Built on Oct 30 2023 03:04:42 (1698660263) cl/577737722
  platform_version = xla_bridge.get_backend().platform_version.split('\n')[-1]
  results = re.findall(r'\(.*?\)', platform_version)
  if len(results) != 1:
    return True
  build_date = date.fromtimestamp(int(results[0][1:-1]))
  return build_date >= date

def pjrt_c_api_version_at_least(major_version: int, minor_version: int):
  pjrt_c_api_versions = xla_bridge.backend_pjrt_c_api_version()
  if pjrt_c_api_versions is None:
    return True
  return pjrt_c_api_versions >= (major_version, minor_version)

def get_tpu_version() -> int:
  if device_under_test() != "tpu":
    raise ValueError("Device is not TPU")
  kind = jax.devices()[0].device_kind
  if kind.endswith(' lite'):
    kind = kind[:-len(' lite')]
  assert kind[:-1] == "TPU v", kind
  return int(kind[-1])

def is_device_tpu_at_least(version: int) -> bool:
  if device_under_test() != "tpu":
    return False
  return get_tpu_version() >= version

def is_device_tpu(version: int | None = None, variant: str = "") -> bool:
  if device_under_test() != "tpu":
    return False
  if version is None:
    return True
  device_kind = jax.devices()[0].device_kind
  expected_version = f"v{version}{variant}"
  # Special case v5e until the name is updated in device_kind
  if expected_version == "v5e":
    return "v5 lite" in device_kind
  elif expected_version == "v6e":
    return "v6 lite" in device_kind
  return expected_version in device_kind

def is_cuda_compute_capability_at_least(capability: str) -> bool:
  if not is_device_cuda():
    return False
  d, *_ = jax.local_devices(backend="gpu")
  target = tuple(int(x) for x in capability.split("."))
  current = tuple(int(x) for x in d.compute_capability.split("."))
  return current >= target

def is_cuda_compute_capability_equal(capability: str) -> bool:
  if not is_device_cuda():
    return False
  d, *_ = jax.local_devices(backend="gpu")
  target = tuple(int(x) for x in capability.split("."))
  current = tuple(int(x) for x in d.compute_capability.split("."))
  return current == target

def _get_device_tags():
  """returns a set of tags defined for the device under test"""
  if is_device_rocm():
    device_tags = {device_under_test(), "rocm"}
  elif is_device_cuda():
    device_tags = {device_under_test(), "cuda"}
  elif device_under_test() == "METAL":
    device_tags = {device_under_test(), "gpu"}
  else:
    device_tags = {device_under_test()}
  return device_tags

def test_device_matches(device_types: Iterable[str]) -> bool:
  assert not isinstance(
      device_types, str
  ), 'device_types should be a list of strings'
  tags = _get_device_tags()
  for device_type in device_types:
    assert isinstance(device_type, str), device_type
    if device_type in tags:
      return True
  return False

test_device_matches.__test__ = False  # This isn't a test case, pytest.

def _device_filter(predicate):
  def skip(test_method):
    @functools.wraps(test_method)
    def test_method_wrapper(self, *args, **kwargs):
      device_tags = _get_device_tags()
      if not predicate():
        test_name = getattr(test_method, '__name__', '[unknown test]')
        raise unittest.SkipTest(
          f"{test_name} not supported on device with tags {device_tags}.")
      return test_method(self, *args, **kwargs)
    return test_method_wrapper
  return skip

def skip_on_devices(*disabled_devices):
  """A decorator for test methods to skip the test on certain devices."""
  return _device_filter(lambda: not test_device_matches(disabled_devices))

def run_on_devices(*enabled_devices):
  """A decorator for test methods to run the test only on certain devices."""
  return _device_filter(lambda: test_device_matches(enabled_devices))

def device_supports_buffer_donation():
  """A decorator for test methods to run the test only on devices that support
  buffer donation."""
  return _device_filter(
      lambda: test_device_matches(mlir._platforms_with_donation)
  )


def request_cpu_devices(nr_devices: int):
  """Requests at least `nr_devices` CPU devices.

  request_cpu_devices should be called at the top-level of a test module before
  main() runs.

  It is not guaranteed that the number of CPU devices will be exactly
  `nr_devices`: it may be more or less, depending on how exactly the test is
  invoked. Test cases that require a specific number of devices should skip
  themselves if that number is not met.
  """
  if xla_bridge.NUM_CPU_DEVICES.value < nr_devices:
    xla_bridge.get_backend.cache_clear()
    config.update("jax_num_cpu_devices", nr_devices)


def skip_on_flag(flag_name, skip_value):
  """A decorator for test methods to skip the test when flags are set."""
  def skip(test_method):        # pylint: disable=missing-docstring
    @functools.wraps(test_method)
    def test_method_wrapper(self, *args, **kwargs):
      flag_value = config._read(flag_name)
      if flag_value == skip_value:
        test_name = getattr(test_method, '__name__', '[unknown test]')
        raise unittest.SkipTest(
          f"{test_name} not supported when FLAGS.{flag_name} is {flag_value}")
      return test_method(self, *args, **kwargs)
    return test_method_wrapper
  return skip


def pytest_mark_if_available(marker: str):
  """A decorator for test classes or methods to pytest.mark if installed."""
  def wrap(func_or_class):
    try:
      import pytest
    except ImportError:
      return func_or_class
    return getattr(pytest.mark, marker)(func_or_class)
  return wrap


def is_running_under_pytest():
  return "pytest" in sys.modules


def skip_under_pytest(reason: str):
  """A decorator for test methods to skip the test when run under pytest."""
  reason = "Running under pytest: " + reason
  def skip(test_method):
    return unittest.skipIf(is_running_under_pytest(), reason)(test_method)
  return skip


def format_test_name_suffix(opname, shapes, dtypes):
  arg_descriptions = (format_shape_dtype_string(shape, dtype)
                      for shape, dtype in zip(shapes, dtypes))
  return '{}_{}'.format(opname.capitalize(), '_'.join(arg_descriptions))


# We use special symbols, represented as singleton objects, to distinguish
# between NumPy scalars, Python scalars, and 0-D arrays.
class ScalarShape:
  def __len__(self): return 0
  def __getitem__(self, i): raise IndexError(f"index {i} out of range.")
class _NumpyScalar(ScalarShape): pass
class _PythonScalar(ScalarShape): pass
NUMPY_SCALAR_SHAPE = _NumpyScalar()
PYTHON_SCALAR_SHAPE = _PythonScalar()


# Some shape combinations don't make sense.
def is_valid_shape(shape, dtype):
  if shape == PYTHON_SCALAR_SHAPE:
    return dtype == np.dtype(type(np.array(0, dtype=dtype).item()))
  return True


def _dims_of_shape(shape):
  """Converts `shape` to a tuple of dimensions."""
  if type(shape) in (list, tuple):
    return shape
  elif isinstance(shape, ScalarShape):
    return ()
  elif np.ndim(shape) == 0:
    return (shape,)
  else:
    raise TypeError(type(shape))


def _cast_to_shape(value, shape, dtype):
  """Casts `value` to the correct Python type for `shape` and `dtype`."""
  if shape is NUMPY_SCALAR_SHAPE:
    # explicitly cast to NumPy scalar in case `value` is a Python scalar.
    return np.dtype(dtype).type(value)
  elif shape is PYTHON_SCALAR_SHAPE:
    # explicitly cast to Python scalar via https://stackoverflow.com/a/11389998
    return np.asarray(value).item()
  elif type(shape) in (list, tuple):
    assert np.shape(value) == tuple(shape)
    return value
  elif np.ndim(shape) == 0:
    assert np.shape(value) == (shape,)
    return value
  else:
    raise TypeError(type(shape))


def dtype_str(dtype):
  return np.dtype(dtype).name


def format_shape_dtype_string(shape, dtype):
  if isinstance(shape, np.ndarray):
    return f'{dtype_str(dtype)}[{shape}]'
  elif isinstance(shape, list):
    shape = tuple(shape)
  return _format_shape_dtype_string(shape, dtype)

@functools.lru_cache(maxsize=64)
def _format_shape_dtype_string(shape, dtype):
  if shape is NUMPY_SCALAR_SHAPE:
    return dtype_str(dtype)
  elif shape is PYTHON_SCALAR_SHAPE:
    return 'py' + dtype_str(dtype)
  elif type(shape) is tuple:
    shapestr = ','.join(str(dim) for dim in shape)
    return f'{dtype_str(dtype)}[{shapestr}]'
  elif type(shape) is int:
    return f'{dtype_str(dtype)}[{shape},]'
  else:
    raise TypeError(type(shape))


def _rand_dtype(rand, shape, dtype, scale=1., post=lambda x: x):
  """Produce random values given shape, dtype, scale, and post-processor.

  Args:
    rand: a function for producing random values of a given shape, e.g. a
      bound version of either np.RandomState.randn or np.RandomState.rand.
    shape: a shape value as a tuple of positive integers.
    dtype: a numpy dtype.
    scale: optional, a multiplicative scale for the random values (default 1).
    post: optional, a callable for post-processing the random values (default
      identity).

  Returns:
    An ndarray of the given shape and dtype using random values based on a call
    to rand but scaled, converted to the appropriate dtype, and post-processed.
  """
  if _dtypes.issubdtype(dtype, np.unsignedinteger):
    r = lambda: np.asarray(scale * abs(rand(*_dims_of_shape(shape)))).astype(dtype)
  else:
    r = lambda: np.asarray(scale * rand(*_dims_of_shape(shape))).astype(dtype)
  if _dtypes.issubdtype(dtype, np.complexfloating):
    vals = r() + 1.0j * r()
  else:
    vals = r()
  return _cast_to_shape(np.asarray(post(vals), dtype), shape, dtype)


def rand_fullrange(rng, standardize_nans=False):
  """Random numbers that span the full range of available bits."""
  def gen(shape, dtype, post=lambda x: x):
    dtype = np.dtype(dtype)
    size = dtype.itemsize * math.prod(_dims_of_shape(shape))
    vals = rng.randint(0, np.iinfo(np.uint8).max, size=size, dtype=np.uint8)
    vals = post(vals).view(dtype)
    if shape is PYTHON_SCALAR_SHAPE:
      # Sampling from the full range of the largest available uint type
      # leads to overflows in this case; sample from signed ints instead.
      if dtype == np.uint64:
        vals = vals.astype(np.int64)
      elif dtype == np.uint32 and not config.enable_x64.value:
        vals = vals.astype(np.int32)
    vals = vals.reshape(shape)
    # Non-standard NaNs cause errors in numpy equality assertions.
    if standardize_nans and np.issubdtype(dtype, np.floating):
      vals[np.isnan(vals)] = np.nan
    return _cast_to_shape(vals, shape, dtype)
  return gen


def rand_default(rng, scale=3):
  return partial(_rand_dtype, rng.randn, scale=scale)


def rand_nonzero(rng):
  post = lambda x: np.where(x == 0, np.array(1, dtype=x.dtype), x)
  return partial(_rand_dtype, rng.randn, scale=3, post=post)


def rand_positive(rng):
  post = lambda x: x + 1
  return partial(_rand_dtype, rng.rand, scale=2, post=post)


def rand_small(rng):
  return partial(_rand_dtype, rng.randn, scale=1e-3)


def rand_not_small(rng, offset=10.):
  post = lambda x: x + np.where(x > 0, offset, -offset)
  return partial(_rand_dtype, rng.randn, scale=3., post=post)


def rand_small_positive(rng):
  return partial(_rand_dtype, rng.rand, scale=2e-5)

def rand_uniform(rng, low=0.0, high=1.0):
  assert low < high
  post = lambda x: x * (high - low) + low
  return partial(_rand_dtype, rng.rand, post=post)


def rand_some_equal(rng):

  def post(x):
    x_ravel = x.ravel()
    if len(x_ravel) == 0:
      return x
    flips = rng.rand(*np.shape(x)) < 0.5
    return np.where(flips, x_ravel[0], x)

  return partial(_rand_dtype, rng.randn, scale=100., post=post)


def rand_some_inf(rng):
  """Return a random sampler that produces infinities in floating types."""
  base_rand = rand_default(rng)

  # TODO: Complex numbers are not correctly tested
  # If blocks should be switched in order, and relevant tests should be fixed
  def rand(shape, dtype):
    """The random sampler function."""
    if not _dtypes.issubdtype(dtype, np.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    if _dtypes.issubdtype(dtype, np.complexfloating):
      base_dtype = np.real(np.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             np.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = np.where(posinf_flips, np.array(np.inf, dtype=dtype), vals)
    vals = np.where(neginf_flips, np.array(-np.inf, dtype=dtype), vals)

    return _cast_to_shape(np.asarray(vals, dtype=dtype), shape, dtype)

  return rand

def rand_some_nan(rng):
  """Return a random sampler that produces nans in floating types."""
  base_rand = rand_default(rng)

  def rand(shape, dtype):
    """The random sampler function."""
    if _dtypes.issubdtype(dtype, np.complexfloating):
      base_dtype = np.real(np.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             np.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    if not _dtypes.issubdtype(dtype, np.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    dims = _dims_of_shape(shape)
    r = rng.rand(*dims)
    nan_flips = r < 0.1
    neg_nan_flips = r < 0.05

    vals = base_rand(shape, dtype)
    vals = np.where(nan_flips, np.array(np.nan, dtype=dtype), vals)
    vals = np.where(neg_nan_flips, np.array(-np.nan, dtype=dtype), vals)

    return _cast_to_shape(np.asarray(vals, dtype=dtype), shape, dtype)

  return rand

def rand_some_inf_and_nan(rng):
  """Return a random sampler that produces infinities in floating types."""
  base_rand = rand_default(rng)

  # TODO: Complex numbers are not correctly tested
  # If blocks should be switched in order, and relevant tests should be fixed
  def rand(shape, dtype):
    """The random sampler function."""
    if not _dtypes.issubdtype(dtype, np.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    if _dtypes.issubdtype(dtype, np.complexfloating):
      base_dtype = np.real(np.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             np.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1
    nan_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = np.where(posinf_flips, np.array(np.inf, dtype=dtype), vals)
    vals = np.where(neginf_flips, np.array(-np.inf, dtype=dtype), vals)
    vals = np.where(nan_flips, np.array(np.nan, dtype=dtype), vals)

    return _cast_to_shape(np.asarray(vals, dtype=dtype), shape, dtype)

  return rand

# TODO(mattjj): doesn't handle complex types
def rand_some_zero(rng):
  """Return a random sampler that produces some zeros."""
  base_rand = rand_default(rng)

  def rand(shape, dtype):
    """The random sampler function."""
    dims = _dims_of_shape(shape)
    zeros = rng.rand(*dims) < 0.5

    vals = base_rand(shape, dtype)
    vals = np.where(zeros, np.array(0, dtype=dtype), vals)

    return _cast_to_shape(np.asarray(vals, dtype=dtype), shape, dtype)

  return rand


def rand_int(rng, low=0, high=None):
  def fn(shape, dtype):
    nonlocal high
    gen_dtype = dtype if np.issubdtype(dtype, np.integer) else np.int64
    if low == 0 and high is None:
      if np.issubdtype(dtype, np.integer):
        high = np.iinfo(dtype).max
      else:
        raise ValueError("rand_int requires an explicit `high` value for "
                         "non-integer types.")
    return rng.randint(low, high=high, size=shape,
                       dtype=gen_dtype).astype(dtype)
  return fn

def rand_unique_int(rng, high=None):
  def fn(shape, dtype):
    return rng.choice(np.arange(high or math.prod(shape), dtype=dtype),
                      size=shape, replace=False)
  return fn

def rand_indices_unique_along_axis(rng):
  """Sample an array of given shape containing indices up to dim (exclusive),
  such that the indices are unique along the given axis.
  Optionally, convert some of the resulting indices to negative indices."""
  def fn(dim, shape, axis, allow_negative=True):
    batch_size = math.prod(shape[:axis] + shape[axis:][1:])
    idx = [
      rng.choice(dim, size=shape[axis], replace=False)
      for _ in range(batch_size)
    ]
    idx = np.array(idx).reshape(batch_size, shape[axis])
    idx = idx.reshape(shape[:axis] + shape[axis:][1:] + (shape[axis],))
    idx = np.moveaxis(idx, -1, axis)

    # assert that indices are unique along the given axis
    count = partial(np.bincount, minlength=dim)
    assert (np.apply_along_axis(count, axis, idx) <= 1).all()

    if allow_negative:
      mask = rng.choice([False, True], idx.shape)
      idx[mask] -= dim
    return idx

  return fn

def rand_bool(rng):
  def generator(shape, dtype):
    return _cast_to_shape(
      np.asarray(rng.rand(*_dims_of_shape(shape)) < 0.5, dtype=dtype),
      shape, dtype)
  return generator

def check_raises(thunk, err_type, msg):
  try:
    thunk()
    assert False
  except err_type as e:
    assert str(e).startswith(msg), f"\n{e}\n\n{msg}\n"

def check_raises_regexp(thunk, err_type, pattern):
  try:
    thunk()
    assert False
  except err_type as e:
    assert re.match(pattern, str(e)), f"{e}\n\n{pattern}\n"


def iter_eqns(jaxpr):
  # TODO(necula): why doesn't this search in params?
  yield from jaxpr.eqns
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from iter_eqns(subjaxpr)

def assert_dot_precision(expected_precision, fun, *args):
  jaxpr = api.make_jaxpr(fun)(*args)
  precisions = [eqn.params['precision'] for eqn in iter_eqns(jaxpr.jaxpr)
                if eqn.primitive == lax.dot_general_p]
  for precision in precisions:
    msg = f"Unexpected precision: {expected_precision} != {precision}"
    if isinstance(precision, tuple):
      assert precision[0] == expected_precision, msg
      assert precision[1] == expected_precision, msg
    else:
      assert precision == expected_precision, msg

def assert_dot_preferred_element_type(expected, fun, *args, **kwargs):
  jaxpr = api.make_jaxpr(partial(fun, **kwargs))(*args)
  pref_eltypes = [eqn.params['preferred_element_type'] for eqn in iter_eqns(jaxpr.jaxpr)
                   if eqn.primitive == lax.dot_general_p]
  for pref_eltype in pref_eltypes:
    msg = f"Unexpected preferred_element_type: {expected} != {pref_eltype}"
    assert expected == pref_eltype, msg

def cases_from_gens(*gens):
  sizes = [1, 3, 10]
  cases_per_size = int(NUM_GENERATED_CASES.value / len(sizes)) + 1
  for size in sizes:
    for i in range(cases_per_size):
      yield (f'_{size}_{i}',) + tuple(gen(size) for gen in gens)

def named_cases_from_sampler(gen):
  seen = set()
  retries = 0
  rng = npr.RandomState(42)
  def choose_one(x):
    if not isinstance(x, (list, tuple)):
      x = list(x)
    return [x[rng.randint(len(x))]]
  while (len(seen) < NUM_GENERATED_CASES.value and
         retries < _MAX_CASES_SAMPLING_RETRIES.value):
    retries += 1
    cases = list(gen(choose_one))
    if not cases:
      continue
    if len(cases) > 1:
      raise RuntimeError("Generator is expected to only return a single case when sampling")
    case = cases[0]
    if case["testcase_name"] in seen:
      continue
    retries = 0
    seen.add(case["testcase_name"])
    yield case


# Random sampling for every parameterized test is expensive. Do it once and
# cache the result.
@functools.cache
def _choice(n, m):
  rng = np.random.RandomState(42)
  return rng.choice(n, size=m, replace=False)

def sample_product_testcases(*args, **kw):
  """Non-decorator form of sample_product."""
  args = [list(arg) for arg in args]
  kw = [(k, list(v)) for k, v in kw.items()]
  n = math.prod(len(a) for a in args) * math.prod(len(v) for _, v in kw)
  testcases = []
  for i in _choice(n, min(n, NUM_GENERATED_CASES.value)):
    testcase = {}
    for a in args:
      testcase.update(a[i % len(a)])
      i //= len(a)
    for k, v in kw:
      testcase[k] = v[i % len(v)]
      i //= len(v)
    testcases.append(testcase)
  return testcases

def sample_product(*args, **kw):
  """Decorator that samples from a cartesian product of test cases.

  Similar to absltest.parameterized.product(), except that it samples from the
  cartesian product rather than returning the whole thing.

  Arguments:
    *args: each positional argument is a list of dictionaries. The entries
      in a dictionary correspond to name=value argument pairs; one dictionary
      will be chosen for each test case. This allows multiple parameters to be
      correlated.
    **kw: each keyword argument is a list of values. One value will be chosen
      for each test case.
  """
  return parameterized.parameters(*sample_product_testcases(*args, **kw))

# We use a reader-writer lock to protect test execution. Tests that may run in
# parallel acquire a read lock; tests that are not thread-safe acquire a write
# lock.
if hasattr(util, 'Mutex'):
  _test_rwlock = util.Mutex()

  def _run_one_test(test: unittest.TestCase, result: ThreadSafeTestResult):
    if getattr(test.__class__, "thread_hostile", False):
      _test_rwlock.writer_lock()
      try:
        test(result)  # type: ignore
      finally:
        _test_rwlock.writer_unlock()
    else:
      _test_rwlock.reader_lock()
      try:
        test(result)  # type: ignore
      finally:
        _test_rwlock.reader_unlock()


  @contextmanager
  def thread_unsafe_test():
    """Decorator for tests that are not thread-safe.

    Note: this decorator (naturally) only applies to what it wraps, not to, say,
    code in separate setUp() or tearDown() methods.
    """
    if TEST_NUM_THREADS.value <= 0:
      yield
      return

    _test_rwlock.assert_reader_held()
    _test_rwlock.reader_unlock()
    _test_rwlock.writer_lock()
    try:
      yield
    finally:
      _test_rwlock.writer_unlock()
      _test_rwlock.reader_lock()
else:
  # TODO(phawkins): remove this branch when jaxlib 0.5.0 is the minimum.
  _test_rwlock = threading.Lock()

  def _run_one_test(test: unittest.TestCase, result: ThreadSafeTestResult):
    _test_rwlock.acquire()
    try:
      test(result)  # type: ignore
    finally:
      _test_rwlock.release()


  @contextmanager
  def thread_unsafe_test():
    yield  # No reader-writer lock, so we get no parallelism.


def thread_unsafe_test_class():
  "Decorator that marks a TestCase class as thread-hostile."
  def f(klass):
    assert issubclass(klass, unittest.TestCase), type(klass)
    klass.thread_hostile = True
    return klass
  return f


class ThreadSafeTestResult:
  """
  Wraps a TestResult to make it thread safe.

  We do this by accumulating API calls and applying them in a batch under a
  lock at the conclusion of each test case.

  We duck type instead of inheriting from TestResult because we aren't actually
  a perfect implementation of TestResult, and would rather get a loud error
  for things we haven't implemented.
  """
  def __init__(self, lock: threading.Lock, result: unittest.TestResult):
    self.lock = lock
    self.test_result = result
    self.actions: list[Callable] = []

  def startTest(self, test: unittest.TestCase):
    del test
    self.start_time = time.time()

  def stopTest(self, test: unittest.TestCase):
    stop_time = time.time()
    with self.lock:
      # If test_result is an ABSL _TextAndXMLTestResult we override how it gets
      # the time. This affects the timing that shows up in the XML output
      # consumed by CI.
      time_getter = getattr(self.test_result, "time_getter", None)
      try:
        self.test_result.time_getter = lambda: self.start_time
        self.test_result.startTest(test)
        for callback in self.actions:
          callback()
        self.test_result.time_getter = lambda: stop_time
        self.test_result.stopTest(test)
      finally:
        if time_getter is not None:
          self.test_result.time_getter = time_getter

  def addSuccess(self, test: unittest.TestCase):
    self.actions.append(lambda: self.test_result.addSuccess(test))

  def addSkip(self, test: unittest.TestCase, reason: str):
    self.actions.append(lambda: self.test_result.addSkip(test, reason))

  def addError(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addError(test, err))

  def addFailure(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addFailure(test, err))

  def addExpectedFailure(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addExpectedFailure(test, err))

  def addDuration(self, test: unittest.TestCase, elapsed):
    self.actions.append(lambda: self.test_result.addDuration(test, elapsed))


class JaxTestSuite(unittest.TestSuite):
  """Runs tests in parallel using threads if TEST_NUM_THREADS is > 1.

  Caution: this test suite does not run setUpClass or setUpModule methods if
  thread parallelism is enabled.
  """

  def __init__(self, suite: unittest.TestSuite):
    super().__init__(list(suite))

  def run(self, result: unittest.TestResult, debug: bool = False) -> unittest.TestResult:
    if TEST_NUM_THREADS.value <= 0:
      return super().run(result)

    test_warning_util.install_threadsafe_warning_handlers()

    executor = ThreadPoolExecutor(TEST_NUM_THREADS.value)
    lock = threading.Lock()
    futures = []

    def run_test(test):
      "Recursively runs tests in a test suite or test case."
      if isinstance(test, unittest.TestSuite):
        for subtest in test:
          run_test(subtest)
      else:
        test_result = ThreadSafeTestResult(lock, result)
        futures.append(executor.submit(_run_one_test, test, test_result))

    with executor:
      run_test(self)
      for future in futures:
        future.result()

    return result


class JaxTestLoader(absltest.TestLoader):
  suiteClass = JaxTestSuite

  def getTestCaseNames(self, testCaseClass):
    names = super().getTestCaseNames(testCaseClass)
    if _TEST_TARGETS.value:
      pattern = re.compile(_TEST_TARGETS.value)
      names = [name for name in names
               if pattern.search(f"{testCaseClass.__name__}.{name}")]
    if _EXCLUDE_TEST_TARGETS.value:
      pattern = re.compile(_EXCLUDE_TEST_TARGETS.value)
      names = [name for name in names
               if not pattern.search(f"{testCaseClass.__name__}.{name}")]
    return names


def with_config(**kwds):
  """Test case decorator for subclasses of JaxTestCase"""
  def decorator(cls):
    assert inspect.isclass(cls) and issubclass(cls, JaxTestCase), "@with_config can only wrap JaxTestCase class definitions."
    cls._default_thread_local_config = {}
    for b in cls.__bases__:
      cls._default_thread_local_config.update(b._default_thread_local_config)
    cls._default_thread_local_config.update(kwds)
    return cls
  return decorator

def with_global_config(**kwds):
  """Test case decorator for subclasses of JaxTestCase"""
  def decorator(cls):
    assert inspect.isclass(cls) and issubclass(cls, JaxTestCase), "@with_config can only wrap JaxTestCase class definitions."
    cls._default_global_config = {}
    for b in cls.__bases__:
      cls._default_global_config.update(b._default_global_config)
    cls._default_global_config.update(kwds)
    return cls
  return decorator


def promote_like_jnp(fun, inexact=False):
  """Decorator that promotes the arguments of `fun` to `jnp.result_type(*args)`.

  jnp and np have different type promotion semantics; this decorator allows
  tests make an np reference implementation act more like a jnp
  implementation.
  """
  _promote = promote_dtypes_inexact if inexact else promote_dtypes
  def wrapper(*args, **kw):
    flat_args, tree = tree_flatten(args)
    args = tree_unflatten(tree, _promote(*flat_args))
    return fun(*args, **kw)
  return wrapper

@contextmanager
def global_config_context(**kwds):
  original_config = {}
  try:
    for key, value in kwds.items():
      original_config[key] = config._read(key)
      config.update(key, value)
    yield
  finally:
    for key, value in original_config.items():
      config.update(key, value)

@contextmanager
def thread_local_config_context(**kwds):
  stack = ExitStack()
  for config_name, value in kwds.items():
    stack.enter_context(config.config_states[config_name](value))
  try:
    yield
  finally:
    stack.close()

class NotPresent:
  def __repr__(self):
    return "<not present>"


@contextmanager
def assert_global_configs_unchanged():
  starting_config = jax.config.values.copy()
  yield
  ending_config = jax.config.values

  if starting_config == ending_config:
    return
  differing = {k: (starting_config.get(k, NotPresent()), ending_config.get(k, NotPresent()))
                for k in (starting_config.keys() | ending_config.keys())
                if (k not in starting_config or k not in ending_config
                    or starting_config[k] != ending_config[k])}
  raise AssertionError(f"Test changed global config values. Differing values are: {differing}")


class JaxTestCase(parameterized.TestCase):
  """Base class for JAX tests including numerical checks and boilerplate."""
  _default_global_config: dict[str, Any] = {}
  _default_thread_local_config = {
    'jax_enable_checks': True,
    'jax_numpy_dtype_promotion': 'strict',
    'jax_numpy_rank_promotion': 'raise',
    'jax_traceback_filtering': 'off',
    'jax_legacy_prng_key': 'error',
  }

  _context_stack: ExitStack | None = None


  def setUp(self):
    super().setUp()
    self.enter_context(assert_global_configs_unchanged())

    # We use the adler32 hash for two reasons.
    # a) it is deterministic run to run, unlike hash() which is randomized.
    # b) it returns values in int32 range, which RandomState requires.
    self._rng = npr.RandomState(zlib.adler32(self._testMethodName.encode()))

    # TODO(phawkins): use TestCase.enterContext once Python 3.11 is the minimum
    # version.
    self._context_stack = ExitStack()
    self.addCleanup(self._context_stack.close)
    stack = self._context_stack
    stack.enter_context(global_config_context(**self._default_global_config))
    for config_name, value in self._default_thread_local_config.items():
      stack.enter_context(jax._src.config.config_states[config_name](value))

    if TEST_WITH_PERSISTENT_COMPILATION_CACHE.value:
      assert TEST_NUM_THREADS.value <= 1, "Persistent compilation cache is not thread-safe."
      stack.enter_context(config.enable_compilation_cache(True))
      stack.enter_context(config.raise_persistent_cache_errors(True))
      stack.enter_context(config.persistent_cache_min_compile_time_secs(0))
      stack.enter_context(config.persistent_cache_min_entry_size_bytes(0))
      tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
      stack.enter_context(config.compilation_cache_dir(tmp_dir))
      stack.callback(compilation_cache.reset_cache)

  def tearDown(self) -> None:
    assert core.reset_trace_state()
    super().tearDown()

  def rng(self):
    return self._rng

  def assertDeprecationWarnsOrRaises(self, deprecation_id: str, message: str):
    """Assert warning or error, depending on deprecation state.

    For use with functions that call :func:`jax._src.deprecations.warn`.
    """
    if deprecations.is_accelerated(deprecation_id):
      return self.assertRaisesRegex(ValueError, message)
    else:
      return self.assertWarnsRegex(DeprecationWarning, message)

  def assertArraysEqual(self, x, y, *, check_dtypes=True, err_msg='',
                        allow_object_dtype=False, verbose=True):
    """Assert that x and y arrays are exactly equal."""
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    x = np.asarray(x)
    y = np.asarray(y)

    if (not allow_object_dtype) and (x.dtype == object or y.dtype == object):
      # See https://github.com/jax-ml/jax/issues/17867
      raise TypeError(
        "assertArraysEqual may be poorly behaved when np.asarray casts to dtype=object. "
        "If comparing PRNG keys, consider random_test.KeyArrayTest.assertKeysEqual. "
        "If comparing collections of arrays, consider using assertAllClose. "
        "To let this test proceed anyway, pass allow_object_dtype=True.")

    # Work around https://github.com/numpy/numpy/issues/18992
    with np.errstate(over='ignore'):
      np.testing.assert_array_equal(x, y, err_msg=err_msg,
                                    verbose=verbose)

  def assertArraysAllClose(self, x, y, *, check_dtypes=True, atol=None,
                           rtol=None, err_msg=''):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
    rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    if not config.enable_x64.value and canonicalize_dtypes:
      self.assertEqual(_dtypes.canonicalize_dtype(_dtype(x), allow_extended_dtype=True),
                       _dtypes.canonicalize_dtype(_dtype(y), allow_extended_dtype=True))
    else:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(self, x, y, *, check_dtypes=True, atol=None, rtol=None,
                     canonicalize_dtypes=True, err_msg=''):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self.assertAllClose(x[k], y[k], check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                            err_msg=err_msg)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(is_sequence(y) and not hasattr(y, '__array__'))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assertAllClose(x_elt, y_elt, check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                            err_msg=err_msg)
    elif hasattr(x, '__array__') or np.isscalar(x):
      self.assertTrue(hasattr(y, '__array__') or np.isscalar(y))
      if check_dtypes:
        self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
      x = np.asarray(x)
      y = np.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
                                err_msg=err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assertMultiLineStrippedEqual(self, expected, what):
    """Asserts two strings are equal, after dedenting and stripping each line."""
    expected = textwrap.dedent(expected)
    what = textwrap.dedent(what)
    ignore_space_re = re.compile(r'\s*\n\s*')
    expected_clean = re.sub(ignore_space_re, '\n', expected.strip())
    what_clean = re.sub(ignore_space_re, '\n', what.strip())
    if what_clean != expected_clean:
      # Print it so we can copy-and-paste it into the test
      print(f"Found\n{what}\n")
    self.assertMultiLineEqual(expected_clean, what_clean,
                              msg=f"Found\n{what}\nExpecting\n{expected}")


  @contextmanager
  def assertNoWarnings(self):
    with test_warning_util.raise_on_warnings():
      yield

  # We replace assertWarns and assertWarnsRegex with functions that use the
  # thread-safe warning utilities. Unlike the unittest versions these only
  # function as context managers.
  @contextmanager
  def assertWarns(self, warning, *, msg=None):
    with test_warning_util.record_warnings() as ws:
      yield
    for w in ws:
      if not isinstance(w.message, warning):
        continue
      if msg is not None and msg not in str(w.message):
        continue
      return
    self.fail(f"Expected warning not found {warning}:'{msg}', got "
              f"{ws}")

  @contextmanager
  def assertWarnsRegex(self, warning, regex):
    if regex is not None:
        regex = re.compile(regex)

    with test_warning_util.record_warnings() as ws:
      yield
    for w in ws:
      if not isinstance(w.message, warning):
        continue
      if regex is not None and not regex.search(str(w.message)):
        continue
      return
    self.fail(f"Expected warning not found {warning}:'{regex}', got "
              f"{ws}")


  def _CompileAndCheck(self, fun, args_maker, *, check_dtypes=True, tol=None,
                       rtol=None, atol=None, check_cache_misses=True):
    """Helper method for running JAX compilation and allclose assertions."""
    args = args_maker()

    def wrapped_fun(*args):
      self.assertTrue(python_should_be_executing)
      return fun(*args)

    python_should_be_executing = True
    python_ans = fun(*args)

    python_shapes = tree_map(lambda x: np.shape(x), python_ans)
    np_shapes = tree_map(lambda x: np.shape(np.asarray(x)), python_ans)
    self.assertEqual(python_shapes, np_shapes)

    cache_misses = dispatch.xla_primitive_callable.cache_info().misses
    python_ans = fun(*args)
    if check_cache_misses and TEST_NUM_THREADS.value <= 1:
      self.assertEqual(
          cache_misses, dispatch.xla_primitive_callable.cache_info().misses,
          "Compilation detected during second call of {} in op-by-op "
          "mode.".format(fun))

    cfun = api.jit(wrapped_fun)
    python_should_be_executing = True
    monitored_ans = cfun(*args)

    python_should_be_executing = False
    compiled_ans = cfun(*args)

    self.assertAllClose(python_ans, monitored_ans, check_dtypes=check_dtypes,
                        atol=atol or tol, rtol=rtol or tol)
    self.assertAllClose(python_ans, compiled_ans, check_dtypes=check_dtypes,
                        atol=atol or tol, rtol=rtol or tol)

    args = args_maker()

    python_should_be_executing = True
    python_ans = fun(*args)

    python_should_be_executing = False
    compiled_ans = cfun(*args)

    self.assertAllClose(python_ans, compiled_ans, check_dtypes=check_dtypes,
                        atol=atol or tol, rtol=rtol or tol)

  def _CheckAgainstNumpy(self, numpy_reference_op, lax_op, args_maker,
                         check_dtypes=True, tol=None, atol=None, rtol=None,
                         canonicalize_dtypes=True):
    args = args_maker()
    lax_ans = lax_op(*args)
    numpy_ans = numpy_reference_op(*args)
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
                        atol=atol or tol, rtol=rtol or tol,
                        canonicalize_dtypes=canonicalize_dtypes)

_PJIT_IMPLEMENTATION = jax.jit
_PJIT_IMPLEMENTATION._name = "jit"
_NOOP_JIT_IMPLEMENTATION = lambda x, *args, **kwargs: x
_NOOP_JIT_IMPLEMENTATION._name = "noop"

JIT_IMPLEMENTATION = (
  _PJIT_IMPLEMENTATION,
  _NOOP_JIT_IMPLEMENTATION,
)

class BufferDonationTestCase(JaxTestCase):
  def assertDeleted(self, x):
    self.assertTrue(x.is_deleted())

  def assertNotDeleted(self, x):
    self.assertFalse(x.is_deleted())


ignore_warning = test_warning_util.ignore_warning

# -------------------- Mesh parametrization helpers --------------------

MeshSpec = list[tuple[str, int]]

@contextmanager
def with_mesh(named_shape: MeshSpec) -> Generator[None, None, None]:
  """Test utility for setting up meshes given mesh data from `schedules`."""
  # This is similar to the `with_mesh` function above, but isn't a decorator.
  axis_names, shape = unzip2(named_shape)
  size = math.prod(shape)
  local_devices = list(jax.local_devices())
  if len(local_devices) < size:
    raise unittest.SkipTest(f"Test requires {size} local devices")
  mesh_devices = np.array(local_devices[:size]).reshape(shape)  # type: ignore
  with jax.sharding.Mesh(mesh_devices, axis_names):
    yield

def with_mesh_from_kwargs(f):
  return lambda *args, **kwargs: with_mesh(kwargs['mesh'])(f)(*args, **kwargs)

def with_and_without_mesh(f):
  return parameterized.named_parameters(
    {"testcase_name": name, "mesh": mesh, "axis_resources": axis_resources}
    for name, mesh, axis_resources in (
      ('', (), ()),
      ('Mesh', (('x', 2),), (('i', 'x'),))
    ))(with_mesh_from_kwargs(f))

def with_user_mesh(sizes, names, axis_types=None):
  axis_types = ({mesh_lib.AxisTypes.Visible: names}
                if axis_types is None else axis_types)
  def decorator(fn):
    def mesh_fn(*args, **kwargs):
      mesh = create_mesh(sizes, names, axis_types=axis_types)
      with mesh_lib.use_mesh(mesh):
        return fn(*args, **kwargs, mesh=mesh)
    return mesh_fn
  return decorator


def create_mesh(mesh_shape, axis_names, iota_order=False, axis_types=None):
  size = math.prod(mesh_shape)
  if len(jax.devices()) < size:
    raise unittest.SkipTest(f"Test requires {size} global devices.")
  if iota_order:
    devices = sorted(jax.devices(), key=lambda d: d.id)
    mesh_devices = np.array(devices[:size]).reshape(mesh_shape)
    return jax.sharding.Mesh(mesh_devices, axis_names, axis_types=axis_types)
  else:
    return jax.make_mesh(mesh_shape, axis_names, axis_types=axis_types)

class _cached_property:
  null = object()

  def __init__(self, method):
    self._method = method
    self._value = self.null

  def __get__(self, obj, cls):
    if self._value is self.null:
      self._value = self._method(obj)
    return self._value


class _LazyDtypes:
  """A class that unifies lists of supported dtypes.

  These could be module-level constants, but device_under_test() is not always
  known at import time, so we need to define these lists lazily.
  """
  def supported(self, dtypes):
    supported = supported_dtypes()
    return type(dtypes)(d for d in dtypes if d in supported)

  @_cached_property
  def custom_floats(self):
    float_dtypes = [
      _dtypes.bfloat16,
      _dtypes.float8_e4m3b11fnuz,
      _dtypes.float8_e4m3fn,
      _dtypes.float8_e4m3fnuz,
      _dtypes.float8_e5m2,
      _dtypes.float8_e5m2fnuz,
    ]
    if _dtypes.float8_e3m4 is not None:
      float_dtypes += [_dtypes.float8_e3m4]
    if _dtypes.float8_e4m3 is not None:
      float_dtypes += [_dtypes.float8_e4m3]
    return self.supported(float_dtypes)

  @_cached_property
  def floating(self):
    return self.supported([np.float32, np.float64])

  @_cached_property
  def all_floating(self):
    return self.supported([_dtypes.bfloat16, np.float16, np.float32, np.float64])

  @_cached_property
  def integer(self):
    return self.supported([np.int32, np.int64])

  @_cached_property
  def all_integer(self):
    return self.supported([np.int8, np.int16, np.int32, np.int64])

  @_cached_property
  def unsigned(self):
    return self.supported([np.uint32, np.uint64])

  @_cached_property
  def all_unsigned(self):
    return self.supported([np.uint8, np.uint16, np.uint32, np.uint64])

  @_cached_property
  def complex(self):
    return self.supported([np.complex64, np.complex128])

  @_cached_property
  def boolean(self):
    return self.supported([np.bool_])

  @_cached_property
  def inexact(self):
    return self.floating + self.complex

  @_cached_property
  def all_inexact(self):
    return self.all_floating + self.complex

  @_cached_property
  def numeric(self):
    return self.floating + self.integer + self.unsigned + self.complex

  @_cached_property
  def all(self):
    return (self.all_floating + self.all_integer + self.all_unsigned +
            self.complex + self.boolean)


dtypes = _LazyDtypes()


def strict_promotion_if_dtypes_match(dtypes):
  """
  Context manager to enable strict promotion if all dtypes match,
  and enable standard dtype promotion otherwise.
  """
  if all(dtype == dtypes[0] for dtype in dtypes):
    return jax.numpy_dtype_promotion('strict')
  return jax.numpy_dtype_promotion('standard')

_version_regex = re.compile(r"([0-9]+(?:\.[0-9]+)*)(?:(rc|dev).*)?")
def parse_version(v: str) -> tuple[int, ...]:
  m = _version_regex.match(v)
  if m is None:
    raise ValueError(f"Unable to parse version '{v}'")
  return tuple(int(x) for x in m.group(1).split('.'))

def numpy_version():
  return parse_version(np.__version__)

def parameterized_filterable(*,
    kwargs: Sequence[dict[str, Any]],
    testcase_name: Callable[[dict[str, Any]], str] | None = None,
    one_containing: str | None = None,
):
  """Decorator for named parameterized tests, with filtering support.

  Works like ``parameterized.named_parameters``, except that it sanitizes the test
  names so that we can use ``pytest -k`` and ``python test.py -k`` test filtering.
  This means, e.g., that many special characters are replaced with `_`.
  It also supports the ``one_containing`` arg to select one of the tests, while
  leaving the name unchanged, which is useful for IDEs to be able to easily
  pick up the enclosing test name.

  Usage:
     @jtu.parameterized_filterable(
       # one_containing="a_4",
       [dict(a=4, b=5),
        dict(a=5, b=4)])
     def test_my_test(self, *, a, b): ...

  Args:
    kwargs: Each entry is a set of kwargs to be passed to the test function.
    testcase_name: Optionally, a function to construct the testcase_name from
      one kwargs dict. If not given then ``kwargs`` may contain ``testcase_name`` and
      otherwise the test case name is constructed as ``str(kwarg)``.
      We sanitize the test names to work with -k test filters. See
      ``sanitize_test_name``.
    one_containing: If given, then leaves the test name unchanged, and use
      only one of the ``kwargs`` whose `testcase_name` includes ``one_containing``.
  """
  # Ensure that all kwargs contain a testcase_name
  kwargs_with_testcase_name: Sequence[dict[str, Any]]
  if testcase_name is not None:
    kwargs_with_testcase_name = [
      dict(testcase_name=sanitize_test_name(str(testcase_name(kw))), **kw)
      for kw in kwargs]
  else:
    for kw in kwargs:
      testcase_name = kw.get("testcase_name")
      if testcase_name is None:
        testcase_name = "_".join(f"{k}={kw[k]}"  # type: ignore
                                 for k in sorted(kw.keys()))
      kw["testcase_name"] = sanitize_test_name(testcase_name)  # type: ignore

    kwargs_with_testcase_name = kwargs
  if one_containing is not None:
    filtered = tuple(kw for kw in kwargs_with_testcase_name
                     if one_containing in kw["testcase_name"])
    assert filtered, (
      f"No testcase_name contains '{one_containing}'. "
      "The testcase_name values are\n  " +
      "\n  ".join(kw["testcase_name"] for kw in kwargs_with_testcase_name))
    kw = filtered[0]
    kw["testcase_name"] = ""
    return parameterized.named_parameters([kw])
  else:
    return parameterized.named_parameters(*kwargs_with_testcase_name)

@contextmanager
def register_event_duration_listener(callback):
  """Manages registering/unregistering an event duration listener callback."""
  try:
    monitoring.register_event_duration_secs_listener(callback)
    yield
  finally:
    monitoring._unregister_event_duration_listener_by_callback(callback)


@contextmanager
def set_env(**kwargs):
  """Context manager to temporarily set/unset one or more environment variables.

  Examples:

    >>> import os
    >>> os.environ['my_var'] = 'original'

    >>> with set_env(my_var=None, other_var='some_value'):
    ...   print("my_var is set:", 'my_var' in os.environ)
    ...   print("other_var =", os.environ['other_var'])
    ...
    my_var is set: False
    other_var = some_value

    >>> os.environ['my_var']
    'original'
    >>> 'other_var' in os.environ
    False
  """
  original = {key: os.environ.pop(key, None) for key in kwargs}
  os.environ.update({k: v for k, v in kwargs.items() if v is not None})
  try:
    yield
  finally:
    _ = [os.environ.pop(key, None) for key in kwargs]
    os.environ.update({k: v for k, v in original.items() if v is not None})

def fwd_bwd_jaxprs(f, *example_args):
  fwd_jaxpr, (y_shape, res_shape) = jax.make_jaxpr(
      lambda *args: jax.vjp(f, *args), return_shape=True)(*example_args)
  bwd_jaxpr = jax.make_jaxpr(lambda res, outs: res(outs))(res_shape, y_shape)
  return fwd_jaxpr, bwd_jaxpr


def numpy_vecdot(x, y, axis):
  """Implementation of numpy.vecdot for testing on numpy < 2.0.0"""
  if numpy_version() >= (2, 0, 0):
    raise ValueError("should be calling vecdot directly on numpy 2.0.0")
  x = np.moveaxis(x, axis, -1)
  y = np.moveaxis(y, axis, -1)
  x, y = np.broadcast_arrays(x, y)
  return np.matmul(np.conj(x[..., None, :]), y[..., None])[..., 0, 0]


def complex_plane_sample(dtype, size_re=10, size_im=None):
  """Return a 2-D array of complex numbers that covers the complex plane
     with a grid of samples.

     The size of the grid is (3 + 2 * size_im) x (3 + 2 * size_re)
     that includes infinity points, extreme finite points, and the
     specified number of points from real and imaginary axis.

     For example:

     >>> print(complex_plane_sample(np.complex64, 0, 3))
     [[-inf          -infj   0.          -infj  inf          -infj]
      [-inf-3.4028235e+38j   0.-3.4028235e+38j  inf-3.4028235e+38j]
      [-inf-2.0000000e+00j   0.-2.0000000e+00j  inf-2.0000000e+00j]
      [-inf-1.1754944e-38j   0.-1.1754944e-38j  inf-1.1754944e-38j]
      [-inf+0.0000000e+00j   0.+0.0000000e+00j  inf+0.0000000e+00j]
      [-inf+1.1754944e-38j   0.+1.1754944e-38j  inf+1.1754944e-38j]
      [-inf+2.0000000e+00j   0.+2.0000000e+00j  inf+2.0000000e+00j]
      [-inf+3.4028235e+38j   0.+3.4028235e+38j  inf+3.4028235e+38j]
      [-inf          +infj   0.          +infj  inf          +infj]]

  """
  if size_im is None:
    size_im = size_re
  finfo = np.finfo(dtype)

  machine = platform.machine()
  is_arm_cpu = machine.startswith('aarch') or machine.startswith('arm')
  smallest = np.nextafter(finfo.tiny, finfo.max) if is_arm_cpu and platform.system() == 'Darwin' else finfo.tiny

  def make_axis_points(size):
    prec_dps_ratio = 3.3219280948873626
    logmin = logmax = finfo.maxexp / prec_dps_ratio
    logtiny = finfo.minexp / prec_dps_ratio
    axis_points = np.zeros(3 + 2 * size, dtype=finfo.dtype)

    with ignore_warning(category=RuntimeWarning):
      # Silence RuntimeWarning: overflow encountered in cast
      half_neg_line = -np.logspace(logmin, logtiny, size, dtype=finfo.dtype)
      half_line = -half_neg_line[::-1]
      axis_points[-size - 1:-1] = half_line
      axis_points[1:size + 1] = half_neg_line

    if size > 1:
      axis_points[1] = finfo.min
      axis_points[-2] = finfo.max
    if size > 0:
      axis_points[size] = -smallest
      axis_points[-size - 1] = smallest
    axis_points[0] = -np.inf
    axis_points[-1] = np.inf
    return axis_points

  real_axis_points = make_axis_points(size_re)
  imag_axis_points = make_axis_points(size_im)

  real_part = real_axis_points.reshape((-1, 3 + 2 * size_re)).repeat(3 + 2 * size_im, 0).astype(dtype)

  imag_part = imag_axis_points.repeat(2).view(dtype)
  imag_part.real[:] = 0
  imag_part = imag_part.reshape((3 + 2 * size_im, -1)).repeat(3 + 2 * size_re, 1)

  return real_part + imag_part


class vectorize_with_mpmath(np.vectorize):
  """Same as numpy.vectorize but using mpmath backend for function evaluation.
  """

  map_float_to_complex = dict(float16='complex32', float32='complex64', float64='complex128', float128='complex256', longdouble='clongdouble')
  map_complex_to_float = {v: k for k, v in map_float_to_complex.items()}

  float_prec = dict(
    # float16=11,
    float32=24,
    float64=53,
    # float128=113,
    # longdouble=113
  )

  float_minexp = dict(
    float16=-14,
    float32=-126,
    float64=-1022,
    float128=-16382
  )

  float_maxexp = dict(
    float16=16,
    float32=128,
    float64=1024,
    float128=16384,
  )

  def __init__(self, *args, **kwargs):
    mpmath = kwargs.pop('mpmath', None)
    if mpmath is None:
      raise ValueError('vectorize_with_mpmath: no mpmath argument specified')
    self.extra_prec_multiplier = kwargs.pop('extra_prec_multiplier', 0)
    self.extra_prec = kwargs.pop('extra_prec', 0)
    self.mpmath = mpmath
    self.contexts = dict()
    self.contexts_inv = dict()
    for fp_format, prec in self.float_prec.items():
      ctx = self.mpmath.mp.clone()
      ctx.prec = prec
      self.contexts[fp_format] = ctx
      self.contexts_inv[ctx] = fp_format

    super().__init__(*args, **kwargs)

  def get_context(self, x):
    if isinstance(x, (np.ndarray, np.floating, np.complexfloating)):
      fp_format = str(x.dtype)
      fp_format = self.map_complex_to_float.get(fp_format, fp_format)
      return self.contexts[fp_format]
    raise NotImplementedError(f'get mpmath context from {type(x).__name__} instance')

  def nptomp(self, x):
    """Convert numpy array/scalar to an array/instance of mpmath number type.
    """
    if isinstance(x, np.ndarray):
      return np.fromiter(map(self.nptomp, x.flatten()), dtype=object).reshape(x.shape)
    elif isinstance(x, np.floating):
      mpmath = self.mpmath
      ctx = self.get_context(x)
      prec, rounding = ctx._prec_rounding
      if np.isposinf(x):
        return ctx.make_mpf(mpmath.libmp.finf)
      elif np.isneginf(x):
        return ctx.make_mpf(mpmath.libmp.fninf)
      elif np.isnan(x):
        return ctx.make_mpf(mpmath.libmp.fnan)
      elif np.isfinite(x):
        mantissa, exponent = np.frexp(x)
        man = int(np.ldexp(mantissa, prec))
        exp = int(exponent - prec)
        r = ctx.make_mpf(mpmath.libmp.from_man_exp(man, exp, prec, rounding))
        assert ctx.isfinite(r), r._mpf_
        return r
    elif isinstance(x, np.complexfloating):
      re, im = self.nptomp(x.real), self.nptomp(x.imag)
      return re.context.make_mpc((re._mpf_, im._mpf_))
    raise NotImplementedError(f'convert {type(x).__name__} instance to mpmath number type')

  def mptonp(self, x):
    """Convert mpmath instance to numpy array/scalar type.
    """
    if isinstance(x, np.ndarray) and x.dtype.kind == 'O':
      x_flat = x.flatten()
      item = x_flat[0]
      ctx = item.context
      fp_format = self.contexts_inv[ctx]
      if isinstance(item, ctx.mpc):
        dtype = getattr(np, self.map_float_to_complex[fp_format])
      elif isinstance(item, ctx.mpf):
        dtype = getattr(np, fp_format)
      else:
        dtype = None
      if dtype is not None:
        return np.fromiter(map(self.mptonp, x_flat), dtype=dtype).reshape(x.shape)
    elif isinstance(x, self.mpmath.ctx_mp.mpnumeric):
      ctx = x.context
      if isinstance(x, ctx.mpc):
        fp_format = self.contexts_inv[ctx]
        dtype = getattr(np, self.map_float_to_complex[fp_format])
        r = dtype().reshape(1).view(getattr(np, fp_format))
        r[0] = self.mptonp(x.real)
        r[1] = self.mptonp(x.imag)
        return r.view(dtype)[0]
      elif isinstance(x, ctx.mpf):
        fp_format = self.contexts_inv[ctx]
        dtype = getattr(np, fp_format)
        if ctx.isfinite(x):
          sign, man, exp, bc = self.mpmath.libmp.normalize(*x._mpf_, *ctx._prec_rounding)
          assert bc >= 0, (sign, man, exp, bc, x._mpf_)
          if exp + bc < self.float_minexp[fp_format]:
            return -ctx.zero if sign else ctx.zero
          if exp + bc > self.float_maxexp[fp_format]:
            return ctx.ninf if sign else ctx.inf
          man = dtype(-man if sign else man)
          r = np.ldexp(man, exp)
          assert np.isfinite(r), (x, r, x._mpf_, man)
          return r
        elif ctx.isnan(x):
          return dtype(np.nan)
        elif ctx.isinf(x):
          return dtype(-np.inf if x._mpf_[0] else np.inf)
    raise NotImplementedError(f'convert {type(x)} instance to numpy floating point type')

  def __call__(self, *args, **kwargs):
    mp_args = []
    context = None
    for a in args:
      if isinstance(a, (np.ndarray, np.floating, np.complexfloating)):
        mp_args.append(self.nptomp(a))
        if context is None:
          context = self.get_context(a)
        else:
          assert context is self.get_context(a)
      else:
        mp_args.append(a)

    extra_prec = int(context.prec * self.extra_prec_multiplier) + self.extra_prec
    with context.extraprec(extra_prec):
      result = super().__call__(*mp_args, **kwargs)

    if isinstance(result, tuple):
      lst = []
      for r in result:
        if ((isinstance(r, np.ndarray) and r.dtype.kind == 'O')
            or isinstance(r, self.mpmath.ctx_mp.mpnumeric)):
          r = self.mptonp(r)
        lst.append(r)
      return tuple(lst)

    if ((isinstance(result, np.ndarray) and result.dtype.kind == 'O')
        or isinstance(result, self.mpmath.ctx_mp.mpnumeric)):
      return self.mptonp(result)

    return result


class numpy_with_mpmath:
  """Namespace of universal functions on numpy arrays that use mpmath
  backend for evaluation and return numpy arrays as outputs.
  """

  _provides = [
    'abs', 'absolute', 'sqrt', 'exp', 'expm1', 'exp2',
    'log', 'log1p', 'log10', 'log2',
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
    'square', 'positive', 'negative', 'conjugate', 'sign', 'sinc',
    'normalize',
  ]

  _mp_names = dict(
    abs='absmin', absolute='absmin',
    log='ln',
    arcsin='asin', arccos='acos', arctan='atan',
    arcsinh='asinh', arccosh='acosh', arctanh='atanh',
  )

  def __init__(self, mpmath, extra_prec_multiplier=0, extra_prec=0):
    self.mpmath = mpmath

    for name in self._provides:
      mp_name = self._mp_names.get(name, name)

      if hasattr(self, name):
        op = getattr(self, name)
      else:

        def op(x, mp_name=mp_name):
          return getattr(x.context, mp_name)(x)

      setattr(self, name, vectorize_with_mpmath(op, mpmath=mpmath, extra_prec_multiplier=extra_prec_multiplier, extra_prec=extra_prec))

  # The following function methods operate on mpmath number instances.
  # The corresponding function names must be listed in
  # numpy_with_mpmath._provides list.

  def square(self, x):
    return x * x

  def positive(self, x):
    return x

  def negative(self, x):
    return -x

  def sqrt(self, x):
    ctx = x.context
    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in sqrt(+-inf+-infj) evaluation (see mpmath/mpmath#776).
      # TODO(pearu): remove this function when mpmath 1.4 or newer
      # will be the required test dependency.
      if ctx.isinf(x.imag):
        return ctx.make_mpc((ctx.inf._mpf_, x.imag._mpf_))
    return ctx.sqrt(x)

  def expm1(self, x):
    return x.context.expm1(x)

  def log1p(self, x):
    ctx = x.context
    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in log(+-inf+-infj) evaluation (see mpmath/mpmath#774).
      # TODO(pearu): remove this function when mpmath 1.4 or newer
      # will be the required test dependency.
      if ctx.isinf(x.real) and ctx.isinf(x.imag):
        pi = ctx.pi
        if x.real > 0 and x.imag > 0:
          return ctx.make_mpc((x.real._mpf_, (pi / 4)._mpf_))
        if x.real > 0 and x.imag < 0:
          return ctx.make_mpc((x.real._mpf_, (-pi / 4)._mpf_))
        if x.real < 0 and x.imag < 0:
          return ctx.make_mpc(((-x.real)._mpf_, (-3 * pi / 4)._mpf_))
        if x.real < 0 and x.imag > 0:
          return ctx.make_mpc(((-x.real)._mpf_, (3 * pi / 4)._mpf_))
    return ctx.log1p(x)

  def tan(self, x):
    ctx = x.context
    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in tan(+-inf+-infj) evaluation (see mpmath/mpmath#781).
      # TODO(pearu): remove this function when mpmath 1.4 or newer
      # will be the required test dependency.
      if ctx.isinf(x.imag) and (ctx.isinf(x.real) or ctx.isfinite(x.real)):
        if x.imag > 0:
          return ctx.make_mpc((ctx.zero._mpf_, ctx.one._mpf_))
        return ctx.make_mpc((ctx.zero._mpf_, (-ctx.one)._mpf_))
      if ctx.isinf(x.real) and ctx.isfinite(x.imag):
        return ctx.make_mpc((ctx.nan._mpf_, ctx.nan._mpf_))
    return ctx.tan(x)

  def tanh(self, x):
    ctx = x.context
    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in tanh(+-inf+-infj) evaluation (see mpmath/mpmath#781).
      # TODO(pearu): remove this function when mpmath 1.4 or newer
      # will be the required test dependency.
      if ctx.isinf(x.imag) and (ctx.isinf(x.real) or ctx.isfinite(x.real)):
        if x.imag > 0:
          return ctx.make_mpc((ctx.zero._mpf_, ctx.one._mpf_))
        return ctx.make_mpc((ctx.zero._mpf_, (-ctx.one)._mpf_))
      if ctx.isinf(x.real) and ctx.isfinite(x.imag):
        return ctx.make_mpc((ctx.nan._mpf_, ctx.nan._mpf_))
    return ctx.tanh(x)

  def log2(self, x):
    return x.context.ln(x) / x.context.ln2

  def log10(self, x):
    return x.context.ln(x) / x.context.ln10

  def exp2(self, x):
    return x.context.exp(x * x.context.ln2)

  def arcsin(self, x):
    ctx = x.context
    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in asin(+-inf+-infj) evaluation (see
      # mpmath/mpmath#793).
      # TODO(pearu): remove the if-block below when mpmath 1.4 or
      # newer will be the required test dependency.
      pi = ctx.pi
      inf = ctx.inf
      zero = ctx.zero
      if ctx.isinf(x.real):
        sign_real = -1 if x.real < 0 else 1
        real = sign_real * pi / (4 if ctx.isinf(x.imag) else 2)
        imag = -inf if x.imag < 0 else inf
        return ctx.make_mpc((real._mpf_, imag._mpf_))
      elif ctx.isinf(x.imag):
        return ctx.make_mpc((zero._mpf_, x.imag._mpf_))

      # On branch cut, mpmath.mp.asin returns different value compared
      # to mpmath.fp.asin and numpy.arcsin (see
      # mpmath/mpmath#786). The following if-block ensures
      # compatibility with numpy.arcsin.
      if x.real > 1 and x.imag == 0:
        return ctx.asin(x).conjugate()

    return ctx.asin(x)

  def arccos(self, x):
    ctx = x.context

    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in acos(+-inf+-infj) evaluation (see
      # mpmath/mpmath#793).
      # TODO(pearu): remove the if-block below when mpmath 1.4 or
      # newer will be the required test dependency.
      pi = ctx.pi
      inf = ctx.inf
      zero = ctx.zero

      if ctx.isinf(x.imag):
        if ctx.isinf(x.real):
          real = pi / 4 if x.real > 0 else 3 * pi / 4
        else:
          real = pi / 2
        imag = inf if x.imag < 0 else -inf
        return ctx.make_mpc((real._mpf_, imag._mpf_))
      elif ctx.isinf(x.real):
        inf = ctx.inf
        sign_imag = -1 if x.imag < 0 else 1
        real = zero if x.real > 0 else pi
        return ctx.make_mpc((real._mpf_, (-sign_imag * inf)._mpf_))
      # On branch cut, mpmath.mp.acos returns different value
      # compared to mpmath.fp.acos and numpy.arccos. The
      # following if-block ensures compatibility with
      # numpy.arccos.
      if x.imag == 0 and x.real > 1:
        return -ctx.acos(x)

    return ctx.acos(x)

  def arcsinh(self, x):
    ctx = x.context

    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in asinh(+-inf+-infj) evaluation
      # (see mpmath/mpmath#749).
      # TODO(pearu): remove the if-block below when mpmath 1.4 or
      # newer will be the required test dependency.
      pi = ctx.pi
      inf = ctx.inf
      zero = ctx.zero
      if ctx.isinf(x.imag):
        sign_imag = -1 if x.imag < 0 else 1
        real = -inf if x.real < 0 else inf
        imag = sign_imag * pi / (4 if ctx.isinf(x.real) else 2)
        return ctx.make_mpc((real._mpf_, imag._mpf_))
      elif ctx.isinf(x.real):
        return ctx.make_mpc((x.real._mpf_, zero._mpf_))

      # On branch cut, mpmath.mp.asinh returns different value
      # compared to mpmath.fp.asinh and numpy.arcsinh (see
      # mpmath/mpmath#786).  The following if-block ensures
      # compatibility with numpy.arcsinh.
      if x.real == 0 and x.imag < -1:
        return (-ctx.asinh(x)).conjugate()
    return ctx.asinh(x)

  def arccosh(self, x):
    ctx = x.context

    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in acosh(+-inf+-infj) evaluation
      # (see mpmath/mpmath#749).
      pi = ctx.pi
      inf = ctx.inf
      zero = ctx.zero
      if ctx.isinf(x.real):
        sign_imag = -1 if x.imag < 0 else 1
        imag = (
          (3 if x.real < 0 else 1) * sign_imag * pi / 4
          if ctx.isinf(x.imag)
          else (sign_imag * pi if x.real < 0 else zero)
        )
        return ctx.make_mpc((inf._mpf_, imag._mpf_))
      elif ctx.isinf(x.imag):
        sign_imag = -1 if x.imag < 0 else 1
        imag = sign_imag * pi / 2
        return ctx.make_mpc((inf._mpf_, imag._mpf_))
    return ctx.acosh(x)

  def arctan(self, x):
    ctx = x.context

    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in atan(+-inf+-infj) evaluation
      # (see mpmath/mpmath#775 with the fix).
      # TODO(pearu): remove the if-block below when mpmath 1.4 or
      # newer will be the required test dependency.
      pi = ctx.pi
      zero = ctx.zero
      if ctx.isinf(x.real) or ctx.isinf(x.imag):
        if x.real < 0:
          return ctx.make_mpc(((-pi / 2)._mpf_, zero._mpf_))
        return ctx.make_mpc(((pi / 2)._mpf_, zero._mpf_))

      # On branch cut, mpmath.mp.atan returns different value compared
      # to mpmath.fp.atan and numpy.arctan (see mpmath/mpmath#865).
      # The following if-block ensures compatibility with
      # numpy.arctan.
      if x.real == 0 and x.imag < -1:
        return (-ctx.atan(x)).conjugate()
    return ctx.atan(x)

  def arctanh(self, x):
    ctx = x.context

    if isinstance(x, ctx.mpc):
      # Workaround mpmath 1.3 bug in atanh(+-inf+-infj) evaluation
      # (see mpmath/mpmath#775 with the fix).
      # TODO(pearu): remove the if-block below when mpmath 1.4 or
      # newer will be the required test dependency.
      pi = ctx.pi
      zero = ctx.zero
      if ctx.isinf(x.real) or ctx.isinf(x.imag):
        if x.imag < 0:
          return ctx.make_mpc((zero._mpf_, (-pi / 2)._mpf_))
        return ctx.make_mpc((zero._mpf_, (pi / 2)._mpf_))

      # On branch cut, mpmath.mp.atanh returns different value
      # compared to mpmath.fp.atanh and numpy.arctanh.  The following
      # if-block ensures compatibility with numpy.arctanh.
      if x.imag == 0 and x.real > 1:
        return ctx.atanh(x).conjugate()
    return ctx.atanh(x)

  def normalize(self, exact, reference, value):
    """Normalize reference and value using precision defined by the
    difference of exact and reference.
    """
    def worker(ctx, s, e, r, v):
      ss, sm, se, sbc = s._mpf_
      es, em, ee, ebc = e._mpf_
      rs, rm, re, rbc = r._mpf_
      vs, vm, ve, vbc = v._mpf_

      if not (ctx.isfinite(e) and ctx.isfinite(r) and ctx.isfinite(v)):
        return r, v

      me = min(se, ee, re, ve)

      # transform mantissa parts to the same exponent base
      sm_e = sm << (se - me)
      em_e = em << (ee - me)
      rm_e = rm << (re - me)
      vm_e = vm << (ve - me)

      # find matching higher and non-matching lower bits of e and r
      sm_b = bin(sm_e)[2:] if sm_e else ''
      em_b = bin(em_e)[2:] if em_e else ''
      rm_b = bin(rm_e)[2:] if rm_e else ''
      vm_b = bin(vm_e)[2:] if vm_e else ''

      m = max(len(sm_b), len(em_b), len(rm_b), len(vm_b))
      em_b = '0' * (m - len(em_b)) + em_b
      rm_b = '0' * (m - len(rm_b)) + rm_b

      c1 = 0
      for b0, b1 in zip(em_b, rm_b):
        if b0 != b1:
          break
        c1 += 1
      c0 = m - c1

      # truncate r and v mantissa
      rm_m = rm_e >> c0
      vm_m = vm_e >> c0

      # normalized r and v
      nr = ctx.make_mpf((rs, rm_m, -c1, len(bin(rm_m)) - 2)) if rm_m else (-ctx.zero if rs else ctx.zero)
      nv = ctx.make_mpf((vs, vm_m, -c1, len(bin(vm_m)) - 2)) if vm_m else (-ctx.zero if vs else ctx.zero)

      return nr, nv

    ctx = exact.context
    scale = abs(exact)
    if isinstance(exact, ctx.mpc):
      rr, rv = worker(ctx, scale, exact.real, reference.real, value.real)
      ir, iv = worker(ctx, scale, exact.imag, reference.imag, value.imag)
      return ctx.make_mpc((rr._mpf_, ir._mpf_)), ctx.make_mpc((rv._mpf_, iv._mpf_))
    elif isinstance(exact, ctx.mpf):
      return worker(ctx, scale, exact, reference, value)
    else:
      assert 0  # unreachable

# Hypothesis testing support
def setup_hypothesis(max_examples=30) -> None:
  """Sets up the hypothesis profiles.

  Sets up the hypothesis testing profiles, and selects the one specified by
  the ``JAX_HYPOTHESIS_PROFILE`` environment variable (or the
  ``--jax_hypothesis_profile`` configuration.

  Args:
    max_examples: the maximum number of hypothesis examples to try, when using
      the default "deterministic" profile.
  """
  try:
    import hypothesis as hp
  except (ModuleNotFoundError, ImportError):
    return

  # In our tests we often use subclasses with slightly different class variables
  # to generate whole suites of parameterized tests, but this approach does not
  # work well with Hypothesis databases, which use some function of the method
  # identity to generate keys. But, if the method is defined in a superclass,
  # all subclasses share the same key. This key collision can lead to confusing
  # false positives in other health checks.
  #
  # Still, as far as I understand, for as long as we don't use the example
  # database, it should be perfectly safe to suppress this health check. This
  # seems simpler than rewriting our tests that trigger this behavior. See
  # the end of https://github.com/HypothesisWorks/hypothesis/issues/3446 for
  # more context.
  suppressed_checks = []
  if hasattr(hp.HealthCheck, "differing_executors"):
    suppressed_checks.append(hp.HealthCheck.differing_executors)

  hp.settings.register_profile(
      "deterministic",
      database=None,
      derandomize=True,
      deadline=None,
      max_examples=max_examples,
      print_blob=True,
      suppress_health_check=suppressed_checks,
  )
  hp.settings.register_profile(
      "interactive",
      parent=hp.settings.load_profile("deterministic"),
      max_examples=1,
      report_multiple_bugs=False,
      verbosity=hp.Verbosity.verbose,
      # Don't try and shrink
      phases=(
          hp.Phase.explicit,
          hp.Phase.reuse,
          hp.Phase.generate,
          hp.Phase.target,
          hp.Phase.explain,
      ),
  )
  profile = HYPOTHESIS_PROFILE.value
  logging.info("Using hypothesis profile: %s", profile)
  hp.settings.load_profile(profile)
