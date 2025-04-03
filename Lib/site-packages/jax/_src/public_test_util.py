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

from functools import partial
import operator

from jax._src import api
from jax._src import config
from jax._src import dtypes as _dtypes
from jax._src.tree_util import tree_map, tree_reduce

import numpy as np


# The only functions intended to be exported are these; they should be used via
# jax.test_util. All other functionality appearing here is for internal use only,
# and may be changed or removed at any time and without any deprecation cycle.
__all__ = ['check_grads', 'check_jvp', 'check_vjp']


EPS = 1e-4


def _dtype(x):
  if hasattr(x, 'dtype'):
    return x.dtype
  elif type(x) in _dtypes.python_scalar_dtypes:
    return np.dtype(_dtypes.python_scalar_dtypes[type(x)])
  else:
    return np.asarray(x).dtype


_default_tolerance = {
    _dtypes.float0: 0,
    np.dtype(np.bool_): 0,
    np.dtype(_dtypes.int4): 0,
    np.dtype(np.int8): 0,
    np.dtype(np.int16): 0,
    np.dtype(np.int32): 0,
    np.dtype(np.int64): 0,
    np.dtype(_dtypes.uint4): 0,
    np.dtype(np.uint8): 0,
    np.dtype(np.uint16): 0,
    np.dtype(np.uint32): 0,
    np.dtype(np.uint64): 0,
    np.dtype(_dtypes.float8_e4m3b11fnuz): 1e-1,
    np.dtype(_dtypes.float8_e4m3fn): 1e-1,
    np.dtype(_dtypes.float8_e4m3fnuz): 1e-1,
    np.dtype(_dtypes.float8_e5m2): 1e-1,
    np.dtype(_dtypes.float8_e5m2fnuz): 1e-1,
    np.dtype(_dtypes.bfloat16): 1e-2,
    np.dtype(np.float16): 1e-3,
    np.dtype(np.float32): 1e-6,
    np.dtype(np.float64): 1e-15,
    np.dtype(np.complex64): 1e-6,
    np.dtype(np.complex128): 1e-15,
}

if _dtypes.int2 is not None:
  assert _dtypes.uint2 is not None
  _default_tolerance[np.dtype(_dtypes.int2)] = 0
  _default_tolerance[np.dtype(_dtypes.uint2)] = 0

def default_tolerance():
  return _default_tolerance


default_gradient_tolerance = {
  np.dtype(_dtypes.float8_e4m3b11fnuz): 1e-1,
  np.dtype(_dtypes.float8_e4m3fn): 1e-1,
  np.dtype(_dtypes.float8_e4m3fnuz): 1e-1,
  np.dtype(_dtypes.float8_e5m2): 1e-1,
  np.dtype(_dtypes.float8_e5m2fnuz): 1e-1,
  np.dtype(_dtypes.bfloat16): 1e-1,
  np.dtype(np.float16): 1e-2,
  np.dtype(np.float32): 2e-3,
  np.dtype(np.float64): 1e-5,
  np.dtype(np.complex64): 1e-3,
  np.dtype(np.complex128): 1e-5,
}

# TODO: make this unconditional when ml_dtypes>=0.5.0 is required
if _dtypes.float8_e3m4 is not None:
  _default_tolerance[np.dtype(_dtypes.float8_e3m4)] = 1e-1
  default_gradient_tolerance[np.dtype(_dtypes.float8_e3m4)] = 1e-1
if _dtypes.float8_e4m3 is not None:
  _default_tolerance[np.dtype(_dtypes.float8_e4m3)] = 1e-1
  default_gradient_tolerance[np.dtype(_dtypes.float8_e4m3)] = 1e-1

def is_python_scalar(val):
  return not isinstance(val, np.generic) and isinstance(val, (bool, int, float, complex))

def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=''):
  if a.dtype == b.dtype == _dtypes.float0:
    np.testing.assert_array_equal(a, b, err_msg=err_msg)
    return

  custom_float_dtypes = [
    _dtypes.float8_e4m3b11fnuz,
    _dtypes.float8_e4m3fn,
    _dtypes.float8_e4m3fnuz,
    _dtypes.float8_e5m2,
    _dtypes.float8_e5m2fnuz,
    _dtypes.bfloat16,
  ]

  if _dtypes.float8_e4m3 is not None:
    custom_float_dtypes.insert(0, _dtypes.float8_e4m3)
  if _dtypes.float8_e3m4 is not None:
    custom_float_dtypes.insert(0, _dtypes.float8_e3m4)

  def maybe_upcast(x):
    if x.dtype in custom_float_dtypes:
      return x.astype(np.float32)
    # TODO(reedwm): Upcasting int2/int4 to int8 will no longer be necessary once
    # JAX depends on a version of ml_dtypes which contains
    # https://github.com/jax-ml/ml_dtypes/commit/348fd3704306cae97f617c38045cee6bc416bf10.
    if x.dtype in _dtypes._intn_dtypes:
      return x.astype(np.int8 if _dtypes.iinfo(x.dtype).min < 0 else np.uint8)
    return x

  a = maybe_upcast(a)
  b = maybe_upcast(b)

  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  with np.errstate(invalid='ignore'):
    # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
    # value errors. It should not do that.
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)

def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {np.dtype(key): value for key, value in tol.items()}
  dtype = _dtypes.canonicalize_dtype(np.dtype(dtype))
  return tol.get(dtype, default_tolerance()[dtype])


def _assert_numpy_close(a, b, atol=None, rtol=None, err_msg=''):
  a, b = np.asarray(a), np.asarray(b)
  assert a.shape == b.shape
  atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
  rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
  _assert_numpy_allclose(a, b, atol=atol * a.size, rtol=rtol * b.size,
                         err_msg=err_msg)


def check_close(xs, ys, atol=None, rtol=None, err_msg=''):
  assert_close = partial(_assert_numpy_close, atol=atol, rtol=rtol,
                         err_msg=err_msg)
  tree_map(assert_close, xs, ys)


def _check_dtypes_match(xs, ys):
  def _assert_dtypes_match(x, y):
    if config.enable_x64.value:
      assert _dtype(x) == _dtype(y)
    else:
      assert (_dtypes.canonicalize_dtype(_dtype(x)) ==
              _dtypes.canonicalize_dtype(_dtype(y)))
  tree_map(_assert_dtypes_match, xs, ys)


def inner_prod(xs, ys):
  def contract(x, y):
    return np.real(np.dot(np.conj(x).reshape(-1), y.reshape(-1)))
  return tree_reduce(np.add, tree_map(contract, xs, ys))


def _safe_subtract(x, y, *, dtype):
  """Subtraction that with `inf - inf == 0` semantics."""
  with np.errstate(invalid='ignore'):
    return np.where(np.equal(x, y), np.array(0, dtype),
                    np.subtract(x, y, dtype=dtype))

def _preserve_input_types(f):
  def wrapped(*args):
    dtype = _dtype(args[0])
    result = np.array(f(*args), dtype=dtype)
    if all(is_python_scalar(arg) for arg in args):
      result = result.item()
    return result
  return wrapped


add = partial(tree_map, _preserve_input_types(operator.add))
sub = partial(tree_map, _preserve_input_types(operator.sub))
safe_sub = partial(tree_map,
                   lambda x, y: _safe_subtract(x, y, dtype=_dtype(x)))
conj = partial(tree_map, _preserve_input_types(np.conj))


def scalar_mul(xs, a):
  def mul(x):
    dtype = _dtype(x)
    result = np.multiply(x, np.array(a, dtype=dtype), dtype=dtype)
    return result.item() if is_python_scalar(x) else result
  return tree_map(mul, xs)


def rand_like(rng, x):
  shape = np.shape(x)
  dtype = _dtype(x)
  randn = lambda: np.asarray(rng.randn(*shape), dtype=dtype)
  if _dtypes.issubdtype(dtype, np.complexfloating):
    result = randn() + dtype.type(1.0j) * randn()
  else:
    result = randn()
  return result.item() if is_python_scalar(x) else result


def numerical_jvp(f, primals, tangents, eps=EPS):
  delta = scalar_mul(tangents, eps)
  f_pos = f(*add(primals, delta))
  f_neg = f(*sub(primals, delta))
  return scalar_mul(safe_sub(f_pos, f_neg), 0.5 / eps)


def _merge_tolerance(tol, default):
  if tol is None:
    return default
  if not isinstance(tol, dict):
    return tol
  out = default.copy()
  for k, v in tol.items():
    out[np.dtype(k)] = v
  return out


def check_jvp(f, f_jvp, args, atol=None, rtol=None, eps=EPS, err_msg=''):
  atol = _merge_tolerance(atol, default_gradient_tolerance)
  rtol = _merge_tolerance(rtol, default_gradient_tolerance)
  rng = np.random.RandomState(0)
  tangent = tree_map(partial(rand_like, rng), args)
  v_out, t_out = f_jvp(args, tangent)
  _check_dtypes_match(v_out, t_out)
  v_out_expected = f(*args)
  _check_dtypes_match(v_out, v_out_expected)
  t_out_expected = numerical_jvp(f, args, tangent, eps=eps)
  # In principle we should expect exact equality of v_out and v_out_expected,
  # but due to nondeterminism especially on GPU (e.g., due to convolution
  # autotuning) we only require "close".
  check_close(v_out, v_out_expected, atol=atol, rtol=rtol,
              err_msg=f'{err_msg} primal' if err_msg else 'primal')
  check_close(t_out, t_out_expected, atol=atol, rtol=rtol,
              err_msg=f'{err_msg} tangent' if err_msg else 'tangent')


def check_vjp(f, f_vjp, args, atol=None, rtol=None, eps=EPS, err_msg=''):
  atol = _merge_tolerance(atol, default_gradient_tolerance)
  rtol = _merge_tolerance(rtol, default_gradient_tolerance)
  _rand_like = partial(rand_like, np.random.RandomState(0))
  v_out, vjpfun = f_vjp(*args)
  v_out_expected = f(*args)
  check_close(v_out, v_out_expected, atol=atol, rtol=rtol,
              err_msg=f'{err_msg} primal' if err_msg else 'primal')
  tangent = tree_map(_rand_like, args)
  tangent_out = numerical_jvp(f, args, tangent, eps=eps)
  cotangent = tree_map(_rand_like, v_out)
  cotangent_out = conj(vjpfun(conj(cotangent)))
  ip = inner_prod(tangent, cotangent_out)
  ip_expected = inner_prod(tangent_out, cotangent)
  check_close(ip, ip_expected, atol=atol, rtol=rtol,
              err_msg=(f'{err_msg} cotangent projection'
                       if err_msg else 'cotangent projection'))


def check_grads(f, args, order,
                modes=("fwd", "rev"), atol=None, rtol=None, eps=None):
  """Check gradients from automatic differentiation against finite differences.

  Gradients are only checked in a single randomly chosen direction, which
  ensures that the finite difference calculation does not become prohibitively
  expensive even for large input/output spaces.

  Args:
    f: function to check at ``f(*args)``.
    args: tuple of argument values.
    order: forward and backwards gradients up to this order are checked.
    modes: lists of gradient modes to check ('fwd' and/or 'rev').
    atol: absolute tolerance for gradient equality.
    rtol: relative tolerance for gradient equality.
    eps: step size used for finite differences.

  Raises:
    AssertionError: if gradients do not match.
  """
  args = tuple(args)
  eps = eps or EPS

  _check_jvp = partial(check_jvp, atol=atol, rtol=rtol, eps=eps)
  _check_vjp = partial(check_vjp, atol=atol, rtol=rtol, eps=eps)

  def _check_grads(f, args, order, err_msg=''):
    if "fwd" in modes:
      fwd_msg = f'JVP of {err_msg}' if err_msg else 'JVP'
      _check_jvp(f, partial(api.jvp, f), args, err_msg=fwd_msg)
      if order > 1:
        _check_grads(partial(api.jvp, f), (args, args), order - 1, fwd_msg)

    if "rev" in modes:
      rev_msg = f'VJP of {err_msg}' if err_msg else 'VJP'
      _check_vjp(f, partial(api.vjp, f), args, err_msg=rev_msg)
      if order > 1:
        def f_vjp(*args):
          out_primal_py, vjp_py = api.vjp(f, *args)
          return vjp_py(out_primal_py)
        _check_grads(f_vjp, args, order - 1, rev_msg)

  _check_grads(f, args, order)
