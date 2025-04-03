# Copyright 2021 The JAX Authors.
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

"""
Sparsify transform
==================

This is an experimental JAX transform that will allow arbitrary JAX functions to accept
sparse matrices as inputs, so long as sparse rules are implemented for the primitives
called by the function.

For example:

>>> import jax.numpy as jnp
>>> from jax import random
>>> from jax.experimental.sparse import BCOO, sparsify

>>> mat = random.uniform(random.key(1701), (5, 5))
>>> mat = mat.at[mat < 0.5].set(0)
>>> vec = random.uniform(random.key(42), (5,))

>>> def f(mat, vec):
...   return -(jnp.sin(mat) @ vec)
...
>>> f(mat, vec)
Array([-1.2655463 , -0.52060574, -0.14522289, -0.10817424,
       -0.15574613], dtype=float32)

>>> mat_sparse = BCOO.fromdense(mat)
>>> mat_sparse
BCOO(float32[5, 5], nse=8)

>>> sparsify(f)(mat_sparse, vec)
Array([-1.2655463 , -0.52060574, -0.14522289, -0.10817424,
       -0.15574613], dtype=float32)
"""

from collections.abc import Callable, Sequence
import functools
from typing import Any, NamedTuple

import numpy as np

import jax
from jax import lax
from jax._src import config
from jax._src import core
from jax._src.custom_derivatives import lift_jvp
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import sharding_impls
from jax.experimental.sparse.bcoo import bcoo_multiply_dense, bcoo_multiply_sparse
import jax.numpy as jnp
from jax._src.api_util import flatten_fun_nokwargs
from jax._src.lib import pytree
from jax._src.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from jax.util import safe_map, safe_zip, split_list
from jax._src.lax.control_flow import _check_tree_and_avals
from jax._src.numpy import lax_numpy
from jax.experimental import sparse
from jax.experimental.sparse import BCOO, BCSR

sparse_rules_bcoo : dict[core.Primitive, Callable] = {}
sparse_rules_bcsr : dict[core.Primitive, Callable] = {}

_zero_preserving_linear_unary_primitives = [
  lax.conj_p,
  lax.copy_p,
  lax.imag_p,
  lax.neg_p,
  lax.real_p,
]

_zero_preserving_unary_primitives = [
  lax.abs_p,
  lax.asin_p,
  lax.asinh_p,
  lax.atan_p,
  lax.atanh_p,
  lax.bessel_i1e_p,
  lax.expm1_p,
  lax.log1p_p,
  lax.sign_p,
  lax.sin_p,
  lax.sinh_p,
  lax.sqrt_p,
  lax.square_p,
  lax.tan_p,
  lax.tanh_p,
  lax.convert_element_type_p,
]

_densifying_primitives : list[core.Primitive] = [
  lax.acos_p,
  lax.acosh_p,
  lax.bessel_i0e_p,
  lax.cos_p,
  lax.cosh_p,
  lax.eq_p,
  lax.exp_p,
  lax.ge_p,
  lax.gt_p,
  lax.le_p,
  lax.lt_p,
  lax.log_p,
  lax.ne_p,
  lax.xor_p
]

def _raise_unimplemented_primitive(primitive):
  if primitive in _densifying_primitives:
    raise NotImplementedError(f"sparse rule for {primitive} is not implemented because it "
                              "would result in dense output. If this is your intent, use "
                              "sparse.todense() to convert your arguments to dense matrices.")
  raise NotImplementedError(f"sparse rule for {primitive} is not implemented.")


Array = Any
ArrayOrSparse = Any


class SparsifyEnv:
  """Environment for sparse jaxpr evaluation.

  The environment is essentially a collection of buffers and/or tracers
  that may be shared between one or more SparsifyValue objects, which
  represent sparse or dense arrays via indices into the list of buffers.
  """
  _buffers : list[Array]

  def __init__(self, bufs=()):
    self._buffers = list(bufs)

  def _push(self, arr: Array) -> int:
    self._buffers.append(jnp.asarray(arr))
    return len(self._buffers) - 1

  def data(self, spvalue: SparsifyValue) -> Array:
    """Get the data buffer associated with a SparsifyValue."""
    if spvalue.data_ref is None:
      raise RuntimeError("Internal: requested data from spvalue with data_ref=None")
    return self._buffers[spvalue.data_ref]

  def indices(self, spvalue: SparsifyValue) -> Array:
    """Get the indices buffer associated with a SparsifyValue."""
    if spvalue.indices_ref is None:
      raise RuntimeError("Internal: requested indices from spvalue with indices_ref=None")
    return self._buffers[spvalue.indices_ref]

  def indptr(self, spvalue: SparsifyValue) -> Array:
    """Get the BCSR indptr buffer associated with a SparsifyValue."""
    if spvalue.indptr_ref is None:
      raise RuntimeError("Internal: requested indices from spvalue with indptr_ref=None")
    return self._buffers[spvalue.indptr_ref]

  def dense(self, data):
    """Add a new dense array to the sparsify environment."""
    return SparsifyValue(np.shape(data), self._push(data))

  def sparse(self, shape, data=None, indices=None, indptr=None,
             *, data_ref=None, indices_ref=None, indptr_ref=None,
             indices_sorted=False, unique_indices=False):
    """Add a new sparse array to the sparsify environment."""
    if data is not None:
      assert data_ref is None
      data_ref = self._push(data)
    else:
      assert data_ref is not None and data_ref < len(self._buffers)

    if indices is not None:
      assert indices_ref is None
      indices_ref = self._push(indices)
    else:
      assert indices_ref is not None and indices_ref < len(self._buffers)

    if indptr is not None:
      assert indptr_ref is None
      indptr_ref = self._push(indptr)
    elif indptr_ref is not None:
      assert indptr_ref < len(self._buffers)

    return SparsifyValue(shape, data_ref, indices_ref, indptr_ref,
                         indices_sorted=indices_sorted, unique_indices=unique_indices)


class SparsifyValue(NamedTuple):
  shape: tuple[int, ...]
  data_ref: int | None
  indices_ref: int | None = None
  indptr_ref: int | None = None
  indices_sorted: bool | None = False
  unique_indices: bool | None = False

  @property
  def ndim(self):
    return len(self.shape)

  def is_sparse(self):
    return self.indices_ref is not None

  def is_dense(self):
    return self.indices_ref is None

  def is_bcoo(self):
    return self.is_sparse() and self.indptr_ref is None

  def is_bcsr(self):
    return self.is_sparse() and self.indptr_ref is not None


_is_sparse_obj = lambda arg: isinstance(arg, (BCOO, BCSR))
_is_spvalue = lambda arg: isinstance(arg, SparsifyValue)


def arrays_to_spvalues(
    spenv: SparsifyEnv,
    args: Any
    ) -> Any:
  """Convert a pytree of (sparse) arrays to an equivalent pytree of spvalues."""
  def array_to_spvalue(arg):
    if isinstance(arg, BCOO):
      return spenv.sparse(arg.shape, arg.data, arg.indices,
                          indices_sorted=arg.indices_sorted,
                          unique_indices=arg.unique_indices)
    elif isinstance(arg, BCSR):
      return spenv.sparse(arg.shape, arg.data, arg.indices, arg.indptr,
                          indices_sorted=arg.indices_sorted,
                          unique_indices=arg.unique_indices)
    else:
      return spenv.dense(arg)
  return tree_map(array_to_spvalue, args, is_leaf=_is_sparse_obj)


def spvalues_to_arrays(
    spenv: SparsifyEnv,
    spvalues: Any,
    ) -> Any:
  """Convert a pytree of spvalues to an equivalent pytree of (sparse) arrays."""
  def spvalue_to_array(spvalue):
    if spvalue.is_bcoo():
      return BCOO((spenv.data(spvalue), spenv.indices(spvalue)),
                  shape=spvalue.shape, indices_sorted=spvalue.indices_sorted,
                  unique_indices=spvalue.unique_indices)
    elif spvalue.is_bcsr():
      return BCSR((spenv.data(spvalue), spenv.indices(spvalue), spenv.indptr(spvalue)),
                  shape=spvalue.shape, indices_sorted=spvalue.indices_sorted,
                  unique_indices=spvalue.unique_indices)
    else:
      return spenv.data(spvalue)
  return tree_map(spvalue_to_array, spvalues, is_leaf=_is_spvalue)


def spvalues_to_avals(
    spenv: SparsifyEnv,
    spvalues: Any,
    ) -> Any:
  """Convert a pytree of spvalues to an equivalent pytree of abstract values."""
  def spvalue_to_aval(spvalue):
    data = spenv.data(spvalue)
    return core.ShapedArray(spvalue.shape, data.dtype, data.aval.weak_type)
  return tree_map(spvalue_to_aval, spvalues, is_leaf=_is_spvalue)


# ------------------------------------------------------------------------------
# Implementation of sparsify() using tracers.

class SparseTracer(core.Tracer):
  def __init__(self, trace: core.Trace, *, spvalue):
    self._spvalue = spvalue
    self._trace = trace

  @property
  def spenv(self):
    if not hasattr(self._trace, 'spenv'):
      raise RuntimeError("Internal: trace does not have spenv defined.")
    return self._trace.spenv

  @property
  def aval(self):
    return spvalues_to_avals(self.spenv, [self._spvalue])[0]

  def full_lower(self):
    return self

class SparseTrace(core.Trace):

  def __init__(self, parent_trace, tag, spenv):
    self.parent_trace = parent_trace
    self.tag = tag
    self.spenv = spenv

  def to_sparse_tracer(self, val):
    if isinstance(val, SparseTracer) and self.tag is val._trace.tag:
      return val
    else:
      with core.set_current_trace(self.parent_trace):
        spvalue, = arrays_to_spvalues(self.spenv, [val])
      return SparseTracer(self, spvalue=spvalue)

  def process_primitive(self, primitive, tracers, params):
    tracers = [self.to_sparse_tracer(t) for t in tracers]
    spvalues = [t._spvalue for t in tracers]
    if any(spvalue.is_sparse() for spvalue in spvalues):
      if primitive not in sparse_rules_bcoo:
        _raise_unimplemented_primitive(primitive)
      with core.set_current_trace(self.parent_trace):
        out_spvalues = sparse_rules_bcoo[primitive](self.spenv, *(t._spvalue for t in tracers), **params)
    else:
      out_bufs = primitive.bind_with_trace(self.parent_trace, tuple(self.spenv.data(spvalue) for spvalue in spvalues), params)
      out_spvalues = arrays_to_spvalues(self.spenv, out_bufs if primitive.multiple_results else [out_bufs])
    out_tracers = tuple(SparseTracer(self, spvalue=spvalue) for spvalue in out_spvalues)
    return out_tracers if primitive.multiple_results else out_tracers[0]

  def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params):
    assert False
    spvalues = tuple(t._spvalue for t in tracers)
    in_bufs = self.spenv._buffers
    fun, out_spvalues = sparsify_subtrace(f, self.main, spvalues)
    if any(params['donated_invars']):
      raise NotImplementedError("sparsify does not support donated_invars")
    params = dict(params, donated_invars=tuple(False for buf in in_bufs))
    bufs_out = call_primitive.bind(fun, *in_bufs, **params)
    return [SparseTracer(self, spvalue=spvalue) for spvalue in out_spvalues()]

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
    # TODO(jakevdp): handle the jvp here
    del primitive, jvp, symbolic_zeros
    with core.set_current_trace(self):
      return fun.call_wrapped(*tracers)

@lu.transformation_with_aux2
def sparsify_subtrace(f, store, tag, spenv, spvalues, *bufs):
  with core.take_current_trace() as parent:
    trace = SparseTrace(parent, tag, spenv)
    with core.set_current_trace(trace):
      in_tracers = [SparseTracer(trace, spvalue=spvalue) for spvalue in spvalues]
      outs = f(*in_tracers)
      out_traces = [trace.to_sparse_tracer(out) for out in outs]
      buffers = spenv._buffers
  store.store([out._spvalue for out in out_traces])
  return buffers

def sparsify_fun(wrapped_fun, args: list[ArrayOrSparse]):
  tag = core.TraceTag()
  spenv = SparsifyEnv()
  spvalues = arrays_to_spvalues(spenv, args)
  in_bufs = spenv._buffers
  fun, out_spvalues = sparsify_subtrace(wrapped_fun, tag, spenv, spvalues)
  out_bufs = fun.call_wrapped(*in_bufs)
  spenv = SparsifyEnv(out_bufs)
  return spvalues_to_arrays(spenv, out_spvalues())

def _sparsify_with_tracer(fun):
  """Implementation of sparsify() using tracers."""
  @functools.wraps(fun)
  def _wrapped(*args):
    args_flat, in_tree = tree_flatten(args, is_leaf=_is_sparse_obj)
    wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    out = sparsify_fun(wrapped_fun, args_flat)
    return tree_unflatten(out_tree(), out)
  return _wrapped

# ------------------------------------------------------------------------------
# Implementation of sparsify() using a jaxpr interpreter.

def eval_sparse(
    jaxpr: core.Jaxpr,
    consts: Sequence[Array],  # all consts are dense
    spvalues: Sequence[SparsifyValue],  # mix of sparse and dense pointers into spenv
    spenv: SparsifyEnv,
) -> Sequence[SparsifyValue]:
  env : dict[core.Var, SparsifyValue] = {}

  def read(var: core.Atom) -> SparsifyValue:
    # all literals are dense
    if isinstance(var, core.Literal):
      return spenv.dense(var.val)
    else:
      assert isinstance(var, core.Var)
      return env[var]

  def write_buffer(var: core.Var, a: Array) -> None:
    if isinstance(var, core.DropVar):
      return
    env[var] = spenv.dense(a)

  def write(var: core.Var, a: SparsifyValue) -> None:
    if isinstance(var, core.DropVar):
      return
    assert a is not None
    env[var] = a

  safe_map(write_buffer, jaxpr.constvars, consts)
  safe_map(write, jaxpr.invars, spvalues)

  for eqn in jaxpr.eqns:
    prim = eqn.primitive
    invals = safe_map(read, eqn.invars)
    if any(val.is_bcsr() for val in invals):
      if prim not in sparse_rules_bcsr:
        _raise_unimplemented_primitive(prim)
      out = sparse_rules_bcsr[prim](spenv, *invals, **eqn.params)
    elif any(val.is_bcoo() for val in invals):
      if prim not in sparse_rules_bcoo:
        _raise_unimplemented_primitive(prim)
      out = sparse_rules_bcoo[prim](spenv, *invals, **eqn.params)
    else:
      out_bufs = prim.bind(*(spenv.data(val) for val in invals), **eqn.params)
      out_bufs = out_bufs if prim.multiple_results else [out_bufs]
      out = []
      for buf, outvar in safe_zip(out_bufs, eqn.outvars):
        if isinstance(outvar, core.DropVar):
          out.append(None)
        else:
          out.append(spenv.dense(buf))
    safe_map(write, eqn.outvars, out)

  return safe_map(read, jaxpr.outvars)

def sparsify_raw(f):

  def wrapped(
      spenv: SparsifyEnv, *spvalues: SparsifyValue, **params: Any
  ) -> tuple[Sequence[SparsifyValue], pytree.PyTreeDef]:
    spvalues_flat, in_tree = tree_flatten(spvalues, is_leaf=_is_spvalue)
    in_avals_flat = spvalues_to_avals(spenv, spvalues_flat)
    wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(f, params), in_tree)
    jaxpr, out_avals_flat, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals_flat)
    result = eval_sparse(jaxpr, consts, spvalues_flat, spenv)
    if len(out_avals_flat) != len(result):
      raise Exception("Internal: eval_sparse does not return expected number of arguments. "
                      "Got {result} for avals {out_avals_flat}")
    return result, out_tree()

  return wrapped

def _sparsify_with_interpreter(f):
  """Implementation of sparsify() using jaxpr interpreter."""
  f_raw = sparsify_raw(f)
  @functools.wraps(f)
  def wrapped(*args, **params):
    spenv = SparsifyEnv()
    spvalues = arrays_to_spvalues(spenv, args)
    spvalues_out, out_tree = f_raw(spenv, *spvalues, **params)
    out = spvalues_to_arrays(spenv, spvalues_out)
    return tree_unflatten(out_tree, out)
  return wrapped

def sparsify(f, use_tracer=False):
  """Experimental sparsification transform.

  Examples:

    Decorate JAX functions to make them compatible with :class:`jax.experimental.sparse.BCOO`
    matrices:

    >>> from jax.experimental import sparse

    >>> @sparse.sparsify
    ... def f(M, v):
    ...   return 2 * M.T @ v

    >>> M = sparse.BCOO.fromdense(jnp.arange(12).reshape(3, 4))

    >>> v = jnp.array([3, 4, 2])

    >>> f(M, v)
    Array([ 64,  82, 100, 118], dtype=int32)
  """
  if use_tracer:
    return _sparsify_with_tracer(f)
  else:
    return _sparsify_with_interpreter(f)


# ------------------------------------------------------------------------------
# Sparse rules for various primitives

def _ensure_unique_indices(spenv, spvalue):
  """Return an spvalue representation with deduplicated indices."""
  if spvalue.is_dense() or spvalue.unique_indices:
    return spvalue
  arr = spvalues_to_arrays(spenv, spvalue)
  arr = arr.sum_duplicates(nse=arr.nse, remove_zeros=False)
  return arrays_to_spvalues(spenv, arr)

def _zero_preserving_unary_op(prim, linear):
  def func(spenv, *spvalues, **kwargs):
    assert len(spvalues) == 1
    spvalue = spvalues[0]
    if not linear:
      # For non-linear unary operations, we need to ensure that
      # indices are unique before applying the operator elementwise.
      spvalue = _ensure_unique_indices(spenv, spvalue)
    buf = spenv.data(spvalue)
    buf_out = prim.bind(buf, **kwargs)
    if spvalues[0].is_sparse():
      out_spvalue = spenv.sparse(spvalue.shape, buf_out,
                                 indices_ref=spvalue.indices_ref,
                                 indptr_ref=spvalue.indptr_ref,
                                 indices_sorted=spvalue.indices_sorted,
                                 unique_indices=spvalue.unique_indices)
    else:
      out_spvalue = spenv.dense(buf)
    return (out_spvalue,)
  return func

for _prim in _zero_preserving_unary_primitives:
  sparse_rules_bcoo[_prim] = _zero_preserving_unary_op(_prim, linear=False)
  sparse_rules_bcsr[_prim] = _zero_preserving_unary_op(_prim, linear=False)
for _prim in _zero_preserving_linear_unary_primitives:
  sparse_rules_bcoo[_prim] = _zero_preserving_unary_op(_prim, linear=True)
  sparse_rules_bcsr[_prim] = _zero_preserving_unary_op(_prim, linear=True)

def _standard_sparse_rule(prim, sparse_op):
  def _sparse_rule(spenv, *spvalues, **kwds):
    result = sparse_op(*spvalues_to_arrays(spenv, spvalues), **kwds)
    return arrays_to_spvalues(spenv, result if prim.multiple_results else [result])
  return _sparse_rule

_BCOO_STANDARD_PRIMITIVES = {
  lax.broadcast_in_dim_p: sparse.bcoo_broadcast_in_dim,
  lax.concatenate_p: lambda *a, **k: sparse.bcoo_concatenate(a, **k),
  lax.conv_general_dilated_p: sparse.bcoo_conv_general_dilated,
  lax.dot_general_p: sparse.bcoo_dot_general,
  lax.dynamic_slice_p: lambda *a, **k: sparse.bcoo_dynamic_slice(a[0], a[1:], **k),
  lax.reshape_p: sparse.bcoo_reshape,
  lax.rev_p: sparse.bcoo_rev,
  lax.slice_p: sparse.bcoo_slice,
  lax.squeeze_p: sparse.bcoo_squeeze,
}

for prim, bcoo_impl in _BCOO_STANDARD_PRIMITIVES.items():
  sparse_rules_bcoo[prim] = _standard_sparse_rule(prim, bcoo_impl)

_BCSR_STANDARD_PRIMITIVES = {
  lax.dot_general_p: sparse.bcsr_dot_general,
  lax.broadcast_in_dim_p: sparse.bcsr_broadcast_in_dim,
  lax.concatenate_p: lambda *a, **k: sparse.bcsr_concatenate(a, **k),
}

for prim, bcsr_impl in _BCSR_STANDARD_PRIMITIVES.items():
  sparse_rules_bcsr[prim] = _standard_sparse_rule(prim, bcsr_impl)

def _integer_pow_sparse(spenv, *spvalues, y):
  if y <= 0:
    raise NotImplementedError(f"sparse rule for {lax.integer_pow_p} with non-positive exponent {y} is "
                              "not implemented because it would result in dense output. If this is your "
                              "intent, use sparse.todense() to convert your argument to a dense array.")
  return _zero_preserving_unary_op(lax.integer_pow_p, False)(spenv, *spvalues, y=y)

sparse_rules_bcoo[lax.integer_pow_p] = _integer_pow_sparse
sparse_rules_bcsr[lax.integer_pow_p] = _integer_pow_sparse

def _transpose_sparse(spenv, *spvalues, permutation):
  permutation = tuple(permutation)
  args = spvalues_to_arrays(spenv, spvalues)
  shape = args[0].shape
  mat_transposed = sparse.bcoo_transpose(args[0], permutation=permutation)
  out_shape = tuple(shape[i] for i in permutation)

  n_batch = args[0].indices.ndim - 2
  n_sparse = args[0].indices.shape[-1]
  batch_dims_unchanged = (permutation[:n_batch] == tuple(range(n_batch)))
  dense_dims_unchanged = (permutation[n_batch + n_sparse:] == tuple(range(n_batch + n_sparse, len(shape))))
  sparse_dims_unchanged = (permutation[n_batch:n_batch + n_sparse] == tuple(range(n_batch, n_batch + n_sparse)))

  # Data is unchanged if batch & dense dims are not permuted
  kwds = {}
  if batch_dims_unchanged and dense_dims_unchanged:
    kwds['data_ref'] = spvalues[0].data_ref
  else:
    kwds['data'] = mat_transposed.data

  # Indices unchanged if batch & sparse dims are not permuted
  if batch_dims_unchanged and sparse_dims_unchanged:
    kwds['indices_ref'] = spvalues[0].indices_ref
  else:
    kwds['indices'] = mat_transposed.indices

  kwds['indices_sorted'] = mat_transposed.indices_sorted
  kwds['unique_indices'] = mat_transposed.unique_indices
  spvalue = spenv.sparse(out_shape, **kwds)
  return (spvalue,)

sparse_rules_bcoo[lax.transpose_p] = _transpose_sparse

def _add_sparse(spenv, *spvalues):
  X, Y = spvalues
  out_shape = lax.broadcast_shapes(X.shape, Y.shape)
  if X.is_sparse() and Y.is_sparse():
    if X.shape != Y.shape:
      raise NotImplementedError("Addition between sparse matrices of different shapes.")
    if X.indices_ref == Y.indices_ref:
      out_data = lax.add(spenv.data(X), spenv.data(Y))
      if config.enable_checks.value:
        assert X.indices_sorted == Y.indices_sorted
        assert X.unique_indices == Y.unique_indices
      out_spvalue = spenv.sparse(X.shape, out_data, indices_ref=X.indices_ref,
                                 indices_sorted=X.indices_sorted,
                                 unique_indices=X.unique_indices)
    elif spenv.indices(X).ndim != spenv.indices(Y).ndim or spenv.data(X).ndim != spenv.data(Y).ndim:
      raise NotImplementedError("Addition between sparse matrices with different batch/dense dimensions.")
    else:
      out_indices = lax.concatenate([spenv.indices(X), spenv.indices(Y)], dimension=spenv.indices(X).ndim - 2)
      out_data = lax.concatenate([spenv.data(X), spenv.data(Y)], dimension=spenv.indices(X).ndim - 2)
      out_spvalue = spenv.sparse(X.shape, out_data, out_indices)
  else:
    if Y.is_sparse():
      X, Y = Y, X
    assert X.is_sparse() and Y.is_dense()
    if Y.shape != out_shape:
      raise NotImplementedError(
        "Addition between a sparse array X and a dense array Y is not implemented when "
        "the output shape is larger than Y.shape. This is to prevent silent densification "
        "of a large sparse array. If this is your intent, you can explicitly cast the sparse "
        "array to a dense matrix.")
    X_promoted, Y_promoted = spvalues_to_arrays(spenv, (X, Y))
    out = X_promoted.todense() + Y_promoted
    out_spvalue = spenv.dense(out)

  return (out_spvalue,)

sparse_rules_bcoo[lax.add_p] = _add_sparse

def _sub_sparse(spenv, *spvalues):
  X, Y = spvalues
  if X.is_sparse() and Y.is_sparse():
    return _add_sparse(spenv, X, *sparse_rules_bcoo[lax.neg_p](spenv, Y))
  else:
    raise NotImplementedError("Subtraction between sparse and dense array.")

sparse_rules_bcoo[lax.sub_p] = _sub_sparse

def _mul_sparse(spenv, *spvalues):
  X, Y = spvalues
  if X.is_sparse() and Y.is_sparse():
    if X.indices_ref == Y.indices_ref and X.unique_indices:
      if config.enable_checks.value:
        assert X.indices_sorted == Y.indices_sorted
        assert X.unique_indices == Y.unique_indices
      out_data = lax.mul(spenv.data(X), spenv.data(Y))
      out_spvalue = spenv.sparse(X.shape, out_data, indices_ref=X.indices_ref,
                                 indices_sorted=X.indices_sorted,
                                 unique_indices=True)
    else:
      X_promoted, Y_promoted = spvalues_to_arrays(spenv, spvalues)
      mat = bcoo_multiply_sparse(X_promoted, Y_promoted)
      out_spvalue = spenv.sparse(mat.shape, mat.data, mat.indices)
  else:
    if Y.is_sparse():
      X, Y = Y, X
    X_promoted = spvalues_to_arrays(spenv, X)
    out_data = bcoo_multiply_dense(X_promoted, spenv.data(Y))
    out_spvalue = spenv.sparse(X.shape, out_data, indices_ref=X.indices_ref,
                               indices_sorted=X.indices_sorted,
                               unique_indices=X.unique_indices)

  return (out_spvalue,)

sparse_rules_bcoo[lax.mul_p] = _mul_sparse

def _div_sparse(spenv, *spvalues):
  X, Y = spvalues
  if Y.is_sparse():
    raise NotImplementedError(
      "Division by a sparse array is not implemented because it "
      "would result in dense output. If this is your intent, use "
      "sparse.todense() to convert your arguments to a dense array.")
  X_promoted = spvalues_to_arrays(spenv, X)
  out_data = bcoo_multiply_dense(X_promoted, 1. / spenv.data(Y))
  out_spvalue = spenv.sparse(X.shape, out_data, indices_ref=X.indices_ref,
                              indices_sorted=X.indices_sorted,
                              unique_indices=X.unique_indices)
  return (out_spvalue,)

sparse_rules_bcoo[lax.div_p] = _div_sparse

def _reduce_sum_sparse(spenv, *spvalues, axes):
  X, = spvalues
  X_promoted = spvalues_to_arrays(spenv, X)
  mat = sparse.bcoo_reduce_sum(X_promoted, axes=axes)
  out_shape = mat.shape
  if out_shape == ():
    out_spvalue = spenv.dense(mat.data.sum())
  else:
    out_spvalue = spenv.sparse(out_shape, mat.data, mat.indices)
  return (out_spvalue,)

sparse_rules_bcoo[lax.reduce_sum_p] = _reduce_sum_sparse


def _gather_sparse_rule(spenv, *args, dimension_numbers, slice_sizes, unique_indices,
                        indices_are_sorted, mode, fill_value):
  operand, start_indices = spvalues_to_arrays(spenv, args)
  result = sparse.bcoo_gather(operand, start_indices, dimension_numbers=dimension_numbers,
                              slice_sizes=slice_sizes, unique_indices=unique_indices,
                              indices_are_sorted=indices_are_sorted,
                              mode=mode, fill_value=fill_value)
  return arrays_to_spvalues(spenv, (result,))

sparse_rules_bcoo[lax.gather_p] = _gather_sparse_rule

def _sparsify_jaxpr(spenv, jaxpr, *spvalues):
  # TODO(jakevdp): currently this approach discards all information about
  #   shared data & indices when generating the sparsified jaxpr. The
  #   current approach produces valid sparsified while loops, but they
  #   don't work in corner cases (see associated TODO in sparsify_test.py)
  out_tree: pytree.PyTreeDef | None = None

  @lu.wrap_init
  def wrapped(*args_flat):
    # TODO(frostig,jakevdp): This closes over `spenv`, which can bring
    # in buffers from the "outer scope" as constants. Is this a
    # problem for primitives like cond and while_loop, which always
    # convert constvars to invars when staging out their subjaxprs?
    nonlocal out_tree
    args = tree_unflatten(in_tree, args_flat)
    spvalues = arrays_to_spvalues(spenv, args)
    result = eval_sparse(jaxpr.jaxpr, jaxpr.consts, spvalues, spenv)
    out = spvalues_to_arrays(spenv, result)
    out_flat, out_tree = tree_flatten(out)
    return out_flat

  args = spvalues_to_arrays(spenv, spvalues)
  args_flat, in_tree = tree_flatten(args)
  avals_flat = [core.get_aval(arg) for arg in args_flat]
  sp_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(wrapped, avals_flat)
  sp_jaxpr = pe.ClosedJaxpr(sp_jaxpr, consts)
  assert out_tree is not None
  return sp_jaxpr, out_tree

def _while_sparse(spenv, *spvalues, cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts):
  cond_const_spvalues, body_const_spvalues, init_val_spvalues = split_list(
    spvalues, [cond_nconsts, body_nconsts])

  cond_sp_jaxpr, _ = _sparsify_jaxpr(spenv, cond_jaxpr, *cond_const_spvalues, *init_val_spvalues)
  body_sp_jaxpr, out_tree = _sparsify_jaxpr(spenv, body_jaxpr, *body_const_spvalues, *init_val_spvalues)

  cond_consts, _ = tree_flatten(spvalues_to_arrays(spenv, cond_const_spvalues))
  body_consts, _ = tree_flatten(spvalues_to_arrays(spenv, body_const_spvalues))
  init_vals, _ = tree_flatten(spvalues_to_arrays(spenv, init_val_spvalues))

  out_flat = lax.while_p.bind(*cond_consts, *body_consts, *init_vals,
                              cond_nconsts=len(cond_consts), cond_jaxpr=cond_sp_jaxpr,
                              body_nconsts=len(body_consts), body_jaxpr=body_sp_jaxpr)
  return arrays_to_spvalues(spenv, tree_unflatten(out_tree, out_flat))

sparse_rules_bcoo[lax.while_p] = _while_sparse


def _pjit_sparse(spenv, *spvalues, jaxpr, in_shardings, out_shardings,
                 in_layouts, out_layouts, resource_env, donated_invars, name,
                 keep_unused, inline, compiler_options_kvs):
  if any(donated_invars):
    raise NotImplementedError("sparse xla_call with donated_invars")

  sp_call_jaxpr, out_tree = _sparsify_jaxpr(spenv, jaxpr, *spvalues)
  args_flat, _ = tree_flatten(spvalues_to_arrays(spenv, spvalues))
  donated_invars = tuple(False for arg in args_flat)

  # TODO(yashkatariya, vanderplas): Flatten twice and set the correct sharding
  # for data and indices.
  in_shardings = in_shardings + tuple(
      sharding_impls.UNSPECIFIED
      for _ in range(len(args_flat) - len(in_shardings))
  )
  out_shardings = out_shardings + tuple(
      sharding_impls.UNSPECIFIED
      for _ in range(len(sp_call_jaxpr.out_avals) - len(out_shardings))
  )
  in_layouts = in_layouts + tuple(
      None for _ in range(len(args_flat) - len(in_layouts))
  )
  out_layouts = out_layouts + tuple(
      None for _ in range(len(sp_call_jaxpr.out_avals) - len(out_layouts))
  )

  out_flat = pjit.pjit_p.bind(
      *args_flat,
      jaxpr=sp_call_jaxpr,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      in_layouts=in_layouts,
      out_layouts=out_layouts,
      resource_env=resource_env,
      donated_invars=donated_invars,
      name=name,
      keep_unused=keep_unused,
      inline=inline,
      compiler_options_kvs=compiler_options_kvs)
  return arrays_to_spvalues(spenv, tree_unflatten(out_tree, out_flat))

sparse_rules_bcoo[pjit.pjit_p] = _pjit_sparse


def _duplicate_for_sparse_spvalues(spvalues, params):
  for spvalue, param in safe_zip(spvalues, params):
    yield from [param, param] if spvalue.is_sparse() else [param]

def _scan_sparse(spenv, *spvalues, jaxpr, num_consts, num_carry, **params):
  const_spvalues, carry_spvalues, xs_spvalues = split_list(
    spvalues, [num_consts, num_carry])
  if xs_spvalues:
    # TODO(jakevdp): we don't want to pass xs_spvalues, we want to pass one row
    # of xs spvalues. How to do this?
    raise NotImplementedError("sparse rule for scan with x values.")
  sp_jaxpr, _ = _sparsify_jaxpr(spenv, jaxpr, *const_spvalues, *carry_spvalues, *xs_spvalues)

  consts, _ = tree_flatten(spvalues_to_arrays(spenv, const_spvalues))
  carry, carry_tree = tree_flatten(spvalues_to_arrays(spenv, carry_spvalues))
  xs, xs_tree = tree_flatten(spvalues_to_arrays(spenv, xs_spvalues))

  # params['linear'] has one entry per arg; expand it to match the sparsified args.
  const_linear, carry_linear, xs_linear = split_list(
    params.pop('linear'), [num_consts, num_carry])
  sp_linear = (
    *_duplicate_for_sparse_spvalues(const_spvalues, const_linear),
    *_duplicate_for_sparse_spvalues(carry_spvalues, carry_linear),
    *_duplicate_for_sparse_spvalues(xs_spvalues, xs_linear))

  out = lax.scan_p.bind(*consts, *carry, *xs, jaxpr=sp_jaxpr, linear=sp_linear,
                        num_consts=len(consts), num_carry=len(carry), **params)
  carry_out = tree_unflatten(carry_tree, out[:len(carry)])
  xs_out = tree_unflatten(xs_tree, out[len(carry):])
  return arrays_to_spvalues(spenv, carry_out + xs_out)

sparse_rules_bcoo[lax.scan_p] = _scan_sparse

def _cond_sparse(spenv, pred, *operands, branches, **params):
  sp_branches, treedefs = zip(*(_sparsify_jaxpr(spenv, jaxpr, *operands)
                                for jaxpr in branches))
  _check_tree_and_avals("sparsified true_fun output",
                        treedefs[0], sp_branches[0].out_avals,
                        "sparsified false_fun output",
                        treedefs[1], sp_branches[1].out_avals)
  args, _ = tree_flatten(spvalues_to_arrays(spenv, (pred, *operands)))
  out_flat = lax.cond_p.bind(*args, branches=sp_branches, **params)
  out = tree_unflatten(treedefs[0], out_flat)
  return arrays_to_spvalues(spenv, out)

sparse_rules_bcoo[lax.cond_p] = _cond_sparse

def _todense_sparse_rule(spenv, spvalue, *, tree):
  del tree  # TODO(jakvdp): we should assert that tree is PytreeDef(*)
  out = spvalues_to_arrays(spenv, spvalue).todense()
  return (spenv.dense(out),)

sparse_rules_bcoo[sparse.todense_p] = _todense_sparse_rule
sparse_rules_bcsr[sparse.todense_p] = _todense_sparse_rule

def _custom_jvp_sparse_rule(spenv, *spvalues, **params):
  call_jaxpr = params.pop('call_jaxpr')
  jvp_jaxpr_thunk = params.pop('jvp_jaxpr_thunk')
  num_consts = params.pop('num_consts')
  sp_call_jaxpr, out_tree = _sparsify_jaxpr(spenv, call_jaxpr, *spvalues)
  @lu.wrap_init
  def fun(*arrs):
    sparrs = arrays_to_spvalues(spenv, arrs)
    out = eval_sparse(call_jaxpr.jaxpr, call_jaxpr.consts, sparrs, spenv)
    return spvalues_to_arrays(spenv, out)
  jvp = lift_jvp(num_consts, jvp_jaxpr_thunk)
  invals = spvalues_to_arrays(spenv, spvalues)
  outvals = jax.custom_derivatives.custom_jvp_call_p.bind(fun, jvp, *invals, **params)
  return arrays_to_spvalues(spenv, outvals)

sparse_rules_bcoo[jax.custom_derivatives.custom_jvp_call_p] = _custom_jvp_sparse_rule
sparse_rules_bcsr[jax.custom_derivatives.custom_jvp_call_p] = _custom_jvp_sparse_rule


# ------------------------------------------------------------------------------
# BCOO methods derived from sparsify
# defined here to avoid circular imports

def _sum(self, *args, **kwargs):
  """Sum array along axis."""
  return sparsify(lambda x: x.sum(*args, **kwargs))(self)

def _reshape(self, *args, **kwargs):
  """Returns an array containing the same data with a new shape."""
  return sparsify(lambda x: x.reshape(*args, **kwargs))(self)

def _astype(self, *args, **kwargs):
  """Copy the array and cast to a specified dtype."""
  return sparsify(lambda x: x.astype(*args, **kwargs))(self)

def _bcoo_rewriting_take(arr, idx, indices_are_sorted=False, unique_indices=False,
                           mode=None, fill_value=None):
  # Only sparsify the array argument; sparse indices not yet supported
  result = sparsify(functools.partial(
    lax_numpy._rewriting_take, idx=idx, indices_are_sorted=indices_are_sorted,
    mode=mode, unique_indices=unique_indices, fill_value=fill_value))(arr)
  # Account for a corner case in the rewriting_take implementation.
  if not isinstance(result, BCOO) and np.size(result) == 0:
    result = BCOO.fromdense(result)
  return result

def _sparse_iter(arr):
  return iter(arr[i] for i in range(arr.shape[0]))

_swap_args = lambda f: lambda a, b: f(b, a)

_bcoo_methods = {
  "astype": _astype,
  "reshape": _reshape,
  "sum": _sum,
  "__abs__": sparsify(jnp.abs),
  "__neg__": sparsify(jnp.negative),
  "__pos__": sparsify(jnp.positive),
  "__matmul__": sparsify(jnp.matmul),
  "__rmatmul__": sparsify(_swap_args(jnp.matmul)),
  "__mul__": sparsify(jnp.multiply),
  "__rmul__": sparsify(_swap_args(jnp.multiply)),
  "__truediv__": sparsify(jnp.divide),
  "__rtruediv__": sparsify(_swap_args(jnp.divide)),
  "__add__": sparsify(jnp.add),
  "__radd__": sparsify(_swap_args(jnp.add)),
  "__sub__": sparsify(jnp.subtract),
  "__rsub__": sparsify(_swap_args(jnp.subtract)),
  "__pow__": lambda x, y: sparsify(lambda x: jnp.power(x, y))(x),
  "__rpow__": sparsify(_swap_args(jnp.power)),
  "__getitem__": _bcoo_rewriting_take,
  "__iter__": _sparse_iter,
  "__gt__": sparsify(jnp.greater),
  "__ge__": sparsify(jnp.greater_equal),
  "__lt__": sparsify(jnp.less),
  "__le__": sparsify(jnp.less_equal),
  "__eq__": sparsify(jnp.equal),
  "__ne__": sparsify(jnp.not_equal),
}

for method, impl in _bcoo_methods.items():
  setattr(BCOO, method, impl)

# ------------------------------------------------------------------------------
# BCSR methods derived from sparsify
# defined here to avoid circular imports

def _bcsr_rewriting_take(arr, idx, indices_are_sorted=False, unique_indices=False,
                           mode=None, fill_value=None):
  # Only sparsify the array argument; sparse indices not yet supported
  result = sparsify(functools.partial(
    lax_numpy._rewriting_take, idx=idx, indices_are_sorted=indices_are_sorted,
    mode=mode, unique_indices=unique_indices, fill_value=fill_value))(arr)
  return result

_bcoo_methods = {
  "__matmul__": sparsify(jnp.matmul),
  "__rmatmul__": sparsify(_swap_args(jnp.matmul)),
  "__getitem__": _bcsr_rewriting_take,
}

for method, impl in _bcoo_methods.items():
  setattr(BCSR, method, impl)
