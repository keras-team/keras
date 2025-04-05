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

from collections.abc import Callable, Sequence
import itertools
from typing import Any

import jax
from jax._src import core
from jax import tree_util
from jax._src.api_util import _ensure_index, _ensure_index_tuple
from jax.util import safe_zip
from jax._src.util import split_list, wraps
from jax._src.traceback_util import api_boundary
from jax.experimental.sparse._base import JAXSparse


is_sparse = lambda x: isinstance(x, JAXSparse)


def flatten_fun_for_sparse_ad(fun, argnums: int | tuple[int, ...], args: tuple[Any, ...]):
  argnums_tup = _ensure_index_tuple(argnums)
  assert all(0 <= argnum < len(args) for argnum in argnums_tup)

  # We do a two-step flattening to figure out how argnums maps to args_flat.
  # First, flatten arguments to a list containing sparse and dense objects.
  args_flat1, tree1 = tree_util.tree_flatten(args, is_leaf=is_sparse)
  *leaf_argnums1, end = split_list(range(tree1.num_leaves),
                                   [child.num_leaves for child in tree1.children()])
  assert not end
  argnums_flat1 = list(itertools.chain.from_iterable(
      nums for i, nums in enumerate(leaf_argnums1) if i in argnums_tup))

  # Next, fully flatten to a list of dense buffers.
  args_flat, tree2 = tree_util.tree_flatten(args_flat1)
  *leaf_argnums2, end = split_list(range(tree2.num_leaves),
                                   [child.num_leaves for child in tree2.children()])
  assert not end
  # For sparse args, we only mark the first buffer (the data) for differentiation.
  leaf_argnums2 = [nums[:1] if is_sparse(arg) else nums
                   for arg, nums in safe_zip(args_flat1, leaf_argnums2)]
  argnums_flat = tuple(itertools.chain.from_iterable(
      nums for i, nums in enumerate(leaf_argnums2) if i in argnums_flat1))

  def fun_flat(*args_flat, **kwargs):
    args = tree_util.tree_unflatten(tree1, tree_util.tree_unflatten(tree2, args_flat))
    return fun(*args, **kwargs)

  def reconstruct(i, grad_out):
    bufs, tree = tree_util.tree_flatten(args_flat1[i])
    f_recons = lambda g: tree_util.tree_unflatten(tree, [g, *bufs[1:]])
    for _ in range(grad_out.ndim - bufs[0].ndim):
      f_recons = jax.vmap(f_recons)
    return f_recons(grad_out)

  def postprocess_gradients(grads_out):
    leaf_grads = [None] * tree1.num_leaves
    for i, grad in safe_zip(argnums_flat1, grads_out):
      leaf_grads[i] = reconstruct(i, grad)
    grad_tree = tree_util.tree_unflatten(tree1, leaf_grads)
    grad_tree = tuple(filter(lambda x: jax.tree.leaves(x), grad_tree))
    return grad_tree[0] if len(grad_tree) == 1 else grad_tree

  return fun_flat, argnums_flat, args_flat, postprocess_gradients


def value_and_grad(fun: Callable, argnums: int | Sequence[int] = 0,
                   has_aux=False, **kwargs) -> Callable[..., tuple[Any, Any]]:
  """Sparse-aware version of :func:`jax.value_and_grad`

  Arguments and return values are the same as :func:`jax.value_and_grad`, but when
  taking the gradient with respect to a :class:`jax.experimental.sparse` array, the
  gradient is computed in the subspace defined by the array's sparsity pattern.

  Examples:

    >>> from jax.experimental import sparse
    >>> X = sparse.BCOO.fromdense(jnp.arange(6.))
    >>> y = jnp.ones(6)
    >>> sparse.value_and_grad(lambda X, y: X @ y)(X, y)
    (Array(15., dtype=float32), BCOO(float32[6], nse=5))
  """
  raw_value_and_grad_fun = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux, **kwargs)
  argnums = core.concrete_or_error(_ensure_index, argnums)

  @wraps(fun, docstr=raw_value_and_grad_fun.__doc__, argnums=argnums)
  @api_boundary
  def value_and_grad_fun(*args, **kwargs):
    fun_flat, argnums_flat, args_flat, postprocess_gradients = flatten_fun_for_sparse_ad(fun, argnums, args)
    val_out, grad_out = jax.value_and_grad(fun_flat, argnums=argnums_flat, has_aux=has_aux, **kwargs)(*args_flat)
    return val_out, postprocess_gradients(grad_out)
  return value_and_grad_fun


def grad(fun: Callable, argnums: int | Sequence[int] = 0,
         has_aux=False, **kwargs) -> Callable:
  """Sparse-aware version of :func:`jax.grad`

  Arguments and return values are the same as :func:`jax.grad`, but when taking
  the gradient with respect to a :class:`jax.experimental.sparse` array, the
  gradient is computed in the subspace defined by the array's sparsity pattern.

  Examples:

    >>> from jax.experimental import sparse
    >>> X = sparse.BCOO.fromdense(jnp.arange(6.))
    >>> y = jnp.ones(6)
    >>> sparse.grad(lambda X, y: X @ y)(X, y)
    BCOO(float32[6], nse=5)
  """
  raw_grad_fun = jax.grad(fun, argnums=argnums, **kwargs)
  argnums = core.concrete_or_error(_ensure_index, argnums)

  @wraps(fun, docstr=raw_grad_fun.__doc__, argnums=argnums)
  @api_boundary
  def grad_fun(*args, **kwargs):
    fun_flat, argnums_flat, args_flat, postprocess_gradients = flatten_fun_for_sparse_ad(fun, argnums, args)
    out = jax.grad(fun_flat, argnums=argnums_flat, has_aux=has_aux, **kwargs)(*args_flat)
    if has_aux:
      return postprocess_gradients(out[0]), out[1]
    return postprocess_gradients(out)
  return grad_fun


def jacfwd(fun: Callable, argnums: int | Sequence[int] = 0,
           has_aux: bool = False, **kwargs) -> Callable:
  """Sparse-aware version of :func:`jax.jacfwd`

  Arguments and return values are the same as :func:`jax.jacfwd`, but when taking
  the gradient with respect to a :class:`jax.experimental.sparse` array, the
  gradient is computed in the subspace defined by the array's sparsity pattern.
  Currently this is only implemented for dense outputs.
  """
  raw_jacfwd_fun = jax.jacfwd(fun, argnums=argnums, **kwargs)
  argnums = core.concrete_or_error(_ensure_index, argnums)

  @wraps(fun, docstr=raw_jacfwd_fun.__doc__, argnums=argnums)
  @api_boundary
  def jacfwd_fun(*args, **kwargs):
    fun_flat, argnums_flat, args_flat, postprocess_gradients = flatten_fun_for_sparse_ad(fun, argnums, args)
    out = jax.jacfwd(fun_flat, argnums=argnums_flat, has_aux=has_aux, **kwargs)(*args_flat)
    if has_aux:
      return postprocess_gradients(out[0]), out[1]
    return postprocess_gradients(out)
  return jacfwd_fun


def jacrev(fun: Callable, argnums: int | Sequence[int] = 0,
           has_aux: bool = False, **kwargs) -> Callable:
  """Sparse-aware version of :func:`jax.jacrev`

  Arguments and return values are the same as :func:`jax.jacrev`, but when taking
  the gradient with respect to a :class:`jax.experimental.sparse` array, the
  gradient is computed in the subspace defined by the array's sparsity pattern.
  Currently this is only implemented for dense outputs.
  """
  raw_jacrev_fun = jax.jacrev(fun, argnums=argnums, **kwargs)
  argnums = core.concrete_or_error(_ensure_index, argnums)

  @wraps(fun, docstr=raw_jacrev_fun.__doc__, argnums=argnums)
  @api_boundary
  def jacrev_fun(*args, **kwargs):
    fun_flat, argnums_flat, args_flat, postprocess_gradients = flatten_fun_for_sparse_ad(fun, argnums, args)
    out = jax.jacrev(fun_flat, argnums=argnums_flat, has_aux=has_aux, **kwargs)(*args_flat)
    if has_aux:
      return postprocess_gradients(out[0]), out[1]
    return postprocess_gradients(out)
  return jacrev_fun


jacobian = jacrev
