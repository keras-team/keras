# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities to perform maths on pytrees."""

import functools
import operator
from typing import Any, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax._src import numerics


def tree_add(tree_x: Any, tree_y: Any, *other_trees: Any) -> Any:
  r"""Add two (or more) pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.
    *other_trees: optional other trees to add

  Returns:
    the sum of the two (or more) pytrees.

  .. versionchanged:: 0.2.1
    Added optional ``*other_trees`` argument.
  """
  trees = [tree_x, tree_y, *other_trees]
  return jax.tree.map(lambda *leaves: sum(leaves), *trees)


def tree_sub(tree_x: Any, tree_y: Any) -> Any:
  r"""Subtract two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.

  Returns:
    the difference of the two pytrees.
  """
  return jax.tree.map(operator.sub, tree_x, tree_y)


def tree_mul(tree_x: Any, tree_y: Any) -> Any:
  r"""Multiply two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.

  Returns:
    the product of the two pytrees.
  """
  return jax.tree.map(operator.mul, tree_x, tree_y)


def tree_div(tree_x: Any, tree_y: Any) -> Any:
  r"""Divide two pytrees.

  Args:
    tree_x: first pytree.
    tree_y: second pytree.

  Returns:
    the quotient of the two pytrees.
  """
  return jax.tree.map(operator.truediv, tree_x, tree_y)


def tree_scalar_mul(
    scalar: Union[float, jax.Array],
    tree: Any,
) -> Any:
  r"""Multiply a tree by a scalar.

  In infix notation, the function performs ``out = scalar * tree``.

  Args:
    scalar: scalar value.
    tree: pytree.

  Returns:
    a pytree with the same structure as ``tree``.
  """
  return jax.tree.map(lambda x: scalar * x, tree)


def tree_add_scalar_mul(
    tree_x: Any, scalar: Union[float, jax.Array], tree_y: Any
) -> Any:
  r"""Add two trees, where the second tree is scaled by a scalar.

  In infix notation, the function performs ``out = tree_x + scalar * tree_y``.

  Args:
    tree_x: first pytree.
    scalar: scalar value.
    tree_y: second pytree.

  Returns:
    a pytree with the same structure as ``tree_x`` and ``tree_y``.
  """
  scalar = jnp.asarray(scalar)
  return jax.tree.map(
      lambda x, y: None if x is None else x + scalar.astype(x.dtype) * y,
      tree_x,
      tree_y,
      is_leaf=lambda x: x is None,
  )


_vdot = functools.partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)


def _vdot_safe(a, b):
  return _vdot(jnp.asarray(a), jnp.asarray(b))


def tree_vdot(tree_x: Any, tree_y: Any) -> chex.Numeric:
  r"""Compute the inner product between two pytrees.

  Args:
    tree_x: first pytree to use.
    tree_y: second pytree to use.

  Returns:
    inner product between ``tree_x`` and ``tree_y``, a scalar value.

  Examples:

    >>> optax.tree_utils.tree_vdot(
    ...   {'a': jnp.array([1, 2]), 'b': jnp.array([1, 2])},
    ...   {'a': jnp.array([-1, -1]), 'b': jnp.array([1, 1])},
    ... )
    Array(0, dtype=int32)

  .. note::
    We upcast the values to the highest precision to avoid
    numerical issues.
  """
  vdots = jax.tree.map(_vdot_safe, tree_x, tree_y)
  return jax.tree.reduce(operator.add, vdots, initializer=0)


def tree_sum(tree: Any) -> chex.Numeric:
  """Compute the sum of all the elements in a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  sums = jax.tree.map(jnp.sum, tree)
  return jax.tree.reduce(operator.add, sums, initializer=0)


def tree_max(tree: Any) -> chex.Numeric:
  """Compute the max of all the elements in a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  maxes = jax.tree.map(jnp.max, tree)
  # initializer=-jnp.inf should work but pytype wants a jax.Array.
  return jax.tree.reduce(jnp.maximum, maxes, initializer=jnp.array(-jnp.inf))


def _square(leaf):
  return jnp.square(leaf.real) + jnp.square(leaf.imag)


def tree_l2_norm(tree: Any, squared: bool = False) -> chex.Numeric:
  """Compute the l2 norm of a pytree.

  Args:
    tree: pytree.
    squared: whether the norm should be returned squared or not.

  Returns:
    a scalar value.
  """
  squared_tree = jax.tree.map(_square, tree)
  sqnorm = tree_sum(squared_tree)
  if squared:
    return sqnorm
  else:
    return jnp.sqrt(sqnorm)


def tree_l1_norm(tree: Any) -> chex.Numeric:
  """Compute the l1 norm of a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  abs_tree = jax.tree.map(jnp.abs, tree)
  return tree_sum(abs_tree)


def tree_linf_norm(tree: Any) -> chex.Numeric:
  """Compute the l-infinity norm of a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  abs_tree = jax.tree.map(jnp.abs, tree)
  return tree_max(abs_tree)


def tree_zeros_like(
    tree: Any,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Any:
  """Creates an all-zeros tree with the same structure.

  Args:
    tree: pytree.
    dtype: optional dtype to use for the tree of zeros.

  Returns:
    an all-zeros tree with the same structure as ``tree``.
  """
  return jax.tree.map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)


def tree_ones_like(
    tree: Any,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Any:
  """Creates an all-ones tree with the same structure.

  Args:
    tree: pytree.
    dtype: optional dtype to use for the tree of ones.

  Returns:
    an all-ones tree with the same structure as ``tree``.
  """
  return jax.tree.map(lambda x: jnp.ones_like(x, dtype=dtype), tree)


def tree_full_like(
    tree: Any,
    fill_value: jax.typing.ArrayLike,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Any:
  """Creates an identical tree where all tensors are filled with ``fill_value``.

  Args:
    tree: pytree.
    fill_value: the fill value for all tensors in the tree.
    dtype: optional dtype to use for the tensors in the tree.

  Returns:
    an tree with the same structure as ``tree``.
  """
  return jax.tree.map(lambda x: jnp.full_like(x, fill_value, dtype=dtype), tree)


def tree_clip(
    tree: Any,
    min_value: Optional[jax.typing.ArrayLike],
    max_value: Optional[jax.typing.ArrayLike],
) -> Any:
  """Creates an identical tree where all tensors are clipped to `[min, max]`.

  Args:
    tree: pytree.
    min_value: min value to clip all tensors to.
    max_value: max value to clip all tensors to.

  Returns:
    an tree with the same structure as ``tree``.

  .. versionadded:: 0.2.3
  """
  return jax.tree.map(lambda g: jnp.clip(g, min_value, max_value), tree)


def tree_update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree.map(
      lambda g, t: (
          (1 - decay) * (g**order) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )


def tree_update_infinity_moment(updates, moments, decay, eps):
  """Compute the exponential moving average of the infinity norm."""
  return jax.tree.map(
      lambda g, t: (
          jnp.maximum(jnp.abs(g) + eps, decay * t) if g is not None else g
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )


def tree_update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g**order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return numerics.abs_sq(g) ** half_order

  return jax.tree.map(
      lambda g, t: (
          (1 - decay) * orderth_norm(g) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )


@functools.partial(jax.jit, inline=True)
def tree_bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree.map(lambda t: t / bias_correction_.astype(t.dtype), moment)


def tree_where(condition, tree_x, tree_y):
  """Select tree_x values if condition is true else tree_y values.

  Args:
    condition: boolean specifying which values to select from tree x or tree_y
    tree_x: pytree chosen if condition is True
    tree_y: pytree chosen if condition is False

  Returns:
    tree_x or tree_y depending on condition.
  """
  return jax.tree.map(lambda x, y: jnp.where(condition, x, y), tree_x, tree_y)
