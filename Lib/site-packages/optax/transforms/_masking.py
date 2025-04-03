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
"""Wrappers that mask out part of the parameters when applying a transform."""

from collections.abc import Callable
from typing import Any, NamedTuple, Union

import jax
from optax._src import base
from optax.tree_utils import _state_utils


class MaskedState(NamedTuple):
  """Maintains inner transform state for masked transformations."""

  inner_state: Any


class MaskedNode(NamedTuple):
  """A node used to mask out unspecified parts of a tree.

  This node is ignored when mapping functions across the tree e.g. using
  `jax.tree.map` since it is a container without children. It can
  therefore be used to mask out parts of a tree.
  """


def _mask_callable(
    mask: Union[base.PyTree, Callable[[base.Params], base.PyTree]],
):
  callable_leaves = jax.tree.leaves(jax.tree.map(callable, mask))
  return (len(callable_leaves) > 0) and all(callable_leaves)  # pylint:disable=g-explicit-length-test


def masked(
    inner: base.GradientTransformation,
    mask: Union[base.PyTree, Callable[[base.Params], base.PyTree]],
    *,
    mask_compatible_extra_args: bool = False,
) -> base.GradientTransformationExtraArgs:
  """Mask updates so only some are transformed, the rest are passed through.

  For example, it is common to skip weight decay for BatchNorm scale and all
  bias parameters. Since in many networks, these are the only 1D parameters,
  you may for instance create a mask function to mask them out as follows::

    mask_fn = lambda p: jax.tree.map(lambda x: x.ndim != 1, p)
    weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask_fn)

  You may alternatively create the mask pytree upfront::

    mask = jax.tree.map(lambda x: x.ndim != 1, params)
    weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask)

  For the ``inner`` transform, state will only be stored for the parameters that
  have a mask value of ``True``.

  Note that, when using ``tree_map_params``, it may be required to pass the
  argument `is_leaf=lambda v: isinstance(v, optax.MaskedNode)`, if the tree
  map needs to take additional arguments with the same shape as the original
  input tree.

  Args:
    inner: Inner transformation to mask.
    mask: a PyTree with same structure as (or a prefix of) the params PyTree, or
      a Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, ``True`` for leaves/subtrees you want to apply the
      transformation to, and ``False`` for those you want to skip. The mask must
      be static for the gradient transformation to be jit-compilable.
    mask_compatible_extra_args: whether to also apply the same masking to
      extra_arg fields with the same tree structure as params/updates.

  Returns:
    New :class:`optax.GradientTransformationExtraArgs` wrapping ``inner``.
  """
  inner = base.with_extra_args_support(inner)

  def mask_pytree(pytree, mask_tree):
    return jax.tree.map(
        lambda m, p: p if m else MaskedNode(), mask_tree, pytree
    )

  # It is possible that `extra_args` of update_fn has pytrees with the same
  # structure as params/updates, e.g. parameter tags. This function applies
  # the mask to those pytrees.
  def maybe_mask_values(pytree_dict, base_pytree, mask_tree):
    base_structure = jax.tree.structure(base_pytree)

    def _maybe_mask(pytree):
      if mask_compatible_extra_args and (
          jax.tree.structure(pytree) == base_structure
      ):
        return mask_pytree(pytree, mask_tree)
      else:
        return pytree

    return {k: _maybe_mask(v) for k, v in pytree_dict.items()}

  def init_fn(params):
    # This is a workaround to make tree_map_params work with masking.
    # The API of `masked` takes a mask on construction, instead of at init.
    # This means that this gradient transformation can only work for parameter
    # trees that match the shape of the mask. Technically this breaks the API
    # of optax, and this causes tree_map_params to break. This is because
    # tree_map_params calls init with a placeholder in order to detect copies
    # of the parameter tree. As a (slightly ugly) workaround, we detect when
    # the init is being called by tree_map_params, and pass the placeholder
    # down without masking. This is safe, since tree_map_params does not impose
    # any particular constraints on the shape of the parameter tree, as long
    # as tree_map_params is being called on a tree with the correct structure.
    # See wrappers_test for proof that this works!
    if isinstance(params, _state_utils._ParamsPlaceholder):  # pylint:disable=protected-access
      return MaskedState(inner_state=inner.init(params))

    mask_tree = mask(params) if _mask_callable(mask) else mask
    masked_params = mask_pytree(params, mask_tree)
    return MaskedState(inner_state=inner.init(masked_params))

  def update_fn(updates, state, params=None, **extra_args):
    mask_tree = mask(updates) if _mask_callable(mask) else mask
    masked_extra_args = maybe_mask_values(extra_args, updates, mask_tree)
    masked_updates = mask_pytree(updates, mask_tree)
    masked_params = None if params is None else mask_pytree(params, mask_tree)

    new_masked_updates, new_inner_state = inner.update(
        masked_updates, state.inner_state, masked_params, **masked_extra_args
    )

    new_updates = jax.tree.map(
        lambda m, new_u, old_u: new_u if m else old_u,
        mask_tree,
        new_masked_updates,
        updates,
    )
    return new_updates, MaskedState(inner_state=new_inner_state)

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
