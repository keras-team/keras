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
"""Gradient clipping transformations.

Note that complex numbers are also supported, see
https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
"""

import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import linear_algebra
from optax._src import numerics


def clip(max_delta: chex.Numeric) -> base.GradientTransformation:
  """Clips updates element-wise, to be in ``[-max_delta, +max_delta]``.

  Args:
    max_delta: The maximum absolute value for each element in the update.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params
    return otu.tree_clip(updates, -max_delta, max_delta), state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def clip_by_block_rms(threshold: float) -> base.GradientTransformation:
  """Clips updates to a max rms for the gradient of each param vector or matrix.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    threshold: The maximum rms for the gradient of each param vector or matrix.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params

    def _clip_fn(u):
      clip_denom = jnp.maximum(
          1.0, jnp.sqrt(jnp.mean(numerics.abs_sq(u))) / threshold
      )
      return u / clip_denom

    updates = jax.tree.map(_clip_fn, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def clip_by_global_norm(max_norm: float) -> base.GradientTransformation:
  """Clips updates using their global norm.

  Args:
    max_norm: The maximum global norm for an update.

  Returns:
    A :class:`optax.GradientTransformation` object.

  References:
    Pascanu et al., `On the difficulty of training Recurrent Neural Networks
    <https://arxiv.org/abs/1211.5063>`_, 2012
  """

  def update_fn(updates, state, params=None):
    del params
    g_norm = linear_algebra.global_norm(updates)
    # TODO(b/163995078): revert back to the following (faster) implementation
    # once analyzed how it affects backprop through update (e.g. meta-gradients)
    # g_norm = jnp.maximum(max_norm, g_norm)
    # updates = jax.tree.map(lambda t: (t / g_norm) * max_norm, updates)
    trigger = jnp.squeeze(g_norm < max_norm)
    chex.assert_shape(trigger, ())  # A scalar.

    def clip_fn(t):
      return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_norm)

    updates = jax.tree.map(clip_fn, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def _check_arrays_have_batch_dim(grads: chex.ArrayTree) -> bool:
  """Checks that each array in grads has a batch dimension in the 0th axis."""
  grads = jax.tree.flatten(grads)[0]
  batch_size = grads[0].shape[0]
  return all(g.ndim >= 1 and batch_size == g.shape[0] for g in grads)


def per_example_global_norm_clip(
    grads: chex.ArrayTree, l2_norm_clip: float
) -> tuple[chex.ArrayTree, jax.Array]:
  """Applies gradient clipping per-example using their global norm.

  Args:
    grads: flattened update; the function expects each array in this list to
      have a batch dimension on the 0th axis.
    l2_norm_clip: maximum L2 norm of the per-example gradients.

  Returns:
    A tuple containing sum of the clipped per-example grads, and the number of
    per-example grads that were clipped.

  Example:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> grads = [jnp.array([[0, 0, 0], [0, 3, 4], [4, 0, 3], [3, 4, 0]])]
    >>> optax.per_example_global_norm_clip(grads, jnp.inf)
    ([Array([7., 7., 7.], dtype=float32)], Array(0, dtype=int32))
    >>> optax.per_example_global_norm_clip(grads, 0.0)
    ([Array([0., 0., 0.], dtype=float32)], Array(3, dtype=int32))
    >>> optax.per_example_global_norm_clip(grads, 1.25)
    ([Array([1.75, 1.75, 1.75], dtype=float32)], Array(3, dtype=int32))

  References:
    Abadi et al., `Deep Learning with Differential Privacy
    <https://arxiv.org/abs/1607.00133>`_, 2016

  .. seealso::
    :func:`optax.contrib.differentially_private_aggregate` for more realistic
    example usages.
  """

  if not _check_arrays_have_batch_dim(grads):
    raise ValueError(
        'Unlike other transforms, `per_example_global_norm_clip` expects'
        ' `grads` to have a batch dimension in the 0th axis.'
    )

  global_grad_norms = jax.vmap(linear_algebra.global_norm)(grads)
  multipliers = jnp.nan_to_num(
      jnp.minimum(l2_norm_clip / global_grad_norms, 1.0), nan=1.0
  )
  num_clipped = jnp.sum(multipliers < 1.0)
  clipped_sum = jax.tree.map(
      lambda g: jnp.tensordot(multipliers, g, axes=1), grads
  )
  return clipped_sum, num_clipped


def per_example_layer_norm_clip(
    grads: chex.ArrayTree, global_l2_norm_clip: float, uniform: bool = True
) -> tuple[chex.ArrayTree, chex.ArrayTree]:
  """Applies gradient clipping per-example using per-layer norms.

  If len(grads) == 1, this function is equivalent to
  optax.per_example_global_norm_clip.  If len(grads) > 1, each array in grads
  will be independently clipped to a value ``C_i`` documented below.

  Let ``C = global_l2_norm_clip value``. Then per-layer clipping is done as
  follows:

  1. If ``uniform`` is ``True``, each of the ``K`` layers has an individual clip
  norm of ``C / sqrt(K)``.

  2. If ``uniform`` is ``False``, each of the ``K`` layers has an individual
  clip norm of ``C * sqrt(D_i / D)`` where ``D_i`` is the number of parameters
  in layer ``i``, and ``D`` is the total number of parameters in the model.

  Args:
    grads: flattened update; i.e. a list of gradients in which each item is the
      gradient for one layer; the function expects these to have a batch
      dimension on the 0th axis.
    global_l2_norm_clip: overall L2 clip norm to use.
    uniform: If `True`, per-layer clip norm is ``global_l2_norm_clip/sqrt(L)``,
      where ``L`` is the number of layers. Otherwise, per-layer clip norm is
      ``global_l2_norm_clip * sqrt(f)``, where ``f`` is the fraction of total
      model parameters that are in this layer.

  Returns:
    A tuple containing sum of the clipped per-example grads and the number of
    per-example grads that were clipped for each layer.

  Example:
    >>> import optax
    >>> import jax.numpy as jnp
    >>> grads = [jnp.array([[0, 0, 0], [0, 3, 4], [4, 0, 3], [3, 4, 0]])]
    >>> optax.per_example_layer_norm_clip(grads, jnp.inf)
    ([Array([7., 7., 7.], dtype=float32)], [Array(0, dtype=int32)])
    >>> optax.per_example_layer_norm_clip(grads, 0.0)
    ([Array([0., 0., 0.], dtype=float32)], [Array(3, dtype=int32)])
    >>> optax.per_example_layer_norm_clip(grads, 1.25)
    ([Array([1.75, 1.75, 1.75], dtype=float32)], [Array(3, dtype=int32)])

  References:
    McMahan et al., `Learning Differentially Private Recurrent Language Models
    <https://arxiv.org/abs/1710.06963>`_, 2017
  """

  if not _check_arrays_have_batch_dim(grads):
    raise ValueError(
        'Unlike other transforms, `per_example_layer_norm_clip` expects'
        ' `grads` to have a batch dimension in the 0th axis; got shapes:'
        f' {jax.tree.map(jnp.shape, grads)}.'
    )

  # Compute per-layer clip norms, based on whether we are using uniform
  # variant or not.
  if uniform:
    # Create list of length `num_layers` of per-layer clip norm.
    num_layers = len(jax.tree.leaves(grads))
    layer_clip_norms = jax.tree.map(
        lambda _: global_l2_norm_clip * (1.0 / num_layers) ** 0.5, grads
    )
  else:
    total_params = jax.tree.reduce(lambda x, g: x + g[0].size, grads, 0)
    layer_clip_norms = jax.tree.map(
        lambda g: global_l2_norm_clip * (g[0].size / total_params) ** 0.5, grads
    )

  result = jax.tree.map(per_example_global_norm_clip, grads, layer_clip_norms)
  return jax.tree.transpose(
      outer_treedef=jax.tree.structure(grads),
      inner_treedef=jax.tree.structure((0, 0)),
      pytree_to_transpose=result,
  )


def unitwise_norm(x: chex.Array) -> chex.Array:
  """Computes norms of each output unit separately."""
  if jnp.squeeze(x).ndim <= 1:  # Scalars and vectors
    squared_norm = jnp.sum(numerics.abs_sq(x), keepdims=True)
  # Note that this assumes parameters with a shape of length 3 are multihead
  # linear parameters--if you wish to apply AGC to 1D convs, you may need
  # to modify this line.
  elif x.ndim in (2, 3):  # Linear layers of shape IO or multihead linear
    squared_norm = jnp.sum(numerics.abs_sq(x), axis=0, keepdims=True)
  elif x.ndim == 4:  # Conv kernels of shape HWIO
    squared_norm = jnp.sum(numerics.abs_sq(x), axis=(0, 1, 2), keepdims=True)
  else:
    raise ValueError(
        f'Expected parameter with shape in {1, 2, 3, 4}, got {x.shape}.'
    )
  chex.assert_is_broadcastable(squared_norm.shape, x.shape)
  return jnp.broadcast_to(jnp.sqrt(squared_norm), x.shape)


def unitwise_clip(
    g_norm: chex.Array,
    max_norm: chex.Array,
    grad: chex.Array,
    div_eps: float = 1e-6,
) -> chex.Array:
  """Applies gradient clipping unit-wise."""
  # This little max(., div_eps) is distinct from the normal eps and just
  # prevents division by zero. It technically should be impossible to engage.
  clipped_grad = grad * (max_norm / jnp.maximum(g_norm, div_eps))
  chex.assert_equal_shape((g_norm, max_norm, grad, clipped_grad))
  return jnp.where(g_norm < max_norm, grad, clipped_grad)


def adaptive_grad_clip(
    clipping: float, eps: float = 1e-3
) -> base.GradientTransformation:
  """Clips updates to be at most ``clipping * parameter_norm``, unit-wise.

  Args:
    clipping: The maximum allowed ratio of update norm to parameter norm.
    eps: An epsilon term to prevent clipping of zero-initialized params.

  Returns:
    A :class:`optax.GradientTransformation` object.

  References:
    Brock et al., `High-Performance Large-Scale Image Recognition Without
    Normalization <https://arxiv.org/abs/2102.06171`_, 2021
  """

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    g_norm, p_norm = jax.tree.map(unitwise_norm, (updates, params))
    # Maximum allowable norm.
    max_norm = jax.tree.map(lambda x: clipping * jnp.maximum(x, eps), p_norm)
    # If grad norm > clipping * param_norm, rescale.
    updates = jax.tree.map(unitwise_clip, g_norm, max_norm, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)
