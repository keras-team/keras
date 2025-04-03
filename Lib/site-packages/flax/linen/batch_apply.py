# Copyright 2023 The Flax Authors.
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

"""Batch apply."""

import jax, jax.numpy as jnp
import numpy as np


def ndim_at_least(x, num_dims):
  if not (isinstance(x, jax.Array) or isinstance(x, np.ndarray)):
    x = jnp.asarray(x)
  return x.ndim >= num_dims

def arbitrary_mergeable_leaf(min_num_dims, args, kwargs):
  for a in jax.tree_util.tree_leaves(args):
    if ndim_at_least(a, min_num_dims):
      return a
  for k in jax.tree_util.tree_leaves(kwargs):
    if ndim_at_least(k, min_num_dims):
      return k
  # Couldn't find a satisfactory leaf.
  return None

def merge_leading_dims(x, num_dims):
  """Merge leading dimensions."""
  # Don't merge if there aren't dimensions to merge.
  if not ndim_at_least(x, num_dims):
    return x

  new_shape = (np.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
  return x.reshape(new_shape)

def split_leading_dim(x, to_dim):
  new_shape = to_dim + x.shape[1:]
  return x.reshape(new_shape)

class BatchApply:
  r"""Temporarily merges leading dimensions of input tensors.

  Merges the leading dimensions of a tensor into a single dimension, runs the
  given callable, then splits the leading dimension of the result to match the
  input.

  Input arrays whose rank is smaller than the number of dimensions to collapse
  are passed unmodified.

  This may be useful for applying a module to each timestep of e.g. a
  ``[Time, Batch, ...]`` array.

  For some ``f``\ s and platforms, this may be more efficient than
  :func:`jax.vmap`, especially when combined with other transformations like
  :func:`jax.grad`.

  Example usage::

    >>> import jax, jax.numpy as jnp

    >>> a = jax.random.normal(jax.random.key(0), [2, 3, 4])
    >>> b = jax.random.normal(jax.random.key(1), [4])

    >>> def raises(a, b):
    ...   if len(a.shape) != 2:
    ...     raise ValueError("a must be shape 2")
    ...   if len(b.shape) != 1:
    ...     raise ValueError("b must be shape 1")
    ...   return jnp.dot(a, b)

    >>> out = BatchApply(raises)(a, b)
    >>> expected_merged_leading = raises(a.reshape(2*3, 4), b)
    >>> expected = expected_merged_leading.reshape((2, 3) + expected_merged_leading.shape[1:])
    >>> np.testing.assert_array_equal(out, expected)
  """

  def __init__(self, f, num_dims=2):
    """Constructs a :class:`BatchApply` module.

    Args:
      f: The callable to be applied to the reshaped array.
      num_dims: The number of dimensions to merge.
    """
    self._f = f
    self.num_dims = num_dims

  def __call__(self, *args, **kwargs):
    example = arbitrary_mergeable_leaf(self.num_dims, args, kwargs)
    if example is None:
      raise ValueError(
        'BatchApply requires at least one input with ndim >= '
        f'{self.num_dims}.'
      )

    merge = lambda x: merge_leading_dims(x, self.num_dims)
    split = lambda x: split_leading_dim(x, example.shape[:self.num_dims])
    args = jax.tree_util.tree_map(merge, args)
    kwargs = jax.tree_util.tree_map(merge, kwargs)
    outputs = self._f(*args, **kwargs)
    return jax.tree_util.tree_map(split, outputs)