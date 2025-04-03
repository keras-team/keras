# Copyright 2024 The Flax Authors.
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

"""Stochastic modules."""

from collections.abc import Sequence

import jax.numpy as jnp
from jax import lax, random

from flax.linen.module import Module, compact, merge_param
from flax.typing import PRNGKey


class Dropout(Module):
  """Create a dropout layer.

  .. note::
    When using :meth:`Module.apply() <flax.linen.Module.apply>`, make sure
    to include an RNG seed named ``'dropout'``. Dropout isn't necessary for
    variable initialization.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> class MLP(nn.Module):
    ...   @nn.compact
    ...   def __call__(self, x, train):
    ...     x = nn.Dense(4)(x)
    ...     x = nn.Dropout(0.5, deterministic=not train)(x)
    ...     return x

    >>> model = MLP()
    >>> x = jnp.ones((1, 3))
    >>> variables = model.init(jax.random.key(0), x, train=False) # don't use dropout
    >>> model.apply(variables, x, train=False) # don't use dropout
    Array([[-0.17875527,  1.6255447 , -1.2431065 , -0.02554005]], dtype=float32)
    >>> model.apply(variables, x, train=True, rngs={'dropout': jax.random.key(1)}) # use dropout
    Array([[-0.35751054,  3.2510893 ,  0.        ,  0.        ]], dtype=float32)

  Attributes:
    rate: the dropout probability.  (_not_ the keep rate!)
    broadcast_dims: dimensions that will share the same dropout mask
    deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
      masked, whereas if true, no mask is applied and the inputs are returned as
      is.
    rng_collection: the rng collection name to use when requesting an rng key.
  """

  rate: float
  broadcast_dims: Sequence[int] = ()
  deterministic: bool | None = None
  rng_collection: str = 'dropout'

  @compact
  def __call__(
    self,
    inputs,
    deterministic: bool | None = None,
    rng: PRNGKey | None = None,
  ):
    """Applies a random dropout mask to the input.

    Args:
      inputs: the inputs that should be randomly masked.
      deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
      rng: an optional PRNGKey used as the random key, if not specified, one
        will be generated using ``make_rng`` with the ``rng_collection`` name.

    Returns:
      The masked inputs reweighted to preserve mean.
    """
    deterministic = merge_param(
      'deterministic', self.deterministic, deterministic
    )

    if (self.rate == 0.0) or deterministic:
      return inputs

    # Prevent gradient NaNs in 1.0 edge-case.
    if self.rate == 1.0:
      return jnp.zeros_like(inputs)

    keep_prob = 1.0 - self.rate
    if rng is None:
      rng = self.make_rng(self.rng_collection)
    broadcast_shape = list(inputs.shape)
    for dim in self.broadcast_dims:
      broadcast_shape[dim] = 1
    mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
    mask = jnp.broadcast_to(mask, inputs.shape)
    return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
