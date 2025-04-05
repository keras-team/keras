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
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import lax, random

from flax.nnx import rnglib
from flax.nnx.module import Module, first_from


@dataclasses.dataclass(repr=False)
class Dropout(Module):
  """Create a dropout layer.

  To use dropout, call the :func:`train` method (or pass in
  ``deterministic=False`` in the constructor or during call time).

  To disable dropout, call the :func:`eval` method (or pass in
  ``deterministic=True`` in the constructor or during call time).

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> class MLP(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(in_features=3, out_features=4, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear(x)
    ...     x = self.dropout(x)
    ...     return x

    >>> model = MLP(rngs=nnx.Rngs(0))
    >>> x = jnp.ones((1, 3))

    >>> model.train() # use dropout
    >>> model(x)
    Array([[ 0.       ,  0.       , -1.592019 , -2.5238838]], dtype=float32)

    >>> model.eval() # don't use dropout
    >>> model(x)
    Array([[ 1.0533503, -1.2679932, -0.7960095, -1.2619419]], dtype=float32)

  Args:
    rate: the dropout probability.  (_not_ the keep rate!)
    broadcast_dims: dimensions that will share the same dropout mask
    deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
      masked, whereas if true, no mask is applied and the inputs are returned
      as is.
    rng_collection: the rng collection name to use when requesting an rng key.
    rngs: rng key.
  """

  rate: float
  broadcast_dims: Sequence[int] = ()
  deterministic: bool = False
  rng_collection: str = 'dropout'
  rngs: rnglib.Rngs | None = None

  def __call__(
    self,
    inputs,
    *,
    deterministic: bool | None = None,
    rngs: rnglib.Rngs | None = None,
  ) -> jax.Array:
    """Applies a random dropout mask to the input.

    Args:
      inputs: the inputs that should be randomly masked.
      deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is. The ``deterministic`` flag passed into the call method will take
        precedence over the ``deterministic`` flag passed into the constructor.
      rngs: rng key. The rng key passed into the call method will take
        precedence over the rng key passed into the constructor.

    Returns:
      The masked inputs reweighted to preserve mean.
    """
    deterministic = first_from(
      deterministic,
      self.deterministic,
      error_msg="""No `deterministic` argument was provided to Dropout
          as either a __call__ argument or class attribute""",
    )

    if (self.rate == 0.0) or deterministic:
      return inputs

    # Prevent gradient NaNs in 1.0 edge-case.
    if self.rate == 1.0:
      return jnp.zeros_like(inputs)

    rngs = first_from(
      rngs,
      self.rngs,
      error_msg="""`deterministic` is False, but no `rngs` argument was provided to Dropout
          as either a __call__ argument or class attribute.""",
    )

    keep_prob = 1.0 - self.rate
    rng = rngs[self.rng_collection]()
    broadcast_shape = list(inputs.shape)
    for dim in self.broadcast_dims:
      broadcast_shape[dim] = 1
    mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
    mask = jnp.broadcast_to(mask, inputs.shape)
    return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
