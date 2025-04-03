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

import jax.numpy as jnp
from jax import lax, random


def dropout(scope, inputs, rate, deterministic=False, rng=None):
  """Applies a random dropout mask to the input.
  Args:
    inputs: the inputs that should be randomly masked.
    rate: the probablity of masking out a value.
    deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
      masked, whereas if true, no mask is applied and the inputs are returned as
      is.
    rng: an optional `jax.random.PRNGKey`. By default `nn.make_rng()` will
      be used.
  Returns:
    The masked inputs.
  """
  if rate == 0.0:
    return inputs
  keep_prob = 1.0 - rate

  if deterministic:
    return inputs
  else:
    if rng is None:
      rng = scope.make_rng('dropout')
    mask = random.bernoulli(rng, p=keep_prob, shape=inputs.shape)
    return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
