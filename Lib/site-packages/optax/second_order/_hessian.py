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
"""Functions for computing diagonals of the Hessian wrt to a set of parameters.

Computing the Hessian for neural networks is typically intractable due to the
quadratic memory requirements. Solving for the diagonal can be done cheaply,
with sub-quadratic memory requirements.
"""

from typing import Any

import jax
from jax import flatten_util
import jax.numpy as jnp
from optax.second_order import _base


def _ravel(p: Any) -> jax.Array:
  return flatten_util.ravel_pytree(p)[0]


def hvp(
    loss: _base.LossFn,
    v: jax.Array,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  """Performs an efficient vector-Hessian (of `loss`) product.

  Args:
    loss: the loss function.
    v: a vector of size `ravel(params)`.
    params: model parameters.
    inputs: inputs at which `loss` is evaluated.
    targets: targets at which `loss` is evaluated.

  Returns:
    An Array corresponding to the product of `v` and the Hessian of `loss`
    evaluated at `(params, inputs, targets)`.
  """
  _, unravel_fn = flatten_util.ravel_pytree(params)
  loss_fn = lambda p: loss(p, inputs, targets)
  return jax.jvp(jax.grad(loss_fn), [params], [unravel_fn(v)])[1]


def hessian_diag(
    loss: _base.LossFn,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  """Computes the diagonal hessian of `loss` at (`inputs`, `targets`).

  Args:
    loss: the loss function.
    params: model parameters.
    inputs: inputs at which `loss` is evaluated.
    targets: targets at which `loss` is evaluated.

  Returns:
    A DeviceArray corresponding to the product to the Hessian of `loss`
    evaluated at `(params, inputs, targets)`.
  """
  vs = jnp.eye(_ravel(params).size)
  comp = lambda v: jnp.vdot(v, _ravel(hvp(loss, v, params, inputs, targets)))
  return jax.vmap(comp)(vs)
