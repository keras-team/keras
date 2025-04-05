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
"""Functions for computing diagonals of the fisher information matrix.

Computing the Fisher matrix for neural networks is typically intractable due to
the quadratic memory requirements. Solving for the diagonal can be done cheaply,
with sub-quadratic memory requirements.
"""

from typing import Any

import jax
from jax import flatten_util
import jax.numpy as jnp
from optax.second_order import _base


def _ravel(p: Any) -> jax.Array:
  return flatten_util.ravel_pytree(p)[0]


def fisher_diag(
    negative_log_likelihood: _base.LossFn,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
  """Computes the diagonal of the (observed) Fisher information matrix.

  Args:
    negative_log_likelihood: the negative log likelihood function with expected
      signature `loss = fn(params, inputs, targets)`.
    params: model parameters.
    inputs: inputs at which `negative_log_likelihood` is evaluated.
    targets: targets at which `negative_log_likelihood` is evaluated.

  Returns:
    An Array corresponding to the product to the Hessian of
    `negative_log_likelihood` evaluated at `(params, inputs, targets)`.
  """
  return jnp.square(
      _ravel(jax.grad(negative_log_likelihood)(params, inputs, targets))
  )
