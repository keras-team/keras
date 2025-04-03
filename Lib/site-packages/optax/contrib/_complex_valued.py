# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Complex-valued optimization.

When using `split_real_and_imaginary` to wrap an optimizer, we split the complex
parameters and updates into pairs of real ones before sending them to the
`update` of the wrapped optimizer, and merge the pairs of transformed real
updates into complex ones afterward. In this way, optimizers on complex
parameters behave the same way as if they were running on two real parameters.

Note that the convention of conjugate for complex gradients in JAX is different
from that in PyTorch and other frameworks, and we need to manually conjugate the
gradients between `jax.grad` and `optimizer.update`.

See details at https://github.com/deepmind/optax/issues/196
"""

from typing import NamedTuple, Union

import chex
import jax
import jax.numpy as jnp
from optax._src import base


class SplitRealAndImaginaryArrays(NamedTuple):
  """A pair of real arrays split from a complex array."""

  real: chex.Array
  imaginary: chex.Array


def _complex_to_real_pair(
    x: chex.Array,
) -> Union[chex.Array, SplitRealAndImaginaryArrays]:
  """Splits a complex array into a `SplitRealAndImaginaryArrays`.

  Args:
    x: The input array, can be complex or real.

  Returns:
    `SplitRealAndImaginaryArrays` if the input is a complex array. If the
    input is a real array, it is passed through unmodified.
  """
  if jnp.iscomplexobj(x):
    return SplitRealAndImaginaryArrays(x.real, x.imag)
  else:
    return x


def _real_pair_to_complex(
    x: Union[chex.Array, SplitRealAndImaginaryArrays],
) -> chex.Array:
  """Merges a `SplitRealAndImaginaryArrays` into a complex array.

  Args:
    x: The input `SplitRealAndImaginaryArrays` or array.

  Returns:
    A complex array obtained from the real and imaginary parts of the
    `SplitRealAndImaginaryArrays`. If the input is not a
    `SplitRealAndImaginaryArrays`, it is passed through unmodified.
  """
  if isinstance(x, SplitRealAndImaginaryArrays):
    return x.real + x.imaginary * 1j
  else:
    return x


class SplitRealAndImaginaryState(NamedTuple):
  """Maintains the inner transformation state for `split_real_and_imaginary`."""

  inner_state: base.OptState


def split_real_and_imaginary(
    inner: base.GradientTransformation,
) -> base.GradientTransformation:
  """Splits the real and imaginary components of complex updates into two.

  The inner transformation processes real parameters and updates, and the
  pairs of transformed real updates are merged into complex updates.

  Parameters and updates that are real before splitting are passed through
  unmodified.

  Args:
    inner: The inner transformation.

  Returns:
    An `optax.GradientTransformation`.
  """

  def init_fn(params):
    params = jax.tree.map(_complex_to_real_pair, params)
    inner_state = inner.init(params)
    return SplitRealAndImaginaryState(inner_state)

  def update_fn(updates, state, params=None):
    inner_state = state.inner_state
    updates = jax.tree.map(_complex_to_real_pair, updates)
    params = jax.tree.map(_complex_to_real_pair, params)
    updates, inner_state = inner.update(updates, inner_state, params)
    updates = jax.tree.map(
        _real_pair_to_complex,
        updates,
        is_leaf=lambda x: isinstance(x, SplitRealAndImaginaryArrays),
    )
    return updates, SplitRealAndImaginaryState(inner_state)

  return base.GradientTransformation(init_fn, update_fn)
