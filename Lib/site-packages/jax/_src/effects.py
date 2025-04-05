# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JAX effects.

JAX uses effects to describe computations that may have side-effects. Effects
are associated with JAX primitive instances and Jaxprs.

A primitive instance with an effect will be protected from dead-code elimination
even if its result is unused.

A special class of effects are the **ordered** effects
(members of `effects.ordered_effects`).
The lowering of a computation with ordered effects will have one additional
input and one additional output for each ordered effect. These appear before
the regular inputs/outputs, and are of type `i1[0]`. These tokens
are threaded through the instructions with ordered effects to ensure that the
compiler will not eliminate, replicate, or reordered the corresponding
instructions.

To ensure the ordering across multiple computations we maintain a
per-thread set of the tokens returned by the last dispatched computation. There
is one token per ordered effect, and it may be sharded over the devices
used by the last dispatched computation. Upon dispatching a
new computation with ordered effects we take the current token, we shard it
on the devices for the computation to be dispatched and we pass it as an input.
Then we update the current token to refer to the token output of
the dispatched computation.

When we have ordered effects, we also use the current token to implement
`jax.barrier` which waits until the current tokens are ready.

The implementation of `jax.barrier` for unordered effects is a bit different,
because for these effects we do not thread tokens in and out of dispatched
computation. Instead, we use a `RuntimeToken`, which is an object returned when
dispatching a computation and on which we can block until is ready. We store
for each thread the `RuntimeToken` returned by the last dispatched computation.

For more details, see the design note:
https://jax.readthedocs.io/en/latest/jep/10657-sequencing-effects.html.
"""

from __future__ import annotations

from collections.abc import Iterable, Set
from typing import Any


class Effect:
  """A generic side-effect."""

Effects = Set[Effect]

class JaxprInputEffect(Effect):
  """A side-effect associated with the input of a jaxpr.

  Note that the `input_index` includes constvars.
  """

  def __init__(self, input_index: Any):
    self.input_index = input_index

  def replace(self, *, input_index: Any | None = None):
    if input_index is None:
      input_index = self.input_index
    return self.__class__(input_index)

  def __eq__(self, other):
    if not isinstance(other, JaxprInputEffect):
      return NotImplemented
    return self.input_index == other.input_index

  def __hash__(self):
    return hash((self.__class__, self.input_index))

  def __repr__(self):
    return f"{self.__class__.__name__}({self.input_index})"

class EffectTypeSet:

  def __init__(self):
    self._effect_types: set[type[Effect]] = set()

  def add_type(self, effect_type: type[Effect]):
    self._effect_types.add(effect_type)

  def contains(self, eff: Effect) -> bool:
    return any(isinstance(eff, eff_type) for eff_type in self._effect_types)

  def filter_in(self, effects: Iterable[Effect]) -> list[Effect]:
    return [eff for eff in effects if self.contains(eff)]

  def filter_not_in(self, effects: Iterable[Effect]) -> list[Effect]:
    return [eff for eff in effects if not self.contains(eff)]


no_effects: Effects = frozenset()
ordered_effects: EffectTypeSet = EffectTypeSet()

# By default, ordered effects are not allowed in multi-device computations,
# because we cannot ensure a total order. Optionally, an effect can be
# declared as shardable, which means that effects will appear in program order
# but for a given program point we may see several side effects on the
# participating devices, and there is no guarantee of their relative ordering.
shardable_ordered_effects: EffectTypeSet = EffectTypeSet()

lowerable_effects: EffectTypeSet = EffectTypeSet()
control_flow_allowed_effects: EffectTypeSet = EffectTypeSet()
custom_derivatives_allowed_effects: EffectTypeSet = EffectTypeSet()
remat_allowed_effects: EffectTypeSet = EffectTypeSet()
