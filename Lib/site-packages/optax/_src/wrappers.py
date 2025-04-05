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
"""Transformation wrappers."""

from collections.abc import Callable
import functools

import chex
import jax.numpy as jnp
from optax._src import base
from optax.transforms import _accumulation
from optax.transforms import _conditionality
from optax.transforms import _layouts
from optax.transforms import _masking


apply_if_finite = _conditionality.apply_if_finite
ApplyIfFiniteState = _conditionality.ApplyIfFiniteState
ConditionFn = _conditionality.ConditionFn
conditionally_mask = _conditionality.conditionally_mask
conditionally_transform = _conditionality.conditionally_transform
ConditionallyMaskState = _conditionality.ConditionallyMaskState
ConditionallyTransformState = _conditionality.ConditionallyTransformState
flatten = _layouts.flatten
masked = _masking.masked
MaskedNode = _masking.MaskedNode
MaskedState = _masking.MaskedState
MultiSteps = _accumulation.MultiSteps
MultiStepsState = _accumulation.MultiStepsState
ShouldSkipUpdateFunction = _accumulation.ShouldSkipUpdateFunction
skip_not_finite = _accumulation.skip_not_finite
skip_large_updates = _accumulation.skip_large_updates


@functools.partial(
    chex.warn_deprecated_function,
    replacement='optax.transforms.maybe_transform',
)
def maybe_update(
    inner: base.GradientTransformation,
    should_update_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> base.GradientTransformationExtraArgs:
  return conditionally_transform(
      inner=inner, should_transform_fn=should_update_fn
  )


MaybeUpdateState = ConditionallyTransformState
