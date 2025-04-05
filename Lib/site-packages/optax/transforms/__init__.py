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
"""The transforms sub-package."""

# pylint: disable=g-importing-member

from optax.transforms._accumulation import ema
from optax.transforms._accumulation import EmaState
from optax.transforms._accumulation import MultiSteps
from optax.transforms._accumulation import MultiStepsState
from optax.transforms._accumulation import ShouldSkipUpdateFunction
from optax.transforms._accumulation import skip_large_updates
from optax.transforms._accumulation import skip_not_finite
from optax.transforms._accumulation import trace
from optax.transforms._accumulation import TraceState
from optax.transforms._adding import add_decayed_weights
from optax.transforms._adding import add_noise
from optax.transforms._adding import AddNoiseState
from optax.transforms._clipping import adaptive_grad_clip
from optax.transforms._clipping import clip
from optax.transforms._clipping import clip_by_block_rms
from optax.transforms._clipping import clip_by_global_norm
from optax.transforms._clipping import per_example_global_norm_clip
from optax.transforms._clipping import per_example_layer_norm_clip
from optax.transforms._clipping import unitwise_clip
from optax.transforms._clipping import unitwise_norm
from optax.transforms._combining import chain
from optax.transforms._combining import named_chain
from optax.transforms._combining import partition
from optax.transforms._combining import PartitionState
from optax.transforms._conditionality import apply_if_finite
from optax.transforms._conditionality import ApplyIfFiniteState
from optax.transforms._conditionality import conditionally_mask
from optax.transforms._conditionality import conditionally_transform
from optax.transforms._conditionality import ConditionallyMaskState
from optax.transforms._conditionality import ConditionallyTransformState
from optax.transforms._conditionality import ConditionFn
from optax.transforms._constraining import keep_params_nonnegative
from optax.transforms._constraining import NonNegativeParamsState
from optax.transforms._constraining import zero_nans
from optax.transforms._constraining import ZeroNansState
from optax.transforms._layouts import flatten
from optax.transforms._masking import masked
from optax.transforms._masking import MaskedNode
from optax.transforms._masking import MaskedState


__all__ = (
    "adaptive_grad_clip",
    "add_decayed_weights",
    "add_noise",
    "AddNoiseState",
    "apply_if_finite",
    "ApplyIfFiniteState",
    "chain",
    "clip_by_block_rms",
    "clip_by_global_norm",
    "clip",
    "conditionally_mask",
    "ConditionallyMaskState",
    "conditionally_transform",
    "ConditionallyTransformState",
    "ema",
    "EmaState",
    "flatten",
    "keep_params_nonnegative",
    "masked",
    "MaskedState",
    "MultiSteps",
    "MultiStepsState",
    "named_chain",
    "NonNegativeParamsState",
    "partition",
    "PartitionState",
    "ShouldSkipUpdateFunction",
    "skip_large_updates",
    "skip_not_finite",
    "trace",
    "TraceState",
    "zero_nans",
    "ZeroNansState",
)
