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
"""Gradient clipping transformations."""

from optax._src import base
from optax.transforms import _clipping


adaptive_grad_clip = _clipping.adaptive_grad_clip
AdaptiveGradClipState = base.EmptyState
ClipState = base.EmptyState
clip = _clipping.clip
clip_by_block_rms = _clipping.clip_by_block_rms
clip_by_global_norm = _clipping.clip_by_global_norm
ClipByGlobalNormState = base.EmptyState
per_example_global_norm_clip = _clipping.per_example_global_norm_clip
per_example_layer_norm_clip = _clipping.per_example_layer_norm_clip
unitwise_norm = _clipping.unitwise_norm
unitwise_clip = _clipping.unitwise_clip
