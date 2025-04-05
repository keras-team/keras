# Copyright 2019 The JAX Authors.
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

"""
Common neural network layer initializers, consistent with definitions
used in Keras and Sonnet.
"""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.nn.initializers import (
  constant as constant,
  Initializer as Initializer,
  delta_orthogonal as delta_orthogonal,
  glorot_normal as glorot_normal,
  glorot_uniform as glorot_uniform,
  he_normal as he_normal,
  he_uniform as he_uniform,
  kaiming_normal as kaiming_normal,
  kaiming_uniform as kaiming_uniform,
  lecun_normal as lecun_normal,
  lecun_uniform as lecun_uniform,
  normal as normal,
  ones as ones,
  orthogonal as orthogonal,
  truncated_normal as truncated_normal,
  uniform as uniform,
  variance_scaling as variance_scaling,
  xavier_normal as xavier_normal,
  xavier_uniform as xavier_uniform,
  zeros as zeros,
)
