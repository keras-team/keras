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

"""Common functions for neural network libraries."""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax.numpy import tanh as tanh
from jax.nn import initializers as initializers
from jax._src.nn.functions import (
  celu as celu,
  elu as elu,
  gelu as gelu,
  glu as glu,
  hard_sigmoid as hard_sigmoid,
  hard_silu as hard_silu,
  hard_swish as hard_swish,
  hard_tanh as hard_tanh,
  leaky_relu as leaky_relu,
  log_sigmoid as log_sigmoid,
  log_softmax as log_softmax,
  logsumexp as logsumexp,
  standardize as standardize,
  one_hot as one_hot,
  relu as relu,
  relu6 as relu6,
  dot_product_attention as dot_product_attention,
  selu as selu,
  sigmoid as sigmoid,
  soft_sign as soft_sign,
  softmax as softmax,
  softplus as softplus,
  sparse_plus as sparse_plus,
  sparse_sigmoid as sparse_sigmoid,
  silu as silu,
  swish as swish,
  squareplus as squareplus,
  mish as mish,
)
