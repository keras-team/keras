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

"""Initializers for Flax."""

# pylint: disable=unused-import
# re-export initializer functions from jax.nn
from jax.nn.initializers import constant as constant
from jax.nn.initializers import delta_orthogonal as delta_orthogonal
from jax.nn.initializers import glorot_normal as glorot_normal
from jax.nn.initializers import glorot_uniform as glorot_uniform
from jax.nn.initializers import he_normal as he_normal
from jax.nn.initializers import he_uniform as he_uniform
from jax.nn.initializers import kaiming_normal as kaiming_normal
from jax.nn.initializers import kaiming_uniform as kaiming_uniform
from jax.nn.initializers import lecun_normal as lecun_normal
from jax.nn.initializers import lecun_uniform as lecun_uniform
from jax.nn.initializers import normal as normal
from jax.nn.initializers import ones as ones
from jax.nn.initializers import orthogonal as orthogonal
from jax.nn.initializers import truncated_normal as truncated_normal
from jax.nn.initializers import uniform as uniform
from jax.nn.initializers import variance_scaling as variance_scaling
from jax.nn.initializers import xavier_normal as xavier_normal
from jax.nn.initializers import xavier_uniform as xavier_uniform
from jax.nn.initializers import zeros as zeros
from flax.typing import Initializer as Initializer

# pylint: enable=unused-import


def zeros_init() -> Initializer:
  """Builds an initializer that returns a constant array full of zeros.

  >>> import jax, jax.numpy as jnp
  >>> from flax.linen.initializers import zeros_init
  >>> zeros_initializer = zeros_init()
  >>> zeros_initializer(jax.random.key(42), (2, 3), jnp.float32)
  Array([[0., 0., 0.],
         [0., 0., 0.]], dtype=float32)
  """
  return zeros


def ones_init() -> Initializer:
  """Builds an initializer that returns a constant array full of ones.

  >>> import jax, jax.numpy as jnp
  >>> from flax.linen.initializers import ones_init
  >>> ones_initializer = ones_init()
  >>> ones_initializer(jax.random.key(42), (3, 2), jnp.float32)
  Array([[1., 1.],
         [1., 1.],
         [1., 1.]], dtype=float32)
  """
  return ones
