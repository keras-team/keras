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

"""Flax Neural Network api."""

# pylint: disable=g-multiple-import
# re-export commonly used modules and functions
from flax.linen import activation as activation
from flax.linen import initializers as initializers
from flax.linen.activation import (
    celu as celu,
    elu as elu,
    gelu as gelu,
    glu as glu,
    leaky_relu as leaky_relu,
    log_sigmoid as log_sigmoid,
    log_softmax as log_softmax,
    relu as relu,
    sigmoid as sigmoid,
    silu as silu,
    soft_sign as soft_sign,
    softmax as softmax,
    softplus as softplus,
    swish as swish,
    tanh as tanh,
)
from flax.linen.pooling import (avg_pool as avg_pool, max_pool as max_pool)
from .attention import (
    dot_product_attention as dot_product_attention,
    multi_head_dot_product_attention as multi_head_dot_product_attention,
)
from .linear import (
    Embedding as Embedding,
    conv_transpose as conv_transpose,
    conv as conv,
    dense_general as dense_general,
    dense as dense,
    embedding as embedding,
)
from .normalization import (
    batch_norm as batch_norm,
    group_norm as group_norm,
    layer_norm as layer_norm,
)
from .stochastic import dropout as dropout

# pylint: enable=g-multiple-import
