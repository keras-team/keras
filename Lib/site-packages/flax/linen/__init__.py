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

"""The Flax Module system."""


# pylint: disable=g-multiple-import,useless-import-alias
# re-export commonly used modules and functions
from flax.core import (
    DenyList as DenyList,
    FrozenDict as FrozenDict,
    broadcast as broadcast,
    meta as meta,
)
from flax.core.meta import (
    PARTITION_NAME as PARTITION_NAME,
    Partitioned as Partitioned,
    get_partition_spec as get_partition_spec,
    get_sharding as get_sharding,
    unbox as unbox,
    with_partitioning as with_partitioning,
)
from flax.core.spmd import (
    get_logical_axis_rules as get_logical_axis_rules,
    logical_axis_rules as logical_axis_rules,
    set_logical_axis_rules as set_logical_axis_rules,
)
from .activation import (
    PReLU as PReLU,
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
    normalize as normalize,
    one_hot as one_hot,
    relu6 as relu6,
    relu as relu,
    selu as selu,
    sigmoid as sigmoid,
    silu as silu,
    soft_sign as soft_sign,
    softmax as softmax,
    softplus as softplus,
    standardize as standardize,
    swish as swish,
    tanh as tanh,
)
from .attention import (
    MultiHeadAttention as MultiHeadAttention,
    MultiHeadDotProductAttention as MultiHeadDotProductAttention,
    SelfAttention as SelfAttention,
    combine_masks as combine_masks,
    dot_product_attention_weights as dot_product_attention_weights,
    dot_product_attention as dot_product_attention,
    make_attention_mask as make_attention_mask,
    make_causal_mask as make_causal_mask,
)
from .batch_apply import BatchApply as BatchApply
from .combinators import Sequential as Sequential
from .fp8_ops import (
    Fp8DirectDotGeneralOp as Fp8DirectDotGeneralOp,
    Fp8DotGeneralOp as Fp8DotGeneralOp,
    NANOOFp8DotGeneralOp as NANOOFp8DotGeneralOp,
)
from .initializers import (
    ones_init as ones_init,
    ones as ones,
    zeros_init as zeros_init,
    zeros as zeros,
)
from .linear import (
    ConvLocal as ConvLocal,
    ConvTranspose as ConvTranspose,
    Conv as Conv,
    DenseGeneral as DenseGeneral,
    Dense as Dense,
    Einsum as Einsum,
    Embed as Embed,
)
from .module import (
    Module as Module,
    Variable as Variable,
    apply as apply,
    compact_name_scope as compact_name_scope,
    compact as compact,
    disable_named_call as disable_named_call,
    enable_named_call as enable_named_call,
    init_with_output as init_with_output,
    init as init,
    intercept_methods as intercept_methods,
    merge_param as merge_param,
    nowrap as nowrap,
    override_named_call as override_named_call,
    share_scope as share_scope,
)
from .normalization import (
    BatchNorm as BatchNorm,
    GroupNorm as GroupNorm,
    InstanceNorm as InstanceNorm,
    LayerNorm as LayerNorm,
    RMSNorm as RMSNorm,
    SpectralNorm as SpectralNorm,
    WeightNorm as WeightNorm,
)
from .pooling import (avg_pool as avg_pool, max_pool as max_pool, pool as pool)
from .recurrent import (
    Bidirectional as Bidirectional,
    ConvLSTMCell as ConvLSTMCell,
    GRUCell as GRUCell,
    LSTMCell as LSTMCell,
    MGUCell as MGUCell,
    OptimizedLSTMCell as OptimizedLSTMCell,
    RNNCellBase as RNNCellBase,
    RNN as RNN,
    SimpleCell as SimpleCell,
)
from .spmd import (
    LogicallyPartitioned as LogicallyPartitioned,
    logical_to_mesh,
    logical_to_mesh_axes,
    logical_to_mesh_sharding,
    with_logical_constraint,
    with_logical_partitioning as with_logical_partitioning,
)
from .stochastic import Dropout as Dropout
from .summary import tabulate
from .transforms import (
    add_metadata_axis,
    checkpoint as checkpoint,
    cond as cond,
    custom_vjp as custom_vjp,
    fold_rngs as fold_rngs,
    grad as grad,
    jit as jit,
    jvp as jvp,
    map_variables as map_variables,
    named_call as named_call,
    remat_scan as remat_scan,
    remat as remat,
    scan as scan,
    switch as switch,
    value_and_grad as value_and_grad,
    vjp as vjp,
    vmap as vmap,
    while_loop as while_loop,
)
# pylint: enable=g-multiple-import
