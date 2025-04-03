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

from .axes_scan import broadcast as broadcast
from .frozen_dict import (
    FrozenDict as FrozenDict,
    copy as copy,
    freeze as freeze,
    pop as pop,
    pretty_repr as pretty_repr,
    unfreeze as unfreeze,
)
from .lift import (
    custom_vjp as custom_vjp,
    jit as jit,
    jvp as jvp,
    remat_scan as remat_scan,
    remat as remat,
    scan as scan,
    vjp as vjp,
    vmap as vmap,
    while_loop as while_loop,
)
from .meta import (
    AxisMetadata as AxisMetadata,
    map_axis_meta as map_axis_meta,
    unbox as unbox,
)
from .scope import (
    DenyList as DenyList,
    Scope as Scope,
    apply as apply,
    bind as bind,
    init as init,
    lazy_init as lazy_init,
)
from .tracers import (
    check_trace_level as check_trace_level,
    current_trace as current_trace,
)

from flax.typing import (
    Array as Array,
)
