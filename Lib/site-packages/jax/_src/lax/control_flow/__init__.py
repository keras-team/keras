# Copyright 2022 The JAX Authors.
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
"""Module for the control flow primitives."""
from jax._src.lax.control_flow.loops import (
    associative_scan as associative_scan,
    cummax as cummax,
    cummax_p as cummax_p,
    cummin as cummin,
    cummin_p as cummin_p,
    cumlogsumexp as cumlogsumexp,
    cumlogsumexp_p as cumlogsumexp_p,
    cumprod as cumprod,
    cumprod_p as cumprod_p,
    cumsum as cumsum,
    cumsum_p as cumsum_p,
    cumred_reduce_window_impl as cumred_reduce_window_impl,
    fori_loop as fori_loop,
    map as map,
    scan as scan,
    scan_p as scan_p,
    _scan_impl as _scan_impl,
    while_loop as while_loop,
    while_p as while_p,
)
from jax._src.lax.control_flow.conditionals import (
    cond as cond,
    cond_p as cond_p,
    switch as switch,
    platform_dependent as platform_dependent,
    platform_index_p as platform_index_p,
)
from jax._src.lax.control_flow.solves import (
    custom_linear_solve as custom_linear_solve,
    custom_root as custom_root,
    _custom_linear_solve_impl as _custom_linear_solve_impl,
    linear_solve_p as linear_solve_p,
)
# Private utilities used elsewhere in JAX
# TODO(sharadmv): lift them into a more common place
from jax._src.lax.control_flow.common import (
    _initial_style_open_jaxpr as _initial_style_open_jaxpr,
    _initial_style_jaxpr as _initial_style_jaxpr,
    _initial_style_jaxprs_with_common_consts as _initial_style_jaxprs_with_common_consts,
    _check_tree_and_avals as _check_tree_and_avals,

)
# TODO(mattjj): fix dependent library which expects optimization_barrier_p here
from jax._src.lax.lax import optimization_barrier_p as optimization_barrier_p
