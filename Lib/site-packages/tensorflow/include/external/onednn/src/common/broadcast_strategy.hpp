/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_BROADCAST_STRATEGY_HPP
#define COMMON_BROADCAST_STRATEGY_HPP

#include <array>
#include <set>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"

namespace dnnl {
namespace impl {

using output_dims_t = std::array<dim_t, DNNL_MAX_NDIMS>;

enum class broadcasting_strategy_t {
    // [n, c, d, h, w]
    scalar, // [1, 1, 1, 1, 1] // Channel_shared
    per_oc, // [1, c, 1, 1, 1] // Channel-wise
    per_oc_spatial, // [1, c, 1, 1, 1] specific case for binary kernel nchw format
    per_mb, // [n, 1, 1, 1, 1] // broadcast per batch
    per_mb_spatial, // [n, 1, d, h, w] // Broadcast only channel
    per_mb_w, // [n, 1, 1, 1, w] // Broadcast per batch and width
    per_w, // [1, 1, 1, 1, w] // Broadcast per width
    shared_axes, // [n, 1, d, h, 1] // General case broadcast (any combination)
    batch, // [1, c, d, h, w] // Broadcast only batch
    spatial, // [n, c, 1, 1, 1] // Broadcast spatial dimensions
    no_broadcast, // [n, c, d, h, w]
    unsupported
};

using bcast_set_t = std::set<broadcasting_strategy_t>;

inline const bcast_set_t &default_strategies() {
    static const bcast_set_t s
            = {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::no_broadcast};
    return s;
}

output_dims_t make_output_dims(const memory_desc_wrapper &dst_d);

broadcasting_strategy_t get_rhs_arg_broadcasting_strategy(
        const memory_desc_t &rhs_arg_md, const memory_desc_wrapper &dst_d);

broadcasting_strategy_t get_rhs_arg_broadcasting_strategy(
        const memory_desc_t &rhs_arg_md, const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

} // namespace impl
} // namespace dnnl

#endif
