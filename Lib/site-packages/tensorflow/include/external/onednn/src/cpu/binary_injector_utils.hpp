/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CPU_BINARY_INJECTOR_UTILS_HPP
#define CPU_BINARY_INJECTOR_UTILS_HPP

#include <tuple>
#include <vector>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace binary_injector_utils {
/*
 * Extracts pointers to tensors passed by user as binary postops rhs (right-hand-side)
 * arguments (arg1 from binary postop) from execution context. Those pointers are placed
 * in vector in order of binary post-op appearance inside post_ops_t structure. Returned vector
 * usually is passed to kernel during execution phase in runtime params.
 * @param first_arg_idx_offset - offset for indexation of binary postop arguments
 * (used for fusions with dw convolutions)
 */
std::vector<const void *> prepare_binary_args(const post_ops_t &post_ops,
        const dnnl::impl::exec_ctx_t &ctx,
        const unsigned first_arg_idx_offset = 0);

bool bcast_strategy_present(
        const std::vector<broadcasting_strategy_t> &post_ops_bcasts,
        const broadcasting_strategy_t bcast_strategy);

std::vector<broadcasting_strategy_t> extract_bcast_strategies(
        const std::vector<dnnl_post_ops::entry_t> &post_ops,
        const memory_desc_wrapper &dst_md);

memory_desc_t get_src1_desc(
        const post_ops_t::entry_t &post_op, const memory_desc_wrapper &dst_d);

/*
 * Returns a tuple of bools, which size is equal to number of bcast
 * strategies passed in. Values at consecutive positions indicate existence of
 * binary postop with a particular bcast strategy in post_ops vector.
 */
template <typename... Str>
auto bcast_strategies_present_tup(
        const std::vector<dnnl_post_ops::entry_t> &post_ops,
        const memory_desc_wrapper &dst_md, Str... bcast_strategies)
        -> decltype(std::make_tuple((bcast_strategies, false)...)) {
    const auto post_ops_bcasts = extract_bcast_strategies(post_ops, dst_md);
    return std::make_tuple(
            bcast_strategy_present(post_ops_bcasts, bcast_strategies)...);
}

} // namespace binary_injector_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
