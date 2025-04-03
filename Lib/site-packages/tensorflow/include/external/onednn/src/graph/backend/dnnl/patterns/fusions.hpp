/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_PATTERNS_FUSIONS_HPP
#define GRAPH_BACKEND_DNNL_PATTERNS_FUSIONS_HPP

#include "graph/utils/pm/pass_manager.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

#define DNNL_BACKEND_REGISTER_PATTERN_DECLARE(pattern_class_) \
    void register_##pattern_class_(graph::pass::pass_registry_t &registry);

DNNL_BACKEND_REGISTER_PATTERN_DECLARE(conv_block_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(conv_post_ops)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(matmul_post_ops)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(sdp)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(binary_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(bn_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(convtranspose_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(eltwise_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(interpolate_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(pool_post_ops)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(quantize_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(reduction_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(reorder_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(shuffle_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(single_op_pass)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(softmax_post_ops)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(layernorm_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(sum_fusion)
DNNL_BACKEND_REGISTER_PATTERN_DECLARE(concat_fusion)

#undef DNNL_BACKEND_REGISTER_PATTERN_DECLARE

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
