/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_LAYOUT_PROPAGATOR_HPP
#define GRAPH_BACKEND_DNNL_LAYOUT_PROPAGATOR_HPP

#include <memory>
#include <utility>
#include <vector>
#include <type_traits>

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/subgraph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

using layout_propagator_func
        = std::function<status_t(std::shared_ptr<op_t> &, const dnnl::engine &,
                fusion_info_mgr_t &, pd_cache_t &, subgraph_rewriter_t &)>;

status_t insert_reorder_before(std::shared_ptr<op_t> &, size_t,
        const dnnl::memory::desc &, const dnnl::engine &, fusion_info_mgr_t &,
        pd_cache_t &, subgraph_rewriter_t &);

status_t insert_reorder_after(std::shared_ptr<op_t> &, size_t,
        const dnnl::memory::desc &, const dnnl::engine &, fusion_info_mgr_t &,
        pd_cache_t &, subgraph_rewriter_t &);

#define DECLARE_LAYOUT_PROPAGATOR(op_name) \
    status_t layout_propagator_for_##op_name(std::shared_ptr<op_t> &, \
            const dnnl::engine &, fusion_info_mgr_t &, pd_cache_t &, \
            subgraph_rewriter_t &);

DECLARE_LAYOUT_PROPAGATOR(conv);
DECLARE_LAYOUT_PROPAGATOR(deconv);
DECLARE_LAYOUT_PROPAGATOR(deconv_bwd_data);
DECLARE_LAYOUT_PROPAGATOR(deconv_bwd_weights);
DECLARE_LAYOUT_PROPAGATOR(eltwise);
DECLARE_LAYOUT_PROPAGATOR(eltwise_bwd);
DECLARE_LAYOUT_PROPAGATOR(binary);
DECLARE_LAYOUT_PROPAGATOR(concat);
DECLARE_LAYOUT_PROPAGATOR(shuffle);
DECLARE_LAYOUT_PROPAGATOR(matmul);
DECLARE_LAYOUT_PROPAGATOR(pool);
DECLARE_LAYOUT_PROPAGATOR(pool_bwd);
DECLARE_LAYOUT_PROPAGATOR(batchnorm);
DECLARE_LAYOUT_PROPAGATOR(batchnorm_bwd);
DECLARE_LAYOUT_PROPAGATOR(prelu);
DECLARE_LAYOUT_PROPAGATOR(prelu_bwd);
DECLARE_LAYOUT_PROPAGATOR(layernorm);
DECLARE_LAYOUT_PROPAGATOR(layernorm_bwd);
DECLARE_LAYOUT_PROPAGATOR(permute);
DECLARE_LAYOUT_PROPAGATOR(to_group);
DECLARE_LAYOUT_PROPAGATOR(from_group);
DECLARE_LAYOUT_PROPAGATOR(reshape);
DECLARE_LAYOUT_PROPAGATOR(transpose);
DECLARE_LAYOUT_PROPAGATOR(unsqueeze);
DECLARE_LAYOUT_PROPAGATOR(squeeze);
DECLARE_LAYOUT_PROPAGATOR(reorder);
DECLARE_LAYOUT_PROPAGATOR(mul_scales);
DECLARE_LAYOUT_PROPAGATOR(bn_folding);
DECLARE_LAYOUT_PROPAGATOR(conv_bwd_data);
DECLARE_LAYOUT_PROPAGATOR(conv_bwd_weights);
DECLARE_LAYOUT_PROPAGATOR(resampling);
DECLARE_LAYOUT_PROPAGATOR(resampling_bwd);
DECLARE_LAYOUT_PROPAGATOR(sum);
DECLARE_LAYOUT_PROPAGATOR(softmax);
DECLARE_LAYOUT_PROPAGATOR(softmax_bwd);
DECLARE_LAYOUT_PROPAGATOR(reduction);
DECLARE_LAYOUT_PROPAGATOR(constant_filler);
DECLARE_LAYOUT_PROPAGATOR(sub_zps);
DECLARE_LAYOUT_PROPAGATOR(add_zps);

#undef DECLARE_LAYOUT_PROPAGATOR

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
