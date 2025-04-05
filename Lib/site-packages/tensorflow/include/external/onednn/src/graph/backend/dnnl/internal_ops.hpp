/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

// We define those internal used operators in this file. For those operators
// defined on API can be found at src/interface/c_types_map.hpp.

#ifndef GRAPH_BACKEND_DNNL_INTERNAL_OPS_HPP
#define GRAPH_BACKEND_DNNL_INTERNAL_OPS_HPP

#include <string>
#include <vector>

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace op_kind {

// X(s, v):
// s will be the internal op kind value, can be accessed via graph::op_kind::s.
// v will be used to define the name string of each op kind.
#define INTERNAL_OPS \
    X(dnnl_mul_scales, Dnnl_mul_scales) \
    X(dnnl_constant_scales, Dnnl_constant_scales) \
    X(dnnl_add_zps, Dnnl_add_zps) \
    X(dnnl_sub_zps, Dnnl_sub_zps) \
    X(dnnl_constant_zps, Dnnl_constant_zps) \
    X(dnnl_permute, Dnnl_permute) \
    X(dnnl_to_group, Dnnl_to_group) \
    X(dnnl_from_group, Dnnl_from_group) \
    X(dnnl_unsqueeze, Dnnl_unsqueeze) \
    X(dnnl_squeeze, Dnnl_squeeze) \
    X(dnnl_reshape, Dnnl_reshape) \
    X(dnnl_transpose, Dnnl_transpose) \
    X(dnnl_convolution, Dnnl_convolution) \
    X(dnnl_convtranspose, Dnnl_convtranspose) \
    X(dnnl_pool, Dnnl_pool) \
    X(dnnl_bn_folding, Dnnl_bn_folding) \
    X(dnnl_conv_bwd_data, Dnnl_conv_bwd_data) \
    X(dnnl_batchnorm, Dnnl_batchnorm) \
    X(dnnl_binary, Dnnl_binary) \
    X(dnnl_eltwise, Dnnl_eltwise) \
    X(dnnl_eltwise_bwd, Dnnl_eltwise_bwd) \
    X(dnnl_shuffle, Dnnl_shuffle) \
    X(dnnl_sum, Dnnl_sum) \
    X(dnnl_reduction, Dnnl_reduction) \
    X(dnnl_prelu, Dnnl_prelu) \
    X(dnnl_prelu_bwd, Dnnl_prelu_bwd) \
    X(dnnl_batchnorm_bwd, Dnnl_batchnorm_bwd) \
    X(dnnl_softmax_bwd, Dnnl_softmax_bwd) \
    X(dnnl_logsoftmax_bwd, Dnnl_logsoftmax_bwd) \
    X(dnnl_resampling, Dnnl_resampling) \
    X(dnnl_resampling_bwd, Dnnl_resampling_bwd) \
    X(dnnl_concat, Dnnl_concat) \
    X(dnnl_layernorm_bwd, Dnnl_layernorm_bwd) \
    X(dnnl_conv_bwd_weights, Dnnl_conv_bwd_weights) \
    X(dnnl_pool_bwd, Dnnl_pool_bwd) \
    X(dnnl_matmul, Dnnl_matmul) \
    X(dnnl_softmax, Dnnl_softmax) \
    X(dnnl_logsoftmax, Dnnl_logsoftmax) \
    X(dnnl_layernorm, Dnnl_layernorm) \
    X(dnnl_reorder, Dnnl_reorder) \
    X(dnnl_convtranspose_bwd_data, Dnnl_convtranspose_bwd_data) \
    X(dnnl_convtranspose_bwd_weights, Dnnl_convtranspose_bwd_weights)

enum kind_t {
    kDNNL_INTERNAL_OP_STARTER = 0x1234,
#define X(s, v) k##v,
    INTERNAL_OPS
#undef X
};

#define X(s, v) const op_kind_t s = static_cast<op_kind_t>(k##v);
INTERNAL_OPS
#undef X

#define X(s, v) #v,
const std::vector<std::string> internal_op_strings = {INTERNAL_OPS};
#undef X

#undef INTERNAL_OPS

} // namespace op_kind
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
