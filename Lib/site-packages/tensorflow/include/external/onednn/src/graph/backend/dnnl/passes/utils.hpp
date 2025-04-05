/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_PASSES_UTILS_HPP
#define GRAPH_BACKEND_DNNL_PASSES_UTILS_HPP

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/value.hpp"

#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/subgraph.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// The pass_pipeline_t class is used to manage all transformation passes to run
// on a subgraph. Users should add passes need to run to the pipeline with a
// user defined order. And then call the run() method to run those added passes.
// After running each pass, the pipeline will choose to visualize the processed
// subgraph by using the visualizer.
class pass_pipeline_t {
public:
    using pass_signature
            = std::function<status_t(std::shared_ptr<subgraph_t> &)>;

    pass_pipeline_t() = default;

    pass_pipeline_t(const subgraph_visualizer_t &vis,
            bool enable_validator = true, bool enable_visualizer = true)
        : visualizer_(vis)
        , is_layout_sensitive_(false)
        , is_memory_sensitive_(false)
        , enable_validator_(enable_validator)
        , enable_visualizer_(enable_visualizer) {};

    // Reset the visualize arguments
    void reset_visualize_arg(
            bool is_layout_sensitive, bool is_memory_sensitive) {
        is_layout_sensitive_ = is_layout_sensitive;
        is_memory_sensitive_ = is_memory_sensitive;
    }

    // Add a pass to the pipeline. The current visualize arguments will be
    // recorded for the added pass and be used when visualize the subgraph
    // processed by this pass.
    void add_pass(const pass_signature &apass, const std::string &name) {
        passes_.emplace_back(apass);
        names_.emplace_back(name);
        is_layout_sensitives_.push_back(is_layout_sensitive_);
        is_memory_sensitives_.push_back(is_memory_sensitive_);
    }

    // Run all added passes
    status_t run(std::shared_ptr<subgraph_t> &sg) {
        status_t ret;
        for (size_t i = 0; i < passes_.size(); i++) {
            ret = passes_[i](sg);
            if (ret != status::success) { return ret; }

            // Dump the subgraph to dot file
            if (enable_visualizer_) {
                visualizer_.run(sg, names_[i], is_layout_sensitives_[i],
                        is_memory_sensitives_[i]);
            }

            // Validate the subgraph after each pass
            if (enable_validator_) { ret = validator_.run(sg); }
            if (ret != status::success) { return ret; }
        }
        return status::success;
    }

private:
    // The added passes and their names
    std::vector<pass_signature> passes_;
    std::vector<std::string> names_;

    // The recorded visualize arguments for each pass
    std::vector<bool> is_layout_sensitives_;
    std::vector<bool> is_memory_sensitives_;

    subgraph_visualizer_t visualizer_;
    subgraph_validator_t validator_;

    // The current visualize arguments
    bool is_layout_sensitive_;
    bool is_memory_sensitive_;

    bool enable_validator_;
    bool enable_visualizer_;
};

#define BACKEND_DNNL_ADD_PASS(pipeline, pass) pipeline.add_pass(pass, #pass)

status_t set_given_inputs_outputs(std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs);

status_t set_given_inputs_outputs(std::vector<std::shared_ptr<op_t>> &subgraph,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs);

void set_weight_bias_constant(std::shared_ptr<subgraph_t> &sg);

inline bool is_preprocess_op(op_t &op) {
    static const std::set<op_kind_t> preprocess_ops = {op_kind::dnnl_permute,
            op_kind::dnnl_to_group, op_kind::dnnl_from_group,
            op_kind::dnnl_unsqueeze, op_kind::dnnl_squeeze,
            op_kind::dnnl_reshape, op_kind::dnnl_transpose};
    return preprocess_ops.count(op.get_kind()) != 0;
}

void merge_common_eltwise_attrs(
        const std::shared_ptr<op_t> &org_op, std::shared_ptr<op_t> &new_op);

inline const std::map<op_kind_t, dnnl::algorithm> &get_eltwise_alg_map() {
    static const std::map<op_kind_t, dnnl::algorithm> &eltwise_alg_map = {
            {graph::op_kind::Abs, dnnl::algorithm::eltwise_abs},
            {graph::op_kind::Clamp, dnnl::algorithm::eltwise_clip_v2},
            {graph::op_kind::Elu, dnnl::algorithm::eltwise_elu},
            {graph::op_kind::Exp, dnnl::algorithm::eltwise_exp},
            {graph::op_kind::GELU, dnnl::algorithm::eltwise_gelu_erf},
            {graph::op_kind::HardSigmoid, dnnl::algorithm::eltwise_hardsigmoid},
            {graph::op_kind::HardSwish, dnnl::algorithm::eltwise_hardswish},
            {graph::op_kind::LeakyReLU, dnnl::algorithm::eltwise_relu},
            {graph::op_kind::Log, dnnl::algorithm::eltwise_log},
            {graph::op_kind::Mish, dnnl::algorithm::eltwise_mish},
            {graph::op_kind::ReLU, dnnl::algorithm::eltwise_relu},
            {graph::op_kind::Round, dnnl::algorithm::eltwise_round},
            {graph::op_kind::Sigmoid, dnnl::algorithm::eltwise_logistic},
            {graph::op_kind::Sqrt, dnnl::algorithm::eltwise_sqrt},
            {graph::op_kind::Square, dnnl::algorithm::eltwise_square},
            {graph::op_kind::Tanh, dnnl::algorithm::eltwise_tanh}};
    return eltwise_alg_map;
}

inline dnnl::algorithm get_eltwise_bwd_alg(op_kind_t kind, bool use_dst) {
    using algo = dnnl::algorithm;
    switch (kind) {
        case graph::op_kind::AbsBackward: return algo::eltwise_abs;
        case graph::op_kind::ClampBackward:
            if (use_dst) return algo::eltwise_clip_v2_use_dst_for_bwd;
            return algo::eltwise_clip_v2;
        case graph::op_kind::EluBackward:
            if (use_dst) return algo::eltwise_elu_use_dst_for_bwd;
            return algo::eltwise_elu;
        case graph::op_kind::GELUBackward: return algo::eltwise_gelu_erf;
        case graph::op_kind::HardSigmoidBackward:
            return algo::eltwise_hardsigmoid;
        case graph::op_kind::HardSwishBackward: return algo::eltwise_hardswish;
        case graph::op_kind::MishBackward: return algo::eltwise_mish;
        case graph::op_kind::ReLUBackward:
            if (use_dst) return algo::eltwise_relu_use_dst_for_bwd;
            return algo::eltwise_relu;
        case graph::op_kind::SigmoidBackward:
            if (use_dst) return algo::eltwise_logistic_use_dst_for_bwd;
            return algo::eltwise_logistic;
        case graph::op_kind::SqrtBackward:
            if (use_dst) return algo::eltwise_sqrt_use_dst_for_bwd;
            return algo::eltwise_sqrt;
        case graph::op_kind::TanhBackward:
            if (use_dst) return algo::eltwise_tanh_use_dst_for_bwd;
            return algo::eltwise_tanh;
        default: return algo::undef;
    }
}

inline const std::map<op_kind_t, dnnl::algorithm> &get_reduction_alg_map() {
    static const std::map<op_kind_t, dnnl::algorithm> &reduction_alg_map = {
            {graph::op_kind::ReduceL1,
                    dnnl::algorithm::reduction_norm_lp_power_p_sum},
            {graph::op_kind::ReduceL2, dnnl::algorithm::reduction_norm_lp_sum},
            {graph::op_kind::ReduceMax, dnnl::algorithm::reduction_max},
            {graph::op_kind::ReduceMean, dnnl::algorithm::reduction_mean},
            {graph::op_kind::ReduceMin, dnnl::algorithm::reduction_min},
            {graph::op_kind::ReduceProd, dnnl::algorithm::reduction_mul},
            {graph::op_kind::ReduceSum, dnnl::algorithm::reduction_sum},
    };
    return reduction_alg_map;
}

inline bool is_eltwise_kind(op_kind_t kind) {
    const std::set<op_kind_t> eltwise_kinds {
            graph::op_kind::Abs,
            graph::op_kind::Clamp,
            graph::op_kind::Elu,
            graph::op_kind::Exp,
            graph::op_kind::GELU,
            graph::op_kind::HardSigmoid,
            graph::op_kind::HardSwish,
            graph::op_kind::LeakyReLU,
            graph::op_kind::Log,
            graph::op_kind::Mish,
            graph::op_kind::ReLU,
            graph::op_kind::Round,
            graph::op_kind::Sigmoid,
            graph::op_kind::SoftPlus,
            graph::op_kind::Sqrt,
            graph::op_kind::Square,
            graph::op_kind::Tanh,
    };
    return eltwise_kinds.find(kind) != eltwise_kinds.end();
}

inline bool is_eltwise_bwd_kind(op_kind_t kind) {
    const std::set<op_kind_t> eltwise_bwd_kinds {
            graph::op_kind::AbsBackward,
            graph::op_kind::ClampBackward,
            graph::op_kind::EluBackward,
            graph::op_kind::GELUBackward,
            graph::op_kind::HardSigmoidBackward,
            graph::op_kind::HardSwishBackward,
            graph::op_kind::MishBackward,
            graph::op_kind::ReLUBackward,
            graph::op_kind::SigmoidBackward,
            graph::op_kind::SqrtBackward,
            graph::op_kind::TanhBackward,
    };
    return eltwise_bwd_kinds.find(kind) != eltwise_bwd_kinds.end();
}

inline bool is_binary_kind(op_kind_t kind) {
    const static std::set<op_kind_t> binary_kinds = {
            graph::op_kind::Add,
            graph::op_kind::Subtract,
            graph::op_kind::Multiply,
            graph::op_kind::Divide,
            graph::op_kind::Minimum,
            graph::op_kind::Maximum,
    };
    return binary_kinds.find(kind) != binary_kinds.end();
}

inline bool is_reduction_kind(op_kind_t kind) {
    const static std::set<op_kind_t> reduction_kinds = {
            graph::op_kind::ReduceL1,
            graph::op_kind::ReduceL2,
            graph::op_kind::ReduceMax,
            graph::op_kind::ReduceMean,
            graph::op_kind::ReduceMin,
            graph::op_kind::ReduceProd,
            graph::op_kind::ReduceSum,
    };
    return reduction_kinds.find(kind) != reduction_kinds.end();
}

std::vector<value_t *> get_constant_block_output_values(
        const std::shared_ptr<subgraph_t> &sg);

status_t infer_shape(std::shared_ptr<subgraph_t> &sg);

const std::map<op_kind_t, dnnl::algorithm> &get_binary_alg_map();

// (3, 4) * (3, 4) is doable
// (1, 4) * (3, 4) is doable
// (3, 4, 5) * (4, 5) is doable
// (3, 4, 5) * (1, 5) is doable
// (3, 4, 5) * (2, 4, 5) is NOT doable
bool binary_doable(
        const std::vector<dim_t> &shape_0, const std::vector<dim_t> &shape_1);

bool prelu_doable(const std::vector<dim_t> &src_dims,
        const std::vector<dim_t> &wei_dims, const std::string &data_format,
        const bool per_channel_broadcast);

// Checks whether chain of Reshape, Transpose, Reshape is fusible
// to dnnl_shuffle. Returns following pair:
// (is_fusible, (axis, groups))
// axis and groups store relevant information only when 'is_fusible = true'.
std::pair<bool, std::pair<size_t, int64_t>> shuffle_fusible(
        const op_t *reshape0, op_t *reshape1, op_t *transpose);

// For some shapes, post binary will run into oneDNN's ref path and has poor
// performance. So, we check the shape in this function and only make
// per_tensor, per_channel, per_mb_w(MatMul) and full tensor broadcast
// binary able to be fused.
bool post_binary_fusible(const op_t *base_op, const op_t *bin_op);

// oneDNN support post depthwise conv fusion. This function is used to check if
// two conv ops can be fused as a conv + depthwise pattern.
bool post_depthwise_conv_fusible(
        const op_t *base_conv_op, const op_t *post_conv_op);

// Get the map between base op kind and fusible post ops kinds. The map is
// determined by oneDNN's fusion capability and may change. For example, a
// dnnl_eltwise op can't fuse dnnl_eltwise op, but dnnl_convolution can.
const std::unordered_map<op_kind_t, std::unordered_set<op_kind_t>> &
get_post_ops_fusible_map();

std::string kind2str(op_kind_t kind);

// This function is used to check if a dnnl_reorder op is converted from or act
// as a TypeCast op. This function will only return true for a dnnl_reorder op
// which only has different input/output data type.
bool is_typecast(const op_t *op);

bool with_runtime_scales(const std::shared_ptr<op_t> &op,
        const fusion_info_mgr_t &mgr, bool is_input, size_t indice);

bool with_runtime_dst_scales(
        const std::shared_ptr<op_t> &op, const fusion_info_mgr_t &mgr);

bool with_runtime_zps(const std::shared_ptr<op_t> &op,
        const fusion_info_mgr_t &mgr, bool is_input, size_t indice);

// This function is used to check if a dnnl_reorder op is converted from or act
// as a Reorder op. This function will only return true for a dnnl_reorder op
// which is not for TypeCast or Quantization.
bool is_layout_reorder(const op_t *op);

// This function is used to clone a dnnl_mul_scales op
std::shared_ptr<op_t> clone_mul_scales(const std::shared_ptr<op_t> &scale_op);

// This function is used to inverse scales of a dnnl_mul_scales op
bool inverse_mul_scales(std::shared_ptr<op_t> &scale_op);

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
