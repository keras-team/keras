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
#ifndef GRAPH_BACKEND_DNNL_PATTERNS_UTILS_HPP
#define GRAPH_BACKEND_DNNL_PATTERNS_UTILS_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/value.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

template <int64_t N>
bool check_zps_values(op_t *op) {
    if (op->has_attr(op_attr::zps) == false) return true;
    auto zps = op->get_attr<std::vector<int64_t>>(op_attr::zps);
    return std::all_of(
            zps.begin(), zps.end(), [](int64_t i) { return i == N; });
}

template <size_t N>
bool check_input_num(op_t *op) {
    return op->num_inputs() == N;
}

template <size_t N>
bool check_output_num(op_t *op) {
    return op->num_outputs() == N;
}

template <data_type_t DTYPE>
bool check_input_dtype(op_t *op) {
    for (size_t i = 0; i < op->num_inputs(); ++i) {
        const logical_tensor_t &iport
                = op->get_input_value(i)->get_logical_tensor();
        if (iport.data_type != DTYPE) return false;
    }

    return true;
}

template <data_type_t DTYPE, size_t N>
bool check_input_dtype_from_offset(op_t *op) {
    if (N >= op->num_inputs()) return true;
    for (size_t i = N; i < op->num_inputs(); ++i) {
        const logical_tensor_t &iport
                = op->get_input_value(i)->get_logical_tensor();
        if (iport.data_type != DTYPE) return false;
    }

    return true;
}

template <dim N>
static inline bool check_conv_weight_size(op_t *op) {
    std::string weight_fmt = op->get_attr<std::string>(op_attr::weights_format);
    const logical_tensor_t &weight_lt
            = op->get_input_value(1)->get_logical_tensor();
    const auto weight_lt_wrapper = logical_tensor_wrapper_t(weight_lt);
    if (weight_lt_wrapper.ndims() == DNNL_GRAPH_UNKNOWN_NDIMS) { return false; }
    dims fil_sp = weight_lt_wrapper.get_weight_spatial_dims(weight_fmt);
    bool all_equal = std::all_of(
            fil_sp.begin(), fil_sp.end(), [](dim value) { return value == N; });
    return all_equal;
}

template <data_type_t DTYPE>
bool check_output_dtype(op_t *op) {
    for (size_t i = 0; i < op->num_outputs(); ++i) {
        const logical_tensor_t &oport
                = op->get_output_value(i)->get_logical_tensor();
        if (oport.data_type != DTYPE) return false;
    }

    return true;
}

template <size_t N>
bool check_producer_input_num(op_t *op) {
    op_t *producer = op->get_input_op(0);
    return producer->num_inputs() == N;
}

inline bool check_qtype_equal_to_per_tensor(op_t *op) {
    std::string qtype = op->get_attr<std::string>(op_attr::qtype);
    return qtype == "per_tensor";
}

inline bool check_begin_norm_axis_attr(const op_t *op) {
    const logical_tensor_t &src_lt
            = op->get_input_value(0)->get_logical_tensor();
    const auto src_lt_wrapper = logical_tensor_wrapper_t(src_lt);
    const auto ndims = src_lt_wrapper.ndims();

    if (op->has_attr(op_attr::begin_norm_axis)) {
        const auto begin_norm_axis
                = op->get_attr<int64_t>(op_attr::begin_norm_axis);
        if (ndims == DNNL_GRAPH_UNKNOWN_NDIMS) return begin_norm_axis == -1;
        return begin_norm_axis == -1 || begin_norm_axis == ndims - 1;
    }
    return true;
}

inline const std::vector<op_kind_t> &get_unary_ops() {
    const static std::vector<op_kind_t> unary = {
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
            graph::op_kind::Sigmoid,
            graph::op_kind::SoftPlus,
            graph::op_kind::ReLU,
            graph::op_kind::Round,
            graph::op_kind::Sqrt,
            graph::op_kind::Square,
            graph::op_kind::Tanh,
    };

    return unary;
}

inline const std::vector<op_kind_t> &get_unary_bwd_ops() {
    const static std::vector<op_kind_t> unary_bwd = {
            graph::op_kind::AbsBackward,
            graph::op_kind::ClampBackward,
            graph::op_kind::EluBackward,
            graph::op_kind::GELUBackward,
            graph::op_kind::HardSigmoidBackward,
            graph::op_kind::HardSwishBackward,
            graph::op_kind::MishBackward,
            graph::op_kind::SigmoidBackward,
            graph::op_kind::SoftPlusBackward,
            graph::op_kind::ReLUBackward,
            graph::op_kind::SqrtBackward,
            graph::op_kind::TanhBackward,
    };

    return unary_bwd;
}

inline const std::vector<op_kind_t> &get_binary_ops() {
    const static std::vector<op_kind_t> binary = {
            graph::op_kind::Add,
            graph::op_kind::Multiply,
            graph::op_kind::Maximum,
            graph::op_kind::Minimum,
            graph::op_kind::Divide,
            graph::op_kind::Subtract,
    };

    return binary;
}

inline const std::vector<op_kind_t> &get_unary_binary_ops() {
    const static std::vector<op_kind_t> unary_binary = {
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
            graph::op_kind::Sigmoid,
            graph::op_kind::SoftPlus,
            graph::op_kind::ReLU,
            graph::op_kind::Round,
            graph::op_kind::Sqrt,
            graph::op_kind::Square,
            graph::op_kind::Tanh,
            graph::op_kind::Add,
            graph::op_kind::Multiply,
            graph::op_kind::Maximum,
            graph::op_kind::Minimum,
            graph::op_kind::Divide,
            graph::op_kind::Subtract,
    };

    return unary_binary;
}

// Optional Quantize for weight will only be fused
// when:
// 1. input logical tensor has constant property type
// 2. the optional Quantize has a Wildcard producer
// 3. the optional Quantize has no producer
inline bool check_if_constant_weight(op_t *op) {
    const auto &in_value = op->get_input_value(0);
    if (in_value->get_logical_tensor().property
            == graph::property_type::constant) {
        return true;
    }
    if (in_value->has_producer()) {
        return in_value->get_producer().get_kind() == graph::op_kind::Wildcard;
    } else {
        return true;
    }
}

inline bool is_int8_quantization(const op_t *op) {
    const op_kind_t kind = op->get_kind();
    if (kind == graph::op_kind::Quantize) {
        const auto &out = op->get_output_value(0)->get_logical_tensor();
        return graph::utils::one_of(
                out.data_type, graph::data_type::s8, graph::data_type::u8);
    } else if (kind == graph::op_kind::Dequantize) {
        const auto &in = op->get_input_value(0)->get_logical_tensor();
        return graph::utils::one_of(
                in.data_type, graph::data_type::s8, graph::data_type::u8);
    } else {
        return false;
    }
}

// Optional BiasAdd after operator like Conv/ConvTranspose/Matmul. If
// `maybe_typecase` is true, there will also be an optional TypeCast before the
// 2nd input of BiasAdd.
inline graph::utils::pm::repetition_t *optional_bias_add(
        const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
        graph::utils::pm::pb_op_t *input, bool maybe_typecast = false) {
    auto popt_bias_graph = std::make_shared<graph::utils::pm::pb_graph_t>();
    graph::utils::pm::pb_op_t *pbias = nullptr;
    if (maybe_typecast) {
        auto popt_tc_graph = std::make_shared<graph::utils::pm::pb_graph_t>();
        graph::utils::pm::pb_op_t *typecast_bias
                = popt_tc_graph->append_op(graph::op_kind::TypeCast);
        typecast_bias->append_decision_function(
                check_output_dtype<graph::data_type::bf16>);
        popt_tc_graph->create_input_port(0, typecast_bias, 0);
        popt_tc_graph->create_output_port(0, typecast_bias, 0);
        auto popt_tc = popt_bias_graph->append_optional(popt_tc_graph);
        pbias = popt_bias_graph->append_op(graph::op_kind::BiasAdd,
                graph::utils::pm::in_edges_t {in_edge(1, popt_tc, 0)});
    } else {
        pbias = popt_bias_graph->append_op(graph::op_kind::BiasAdd);
    }
    pbias->append_decision_function(check_producer_input_num<2>);
    popt_bias_graph->create_input_port(0, pbias, 0);
    popt_bias_graph->create_output_port(0, pbias, 0);
    auto popt_bias = pgraph->append_optional(popt_bias_graph,
            graph::utils::pm::in_edges_t {in_edge(0, input, 0)});
    return popt_bias;
}

inline graph::utils::pm::repetition_t *post_quantized_add(
        const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
        graph::utils::pm::pb_node_t *input, bool check_zps = false) {
    graph::utils::pm::pb_op_t *pdequant_add
            = pgraph->append_op(graph::op_kind::Dequantize);
    pdequant_add->append_decision_function(is_int8_quantization);
    if (check_zps) pdequant_add->append_decision_function(check_zps_values<0>);
    graph::utils::pm::pb_op_t *padd = pgraph->append_op(graph::op_kind::Add,
            graph::utils::pm::in_edges_t {
                    in_edge(0, input, 0), in_edge(1, pdequant_add, 0)});

    // post ops
    auto postop_graph = std::make_shared<graph::utils::pm::pb_graph_t>();
    graph::utils::pm::pb_op_t *pop
            = postop_graph->append_alternation(get_unary_binary_ops());
    pop->allow_internal_inputs();
    postop_graph->create_input_port(0, pop, 0);
    postop_graph->create_input_port(1, pop, 1);
    postop_graph->create_output_port(0, pop, 0);

    auto prep = pgraph->append_repetition(postop_graph, {0, 0}, 0,
            MAX_REPETITION, graph::utils::pm::in_edges_t {in_edge(0, padd, 0)});
    return prep;
}

/*
    if optional_qout is true:
        pattern is [ [Multiply / Divide]* - Quantize]*
    else:
        pattern is [Multiply / Divide]* - Quantize
*/
inline graph::utils::pm::pb_node_t *optional_smooth_quant(
        const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
        graph::utils::pm::pb_node_t *input, bool optional_qout = false) {
    auto optional_graph = std::make_shared<graph::utils::pm::pb_graph_t>();
    graph::utils::pm::pb_op_t *smooth_op = optional_graph->append_alternation(
            {graph::op_kind::Multiply, graph::op_kind::Divide});
    optional_graph->create_input_port(0, smooth_op, 0);
    optional_graph->create_output_port(0, smooth_op, 0);
    auto popt_qout_graph = std::make_shared<graph::utils::pm::pb_graph_t>();
    auto p_curr_graph = optional_qout ? popt_qout_graph : pgraph;
    auto opt = optional_qout
            ? p_curr_graph->append_optional(optional_graph)
            : p_curr_graph->append_optional(optional_graph,
                    graph::utils::pm::in_edges_t {in_edge(0, input, 0)});
    graph::utils::pm::pb_op_t *quant_out
            = p_curr_graph->append_op(graph::op_kind::Quantize,
                    graph::utils::pm::in_edges_t {in_edge(0, opt, 0)});
    quant_out->append_decision_function(is_int8_quantization);
    if (optional_qout) {
        p_curr_graph->create_input_port(0, opt, 0);
        p_curr_graph->create_output_port(0, quant_out, 0);
        auto opt_qout = pgraph->append_optional(p_curr_graph,
                graph::utils::pm::in_edges_t {in_edge(0, input, 0)});
        return opt_qout;
    } else {
        return quant_out;
    }
}

// Optional Select
inline graph::utils::pm::repetition_t *optional_select(
        const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
        graph::utils::pm::pb_node_t *input, int input_index) {
    auto popt_select_graph = std::make_shared<graph::utils::pm::pb_graph_t>();

    graph::utils::pm::pb_op_t *select_op
            = popt_select_graph->append_op(graph::op_kind::Select);

    popt_select_graph->create_input_port(0, select_op, 0);
    popt_select_graph->create_input_port(1, select_op, 1);
    popt_select_graph->create_input_port(2, select_op, 2);
    popt_select_graph->create_output_port(0, select_op, 0);
    auto pselect = pgraph->append_optional(popt_select_graph,
            graph::utils::pm::in_edges_t {in_edge(input_index, input, 0)});
    return pselect;
}

// Optional (transpose + reorder/staticReshape)
inline graph::utils::pm::repetition_t *optional_transpose_reshape(
        const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
        graph::utils::pm::pb_node_t *input, int input_index) {
    auto popt_graph = std::make_shared<graph::utils::pm::pb_graph_t>();

    graph::utils::pm::pb_op_t *transpose
            = popt_graph->append_op(graph::op_kind::StaticTranspose);
    graph::utils::pm::pb_op_t *reshape_out = popt_graph->append_alternation(
            {graph::op_kind::Reorder, graph::op_kind::StaticReshape},
            {in_edge(0, transpose, 0)});
    popt_graph->create_input_port(0, transpose, 0);
    popt_graph->create_output_port(0, reshape_out, 0);
    auto popt_transpose_reshape = pgraph->append_optional(popt_graph,
            graph::utils::pm::in_edges_t {in_edge(input_index, input, 0)});
    return popt_transpose_reshape;
}

inline graph::utils::pm::pb_node_t *create_dequant_matmul(
        const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
        graph::utils::pm::pb_node_t *input, bool is_bf16 = false,
        bool is_int8 = false) {
    graph::utils::pm::in_edges_t in_edges;
    if (input) {
        in_edges = graph::utils::pm::in_edges_t {in_edge(0, input, 0)};
    }
    if (is_int8) {
        auto dequantize_A
                = pgraph->append_op(graph::op_kind::Dequantize, in_edges);
        auto dequantize_B = pgraph->append_op(graph::op_kind::Dequantize);
        if (is_bf16) {
            auto typecast_A = pgraph->append_op(
                    graph::op_kind::TypeCast, {in_edge(0, dequantize_A, 0)});
            auto typecast_B = pgraph->append_op(
                    graph::op_kind::TypeCast, {in_edge(0, dequantize_B, 0)});
            in_edges = graph::utils::pm::in_edges_t {
                    in_edge(0, typecast_A, 0), in_edge(1, typecast_B, 0)};
        } else {
            in_edges = graph::utils::pm::in_edges_t {
                    in_edge(0, dequantize_A, 0), in_edge(1, dequantize_B, 0)};
        }
    }
    auto matmul = pgraph->append_op(graph::op_kind::MatMul, in_edges);
    return matmul;
}

// only for single input and single output op
inline graph::utils::pm::pb_node_t *append_siso_repetition_subgraph(
        const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
        graph::op_kind_t kind, graph::utils::pm::pb_node_t *input,
        int rep_min = 0, int rep_max = 2) {
    graph::utils::pm::in_edges_t in_edges;
    if (input) {
        in_edges = graph::utils::pm::in_edges_t {in_edge(0, input, 0)};
    }
    auto rep_subgraph = std::make_shared<graph::utils::pm::pb_graph_t>();
    auto single_op = rep_subgraph->append_op(kind);
    rep_subgraph->create_input_port(0, single_op, 0);
    rep_subgraph->create_output_port(0, single_op, 0);
    auto rep = pgraph->append_repetition(
            rep_subgraph, {0, 0}, rep_min, rep_max, in_edges);
    return rep;
}

inline graph::utils::pm::pb_node_t *append_optional_typecast_quantize(
        const std::shared_ptr<graph::utils::pm::pb_graph_t> &pgraph,
        graph::utils::pm::pb_node_t *input, bool is_bf16 = false) {
    auto subgraph = std::make_shared<graph::utils::pm::pb_graph_t>();
    graph::utils::pm::in_edges_t in_edges;
    graph::utils::pm::pb_node_t *subgraph_in_node;
    if (is_bf16) {
        auto typecast_output = subgraph->append_op(graph::op_kind::TypeCast);
        in_edges
                = graph::utils::pm::in_edges_t {in_edge(0, typecast_output, 0)};
        subgraph_in_node = typecast_output;
    }
    auto quantize = subgraph->append_op(graph::op_kind::Quantize, in_edges);
    if (!is_bf16) { subgraph_in_node = quantize; }
    subgraph->create_input_port(0, subgraph_in_node, 0);
    subgraph->create_output_port(0, quantize, 0);
    auto output = pgraph->append_optional(subgraph, {in_edge(0, input, 0)});
    return output;
}

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
