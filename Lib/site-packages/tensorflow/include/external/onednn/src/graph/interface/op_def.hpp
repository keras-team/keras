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
#ifndef GRAPH_INTERFACE_OP_DEF_HPP
#define GRAPH_INTERFACE_OP_DEF_HPP

#include <limits>
#include <set>
#include <vector>

#include "graph/interface/op_def_constraint.hpp"
#include "graph/interface/op_schema.hpp"
#include "graph/interface/shape_infer.hpp"

namespace dnnl {
namespace impl {
namespace graph {

DNNL_GRAPH_OP_SCHEMA(Abs, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(AbsBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Add, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "src_0", "T")
                .set_input(1, "src_1", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(AvgPool, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::exclude_pad, true, attribute_kind::b)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::rounding_type, false, attribute_kind::s,
                        "floor")
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_pool_output_shape)
                .set_op_def_constraint_function(check_pads))

DNNL_GRAPH_OP_SCHEMA(AvgPoolBackward, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_input(1, "src_shape", "T1")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::exclude_pad, true, attribute_kind::b)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::src_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T1", {data_type::s32})
                .set_shape_inference_function(infer_pool_bwd_output_shape)
                .set_op_def_constraint_function(check_avgpool_bwd_input_shape)
                .set_op_def_constraint_function(check_pads))

DNNL_GRAPH_OP_SCHEMA(BatchNormInference, 1,
        op_schema_t()
                .set_num_inputs(5)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "gamma", "T2")
                .set_input(2, "beta", "T2")
                .set_input(3, "mean", "T2")
                .set_input(4, "variance", "T2")
                .set_output(0, "dst", "T1")
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_bn_data_type))

DNNL_GRAPH_OP_SCHEMA(BatchNormForwardTraining, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_num_outputs(5)
                .set_input(0, "src", "T1")
                .set_input(1, "mean", "T2")
                .set_input(2, "variance", "T2")
                .set_input(3, "gamma", "T2")
                .set_input(4, "beta", "T2")
                .set_output(0, "dst", "T1")
                .set_output(1, "running_mean", "T2")
                .set_output(2, "running_variance", "T2")
                .set_output(3, "batch_mean", "T2")
                .set_output(4, "batch_variance", "T2")
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::momentum, false, attribute_kind::f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_bn_fwd_train_output_shape)
                .set_op_def_constraint_function(check_bn_data_type))

DNNL_GRAPH_OP_SCHEMA(BatchNormTrainingBackward, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 2, 3}))
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "mean", "T2")
                .set_input(3, "variance", "T2")
                .set_input(4, "gamma", "T2")
                .set_output(0, "diff_src", "T1")
                .set_output(1, "diff_gamma", "T2")
                .set_output(2, "diff_beta", "T2")
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_bn_bwd_output_shape)
                .set_op_def_constraint_function(check_bn_data_type))

DNNL_GRAPH_OP_SCHEMA(BiasAdd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "bias", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_bias_add_output_shape))

DNNL_GRAPH_OP_SCHEMA(BiasAddBackward, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_output(0, "diff_bias", "T")
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_bias_backprop_output_shape))

DNNL_GRAPH_OP_SCHEMA(Clamp, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::min, true, attribute_kind::f)
                .set_attr(op_attr::max, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(ClampBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::min, true, attribute_kind::f)
                .set_attr(op_attr::max, true, attribute_kind::f)
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Concat, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 64}))
                .set_num_outputs(1)
                .set_input(0, "src_i", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::axis, true, attribute_kind::i)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_concat_output_shape))

DNNL_GRAPH_OP_SCHEMA(Convolution, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "weights", "T")
                .set_input(2, "bias", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvolutionBackwardData, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T1")
                .set_input(1, "weights", "T1")
                .set_input(2, "dst_shape", "T2")
                .set_output(0, "diff_src", "T1")
                .set_attr(op_attr::output_padding, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_attr(op_attr::dst_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_conv_bprop_data_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_op_def_constraint_function(
                        check_conv_bwd_data_output_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvolutionBackwardWeights, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "weights_shape", "T2")
                .set_output(0, "diff_weights", "T1")
                .set_attr(op_attr::weights_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_conv_bprop_filters_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_op_def_constraint_function(
                        check_conv_bwd_weights_weights_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTranspose, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "weights", "T")
                .set_input(2, "bias", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::output_padding, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_convtranspose_output_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONVTRANSPOSE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTransposeBackwardData, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_input(1, "weights", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_convtranspose_bprop_data_output_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONVTRANSPOSE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTransposeBackwardWeights, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "weights_shape", "T2")
                .set_output(0, "diff_weights", "T1")
                .set_attr(op_attr::weights_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_convtranspose_bprop_filters_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_op_def_constraint_function(
                        check_conv_bwd_weights_weights_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONVTRANSPOSE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Divide, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src_0", "T1")
                .set_input(1, "src_1", "T2")
                .set_output(0, "dst", "T3")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T3", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Elu, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(EluBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(End, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(0)
                .set_input(0, "src", "T")
                .set_type_constraints("T",
                        {data_type::f32, data_type::f16, data_type::bf16,
                                data_type::s8, data_type::u8, data_type::s32,
                                data_type::undef}))

DNNL_GRAPH_OP_SCHEMA(Exp, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GELU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GELUBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSigmoid, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_attr(op_attr::beta, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSigmoidBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_attr(op_attr::beta, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSwish, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSwishBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Interpolate, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "sizes", "T2")
                .set_output(0, "dst", "T1")
                .set_attr(op_attr::mode, true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::sizes, false, attribute_kind::is)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::coordinate_transformation_mode, false,
                        attribute_kind::s, "half_pixel",
                        {"half_pixel", "align_corners"})
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_interpolate_output_shape)
                .set_op_def_constraint_function(check_interpolate_sizes_scales))

DNNL_GRAPH_OP_SCHEMA(InterpolateBackward, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "sizes", "T2")
                .set_output(0, "diff_src", "T1")
                .set_attr(op_attr::mode, true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::coordinate_transformation_mode, false,
                        attribute_kind::s, "half_pixel",
                        {"half_pixel", "align_corners"})
                .set_attr(op_attr::sizes, false, attribute_kind::is)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_interpolate_sizes_scales))

DNNL_GRAPH_OP_SCHEMA(LayerNorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
                .set_input(0, "src", "T1")
                .set_input(1, "gamma", "T2")
                .set_input(2, "beta", "T2")
                .set_output(0, "dst", "T1")
                .set_output(1, "mean", "T2")
                .set_output(2, "variance", "T2")
                .set_attr(op_attr::keep_stats, false, attribute_kind::b, true)
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_norm_output_shape)
                .set_op_def_constraint_function(check_ln_data_type)
                .set_op_def_constraint_function(check_ln_fwd_outputs_num))

DNNL_GRAPH_OP_SCHEMA(LayerNormBackward, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5, 6}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "mean", "T2")
                .set_input(3, "variance", "T2")
                .set_input(4, "gamma", "T2")
                .set_input(5, "beta", "T2")
                .set_output(0, "diff_src", "T1")
                .set_output(1, "diff_gamma", "T2")
                .set_output(2, "diff_beta", "T2")
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_norm_bprop_output_shape)
                .set_op_def_constraint_function(check_ln_data_type)
                .set_op_def_constraint_function(check_ln_bwd_use_affine))

DNNL_GRAPH_OP_SCHEMA(LeakyReLU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Log, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogSoftmax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(-1))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogSoftmaxBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_input(1, "dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(-1))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(MatMul, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "weights", "T")
                .set_input(2, "bias", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Maximum, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "src_0", "T")
                .set_input(1, "src_1", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(MaxPool, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::dilations, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::rounding_type, false, attribute_kind::s,
                        "floor")
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_pool_output_shape)
                .set_op_def_constraint_function(check_pads)
                .set_op_def_constraint_function(check_maxpool_dilations))

DNNL_GRAPH_OP_SCHEMA(MaxPoolBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_attr(op_attr::dilations, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_pool_bwd_output_shape)
                .set_op_def_constraint_function(check_pads)
                .set_op_def_constraint_function(check_maxpool_dilations))

DNNL_GRAPH_OP_SCHEMA(Minimum, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "src_0", "T")
                .set_input(1, "src_1", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Mish, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(MishBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

// TODO(Yixin): for Multiply. input and output needs to have the same dtypes
// But in current pytorch bridge's type promotion system, there's no
// such constraints. So this feature is postponed.
DNNL_GRAPH_OP_SCHEMA(Multiply, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "src_0", "T1")
                .set_input(1, "src_1", "T2")
                .set_output(0, "dst", "T3")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T3", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Pow, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::beta, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(PReLU, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "slope", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::per_channel_broadcast, false,
                        attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(PReLUBackward, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(2)
                .set_input(0, "src", "T")
                .set_input(1, "slope", "T")
                .set_input(2, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_output(1, "diff_slope", "T")
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_prelu_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReduceL1, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceL2, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMax, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMean, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMin, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceProd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceSum, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReLU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReLUBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Round, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Select, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "cond", "T1")
                .set_input(1, "src_0", "T2")
                .set_input(2, "src_1", "T2")
                .set_output(0, "dst", "T2")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints("T1", {data_type::boolean})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_select_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sigmoid, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SigmoidBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftMax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftMaxBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_input(1, "dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftPlus, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::beta, false, attribute_kind::f, 1.f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftPlusBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::beta, false, attribute_kind::f, 1.f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sqrt, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SqrtBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Square, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SquaredDifference, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src_0", "T")
                .set_input(1, "src_1", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Subtract, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src_0", "T")
                .set_input(1, "src_1", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Tanh, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(TanhBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Wildcard, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>(
                        {0, std::numeric_limits<size_t>::max()}))
                .set_outputs_option(op_schema_t::param_num_option::variadic)
                .set_num_outputs(std::set<size_t>(
                        {0, std::numeric_limits<size_t>::max()}))
                .set_input(0, "src", "any")
                .set_output(0, "dst", "any")
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(Quantize, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_output(0, "dst", "T2")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::scales, true, attribute_kind::fs)
                // for symmetric quantization or fp8 quantization, zps is not required.
                .set_attr(op_attr::zps, false, attribute_kind::is)
                .set_type_constraints("T1", {data_type::f32})
                .set_type_constraints("T2",
                        {data_type::u8, data_type::s8, data_type::f8_e5m2,
                                data_type::f8_e4m3})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_quant_dequant_scales_zps))

DNNL_GRAPH_OP_SCHEMA(Dequantize, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_output(0, "dst", "T2")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::scales, true, attribute_kind::fs)
                // for symmetric quantization or fp8 quantization, zps is not required.
                .set_attr(op_attr::zps, false, attribute_kind::is)
                .set_type_constraints("T1",
                        {data_type::u8, data_type::s8, data_type::f8_e5m2,
                                data_type::f8_e4m3})
                .set_type_constraints("T2", {data_type::f32})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_quant_dequant_scales_zps))

DNNL_GRAPH_OP_SCHEMA(Reorder, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(TypeCast, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_output(0, "dst", "T2")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_typecast_data_type))

DNNL_GRAPH_OP_SCHEMA(StaticReshape, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::shape, true, attribute_kind::is)
                .set_attr(op_attr::special_zero, true, attribute_kind::b)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_static_reshape_output_shape))

DNNL_GRAPH_OP_SCHEMA(StaticTranspose, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::order, true, attribute_kind::is)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_static_transpose_output_shape))

DNNL_GRAPH_OP_SCHEMA(DynamicQuantize, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "scales", "T1")
                .set_input(2, "zps", "T2")
                .set_output(0, "dst", "T3")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_type_constraints("T1", {data_type::f32})
                .set_type_constraints(
                        "T2", {data_type::u8, data_type::s8, data_type::s32})
                .set_type_constraints("T3", {data_type::u8, data_type::s8})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(
                        check_dyn_quant_dequant_scales_zps))

DNNL_GRAPH_OP_SCHEMA(DynamicDequantize, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "scales", "T2")
                .set_input(2, "zps", "T3")
                .set_output(0, "dst", "T2")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_type_constraints("T1", {data_type::u8, data_type::s8})
                .set_type_constraints("T2", {data_type::f32})
                .set_type_constraints(
                        "T3", {data_type::u8, data_type::s8, data_type::s32})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(
                        check_dyn_quant_dequant_scales_zps))

DNNL_GRAPH_OP_SCHEMA(Reciprocal, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
