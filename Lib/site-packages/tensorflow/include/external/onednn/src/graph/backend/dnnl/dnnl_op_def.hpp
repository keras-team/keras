/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_DNNL_OP_DEF_HPP
#define GRAPH_BACKEND_DNNL_DNNL_OP_DEF_HPP

#include <limits>
#include <set>
#include <vector>

#include "graph/interface/op_schema.hpp"
#include "graph/interface/shape_infer.hpp"

#include "graph/backend/dnnl/dnnl_shape_infer.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/layout_propagator.hpp"
#include "graph/backend/dnnl/op_executable.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

#define SET_ATTR_IS_CONSTANT \
    set_attr(op_attr::is_constant, false, attribute_kind::b, false)

#define SET_EXECUTABLE_CREATOR(func) \
    set_additional_item<executable_creator_func>("executable_creator", {func})

#define SET_ARG_INDICES_GETTER(executable_class) \
    set_additional_item<arg_indices_getter_func>( \
            "arg_indices_getter", {executable_class::get_arg_indices})

#define SET_LAYOUT_PROPAGATOR(func) \
    set_additional_item<layout_propagator_func>("layout_propagator", {func})

#define SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS \
    set_attr(op_attr::strides, true, attribute_kind::is) \
            .set_attr(op_attr::pads_begin, true, attribute_kind::is) \
            .set_attr(op_attr::pads_end, true, attribute_kind::is) \
            .set_attr(op_attr::dilations, true, attribute_kind::is) \
            .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None", \
                    {"None", "SAME_UPPER", "SAME_LOWER", "VALID"}) \
            .set_attr(op_attr::groups, false, attribute_kind::i, (int64_t)1) \
            .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC", \
                    {"NXC", "NCX"}) \
            .set_attr(op_attr::weights_format, false, attribute_kind::s, \
                    "XOI", {"XOI", "IOX", "OIX"})

template <typename T>
op_schema_t get_op_schema();

DNNL_GRAPH_OP_SCHEMA(dnnl_mul_scales, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "x")
                .set_input(1, "scales")
                .set_output(0, "y")
                .set_output(1, "scratchpad")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::scales, false, attribute_kind::fs,
                        std::vector<float>())
                .set_attr(op_attr::with_runtime_scales, false,
                        attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_mul_scales)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<reorder_executable_t>)
                .SET_ARG_INDICES_GETTER(reorder_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_constant_scales, 1,
        op_schema_t()
                .set_num_inputs(0)
                .set_num_outputs(1)
                .set_output(0, "output")
                .set_attr(op_attr::scales, true, attribute_kind::fs)
                .set_attr(op_attr::shape, true,
                        attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_constant_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_constant_filler)
                .SET_EXECUTABLE_CREATOR(executable_creator<const_scales_filler>)
                .SET_ARG_INDICES_GETTER(const_scales_filler))

DNNL_GRAPH_OP_SCHEMA(dnnl_add_zps, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_input(1, "zps")
                .set_output(0, "y")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::zps, false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr(op_attr::with_runtime_zps, false, attribute_kind::b,
                        false)
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_add_zps)
                .SET_EXECUTABLE_CREATOR(dummy_executable_creator)
                .set_additional_item<arg_indices_getter_func>(
                        "arg_indices_getter", {dummy_arg_indices_getter}))

DNNL_GRAPH_OP_SCHEMA(dnnl_sub_zps, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_input(1, "zps")
                .set_output(0, "y")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::zps, false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr(op_attr::with_runtime_zps, false, attribute_kind::b,
                        false)
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_sub_zps)
                .SET_EXECUTABLE_CREATOR(dummy_executable_creator)
                .set_additional_item<arg_indices_getter_func>(
                        "arg_indices_getter", {dummy_arg_indices_getter}))

DNNL_GRAPH_OP_SCHEMA(dnnl_constant_zps, 1,
        op_schema_t()
                .set_num_inputs(0)
                .set_num_outputs(1)
                .set_output(0, "output")
                .set_attr(op_attr::zps, true, attribute_kind::is)
                .set_attr(op_attr::shape, true,
                        attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_constant_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_constant_filler)
                .SET_EXECUTABLE_CREATOR(executable_creator<const_zps_filler>)
                .SET_ARG_INDICES_GETTER(const_zps_filler))

// The logical axes will be permuted in the following manner:
// for (i = 0; i < ndims(); i++)
//     new_desc.dims()[permutation[i]] = dims()[i];
//
// Note: the permutation attr in dnnl_permute is quite different from the order
// attr in dnnl_transpose. The later one is inherited from StaticTranspose op
// and are used in the following manner:
// for (i = 0; i < ndims(); i++)
//     new_desc.dims()[i] = dims()[order[i]];
DNNL_GRAPH_OP_SCHEMA(dnnl_permute, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::permutation, false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_permute_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_permute)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_to_group, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::groups, false, attribute_kind::i, (int64_t)1)
                .set_attr(op_attr::is_convtranspose, false, attribute_kind::b,
                        false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_to_group_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_to_group)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

// This op is used for grouped conv/deconv backward weight to convert a [g,
// oc/g, ic, kh, kw] shaped weight tensor to a [oc, ic, kh, kw] weight tensor.
// The former shaped weight tensor is required by oneDNN primitive, but the
// later one is required by oneDNN Graph users
DNNL_GRAPH_OP_SCHEMA(dnnl_from_group, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::groups, false, attribute_kind::i, (int64_t)1)
                .set_attr(op_attr::is_convtranspose, false, attribute_kind::b,
                        false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_from_group_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_from_group)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_unsqueeze, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::axes, false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache

                .set_shape_inference_function(infer_unsqueeze_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_unsqueeze)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_squeeze, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::axes, false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_squeeze_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_squeeze)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_reshape, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data")
                .set_output(0, "output")
                .set_attr(op_attr::shape, true, attribute_kind::is)
                .set_attr(op_attr::special_zero, true, attribute_kind::b)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_static_reshape_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_reshape)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_transpose, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data")
                .set_output(0, "output")
                .set_attr(op_attr::order, true, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(
                        infer_static_transpose_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_transpose)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_convolution, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "filter")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from Convolution.
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::with_bias, false, attribute_kind::b, false)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_conv_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_conv)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<conv_fwd_executable_t>)
                .SET_ARG_INDICES_GETTER(conv_fwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_convtranspose, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "weight")
                .set_input(2, "bias")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from ConvTranspose.
                .set_attr(op_attr::output_padding, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::with_bias, false, attribute_kind::b, false)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_deconv)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<deconv_fwd_executable_t>)
                .SET_ARG_INDICES_GETTER(deconv_fwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_convtranspose_bwd_data, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "output_delta")
                .set_input(1, "filter")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                // Attributes inherited from ConvTransposeBackwardData.
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_bwd_data_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_deconv_bwd_data)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<deconv_bwd_data_executable_t>)
                .SET_ARG_INDICES_GETTER(deconv_bwd_data_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_convtranspose_bwd_weights, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "output_delta")
                .set_input(2, "filter_shape")
                .set_output(0, "filter_delta")
                .set_output(1, "scratchpad")
                // Attributes inherited from ConvTransposeBackwardWeights.
                .set_attr(op_attr::weights_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_bwd_weight_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_deconv_bwd_weights)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<deconv_bwd_weights_executable_t>)
                .SET_ARG_INDICES_GETTER(deconv_bwd_weights_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_pool, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3}))
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                .set_output(2, "workspace")
                // Attributes inherited from MaxPool and AvgPool.
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::exclude_pad, false, attribute_kind::b)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::dilations, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::rounding_type, false, attribute_kind::s,
                        "floor")
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::kind, true, attribute_kind::s)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::is_training, false, attribute_kind::b)
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_pool_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_pool)
                .SET_EXECUTABLE_CREATOR(executable_creator<pool_executable_t>)
                .SET_ARG_INDICES_GETTER(pool_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_pool_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_num_outputs(2)
                .set_input(0, "output_delta")
                .set_input(1, "output_forward_indices")
                .set_input(2, "forward_src")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::exclude_pad, false, attribute_kind::b)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_attr(op_attr::dilations, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::src_shape, true, attribute_kind::is)
                // New added attributes
                .set_attr(op_attr::kind, true, attribute_kind::s)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_pool_bwd_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_pool_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<pool_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(pool_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_prelu, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "data")
                .set_input(1, "slope")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from PReLU
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::per_channel_broadcast, false,
                        attribute_kind::b, true)
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_prelu)
                .SET_EXECUTABLE_CREATOR(executable_creator<prelu_executable_t>)
                .SET_ARG_INDICES_GETTER(prelu_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_prelu_bwd, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(3)
                .set_input(0, "input_forward")
                .set_input(1, "slope")
                .set_input(2, "output_delta")
                .set_output(0, "input_delta")
                .set_output(1, "slope_delta")
                .set_output(2, "scratchpad")
                // Attributes inherited from PReLUBackward
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_prelu_bwd_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_prelu_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<prelu_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(prelu_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_bn_folding, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({5, 6}))
                .set_num_outputs(3)
                .set_input(0, "weight")
                .set_input(1, "bias")
                .set_input(2, "gamma")
                .set_input(3, "beta")
                .set_input(4, "mean")
                .set_input(5, "variance")
                .set_output(0, "updated_weight")
                .set_output(1, "updated_bias")
                .set_output(2, "scratchpad")
                // No corresponding frontend op
                // Attributes
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::with_bias, false, attribute_kind::b, false)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::weights_format, false, attribute_kind::s,
                        "XIO", {"XIO", "OIX"})
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_bn_folding_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_bn_folding)
                .SET_EXECUTABLE_CREATOR(executable_creator<bn_folding_t>)
                .SET_ARG_INDICES_GETTER(bn_folding_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_conv_bwd_data, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "weight")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from ConvolutionBackwardData.
                .set_attr(op_attr::output_padding, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_attr(op_attr::dst_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_bwd_data_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_conv_bwd_data)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<conv_bwd_data_executable_t>)
                .SET_ARG_INDICES_GETTER(conv_bwd_data_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_conv_bwd_weights, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "output_delta")
                .set_output(0, "weight_delta")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::weights_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_bwd_weight_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_conv_bwd_weights)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<conv_bwd_weights_executable_t>)
                .SET_ARG_INDICES_GETTER(conv_bwd_weights_executable_t))

// Note: if `is_training` is False, the `gamma` and `beta` are the second and
// third input (required), while `is_training` is True, the `gamma` and `beta`
// are the last two inputs (optional).
DNNL_GRAPH_OP_SCHEMA(dnnl_batchnorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3, 6, 7}))
                .set_input(0, "input")
                .set_input(1, "gamma")
                .set_input(2, "beta")
                .set_input(3, "mean")
                .set_input(4, "variance")
                .set_output(0, "output")
                .set_output(1, "running mean")
                .set_output(2, "running variance")
                .set_output(3, "batch mean")
                .set_output(4, "batch variance")
                .set_output(5, "scratchpad")
                .set_output(6, "workspace")
                // Attributes inherited from BatchNormInference and
                // BatchNormForwardTraining op
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::momentum, false, attribute_kind::f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::is_training, false, attribute_kind::b)
                .set_attr(op_attr::fuse_relu, false, attribute_kind::b)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_batchnorm_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_batchnorm)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<batchnorm_executable_t>)
                .SET_ARG_INDICES_GETTER(batchnorm_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_batchnorm_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3, 4}))
                .set_input(0, "input")
                .set_input(1, "output_delta")
                .set_input(2, "mean")
                .set_input(3, "variance")
                .set_input(4, "gamma")
                .set_output(0, "input_delta")
                .set_output(1, "gamma_delta")
                .set_output(2, "beta_delta")
                .set_output(3, "scratchpad")
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(
                        infer_dnnl_batchnorm_bwd_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_batchnorm_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<batchnorm_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(batchnorm_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_resampling_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(2)
                .set_input(0, "data")
                .set_input(1, "output_delta")
                .set_input(2, "sizes")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::mode, true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::coordinate_transformation_mode, false,
                        attribute_kind::s, "half_pixel",
                        {"half_pixel", "align_corners"})
                .set_attr(op_attr::sizes, false, attribute_kind::is)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_resampling_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<resampling_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(resampling_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_sum, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_sum)
                .SET_EXECUTABLE_CREATOR(executable_creator<sum_executable_t>)
                .SET_ARG_INDICES_GETTER(sum_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_binary, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "a")
                .set_input(1, "b")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from front binary ops (Add, Multiply,
                // ...).
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                // Attributes inherited from front BiasAdd ops, will only take
                // effect when is_bias_add attr is true
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::is_bias_add, false, attribute_kind::b, false)
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::alg_kind, true, attribute_kind::i)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_binary_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_binary)
                .SET_EXECUTABLE_CREATOR(executable_creator<binary_executable_t>)
                .SET_ARG_INDICES_GETTER(binary_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_eltwise, 1,
        op_schema_t()
                // dnnl_eltwise can fuse dnnl_binary, so its input number is
                // variadic
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from front eltwise ops
                .set_attr(op_attr::alpha, false, attribute_kind::f, 0.f)
                .set_attr(op_attr::beta, false, attribute_kind::f, 0.f)
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::alg_kind, true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_eltwise)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<eltwise_executable_t>)
                .SET_ARG_INDICES_GETTER(eltwise_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_eltwise_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "forward_data")
                .set_input(1, "output_delta")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::alpha, false, attribute_kind::f, 0.f)
                .set_attr(op_attr::beta, false, attribute_kind::f, 0.f)
                .set_attr(op_attr::use_dst, false, attribute_kind::b, false)
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::alg_kind, true, attribute_kind::i)
                .set_attr(op_attr::fwd_alg_kind, true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_eltwise_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<eltwise_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(eltwise_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_shuffle, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // No corresponding frontend op
                // Attributes
                .set_attr(op_attr::axis, true, attribute_kind::i)
                .set_attr(op_attr::groups, true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_shuffle)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<shuffle_executable_t>)
                .SET_ARG_INDICES_GETTER(shuffle_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_reduction, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "axes")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from front reduction ops
                .SET_REDUCE_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::alg_kind, true, attribute_kind::i)
                .set_attr(op_attr::p, false, attribute_kind::f, 0.0f)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_reduction)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<reduction_executable_t>)
                .SET_ARG_INDICES_GETTER(reduction_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_softmax_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "output_delta")
                .set_input(1, "forward_result")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                // Attributes inherited from SoftMaxBackward
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_softmax_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<softmax_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(softmax_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_logsoftmax_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "output_delta")
                .set_input(1, "forward_result")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                // Attributes inherited from LogSoftmaxBackward
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)-1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_softmax_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<softmax_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(softmax_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_resampling, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(2)
                .set_input(0, "data")
                .set_input(1, "sizes")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from Interpolate.
                .set_attr(op_attr::mode, true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::sizes, false, attribute_kind::is)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::coordinate_transformation_mode, false,
                        attribute_kind::s, "half_pixel",
                        {"half_pixel", "align_corners"})
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_interpolate_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_resampling)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<resampling_executable_t>)
                .SET_ARG_INDICES_GETTER(resampling_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_concat, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 64}))
                .set_num_outputs(2)
                .set_input(0, "a")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from Concat
                .set_attr(op_attr::axis, true, attribute_kind::i)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_concat_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_concat)
                .SET_EXECUTABLE_CREATOR(executable_creator<concat_executable_t>)
                .SET_ARG_INDICES_GETTER(concat_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_layernorm_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5, 6}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 4}))
                .set_input(0, "input_forward")
                .set_input(1, "output_delta")
                .set_input(2, "mean")
                .set_input(3, "variance")
                .set_input(4, "gamma")
                .set_input(5, "beta")
                .set_output(0, "input_delta")
                .set_output(1, "gamma_delta")
                .set_output(2, "beta_delta")
                .set_output(3, "scratchpad")
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_norm_bprop_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_layernorm_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<layernorm_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(layernorm_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_matmul, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "src0")
                .set_input(1, "src1")
                .set_input(2, "bias")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from MatMul.
                .SET_MATMUL_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(op_attr::with_bias, false, attribute_kind::b, false)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::keep_dst_layout, false, attribute_kind::b,
                        false)
                // Analysis rules
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_matmul)
                .SET_EXECUTABLE_CREATOR(executable_creator<matmul_executable_t>)
                .SET_ARG_INDICES_GETTER(matmul_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_softmax, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from SoftMax
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_softmax)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<softmax_executable_t>)
                .SET_ARG_INDICES_GETTER(softmax_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_logsoftmax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from LogSoftmax
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_softmax)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<softmax_executable_t>)
                .SET_ARG_INDICES_GETTER(softmax_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_layernorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 4}))
                .set_input(0, "input")
                .set_input(1, "gamma")
                .set_input(2, "beta")
                .set_output(0, "output")
                .set_output(1, "mean")
                .set_output(2, "variance")
                .set_output(3, "scratchpad")
                // Attributes inherited from LayerNorm
                .set_attr(op_attr::keep_stats, false, attribute_kind::b, true)
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_norm_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_layernorm)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<layernorm_executable_t>)
                .SET_ARG_INDICES_GETTER(layernorm_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_reorder, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // TODO(xxx) Multiple ops will be mapped to dnnl_reorder
                // finally, how to deal with the attrs?
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                // Attributes
                .set_attr(op_attr::fusion_info_key, false, attribute_kind::i,
                        (int64_t)-1)
                .set_attr(
                        op_attr::change_layout, false, attribute_kind::b, false)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::src_zps, false, attribute_kind::is)
                .set_attr(op_attr::dst_zps, false, attribute_kind::is)
                .set_attr(op_attr::with_runtime_scales, false,
                        attribute_kind::b, false)
                .set_attr(op_attr::with_runtime_src_zps, false,
                        attribute_kind::b, false)
                .set_attr(op_attr::with_runtime_dst_zps, false,
                        attribute_kind::b, false)
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(-1))
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_reorder)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<reorder_executable_t>)
                .SET_ARG_INDICES_GETTER(reorder_executable_t))

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
