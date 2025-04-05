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
#ifndef GRAPH_BACKEND_DNNL_PASSES_TRANSFORM_HPP
#define GRAPH_BACKEND_DNNL_PASSES_TRANSFORM_HPP

#include <memory>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/dnnl/subgraph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t check_with_bias(std::shared_ptr<subgraph_t> &sg);

status_t fuse_bias_add(std::shared_ptr<subgraph_t> &sg);

status_t fold_mul_scales(std::shared_ptr<subgraph_t> &sg);

status_t fuse_to_int8_pool(std::shared_ptr<subgraph_t> &sg);

status_t defer_src_zps_for_pool(std::shared_ptr<subgraph_t> &sg);

status_t fuse_to_int8_concat(std::shared_ptr<subgraph_t> &sg);

status_t fuse_to_shuffle(std::shared_ptr<subgraph_t> &sg);

status_t replace_quant_data_with_binary_post_op(
        std::shared_ptr<subgraph_t> &sg);

status_t fuse_post_ops(std::shared_ptr<subgraph_t> &sg);

status_t fuse_src_zero_points(std::shared_ptr<subgraph_t> &sg);

status_t fuse_dst_zero_points(std::shared_ptr<subgraph_t> &sg);

status_t fuse_reciprocal_mul_to_div(std::shared_ptr<subgraph_t> &sg);

status_t insert_bn_folding(std::shared_ptr<subgraph_t> &sg);

status_t conv_bwd_data_canonicalization(std::shared_ptr<subgraph_t> &sg);

status_t conv_bwd_weights_canonicalization(std::shared_ptr<subgraph_t> &sg);

status_t pool_fwd_canonicalization(std::shared_ptr<subgraph_t> &sg);

status_t pool_bwd_canonicalization(std::shared_ptr<subgraph_t> &sg);

status_t fuse_mul_sigmoid_to_swish(std::shared_ptr<subgraph_t> &sg);

/// translate mixed int8/bf16 matmul/convolution subgraph to x8x8bf16 subgraph
///
///     | (u8/s8)  | (u8/s8)               | (u8/s8)  | (u8/s8)
///  dequant    dequant                 dequant    dequant
///     | (f32)    | (f32)                 | (f32)    | (f32)
///  typecast  typecast         -->         \        /
/// (bf16) \     / (bf16)                     matmul/conv
///      matmul/conv                             | (bf16)
///          | (bf16)
///
status_t fuse_typecast_to_matmul_or_conv(std::shared_ptr<subgraph_t> &sg);

/// translate mixed int8/bf16 matmul+add subgraph to x8x8bf16 subgraph
///
///     | (u8/s8)  | (u8/s8)               | (u8/s8)          | (u8/s8)
///  dequant    dequant    | (u8/s8)            dequant    dequant    | (u8/s8)
/// (f32) \     / (f32) dequant                (f32) \     / (f32) dequant
///        matmul      / (fp32)                       matmul      / (fp32)
///           \     typecast                            \ (fp32) /
///     (bf16) \   / (bf16)                                 add
///             add                                          | (bf16)
///              | (bf16)
status_t fuse_typecast_to_add(std::shared_ptr<subgraph_t> &sg);

/// fuse post typecast (f32<->bf16/f16) to matmul/conv/eltwise/binary/softmax/layernorm
///
///          |                 -->              |
///    matmul/conv/eltwise/              matmul/conv/eltwise/
///  binary/softmax/layernorm          binary/softmax/layernorm
///          |                                  |
///      (post_ops)                         (post_ops)
///          |                                  |
///       typecast
///          |
status_t fuse_post_typecast_to_predecessor(std::shared_ptr<subgraph_t> &sg);

status_t batchnorm_bwd_canonicalization(std::shared_ptr<subgraph_t> &sg);

/// translate the subgraph containing chain of Adds into dnnl_sum
///   in0   in1
///     \    /
///      Add   in2         in0  in1  in2
///        \   /             \   |   / ...
///         Add  in3    -->     sum
///           \   /
///            Add
///            ...
status_t fuse_to_dnnl_sum(std::shared_ptr<subgraph_t> &sg);

// This pass is used to insert unsqueeze op before dnnl_binary op's inputs to
// make the input shape meet the requirement of oneDNN binary primitive
status_t binary_canonicalization(std::shared_ptr<subgraph_t> &sg);

// This pass is used to swap two inputs to broadcast src1 which is optimized in
// oneDNN binary primitive. Notice that this should be applied after
// binary_canonicalization and infer_shape
status_t binary_broadcast_swap(std::shared_ptr<subgraph_t> &sg);

// This pass is used to fuse those adjacent reorders.
status_t fuse_adjacent_reorders(std::shared_ptr<subgraph_t> &sg);

status_t fuse_typecast_to_mul_scales(std::shared_ptr<subgraph_t> &sg);

// This pass handle dynamic quantization:mul_scale+add_zp
status_t fuse_dynamic_mul_scales_add_zps(std::shared_ptr<subgraph_t> &sg);

// This pass handle dynamic dequantization:sub_zp+mul_scale
status_t fuse_dynamic_sub_zps_mul_scales(std::shared_ptr<subgraph_t> &sg);

// This pass is used to convert single mul_scale,add_zp,sub_zp to reorder
// After "remove_quant_data_with_no_effect", maybe there is only single op.
impl::status_t convert_dynamic_quantize_ops(std::shared_ptr<subgraph_t> &sg);

status_t reorder_canonicalization(std::shared_ptr<subgraph_t> &sg);

status_t softmax_bwd_canonicalization(std::shared_ptr<subgraph_t> &sg);

/// A simple common reorder elimination pass which can perform the following
/// optimization if two reorder ops are equal:
///              val             val
///             /   \             |
///        reorder reorder  --> reorder
///             |    |          /   \  ...
///            op3  op4        op3  op4
status_t common_reorder_elimination(std::shared_ptr<subgraph_t> &sg);

// This pass currently can be used for int8 Pooling and int8 Eltwise only (as
// they are not supporting quantization-related attributes). Scales will get
// combined only if there is a single binary post-op.
status_t combine_binary_post_op_scales(std::shared_ptr<subgraph_t> &sg);

// This pass will remove OPs like mul_scales and add_zps in the following
// scenarios:
// - scales = [1] or [1, ..., 1]
// - zero points = [0] or [0, ..., 0]
status_t remove_quant_data_with_no_effect(std::shared_ptr<subgraph_t> &sg);

// This pass will move per_tensor quantize before Reshape and Transpose. So that
// it can have the opportunity to be fused into computation operators
impl::status_t lift_up_quantize(std::shared_ptr<subgraph_t> &sg);

// This pass will move typecast before Reshape and Transpose. So that it can
// have the opportunity to be fused into computation operators
impl::status_t lift_up_typecast(std::shared_ptr<subgraph_t> &sg);

/// This pass will move add after matmul and insert reshape and transpose before
/// src1 of add. So that it can have the opportunity to be fused into matmul.
///                                                          src1(4D)
///                                                          /
///          |                                  |        transpose
///          |                                  |          / (4D)
///        matmul(3D)                         matmul    reshape
///          |                                  |      / (3D)
///        reshape             ----->         add(3D)
///          | (4D)                             |
///       transpose   src1(4D)                reshape
///          |      /                           |(4D)
///         add(4D)                          transpose
///          |                                  |
///
impl::status_t lift_up_post_add_for_matmul(std::shared_ptr<subgraph_t> &sg);

// This pass will move reshape before Quantize and Dequantize for depthwiseconv.
// So that it can have the opportunity to be fused into computation operators
impl::status_t lift_up_weight_reshape_for_depthwiseconv(
        std::shared_ptr<subgraph_t> &sg);

// This pass will compute matmul with the src layout of transpose before matmul
impl::status_t fuse_src_transpose_to_matmul(std::shared_ptr<subgraph_t> &sg);

// This pass will compute matmul with the dst layout of following transpose if
// the operator after transpose need a dense layout
impl::status_t fuse_dst_transpose_to_matmul(std::shared_ptr<subgraph_t> &sg);

// This pass will fold add_zps into the previous sub_zps with new_zps = sub_zps
// - add_zps
impl::status_t fold_sub_zps_add_zps(std::shared_ptr<subgraph_t> &sg);

status_t convert_to_runtime_src_zero_points(std::shared_ptr<subgraph_t> &sg);

status_t convert_to_runtime_dst_zero_points(std::shared_ptr<subgraph_t> &sg);

status_t convert_runtime_mul_scales(std::shared_ptr<subgraph_t> &sg);

status_t convert_runtime_zero_points(std::shared_ptr<subgraph_t> &sg);

status_t convert_to_runtime_src_scales(std::shared_ptr<subgraph_t> &sg);

status_t fuse_src_scales(std::shared_ptr<subgraph_t> &sg);

status_t convert_to_runtime_dst_scales(std::shared_ptr<subgraph_t> &sg);

status_t fuse_dst_scales(std::shared_ptr<subgraph_t> &sg);

status_t convert_bias_to_f32(std::shared_ptr<subgraph_t> &sg);

status_t expand_convtranspose_scales(std::shared_ptr<subgraph_t> &sg);

// swap relu and mul_scales so that mul_scales can be folded into previous
// layers:
///        bn                                  bn
///         |                                   |
///       relu                              mul_scales
///         |                                   |
///     mul_scales                             relu
///         |                                   |
impl::status_t swap_relu_mul_scales(std::shared_ptr<subgraph_t> &sg);

/// This pass will move the effect of dequant to gamma and beta
/// Formula:
///  original: dst = (gamma * (src - mean) / sqrt(variance + epsilon)) + beta
///  apply_pre_mul_scale:
///      dst = (gamma * (src * scale - mean) / sqrt(variance + epsilon)) + beta
///  ==> dst = (gamma * scale * (src - mean / scale) /
///  sqrt(variance + epsilon)) + beta
///  ==> dst = (new_gamma * (src - new_mean) / sqrt(variance + epsilon))
///  + beta
impl::status_t fold_pre_mul_scale_into_bn(std::shared_ptr<subgraph_t> &sg);

/// This pass will move the effect of quant to gamma and beta
/// Formula:
///  original: dst = (gamma * (src - mean) / sqrt(variance + epsilon)) + beta
///  apply_post_mul_scale:
///      dst = ((gamma * (src - mean) / sqrt(variance + epsilon)) + beta) *
///  scale
///  ==> dst = (gamma * scale) * (src - mean) / sqrt(variance + epsilon) +
///  (beta * scale)
///  ==> dst = (new_gamma * (src - mean) / sqrt(variance + epsilon)) + new_beta
impl::status_t fold_post_mul_scale_into_bn(std::shared_ptr<subgraph_t> &sg);

/// This pass replaces the output logical tensor to remove the consumer. It is
/// mainly to use the "get_output_ops" function.
impl::status_t replace_select_values(std::shared_ptr<subgraph_t> &sg);
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
