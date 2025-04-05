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

#ifndef GRAPH_INTERFACE_C_TYPES_MAP_HPP
#define GRAPH_INTERFACE_C_TYPES_MAP_HPP

#include <string>
#include <vector>
#include <type_traits>

#include "oneapi/dnnl/dnnl_graph_sycl.h"
#include "oneapi/dnnl/dnnl_graph_types.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.h"
#endif

namespace dnnl {
namespace impl {
namespace graph {

using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;
using dims = std::vector<dim_t>;

using status_t = dnnl_status_t;
namespace status {
const status_t success = dnnl_success;
const status_t out_of_memory = dnnl_out_of_memory;
const status_t invalid_arguments = dnnl_invalid_arguments;
const status_t unimplemented = dnnl_unimplemented;
const status_t last_impl_reached = dnnl_last_impl_reached;
const status_t runtime_error = dnnl_runtime_error;
const status_t not_required = dnnl_not_required;
const status_t invalid_graph = dnnl_invalid_graph;
const status_t invalid_graph_op = dnnl_invalid_graph_op;
const status_t invalid_shape = dnnl_invalid_shape;
const status_t invalid_data_type = dnnl_invalid_data_type;
} // namespace status

using data_type_t = dnnl_data_type_t;
namespace data_type {
const data_type_t undef = dnnl_data_type_undef;
const data_type_t f16 = dnnl_f16;
const data_type_t bf16 = dnnl_bf16;
const data_type_t f32 = dnnl_f32;
const data_type_t s32 = dnnl_s32;
const data_type_t s8 = dnnl_s8;
const data_type_t u8 = dnnl_u8;
const data_type_t boolean = dnnl_boolean;
const data_type_t f8_e5m2 = dnnl_f8_e5m2;
const data_type_t f8_e4m3 = dnnl_f8_e4m3;
} // namespace data_type

using partition_policy_t = dnnl_graph_partition_policy_t;
namespace partition_policy {
const partition_policy_t fusion = dnnl_graph_partition_policy_fusion;
const partition_policy_t debug = dnnl_graph_partition_policy_debug;
} // namespace partition_policy

// partition kind is moved from API to internal.
enum class partition_kind_t {
    undef = 0,
    convolution_post_ops = 1,
    convtranspose_post_ops = 2,
    interpolate_post_ops = 3,
    matmul_post_ops = 4,
    reduction_post_ops = 5,
    unary_post_ops = 6,
    binary_post_ops = 7,
    pooling_post_ops = 8,
    batch_norm_post_ops = 9,
    misc_post_ops = 10,
    quantized_convolution_post_ops = 11,
    quantized_convtranspose_post_ops = 12,
    quantized_matmul_post_ops = 13,
    quantized_unary_post_ops = 14,
    quantized_pooling_post_ops = 15,
    misc_quantized_post_ops = 16,
    convolution_backward_post_ops = 17,
    mha = 18,
    mlp = 19,
    quantized_mha = 20,
    quantized_mlp = 21,
    residual_conv_blocks = 22,
    quantized_residual_conv_blocks = 23,
    concat_fusion_memory_optim = 24,
    sdp = 25,
    quantized_sdp = 26
};

using engine_kind_t = dnnl_engine_kind_t;
namespace engine_kind {
const engine_kind_t any_engine = dnnl_any_engine;
const engine_kind_t cpu = dnnl_cpu;
const engine_kind_t gpu = dnnl_gpu;
} // namespace engine_kind

using fpmath_mode_t = dnnl_fpmath_mode_t;
namespace fpmath_mode {
const fpmath_mode_t strict = dnnl_fpmath_mode_strict;
const fpmath_mode_t bf16 = dnnl_fpmath_mode_bf16;
const fpmath_mode_t f16 = dnnl_fpmath_mode_f16;
const fpmath_mode_t any = dnnl_fpmath_mode_any;
const fpmath_mode_t tf32 = dnnl_fpmath_mode_tf32;
}; // namespace fpmath_mode

using op_kind_t = typename std::underlying_type<dnnl_graph_op_kind_t>::type;
namespace op_kind {
const op_kind_t Abs = dnnl_graph_op_abs;
const op_kind_t AbsBackward = dnnl_graph_op_abs_backward;
const op_kind_t Add = dnnl_graph_op_add;
const op_kind_t AvgPool = dnnl_graph_op_avg_pool;
const op_kind_t AvgPoolBackward = dnnl_graph_op_avg_pool_backward;
const op_kind_t BatchNormForwardTraining
        = dnnl_graph_op_batch_norm_forward_training;
const op_kind_t BatchNormInference = dnnl_graph_op_batch_norm_inference;
const op_kind_t BatchNormTrainingBackward = dnnl_graph_op_batch_norm_backward;
const op_kind_t BiasAdd = dnnl_graph_op_bias_add;
const op_kind_t BiasAddBackward = dnnl_graph_op_bias_add_backward;
const op_kind_t Clamp = dnnl_graph_op_clamp;
const op_kind_t ClampBackward = dnnl_graph_op_clamp_backward;
const op_kind_t Concat = dnnl_graph_op_concat;
const op_kind_t Convolution = dnnl_graph_op_convolution;
const op_kind_t ConvolutionBackwardData
        = dnnl_graph_op_convolution_backward_data;
const op_kind_t ConvolutionBackwardWeights
        = dnnl_graph_op_convolution_backward_weights;
const op_kind_t ConvTranspose = dnnl_graph_op_conv_transpose;
const op_kind_t ConvTransposeBackwardData
        = dnnl_graph_op_conv_transpose_backward_data;
const op_kind_t ConvTransposeBackwardWeights
        = dnnl_graph_op_conv_transpose_backward_weights;
const op_kind_t Dequantize = dnnl_graph_op_dequantize;
const op_kind_t Divide = dnnl_graph_op_divide;
const op_kind_t DynamicDequantize = dnnl_graph_op_dynamic_dequantize;
const op_kind_t DynamicQuantize = dnnl_graph_op_dynamic_quantize;
const op_kind_t Elu = dnnl_graph_op_elu;
const op_kind_t EluBackward = dnnl_graph_op_elu_backward;
const op_kind_t End = dnnl_graph_op_end;
const op_kind_t Exp = dnnl_graph_op_exp;
const op_kind_t GELU = dnnl_graph_op_gelu;
const op_kind_t GELUBackward = dnnl_graph_op_gelu_backward;
const op_kind_t HardSigmoid = dnnl_graph_op_hard_sigmoid;
const op_kind_t HardSigmoidBackward = dnnl_graph_op_hard_sigmoid_backward;
const op_kind_t HardSwish = dnnl_graph_op_hard_swish;
const op_kind_t HardSwishBackward = dnnl_graph_op_hard_swish_backward;
const op_kind_t Interpolate = dnnl_graph_op_interpolate;
const op_kind_t InterpolateBackward = dnnl_graph_op_interpolate_backward;
const op_kind_t LayerNorm = dnnl_graph_op_layer_norm;
const op_kind_t LayerNormBackward = dnnl_graph_op_layer_norm_backward;
const op_kind_t LeakyReLU = dnnl_graph_op_leaky_relu;
const op_kind_t Log = dnnl_graph_op_log;
const op_kind_t LogSoftmax = dnnl_graph_op_log_softmax;
const op_kind_t LogSoftmaxBackward = dnnl_graph_op_log_softmax_backward;
const op_kind_t MatMul = dnnl_graph_op_matmul;
const op_kind_t Maximum = dnnl_graph_op_maximum;
const op_kind_t MaxPool = dnnl_graph_op_max_pool;
const op_kind_t MaxPoolBackward = dnnl_graph_op_max_pool_backward;
const op_kind_t Minimum = dnnl_graph_op_minimum;
const op_kind_t Mish = dnnl_graph_op_mish;
const op_kind_t MishBackward = dnnl_graph_op_mish_backward;
const op_kind_t Multiply = dnnl_graph_op_multiply;
const op_kind_t Pow = dnnl_graph_op_pow;
const op_kind_t PReLU = dnnl_graph_op_prelu;
const op_kind_t PReLUBackward = dnnl_graph_op_prelu_backward;
const op_kind_t Quantize = dnnl_graph_op_quantize;
const op_kind_t Reciprocal = dnnl_graph_op_reciprocal;
const op_kind_t ReduceL1 = dnnl_graph_op_reduce_l1;
const op_kind_t ReduceL2 = dnnl_graph_op_reduce_l2;
const op_kind_t ReduceMax = dnnl_graph_op_reduce_max;
const op_kind_t ReduceMean = dnnl_graph_op_reduce_mean;
const op_kind_t ReduceMin = dnnl_graph_op_reduce_min;
const op_kind_t ReduceProd = dnnl_graph_op_reduce_prod;
const op_kind_t ReduceSum = dnnl_graph_op_reduce_sum;
const op_kind_t ReLU = dnnl_graph_op_relu;
const op_kind_t ReLUBackward = dnnl_graph_op_relu_backward;
const op_kind_t Reorder = dnnl_graph_op_reorder;
const op_kind_t Round = dnnl_graph_op_round;
const op_kind_t Select = dnnl_graph_op_select;
const op_kind_t Sigmoid = dnnl_graph_op_sigmoid;
const op_kind_t SigmoidBackward = dnnl_graph_op_sigmoid_backward;
const op_kind_t SoftMax = dnnl_graph_op_softmax;
const op_kind_t SoftMaxBackward = dnnl_graph_op_softmax_backward;
const op_kind_t SoftPlus = dnnl_graph_op_softplus;
const op_kind_t SoftPlusBackward = dnnl_graph_op_softplus_backward;
const op_kind_t Sqrt = dnnl_graph_op_sqrt;
const op_kind_t SqrtBackward = dnnl_graph_op_sqrt_backward;
const op_kind_t Square = dnnl_graph_op_square;
const op_kind_t SquaredDifference = dnnl_graph_op_squared_difference;
const op_kind_t StaticReshape = dnnl_graph_op_static_reshape;
const op_kind_t StaticTranspose = dnnl_graph_op_static_transpose;
const op_kind_t Subtract = dnnl_graph_op_subtract;
const op_kind_t Tanh = dnnl_graph_op_tanh;
const op_kind_t TanhBackward = dnnl_graph_op_tanh_backward;
const op_kind_t TypeCast = dnnl_graph_op_type_cast;
const op_kind_t Wildcard = dnnl_graph_op_wildcard;
const op_kind_t LastSymbol = dnnl_graph_op_last_symbol;
} // namespace op_kind

using op_attr_t = typename std::underlying_type<dnnl_graph_op_attr_t>::type;
namespace op_attr {
const op_attr_t undef = dnnl_graph_op_attr_undef;

const op_attr_t alpha = dnnl_graph_op_attr_alpha;
const op_attr_t beta = dnnl_graph_op_attr_beta;
const op_attr_t epsilon = dnnl_graph_op_attr_epsilon;
const op_attr_t max = dnnl_graph_op_attr_max;
const op_attr_t min = dnnl_graph_op_attr_min;
const op_attr_t momentum = dnnl_graph_op_attr_momentum;

const op_attr_t scales = dnnl_graph_op_attr_scales;

const op_attr_t axis = dnnl_graph_op_attr_axis;
const op_attr_t begin_norm_axis = dnnl_graph_op_attr_begin_norm_axis;
const op_attr_t groups = dnnl_graph_op_attr_groups;

const op_attr_t axes = dnnl_graph_op_attr_axes;
const op_attr_t dilations = dnnl_graph_op_attr_dilations;
const op_attr_t weights_shape = dnnl_graph_op_attr_weights_shape;
const op_attr_t src_shape = dnnl_graph_op_attr_src_shape;
const op_attr_t kernel = dnnl_graph_op_attr_kernel;
const op_attr_t order = dnnl_graph_op_attr_order;
const op_attr_t output_padding = dnnl_graph_op_attr_output_padding;
const op_attr_t dst_shape = dnnl_graph_op_attr_dst_shape;
const op_attr_t pads_begin = dnnl_graph_op_attr_pads_begin;
const op_attr_t pads_end = dnnl_graph_op_attr_pads_end;
const op_attr_t shape = dnnl_graph_op_attr_shape;
const op_attr_t sizes = dnnl_graph_op_attr_sizes;
const op_attr_t strides = dnnl_graph_op_attr_strides;
const op_attr_t zps = dnnl_graph_op_attr_zps;

const op_attr_t exclude_pad = dnnl_graph_op_attr_exclude_pad;
const op_attr_t keep_dims = dnnl_graph_op_attr_keep_dims;
const op_attr_t keep_stats = dnnl_graph_op_attr_keep_stats;
const op_attr_t per_channel_broadcast
        = dnnl_graph_op_attr_per_channel_broadcast;
const op_attr_t special_zero = dnnl_graph_op_attr_special_zero;
const op_attr_t transpose_a = dnnl_graph_op_attr_transpose_a;
const op_attr_t transpose_b = dnnl_graph_op_attr_transpose_b;
const op_attr_t use_affine = dnnl_graph_op_attr_use_affine;
const op_attr_t use_dst = dnnl_graph_op_attr_use_dst;

const op_attr_t auto_broadcast = dnnl_graph_op_attr_auto_broadcast;
const op_attr_t auto_pad = dnnl_graph_op_attr_auto_pad;
const op_attr_t coordinate_transformation_mode
        = dnnl_graph_op_attr_coordinate_transformation_mode;
const op_attr_t data_format = dnnl_graph_op_attr_data_format;
const op_attr_t weights_format = dnnl_graph_op_attr_weights_format;
const op_attr_t mode = dnnl_graph_op_attr_mode;
const op_attr_t qtype = dnnl_graph_op_attr_qtype;
const op_attr_t rounding_type = dnnl_graph_op_attr_rounding_type;

// Used to indicate the end of all external attributes, note all the new
// attribute should be added above this one.
const op_attr_t end = dnnl_graph_op_attr_end;

// internal attributes
const op_attr_t matched = 0x100;
const op_attr_t backend = 0x101;
const op_attr_t partition_id = 0x102;
const op_attr_t op_depth = 0x103;
} // namespace op_attr

using logical_tensor_t = dnnl_graph_logical_tensor_t;

using layout_type_t = dnnl_graph_layout_type_t;
namespace layout_type {
const layout_type_t undef = dnnl_graph_layout_type_undef;
const layout_type_t any = dnnl_graph_layout_type_any;
const layout_type_t strided = dnnl_graph_layout_type_strided;
const layout_type_t opaque = dnnl_graph_layout_type_opaque;
} // namespace layout_type

using property_type_t = dnnl_graph_tensor_property_t;
namespace property_type {
const property_type_t undef = dnnl_graph_tensor_property_undef;
const property_type_t variable = dnnl_graph_tensor_property_variable;
const property_type_t constant = dnnl_graph_tensor_property_constant;
} // namespace property_type

using attribute_kind_t = size_t;
namespace attribute_kind {
const attribute_kind_t f = 0;
const attribute_kind_t fs = 1;
const attribute_kind_t i = 2;
const attribute_kind_t is = 3;
const attribute_kind_t s = 4;
const attribute_kind_t b = 5;
} // namespace attribute_kind

using allocator_t = dnnl_graph_allocator;
using host_allocate_f = dnnl_graph_host_allocate_f;
using host_deallocate_f = dnnl_graph_host_deallocate_f;
using sycl_allocate_f = dnnl_graph_sycl_allocate_f;
using sycl_deallocate_f = dnnl_graph_sycl_deallocate_f;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
using ocl_allocate_f = dnnl_graph_ocl_allocate_f;
using ocl_deallocate_f = dnnl_graph_ocl_deallocate_f;
#endif
using inplace_pair_t = dnnl_graph_inplace_pair_t;

using graph_t = dnnl_graph_graph;
using op_t = dnnl_graph_op;
using partition_t = dnnl_graph_partition;
using compiled_partition_t = dnnl_graph_compiled_partition;
using tensor_t = dnnl_graph_tensor;

// oneDNN common objects
using engine_t = dnnl_engine;
using stream_t = dnnl_stream;

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
