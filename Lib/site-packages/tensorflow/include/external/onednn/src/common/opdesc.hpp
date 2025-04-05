/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef COMMON_OPDESC_HPP
#define COMMON_OPDESC_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/gemm_types.hpp"

namespace dnnl {
namespace impl {

struct reorder_desc_t {
    primitive_kind_t primitive_kind;
    const memory_desc_t *src_md;
    const memory_desc_t *dst_md;
    engine_kind_t src_engine_kind;
    engine_kind_t dst_engine_kind;
    bool is_cross_engine;
};

struct concat_desc_t {
    concat_desc_t() = default;
    concat_desc_t(primitive_kind_t primitive_kind, const memory_desc_t *dst_md,
            dim_t n, dim_t concat_dimension,
            const memory_desc_t *const *src_mds)
        : primitive_kind(primitive_kind)
        , dst_md(dst_md)
        , n(n)
        , concat_dimension(concat_dimension) {
        for (dim_t i = 0; i < n; i++)
            this->src_mds.push_back(src_mds[i]);
    }

    primitive_kind_t primitive_kind;
    const memory_desc_t *dst_md;
    dim_t n;
    dim_t concat_dimension;
    std::vector<const memory_desc_t *> src_mds;
};

struct sum_desc_t {
    sum_desc_t() = default;
    sum_desc_t(primitive_kind_t primitive_kind, const memory_desc_t *dst_md,
            dim_t n, const float *scales, const memory_desc_t *const *src_mds)
        : primitive_kind(primitive_kind), dst_md(dst_md), n(n), scales(scales) {
        for (dim_t i = 0; i < n; i++)
            this->src_mds.push_back(src_mds[i]);
    }

    primitive_kind_t primitive_kind;
    const memory_desc_t *dst_md;
    dim_t n;
    const float *scales;
    std::vector<const memory_desc_t *> src_mds;
};

struct zero_pad_desc_t {
    primitive_kind_t primitive_kind;
};

struct inner_product_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_inner_product.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: forward_training,
    // forward_inference, backward_data,
    // backward_weights, and backward_bias.
    prop_kind_t prop_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Weights memory descriptor.
    memory_desc_t weights_desc;
    // Weights gradient memory descriptor.
    memory_desc_t diff_weights_desc;
    // Bias memory descriptor.
    memory_desc_t bias_desc;
    // Bias gradient memory descriptor.
    memory_desc_t diff_bias_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // The accumulator data type.
    data_type_t accum_data_type;
};

struct convolution_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_convolution.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward_data,
    // #dnnl_backward_weights, and #dnnl_backward_bias.
    prop_kind_t prop_kind;
    // The kind of the convolution algorithm. Possible values:
    // #dnnl_convolution_direct.
    alg_kind_t alg_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Weights memory descriptor.
    memory_desc_t weights_desc;
    // Weights gradient memory descriptor.
    memory_desc_t diff_weights_desc;
    // Bias memory descriptor.
    memory_desc_t bias_desc;
    // Bias gradient memory descriptor.
    memory_desc_t diff_bias_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // Convolution strides in each spatial dimension.
    dims_t strides;
    // Convolution dilates in each spatial dimension.
    dims_t dilates;
    // Padding in each spatial dimension. padding[0] is a padding in the
    // beginning (@p padding_l), padding[1] is a padding in the end (@p
    // padding_r).
    dims_t padding[2];
    // The accumulator data type. Initialized automatically.
    data_type_t accum_data_type;
};

// A descriptor of a deconvolution operation.
using deconvolution_desc_t = convolution_desc_t;

// A descriptor of a shuffle operation.
struct shuffle_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_shuffle.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // Source or source gradient memory descriptor.
    memory_desc_t src_desc;
    // Destination or destination gradient memory descriptor.
    memory_desc_t dst_desc;
    // Axis for shuffling.
    int axis;
    // Number of groups.
    dim_t group_size;
};

// A descriptor of resampling operation.
struct resampling_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_resampling.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward_data,
    prop_kind_t prop_kind;
    // The kind of the resampling algorithm. Possible values:
    // #dnnl_resampling_nearest, #dnnl_resampling_linear.
    alg_kind_t alg_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // Resampling factor in each spatial dimension.
    float factors[DNNL_MAX_NDIMS];
};

// A descriptor of a matrix multiplication operation.
//
// 2D case:
//     dst[m, n] = src[m, k] * weights[k, n] + bias[m, n]
//
// 3D case:
//     dst[mb, m, n] = src[mb, m, k] * weights[mb, k, n] + bias[mb, m, n]
struct matmul_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_matmul.
    primitive_kind_t primitive_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Weights memory descriptor.
    memory_desc_t weights_desc;
    // Bias memory descriptor.
    memory_desc_t bias_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // The accumulator data type. Initialized automatically.
    data_type_t accum_data_type;
};

// A descriptor of a element-wise operation.
struct eltwise_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_eltwise.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // The kind of eltwise algorithm. Possible values: #dnnl_eltwise_relu,
    // #dnnl_eltwise_tanh, #dnnl_eltwise_elu, #dnnl_eltwise_square,
    // #dnnl_eltwise_abs, #dnnl_eltwise_sqrt, #dnnl_eltwise_linear,
    // #dnnl_eltwise_soft_relu, #dnnl_eltwise_logistic, #dnnl_eltwise_exp,
    // #dnnl_eltwise_gelu_tanh, #dnnl_eltwise_swish, #dnnl_eltwise_log,
    // #dnnl_eltwise_clip, #dnnl_eltwise_clip_v2, #dnnl_eltwise_pow,
    // #dnnl_eltwise_gelu_erf, #dnnl_eltwise_round,
    // #dnnl_eltwise_mish, #dnnl_eltwise_hardswish, #dnnl_eltwise_hardsigmoid.
    // Possible values for passing destination memory on backward:
    // #dnnl_eltwise_relu_use_dst_for_bwd, #dnnl_eltwise_tanh_use_dst_for_bwd,
    // #dnnl_eltwise_elu_use_dst_for_bwd, #dnnl_eltwise_sqrt_use_dst_for_bwd,
    // #dnnl_eltwise_logistic_use_dst_for_bwd,
    // #dnnl_eltwise_exp_use_dst_for_bwd,
    // #dnnl_eltwise_clip_v2_use_dst_for_bwd.
    alg_kind_t alg_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // Algorithm specific parameter.
    // Accordance table:
    //  - #dnnl_eltwise_relu: @p alpha -- negative slope, @p beta ignored
    //  - #dnnl_eltwise_tanh: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_elu: @p alpha -- negative slope, @p beta ignored
    //  - #dnnl_eltwise_square: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_abs: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_sqrt: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_linear: @p alpha -- scale, @p beta -- shift
    //  - #dnnl_eltwise_soft_relu: @p alpha -- soft_relu arg scaling, @p beta ignored
    //  - #dnnl_eltwise_logistic: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_exp: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_gelu_tanh: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_swish: @p alpha -- sigmoid arg scaling, @p beta ignored
    //  - #dnnl_eltwise_log: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_clip: @p alpha -- lower bound, @p beta -- upper bound
    //  - #dnnl_eltwise_clip_v2: @p alpha -- lower bound, @p beta -- upper bound
    //  - #dnnl_eltwise_pow: @p alpha -- scale, @p beta -- exponent
    //  - #dnnl_eltwise_gelu_erf: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_round: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_mish: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_hardswish: @p alpha and @p beta ignored
    //  - #dnnl_eltwise_hardsigmoid: @p alpha -- scale, @p beta -- shift
    float alpha, beta;
};

// A descriptor of a Batch Normalization operation.
struct batch_normalization_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_batch_normalization.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // Scale and/or shift data and gradient memory descriptor.
    // Scaleshift memory descriptor uses 1D #dnnl_x format[Channels].
    memory_desc_t scaleshift_desc;
    memory_desc_t diff_scaleshift_desc;
    // Statistics memory descriptor.
    //
    // Statistics (mean or variance) descriptor use 1D #dnnl_x format[Channels].
    memory_desc_t stat_desc;
    // Batch normalization epsilon parameter.
    float batch_norm_epsilon;
    unsigned flags;
};

// A descriptor of a Group Normalization operation.
struct group_normalization_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_group_normalization.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Scale and/or shift data and gradient memory descriptor.
    // Scaleshift memory descriptor uses 1D #dnnl_x format[Channels].
    memory_desc_t scaleshift_desc;
    memory_desc_t diff_scaleshift_desc;
    // Mean and variance data memory descriptors.
    // Statistics (mean and variance) memory descriptor uses 2D #dnnl_ab
    // format[Batch, groups].
    memory_desc_t stat_desc;
    // Group normalization groups parameter.
    dim_t groups;
    // Group normalization epsilon parameter.
    float group_norm_epsilon;
    unsigned flags;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
};

// A descriptor of a Layer Normalization operation.
struct layer_normalization_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_layer_normalization.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Scale and shift data and gradient memory descriptors.
    // Scaleshift memory descriptor uses 1D #dnnl_x format[normalized_dim].
    // Normalized_dim is equal to the last logical dimension of the source
    // tensor across which normalization is performed.
    memory_desc_t data_scaleshift_desc;
    memory_desc_t diff_data_scaleshift_desc;
    // Mean and variance data memory descriptors.
    //
    // Statistics (mean and variance) memory descriptor is the k-dimensional tensor
    // where k is equal to data_tensor_ndims - 1 and may have any plain
    // (stride[last_dim] == 1) user-provided format.
    memory_desc_t stat_desc;
    // Layer normalization epsilon parameter.
    float layer_norm_epsilon;
    unsigned flags;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
};

// A descriptor of a Local Response Normalization (LRN) operation.
struct lrn_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_lrn.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // LRN algorithm. Possible values: #dnnl_lrn_within_channel and
    // #dnnl_lrn_across_channels.
    alg_kind_t alg_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // The number of channels to sum over (for cross-channel LRN) or the side
    // length of the square region to sum over (for within-channel LRN).
    dim_t local_size;
    // LRN alpha parameter.
    float lrn_alpha;
    // LRN beta parameter.
    float lrn_beta;
    // LRN k parameter.
    float lrn_k;
};

// A descriptor of reduction operation.
struct reduction_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_reduction.
    primitive_kind_t primitive_kind;
    // The kind of reduction algorithm. Possible values:
    // #dnnl_reduction_max, #dnnl_reduction_min, #dnnl_reduction_sum,
    // #dnnl_reduction_mul, #dnnl_reduction_mean, #dnnl_reduction_norm_lp_max,
    // #dnnl_reduction_norm_lp_sum, #dnnl_reduction_norm_lp_power_p_max,
    // #dnnl_reduction_norm_lp_power_p_sum.
    alg_kind_t alg_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Algorithm specific parameters.
    // Accordance table:
    // #dnnl_reduction_max: @p p and @p eps are ignored
    // #dnnl_reduction_min: @p p and @p eps are ignored
    // #dnnl_reduction_norm_lp_max: @p p -- power, @p eps -- epsilon
    // #dnnl_reduction_norm_lp_sum: @p p -- power, @p eps -- epsilon
    // #dnnl_reduction_norm_lp_power_p_max: @p p -- power, @p eps -- epsilon
    // #dnnl_reduction_norm_lp_power_p_sum: @p p -- power, @p eps -- epsilon
    // #dnnl_reduction_sum: @p p and @p eps are ignored
    // #dnnl_reduction_mul: @p p and @p eps are ignored
    // #dnnl_reduction_mean: @p p and @p eps are ignored
    float p, eps;
};

/// A descriptor of a Softmax operation.
struct softmax_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_softmax.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // The axis along which to perform the softmax.
    int softmax_axis;
    // Softmax algorithm. Possible values: #dnnl_softmax_accurate and
    // #dnnl_softmax_log.
    alg_kind_t alg_kind;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
};

// A descriptor of a binary operation.
struct binary_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_binary.
    primitive_kind_t primitive_kind;
    // The kind of the binary algorithm. Possible values:
    // #dnnl_binary_add, #dnnl_binary_mul, #dnnl_binary_max, #dnnl_binary_min,
    // #dnnl_binary_div and #dnnl_binary_sub.
    alg_kind_t alg_kind;
    // Source memory descriptors.
    memory_desc_t src_desc[2];
    // Destination memory descriptor.
    memory_desc_t dst_desc;
};

/// A descriptor of a PReLU operation.
struct prelu_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_prelu.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward
    prop_kind_t prop_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Learnable parameter alpha memory descriptor.
    // Alpha describes negative slope.
    memory_desc_t weights_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Learnable parameter alpha gradient memory descriptor.
    memory_desc_t diff_weights_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
};

// A descriptor of a pooling operation.
struct pooling_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_pooling.
    primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    prop_kind_t prop_kind;
    // The kind of pooling algorithm.
    // Possible values: #dnnl_pooling_max,
    // #dnnl_pooling_avg_include_padding, and
    // #dnnl_pooling_avg_exclude_padding.
    alg_kind_t alg_kind;
    // Source memory descriptor.
    memory_desc_t src_desc;
    // Source gradient memory descriptor.
    memory_desc_t diff_src_desc;
    // Destination memory descriptor.
    memory_desc_t dst_desc;
    // Destination gradient memory descriptor.
    memory_desc_t diff_dst_desc;
    // Pooling kernel strides for spatial dimensions.
    dims_t strides;
    // Pooling kernel spatial dimensions.
    dims_t kernel;
    // Padding in each spatial dimension. padding[0] is a padding in the
    // beginning (@p padding_l), padding[1] is a padding in the end (@p
    // padding_r).
    dims_t padding[2];
    // The accumulator data type. Initialized automatically.
    data_type_t accum_data_type;
    // Pooling dilations for spatial dimensions.
    dims_t dilation;
};

// A descriptor for an RNN operation.
struct rnn_desc_t {
    // The kind of primitive. Used for self-identifying the primitive
    // descriptor. Must be #dnnl_rnn.
    dnnl_primitive_kind_t primitive_kind;
    // The kind of propagation. Possible values: #dnnl_forward_training,
    // #dnnl_forward_inference, and #dnnl_backward.
    prop_kind_t prop_kind;
    // RNN cell kind. Must be one of #dnnl_vanilla_rnn,
    // #dnnl_vanilla_lstm, #dnnl_vanilla_gru, or #dnnl_lbr_gru.
    alg_kind_t cell_kind;
    // The direction of RNN primitive execution.
    rnn_direction_t direction;
    // Source layer memory descriptor.
    memory_desc_t src_layer_desc;
    // Source iteration memory descriptor for hidden state.
    memory_desc_t src_iter_desc;
    // Source iteration memory descriptor for cell state.
    memory_desc_t src_iter_c_desc;
    // Weights layer memory descriptor.
    memory_desc_t weights_layer_desc;
    // Weights iteration memory descriptor.
    memory_desc_t weights_iter_desc;
    // Bias memory descriptor.
    memory_desc_t bias_desc;
    // Destination layer memory descriptor.
    memory_desc_t dst_layer_desc;
    // Destination iter memory descriptor for hidden state.
    memory_desc_t dst_iter_desc;
    // Destination iter memory descriptor for cell state.
    memory_desc_t dst_iter_c_desc;
    // Weights peephole memory descriptor.
    // This memory descriptor is equal to zero memory descriptor in case of
    // non-peephole LSTMs and other non-LSTM RNNs.
    memory_desc_t weights_peephole_desc;
    // Weights projection memory descriptor.
    // This memory descriptor is equal to zero memory descriptor in case of
    // non-projection LSTMs and other non-LSTM RNNs.
    memory_desc_t weights_projection_desc;

    // Source gradient layer memory descriptor.
    memory_desc_t diff_src_layer_desc;
    // Source gradient iter memory descriptor for hidden state.
    memory_desc_t diff_src_iter_desc;
    // Source gradient iter memory descriptor for cell state.
    memory_desc_t diff_src_iter_c_desc;
    // Weights gradient layer memory descriptor.
    memory_desc_t diff_weights_layer_desc;
    // Weights gradient iter memory descriptor.
    memory_desc_t diff_weights_iter_desc;
    // Bias gradient memory descriptor.
    memory_desc_t diff_bias_desc;
    // Destination gradient layer memory descriptor.
    memory_desc_t diff_dst_layer_desc;
    // Destination gradient iteration memory descriptor for hidden state.
    memory_desc_t diff_dst_iter_desc;
    // Destination gradient iteration memory descriptor for cell state.
    memory_desc_t diff_dst_iter_c_desc;
    // Weights gradient peephole memory descriptor.
    // This memory descriptor is equal to zero memory descriptor in case of
    // non-peephole LSTMs and other non-LSTM RNNs.
    memory_desc_t diff_weights_peephole_desc;
    // Weights gradient projection memory descriptor.
    // This memory descriptor is equal to zero memory descriptor in case of
    // non-projection LSTMs and other non-LSTM RNNs.
    memory_desc_t diff_weights_projection_desc;

    // RNN cell flags
    unsigned int flags;
    // Activation function used for vanilla_rnn cell kind.
    // Must be either #dnnl_eltwise_relu or #dnnl_eltwise_tanh.
    alg_kind_t activation_kind;
    float alpha;
    float beta;
};

struct op_desc_t {
    union {
        primitive_kind_t kind;
        convolution_desc_t convolution;
        deconvolution_desc_t deconvolution;
        shuffle_desc_t shuffle;
        pooling_desc_t pooling;
        prelu_desc_t prelu;
        eltwise_desc_t eltwise;
        softmax_desc_t softmax;
        lrn_desc_t lrn;
        batch_normalization_desc_t batch_normalization;
        group_normalization_desc_t group_normalization;
        layer_normalization_desc_t layer_normalization;
        inner_product_desc_t inner_product;
        rnn_desc_t rnn;
        gemm_desc_t gemm;
        concat_desc_t concat;
        reorder_desc_t reorder;
        sum_desc_t sum;
        binary_desc_t binary;
        matmul_desc_t matmul;
        resampling_desc_t resampling;
        zero_pad_desc_t zero_pad;
        reduction_desc_t reduction;
    };

#define DECL_CTOR_AND_CONVERTERS(c_type) \
    op_desc_t(const c_type &) = delete; \
    static op_desc_t *convert_from_c(c_type *_) { \
        return reinterpret_cast<op_desc_t *>(_); \
    } \
    static const op_desc_t *convert_from_c(const c_type *_) { \
        return reinterpret_cast<const op_desc_t *>(_); \
    }

    DECL_CTOR_AND_CONVERTERS(convolution_desc_t);
    DECL_CTOR_AND_CONVERTERS(shuffle_desc_t);
    DECL_CTOR_AND_CONVERTERS(pooling_desc_t);
    DECL_CTOR_AND_CONVERTERS(prelu_desc_t);
    DECL_CTOR_AND_CONVERTERS(eltwise_desc_t);
    DECL_CTOR_AND_CONVERTERS(softmax_desc_t);
    DECL_CTOR_AND_CONVERTERS(lrn_desc_t);
    DECL_CTOR_AND_CONVERTERS(batch_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(group_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(layer_normalization_desc_t);
    DECL_CTOR_AND_CONVERTERS(inner_product_desc_t);
    DECL_CTOR_AND_CONVERTERS(rnn_desc_t);
    DECL_CTOR_AND_CONVERTERS(gemm_desc_t);
    DECL_CTOR_AND_CONVERTERS(concat_desc_t);
    DECL_CTOR_AND_CONVERTERS(reorder_desc_t);
    DECL_CTOR_AND_CONVERTERS(sum_desc_t);
    DECL_CTOR_AND_CONVERTERS(binary_desc_t);
    DECL_CTOR_AND_CONVERTERS(matmul_desc_t);
    DECL_CTOR_AND_CONVERTERS(resampling_desc_t);
    DECL_CTOR_AND_CONVERTERS(zero_pad_desc_t);
    DECL_CTOR_AND_CONVERTERS(reduction_desc_t);

    // concat_desc_t and sum_desc_t have data members which have non-trivial
    // special member functions hence the default destructor is implicitly
    // deleted by the compiler which causes a warning on Windows so we should
    // delete the destructor explicitly.
    ~op_desc_t() = delete;

#undef DECL_CTOR_AND_CONVERTERS
};

} // namespace impl
} // namespace dnnl

#endif
