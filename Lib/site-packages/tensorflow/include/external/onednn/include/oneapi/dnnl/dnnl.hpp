/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

/// @file
/// C++ API

#ifndef ONEAPI_DNNL_DNNL_HPP
#define ONEAPI_DNNL_DNNL_HPP

#include "oneapi/dnnl/dnnl_config.h"

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_common.hpp"

/// @endcond

/// @addtogroup dnnl_api oneDNN API
/// @{

/// oneDNN namespace
namespace dnnl {

/// @addtogroup dnnl_api_utils Utilities
/// Utility types and definitions.
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <typename T>
void validate_container_size(const T &v, const char *error_message,
        int min_size = 1, int max_size = -1) {
    const int size = (int)v.size();
    if (size < min_size || (max_size >= 0 && size > max_size))
        DNNL_THROW_ERROR(dnnl_invalid_arguments, error_message);
}
/// @endcond

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_memory_desc_t> {
    static dnnl_status_t destructor(dnnl_memory_desc_t p) {
        return dnnl_memory_desc_destroy(p);
    }
};

template <>
struct handle_traits<dnnl_memory_t> {
    static dnnl_status_t destructor(dnnl_memory_t p) {
        return dnnl_memory_destroy(p);
    }
};

template <>
struct handle_traits<dnnl_primitive_desc_t> {
    static dnnl_status_t destructor(dnnl_primitive_desc_t p) {
        return dnnl_primitive_desc_destroy(p);
    }
};

template <>
struct handle_traits<dnnl_primitive_t> {
    static dnnl_status_t destructor(dnnl_primitive_t p) {
        return dnnl_primitive_destroy(p);
    }
};

/// @endcond

/// @} dnnl_api_utils

struct stream;
struct memory;
struct primitive_desc;

/// @addtogroup dnnl_api_primitives Primitives
/// Compute primitives
/// @sa @ref dev_guide_basic_concepts
/// @{

/// @addtogroup dnnl_api_primitives_common Common
/// Common operations to create, destroy and inspect primitives
/// @{

/// Base class for all computational primitives.
struct primitive : public handle<dnnl_primitive_t> {
    /// Kinds of primitives supported by the library.
    enum class kind {
        /// Undefined primitive
        undef = dnnl_undefined_primitive,
        /// A reorder primitive.
        reorder = dnnl_reorder,
        /// A shuffle primitive.
        shuffle = dnnl_shuffle,
        /// A (out-of-place) tensor concatenation primitive.
        concat = dnnl_concat,
        /// A summation primitive.
        sum = dnnl_sum,
        /// A convolution primitive.
        convolution = dnnl_convolution,
        /// A deconvolution primitive.
        deconvolution = dnnl_deconvolution,
        /// An element-wise primitive.
        eltwise = dnnl_eltwise,
        /// An LRN primitive.
        lrn = dnnl_lrn,
        /// A batch normalization primitive.
        batch_normalization = dnnl_batch_normalization,
        /// An inner product primitive.
        inner_product = dnnl_inner_product,
        /// An RNN primitive.
        rnn = dnnl_rnn,
        /// A binary primitive.
        binary = dnnl_binary,
        /// A matmul (matrix multiplication) primitive.
        matmul = dnnl_matmul,
        /// A resampling primitive.
        resampling = dnnl_resampling,
        /// A pooling primitive.
        pooling = dnnl_pooling,
        /// A reduction primitive.
        reduction = dnnl_reduction,
        /// A PReLU primitive.
        prelu = dnnl_prelu,
        /// A softmax primitive.
        softmax = dnnl_softmax,
        /// A layer normalization primitive.
        layer_normalization = dnnl_layer_normalization,
        /// A group normalization primitive
        group_normalization = dnnl_group_normalization,
    };

    using handle::handle;

    /// Default constructor. Constructs an empty object.
    primitive() = default;

    /// Constructs a primitive from a C API primitive descriptor.
    ///
    /// @param c_pd C API primitive descriptor.
    primitive(const_dnnl_primitive_desc_t c_pd);

    /// Constructs a primitive from a C API primitive descriptor and a cache blob.
    ///
    /// @param c_pd C API primitive descriptor.
    /// @param cache_blob Cache blob.
    primitive(const_dnnl_primitive_desc_t c_pd,
            const std::vector<uint8_t> &cache_blob);

    /// Constructs a primitive from a primitive descriptor.
    ///
    /// @param pd Primitive descriptor.
    primitive(const primitive_desc &pd);

    /// Constructs a primitive from a primitive descriptor and a cache blob.
    ///
    /// @param pd Primitive descriptor.
    /// @param cache_blob Cache blob.
    primitive(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob);

    /// Returns the C API primitive descriptor of the underlying C API
    /// primitive.
    ///
    /// @returns The underlying C API primitive descriptor.
    inline const_dnnl_primitive_desc_t get_primitive_desc() const;

    /// Returns the kind of the primitive.
    ///
    /// @returns The primitive kind.
    inline kind get_kind() const;

    /// Returns a cache blob for the primitive.
    ///
    /// @returns Vector containing the cache blob.
    ///
    /// @note The cache blob can be empty. It's the user's responsibility to
    ///     check whether it's empty prior to passing it to the primitive
    ///     constructor.
    inline std::vector<uint8_t> get_cache_blob() const;

    /// Executes computations specified by the primitive in a specified stream.
    ///
    /// Arguments are passed via an arguments map containing <index,
    /// memory object> pairs. The index must be one of the `DNNL_ARG_*` values
    /// such as `DNNL_ARG_SRC`, and the memory must have a memory descriptor
    /// matching the one returned by
    /// primitive_desc::query_md(#query::exec_arg_md, index) unless using
    /// dynamic shapes (see #DNNL_RUNTIME_DIM_VAL).
    ///
    /// @param astream Stream object. The stream must belong to the same engine
    ///     as the primitive.
    /// @param args Arguments map.
    void execute(const stream &astream,
            const std::unordered_map<int, memory> &args) const;
};

/// Converts primitive kind enum value from C++ API to C API type.
///
/// @param akind C++ API primitive kind enum value.
/// @returns Corresponding C API primitive kind enum value.
inline dnnl_primitive_kind_t convert_to_c(primitive::kind akind) {
    return static_cast<dnnl_primitive_kind_t>(akind);
}

const_dnnl_primitive_desc_t primitive::get_primitive_desc() const {
    const_dnnl_primitive_desc_t pd;
    error::wrap_c_api(dnnl_primitive_get_primitive_desc(get(), &pd),
            "could not get a primitive descriptor from a primitive");
    return pd;
}

dnnl::primitive::kind primitive::get_kind() const {
    const_dnnl_primitive_desc_t pd = get_primitive_desc();
    // TODO (Roma): the code below is only needed because get_primitive_desc
    // returns a C type.
    dnnl_primitive_kind_t kind;
    error::wrap_c_api(dnnl_primitive_desc_query(
                              pd, dnnl_query_primitive_kind, 0, (void *)&kind),
            "could not get a primitive kind from a primitive descriptor");
    return static_cast<dnnl::primitive::kind>(kind);
}

std::vector<uint8_t> primitive::get_cache_blob() const {
    size_t size;
    error::wrap_c_api(dnnl_primitive_get_cache_blob(get(), &size, nullptr),
            "could not get cache blob size from a primitive");

    std::vector<uint8_t> cache_blob(size);
    error::wrap_c_api(
            dnnl_primitive_get_cache_blob(get(), &size, cache_blob.data()),
            "could not get a cache blob from a primitive");
    return cache_blob;
}

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_attributes
///
/// A container for parameters that extend primitives behavior.
///
/// Attributes can also contain Post-ops, which are computations executed
/// after the primitive.
///
/// @sa @ref dev_guide_attributes
/// @sa @ref dev_guide_attributes_post_ops
///
/// @{

/// Scratchpad mode
enum class scratchpad_mode {
    /// The library manages the scratchpad allocation according to the policy
    /// specified by the `DNNL_ENABLE_CONCURRENT_EXEC`
    /// [build option](@ref dev_guide_build_options) (default).
    ///
    /// When `DNNL_ENABLE_CONCURRENT_EXEC=OFF` (default), the library
    /// scratchpad is common to all primitives to reduce the memory footprint.
    /// This configuration comes with limited thread-safety properties, namely
    /// primitives can be created and executed in parallel but cannot migrate
    /// between threads (in other words, each primitive should be executed in
    /// the same thread it was created in).
    ///
    /// When `DNNL_ENABLE_CONCURRENT_EXEC=ON`, the library scratchpad is
    /// private to each primitive. The memory footprint is larger than when
    /// using `DNNL_ENABLE_CONCURRENT_EXEC=OFF` but different primitives can be
    /// created and run concurrently (the same primitive cannot be run
    /// concurrently from two different threads though).
    library = dnnl_scratchpad_mode_library,
    /// The user manages the scratchpad allocation by querying and providing
    /// the scratchpad memory to primitives. This mode is thread-safe as long
    /// as the scratchpad buffers are not used concurrently by two primitive
    /// executions.
    user = dnnl_scratchpad_mode_user,
};

/// Converts a scratchpad mode enum value from C++ API to C API type.
///
/// @param mode C++ API scratchpad mode enum value.
/// @returns Corresponding C API scratchpad mode enum value.
inline dnnl_scratchpad_mode_t convert_to_c(scratchpad_mode mode) {
    return static_cast<dnnl_scratchpad_mode_t>(mode);
}

/// Propagation kind.
enum class prop_kind {
    /// Undefined propagation kind.
    undef = dnnl_prop_kind_undef,
    /// Forward data propagation (training mode). In this mode, primitives
    /// perform computations necessary for subsequent backward propagation.
    forward_training = dnnl_forward_training,
    /// Forward data propagation (inference mode). In this mode, primitives
    /// perform only computations that are necessary for inference and omit
    /// computations that are necessary only for backward propagation.
    forward_inference = dnnl_forward_inference,
    /// Forward data propagation,
    /// alias for #dnnl::prop_kind::forward_training.
    forward = dnnl_forward,
    /// Backward propagation (with respect to all parameters).
    backward = dnnl_backward,
    /// Backward data propagation.
    backward_data = dnnl_backward_data,
    /// Backward weights propagation.
    backward_weights = dnnl_backward_weights,
    /// Backward bias propagation.
    backward_bias = dnnl_backward_bias
};

/// Converts propagation kind enum value from C++ API to C API type.
///
/// @param akind C++ API propagation kind enum value.
/// @returns Corresponding C API propagation kind enum value.
inline dnnl_prop_kind_t convert_to_c(prop_kind akind) {
    return static_cast<dnnl_prop_kind_t>(akind);
}

/// Kinds of algorithms.
enum class algorithm {
    /// Undefined algorithm
    undef = dnnl_alg_kind_undef,
    /// Convolution algorithm that is chosen to be either direct or Winograd
    /// automatically
    convolution_auto = dnnl_convolution_auto,
    /// Direct convolution
    convolution_direct = dnnl_convolution_direct,
    /// Winograd convolution
    convolution_winograd = dnnl_convolution_winograd,
    /// Direct deconvolution
    deconvolution_direct = dnnl_deconvolution_direct,
    /// Winograd deconvolution
    deconvolution_winograd = dnnl_deconvolution_winograd,
    /// Elementwise: rectified linear unit (ReLU)
    eltwise_relu = dnnl_eltwise_relu,
    /// Elementwise: hyperbolic tangent non-linearity (tanh)
    eltwise_tanh = dnnl_eltwise_tanh,
    /// Elementwise: exponential linear unit (ELU)
    eltwise_elu = dnnl_eltwise_elu,
    /// Elementwise: square
    eltwise_square = dnnl_eltwise_square,
    /// Elementwise: abs
    eltwise_abs = dnnl_eltwise_abs,
    /// Elementwise: square root
    eltwise_sqrt = dnnl_eltwise_sqrt,
    /// Elementwise: swish (\f$x \cdot sigmoid(a \cdot x)\f$)
    eltwise_swish = dnnl_eltwise_swish,
    /// Elementwise: linear
    eltwise_linear = dnnl_eltwise_linear,
    /// Elementwise: soft_relu
    eltwise_soft_relu = dnnl_eltwise_soft_relu,
    /// Elementwise: mish
    eltwise_mish = dnnl_eltwise_mish,
    /// Elementwise: logistic
    eltwise_logistic = dnnl_eltwise_logistic,
    /// Elementwise: exponent
    eltwise_exp = dnnl_eltwise_exp,
    /// Elementwise: tanh-based gelu
    eltwise_gelu_tanh = dnnl_eltwise_gelu_tanh,
    /// Elementwise: erf-based gelu
    eltwise_gelu_erf = dnnl_eltwise_gelu_erf,
    /// Elementwise: natural logarithm
    eltwise_log = dnnl_eltwise_log,
    /// Elementwise: clip
    eltwise_clip = dnnl_eltwise_clip,
    /// Eltwise: clip version 2
    eltwise_clip_v2 = dnnl_eltwise_clip_v2,
    /// Elementwise: pow
    eltwise_pow = dnnl_eltwise_pow,
    /// Elementwise: round
    eltwise_round = dnnl_eltwise_round,
    /// Elementwise: hardswish
    eltwise_hardswish = dnnl_eltwise_hardswish,
    /// Elementwise: hardsigmoid
    eltwise_hardsigmoid = dnnl_eltwise_hardsigmoid,
    /// Elementwise: rectified linar unit (ReLU) (dst for backward)
    eltwise_relu_use_dst_for_bwd = dnnl_eltwise_relu_use_dst_for_bwd,
    /// Elementwise: hyperbolic tangent non-linearity (tanh) (dst for backward)
    eltwise_tanh_use_dst_for_bwd = dnnl_eltwise_tanh_use_dst_for_bwd,
    /// Elementwise: exponential linear unit (ELU) (dst for backward)
    eltwise_elu_use_dst_for_bwd = dnnl_eltwise_elu_use_dst_for_bwd,
    /// Elementwise: square root (dst for backward)
    eltwise_sqrt_use_dst_for_bwd = dnnl_eltwise_sqrt_use_dst_for_bwd,
    /// Elementwise: logistic (dst for backward)
    eltwise_logistic_use_dst_for_bwd = dnnl_eltwise_logistic_use_dst_for_bwd,
    /// Elementwise: exponent (dst for backward)
    eltwise_exp_use_dst_for_bwd = dnnl_eltwise_exp_use_dst_for_bwd,
    /// Elementwise: clip version 2 (dst for backward)
    eltwise_clip_v2_use_dst_for_bwd = dnnl_eltwise_clip_v2_use_dst_for_bwd,
    /// Local response normalization (LRN) across multiple channels
    lrn_across_channels = dnnl_lrn_across_channels,
    /// LRN within a single channel
    lrn_within_channel = dnnl_lrn_within_channel,
    /// Max pooling
    pooling_max = dnnl_pooling_max,
    /// Average pooling include padding
    pooling_avg_include_padding = dnnl_pooling_avg_include_padding,
    /// Average pooling exclude padding
    pooling_avg_exclude_padding = dnnl_pooling_avg_exclude_padding,
    /// RNN cell
    vanilla_rnn = dnnl_vanilla_rnn,
    /// LSTM cell
    vanilla_lstm = dnnl_vanilla_lstm,
    /// GRU cell
    vanilla_gru = dnnl_vanilla_gru,
    /// GRU cell with linear before reset. Differs from the vanilla GRU
    /// in how the new memory gate is calculated:
    /// \f$c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h})) \f$
    /// LRB GRU expects 4 bias tensors on input:
    /// \f$[b_{u}, b_{r}, b_{c_x}, b_{c_h}]\f$
    lbr_gru = dnnl_lbr_gru,
    /// AUGRU cell
    vanilla_augru = dnnl_vanilla_augru,
    /// AUGRU cell with linear before reset
    lbr_augru = dnnl_lbr_augru,
    /// Binary add
    binary_add = dnnl_binary_add,
    /// Binary mul
    binary_mul = dnnl_binary_mul,
    /// Binary max
    binary_max = dnnl_binary_max,
    /// Binary min
    binary_min = dnnl_binary_min,
    /// Binary div
    binary_div = dnnl_binary_div,
    /// Binary sub
    binary_sub = dnnl_binary_sub,
    /// Binary greater than or equal
    binary_ge = dnnl_binary_ge,
    /// Binary greater than
    binary_gt = dnnl_binary_gt,
    /// Binary less than or equal
    binary_le = dnnl_binary_le,
    /// Binary less than
    binary_lt = dnnl_binary_lt,
    /// Binary equal
    binary_eq = dnnl_binary_eq,
    /// Binary not equal
    binary_ne = dnnl_binary_ne,
    /// Nearest Neighbor resampling method
    resampling_nearest = dnnl_resampling_nearest,
    /// Linear (Bilinear, Trilinear) resampling method
    resampling_linear = dnnl_resampling_linear,
    /// Reduction using max operation
    reduction_max = dnnl_reduction_max,
    /// Reduction using min operation
    reduction_min = dnnl_reduction_min,
    /// Reduction using sum operation
    reduction_sum = dnnl_reduction_sum,
    /// Reduction using mul operation
    reduction_mul = dnnl_reduction_mul,
    /// Reduction using mean operation
    reduction_mean = dnnl_reduction_mean,
    /// Reduction using norm_lp_max operation
    reduction_norm_lp_max = dnnl_reduction_norm_lp_max,
    /// Reduction using norm_lp_sum operation
    reduction_norm_lp_sum = dnnl_reduction_norm_lp_sum,
    /// Reduction using norm_lp_power_p_max operation
    reduction_norm_lp_power_p_max = dnnl_reduction_norm_lp_power_p_max,
    /// Reduction using norm_lp_power_p_sum operation
    reduction_norm_lp_power_p_sum = dnnl_reduction_norm_lp_power_p_sum,
    /// Softmax, numerically stable
    softmax_accurate = dnnl_softmax_accurate,
    /// LogSoftmax, numerically stable
    softmax_log = dnnl_softmax_log,
};

/// Converts algorithm kind enum value from C++ API to C API type.
/// @param aalgorithm C++ API algorithm kind enum value.
/// @returns Corresponding C API algorithm kind enum value.
inline dnnl_alg_kind_t convert_to_c(algorithm aalgorithm) {
    return static_cast<dnnl_alg_kind_t>(aalgorithm);
}

/// @} dnnl_api_attributes

/// @addtogroup dnnl_api_primitives_common
/// @{

/// Flags for normalization primitives.
enum class normalization_flags : unsigned {
    /// Use no normalization flags. If specified, the library computes mean and
    /// variance on forward propagation for training and inference, outputs them
    /// on forward propagation for training, and computes the respective
    /// derivatives on backward propagation.
    none = dnnl_normalization_flags_none,

    /// Use global statistics. If specified, the library uses mean and
    /// variance provided by the user as an input on forward propagation and
    /// does not compute their derivatives on backward propagation. Otherwise,
    /// the library computes mean and variance on forward propagation for
    /// training and inference, outputs them on forward propagation for
    /// training, and computes the respective derivatives on backward
    /// propagation.
    use_global_stats = dnnl_use_global_stats,

    /// Use scale parameter. If specified, the user is expected to pass scale as
    /// input on forward propagation. On backward propagation of type
    /// #dnnl::prop_kind::backward, the library computes its derivative.
    use_scale = dnnl_use_scale,

    /// Use shift parameter. If specified, the user is expected to pass shift as
    /// input on forward propagation. On backward propagation of type
    /// #dnnl::prop_kind::backward, the library computes its derivative.
    use_shift = dnnl_use_shift,

    /// Fuse normalization with ReLU. On training, normalization will require
    /// the workspace to implement backward propagation. On inference, the
    /// workspace is not required and behavior is the same as when normalization
    /// is fused with ReLU using the post-ops API.
    fuse_norm_relu = dnnl_fuse_norm_relu,

    /// Fuse normalization with elementwise binary Add and then fuse with ReLU.
    /// On training, normalization will require the workspace to implement
    /// backward propagation. On inference, the workspace is not required.
    fuse_norm_add_relu = dnnl_fuse_norm_add_relu,
};

/// Converts normalization flags enum value from C++ API to C API type.
/// @param flags C++ API normalization flags enum value.
/// @returns Corresponding C API normalization flags enum value.
inline dnnl_normalization_flags_t convert_to_c(normalization_flags flags) {
    return static_cast<dnnl_normalization_flags_t>(flags);
}

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_rnn
/// @{

/// RNN cell flags.
enum class rnn_flags : unsigned {
    /// Undefined RNN flags
    undef = dnnl_rnn_flags_undef,
    /// Do not add weights gradient to existing diff_weights memory
    diff_weights_overwrite = dnnl_rnn_flags_diff_weights_overwrite,
};

/// Converts RNN cell flags enum value from C++ API to C API type.
/// @param flags C++ API RNN cell flags enum value.
/// @returns Corresponding C API RNN cell flags enum value.
inline dnnl_rnn_flags_t convert_to_c(rnn_flags flags) {
    return static_cast<dnnl_rnn_flags_t>(flags);
}

DNNL_DEFINE_BITMASK_OPS(normalization_flags)
DNNL_DEFINE_BITMASK_OPS(rnn_flags)

/// A direction of RNN primitive execution
enum class rnn_direction {
    /// Undefined RNN direction.
    undef = dnnl_rnn_direction_undef,
    /// Unidirectional execution of RNN primitive from left to right.
    unidirectional_left2right = dnnl_unidirectional_left2right,
    /// Unidirectional execution of RNN primitive from right to left.
    unidirectional_right2left = dnnl_unidirectional_right2left,
    /// Bidirectional execution of RNN primitive with concatenation of the
    /// results.
    bidirectional_concat = dnnl_bidirectional_concat,
    /// Bidirectional execution of RNN primitive with summation of the
    /// results.
    bidirectional_sum = dnnl_bidirectional_sum,
};

/// Converts RNN direction enum value from C++ API to C API type.
/// @param dir C++ API RNN direction enum value.
/// @returns Corresponding C API RNN direction enum value.
inline dnnl_rnn_direction_t convert_to_c(rnn_direction dir) {
    return static_cast<dnnl_rnn_direction_t>(dir);
}

/// @} dnnl_api_rnn

/// @addtogroup dnnl_api_primitives_common
/// @{

/// Primitive descriptor query specification.
///
/// In general, queries are not used with the C++ API because most queries are
/// implemented as class members.
///
/// See @ref dnnl_query_t for more information.
enum class query {
    /// no query
    undef = dnnl_query_undef,

    /// execution engine
    engine = dnnl_query_engine,
    /// primitive kind
    primitive_kind = dnnl_query_primitive_kind,

    /// number of inputs expected
    num_of_inputs_s32 = dnnl_query_num_of_inputs_s32,
    /// number of outputs expected
    num_of_outputs_s32 = dnnl_query_num_of_outputs_s32,

    /// runtime estimation (seconds), unimplemented
    time_estimate_f64 = dnnl_query_time_estimate_f64,
    /// memory required for scratchpad (bytes)
    ///
    /// @sa @ref dev_guide_attributes_scratchpad
    memory_consumption_s64 = dnnl_query_memory_consumption_s64,

    /// scratchpad engine
    ///
    /// engine to be used for creating scratchpad memory
    scratchpad_engine = dnnl_query_scratchpad_engine,

    /// reorder source engine
    reorder_src_engine = dnnl_query_reorder_src_engine,
    /// reorder destination engine
    reorder_dst_engine = dnnl_query_reorder_dst_engine,

    /// implementation name
    impl_info_str = dnnl_query_impl_info_str,

    /// propagation kind
    prop_kind = dnnl_query_prop_kind,

    /// size of cache blob ID in bytes
    cache_blob_id_size_s64 = dnnl_query_cache_blob_id_size_s64,

    /// cache blob ID (pointer to array)
    cache_blob_id = dnnl_query_cache_blob_id,

    /// strides
    strides = dnnl_query_strides,
    /// dilations
    dilations = dnnl_query_dilations,
    /// left padding
    padding_l = dnnl_query_padding_l,
    /// right padding
    padding_r = dnnl_query_padding_r,
    /// epsilon
    epsilon_f32 = dnnl_query_epsilon_f32,
    /// flags
    flags = dnnl_query_flags,
    /// algorithm kind
    alg_kind = dnnl_query_alg_kind,
    /// alpha
    alpha_f32 = dnnl_query_alpha_f32,
    /// beta
    beta_f32 = dnnl_query_beta_f32,
    /// axis
    axis_s32 = dnnl_query_axis_s32,
    /// LRN parameter local size
    local_size_s64 = dnnl_query_local_size_s64,
    /// LRN parameter K
    k_f32 = dnnl_query_k_f32,
    /// Reduction parameter P
    p_f32 = dnnl_query_p_f32,
    /// Resampling parameter factors
    factors = dnnl_query_factors,
    /// RNN parameter cell kind
    cell_kind = dnnl_query_cell_kind,
    /// RNN parameter direction
    direction = dnnl_query_direction,
    /// RNN parameter activation kind
    activation_kind = dnnl_query_activation_kind,
    /// Pooling parameter kernel
    kernel = dnnl_query_kernel,
    /// Shuffle parameter group size
    group_size_s64 = dnnl_query_group_size_s64,

    /// source memory desc
    src_md = dnnl_query_src_md,
    /// source gradient (diff) memory desc
    diff_src_md = dnnl_query_diff_src_md,
    /// weights memory descriptor desc
    weights_md = dnnl_query_weights_md,
    /// weights gradient (diff) memory desc
    diff_weights_md = dnnl_query_diff_weights_md,
    /// destination memory desc
    dst_md = dnnl_query_dst_md,
    /// destination gradient (diff) memory desc
    diff_dst_md = dnnl_query_diff_dst_md,
    /// workspace memory desc
    workspace_md = dnnl_query_workspace_md,
    /// scratchpad memory desc
    scratchpad_md = dnnl_query_scratchpad_md,
    /// memory desc of an execute argument
    exec_arg_md = dnnl_query_exec_arg_md,

    /// number of dimensions
    ndims_s32 = dnnl_query_ndims_s32,
    /// vector of dimensions
    dims = dnnl_query_dims,
    /// data type
    data_type = dnnl_query_data_type,
    /// submemory offset
    submemory_offset_s64 = dnnl_query_submemory_offset_s64,
    /// vector of padded dimensions
    padded_dims = dnnl_query_padded_dims,
    /// vector of padded offsets
    padded_offsets = dnnl_query_padded_offsets,
    /// format kind
    format_kind = dnnl_query_format_kind,
    ///  number of innermost blocks
    inner_nblks_s32 = dnnl_query_inner_nblks_s32,
    /// vector of sizes of the innermost blocks
    inner_blks = dnnl_query_inner_blks,
    /// vector of logical indices of the blocks
    inner_idxs = dnnl_query_inner_idxs,
#ifdef DNNL_EXPERIMENTAL_SPARSE
    /// Sparse encoding
    sparse_encoding = dnnl_query_sparse_encoding,
    /// Number of non-zero entries
    nnz_s64 = dnnl_query_nnz_s64,
    /// Number of buffers required for a memory descriptor
    num_handles_s32 = dnnl_query_num_handles_s32,
#endif
};

/// Converts query enum value from C++ API to C API type.
/// @param aquery C++ API query enum value.
/// @returns Corresponding C API query enum value.
inline dnnl_query_t convert_to_c(query aquery) {
    return static_cast<dnnl_query_t>(aquery);
}

/// @} dnnl_api_primitives_common

/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_memory Memory
///
/// A container that describes and stores data. Memory objects can contain
/// data of various types and formats. There are two levels of abstraction:
///
/// 1. **Memory descriptor** -- engine-agnostic logical description of data
///     (number of dimensions, dimension sizes, and data type), and,
///     optionally, the information about the physical format of data in
///     memory. If this information is not known yet, a memory descriptor can
///     be created with #dnnl::memory::format_tag::any. This allows
///     compute-intensive primitives to choose the best format for
///     computation. The user is responsible for reordering the data into the
///     chosen format when formats do not match.
///
///     A memory descriptor can be initialized either by specifying dimensions
///     and a memory format tag or strides for each of them, or by
///     manipulating the dnnl_memory_desc_t structure directly.
///
///     @warning
///         The latter approach requires understanding how the physical data
///         representation is mapped to the structure and is discouraged. This
///         topic is discussed in @ref dev_guide_understanding_memory_formats.
///
///     The user can query the amount of memory required by a memory
///     descriptor using the #dnnl::memory::desc::get_size() function. The
///     size of data in general cannot be computed as the product of
///     dimensions multiplied by the size of the data type. So users are
///     required to use this function for better code portability.
///
///     Two memory descriptors can be compared using the equality and
///     inequality operators.  The comparison is especially useful when
///     checking whether it is necessary to reorder data from the user's data
///     format to a primitive's format.
///
/// 2. **Memory object** -- an engine-specific object that handles the memory
///     buffer and its description (a memory descriptor). For the CPU engine or
///     with USM, the memory buffer handle is simply a pointer to @c void. The
///     memory buffer can be queried using #dnnl::memory::get_data_handle() and
///     set using #dnnl::memory::set_data_handle(). The underlying SYCL buffer,
///     when used, can be queried using #dnnl::sycl_interop::get_buffer and set
///     using #dnnl::sycl_interop::set_buffer. A memory object can also be
///     queried for the underlying memory descriptor and for its engine using
///     #dnnl::memory::get_desc() and dnnl::memory::get_engine().
///
/// Along with ordinary memory descriptors with all dimensions being positive,
/// the library supports *zero-volume*  memory descriptors with one or more
/// dimensions set to zero. This is used to support the NumPy\* convention.
/// If a zero-volume memory is passed to a primitive, the primitive typically
/// does not perform any computations with this memory. For example:
///
/// - A concatenation primitive would ignore all memory object with zeroes in
///   the concat dimension / axis.
///
/// - A forward convolution with a source memory object with zero in the
///   minibatch dimension would always produce a destination memory object
///   with a zero in the minibatch dimension and perform no computations.
///
/// - However, a forward convolution with a zero in one of the weights
///   dimensions is ill-defined and is considered to be an error by the
///   library because there is no clear definition of what the output values
///   should be.
///
/// Memory buffer of a zero-volume memory is never accessed.
///
/// @{

/// Memory object.
///
/// A memory object encapsulates a handle to a memory buffer allocated on a
/// specific engine, tensor dimensions, data type, and memory format, which is
/// the way tensor indices map to offsets in linear memory space. Memory
/// objects are passed to primitives during execution.
struct memory : public handle<dnnl_memory_t> {
    using handle::handle;

    /// Integer type for representing dimension sizes and indices.
    typedef dnnl_dim_t dim;
    /// Vector of dimensions. Implementations are free to force a limit on the
    /// vector's length.
    typedef std::vector<dim> dims;

    /// Helper function that validates that an `std::vector` of dimensions can
    /// be safely converted to the C API array ::dnnl_dims_t. Throws if
    /// validation fails.
    ///
    /// @param v Vector of dimensions.
    /// @param min_size Minimum expected size of the vector.
    template <typename T>
    static void validate_dims(const std::vector<T> &v, int min_size = 0) {
        validate_container_size(
                v, "dimensions are invalid", min_size, DNNL_MAX_NDIMS);
    }

    /// Data type specification.
    enum class data_type {
        /// Undefined data type (used for empty memory descriptors).
        undef = dnnl_data_type_undef,
        /// [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
        /// with a 5-bit exponent and a 2-bit mantissa.
        f8_e5m2 = dnnl_f8_e5m2,
        /// [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
        /// with a 4-bit exponent and a 3-bit mantissa.
        f8_e4m3 = dnnl_f8_e4m3,
        /// [16-bit/half-precision floating point](https://en.wikipedia.org/wiki/Half-precision_floating-point_format).
        f16 = dnnl_f16,
        /// non-standard
        /// [16-bit floating point with 7-bit mantissa](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).
        bf16 = dnnl_bf16,
        /// [32-bit/single-precision floating point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format).
        f32 = dnnl_f32,
        //// [64-bit/double-precision floating point](https://en.wikipedia.org/wiki/Double-precision_floating-point_format).
        f64 = dnnl_f64,
        /// 32-bit signed integer.
        s32 = dnnl_s32,
        /// 8-bit signed integer.
        s8 = dnnl_s8,
        /// 8-bit unsigned integer.
        u8 = dnnl_u8,
        /// 4-bit signed integer.
        s4 = dnnl_s4,
        /// 4-bit unsigned integer.
        u4 = dnnl_u4,
    };

    /// Returns size of data type in bytes.
    /// @returns The number of bytes occupied by data type.
    static size_t data_type_size(data_type adata_type) {
        return dnnl_data_type_size(convert_to_c(adata_type));
    }

    /// Memory format kind
    enum class format_kind {
        /// Undefined memory format kind, used for empty memory descriptors.
        undef = dnnl_format_kind_undef,
        /// A special format kind that indicates that the actual format will be
        /// selected by a primitive automatically.
        any = dnnl_format_kind_any,
        /// A tensor in a generic format described by the stride and blocking
        /// values in each dimension.
        blocked = dnnl_blocked,
#ifdef DNNL_EXPERIMENTAL_SPARSE
        /// Format kind for sparse tensors.
        sparse = dnnl_format_kind_sparse,
#endif
        /// A special format kind that indicates that tensor format is opaque.
        opaque = dnnl_format_kind_opaque,
    };

#ifdef DNNL_EXPERIMENTAL_SPARSE
    /// Sparse encodings.
    enum class sparse_encoding {
            /// Undefined sparse encoding kind, used for empty memory descriptors.
            undef = dnnl_sparse_encoding_undef,
            /// Compressed Sparse Row (CSR) encoding.
            csr = dnnl_csr,
            /// An encoding that is used for an opaque storage schema for
            /// tensors with unstructured sparsity. A memory descriptor with the
            /// packed encoding cannot be used to create a memory object. It can
            /// only be used to create a primitive descriptor to query the
            /// actual memory descriptor (similar to the format tag `any`).
            packed = dnnl_packed,
    };
#endif

    /// Memory format tag specification.
    ///
    /// Memory format tags can be further divided into two categories:
    ///
    ///  - Domain-agnostic names, i.e. names that do not depend on the tensor
    ///    usage in the specific primitive. These names use letters from `a`
    ///    to `f` to denote logical dimensions and form the order in which the
    ///    dimensions are laid in memory. For example,
    ///    #dnnl::memory::format_tag::ab is used to denote a 2D tensor where the
    ///    second logical dimension (denoted as `b`) is the innermost, i.e.
    ///    has stride = 1, and the first logical dimension (`a`) is laid out in
    ///    memory with stride equal to the size of the second dimension. On the
    ///    other hand, #dnnl::memory::format_tag::ba is the transposed version
    ///    of the same tensor: the outermost dimension (`a`) becomes the
    ///    innermost one.
    ///
    ///  - Domain-specific names, i.e. names that make sense only in the
    ///    context of a certain domain, such as CNN. These names are
    ///    aliases to the corresponding domain-agnostic tags and used mostly
    ///    for convenience. For example, #dnnl::memory::format_tag::nc
    ///    is used to denote 2D CNN activations tensor memory format, where
    ///    the channels dimension is the innermost one and the batch dimension
    ///    is the outermost one. Moreover, #dnnl::memory::format_tag::nc is
    ///    an alias for #dnnl::memory::format_tag::ab, because for
    ///    CNN primitives the logical dimensions of activations tensors come
    ///    in order: batch, channels, spatial.  In other words, batch
    ///    corresponds to the first logical dimension (`a`), and channels
    ///    correspond to the second one (`b`).
    ///
    /// The following domain-specific notation applies to memory format tags:
    ///  - @c 'n' denotes the mini-batch dimension
    ///  - @c 'c' denotes a channels dimension
    ///  - When there are multiple channel dimensions (for example,
    ///    in convolution weights tensor), @c 'i' and @c 'o' denote dimensions
    ///    of input and output channels
    ///  - @c 'g' denotes a groups dimension for convolution weights
    ///  - @c 'd', @c 'h', and @c 'w' denote spatial depth, height, and width
    ///    respectively
    ///
    /// See @ref dnnl_format_tag_t for a detailed description.
    enum class format_tag {
        /// Undefined memory format tag
        undef = dnnl_format_tag_undef,
        /// Placeholder memory format tag. Used to instruct the primitive to
        /// select a format automatically.
        any = dnnl_format_tag_any,

        /// plain 1D tensor
        a = dnnl_a,

        /// plain 2D tensor
        ab = dnnl_ab,
        /// permuted 2D tensor
        ba = dnnl_ba,

        /// plain 3D tensor
        abc = dnnl_abc,
        /// permuted 3D tensor
        acb = dnnl_acb,
        /// permuted 3D tensor
        bac = dnnl_bac,
        /// permuted 3D tensor
        bca = dnnl_bca,
        /// permuted 3D tensor
        cba = dnnl_cba,

        /// plain 4D tensor
        abcd = dnnl_abcd,
        /// permuted 4D tensor
        abdc = dnnl_abdc,
        /// permuted 4D tensor
        acbd = dnnl_acbd,
        /// permuted 4D tensor
        acdb = dnnl_acdb,
        /// permuted 4D tensor
        adbc = dnnl_adbc,
        /// permuted 4D tensor
        bacd = dnnl_bacd,
        /// permuted 4D tensor
        bcda = dnnl_bcda,
        /// permuted 4D tensor
        cdba = dnnl_cdba,
        /// permuted 4D tensor
        dcab = dnnl_dcab,

        /// plain 5D tensor
        abcde = dnnl_abcde,
        /// permuted 5D tensor
        abdec = dnnl_abdec,
        /// permuted 5D tensor
        acbde = dnnl_acbde,
        /// permuted 5D tensor
        acdeb = dnnl_acdeb,
        /// permuted 5D tensor
        bacde = dnnl_bacde,
        /// permuted 5D tensor
        bcdea = dnnl_bcdea,
        /// permuted 5D tensor
        cdeba = dnnl_cdeba,
        /// permuted 5D tensor
        decab = dnnl_decab,
        /// permuted 5D tensor
        abced = dnnl_abced,

        /// plain 6D tensor
        abcdef = dnnl_abcdef,
        /// permuted 6D tensor
        abdfce = dnnl_abdfce,
        /// permuted 6D tensor
        acbdef = dnnl_acbdef,
        /// permuted 6D tensor
        abdefc = dnnl_abdefc,
        /// permuted 6D tensor
        defcab = dnnl_defcab,
        /// permuted 6D tensor
        abcdfe = dnnl_abcdfe,

        /// plain 7D tensor
        abcdefg = dnnl_abcdefg,
        /// permuted 7D tensor
        abcdegf = dnnl_abcdegf,

        /// plain 8D tensor
        abcdefgh = dnnl_abcdefgh,
        /// permuted 8D tensor
        abcdefhg = dnnl_abcdefhg,

        /// plain 9D tensor
        abcdefghi = dnnl_abcdefghi,
        /// permuted 9D tensor
        abcdefgih = dnnl_abcdefgih,

        /// plain 10D tensor
        abcdefghij = dnnl_abcdefghij,
        /// permuted 10D tensor
        abcdefghji = dnnl_abcdefghji,

        /// plain 11D tensor
        abcdefghijk = dnnl_abcdefghijk,
        /// permuted 11D tensor
        abcdefghikj = dnnl_abcdefghikj,

        /// plain 12D tensor
        abcdefghijkl = dnnl_abcdefghijkl,
        /// permuted 12D tensor
        abcdefghijlk = dnnl_abcdefghijlk,

        /// 1D tensor; an alias for #dnnl::memory::format_tag::a
        x = a,
        /// 2D CNN activations tensor; an alias for #dnnl::memory::format_tag::ab
        nc = ab,
        /// 2D CNN activations tensor; an alias for #dnnl::memory::format_tag::ba
        cn = ba,
        /// 2D RNN statistics tensor; an alias for #dnnl::memory::format_tag::ab
        tn = ab,
        /// 2D RNN statistics tensor; an alias for #dnnl::memory::format_tag::ba
        nt = ba,
        /// 3D CNN activations tensor; an alias for #dnnl::memory::format_tag::abc
        ncw = abc,
        /// 3D CNN activations tensor; an alias for #dnnl::memory::format_tag::acb
        nwc = acb,
        /// 4D CNN activations tensor; an alias for #dnnl::memory::format_tag::abcd
        nchw = abcd,
        /// 4D CNN activations tensor; an alias for #dnnl::memory::format_tag::acdb
        nhwc = acdb,
        /// 4D CNN activations tensor; an alias for #dnnl::memory::format_tag::bcda
        chwn = bcda,
        /// 5D CNN activations tensor; an alias for #dnnl::memory::format_tag::abcde
        ncdhw = abcde,
        /// 5D CNN activations tensor; an alias for #dnnl::memory::format_tag::acdeb
        ndhwc = acdeb,

        /// 2D CNN weights tensor; an alias for #dnnl::memory::format_tag::ab
        oi = ab,
        /// 2D CNN weights tensor; an alias for #dnnl::memory::format_tag::ba
        io = ba,
        /// 3D CNN weights tensor; an alias for #dnnl::memory::format_tag::abc
        oiw = abc,
        /// 3D CNN weights tensor; an alias for #dnnl::memory::format_tag::acb
        owi = acb,
        /// 3D CNN weights tensor; an alias for #dnnl::memory::format_tag::cba
        wio = cba,
        /// 3D CNN weights tensor; an alias for #dnnl::memory::format_tag::bca
        iwo = bca,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::abcd
        oihw = abcd,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::cdba
        hwio = cdba,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::acdb
        ohwi = acdb,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::bcda
        ihwo = bcda,
        /// 4D CNN weights tensor; an alias for #dnnl::memory::format_tag::bacd
        iohw = bacd,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::abcde
        oidhw = abcde,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::cdeba
        dhwio = cdeba,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::acdeb
        odhwi = acdeb,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::bacde
        iodhw = bacde,
        /// 5D CNN weights tensor; an alias for #dnnl::memory::format_tag::bcdea
        idhwo = bcdea,

        /// 4D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abcd
        goiw = abcd,
        /// 4D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abdc
        gowi = abdc,
        /// 4D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::dcab
        wigo = dcab,
        /// 5D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abdec
        gohwi = abdec,
        /// 5D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abcde
        goihw = abcde,
        /// 5D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::decab
        hwigo = decab,
        /// 5D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::acbde
        giohw = acbde,
        /// 6D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abcdef
        goidhw = abcdef,
        /// 6D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abcdef
        giodhw = acbdef,
        /// 6D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::abdefc
        godhwi = abdefc,
        /// 6D CNN weights tensor with groups; an alias for #dnnl::memory::format_tag::defcab
        dhwigo = defcab,

        /// 3D RNN data tensor in the format (seq_length, batch, input
        /// channels); an alias for #dnnl::memory::format_tag::abc.
        tnc = abc,
        /// 3D RNN data tensor in the format (batch, seq_length, input
        /// channels); an alias for #dnnl::memory::format_tag::bac.
        ntc = bac,
        /// 4D RNN states tensor in the format (num_layers, num_directions,
        /// batch, state channels); an alias for #dnnl::memory::format_tag::abcd.
        ldnc = abcd,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        /// input_channels, num_gates, output_channels);
        /// an alias for #dnnl::memory::format_tag::abcde.
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldigo = abcde,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels, input_channels);
        /// an alias for #dnnl::memory::format_tag::abdec.
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgoi = abdec,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_hidden_state, num_channels_in_recurrent_projection);
        /// an alias for #dnnl::memory::format_tag::abcd.
        ldio = abcd,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_recurrent_projection, num_channels_in_hidden_state);
        /// an alias for #dnnl::memory::format_tag::abdc.
        ldoi = abdc,
        /// 4D RNN bias tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels);
        /// an alias for #dnnl::memory::format_tag::abcd.
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgo = abcd,

        // Opaque blocked formats

        AB16b16a = dnnl_AB16b16a,
        AB16b32a = dnnl_AB16b32a,
        AB16b48a = dnnl_AB16b48a,
        AB16b64a = dnnl_AB16b64a,
        AB8b16a2b = dnnl_AB8b16a2b,
        AB8b32a2b = dnnl_AB8b32a2b,
        AB8b64a2b = dnnl_AB8b64a2b,
        AB4b16a4b = dnnl_AB4b16a4b,
        AB4b32a4b = dnnl_AB4b32a4b,
        AB4b64a4b = dnnl_AB4b64a4b,
        AB16b16a4b = dnnl_AB16b16a4b,
        AB16b32a4b = dnnl_AB16b32a4b,
        AB16b48a4b = dnnl_AB16b48a4b,
        AB16b64a4b = dnnl_AB16b64a4b,
        AB16b16a2b = dnnl_AB16b16a2b,
        AB16b32a2b = dnnl_AB16b32a2b,
        AB16b48a2b = dnnl_AB16b48a2b,
        AB16b64a2b = dnnl_AB16b64a2b,
        Ab4a = dnnl_Ab4a,
        Ab8a = dnnl_Ab8a,
        Abc16a = dnnl_Abc16a,
        ABc16a16b = dnnl_ABc16a16b,
        ABc4a4b = dnnl_ABc4a4b,
        aBc16b = dnnl_aBc16b,
        aBc32b = dnnl_aBc32b,
        ABc16b16a = dnnl_ABc16b16a,
        AcB16b16a = dnnl_AcB16b16a,
        ABc16b32a = dnnl_ABc16b32a,
        AcB16b32a = dnnl_AcB16b32a,
        ABc16b48a = dnnl_ABc16b48a,
        AcB16b48a = dnnl_AcB16b48a,
        ABc16b64a = dnnl_ABc16b64a,
        AcB16b64a = dnnl_AcB16b64a,
        Abc4a = dnnl_Abc4a,
        aBc4b = dnnl_aBc4b,
        ABc4b16a4b = dnnl_ABc4b16a4b,
        AcB4b16a4b = dnnl_AcB4b16a4b,
        ABc4b32a4b = dnnl_ABc4b32a4b,
        AcB4b32a4b = dnnl_AcB4b32a4b,
        ABc4b64a4b = dnnl_ABc4b64a4b,
        AcB4b64a4b = dnnl_AcB4b64a4b,
        ABc2b8a4b = dnnl_ABc2b8a4b,
        ABc16a16b2a = dnnl_ABc16a16b2a,
        ABc16b16a4b = dnnl_ABc16b16a4b,
        ABc16b32a4b = dnnl_ABc16b32a4b,
        ABc16b48a4b = dnnl_ABc16b48a4b,
        ABc16b64a4b = dnnl_ABc16b64a4b,
        ABc16b16a2b = dnnl_ABc16b16a2b,
        ABc16b32a2b = dnnl_ABc16b32a2b,
        ABc16b48a2b = dnnl_ABc16b48a2b,
        ABc16b64a2b = dnnl_ABc16b64a2b,
        ABc4b4a = dnnl_ABc4b4a,
        ABc8a16b2a = dnnl_ABc8a16b2a,
        ABc8a8b = dnnl_ABc8a8b,
        ABc8a4b = dnnl_ABc8a4b,
        aBc8b = dnnl_aBc8b,
        ABc8b16a2b = dnnl_ABc8b16a2b,
        AcB8b16a2b = dnnl_AcB8b16a2b,
        ABc8b32a2b = dnnl_ABc8b32a2b,
        AcB8b32a2b = dnnl_AcB8b32a2b,
        ABc8b64a2b = dnnl_ABc8b64a2b,
        AcB8b64a2b = dnnl_AcB8b64a2b,
        ABc8b8a = dnnl_ABc8b8a,
        AcB8b8a = dnnl_AcB8b8a,
        Abcd8a = dnnl_Abcd8a,
        Abcd16a = dnnl_Abcd16a,
        Abcd32a = dnnl_Abcd32a,
        ABcd16a16b = dnnl_ABcd16a16b,
        aBcd16b = dnnl_aBcd16b,
        aBcd32b = dnnl_aBcd32b,
        ABcd16b16a = dnnl_ABcd16b16a,
        AcdB16b16a = dnnl_AcdB16b16a,
        ABcd16b32a = dnnl_ABcd16b32a,
        AcdB16b32a = dnnl_AcdB16b32a,
        ABcd16b48a = dnnl_ABcd16b48a,
        AcdB16b48a = dnnl_AcdB16b48a,
        ABcd16b64a = dnnl_ABcd16b64a,
        AcdB16b64a = dnnl_AcdB16b64a,
        aBCd16b16c = dnnl_aBCd16b16c,
        aBCd16c16b = dnnl_aBCd16c16b,
        Abcd4a = dnnl_Abcd4a,
        aBcd4b = dnnl_aBcd4b,
        ABcd4b16a4b = dnnl_ABcd4b16a4b,
        AcdB4b16a4b = dnnl_AcdB4b16a4b,
        ABcd4b32a4b = dnnl_ABcd4b32a4b,
        AcdB4b32a4b = dnnl_AcdB4b32a4b,
        ABcd4b64a4b = dnnl_ABcd4b64a4b,
        AcdB4b64a4b = dnnl_AcdB4b64a4b,
        ABcd2b8a4b = dnnl_ABcd2b8a4b,
        ABcd4b4a = dnnl_ABcd4b4a,
        ABcd4a4b = dnnl_ABcd4a4b,
        aBCd4c16b4c = dnnl_aBCd4c16b4c,
        aBCd2c8b4c = dnnl_aBCd2c8b4c,
        ABcd16a16b2a = dnnl_ABcd16a16b2a,
        ABcd16b16a4b = dnnl_ABcd16b16a4b,
        ABcd16b32a4b = dnnl_ABcd16b32a4b,
        ABcd16b48a4b = dnnl_ABcd16b48a4b,
        ABcd16b64a4b = dnnl_ABcd16b64a4b,
        ABcd16b16a2b = dnnl_ABcd16b16a2b,
        ABcd16b32a2b = dnnl_ABcd16b32a2b,
        ABcd16b48a2b = dnnl_ABcd16b48a2b,
        ABcd16b64a2b = dnnl_ABcd16b64a2b,
        aBCd16b16c2b = dnnl_aBCd16b16c2b,
        aBCd16c16b4c = dnnl_aBCd16c16b4c,
        aBCd16c16b2c = dnnl_aBCd16c16b2c,
        aBCd4c4b = dnnl_aBCd4c4b,
        aBCd4b4c = dnnl_aBCd4b4c,
        ABcd8a16b2a = dnnl_ABcd8a16b2a,
        ABcd8a8b = dnnl_ABcd8a8b,
        ABcd8a4b = dnnl_ABcd8a4b,
        ABcd8a2b = dnnl_ABcd8a2b,
        /// 4D tensor blocked by 2nd dimension with block size 8
        aBcd8b = dnnl_aBcd8b,
        ABcd8b16a2b = dnnl_ABcd8b16a2b,
        AcdB8b16a2b = dnnl_AcdB8b16a2b,
        ABcd8b32a2b = dnnl_ABcd8b32a2b,
        AcdB8b32a2b = dnnl_AcdB8b32a2b,
        ABcd8b64a2b = dnnl_ABcd8b64a2b,
        AcdB8b64a2b = dnnl_AcdB8b64a2b,
        aBCd8b16c2b = dnnl_aBCd8b16c2b,
        /// 4D tensor blocked by 1st and 2nd dimension with block size 8
        ABcd8b8a = dnnl_ABcd8b8a,
        AcdB8b8a = dnnl_AcdB8b8a,
        aBCd8b8c = dnnl_aBCd8b8c,
        aBCd8b4c = dnnl_aBCd8b4c,
        aBCd8c16b2c = dnnl_aBCd8c16b2c,
        aBCd8c8b = dnnl_aBCd8c8b,
        Abcde16a = dnnl_Abcde16a,
        Abcde32a = dnnl_Abcde32a,
        ABcde16a16b = dnnl_ABcde16a16b,
        aBcde16b = dnnl_aBcde16b,
        aBcde32b = dnnl_aBcde32b,
        ABcde16b16a = dnnl_ABcde16b16a,
        AcdeB16b16a = dnnl_AcdeB16b16a,
        ABcde16b32a = dnnl_ABcde16b32a,
        AcdeB16b32a = dnnl_AcdeB16b32a,
        ABcde16b48a = dnnl_ABcde16b48a,
        AcdeB16b48a = dnnl_AcdeB16b48a,
        ABcde16b64a = dnnl_ABcde16b64a,
        AcdeB16b64a = dnnl_AcdeB16b64a,
        aBCde16b16c = dnnl_aBCde16b16c,
        aBCde16c16b = dnnl_aBCde16c16b,
        aBCde2c8b4c = dnnl_aBCde2c8b4c,
        Abcde4a = dnnl_Abcde4a,
        aBcde4b = dnnl_aBcde4b,
        ABcde4b4a = dnnl_ABcde4b4a,
        ABcde4a4b = dnnl_ABcde4a4b,
        aBCde4b4c = dnnl_aBCde4b4c,
        aBCde4c16b4c = dnnl_aBCde4c16b4c,
        aBCde16b16c2b = dnnl_aBCde16b16c2b,
        aBCde16c16b4c = dnnl_aBCde16c16b4c,
        aBCde16c16b2c = dnnl_aBCde16c16b2c,
        aBCdef16c16b2c = dnnl_aBCdef16c16b2c,
        aBCde4c4b = dnnl_aBCde4c4b,
        Abcde8a = dnnl_Abcde8a,
        ABcde8a8b = dnnl_ABcde8a8b,
        ABcde8a4b = dnnl_ABcde8a4b,
        aBcde8b = dnnl_aBcde8b,
        ABcde8b16a2b = dnnl_ABcde8b16a2b,
        AcdeB8b16a2b = dnnl_AcdeB8b16a2b,
        ABcde8b32a2b = dnnl_ABcde8b32a2b,
        AcdeB8b32a2b = dnnl_AcdeB8b32a2b,
        ABcde8b64a2b = dnnl_ABcde8b64a2b,
        AcdeB8b64a2b = dnnl_AcdeB8b64a2b,
        ABcde4b16a4b = dnnl_ABcde4b16a4b,
        AcdeB4b16a4b = dnnl_AcdeB4b16a4b,
        ABcde4b32a4b = dnnl_ABcde4b32a4b,
        AcdeB4b32a4b = dnnl_AcdeB4b32a4b,
        ABcde4b64a4b = dnnl_ABcde4b64a4b,
        AcdeB4b64a4b = dnnl_AcdeB4b64a4b,
        ABcde16b16a4b = dnnl_ABcde16b16a4b,
        ABcde16b32a4b = dnnl_ABcde16b32a4b,
        ABcde16b48a4b = dnnl_ABcde16b48a4b,
        ABcde16b64a4b = dnnl_ABcde16b64a4b,
        ABcde16b16a2b = dnnl_ABcde16b16a2b,
        ABcde16b32a2b = dnnl_ABcde16b32a2b,
        ABcde16b48a2b = dnnl_ABcde16b48a2b,
        ABcde16b64a2b = dnnl_ABcde16b64a2b,
        ABcde2b8a4b = dnnl_ABcde2b8a4b,
        aBCde8b16c2b = dnnl_aBCde8b16c2b,
        ABcde8b8a = dnnl_ABcde8b8a,
        AcdeB8b8a = dnnl_AcdeB8b8a,
        aBCde8b8c = dnnl_aBCde8b8c,
        aBCde8b4c = dnnl_aBCde8b4c,
        ABcd4a8b8a4b = dnnl_ABcd4a8b8a4b,
        ABcd2a8b8a2b = dnnl_ABcd2a8b8a2b,
        aBCde4b8c8b4c = dnnl_aBCde4b8c8b4c,
        aBCde2b8c8b2c = dnnl_aBCde2b8c8b2c,
        aBCde8c16b2c = dnnl_aBCde8c16b2c,
        aBCde8c8b = dnnl_aBCde8c8b,
        aBcdef16b = dnnl_aBcdef16b,
        aBCdef16b16c = dnnl_aBCdef16b16c,
        aBCdef16c16b = dnnl_aBCdef16c16b,
        aBcdef4b = dnnl_aBcdef4b,
        aBCdef2c8b4c = dnnl_aBCdef2c8b4c,
        aBCdef4c4b = dnnl_aBCdef4c4b,
        aBCdef4b4c = dnnl_aBCdef4b4c,
        aBCdef8b8c = dnnl_aBCdef8b8c,
        aBCdef8b4c = dnnl_aBCdef8b4c,
        aBCdef8c16b2c = dnnl_aBCdef8c16b2c,
        aBCdef4c16b4c = dnnl_aBCdef4c16b4c,
        aBCdef8c8b = dnnl_aBCdef8c8b,
        aBdc16b = dnnl_aBdc16b,
        aBdc4b = dnnl_aBdc4b,
        aBdc8b = dnnl_aBdc8b,
        aBdC8b2c = dnnl_aBdC8b2c,
        aBdC8b4c = dnnl_aBdC8b4c,
        aBdec16b = dnnl_aBdec16b,
        aBdec4b = dnnl_aBdec4b,
        aBdec8b = dnnl_aBdec8b,
        aBdeC8b2c = dnnl_aBdeC8b2c,
        aBdeC8b4c = dnnl_aBdeC8b4c,
        aBdefc16b = dnnl_aBdefc16b,
        aCBdef16c16b = dnnl_aCBdef16c16b,
        aCBdef16b16c = dnnl_aCBdef16b16c,
        aBdefc4b = dnnl_aBdefc4b,
        aBdefc8b = dnnl_aBdefc8b,
        aBdefC8b2c = dnnl_aBdefC8b2c,
        aBdefC8b4c = dnnl_aBdefC8b4c,
        Acb16a = dnnl_Acb16a,
        Acb4a = dnnl_Acb4a,
        Acb8a = dnnl_Acb8a,
        AcB8a2b = dnnl_AcB8a2b,
        AcB8a4b = dnnl_AcB8a4b,
        aCBd16b16c = dnnl_aCBd16b16c,
        aCBd16c16b = dnnl_aCBd16c16b,
        aCBde16b16c = dnnl_aCBde16b16c,
        aCBde16c16b = dnnl_aCBde16c16b,
        Acdb16a = dnnl_Acdb16a,
        Acdb4a = dnnl_Acdb4a,
        Acdb8a = dnnl_Acdb8a,
        AcdB8a2b = dnnl_AcdB8a2b,
        AcdB8a4b = dnnl_AcdB8a4b,
        Acdeb16a = dnnl_Acdeb16a,
        Acdeb4a = dnnl_Acdeb4a,
        Acdeb8a = dnnl_Acdeb8a,
        AcdeB8a2b = dnnl_AcdeB8a2b,
        AcdeB8a4b = dnnl_AcdeB8a4b,
        BAc16a16b = dnnl_BAc16a16b,
        BAc16b16a = dnnl_BAc16b16a,
        BAcd16a16b = dnnl_BAcd16a16b,
        BAcd16b16a = dnnl_BAcd16b16a,
        ABcd32a32b = dnnl_ABcd32a32b,
        BAcde16b16a = dnnl_BAcde16b16a,
        BAcde16a16b = dnnl_BAcde16a16b,
        aBdec32b = dnnl_aBdec32b,
        Abcdef16a = dnnl_Abcdef16a,
        Abcdef32a = dnnl_Abcdef32a,
        Acdb32a = dnnl_Acdb32a,
        aBCd2b4c2b = dnnl_aBCd2b4c2b,
        aBCde2b4c2b = dnnl_aBCde2b4c2b,
        aBCdef2b4c2b = dnnl_aBCdef2b4c2b,
        aBCd2c4b2c = dnnl_aBCd2c4b2c,
        aBCde2c4b2c = dnnl_aBCde2c4b2c,
        aBCdef2c4b2c = dnnl_aBCdef2c4b2c,
        aBCd4b8c2b = dnnl_aBCd4b8c2b,
        aBCde4b8c2b = dnnl_aBCde4b8c2b,
        aBCdef4b8c2b = dnnl_aBCdef4b8c2b,
        aBCd4c8b2c = dnnl_aBCd4c8b2c,
        aBCde4c8b2c = dnnl_aBCde4c8b2c,
        aBCdef4c8b2c = dnnl_aBCdef4c8b2c,
        AB32a32b8a4b = dnnl_AB32a32b8a4b,
        AB32a32b8a2b = dnnl_AB32a32b8a2b,
        AB8a4b = dnnl_AB8a4b,
        AB8a2b = dnnl_AB8a2b,
        abDc16d = dnnl_abDc16d,
        abDc32d = dnnl_abDc32d,
        abDC32d4c = dnnl_abDC32d4c,
        abCd32c = dnnl_abCd32c,
        abdEc16e = dnnl_abdEc16e,
        abdEc32e = dnnl_abdEc32e,
        abdEC32e2c = dnnl_abdEC32e2c,
        abdEC32e4c = dnnl_abdEC32e4c,
        abdCe16c = dnnl_abdCe16c,
        abdCe32c = dnnl_abdCe32c,
        abdCE32c2e = dnnl_abdCE32c2e,
        aBCdef16c16b4c = dnnl_aBCdef16c16b4c,
        aBdC16b4c = dnnl_aBdC16b4c,
        aBdeC16b4c = dnnl_aBdeC16b4c,
        AcB16a4b = dnnl_AcB16a4b,
        AcdB16a2b = dnnl_AcdB16a2b,
        aBdefC16b4c = dnnl_aBdefC16b4c,
        AcdeB16a4b = dnnl_AcdeB16a4b,

        Acb32a = dnnl_Acb32a,
        AcB32a2b = dnnl_AcB32a2b,
        AcB32a4b = dnnl_AcB32a4b,
        Acb48a = dnnl_Acb48a,
        AcB48a2b = dnnl_AcB48a2b,
        AcB48a4b = dnnl_AcB48a4b,
        Acb64a = dnnl_Acb64a,
        AcB64a2b = dnnl_AcB64a2b,
        AcB64a4b = dnnl_AcB64a4b,
        cBa2b = dnnl_cBa2b,
        cBa4b = dnnl_cBa4b,
        aBdc32b = dnnl_aBdc32b,
        aBdC32b2c = dnnl_aBdC32b2c,
        aBdC32b4c = dnnl_aBdC32b4c,
        aBdc48b = dnnl_aBdc48b,
        aBdC48b2c = dnnl_aBdC48b2c,
        aBdC48b4c = dnnl_aBdC48b4c,
        aBdc64b = dnnl_aBdc64b,
        aBdC64b2c = dnnl_aBdC64b2c,
        aBdC64b4c = dnnl_aBdC64b4c,
        adcb = dnnl_adcb,
        adCb2c = dnnl_adCb2c,
        adCb4c = dnnl_adCb4c,
        AcdB32a2b = dnnl_AcdB32a2b,
        AcdB32a4b = dnnl_AcdB32a4b,
        Acdb48a = dnnl_Acdb48a,
        AcdB48a2b = dnnl_AcdB48a2b,
        AcdB48a4b = dnnl_AcdB48a4b,
        Acdb64a = dnnl_Acdb64a,
        AcdB64a2b = dnnl_AcdB64a2b,
        AcdB64a4b = dnnl_AcdB64a4b,
        cdBa2b = dnnl_cdBa2b,
        cdBa4b = dnnl_cdBa4b,
        aBdeC32b2c = dnnl_aBdeC32b2c,
        aBdeC32b4c = dnnl_aBdeC32b4c,
        aBdec48b = dnnl_aBdec48b,
        aBdeC48b2c = dnnl_aBdeC48b2c,
        aBdeC48b4c = dnnl_aBdeC48b4c,
        aBdec64b = dnnl_aBdec64b,
        aBdeC64b2c = dnnl_aBdeC64b2c,
        aBdeC64b4c = dnnl_aBdeC64b4c,
        adecb = dnnl_adecb,
        adeCb2c = dnnl_adeCb2c,
        adeCb4c = dnnl_adeCb4c,
        Acdeb32a = dnnl_Acdeb32a,
        AcdeB32a2b = dnnl_AcdeB32a2b,
        AcdeB32a4b = dnnl_AcdeB32a4b,
        Acdeb48a = dnnl_Acdeb48a,
        AcdeB48a2b = dnnl_AcdeB48a2b,
        AcdeB48a4b = dnnl_AcdeB48a4b,
        Acdeb64a = dnnl_Acdeb64a,
        AcdeB64a2b = dnnl_AcdeB64a2b,
        AcdeB64a4b = dnnl_AcdeB64a4b,
        cdeBa2b = dnnl_cdeBa2b,
        cdeBa4b = dnnl_cdeBa4b,
        aBdefc32b = dnnl_aBdefc32b,
        aBdefC32b2c = dnnl_aBdefC32b2c,
        aBdefC32b4c = dnnl_aBdefC32b4c,
        aBdefc48b = dnnl_aBdefc48b,
        aBdefC48b2c = dnnl_aBdefC48b2c,
        aBdefC48b4c = dnnl_aBdefC48b4c,
        aBdefc64b = dnnl_aBdefc64b,
        aBdefC64b2c = dnnl_aBdefC64b2c,
        aBdefC64b4c = dnnl_aBdefC64b4c,
        adefcb = dnnl_adefcb,
        adefCb2c = dnnl_adefCb2c,
        adefCb4c = dnnl_adefCb4c,
        ABc32a32b = dnnl_ABc32a32b,
        BAc8a16b2a = dnnl_BAc8a16b2a,
        BAcd8a16b2a = dnnl_BAcd8a16b2a,
        ABcde8a16b2a = dnnl_ABcde8a16b2a,
        aCBd8b16c2b = dnnl_aCBd8b16c2b,
        BAcde8a16b2a = dnnl_BAcde8a16b2a,
        aCBde8b16c2b = dnnl_aCBde8b16c2b,
        ABcde32a32b = dnnl_ABcde32a32b,
        ABc4a8b8a4b = dnnl_ABc4a8b8a4b,
        ABcde4a8b8a4b = dnnl_ABcde4a8b8a4b,
        BAc4b8a8b4a = dnnl_BAc4b8a8b4a,
        BAcd4b8a8b4a = dnnl_BAcd4b8a8b4a,
        BAcde4b8a8b4a = dnnl_BAcde4b8a8b4a,
        aBCd4b8c8b4c = dnnl_aBCd4b8c8b4c,
        aBCdef4b8c8b4c = dnnl_aBCdef4b8c8b4c,
        aBCdef8b16c2b = dnnl_aBCdef8b16c2b,
        aCBdef8b16c2b = dnnl_aCBdef8b16c2b,
        aBdC16b2c = dnnl_aBdC16b2c,
        aBdeC16b2c = dnnl_aBdeC16b2c,
        aBdefC16b2c = dnnl_aBdefC16b2c,
        aBedc16b = dnnl_aBedc16b,
        AcB16a2b = dnnl_AcB16a2b,
        AcdB16a4b = dnnl_AcdB16a4b,
        AcdeB16a2b = dnnl_AcdeB16a2b,
        Adcb16a = dnnl_Adcb16a,
        aCBd4c8b8c4b = dnnl_aCBd4c8b8c4b,
        aCBde4c8b8c4b = dnnl_aCBde4c8b8c4b,
        aCBdef4c8b8c4b = dnnl_aCBdef4c8b8c4b,
        ABc32a16b = dnnl_ABc32a16b,
        ABcd16a32b = dnnl_ABcd16a32b,
        ABcd32a16b = dnnl_ABcd32a16b,
        ABcde32a16b = dnnl_ABcde32a16b,
        AB48a16b = dnnl_AB48a16b,
        AB48a32b = dnnl_AB48a32b,
        ABc40a16b = dnnl_ABc40a16b,
        ABc40a32b = dnnl_ABc40a32b,
        aBC48b16c = dnnl_aBC48b16c,
        aBC48b32c = dnnl_aBC48b32c,
        ABcd40a16b = dnnl_ABcd40a16b,
        ABcd40a32b = dnnl_ABcd40a32b,
        BA16a16b = dnnl_BA16a16b,
        BA16a32b = dnnl_BA16a32b,
        BA16a48b = dnnl_BA16a48b,
        BA16a64b = dnnl_BA16a64b,
        BA16a16b2a = dnnl_BA16a16b2a,
        BA16a32b2a = dnnl_BA16a32b2a,
        BA16a48b2a = dnnl_BA16a48b2a,
        BA16a64b2a = dnnl_BA16a64b2a,
        BA16a16b4a = dnnl_BA16a16b4a,
        BA16a32b4a = dnnl_BA16a32b4a,
        BA16a48b4a = dnnl_BA16a48b4a,
        BA16a64b4a = dnnl_BA16a64b4a,
        decbA16a = dnnl_decbA16a,
        decbA8a = dnnl_decbA8a,
        defcbA16a = dnnl_defcbA16a,
        defcbA8a = dnnl_defcbA8a,
        aCB16b16c = dnnl_aCB16b16c,
        aCB16b32c = dnnl_aCB16b32c,
        aCB16b48c = dnnl_aCB16b48c,
        aCB16b64c = dnnl_aCB16b64c,
        aCB16b16c2b = dnnl_aCB16b16c2b,
        aCB16b32c2b = dnnl_aCB16b32c2b,
        aCB16b48c2b = dnnl_aCB16b48c2b,
        aCB16b64c2b = dnnl_aCB16b64c2b,
        aCB16b16c4b = dnnl_aCB16b16c4b,
        aCB16b32c4b = dnnl_aCB16b32c4b,
        aCB16b48c4b = dnnl_aCB16b48c4b,
        aCB16b64c4b = dnnl_aCB16b64c4b,
        Acb24a = dnnl_Acb24a,
        Acdb24a = dnnl_Acdb24a,
        Acdeb24a = dnnl_Acdeb24a,
        aBdc24b = dnnl_aBdc24b,
        aBdec24b = dnnl_aBdec24b,
        aBdefc24b = dnnl_aBdefc24b,
        AcB24a2b = dnnl_AcB24a2b,
        AcdB24a2b = dnnl_AcdB24a2b,
        AcdeB24a2b = dnnl_AcdeB24a2b,
        aBdC24b2c = dnnl_aBdC24b2c,
        aBdeC24b2c = dnnl_aBdeC24b2c,
        aBdefC24b2c = dnnl_aBdefC24b2c,
        AcB24a4b = dnnl_AcB24a4b,
        AcdB24a4b = dnnl_AcdB24a4b,
        AcdeB24a4b = dnnl_AcdeB24a4b,
        aBdC24b4c = dnnl_aBdC24b4c,
        aBdeC24b4c = dnnl_aBdeC24b4c,
        aBdefC24b4c = dnnl_aBdefC24b4c,
        AB8b32a = dnnl_AB8b32a,
        ABc8b32a = dnnl_ABc8b32a,
        AcB8b32a = dnnl_AcB8b32a,
        ABcd8b32a = dnnl_ABcd8b32a,
        AcdB8b32a = dnnl_AcdB8b32a,
        ABcde8b32a = dnnl_ABcde8b32a,
        AcdeB8b32a = dnnl_AcdeB8b32a,
        AB8b24a = dnnl_AB8b24a,
        ABc8b24a = dnnl_ABc8b24a,
        AcB8b24a = dnnl_AcB8b24a,
        ABcd8b24a = dnnl_ABcd8b24a,
        AcdB8b24a = dnnl_AcdB8b24a,
        ABcde8b24a = dnnl_ABcde8b24a,
        AcdeB8b24a = dnnl_AcdeB8b24a,
        AB8b16a = dnnl_AB8b16a,
        ABc8b16a = dnnl_ABc8b16a,
        AcB8b16a = dnnl_AcB8b16a,
        ABcd8b16a = dnnl_ABcd8b16a,
        AcdB8b16a = dnnl_AcdB8b16a,
        ABcde8b16a = dnnl_ABcde8b16a,
        AcdeB8b16a = dnnl_AcdeB8b16a,
        AB8b8a = dnnl_AB8b8a,

        format_tag_last = dnnl_format_tag_last,

        nCdhw16c = dnnl_nCdhw16c,
        nCdhw4c = dnnl_nCdhw4c,
        nCdhw8c = dnnl_nCdhw8c,
        nChw16c = dnnl_nChw16c,
        nChw4c = dnnl_nChw4c,
        nChw8c = dnnl_nChw8c,
        nCw16c = dnnl_nCw16c,
        nCw4c = dnnl_nCw4c,
        nCw8c = dnnl_nCw8c,
        NCw16n16c = dnnl_NCw16n16c,
        NChw16n16c = dnnl_NChw16n16c,
        NCdhw16n16c = dnnl_NCdhw16n16c,
        NCdhw32n32c = dnnl_NCdhw32n32c,
        NChw32n32c = dnnl_NChw32n32c,
        IOhw16i16o = dnnl_IOhw16i16o,
        OI16i16o = dnnl_OI16i16o,
        OI16i32o = dnnl_OI16i32o,
        OI16i48o = dnnl_OI16i48o,
        OI16i64o = dnnl_OI16i64o,
        OI8i16o2i = dnnl_OI8i16o2i,
        OI8i32o2i = dnnl_OI8i32o2i,
        OI8i64o2i = dnnl_OI8i64o2i,
        OI4i8o4i = dnnl_OI4i8o4i,
        OI4i16o4i = dnnl_OI4i16o4i,
        OI4i24o4i = dnnl_OI4i24o4i,
        OI4i32o4i = dnnl_OI4i32o4i,
        OI4i64o4i = dnnl_OI4i64o4i,
        Ohwi32o = dnnl_Ohwi32o,
        IOdhw16i16o = dnnl_IOdhw16i16o,
        gIOhw16i16o = dnnl_gIOhw16i16o,
        gOhwi32o = dnnl_gOhwi32o,
        Goidhw16g = dnnl_Goidhw16g,
        IOw16o16i = dnnl_IOw16o16i,
        OIw16i16o = dnnl_OIw16i16o,
        OwI16i16o = dnnl_OwI16i16o,
        OIw16i32o = dnnl_OIw16i32o,
        OwI16i32o = dnnl_OwI16i32o,
        OIw16i48o = dnnl_OIw16i48o,
        OwI16i48o = dnnl_OwI16i48o,
        OIw16i64o = dnnl_OIw16i64o,
        OwI16i64o = dnnl_OwI16i64o,
        IOw16i16o = dnnl_IOw16i16o,
        gIOw16i16o = dnnl_gIOw16i16o,
        OIw16o16i = dnnl_OIw16o16i,
        Oiw16o = dnnl_Oiw16o,
        OIw4i8o4i = dnnl_OIw4i8o4i,
        OwI4i8o4i = dnnl_OwI4i8o4i,
        OIw4i16o4i = dnnl_OIw4i16o4i,
        OwI4i16o4i = dnnl_OwI4i16o4i,
        OIw4i24o4i = dnnl_OIw4i24o4i,
        OwI4i24o4i = dnnl_OwI4i24o4i,
        OIw4i32o4i = dnnl_OIw4i32o4i,
        OwI4i32o4i = dnnl_OwI4i32o4i,
        OIw4i64o4i = dnnl_OIw4i64o4i,
        OwI4i64o4i = dnnl_OwI4i64o4i,
        OIw2i8o4i = dnnl_OIw2i8o4i,
        OIw4i4o = dnnl_OIw4i4o,
        OIw4o4i = dnnl_OIw4o4i,
        Oiw4o = dnnl_Oiw4o,
        OIw8i16o2i = dnnl_OIw8i16o2i,
        OwI8i16o2i = dnnl_OwI8i16o2i,
        OIw8i32o2i = dnnl_OIw8i32o2i,
        OwI8i32o2i = dnnl_OwI8i32o2i,
        OIw8i64o2i = dnnl_OIw8i64o2i,
        OwI8i64o2i = dnnl_OwI8i64o2i,
        OIw8i8o = dnnl_OIw8i8o,
        OwI8i8o = dnnl_OwI8i8o,
        OIw8o16i2o = dnnl_OIw8o16i2o,
        OIw8o8i = dnnl_OIw8o8i,
        OIw8o4i = dnnl_OIw8o4i,
        OIw16i16o4i = dnnl_OIw16i16o4i,
        OIw16i32o4i = dnnl_OIw16i32o4i,
        OIw16i48o4i = dnnl_OIw16i48o4i,
        OIw16i64o4i = dnnl_OIw16i64o4i,
        OIw16i16o2i = dnnl_OIw16i16o2i,
        OIw16i32o2i = dnnl_OIw16i32o2i,
        OIw16i48o2i = dnnl_OIw16i48o2i,
        OIw16i64o2i = dnnl_OIw16i64o2i,
        OIw16o16i2o = dnnl_OIw16o16i2o,
        Owi16o = dnnl_Owi16o,
        OwI16o2i = dnnl_OwI16o2i,
        Iwo16i = dnnl_Iwo16i,
        IwO16i2o = dnnl_IwO16i2o,
        IwO16i4o = dnnl_IwO16i4o,
        Owi4o = dnnl_Owi4o,
        Owi8o = dnnl_Owi8o,
        OwI8o2i = dnnl_OwI8o2i,
        OwI8o4i = dnnl_OwI8o4i,
        IOhw16o16i = dnnl_IOhw16o16i,
        Ohwi16o = dnnl_Ohwi16o,
        OhwI16o2i = dnnl_OhwI16o2i,
        Ihwo16i = dnnl_Ihwo16i,
        IhwO16i2o = dnnl_IhwO16i2o,
        IhwO16i4o = dnnl_IhwO16i4o,
        Ohwi4o = dnnl_Ohwi4o,
        Ohwi8o = dnnl_Ohwi8o,
        OhwI8o2i = dnnl_OhwI8o2i,
        OhwI8o4i = dnnl_OhwI8o4i,
        OIhw16i16o = dnnl_OIhw16i16o,
        OhwI16i16o = dnnl_OhwI16i16o,
        OIhw16i32o = dnnl_OIhw16i32o,
        OhwI16i32o = dnnl_OhwI16i32o,
        OIhw16i48o = dnnl_OIhw16i48o,
        OhwI16i48o = dnnl_OhwI16i48o,
        OIhw16i64o = dnnl_OIhw16i64o,
        OhwI16i64o = dnnl_OhwI16i64o,
        OIhw16o16i = dnnl_OIhw16o16i,
        Oihw16o = dnnl_Oihw16o,
        OIhw4i8o4i = dnnl_OIhw4i8o4i,
        OhwI4i8o4i = dnnl_OhwI4i8o4i,
        OIhw4i16o4i = dnnl_OIhw4i16o4i,
        OhwI4i16o4i = dnnl_OhwI4i16o4i,
        OIhw4i24o4i = dnnl_OIhw4i24o4i,
        OhwI4i24o4i = dnnl_OhwI4i24o4i,
        OIhw4i32o4i = dnnl_OIhw4i32o4i,
        OhwI4i32o4i = dnnl_OhwI4i32o4i,
        OIhw4i64o4i = dnnl_OIhw4i64o4i,
        OhwI4i64o4i = dnnl_OhwI4i64o4i,
        OIhw4i4o = dnnl_OIhw4i4o,
        OIhw4o4i = dnnl_OIhw4o4i,
        Oihw4o = dnnl_Oihw4o,
        OIhw8i16o2i = dnnl_OIhw8i16o2i,
        OhwI8i16o2i = dnnl_OhwI8i16o2i,
        OIhw8i32o2i = dnnl_OIhw8i32o2i,
        OhwI8i32o2i = dnnl_OhwI8i32o2i,
        OIhw8i64o2i = dnnl_OIhw8i64o2i,
        OhwI8i64o2i = dnnl_OhwI8i64o2i,
        OIhw8i8o = dnnl_OIhw8i8o,
        OhwI8i8o = dnnl_OhwI8i8o,
        OIhw8o16i2o = dnnl_OIhw8o16i2o,
        OIhw8o8i = dnnl_OIhw8o8i,
        OIhw8o4i = dnnl_OIhw8o4i,
        OIhw2i8o4i = dnnl_OIhw2i8o4i,
        IOdhw16o16i = dnnl_IOdhw16o16i,
        Odhwi16o = dnnl_Odhwi16o,
        OdhwI16o2i = dnnl_OdhwI16o2i,
        Idhwo16i = dnnl_Idhwo16i,
        IdhwO16i2o = dnnl_IdhwO16i2o,
        IdhwO16i4o = dnnl_IdhwO16i4o,
        Odhwi4o = dnnl_Odhwi4o,
        Odhwi8o = dnnl_Odhwi8o,
        OdhwI8o2i = dnnl_OdhwI8o2i,
        OdhwI8o4i = dnnl_OdhwI8o4i,
        OIdhw16i16o = dnnl_OIdhw16i16o,
        OdhwI16i16o = dnnl_OdhwI16i16o,
        OIdhw16i32o = dnnl_OIdhw16i32o,
        OdhwI16i32o = dnnl_OdhwI16i32o,
        OIdhw16i48o = dnnl_OIdhw16i48o,
        OdhwI16i48o = dnnl_OdhwI16i48o,
        OIdhw16i64o = dnnl_OIdhw16i64o,
        OdhwI16i64o = dnnl_OdhwI16i64o,
        OIdhw16o16i = dnnl_OIdhw16o16i,
        OIdhw16o16i2o = dnnl_OIdhw16o16i2o,
        Oidhw16o = dnnl_Oidhw16o,
        OIdhw4i4o = dnnl_OIdhw4i4o,
        OIdhw4o4i = dnnl_OIdhw4o4i,
        Oidhw4o = dnnl_Oidhw4o,
        OIdhw8i16o2i = dnnl_OIdhw8i16o2i,
        OdhwI8i16o2i = dnnl_OdhwI8i16o2i,
        OIdhw8i32o2i = dnnl_OIdhw8i32o2i,
        OdhwI8i32o2i = dnnl_OdhwI8i32o2i,
        OIdhw8i64o2i = dnnl_OIdhw8i64o2i,
        OdhwI8i64o2i = dnnl_OdhwI8i64o2i,
        OIdhw4i8o4i = dnnl_OIdhw4i8o4i,
        OdhwI4i8o4i = dnnl_OdhwI4i8o4i,
        OIdhw4i16o4i = dnnl_OIdhw4i16o4i,
        OdhwI4i16o4i = dnnl_OdhwI4i16o4i,
        OIdhw16i16o4i = dnnl_OIdhw16i16o4i,
        OIdhw16i32o4i = dnnl_OIdhw16i32o4i,
        OIdhw16i48o4i = dnnl_OIdhw16i48o4i,
        OIdhw16i64o4i = dnnl_OIdhw16i64o4i,
        OIdhw16i16o2i = dnnl_OIdhw16i16o2i,
        OIdhw16i32o2i = dnnl_OIdhw16i32o2i,
        OIdhw16i48o2i = dnnl_OIdhw16i48o2i,
        OIdhw16i64o2i = dnnl_OIdhw16i64o2i,
        OIdhw4i24o4i = dnnl_OIdhw4i24o4i,
        OdhwI4i24o4i = dnnl_OdhwI4i24o4i,
        OIdhw4i32o4i = dnnl_OIdhw4i32o4i,
        OdhwI4i32o4i = dnnl_OdhwI4i32o4i,
        OIdhw4i64o4i = dnnl_OIdhw4i64o4i,
        OdhwI4i64o4i = dnnl_OdhwI4i64o4i,
        OIdhw2i8o4i = dnnl_OIdhw2i8o4i,
        OIdhw8i8o = dnnl_OIdhw8i8o,
        OdhwI8i8o = dnnl_OdhwI8i8o,
        OIdhw8o8i = dnnl_OIdhw8o8i,
        OIdhw8o4i = dnnl_OIdhw8o4i,
        gIOw16o16i = dnnl_gIOw16o16i,
        gOIw16i16o = dnnl_gOIw16i16o,
        gOIw16o16i = dnnl_gOIw16o16i,
        gOiw16o = dnnl_gOiw16o,
        gOIw4i16o4i = dnnl_gOIw4i16o4i,
        gOIw2i8o4i = dnnl_gOIw2i8o4i,
        gOIw4i4o = dnnl_gOIw4i4o,
        gOIw4o4i = dnnl_gOIw4o4i,
        gOiw4o = dnnl_gOiw4o,
        gOIw8i16o2i = dnnl_gOIw8i16o2i,
        gOIw8i8o = dnnl_gOIw8i8o,
        gOIw8o16i2o = dnnl_gOIw8o16i2o,
        gOIw8o8i = dnnl_gOIw8o8i,
        gOIw8o4i = dnnl_gOIw8o4i,
        gOIw16i16o4i = dnnl_gOIw16i16o4i,
        gOIw16i16o2i = dnnl_gOIw16i16o2i,
        gOIw16o16i2o = dnnl_gOIw16o16i2o,
        gOwi16o = dnnl_gOwi16o,
        gOwI16o2i = dnnl_gOwI16o2i,
        gIwo16i = dnnl_gIwo16i,
        gIwO16i2o = dnnl_gIwO16i2o,
        gIwO16i4o = dnnl_gIwO16i4o,
        gOwi4o = dnnl_gOwi4o,
        gOwi8o = dnnl_gOwi8o,
        gOwI8o2i = dnnl_gOwI8o2i,
        gOwI8o4i = dnnl_gOwI8o4i,
        Goiw8g = dnnl_Goiw8g,
        Goiw16g = dnnl_Goiw16g,
        gIOhw16o16i = dnnl_gIOhw16o16i,
        gOhwi16o = dnnl_gOhwi16o,
        gOhwI16o2i = dnnl_gOhwI16o2i,
        gIhwo16i = dnnl_gIhwo16i,
        gIhwO16i2o = dnnl_gIhwO16i2o,
        gIhwO16i4o = dnnl_gIhwO16i4o,
        gOhwi4o = dnnl_gOhwi4o,
        gOhwi8o = dnnl_gOhwi8o,
        gOhwI8o2i = dnnl_gOhwI8o2i,
        gOhwI8o4i = dnnl_gOhwI8o4i,
        Goihw16g = dnnl_Goihw16g,
        gOIhw16i16o = dnnl_gOIhw16i16o,
        gOIhw16o16i = dnnl_gOIhw16o16i,
        gOihw16o = dnnl_gOihw16o,
        gOIhw4i16o4i = dnnl_gOIhw4i16o4i,
        gOIhw2i8o4i = dnnl_gOIhw2i8o4i,
        gOIhw4i4o = dnnl_gOIhw4i4o,
        gOIhw4o4i = dnnl_gOIhw4o4i,
        gOihw4o = dnnl_gOihw4o,
        Goihw8g = dnnl_Goihw8g,
        gOIhw8i16o2i = dnnl_gOIhw8i16o2i,
        gOIhw8i8o = dnnl_gOIhw8i8o,
        gOIhw8o16i2o = dnnl_gOIhw8o16i2o,
        OIw4o8i8o4i = dnnl_OIw4o8i8o4i,
        OIdhw4o8i8o4i = dnnl_OIdhw4o8i8o4i,
        OIhw4o8i8o4i = dnnl_OIhw4o8i8o4i,
        OIhw2o8i8o2i = dnnl_OIhw2o8i8o2i,
        gOIw4o8i8o4i = dnnl_gOIw4o8i8o4i,
        gOIdhw4o8i8o4i = dnnl_gOIdhw4o8i8o4i,
        gOIhw4o8i8o4i = dnnl_gOIhw4o8i8o4i,
        gOIhw2o8i8o2i = dnnl_gOIhw2o8i8o2i,
        OIhw16i16o4i = dnnl_OIhw16i16o4i,
        OIhw16i32o4i = dnnl_OIhw16i32o4i,
        OIhw16i48o4i = dnnl_OIhw16i48o4i,
        OIhw16i64o4i = dnnl_OIhw16i64o4i,
        OIhw16i16o2i = dnnl_OIhw16i16o2i,
        OIhw16i32o2i = dnnl_OIhw16i32o2i,
        OIhw16i48o2i = dnnl_OIhw16i48o2i,
        OIhw16i64o2i = dnnl_OIhw16i64o2i,
        OIhw16o16i2o = dnnl_OIhw16o16i2o,
        gOIhw16i16o4i = dnnl_gOIhw16i16o4i,
        gOIhw16i16o2i = dnnl_gOIhw16i16o2i,
        gOIhw16o16i2o = dnnl_gOIhw16o16i2o,
        gOIhw8o8i = dnnl_gOIhw8o8i,
        gOIhw8o4i = dnnl_gOIhw8o4i,
        gIOdhw16i16o = dnnl_gIOdhw16i16o,
        gIOdhw16o16i = dnnl_gIOdhw16o16i,
        gOdhwi16o = dnnl_gOdhwi16o,
        gOdhwI16o2i = dnnl_gOdhwI16o2i,
        gIdhwo16i = dnnl_gIdhwo16i,
        gIdhwO16i2o = dnnl_gIdhwO16i2o,
        gIdhwO16i4o = dnnl_gIdhwO16i4o,
        gOdhwi4o = dnnl_gOdhwi4o,
        gOdhwi8o = dnnl_gOdhwi8o,
        gOdhwI8o2i = dnnl_gOdhwI8o2i,
        gOdhwI8o4i = dnnl_gOdhwI8o4i,
        gOIdhw16i16o = dnnl_gOIdhw16i16o,
        gOIdhw16o16i = dnnl_gOIdhw16o16i,
        gOIdhw16o16i2o = dnnl_gOIdhw16o16i2o,
        gOidhw16o = dnnl_gOidhw16o,
        gOIdhw4i4o = dnnl_gOIdhw4i4o,
        gOIdhw4o4i = dnnl_gOIdhw4o4i,
        gOidhw4o = dnnl_gOidhw4o,
        gOIdhw8i16o2i = dnnl_gOIdhw8i16o2i,
        gOIdhw4i16o4i = dnnl_gOIdhw4i16o4i,
        gOIdhw16i16o4i = dnnl_gOIdhw16i16o4i,
        gOIdhw16i16o2i = dnnl_gOIdhw16i16o2i,
        gOIdhw2i8o4i = dnnl_gOIdhw2i8o4i,
        gOIdhw8i8o = dnnl_gOIdhw8i8o,
        gOIdhw8o8i = dnnl_gOIdhw8o8i,
        gOIdhw8o4i = dnnl_gOIdhw8o4i,
        gOIw2i4o2i = dnnl_gOIw2i4o2i,
        gOIhw2i4o2i = dnnl_gOIhw2i4o2i,
        gOIdhw2i4o2i = dnnl_gOIdhw2i4o2i,
        gOIw2o4i2o = dnnl_gOIw2o4i2o,
        gOIhw2o4i2o = dnnl_gOIhw2o4i2o,
        gOIdhw2o4i2o = dnnl_gOIdhw2o4i2o,
        gOIw4i8o2i = dnnl_gOIw4i8o2i,
        gOIhw4i8o2i = dnnl_gOIhw4i8o2i,
        gOIdhw4i8o2i = dnnl_gOIdhw4i8o2i,
        gOIw4o8i2o = dnnl_gOIw4o8i2o,
        gOIhw4o8i2o = dnnl_gOIhw4o8i2o,
        gOIdhw4o8i2o = dnnl_gOIdhw4o8i2o,

        ldOi16o = abDc16d,
        ldOi32o = abDc32d,
        ldOI32o4i = abDC32d4c,
        ldgOi16o = abdEc16e,
        ldgOi32o = abdEc32e,
        ldgOI32o2i = abdEC32e2c,
        ldgOI32o4i = abdEC32e4c,
        OwI16o4i = dnnl_OwI16o4i,
        OhwI16o4i = dnnl_OhwI16o4i,
        gOwI16o4i = dnnl_gOwI16o4i,
        gOhwI16o4i = dnnl_gOhwI16o4i,
        OdhwI16o4i = dnnl_OdhwI16o4i,
        gOdhwI16o4i = dnnl_gOdhwI16o4i,

        Owi32o = dnnl_Owi32o,
        OwI32o2i = dnnl_OwI32o2i,
        OwI32o4i = dnnl_OwI32o4i,
        Owi48o = dnnl_Owi48o,
        OwI48o2i = dnnl_OwI48o2i,
        OwI48o4i = dnnl_OwI48o4i,
        Owi64o = dnnl_Owi64o,
        OwI64o2i = dnnl_OwI64o2i,
        OwI64o4i = dnnl_OwI64o4i,
        Iwo32i = dnnl_Iwo32i,
        IwO32i2o = dnnl_IwO32i2o,
        IwO32i4o = dnnl_IwO32i4o,
        Iwo48i = dnnl_Iwo48i,
        IwO48i2o = dnnl_IwO48i2o,
        IwO48i4o = dnnl_IwO48i4o,
        Iwo64i = dnnl_Iwo64i,
        IwO64i2o = dnnl_IwO64i2o,
        IwO64i4o = dnnl_IwO64i4o,
        wIo2i = dnnl_wIo2i,
        wIo4i = dnnl_wIo4i,
        gOwi32o = dnnl_gOwi32o,
        gOwI32o2i = dnnl_gOwI32o2i,
        gOwI32o4i = dnnl_gOwI32o4i,
        gOwi48o = dnnl_gOwi48o,
        gOwI48o2i = dnnl_gOwI48o2i,
        gOwI48o4i = dnnl_gOwI48o4i,
        gOwi64o = dnnl_gOwi64o,
        gOwI64o2i = dnnl_gOwI64o2i,
        gOwI64o4i = dnnl_gOwI64o4i,
        gIwo32i = dnnl_gIwo32i,
        gIwO32i2o = dnnl_gIwO32i2o,
        gIwO32i4o = dnnl_gIwO32i4o,
        gIwo48i = dnnl_gIwo48i,
        gIwO48i2o = dnnl_gIwO48i2o,
        gIwO48i4o = dnnl_gIwO48i4o,
        gIwo64i = dnnl_gIwo64i,
        gIwO64i2o = dnnl_gIwO64i2o,
        gIwO64i4o = dnnl_gIwO64i4o,
        gwio = dnnl_gwio,
        gwIo2i = dnnl_gwIo2i,
        gwIo4i = dnnl_gwIo4i,
        OhwI32o = dnnl_OhwI32o,
        OhwI32o2i = dnnl_OhwI32o2i,
        OhwI32o4i = dnnl_OhwI32o4i,
        Ohwi48o = dnnl_Ohwi48o,
        OhwI48o2i = dnnl_OhwI48o2i,
        OhwI48o4i = dnnl_OhwI48o4i,
        Ohwi64o = dnnl_Ohwi64o,
        OhwI64o2i = dnnl_OhwI64o2i,
        OhwI64o4i = dnnl_OhwI64o4i,
        Ihwo32i = dnnl_Ihwo32i,
        IhwO32i2o = dnnl_IhwO32i2o,
        IhwO32i4o = dnnl_IhwO32i4o,
        Ihwo48i = dnnl_Ihwo48i,
        IhwO48i2o = dnnl_IhwO48i2o,
        IhwO48i4o = dnnl_IhwO48i4o,
        Ihwo64i = dnnl_Ihwo64i,
        IhwO64i2o = dnnl_IhwO64i2o,
        IhwO64i4o = dnnl_IhwO64i4o,
        hwIo2i = dnnl_hwIo2i,
        hwIo4i = dnnl_hwIo4i,
        gOhwI32o = dnnl_gOhwI32o,
        gOhwI32o2i = dnnl_gOhwI32o2i,
        gOhwI32o4i = dnnl_gOhwI32o4i,
        gOhwi48o = dnnl_gOhwi48o,
        gOhwI48o2i = dnnl_gOhwI48o2i,
        gOhwI48o4i = dnnl_gOhwI48o4i,
        gOhwi64o = dnnl_gOhwi64o,
        gOhwI64o2i = dnnl_gOhwI64o2i,
        gOhwI64o4i = dnnl_gOhwI64o4i,
        gIhwo32i = dnnl_gIhwo32i,
        gIhwO32i2o = dnnl_gIhwO32i2o,
        gIhwO32i4o = dnnl_gIhwO32i4o,
        gIhwo48i = dnnl_gIhwo48i,
        gIhwO48i2o = dnnl_gIhwO48i2o,
        gIhwO48i4o = dnnl_gIhwO48i4o,
        gIhwo64i = dnnl_gIhwo64i,
        gIhwO64i2o = dnnl_gIhwO64i2o,
        gIhwO64i4o = dnnl_gIhwO64i4o,
        ghwio = dnnl_ghwio,
        ghwIo2i = dnnl_ghwIo2i,
        ghwIo4i = dnnl_ghwIo4i,
        Odhwi32o = dnnl_Odhwi32o,
        OdhwI32o2i = dnnl_OdhwI32o2i,
        OdhwI32o4i = dnnl_OdhwI32o4i,
        Odhwi48o = dnnl_Odhwi48o,
        OdhwI48o2i = dnnl_OdhwI48o2i,
        OdhwI48o4i = dnnl_OdhwI48o4i,
        Odhwi64o = dnnl_Odhwi64o,
        OdhwI64o2i = dnnl_OdhwI64o2i,
        OdhwI64o4i = dnnl_OdhwI64o4i,
        Idhwo32i = dnnl_Idhwo32i,
        IdhwO32i2o = dnnl_IdhwO32i2o,
        IdhwO32i4o = dnnl_IdhwO32i4o,
        Idhwo48i = dnnl_Idhwo48i,
        IdhwO48i2o = dnnl_IdhwO48i2o,
        IdhwO48i4o = dnnl_IdhwO48i4o,
        Idhwo64i = dnnl_Idhwo64i,
        IdhwO64i2o = dnnl_IdhwO64i2o,
        IdhwO64i4o = dnnl_IdhwO64i4o,
        dhwIo2i = dnnl_dhwIo2i,
        dhwIo4i = dnnl_dhwIo4i,
        gOdhwi32o = dnnl_gOdhwi32o,
        gOdhwI32o2i = dnnl_gOdhwI32o2i,
        gOdhwI32o4i = dnnl_gOdhwI32o4i,
        gOdhwi48o = dnnl_gOdhwi48o,
        gOdhwI48o2i = dnnl_gOdhwI48o2i,
        gOdhwI48o4i = dnnl_gOdhwI48o4i,
        gOdhwi64o = dnnl_gOdhwi64o,
        gOdhwI64o2i = dnnl_gOdhwI64o2i,
        gOdhwI64o4i = dnnl_gOdhwI64o4i,
        gIdhwo32i = dnnl_gIdhwo32i,
        gIdhwO32i2o = dnnl_gIdhwO32i2o,
        gIdhwO32i4o = dnnl_gIdhwO32i4o,
        gIdhwo48i = dnnl_gIdhwo48i,
        gIdhwO48i2o = dnnl_gIdhwO48i2o,
        gIdhwO48i4o = dnnl_gIdhwO48i4o,
        gIdhwo64i = dnnl_gIdhwo64i,
        gIdhwO64i2o = dnnl_gIdhwO64i2o,
        gIdhwO64i4o = dnnl_gIdhwO64i4o,
        gdhwio = dnnl_gdhwio,
        gdhwIo2i = dnnl_gdhwIo2i,
        gdhwIo4i = dnnl_gdhwIo4i,
        ldIo32i = dnnl_ldIo32i,
        ldgIo16i = dnnl_ldgIo16i,
        ldgIo32i = dnnl_ldgIo32i,
        ldgIO32i2o = dnnl_ldgIO32i2o,
        nCdhw32c = dnnl_nCdhw32c,
        nChw32c = dnnl_nChw32c,
        nCw32c = dnnl_nCw32c,
        NCw32n16c = dnnl_NCw32n16c,
        NChw32n16c = dnnl_NChw32n16c,
        NCdhw32n16c = dnnl_NCdhw32n16c,
        NCw32n32c = dnnl_NCw32n32c,
        OI16i16o4i = dnnl_OI16i16o4i,
        IOw8o16i2o = dnnl_IOw8o16i2o,
        IOhw8o16i2o = dnnl_IOhw8o16i2o,
        Owhi16o = dnnl_Owhi16o,
        OIdhw8o16i2o = dnnl_OIdhw8o16i2o,
        IOdhw8o16i2o = dnnl_IOdhw8o16i2o,
        Goiw4g = dnnl_Goiw4g,
        gIOw8o16i2o = dnnl_gIOw8o16i2o,
        Goiw32g = dnnl_Goiw32g,
        Goihw4g = dnnl_Goihw4g,
        gIOhw8o16i2o = dnnl_gIOhw8o16i2o,
        Goihw32g = dnnl_Goihw32g,
        gOwhi16o = dnnl_gOwhi16o,
        IOw4i8o8i4o = dnnl_IOw4i8o8i4o,
        IOhw4i8o8i4o = dnnl_IOhw4i8o8i4o,
        IOdhw4i8o8i4o = dnnl_IOdhw4i8o8i4o,
        gIOw4i8o8i4o = dnnl_gIOw4i8o8i4o,
        gIOhw4i8o8i4o = dnnl_gIOhw4i8o8i4o,
        gIOdhw4i8o8i4o = dnnl_gIOdhw4i8o8i4o,
        gOIdhw8o16i2o = dnnl_gOIdhw8o16i2o,
        gIOdhw8o16i2o = dnnl_gIOdhw8o16i2o,
        Goidhw32g = dnnl_Goidhw32g,
        OI16i32o4i = dnnl_OI16i32o4i,
        OI16i48o4i = dnnl_OI16i48o4i,
        OI16i64o4i = dnnl_OI16i64o4i,
        OI16i16o2i = dnnl_OI16i16o2i,
        OI16i32o2i = dnnl_OI16i32o2i,
        OI16i48o2i = dnnl_OI16i48o2i,
        OI16i64o2i = dnnl_OI16i64o2i,
        aBdeC16c16b4c = dnnl_aBdeC16c16b4c,
        AcB16b16a2b = dnnl_AcB16b16a2b,
        aBdC16c16b2c = dnnl_aBdC16c16b2c,
        AcB16b16a4b = dnnl_AcB16b16a4b,
        aBdC16c16b4c = dnnl_aBdC16c16b4c,
        AcdB16b16a2b = dnnl_AcdB16b16a2b,
        aBdefC16c16b4c = dnnl_aBdefC16c16b4c,
        AcdeB16b16a4b = dnnl_AcdeB16b16a4b,
        AcB16b32a2b = dnnl_AcB16b32a2b,
        AcB16b32a4b = dnnl_AcB16b32a4b,
        AcB16b48a2b = dnnl_AcB16b48a2b,
        AcB16b48a4b = dnnl_AcB16b48a4b,
        AcB16b64a2b = dnnl_AcB16b64a2b,
        AcB16b64a4b = dnnl_AcB16b64a4b,
        aBdC16c32b2c = dnnl_aBdC16c32b2c,
        aBdC16c32b4c = dnnl_aBdC16c32b4c,
        aBdC16c48b2c = dnnl_aBdC16c48b2c,
        aBdC16c48b4c = dnnl_aBdC16c48b4c,
        aBdC16c64b2c = dnnl_aBdC16c64b2c,
        aBdC16c64b4c = dnnl_aBdC16c64b4c,
        AcdB16b32a2b = dnnl_AcdB16b32a2b,
        AcdB16b32a4b = dnnl_AcdB16b32a4b,
        AcdB16b48a2b = dnnl_AcdB16b48a2b,
        AcdB16b48a4b = dnnl_AcdB16b48a4b,
        AcdB16b64a2b = dnnl_AcdB16b64a2b,
        AcdB16b64a4b = dnnl_AcdB16b64a4b,
        aBdeC16c32b2c = dnnl_aBdeC16c32b2c,
        aBdeC16c32b4c = dnnl_aBdeC16c32b4c,
        aBdeC16c48b2c = dnnl_aBdeC16c48b2c,
        aBdeC16c48b4c = dnnl_aBdeC16c48b4c,
        aBdeC16c64b2c = dnnl_aBdeC16c64b2c,
        aBdeC16c64b4c = dnnl_aBdeC16c64b4c,
        AcdeB16b32a2b = dnnl_AcdeB16b32a2b,
        AcdeB16b32a4b = dnnl_AcdeB16b32a4b,
        AcdeB16b48a2b = dnnl_AcdeB16b48a2b,
        AcdeB16b48a4b = dnnl_AcdeB16b48a4b,
        AcdeB16b64a2b = dnnl_AcdeB16b64a2b,
        AcdeB16b64a4b = dnnl_AcdeB16b64a4b,
        aBdefC16c32b2c = dnnl_aBdefC16c32b2c,
        aBdefC16c32b4c = dnnl_aBdefC16c32b4c,
        aBdefC16c48b2c = dnnl_aBdefC16c48b2c,
        aBdefC16c48b4c = dnnl_aBdefC16c48b4c,
        aBdefC16c64b2c = dnnl_aBdefC16c64b2c,
        aBdefC16c64b4c = dnnl_aBdefC16c64b4c,
        OwI16i16o2i = dnnl_OwI16i16o2i,
        gOwI16i16o2i = dnnl_gOwI16i16o2i,
        OhwI16i16o2i = dnnl_OhwI16i16o2i,
        gOhwI16i16o2i = dnnl_gOhwI16i16o2i,
        OdhwI16i16o2i = dnnl_OdhwI16i16o2i,
        gOdhwI16i16o2i = dnnl_gOdhwI16i16o2i,
        OwI16i16o4i = dnnl_OwI16i16o4i,
        gOwI16i16o4i = dnnl_gOwI16i16o4i,
        OhwI16i16o4i = dnnl_OhwI16i16o4i,
        gOhwI16i16o4i = dnnl_gOhwI16i16o4i,
        OdhwI16i16o4i = dnnl_OdhwI16i16o4i,
        gOdhwI16i16o4i = dnnl_gOdhwI16i16o4i,
        OwI16i32o2i = dnnl_OwI16i32o2i,
        OwI16i32o4i = dnnl_OwI16i32o4i,
        OwI16i48o2i = dnnl_OwI16i48o2i,
        OwI16i48o4i = dnnl_OwI16i48o4i,
        OwI16i64o2i = dnnl_OwI16i64o2i,
        OwI16i64o4i = dnnl_OwI16i64o4i,
        gOwI16i32o2i = dnnl_gOwI16i32o2i,
        gOwI16i32o4i = dnnl_gOwI16i32o4i,
        gOwI16i48o2i = dnnl_gOwI16i48o2i,
        gOwI16i48o4i = dnnl_gOwI16i48o4i,
        gOwI16i64o2i = dnnl_gOwI16i64o2i,
        gOwI16i64o4i = dnnl_gOwI16i64o4i,
        OhwI16i32o2i = dnnl_OhwI16i32o2i,
        OhwI16i32o4i = dnnl_OhwI16i32o4i,
        OhwI16i48o2i = dnnl_OhwI16i48o2i,
        OhwI16i48o4i = dnnl_OhwI16i48o4i,
        OhwI16i64o2i = dnnl_OhwI16i64o2i,
        OhwI16i64o4i = dnnl_OhwI16i64o4i,
        gOhwI16i32o2i = dnnl_gOhwI16i32o2i,
        gOhwI16i32o4i = dnnl_gOhwI16i32o4i,
        gOhwI16i48o2i = dnnl_gOhwI16i48o2i,
        gOhwI16i48o4i = dnnl_gOhwI16i48o4i,
        gOhwI16i64o2i = dnnl_gOhwI16i64o2i,
        gOhwI16i64o4i = dnnl_gOhwI16i64o4i,
        OdhwI16i32o2i = dnnl_OdhwI16i32o2i,
        OdhwI16i32o4i = dnnl_OdhwI16i32o4i,
        OdhwI16i48o2i = dnnl_OdhwI16i48o2i,
        OdhwI16i48o4i = dnnl_OdhwI16i48o4i,
        OdhwI16i64o2i = dnnl_OdhwI16i64o2i,
        OdhwI16i64o4i = dnnl_OdhwI16i64o4i,
        IdhwO16o32i2o = dnnl_IdhwO16o32i2o,
        IdhwO16o32i4o = dnnl_IdhwO16o32i4o,
        IdhwO16o48i2o = dnnl_IdhwO16o48i2o,
        IdhwO16o48i4o = dnnl_IdhwO16o48i4o,
        IdhwO16o64i2o = dnnl_IdhwO16o64i2o,
        IdhwO16o64i4o = dnnl_IdhwO16o64i4o,
        gOdhwI16i32o2i = dnnl_gOdhwI16i32o2i,
        gOdhwI16i32o4i = dnnl_gOdhwI16i32o4i,
        gOdhwI16i48o2i = dnnl_gOdhwI16i48o2i,
        gOdhwI16i48o4i = dnnl_gOdhwI16i48o4i,
        gOdhwI16i64o2i = dnnl_gOdhwI16i64o2i,
        gOdhwI16i64o4i = dnnl_gOdhwI16i64o4i,
        gIdhwO16o32i2o = dnnl_gIdhwO16o32i2o,
        gIdhwO16o32i4o = dnnl_gIdhwO16o32i4o,
        gIdhwO16o48i2o = dnnl_gIdhwO16o48i2o,
        gIdhwO16o48i4o = dnnl_gIdhwO16o48i4o,
        gIdhwO16o64i2o = dnnl_gIdhwO16o64i2o,
        gIdhwO16o64i4o = dnnl_gIdhwO16o64i4o,
        IwO16o16i2o = dnnl_IwO16o16i2o,
        IwO16o16i4o = dnnl_IwO16o16i4o,
        IhwO16o16i2o = dnnl_IhwO16o16i2o,
        IhwO16o16i4o = dnnl_IhwO16o16i4o,
        IdhwO16o16i2o = dnnl_IdhwO16o16i2o,
        IdhwO16o16i4o = dnnl_IdhwO16o16i4o,
        gIwO16o16i2o = dnnl_gIwO16o16i2o,
        gIwO16o16i4o = dnnl_gIwO16o16i4o,
        gIhwO16o16i2o = dnnl_gIhwO16o16i2o,
        gIhwO16o16i4o = dnnl_gIhwO16o16i4o,
        gIdhwO16o16i2o = dnnl_gIdhwO16o16i2o,
        gIdhwO16o16i4o = dnnl_gIdhwO16o16i4o,
        IwO16o32i2o = dnnl_IwO16o32i2o,
        IwO16o32i4o = dnnl_IwO16o32i4o,
        IwO16o48i2o = dnnl_IwO16o48i2o,
        IwO16o48i4o = dnnl_IwO16o48i4o,
        IwO16o64i2o = dnnl_IwO16o64i2o,
        IwO16o64i4o = dnnl_IwO16o64i4o,
        gIwO16o32i2o = dnnl_gIwO16o32i2o,
        gIwO16o32i4o = dnnl_gIwO16o32i4o,
        gIwO16o48i2o = dnnl_gIwO16o48i2o,
        gIwO16o48i4o = dnnl_gIwO16o48i4o,
        gIwO16o64i2o = dnnl_gIwO16o64i2o,
        gIwO16o64i4o = dnnl_gIwO16o64i4o,
        IhwO16o32i2o = dnnl_IhwO16o32i2o,
        IhwO16o32i4o = dnnl_IhwO16o32i4o,
        IhwO16o48i2o = dnnl_IhwO16o48i2o,
        IhwO16o48i4o = dnnl_IhwO16o48i4o,
        IhwO16o64i2o = dnnl_IhwO16o64i2o,
        IhwO16o64i4o = dnnl_IhwO16o64i4o,
        gIhwO16o32i2o = dnnl_gIhwO16o32i2o,
        gIhwO16o32i4o = dnnl_gIhwO16o32i4o,
        gIhwO16o48i2o = dnnl_gIhwO16o48i2o,
        gIhwO16o48i4o = dnnl_gIhwO16o48i4o,
        gIhwO16o64i2o = dnnl_gIhwO16o64i2o,
        gIhwO16o64i4o = dnnl_gIhwO16o64i4o,
        aBdeC16c16b2c = dnnl_aBdeC16c16b2c,
        aBdefC16c16b2c = dnnl_aBdefC16c16b2c,
        AcdB16b16a4b = dnnl_AcdB16b16a4b,
        AcdeB16b16a2b = dnnl_AcdeB16b16a2b,
        hwioG16g = dnnl_hwioG16g,
        hwioG8g = dnnl_hwioG8g,
        dhwioG16g = dnnl_dhwioG16g,
        dhwioG8g = dnnl_dhwioG8g,
        ABc4a2b = dnnl_ABc4a2b,
        ABc8a2b = dnnl_ABc8a2b,
        ABcd4a2b = dnnl_ABcd4a2b,
        ABcde4a2b = dnnl_ABcde4a2b,
        ABcde8a2b = dnnl_ABcde8a2b,
        ABcd4a8b8a2b = dnnl_ABcd4a8b8a2b,
        NCdhw40n32c = dnnl_NCdhw40n32c,
        NChw40n32c = dnnl_NChw40n32c,
        NCw40n32c = dnnl_NCw40n32c,
        OIdhw4o8i8o2i = dnnl_OIdhw4o8i8o2i,
        OIhw4o8i8o2i = dnnl_OIhw4o8i8o2i,
        OIw4o8i8o2i = dnnl_OIw4o8i8o2i,
        gOIdhw4o8i8o2i = dnnl_gOIdhw4o8i8o2i,
        gOIhw4o8i8o2i = dnnl_gOIhw4o8i8o2i,
        gOIw4o8i8o2i = dnnl_gOIw4o8i8o2i,
        IOdhw4i8o8i2o = dnnl_IOdhw4i8o8i2o,
        IOhw4i8o8i2o = dnnl_IOhw4i8o8i2o,
        IOw4i8o8i2o = dnnl_IOw4i8o8i2o,
        gIOdhw4i8o8i2o = dnnl_gIOdhw4i8o8i2o,
        gIOhw4i8o8i2o = dnnl_gIOhw4i8o8i2o,
        gIOw4i8o8i2o = dnnl_gIOw4i8o8i2o,
        aBCd8b2c = dnnl_aBCd8b2c,
        ABcde40a16b = dnnl_ABcde40a16b,
        ABcde40a32b = dnnl_ABcde40a32b,
        aBCde8b2c = dnnl_aBCde8b2c,
        ABcde4a8b8a2b = dnnl_ABcde4a8b8a2b,
        ABc4a8b8a2b = dnnl_ABc4a8b8a2b,
        aBCdef4b8c8b2c = dnnl_aBCdef4b8c8b2c,
        aBCde4b8c8b2c = dnnl_aBCde4b8c8b2c,
        aBCd4b8c8b2c = dnnl_aBCd4b8c8b2c,
        BAcde4b8a8b2a = dnnl_BAcde4b8a8b2a,
        BAcd4b8a8b2a = dnnl_BAcd4b8a8b2a,
        BAc4b8a8b2a = dnnl_BAc4b8a8b2a,
        aCBdef4c8b8c2b = dnnl_aCBdef4c8b8c2b,
        aCBde4c8b8c2b = dnnl_aCBde4c8b8c2b,
        aCBd4c8b8c2b = dnnl_aCBd4c8b8c2b,
        aBCdef8b2c = dnnl_aBCdef8b2c,
        AB32a16b = dnnl_AB32a16b,
        AB32a32b = dnnl_AB32a32b,
        BA4b8a8b2a = dnnl_BA4b8a8b2a,
        BA4b8a8b4a = dnnl_BA4b8a8b4a,
        aBC32b16c = dnnl_aBC32b16c,
        aBC32b32c = dnnl_aBC32b32c,
        aCB4c8b8c2b = dnnl_aCB4c8b8c2b,
        aCB4c8b8c4b = dnnl_aCB4c8b8c4b,
        ABc2b8a16b4a = dnnl_ABc2b8a16b4a,
        ABcd2b8a16b4a = dnnl_ABcd2b8a16b4a,
        ABcde2b8a16b4a = dnnl_ABcde2b8a16b4a,
        ABc2a8b16a4b = dnnl_ABc2a8b16a4b,
        ABc2a8b16a2b = dnnl_ABc2a8b16a2b,
        ABc2b32a8b = dnnl_ABc2b32a8b,
        ABcd2a8b16a4b = dnnl_ABcd2a8b16a4b,
        ABcd2a8b16a2b = dnnl_ABcd2a8b16a2b,
        aCBd2c8b16c2b = dnnl_aCBd2c8b16c2b,
        ABcd2b32a8b = dnnl_ABcd2b32a8b,
        aBCd2c8b16c2b = dnnl_aBCd2c8b16c2b,
        ABcde2a8b16a4b = dnnl_ABcde2a8b16a4b,
        ABcde2a8b16a2b = dnnl_ABcde2a8b16a2b,
        aCBde2c8b16c2b = dnnl_aCBde2c8b16c2b,
        ABcde2b32a8b = dnnl_ABcde2b32a8b,
        aBC2b8c16b2c = dnnl_aBC2b8c16b2c,
        aBCd2b8c16b2c = dnnl_aBCd2b8c16b2c,
        aBCde2b8c16b2c = dnnl_aBCde2b8c16b2c,
        aBCdef2b8c16b2c = dnnl_aBCdef2b8c16b2c,
        BAcde2b8a16b4a = dnnl_BAcde2b8a16b4a,
        BAcd2b8a16b4a = dnnl_BAcd2b8a16b4a,
        BAc2b8a16b4a = dnnl_BAc2b8a16b4a,
        BAcde2b8a16b2a = dnnl_BAcde2b8a16b2a,
        BAcd2b8a16b2a = dnnl_BAcd2b8a16b2a,
        BAc2b8a16b2a = dnnl_BAc2b8a16b2a,
        aBCde2c8b16c2b = dnnl_aBCde2c8b16c2b,
        aBCdef2c8b16c2b = dnnl_aBCdef2c8b16c2b,
        aCBdef2c8b16c2b = dnnl_aCBdef2c8b16c2b,
        aBCd2b8c16b4c = dnnl_aBCd2b8c16b4c,
        aBCde2b8c16b4c = dnnl_aBCde2b8c16b4c,
        NCdhw40n16c = dnnl_NCdhw40n16c,
        NCw40n16c = dnnl_NCw40n16c,
        NChw40n16c = dnnl_NChw40n16c,
        NCw2c32n8c = dnnl_NCw2c32n8c,
        NChw2c32n8c = dnnl_NChw2c32n8c,
        NCdhw2c32n8c = dnnl_NCdhw2c32n8c,
        OIw2i8o16i4o = dnnl_OIw2i8o16i4o,
        OIhw2i8o16i4o = dnnl_OIhw2i8o16i4o,
        OIdhw2i8o16i4o = dnnl_OIdhw2i8o16i4o,
        OIw2o8i16o4i = dnnl_OIw2o8i16o4i,
        OIw2o8i16o2i = dnnl_OIw2o8i16o2i,
        IOw2i8o16i4o = dnnl_IOw2i8o16i4o,
        IOw2i8o16i2o = dnnl_IOw2i8o16i2o,
        OIhw2o8i16o4i = dnnl_OIhw2o8i16o4i,
        OIhw2o8i16o2i = dnnl_OIhw2o8i16o2i,
        IOhw2i8o16i4o = dnnl_IOhw2i8o16i4o,
        IOhw2i8o16i2o = dnnl_IOhw2i8o16i2o,
        OIdhw2o8i16o4i = dnnl_OIdhw2o8i16o4i,
        OIdhw2o8i16o2i = dnnl_OIdhw2o8i16o2i,
        IOdhw2i8o16i4o = dnnl_IOdhw2i8o16i4o,
        IOdhw2i8o16i2o = dnnl_IOdhw2i8o16i2o,
        gOIw2o8i16o2i = dnnl_gOIw2o8i16o2i,
        gIOw2i8o16i2o = dnnl_gIOw2i8o16i2o,
        gIOhw2i8o16i2o = dnnl_gIOhw2i8o16i2o,
        gIOdhw2i8o16i2o = dnnl_gIOdhw2i8o16i2o,
        gOIhw2o8i16o2i = dnnl_gOIhw2o8i16o2i,
        gOIdhw2o8i16o2i = dnnl_gOIdhw2o8i16o2i,
        gOIw2o8i16o4i = dnnl_gOIw2o8i16o4i,
        gOIhw2o8i16o4i = dnnl_gOIhw2o8i16o4i,
        BA4b8a16b2a = dnnl_BA4b8a16b2a,
        BA4b8a16b4a = dnnl_BA4b8a16b4a,
        aCB4c8b16c2b = dnnl_aCB4c8b16c2b,
        aCB4c8b16c4b = dnnl_aCB4c8b16c4b,
        aCB16c2b = dnnl_aCB16c2b,
        aCB16c4b = dnnl_aCB16c4b,
        BA16b2a = dnnl_BA16b2a,
        BA16b4a = dnnl_BA16b4a,
        BA4b4a = dnnl_BA4b4a,
        BA8b4a = dnnl_BA8b4a,
        aBC16b16c = dnnl_aBC16b16c,
        aBC16b32c = dnnl_aBC16b32c,
        AB16a16b = dnnl_AB16a16b,
        AB16a32b = dnnl_AB16a32b,
        ABcde16a16b2a = dnnl_ABcde16a16b2a,
        aBCdef16b16c2b = dnnl_aBCdef16b16c2b,
        Acedb16a = dnnl_Acedb16a,
        aBdfec16b = dnnl_aBdfec16b,
        Odwhi16o = dnnl_Odwhi16o,
        gOdwhi16o = dnnl_gOdwhi16o,
        abdEC64e2c = dnnl_abdEC64e2c,
        abdEC64e4c = dnnl_abdEC64e4c,
        ldgOI64o2i = abdEC64e2c,
        ldgOI64o4i = abdEC64e4c,
        abCd4c = dnnl_abCd4c,
        abCde4c = dnnl_abCde4c,
        abCdef4c = dnnl_abCdef4c,
        abCde32c = dnnl_abCde32c,
        abCdef32c = dnnl_abCdef32c,
        aCdefB16b32c2b = dnnl_aCdefB16b32c2b,
        aCdefB16b32c4b = dnnl_aCdefB16b32c4b,
        aCdefB16b48c2b = dnnl_aCdefB16b48c2b,
        aCdefB16b48c4b = dnnl_aCdefB16b48c4b,
        aCdefB16b64c2b = dnnl_aCdefB16b64c2b,
        aCdefB16b64c4b = dnnl_aCdefB16b64c4b,
        BcdeA16a32b2a = dnnl_BcdeA16a32b2a,
        BcdeA16a32b4a = dnnl_BcdeA16a32b4a,
        BcdeA16a48b2a = dnnl_BcdeA16a48b2a,
        BcdeA16a48b4a = dnnl_BcdeA16a48b4a,
        BcdeA16a64b2a = dnnl_BcdeA16a64b2a,
        BcdeA16a64b4a = dnnl_BcdeA16a64b4a,
        aCdefb32c = dnnl_aCdefb32c,
        aCdefB32c2b = dnnl_aCdefB32c2b,
        aCdefB32c4b = dnnl_aCdefB32c4b,
        aCdefb48c = dnnl_aCdefb48c,
        aCdefB48c2b = dnnl_aCdefB48c2b,
        aCdefB48c4b = dnnl_aCdefB48c4b,
        aCdefb64c = dnnl_aCdefb64c,
        aCdefB64c2b = dnnl_aCdefB64c2b,
        aCdefB64c4b = dnnl_aCdefB64c4b,
        Bcdea32b = dnnl_Bcdea32b,
        BcdeA32b2a = dnnl_BcdeA32b2a,
        BcdeA32b4a = dnnl_BcdeA32b4a,
        Bcdea48b = dnnl_Bcdea48b,
        BcdeA48b2a = dnnl_BcdeA48b2a,
        BcdeA48b4a = dnnl_BcdeA48b4a,
        Bcdea64b = dnnl_Bcdea64b,
        BcdeA64b2a = dnnl_BcdeA64b2a,
        BcdeA64b4a = dnnl_BcdeA64b4a,
        Bca32b = dnnl_Bca32b,
        BcA32b2a = dnnl_BcA32b2a,
        BcA32b4a = dnnl_BcA32b4a,
        Bca48b = dnnl_Bca48b,
        BcA48b2a = dnnl_BcA48b2a,
        BcA48b4a = dnnl_BcA48b4a,
        Bca64b = dnnl_Bca64b,
        BcA64b2a = dnnl_BcA64b2a,
        BcA64b4a = dnnl_BcA64b4a,
        aCdb32c = dnnl_aCdb32c,
        aCdB32c2b = dnnl_aCdB32c2b,
        aCdB32c4b = dnnl_aCdB32c4b,
        aCdb48c = dnnl_aCdb48c,
        aCdB48c2b = dnnl_aCdB48c2b,
        aCdB48c4b = dnnl_aCdB48c4b,
        aCdb64c = dnnl_aCdb64c,
        aCdB64c2b = dnnl_aCdB64c2b,
        aCdB64c4b = dnnl_aCdB64c4b,
        BcA16a16b2a = dnnl_BcA16a16b2a,
        BcA16a16b4a = dnnl_BcA16a16b4a,
        BcdA16a16b2a = dnnl_BcdA16a16b2a,
        BcdA16a16b4a = dnnl_BcdA16a16b4a,
        BcdeA16a16b2a = dnnl_BcdeA16a16b2a,
        BcdeA16a16b4a = dnnl_BcdeA16a16b4a,
        aCdB16b16c2b = dnnl_aCdB16b16c2b,
        aCdB16b16c4b = dnnl_aCdB16b16c4b,
        aCdeB16b16c2b = dnnl_aCdeB16b16c2b,
        aCdeB16b16c4b = dnnl_aCdeB16b16c4b,
        aCdefB16b16c2b = dnnl_aCdefB16b16c2b,
        aCdefB16b16c4b = dnnl_aCdefB16b16c4b,
        BcA16a32b2a = dnnl_BcA16a32b2a,
        BcA16a32b4a = dnnl_BcA16a32b4a,
        BcA16a48b2a = dnnl_BcA16a48b2a,
        BcA16a48b4a = dnnl_BcA16a48b4a,
        BcA16a64b2a = dnnl_BcA16a64b2a,
        BcA16a64b4a = dnnl_BcA16a64b4a,
        aCdB16b32c2b = dnnl_aCdB16b32c2b,
        aCdB16b32c4b = dnnl_aCdB16b32c4b,
        aCdB16b48c2b = dnnl_aCdB16b48c2b,
        aCdB16b48c4b = dnnl_aCdB16b48c4b,
        aCdB16b64c2b = dnnl_aCdB16b64c2b,
        aCdB16b64c4b = dnnl_aCdB16b64c4b,
        BcdA16a32b2a = dnnl_BcdA16a32b2a,
        BcdA16a32b4a = dnnl_BcdA16a32b4a,
        BcdA16a48b2a = dnnl_BcdA16a48b2a,
        BcdA16a48b4a = dnnl_BcdA16a48b4a,
        BcdA16a64b2a = dnnl_BcdA16a64b2a,
        BcdA16a64b4a = dnnl_BcdA16a64b4a,
        aCdeB16b32c2b = dnnl_aCdeB16b32c2b,
        aCdeB16b32c4b = dnnl_aCdeB16b32c4b,
        aCdeB16b48c2b = dnnl_aCdeB16b48c2b,
        aCdeB16b48c4b = dnnl_aCdeB16b48c4b,
        aCdeB16b64c2b = dnnl_aCdeB16b64c2b,
        aCdeB16b64c4b = dnnl_aCdeB16b64c4b,
        Bca16b = dnnl_Bca16b,
        BcA16b2a = dnnl_BcA16b2a,
        BcA16b4a = dnnl_BcA16b4a,
        Bcda16b = dnnl_Bcda16b,
        BcdA16b2a = dnnl_BcdA16b2a,
        BcdA16b4a = dnnl_BcdA16b4a,
        Bcdea16b = dnnl_Bcdea16b,
        BcdeA16b2a = dnnl_BcdeA16b2a,
        BcdeA16b4a = dnnl_BcdeA16b4a,
        aCdb16c = dnnl_aCdb16c,
        aCdB16c2b = dnnl_aCdB16c2b,
        aCdB16c4b = dnnl_aCdB16c4b,
        aCdeb16c = dnnl_aCdeb16c,
        aCdeB16c2b = dnnl_aCdeB16c2b,
        aCdeB16c4b = dnnl_aCdeB16c4b,
        aCdefb16c = dnnl_aCdefb16c,
        aCdefB16c2b = dnnl_aCdefB16c2b,
        aCdefB16c4b = dnnl_aCdefB16c4b,
        Bcda32b = dnnl_Bcda32b,
        BcdA32b2a = dnnl_BcdA32b2a,
        BcdA32b4a = dnnl_BcdA32b4a,
        Bcda48b = dnnl_Bcda48b,
        BcdA48b2a = dnnl_BcdA48b2a,
        BcdA48b4a = dnnl_BcdA48b4a,
        Bcda64b = dnnl_Bcda64b,
        BcdA64b2a = dnnl_BcdA64b2a,
        BcdA64b4a = dnnl_BcdA64b4a,
        aCdeb32c = dnnl_aCdeb32c,
        aCdeB32c2b = dnnl_aCdeB32c2b,
        aCdeB32c4b = dnnl_aCdeB32c4b,
        aCdeb48c = dnnl_aCdeb48c,
        aCdeB48c2b = dnnl_aCdeB48c2b,
        aCdeB48c4b = dnnl_aCdeB48c4b,
        aCdeb64c = dnnl_aCdeb64c,
        aCdeB64c2b = dnnl_aCdeB64c2b,
        aCdeB64c4b = dnnl_aCdeB64c4b,
        NChw16n32c = dnnl_NChw16n32c,
        goIw4i = dnnl_goIw4i,
        goIw32i = dnnl_goIw32i,
        goIhw4i = dnnl_goIhw4i,
        goIhw32i = dnnl_goIhw32i,
        goIdhw4i = dnnl_goIdhw4i,
        goIdhw32i = dnnl_goIdhw32i,
        cab = dnnl_cab,
        cdab = dnnl_cdab,
        cdeab = dnnl_cdeab,
        woi = dnnl_woi,
        hwoi = dnnl_hwoi,
        dhwoi = dnnl_dhwoi,
        Owi24o = dnnl_Owi24o,
        Ohwi24o = dnnl_Ohwi24o,
        Odhwi24o = dnnl_Odhwi24o,
        gOwi24o = dnnl_gOwi24o,
        gOhwi24o = dnnl_gOhwi24o,
        gOdhwi24o = dnnl_gOdhwi24o,
        OwI24o2i = dnnl_OwI24o2i,
        OhwI24o2i = dnnl_OhwI24o2i,
        OdhwI24o2i = dnnl_OdhwI24o2i,
        gOwI24o2i = dnnl_gOwI24o2i,
        gOhwI24o2i = dnnl_gOhwI24o2i,
        gOdhwI24o2i = dnnl_gOdhwI24o2i,
        OwI24o4i = dnnl_OwI24o4i,
        OhwI24o4i = dnnl_OhwI24o4i,
        OdhwI24o4i = dnnl_OdhwI24o4i,
        gOwI24o4i = dnnl_gOwI24o4i,
        gOhwI24o4i = dnnl_gOhwI24o4i,
        gOdhwI24o4i = dnnl_gOdhwI24o4i,
        OI8i32o = dnnl_OI8i32o,
        OIw8i32o = dnnl_OIw8i32o,
        OwI8i32o = dnnl_OwI8i32o,
        OIhw8i32o = dnnl_OIhw8i32o,
        OhwI8i32o = dnnl_OhwI8i32o,
        OIdhw8i32o = dnnl_OIdhw8i32o,
        OdhwI8i32o = dnnl_OdhwI8i32o,
        OI8i24o = dnnl_OI8i24o,
        OIw8i24o = dnnl_OIw8i24o,
        OwI8i24o = dnnl_OwI8i24o,
        OIhw8i24o = dnnl_OIhw8i24o,
        OhwI8i24o = dnnl_OhwI8i24o,
        OIdhw8i24o = dnnl_OIdhw8i24o,
        OdhwI8i24o = dnnl_OdhwI8i24o,
        OI8i16o = dnnl_OI8i16o,
        OIw8i16o = dnnl_OIw8i16o,
        OwI8i16o = dnnl_OwI8i16o,
        OIhw8i16o = dnnl_OIhw8i16o,
        OhwI8i16o = dnnl_OhwI8i16o,
        OIdhw8i16o = dnnl_OIdhw8i16o,
        OdhwI8i16o = dnnl_OdhwI8i16o,
        OI8i8o = dnnl_OI8i8o,
        AB4b8a4b = dnnl_AB4b8a4b,
        AB4b24a4b = dnnl_AB4b24a4b,
        ABc4b8a4b = dnnl_ABc4b8a4b,
        AcB4b8a4b = dnnl_AcB4b8a4b,
        ABc4b24a4b = dnnl_ABc4b24a4b,
        AcB4b24a4b = dnnl_AcB4b24a4b,
        ABcd4b8a4b = dnnl_ABcd4b8a4b,
        AcdB4b8a4b = dnnl_AcdB4b8a4b,
        ABcd4b24a4b = dnnl_ABcd4b24a4b,
        AcdB4b24a4b = dnnl_AcdB4b24a4b,
        ABcde4b8a4b = dnnl_ABcde4b8a4b,
        AcdeB4b8a4b = dnnl_AcdeB4b8a4b,
        ABcde4b24a4b = dnnl_ABcde4b24a4b,
        AcdeB4b24a4b = dnnl_AcdeB4b24a4b,
        Bca8b = dnnl_Bca8b,
        BcA8b2a = dnnl_BcA8b2a,
        Bcda8b = dnnl_Bcda8b,
        BcdA8b2a = dnnl_BcdA8b2a,
        Bcdea8b = dnnl_Bcdea8b,
        BcdeA8b2a = dnnl_BcdeA8b2a,
        aCdb8c = dnnl_aCdb8c,
        aCdB8c2b = dnnl_aCdB8c2b,
        aCdeb8c = dnnl_aCdeb8c,
        aCdeB8c2b = dnnl_aCdeB8c2b,
        aCdefb8c = dnnl_aCdefb8c,
        aCdefB8c2b = dnnl_aCdefB8c2b,
        Bca24b = dnnl_Bca24b,
        BcA24b2a = dnnl_BcA24b2a,
        Bcda24b = dnnl_Bcda24b,
        BcdA24b2a = dnnl_BcdA24b2a,
        Bcdea24b = dnnl_Bcdea24b,
        BcdeA24b2a = dnnl_BcdeA24b2a,
        aCdb24c = dnnl_aCdb24c,
        aCdB24c2b = dnnl_aCdB24c2b,
        aCdeb24c = dnnl_aCdeb24c,
        aCdeB24c2b = dnnl_aCdeB24c2b,
        aCdefb24c = dnnl_aCdefb24c,
        aCdefB24c2b = dnnl_aCdefB24c2b,
        Iwo8i = dnnl_Iwo8i,
        IwO8i2o = dnnl_IwO8i2o,
        Iwo24i = dnnl_Iwo24i,
        IwO24i2o = dnnl_IwO24i2o,
        Ihwo8i = dnnl_Ihwo8i,
        IhwO8i2o = dnnl_IhwO8i2o,
        Ihwo24i = dnnl_Ihwo24i,
        IhwO24i2o = dnnl_IhwO24i2o,
        Idhwo8i = dnnl_Idhwo8i,
        IdhwO8i2o = dnnl_IdhwO8i2o,
        Idhwo24i = dnnl_Idhwo24i,
        IdhwO24i2o = dnnl_IdhwO24i2o,
        gIwo8i = dnnl_gIwo8i,
        gIwO8i2o = dnnl_gIwO8i2o,
        gIwo24i = dnnl_gIwo24i,
        gIwO24i2o = dnnl_gIwO24i2o,
        gIhwo8i = dnnl_gIhwo8i,
        gIhwO8i2o = dnnl_gIhwO8i2o,
        gIhwo24i = dnnl_gIhwo24i,
        gIhwO24i2o = dnnl_gIhwO24i2o,
        gIdhwo8i = dnnl_gIdhwo8i,
        gIdhwO8i2o = dnnl_gIdhwO8i2o,
        gIdhwo24i = dnnl_gIdhwo24i,
        gIdhwO24i2o = dnnl_gIdhwO24i2o,
        OhwI24o = dnnl_OhwI24o,
        gOhwI24o = dnnl_gOhwI24o,
        AB8b24a2b = dnnl_AB8b24a2b,
        ABc8b24a2b = dnnl_ABc8b24a2b,
        AcB8b24a2b = dnnl_AcB8b24a2b,
        ABcd8b24a2b = dnnl_ABcd8b24a2b,
        AcdB8b24a2b = dnnl_AcdB8b24a2b,
        ABcde8b24a2b = dnnl_ABcde8b24a2b,
        AcdeB8b24a2b = dnnl_AcdeB8b24a2b,
        AB8b8a2b = dnnl_AB8b8a2b,
        ABc8b8a2b = dnnl_ABc8b8a2b,
        AcB8b8a2b = dnnl_AcB8b8a2b,
        ABcd8b8a2b = dnnl_ABcd8b8a2b,
        AcdB8b8a2b = dnnl_AcdB8b8a2b,
        ABcde8b8a2b = dnnl_ABcde8b8a2b,
        AcdeB8b8a2b = dnnl_AcdeB8b8a2b,
        OI8i8o2i = dnnl_OI8i8o2i,
        OI8i24o2i = dnnl_OI8i24o2i,
        OIw8i8o2i = dnnl_OIw8i8o2i,
        OwI8i8o2i = dnnl_OwI8i8o2i,
        OIw8i24o2i = dnnl_OIw8i24o2i,
        OwI8i24o2i = dnnl_OwI8i24o2i,
        OIhw8i8o2i = dnnl_OIhw8i8o2i,
        OhwI8i8o2i = dnnl_OhwI8i8o2i,
        OIhw8i24o2i = dnnl_OIhw8i24o2i,
        OhwI8i24o2i = dnnl_OhwI8i24o2i,
        OIdhw8i8o2i = dnnl_OIdhw8i8o2i,
        OdhwI8i8o2i = dnnl_OdhwI8i8o2i,
        OIdhw8i24o2i = dnnl_OIdhw8i24o2i,
        OdhwI8i24o2i = dnnl_OdhwI8i24o2i,
        BcA8b4a = dnnl_BcA8b4a,
        BcdA8b4a = dnnl_BcdA8b4a,
        BcdeA8b4a = dnnl_BcdeA8b4a,
        aCdB8c4b = dnnl_aCdB8c4b,
        aCdeB8c4b = dnnl_aCdeB8c4b,
        aCdefB8c4b = dnnl_aCdefB8c4b,
        BcA24b4a = dnnl_BcA24b4a,
        BcdA24b4a = dnnl_BcdA24b4a,
        BcdeA24b4a = dnnl_BcdeA24b4a,
        aCdB24c4b = dnnl_aCdB24c4b,
        aCdeB24c4b = dnnl_aCdeB24c4b,
        aCdefB24c4b = dnnl_aCdefB24c4b,
        ABc16a4b = dnnl_ABc16a4b,
        ABcd16a4b = dnnl_ABcd16a4b,
        ABcde16a4b = dnnl_ABcde16a4b,
        IwO8i4o = dnnl_IwO8i4o,
        IwO24i4o = dnnl_IwO24i4o,
        IhwO8i4o = dnnl_IhwO8i4o,
        IhwO24i4o = dnnl_IhwO24i4o,
        IdhwO8i4o = dnnl_IdhwO8i4o,
        IdhwO24i4o = dnnl_IdhwO24i4o,
        gIwO8i4o = dnnl_gIwO8i4o,
        gIwO24i4o = dnnl_gIwO24i4o,
        gIhwO8i4o = dnnl_gIhwO8i4o,
        gIhwO24i4o = dnnl_gIhwO24i4o,
        gIdhwO8i4o = dnnl_gIdhwO8i4o,
        gIdhwO24i4o = dnnl_gIdhwO24i4o,
        BA2a24b = dnnl_BA2a24b,
        aCB2b24c = dnnl_aCB2b24c,
        BA2a8b = dnnl_BA2a8b,
        aCB2b8c = dnnl_aCB2b8c,
        BA8a24b = dnnl_BA8a24b,
        aCB8b24c = dnnl_aCB8b24c,
        BA8a16b = dnnl_BA8a16b,
        aCB8b16c = dnnl_aCB8b16c,
        BA8a8b = dnnl_BA8a8b,
        aCB8b8c = dnnl_aCB8b8c,
        bcad = dnnl_bcad,
        cabd = dnnl_cabd,
        dabc = dnnl_dabc,
    };

    /// A memory descriptor.
    struct desc : public handle<dnnl_memory_desc_t> {
        using handle<dnnl_memory_desc_t>::handle;

        friend struct memory;

        /// Constructs a zero (empty) memory descriptor. Such a memory
        /// descriptor can be used to indicate absence of an argument.
        desc() {
            dnnl_memory_desc_t zero_md = nullptr;
            error::wrap_c_api(
                    dnnl_memory_desc_create_with_tag(&zero_md, 0, nullptr,
                            dnnl_data_type_undef, dnnl_format_tag_undef),
                    "could not create a zero memory descriptor");
            reset(zero_md);
        }

        /// Constructs a memory descriptor.
        ///
        /// @note
        ///     The logical order of dimensions corresponds to the `abc...`
        ///     format tag, and the physical meaning of the dimensions depends
        ///     both on the primitive that would operate on this memory and
        ///     the operation context.
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param aformat_tag Memory format tag.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        desc(const dims &adims, data_type adata_type, format_tag aformat_tag,
                bool allow_empty = false) {
            validate_dims(adims);
            dnnl_memory_desc_t md = nullptr;
            dnnl_status_t status = dnnl_memory_desc_create_with_tag(&md,
                    (int)adims.size(), adims.data(), convert_to_c(adata_type),
                    convert_to_c(aformat_tag));
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not construct a memory descriptor using a "
                        "format tag");
            reset(md);
        }

        /// Constructs a memory descriptor by strides.
        ///
        /// @note
        ///     The logical order of dimensions corresponds to the `abc...`
        ///     format tag, and the physical meaning of the dimensions depends
        ///     both on the primitive that would operate on this memory and
        ///     the operation context.
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param strides Strides for each dimension.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        desc(const dims &adims, data_type adata_type, const dims &strides,
                bool allow_empty = false) {
            validate_dims(adims);
            if (!strides.empty()) validate_dims(strides, (int)adims.size());
            dnnl_memory_desc_t md = nullptr;
            dnnl_status_t status = dnnl_memory_desc_create_with_strides(&md,
                    (int)adims.size(), adims.data(), convert_to_c(adata_type),
                    strides.empty() ? nullptr : &strides[0]);
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not construct a memory descriptor using "
                        "strides");
            reset(md);
        }
#ifdef DNNL_EXPERIMENTAL_SPARSE
        /// Function for creating a memory descriptor for CSR sparse encoding.
        ///
        /// The created memory descriptor will describe a memory object that
        /// contains 3 buffers. The buffers have the following meaning and
        /// assigned numbers (index):
        ///  - 0: values
        ///  - 1: indices
        ///  - 2: pointers
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param nnz Number of non-zero entries.
        /// @param index_dt Data type of indices.
        /// @param pointer_dt Data type of pointers.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        static desc csr(const dims &adims, data_type adata_type, dim nnz,
                data_type index_dt, data_type pointer_dt,
                bool allow_empty = false) {
            validate_dims(adims);
            dnnl_memory_desc_t md = nullptr;
            dnnl_status_t status = dnnl_memory_desc_create_with_csr_encoding(
                    &md, (int)adims.size(), adims.data(),
                    convert_to_c(adata_type), nnz, convert_to_c(index_dt),
                    convert_to_c(pointer_dt));
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a memory descriptor for CSR sparse "
                        "encoding");
            return desc {md};
        }

        /// Function for creating a memory descriptor for packed sparse
        /// encoding.
        ///
        /// The created memory descriptor cannot be used to create a memory
        /// object. It can only be used to create a primitive descriptor to
        /// query the actual memory descriptor (similar to the format tag
        /// `any`).
        ///
        /// @warning
        ///     The meaning and content of the handles of the memory object that
        ///     is created using the queried memory descriptor are unspecified
        ///     therefore using the content is an undefined behavior.
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param nnz Number of non-zero entries.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        static desc packed(const dims &adims, data_type adata_type, dim nnz,
                bool allow_empty = false) {
            validate_dims(adims);
            dnnl_memory_desc_t md = nullptr;
            dnnl_status_t status = dnnl_memory_desc_create_with_packed_encoding(
                    &md, (int)adims.size(), adims.data(),
                    convert_to_c(adata_type), nnz);
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a memory descriptor for packed "
                        "sparse encoding");
            return desc {md};
        }
#endif
        /// Construct a memory descriptor from a C API ::dnnl_memory_desc_t
        /// handle. The resulting handle is not weak and the C handle will be
        /// destroyed during the destruction of the C++ object.
        ///
        /// @param md The C API memory descriptor.
        desc(dnnl_memory_desc_t md) : handle<dnnl_memory_desc_t>(md) {}

        /// Construct a memory descriptor from a binary blob.
        ///
        /// @param blob A binary blob previously queried from a memory descriptor.
        desc(const std::vector<uint8_t> &blob) {
            dnnl_memory_desc_t md = nullptr;
            error::wrap_c_api(
                    dnnl_memory_desc_create_with_blob(&md, blob.data()),
                    "could not create a memory descriptor from blob");
            reset(md);
        }

        /// Constructs a memory descriptor for a region inside an area
        /// described by this memory descriptor.
        //
        /// @param adims Sizes of the region.
        /// @param offsets Offsets to the region from the encompassing
        ///     memory object in each dimension.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A memory descriptor for the region.
        desc submemory_desc(const dims &adims, const dims &offsets,
                bool allow_empty = false) const {
            validate_dims(adims, get_ndims());
            validate_dims(offsets, get_ndims());
            dnnl_memory_desc_t sub_md = nullptr;
            dnnl_status_t status = dnnl_memory_desc_create_submemory(
                    &sub_md, get(), adims.data(), offsets.data());
            if (!allow_empty)
                error::wrap_c_api(status, "could not construct a sub-memory");
            return desc(sub_md);
        }

        /// Constructs a memory descriptor by reshaping an existing one. The
        /// new memory descriptor inherits the data type. This operation is
        /// valid only for memory descriptors that have format_kind set to
        /// #dnnl::memory::format_kind::blocked or
        /// #dnnl::memory::format_kind::any.
        ///
        /// The operation ensures that the transformation of the physical memory
        /// format corresponds to the transformation of the logical dimensions.
        /// If such transformation is impossible, the function either throws an
        /// exception (default) or returns a zero memory descriptor depending on
        /// the `allow_empty` flag.
        ///
        /// The reshape operation can be described as a combination of the
        /// following basic operations:
        /// 1. Add a dimension of size `1`. This is always possible.
        /// 2. Remove a dimension of size `1`. This is possible only if the
        ///    dimension has no padding (i.e.
        ///    `padded_dims[dim] == dims[dim] && dims[dim] == 1`).
        /// 3. Split a dimension into multiple ones. This is possible only if
        ///    the product of all tensor dimensions stays constant and the
        ///    dimension being split does not have padding (i.e.
        ///    `padded_dims[dim] = dims[dim]`).
        /// 4. Join multiple consecutive dimensions into a single one. As in
        ///    the cases above, this requires that the dimensions do not have
        ///    padding and that the memory format is such that in physical
        ///    memory these dimensions are dense and have the same order as
        ///    their logical counterparts. This also assumes that these
        ///    dimensions are not blocked.
        ///    - Here, 'dense' means:
        ///      `stride for dim[i] == (stride for dim[i + 1]) * dim[i + 1]`;
        ///    - And 'same order' means:
        ///      `i < j` if and only if `stride for dim[j] <= stride for dim[i]`.
        ///
        /// @warning
        ///     Some combinations of physical memory layout and/or offsets or
        ///     dimensions may result in a failure to make a reshape.
        ///
        /// @param adims New dimensions. The product of dimensions must
        ///     remain constant.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A new memory descriptor with new dimensions.
        desc reshape(const dims &adims, bool allow_empty = false) const {
            if (get_ndims()) validate_dims(adims, 1);
            dnnl_memory_desc_t out_md = nullptr;
            dnnl_status_t status = dnnl_memory_desc_reshape(
                    &out_md, get(), (int)adims.size(), adims.data());
            if (!allow_empty)
                error::wrap_c_api(
                        status, "could not reshape a memory descriptor");
            return desc(out_md);
        }

        /// Constructs a memory descriptor by permuting axes in an existing
        /// one.
        ///
        /// The physical memory layout representation is adjusted accordingly
        /// to maintain the consistency between the logical and physical parts
        /// of the memory descriptor. The new memory descriptor inherits the
        /// data type.
        ///
        /// The new memory descriptor inherits the data type. This operation is
        /// valid only for memory descriptors that have format_kind set to
        /// #dnnl::memory::format_kind::blocked or
        /// #dnnl::memory::format_kind::any.
        ///
        /// The logical axes will be permuted in the following manner:
        /// @code
        /// for (i = 0; i < get_ndims(); i++)
        ///     new_desc.dims()[permutation[i]] = dims()[i];
        /// @endcode
        ///
        /// Example:
        /// @code
        ///     std::vector<int> permutation = {1, 0}; // swap the first and
        ///                                            // the second axes
        ///     dnnl::memory::desc in_md(
        ///             {2, 3}, data_type, memory::format_tag::ab);
        ///     dnnl::memory::desc expect_out_md(
        ///             {3, 2}, data_type, memory::format_tag::ba);
        ///
        ///     assert(in_md.permute_axes(permutation) == expect_out_md);
        /// @endcode
        ///
        /// @param permutation Axes permutation.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be returned. This flag is optional
        ///     and defaults to false.
        /// @returns A new memory descriptor with new dimensions.
        desc permute_axes(const std::vector<int> &permutation,
                bool allow_empty = false) const {
            validate_dims(permutation, get_ndims());
            dnnl_memory_desc_t out_md = nullptr;
            dnnl_status_t status = dnnl_memory_desc_permute_axes(
                    &out_md, get(), permutation.data());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not permute axes of a memory descriptor");
            return desc(out_md);
        }

        /// Returns a number of dimensions of the memory descriptor.
        ///
        /// @returns A number of dimensions.
        int get_ndims() const { return query_s32(query::ndims_s32); }

        /// Returns padded dimensions of the memory descriptor.
        ///
        /// @returns A copy of the padded dimensions vector.
        memory::dims get_padded_dims() const {
            return query_dims(query::padded_dims);
        }

        /// Returns padded offsets of the memory descriptor.
        ///
        /// @returns A copy of the padded offsets vector.
        memory::dims get_padded_offsets() const {
            return query_dims(query::padded_offsets);
        }

        /// Returns a submemory offset of the memory descriptor.
        ///
        /// @returns A submemory offset.
        memory::dim get_submemory_offset() const {
            dnnl_dim_t submemory_offset;
            dnnl_status_t status = dnnl_memory_desc_query(
                    get(), dnnl_query_submemory_offset_s64, &submemory_offset);
            return status == dnnl_success ? submemory_offset : 0;
        }

        /// Returns strides of the memory descriptor.
        ///
        /// @note
        ///     This API is only applicable to memory descriptors with format
        ///     kind #dnnl_blocked.
        ///
        /// @returns A copy of the strides vector.
        /// @returns An empty #dnnl::memory::dims if the memory descriptor
        ///     does not have strides.
        memory::dims get_strides() const { return query_dims(query::strides); }

        /// Returns a number of inner blocks of the memory descriptor.
        ///
        /// @note
        ///     This API is only applicable to memory descriptors with format
        ///     kind #dnnl_blocked.
        ///
        /// @returns A number of inner blocks.
        int get_inner_nblks() const {
            return query_s32(query::inner_nblks_s32);
        }

        /// Returns inner blocks of the memory descriptor.
        ///
        /// @note
        ///     This API is only applicable to memory descriptors with format
        ///     kind #dnnl_blocked.
        ///
        /// @returns A copy of the inner blocks vector.
        /// @returns An empty #dnnl::memory::dims if the memory descriptor
        ///     does not have inner blocks.
        memory::dims get_inner_blks() const {
            return query_dims(query::inner_blks);
        }

        /// Returns inner indices of the memory descriptor.
        ///
        /// @note
        ///     This API is only applicable to memory descriptors with format
        ///     kind #dnnl_blocked.
        ///
        /// @returns A copy of the inner indices vector.
        /// @returns An empty #dnnl::memory::dims if the memory descriptor
        ///     does not have inner indices.
        memory::dims get_inner_idxs() const {
            return query_dims(query::inner_idxs);
        }

#ifdef DNNL_EXPERIMENTAL_SPARSE
        /// Returns number of handles.
        ///
        /// @returns A number of handles.
        int get_num_handles() const {
            int nhandles;
            dnnl_status_t status = dnnl_memory_desc_query_v2(
                    get(), dnnl_query_num_handles_s32, 0, &nhandles);
            return status == dnnl_success ? nhandles : 0;
        }

        /// Returns a number of non-zero entries of the memory descriptor.
        ///
        /// @returns A number non-zero entries.
        dim get_nnz() const {
            dnnl_dim_t nnz;
            dnnl_status_t status = dnnl_memory_desc_query_v2(
                    get(), dnnl_query_nnz_s64, 0, &nnz);
            return status == dnnl_success ? nnz : 0;
        }

        /// Returns the sparse encoding of the memory descriptor.
        ///
        /// @returns the sparse encoding kind.
        memory::sparse_encoding get_sparse_encoding() const {
            dnnl_sparse_encoding_t sparse_encoding;
            dnnl_status_t status = dnnl_memory_desc_query_v2(
                    get(), dnnl_query_sparse_encoding, 0, &sparse_encoding);
            return status == dnnl_success
                    ? static_cast<dnnl::memory::sparse_encoding>(
                            sparse_encoding)
                    : dnnl::memory::sparse_encoding::undef;
        }

        /// Returns the data type of the memory descriptor.
        ///
        /// @returns The data type.
        memory::data_type get_data_type(int index = 0) const {
            return query_data_type(query::data_type, index);
        }
#else
        /// Returns the data type of the memory descriptor.
        ///
        /// @returns The data type.
        memory::data_type get_data_type() const {
            return query_data_type(query::data_type);
        }
#endif

        /// Returns the format kind of the memory descriptor.
        ///
        /// @returns the format kind.
        memory::format_kind get_format_kind() const {
            dnnl_format_kind_t format_kind;
            dnnl_status_t status = dnnl_memory_desc_query(
                    get(), dnnl_query_format_kind, &format_kind);
            return status == dnnl_success
                    ? static_cast<dnnl::memory::format_kind>(format_kind)
                    : dnnl::memory::format_kind::undef;
        }

        /// Returns dimensions of the memory descriptor.
        ///
        /// Potentially expensive due to the data copy involved.
        /// @returns A copy of the dimensions vector.
        memory::dims get_dims() const { return query_dims(query::dims); }

#ifdef DNNL_EXPERIMENTAL_SPARSE
        /// Returns size of the memory descriptor in bytes.
        /// @param index Data index. Defaults to 0.
        /// @returns The number of bytes required to allocate a memory buffer
        ///     for data with a particular @p index described by this memory
        ///     descriptor including the padding area.
        size_t get_size(int index = 0) const {
            return dnnl_memory_desc_get_size_v2(get(), index);
        }
#else
        /// Returns size of the memory descriptor in bytes.
        /// @returns The number of bytes required to allocate a memory buffer
        ///     for the memory object described by this memory descriptor
        ///     including the padding area.
        size_t get_size() const { return dnnl_memory_desc_get_size(get()); }
#endif

        /// Returns a binary blob associated with the given memory descriptor
        /// @returns The memory descriptor blob associated with the memory descriptor
        std::vector<uint8_t> get_blob() {
            size_t size;
            dnnl_status_t status
                    = dnnl_memory_desc_get_blob(nullptr, &size, get());
            error::wrap_c_api(
                    status, "could not get memory descriptor blob size");

            std::vector<uint8_t> out_blob(size);
            status = dnnl_memory_desc_get_blob(out_blob.data(), &size, get());
            error::wrap_c_api(status, "could not get memory descriptor blob");
            return out_blob;
        }

        /// Checks whether the memory descriptor is zero (empty).
        /// @returns @c true if the memory descriptor describes an empty
        ///     memory and @c false otherwise.
        bool is_zero() const { return get_ndims() == 0; }

        /// An equality operator.
        /// @param other Another memory descriptor.
        /// @returns Whether this and the other memory descriptors have
        ///     the same format tag, dimensions, strides, blocking, etc.
        bool operator==(const desc &other) const {
            return dnnl_memory_desc_equal(get(), other.get()) != 0;
        }

        /// An inequality operator.
        /// @param other Another memory descriptor.
        /// @returns Whether this and the other memory descriptors describe
        ///     different memory.
        bool operator!=(const desc &other) const { return !operator==(other); }

    private:
#ifdef DNNL_EXPERIMENTAL_SPARSE
        memory::data_type query_data_type(query what, int index) const {
            dnnl_data_type_t data_type;
            dnnl_status_t status = dnnl_memory_desc_query_v2(
                    get(), dnnl::convert_to_c(what), index, &data_type);
            return status == dnnl_success
                    ? static_cast<dnnl::memory::data_type>(data_type)
                    : dnnl::memory::data_type::undef;
        }
#else
        memory::data_type query_data_type(query what) const {
            dnnl_data_type_t data_type;
            dnnl_status_t status = dnnl_memory_desc_query(
                    get(), dnnl::convert_to_c(what), &data_type);
            return status == dnnl_success
                    ? static_cast<dnnl::memory::data_type>(data_type)
                    : dnnl::memory::data_type::undef;
        }
#endif

        int query_s32(query what) const {
            int res;
            dnnl_status_t status = dnnl_memory_desc_query(
                    get(), dnnl::convert_to_c(what), &res);
            return status == dnnl_success ? res : 0;
        }

        memory::dims query_dims(query what) const {
            dnnl_dims_t *c_dims;
            dnnl_status_t status = dnnl_memory_desc_query(
                    get(), dnnl::convert_to_c(what), &c_dims);

            const int ndims
                    = (what == query::inner_idxs || what == query::inner_blks)
                    ? get_inner_nblks()
                    : get_ndims();

            return status == dnnl_success
                    ? memory::dims(*c_dims, *c_dims + ndims)
                    : memory::dims {};
        }
    };

    /// Default constructor.
    ///
    /// Constructs an empty memory object, which can be used to indicate
    /// absence of a parameter.
    memory() = default;

#ifdef DNNL_EXPERIMENTAL_SPARSE
    /// Constructs a memory object.
    ///
    /// Unless @p handle is equal to #DNNL_MEMORY_NONE, the constructed memory
    /// object will have the underlying buffer set. In this case, the buffer
    /// will be initialized as if #dnnl::memory::set_data_handle() had been
    /// called.
    ///
    /// @sa memory::set_data_handle()
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    /// @param handle Handle of the memory buffer to use.
    ///     - A pointer to the user-allocated buffer. In this case the library
    ///       doesn't own the buffer.
    ///     - The #DNNL_MEMORY_ALLOCATE special value. Instructs the library to
    ///       allocate the buffer for the memory object. In this case the
    ///       library owns the buffer.
    ///     - #DNNL_MEMORY_NONE to create dnnl::memory without an underlying
    ///       buffer.
    memory(const desc &md, const engine &aengine, void *handle)
        : memory(md, aengine, std::vector<void *> {handle}) {}

    /// Constructs a memory object with multiple handles.
    ///
    /// Unless @p handle is equal to #DNNL_MEMORY_NONE, the constructed memory
    /// object will have the underlying buffer set. In this case, the buffer
    /// will be initialized as if #dnnl::memory::set_data_handle() had been
    /// called.
    ///
    /// @sa memory::set_data_handle()
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    /// @param handles Handles of the memory buffers to use.
    ///     For each element of the @p handles vector the following applies:
    ///     - A pointer to the user-allocated buffer. In this case the library
    ///       doesn't own the buffer.
    ///     - The #DNNL_MEMORY_ALLOCATE special value. Instructs the library to
    ///       allocate the buffer for the memory object. In this case the
    ///       library owns the buffer.
    ///     - #DNNL_MEMORY_NONE Instructs the library to skip allocation of the
    ///       memory buffer.
    memory(const desc &md, const engine &aengine, std::vector<void *> handles) {
        dnnl_memory_t result;
        dnnl_status_t status = dnnl_memory_create_v2(&result, md.get(),
                aengine.get(), (int)handles.size(), handles.data());
        error::wrap_c_api(status, "could not create a memory object");
        reset(result);
    }

    /// Constructs a memory object.
    ///
    /// The underlying buffer(s) for the memory will be allocated by the
    /// library.
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    memory(const desc &md, const engine &aengine) {
        dnnl_status_t status;
        dnnl_memory_t result;
        const int nhandles = md.get_num_handles();

        std::vector<void *> handles(nhandles, DNNL_MEMORY_ALLOCATE);
        status = dnnl_memory_create_v2(&result, md.get(), aengine.get(),
                (int)handles.size(), handles.data());

        error::wrap_c_api(status, "could not create a memory object");
        reset(result);
    }
#else
    /// Constructs a memory object.
    ///
    /// Unless @p handle is equal to #DNNL_MEMORY_NONE, the constructed memory
    /// object will have the underlying buffer set. In this case, the buffer
    /// will be initialized as if #dnnl::memory::set_data_handle() had been
    /// called.
    ///
    /// @sa memory::set_data_handle()
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    /// @param handle Handle of the memory buffer to use.
    ///     - A pointer to the user-allocated buffer. In this case the library
    ///       doesn't own the buffer.
    ///     - The #DNNL_MEMORY_ALLOCATE special value. Instructs the library to
    ///       allocate the buffer for the memory object. In this case the
    ///       library owns the buffer.
    ///     - #DNNL_MEMORY_NONE to create dnnl::memory without an underlying
    ///       buffer.
    memory(const desc &md, const engine &aengine, void *handle) {
        dnnl_memory_t result;
        error::wrap_c_api(
                dnnl_memory_create(&result, md.get(), aengine.get(), handle),
                "could not create a memory object");
        reset(result);
    }

    /// Constructs a memory object.
    ///
    /// The underlying buffer for the memory will be allocated by the library.
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    memory(const desc &md, const engine &aengine)
        : memory(md, aengine, DNNL_MEMORY_ALLOCATE) {}
#endif

    /// Returns the associated memory descriptor.
    desc get_desc() const {
        const_dnnl_memory_desc_t cdesc;
        error::wrap_c_api(dnnl_memory_get_memory_desc(get(), &cdesc),
                "could not get a memory descriptor from a memory object");
        dnnl_memory_desc_t cloned_md = nullptr;
        error::wrap_c_api(dnnl_memory_desc_clone(&cloned_md, cdesc),
                "could not clone a memory descriptor");
        return desc(cloned_md);
    }

    /// Returns the associated engine.
    engine get_engine() const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_memory_get_engine(get(), &c_engine),
                "could not get an engine from a memory object");
        return engine(c_engine, true);
    }

#ifdef DNNL_EXPERIMENTAL_SPARSE
    /// Returns an underlying memory buffer that corresponds to the given index.
    ///
    /// On the CPU engine, or when using USM, this is a pointer to the
    /// allocated memory.
    void *get_data_handle(int index = 0) const {
        void *handle;
        error::wrap_c_api(dnnl_memory_get_data_handle_v2(get(), &handle, index),
                "could not get a native handle from a memory object");
        return handle;
    }

    /// Sets an underlying memory buffer that corresponds to the given index.
    ///
    /// @param handle Memory buffer to use. On the CPU engine or when USM is
    ///     used, the memory buffer is a pointer to the actual data. For OpenCL
    ///     it is a cl_mem. It must have at least
    ///     #dnnl::memory::desc::get_size() bytes allocated.
    /// @param index Memory index to attach the buffer. Defaults to 0.
    void set_data_handle(void *handle, int index = 0) const {
        error::wrap_c_api(dnnl_memory_set_data_handle_v2(get(), handle, index),
                "could not set native handle of a memory object");
    }

    /// Maps a memory object and returns a host-side pointer to a memory
    /// buffer with a copy of its contents. The memory buffer corresponds to
    /// the given index.
    ///
    /// Mapping enables read/write directly from/to the memory contents for
    /// engines that do not support direct memory access.
    ///
    /// Mapping is an exclusive operation - a memory object cannot be used in
    /// other operations until it is unmapped via #dnnl::memory::unmap_data()
    /// call.
    ///
    /// @note
    ///     Any primitives working with the memory should be completed before
    ///     the memory is mapped. Use #dnnl::stream::wait() to synchronize the
    ///     corresponding execution stream.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be suboptimal.
    ///
    /// @tparam T Data type to return a pointer to.
    /// @param index Index of the buffer. Defaults to 0.
    /// @returns Pointer to the mapped memory.
    template <typename T = void>
    T *map_data(int index = 0) const {
        void *mapped_ptr;
        error::wrap_c_api(dnnl_memory_map_data_v2(get(), &mapped_ptr, index),
                "could not map memory object data");
        return static_cast<T *>(mapped_ptr);
    }

    /// Unmaps a memory object and writes back any changes made to the
    /// previously mapped memory buffer. The memory buffer corresponds to
    /// the given index.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be
    ///     suboptimal.
    ///
    /// @param mapped_ptr A pointer previously returned by
    ///     #dnnl::memory::map_data().
    /// @param index Index of the buffer. Defaults to 0.
    void unmap_data(void *mapped_ptr, int index = 0) const {
        error::wrap_c_api(dnnl_memory_unmap_data_v2(get(), mapped_ptr, index),
                "could not unmap memory object data");
    }
#else
    /// Returns the underlying memory buffer.
    ///
    /// On the CPU engine, or when using USM, this is a pointer to the
    /// allocated memory.
    void *get_data_handle() const {
        void *handle;
        error::wrap_c_api(dnnl_memory_get_data_handle(get(), &handle),
                "could not get a native handle from a memory object");
        return handle;
    }

    /// Sets the underlying memory buffer.
    ///
    /// @param handle Memory buffer to use. On the CPU engine or when USM is
    ///     used, the memory buffer is a pointer to the actual data. For OpenCL
    ///     it is a cl_mem. It must have at least
    ///     #dnnl::memory::desc::get_size() bytes allocated.
    void set_data_handle(void *handle) const {
        error::wrap_c_api(dnnl_memory_set_data_handle(get(), handle),
                "could not set native handle of a memory object");
    }

    /// Maps a memory object and returns a host-side pointer to a memory
    /// buffer with a copy of its contents.
    ///
    /// Mapping enables read/write directly from/to the memory contents for
    /// engines that do not support direct memory access.
    ///
    /// Mapping is an exclusive operation - a memory object cannot be used in
    /// other operations until it is unmapped via #dnnl::memory::unmap_data()
    /// call.
    ///
    /// @note
    ///     Any primitives working with the memory should be completed before
    ///     the memory is mapped. Use #dnnl::stream::wait() to synchronize the
    ///     corresponding execution stream.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be suboptimal.
    ///
    /// @tparam T Data type to return a pointer to.
    /// @returns Pointer to the mapped memory.
    template <typename T = void>
    T *map_data() const {
        void *mapped_ptr;
        error::wrap_c_api(dnnl_memory_map_data(get(), &mapped_ptr),
                "could not map memory object data");
        return static_cast<T *>(mapped_ptr);
    }

    /// Unmaps a memory object and writes back any changes made to the
    /// previously mapped memory buffer.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be
    ///     suboptimal.
    ///
    /// @param mapped_ptr A pointer previously returned by
    ///     #dnnl::memory::map_data().
    void unmap_data(void *mapped_ptr) const {
        error::wrap_c_api(dnnl_memory_unmap_data(get(), mapped_ptr),
                "could not unmap memory object data");
    }
#endif

    static dnnl_data_type_t convert_to_c(data_type adata_type) {
        return static_cast<dnnl_data_type_t>(adata_type);
    }
    static dnnl_format_tag_t convert_to_c(format_tag format) {
        return static_cast<dnnl_format_tag_t>(format);
    }
};

inline bool operator==(dnnl_data_type_t a, memory::data_type b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(dnnl_data_type_t a, memory::data_type b) {
    return !(a == b);
}
inline bool operator==(memory::data_type a, dnnl_data_type_t b) {
    return b == a;
}
inline bool operator!=(memory::data_type a, dnnl_data_type_t b) {
    return !(a == b);
}

inline bool operator==(dnnl_format_tag_t a, memory::format_tag b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(dnnl_format_tag_t a, memory::format_tag b) {
    return !(a == b);
}
inline bool operator==(memory::format_tag a, dnnl_format_tag_t b) {
    return b == a;
}
inline bool operator!=(memory::format_tag a, dnnl_format_tag_t b) {
    return !(a == b);
}

/// @} dnnl_api_memory

/// @addtogroup dnnl_api_primitives
/// @{
/// @addtogroup dnnl_api_attributes Attributes
///
/// A container for parameters that extend primitives behavior.
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_post_ops_t> {
    static dnnl_status_t destructor(dnnl_post_ops_t p) {
        return dnnl_post_ops_destroy(p);
    }
};
/// @endcond

/// Post-ops.
///
/// Post-ops are computations executed after the main primitive computations
/// and are attached to the primitive via primitive attributes.
///
/// @sa @ref dev_guide_attributes_post_ops
///
struct post_ops : public handle<dnnl_post_ops_t> {
    using handle<dnnl_post_ops_t>::handle;

    /// Constructs an empty sequence of post-ops.
    post_ops() {
        dnnl_post_ops_t result;
        error::wrap_c_api(
                dnnl_post_ops_create(&result), "could not create post-ops");
        reset(result);
    }

    /// Creates post-ops primitive attribute from a C API ::dnnl_post_ops_t
    /// handle. The resulting handle is not weak and the C handle will be
    /// destroyed during the destruction of the C++ object.
    ///
    /// @param post_ops The C API post-ops primitive attribute.
    post_ops(dnnl_post_ops_t post_ops) : handle<dnnl_post_ops_t>(post_ops) {}

    /// Returns the number of post-ops entries.
    int len() const { return dnnl_post_ops_len(get()); }

    /// Returns the primitive kind of post-op at entry with a certain index.
    /// @param index Index of the post-op to return the kind for.
    /// @returns Primitive kind of the post-op at the specified index.
    primitive::kind kind(int index) const {
        error::wrap_c_api(index < len() ? dnnl_success : dnnl_invalid_arguments,
                "post-ops index is out of range");
        return static_cast<primitive::kind>(
                dnnl_post_ops_get_kind(get(), index));
    }

    /// Appends an accumulation (sum) post-op. Prior to accumulating the
    /// result, the previous value will be will be reduced by zero point
    /// @p zero_point and multiplied by a scaling factor @p scale.
    ///
    /// The kind of this post-op is #dnnl::primitive::kind::sum.
    ///
    /// This feature may improve performance for cases like dequantize the
    /// asymmetrically quantized sum's src1 tensor to f32 domain before
    /// performing the sum operation by subtracting @p zero_point before the
    /// scaling.
    ///
    /// In the simplest case when the accumulation is the only post-op,
    /// the computations will be `dst[:] := scale * (dst[:] - zero_point) +
    /// op(...)` instead of `dst[:] := op(...)`.
    ///
    /// If @p data_type is specified, the original dst tensor will be
    /// reinterpreted as a tensor with the provided data type. Because it is a
    /// reinterpretation, data_type and dst data type should have the same size.
    /// As a result, computations will be `dst[:] <- scale *
    /// (as_data_type(dst[:]) - zero_point) + op(...)` instead of
    /// `dst[:] <- op(...)`.
    ///
    /// @note
    ///     This post-op executes in-place and does not change the
    ///     destination layout.
    ///
    /// @param scale Scaling factor.
    /// @param zero_point Zero point.
    /// @param data_type Data type.
    void append_sum(float scale = 1.f, int32_t zero_point = 0,
            memory::data_type data_type = memory::data_type::undef) {
        error::wrap_c_api(dnnl_post_ops_append_sum(get(), scale, zero_point,
                                  memory::convert_to_c(data_type)),
                "could not append a sum post-op");
    }

    /// Returns the parameters of an accumulation (sum) post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    void get_params_sum(int index, float &scale) const {
        error::wrap_c_api(dnnl_post_ops_get_params_sum(
                                  get(), index, &scale, nullptr, nullptr),
                "could not get parameters of a sum post-op");
    }

    /// Returns the parameters of an accumulation (sum) post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    /// @param data_type Data type of the sum post-op.
    void get_params_sum(
            int index, float &scale, memory::data_type &data_type) const {
        dnnl_data_type_t c_data_type;
        error::wrap_c_api(dnnl_post_ops_get_params_sum(
                                  get(), index, &scale, nullptr, &c_data_type),
                "could not get parameters of a sum post-op");
        data_type = static_cast<memory::data_type>(c_data_type);
    }

    /// Returns the parameters of an accumulation (sum) post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    /// @param zero_point Single scalar int32_t value of zeropoint.
    /// @param data_type Data type of the sum post-op.
    void get_params_sum(int index, float &scale, int32_t &zero_point,
            memory::data_type &data_type) const {
        dnnl_data_type_t c_data_type;
        error::wrap_c_api(dnnl_post_ops_get_params_sum(get(), index, &scale,
                                  &zero_point, &c_data_type),
                "could not get parameters of a sum post-op");
        data_type = static_cast<memory::data_type>(c_data_type);
    }

    /// Appends an elementwise post-op.
    ///
    /// The kind of this post-op is #dnnl::primitive::kind::eltwise.
    ///
    /// In the simplest case when the elementwise is the only post-op, the
    /// computations would be `dst[:] := eltwise_op (op(...))` instead
    /// of `dst[:] <- op(...)`, where eltwise_op is configured with the given
    /// parameters.
    ///
    /// @param aalgorithm Elementwise algorithm.
    /// @param alpha Alpha parameter for the elementwise algorithm.
    /// @param beta Beta parameter for the elementwise algorithm.
    void append_eltwise(algorithm aalgorithm, float alpha, float beta) {
        error::wrap_c_api(dnnl_post_ops_append_eltwise(
                                  get(), convert_to_c(aalgorithm), alpha, beta),
                "could not append an elementwise post-op");
    }

    /// Returns parameters of an elementwise post-op.
    ///
    /// @param index Index of the post-op.
    /// @param aalgorithm Output elementwise algorithm kind.
    /// @param alpha Output alpha parameter for the elementwise algorithm.
    /// @param beta Output beta parameter for the elementwise algorithm.
    void get_params_eltwise(
            int index, algorithm &aalgorithm, float &alpha, float &beta) const {
        dnnl_alg_kind_t c_alg;
        error::wrap_c_api(dnnl_post_ops_get_params_eltwise(
                                  get(), index, &c_alg, &alpha, &beta),
                "could not get parameters of an elementwise post-op");
        aalgorithm = static_cast<dnnl::algorithm>(c_alg);
    }

    /// Appends a depthwise post-op convolution.
    ///
    /// This post-op can only be fused with a 2D 1x1 convolution (convolution
    /// with weights spatial dimension equal to 1 i.e., kh=kw=1).
    ///
    /// The kind of this post-op is #dnnl_convolution.
    ///
    /// The number of outputs for primitive remain same as before. The output
    /// spatial size can be derived as below:
    ///
    /// output_height = ceil(output_height_1x1_convolution, stride)
    /// output_width = ceil(output_width_1x1_convolution, stride)
    ///
    /// See @ref dev_guide_attributes_post_ops_depthwise and
    /// @ref dev_guide_attributes_post_ops_depthwise_fusion for more info.
    ///
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param kernel_size Size of kernel of depthwise post-op
    /// @param stride_size Size of stride of depthwise post-op
    /// @param padding_l_size Size of left and top paddings of depthwise post-op
    void append_dw(memory::data_type weights_data_type,
            memory::data_type bias_data_type, memory::data_type dst_data_type,
            memory::dim kernel_size, memory::dim stride_size,
            memory::dim padding_l_size) {

        error::wrap_c_api(dnnl_post_ops_append_dw(get(),
                                  memory::convert_to_c(weights_data_type),
                                  memory::convert_to_c(bias_data_type),
                                  memory::convert_to_c(dst_data_type),
                                  kernel_size, stride_size, padding_l_size),
                "could not append depthwise post-op");
    }

    /// Returns the parameters of an depthwise post-op.
    ///
    /// @param index Index of the elementwise post-op.
    /// @param weights_data_type Weights data type of depthwise post-op
    /// @param bias_data_type Bias data type of depthwise post-op
    /// @param dst_data_type Output data type of depthwise post-op
    /// @param kernel_size Size of kernel of depthwise post-op
    /// @param stride_size Size of stride of depthwise post-op
    /// @param padding_l_size Size of left and top paddings of depthwise post-op
    void get_params_dw(int index, memory::data_type &weights_data_type,
            memory::data_type &bias_data_type, memory::data_type &dst_data_type,
            memory::dim &kernel_size, memory::dim &stride_size,
            memory::dim &padding_l_size) const {

        dnnl_data_type_t c_weights_data_type;
        dnnl_data_type_t c_bias_data_type;
        dnnl_data_type_t c_dst_data_type;
        dnnl_dim_t c_kernel_size;
        dnnl_dim_t c_stride_size;
        dnnl_dim_t c_padding_l_size;
        error::wrap_c_api(
                dnnl_post_ops_get_params_dw(get(), index, &c_weights_data_type,
                        &c_bias_data_type, &c_dst_data_type, &c_kernel_size,
                        &c_stride_size, &c_padding_l_size),
                "could not get parameters of depthwise post-op");

        weights_data_type = static_cast<memory::data_type>(c_weights_data_type);
        bias_data_type = static_cast<memory::data_type>(c_bias_data_type);
        dst_data_type = static_cast<memory::data_type>(c_dst_data_type);
        kernel_size = c_kernel_size;
        stride_size = c_stride_size;
        padding_l_size = c_padding_l_size;
    }

    /// Appends a binary post-op.
    ///
    /// The kind of this post operation is #dnnl_binary.
    ///
    /// In the simplest case when the binary is the only post operation, the
    /// computations would be:
    ///
    ///     dst[:] <- binary_op (dst[:], another_input[:])
    ///
    /// where binary_op is configured with the given parameters. binary_op
    /// supports broadcast semantics for a second operand.
    ///
    /// @param aalgorithm Binary algorithm for the post-op.
    /// @param src1_desc Memory descriptor of a second operand.
    void append_binary(algorithm aalgorithm, const memory::desc &src1_desc) {
        error::wrap_c_api(dnnl_post_ops_append_binary(get(),
                                  convert_to_c(aalgorithm), src1_desc.get()),
                "could not append a binary post-op");
    }

    /// Returns the parameters of a binary post-op.
    ///
    /// @param index Index of the binary post-op.
    /// @param aalgorithm Output binary algorithm kind.
    /// @param src1_desc Output memory descriptor of a second operand.
    void get_params_binary(
            int index, algorithm &aalgorithm, memory::desc &src1_desc) const {
        dnnl_alg_kind_t c_alg;
        const_dnnl_memory_desc_t cdesc;
        error::wrap_c_api(
                dnnl_post_ops_get_params_binary(get(), index, &c_alg, &cdesc),
                "could not get parameters of a binary post-op");
        aalgorithm = static_cast<dnnl::algorithm>(c_alg);
        dnnl_memory_desc_t cloned_md = nullptr;
        error::wrap_c_api(dnnl_memory_desc_clone(&cloned_md, cdesc),
                "could not clone a memory descriptor");
        src1_desc = memory::desc(cloned_md);
    }

    /// Appends a prelu forward post-op.
    ///
    /// The kind of this post-op is #dnnl::primitive::kind::prelu.
    ///
    /// The post-op can be defined as:
    ///
    ///      dst[:] <- prelu(dst[:], weights[:])
    ///      prelu:
    ///      dst[:] <- dst[:] if dst[:] > 0
    ///      dst[:] <- dst[:] * weights[:] if dst[:] <= 0
    ///
    ///
    /// Example usage:
    /// @code
    ///     int mb = 32, oc = 32,
    ///         oh = 14, ow = 14; // convolution output params
    ///     // unique weights per output channel
    ///     vector<float> weights = { ... };
    ///     int oc_dim = 1; // mb_dim = 0, channel_dim = 1, height_dim = 2, ...
    ///
    ///     // construct a convolution descriptor
    ///     dnnl::convolution::desc conv_d;
    ///
    ///     dnnl::primitive_attr attr;
    ///     attr.append_prelu(1 << oc_dim);
    ///
    ///     dnnl::primitive_desc conv_pd(conv_d, attr, engine);
    ///     memory prelu_weights({{1}, dt::f32, {1}}, eng, weights.data());
    ///
    ///     std::unordered_map<int, memory> conv_args;
    ///
    ///     conv_args.insert(
    ///      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_WEIGHTS, prelu_weights})
    /// @endcode
    ///
    /// @note
    ///     The order of dimensions does not depend on how elements are laid
    ///     out in memory. For example:
    ///     - for a 2D CNN activations tensor the order is always (n, c)
    ///     - for a 4D CNN activations tensor the order is always (n, c, h, w)
    ///     - for a 5D CNN weights tensor the order is always
    ///        (g, oc, ic, kh, kw)
    ///
    ///    Prelu weights tensor is passed in runtime execution phase. Prelu
    ///    weights tensor data type is implicitly assumed as f32 using plain
    ///    layout (a, ab, acb, acdb, acdeb).
    ///
    /// @param mask Defines the correspondence between the output tensor
    ///     dimensions and the prelu weights tensor. The set i-th bit indicates
    ///     that a dedicated weights value is used for each index along that
    ///     dimension. Set the mask to 0 to use a common weights value
    ///     for the whole output tensor.
    void append_prelu(int mask) {
        error::wrap_c_api(dnnl_post_ops_append_prelu(get(), mask),
                "could not append a prelu post-op");
    }

    /// Returns the parameters of a prelu post-op.
    ///
    /// @param index Index of the prelu post-op.
    /// @param mask Weights mask of prelu post-op.
    void get_params_prelu(int index, int &mask) const {
        error::wrap_c_api(dnnl_post_ops_get_params_prelu(get(), index, &mask),
                "could not get parameters of a binary post-op");
    }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_primitive_attr_t> {
    static dnnl_status_t destructor(dnnl_primitive_attr_t p) {
        return dnnl_primitive_attr_destroy(p);
    }
};
/// @endcond

/// Primitive attributes.
///
/// @sa @ref dev_guide_attributes
struct primitive_attr : public handle<dnnl_primitive_attr_t> {
    using handle<dnnl_primitive_attr_t>::handle;

    /// Constructs default (empty) primitive attributes.
    primitive_attr() {
        dnnl_primitive_attr_t result;
        error::wrap_c_api(dnnl_primitive_attr_create(&result),
                "could not create primitive attribute");
        reset(result);
    }

    /// Creates primitive attributes from a C API ::dnnl_primitive_attr_t
    /// handle. The resulting handle is not weak and the C handle will be
    /// destroyed during the destruction of the C++ object.
    ///
    /// @param attr The C API primitive attributes.
    primitive_attr(dnnl_primitive_attr_t attr)
        : handle<dnnl_primitive_attr_t>(attr) {}

    /// Returns the fpmath mode
    fpmath_mode get_fpmath_mode() const {
        dnnl_fpmath_mode_t result;
        error::wrap_c_api(dnnl_primitive_attr_get_fpmath_mode(get(), &result),
                "could not get fpmath mode primitive attribute");
        return fpmath_mode(result);
    }

    /// Returns the fpmath mode
    ///
    /// @param mode Specified fpmath mode.
    /// @param apply_to_int Use floating-point arithmetic for integer primitives.
    void get_fpmath_mode(fpmath_mode &mode, bool &apply_to_int) const {
        dnnl_fpmath_mode_t c_mode;
        int c_apply_to_int;
        error::wrap_c_api(dnnl_primitive_attr_get_fpmath_mode_v2(
                                  get(), &c_mode, &c_apply_to_int),
                "could not get fpmath mode primitive attribute");
        mode = fpmath_mode(c_mode);
        apply_to_int = static_cast<bool>(c_apply_to_int);
    }

    /// Sets fpmath mode.
    ///
    /// @param mode Specified fpmath mode.
    /// @param apply_to_int Boolean. Use of floating-point arithmetic for integer primitives.
    void set_fpmath_mode(fpmath_mode mode, bool apply_to_int = false) {
        error::wrap_c_api(dnnl_primitive_attr_set_fpmath_mode_v2(get(),
                                  dnnl::convert_to_c(mode), apply_to_int),
                "could not set fpmath mode primitive attribute");
    }

    /// Returns the accumulation mode
    accumulation_mode get_accumulation_mode() const {
        dnnl_accumulation_mode_t result;
        error::wrap_c_api(
                dnnl_primitive_attr_get_accumulation_mode(get(), &result),
                "could not get accumulation mode primitive attribute");
        return accumulation_mode(result);
    }

    /// Sets accumulation mode.
    ///
    /// @param mode Specified accumulation mode.
    void set_accumulation_mode(accumulation_mode mode) {
        error::wrap_c_api(dnnl_primitive_attr_set_accumulation_mode(
                                  get(), dnnl::convert_to_c(mode)),
                "could not set accumulation mode primitive attribute");
    }

    /// Returns the deterministic attribute value
    bool get_deterministic() const {
        int result;
        error::wrap_c_api(dnnl_primitive_attr_get_deterministic(get(), &result),
                "could not get deterministic primitive attribute");
        return static_cast<bool>(result);
    }

    /// Sets deterministic attribute value
    ///
    /// @param value Specified deterministic mode.
    void set_deterministic(bool value) {
        error::wrap_c_api(dnnl_primitive_attr_set_deterministic(
                                  get(), static_cast<int>(value)),
                "could not set deterministic primitive attribute");
    }

    /// Returns the scratchpad mode.
    scratchpad_mode get_scratchpad_mode() const {
        dnnl_scratchpad_mode_t result;
        error::wrap_c_api(
                dnnl_primitive_attr_get_scratchpad_mode(get(), &result),
                "could not get scratchpad mode primitive attribute");
        return scratchpad_mode(result);
    }

    /// Sets scratchpad mode.
    ///
    /// @param mode Specified scratchpad mode.
    void set_scratchpad_mode(scratchpad_mode mode) {
        error::wrap_c_api(dnnl_primitive_attr_set_scratchpad_mode(
                                  get(), dnnl::convert_to_c(mode)),
                "could not set scratchpad mode primitive attribute");
    }

    /// Sets scaling factors for primitive operations for a given memory
    /// argument. The scaling factors must be passed at execution time
    /// as an argument with index #DNNL_ARG_ATTR_SCALES | arg.
    ///
    /// @sa dnnl_primitive_attr_set_scales_mask
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p scales
    ///     vector. The set i-th bit indicates that a dedicated scaling factor
    ///     is used for each index along that dimension. Set the mask to 0 to
    ///     use a common scaling factor for the whole output tensor.
    void set_scales_mask(int arg, int mask) {
        error::wrap_c_api(dnnl_primitive_attr_set_scales_mask(get(), arg, mask),
                "could not set scales primitive attribute");
    }

    /// Sets scaling factors for primitive operations for a given memory
    /// argument. The scaling factors must be passed at execution time
    /// as an argument with index #DNNL_ARG_ATTR_SCALES | arg.
    ///
    /// @sa dnnl_primitive_attr_set_scales
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Scales correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated
    ///     scale is used for each index along that dimension. Set the
    ///     mask to 0 to use a common scale for the whole output tensor.
    /// @param groups Scaling factors correspondence groups that define the
    ///     correspondence between the tensor dimensions and the scales array.
    ///     The set i-th dimension indicates a number of groups of scaling
    ///     factors used for that logical dimension in a memory indicated by @p arg.
    /// @param data_type Scaling factors data_type.
    void set_scales(int arg, int mask, const memory::dims &groups,
            memory::data_type data_type = memory::data_type::f32) {
        error::wrap_c_api(dnnl_primitive_attr_set_scales(get(), arg, mask,
                                  (int)groups.size(), groups.data(),
                                  memory::convert_to_c(data_type)),
                "could not set scales primitive attribute");
    }

    /// Sets zero points for primitive operations for a given memory argument.
    /// The zero points must be passed at execution time as an argument with
    /// index #DNNL_ARG_ATTR_ZERO_POINTS | arg.
    ///
    /// @sa dnnl_primitive_attr_set_zero_points_mask
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Zero point correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p
    ///     zero_points vector. The set i-th bit indicates that a dedicated
    ///     zero point is used for each index along that dimension. Set the
    ///     mask to 0 to use a common zero point for the whole output tensor.
    void set_zero_points_mask(int arg, int mask) {
        error::wrap_c_api(
                dnnl_primitive_attr_set_zero_points_mask(get(), arg, mask),
                "could not set zero points primitive attribute");
    }

    /// Sets zero points for primitive operations for a given memory argument.
    /// The zero points must be passed at execution time as an argument with
    /// index #DNNL_ARG_ATTR_ZERO_POINTS | arg.
    ///
    /// @sa dnnl_primitive_attr_set_zero_points
    ///
    /// @param arg Parameter argument index as passed to the
    ///     primitive::execute() call.
    /// @param mask Zero point correspondence mask that defines the
    ///     correspondence between the tensor dimensions and the @p
    ///     zero_points vector. The set i-th bit indicates that a dedicated
    ///     zero point is used for each index along that dimension. Set the
    ///     mask to 0 to use a common zero point for the whole output tensor.
    /// @param groups Zero point factors correspondence groups that define the
    ///     correspondence between the tensor dimensions and the zero_points array.
    ///     The set i-th dimension indicates a number of groups of zero point
    ///     factors used for that logical dimension in a memory indicated by @p arg.
    /// @param data_type Zero point factors data_type.
    void set_zero_points(int arg, int mask, const memory::dims &groups,
            memory::data_type data_type = memory::data_type::s32) {
        error::wrap_c_api(dnnl_primitive_attr_set_zero_points(get(), arg, mask,
                                  (int)groups.size(), groups.data(),
                                  memory::convert_to_c(data_type)),
                "could not set zero points primitive attribute");
    }

    /// Returns post-ops previously set via set_post_ops().
    ///
    /// @returns Post-ops.
    const post_ops get_post_ops() const {
        const_dnnl_post_ops_t const_c_post_ops;
        error::wrap_c_api(
                dnnl_primitive_attr_get_post_ops(get(), &const_c_post_ops),
                "could not get post-ops primitive attribute");
        dnnl_post_ops_t c_post_ops;
        error::wrap_c_api(dnnl_post_ops_clone(&c_post_ops, const_c_post_ops),
                "could not clone post-ops primitive attribute");
        return post_ops(c_post_ops);
    }

    /// Sets post-ops.
    ///
    /// @note
    ///     There is no way to check whether the post-ops would be supported
    ///     by the target primitive. Any error will be reported
    ///     by the respective primitive descriptor constructor.
    ///
    /// @param ops Post-ops object to copy post-ops from.
    void set_post_ops(const post_ops ops) {
        error::wrap_c_api(dnnl_primitive_attr_set_post_ops(get(), ops.get()),
                "could not set post-ops primitive attribute");
    }

    /// Sets quantization scale and shift parameters for RNN data tensors.
    ///
    /// For performance reasons, the low-precision configuration of the RNN
    /// primitives expect input activations to have the unsigned 8-bit integer
    /// data type. The scale and shift parameters are used to quantize
    /// floating-point data to unsigned integer and must be passed to the RNN
    /// primitive using attributes.
    ///
    /// The quantization formula is `scale * data + shift`.
    ///
    /// Example usage:
    /// @code
    ///     // RNN parameters
    ///     int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
    ///     // Activations quantization parameters
    ///     float scale = 63.f, shift = 64.f;
    ///
    ///     primitive_attr attr;
    ///
    ///     // Set scale and shift for int8 quantization of activation
    ///     attr.set_rnn_data_qparams(scale, shift);
    ///
    ///     // Create an RNN primitive descriptor.
    ///     vanilla_rnn_forward::primitive_desc rnn_d(
    ///             engine, /* arguments */, attr);
    /// @endcode
    ///
    /// @note
    ///     Quantization scale and shift are common for src_layer, src_iter,
    ///     dst_iter, and dst_layer.
    ///
    /// @param scale The value to scale the data by.
    /// @param shift The value to shift the data by.
    void set_rnn_data_qparams(float scale, float shift) {
        error::wrap_c_api(
                dnnl_primitive_attr_set_rnn_data_qparams(get(), scale, shift),
                "could not set RNN data quantization parameters primitive "
                "attribute");
    }

    /// Returns the quantization scale and shift parameters for RNN data
    /// tensors.
    ///
    /// @note
    ///     Quantization scale and shift are common for src_layer, src_iter,
    ///     dst_iter, and dst_layer.
    ///
    /// @param scale The value to scale the data by.
    /// @param shift The value to shift the data by.
    void get_rnn_data_qparams(float &scale, float &shift) {
        float c_scale, c_shift;
        error::wrap_c_api(dnnl_primitive_attr_get_rnn_data_qparams(
                                  get(), &c_scale, &c_shift),
                "could not set RNN data quantization parameters primitive "
                "attribute");
        scale = c_scale;
        shift = c_shift;
    }

    /// Sets quantization scaling factors for RNN weights tensors. The
    /// low-precision configuration of the RNN primitives expect input weights
    /// to use the signed 8-bit integer data type. The scaling factors are
    /// used to quantize floating-point data to signed integer and must be
    /// passed to RNN primitives using attributes.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @note
    ///     Quantization scales are common for weights_layer and
    ///     weights_iteration
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void set_rnn_weights_qparams(int mask, const std::vector<float> &scales) {
        error::wrap_c_api(dnnl_primitive_attr_set_rnn_weights_qparams(get(),
                                  (int)scales.size(), mask, scales.data()),
                "could not set RNN weights quantization parameters primitive "
                "attribute");
    }

    /// Returns the quantization scaling factors for RNN projection weights
    /// tensors.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void get_rnn_weights_qparams(int &mask, std::vector<float> &scales) {
        dnnl_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(dnnl_primitive_attr_get_rnn_weights_qparams(
                                  get(), &count, &c_mask, &c_scales),
                "could not get primitive RNN weights quantization "
                "parameters attributes");
        scales.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; c++)
            scales[c] = c_scales[c];
    }

    /// Sets quantization scaling factors for RNN projection weights tensors.
    //  The low-precision configuration of the RNN primitives expect input
    //  weights to use the signed 8-bit integer data type. The scaling factors
    //  are used to quantize floating-point data to signed integer and must be
    /// passed to RNN primitives using attributes.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @note
    ///     Quantization scales are common for weights_layer and
    ///     weights_iteration
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void set_rnn_weights_projection_qparams(
            int mask, const std::vector<float> &scales) {
        error::wrap_c_api(
                dnnl_primitive_attr_set_rnn_weights_projection_qparams(
                        get(), (int)scales.size(), mask, scales.data()),
                "could not set primitive RNN weights projection quantization "
                "parameters attributes");
    }

    /// Returns the quantization scaling factors for RNN projection weights
    /// tensors.
    ///
    /// @note
    ///     The dimension order is always native and does not depend on the
    ///     actual layout used. For example, five-dimensional weights always
    ///     have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///     correspondence between the output tensor dimensions and the @p
    ///     scales vector. The set i-th bit indicates that a dedicated scaling
    ///     factor should be used each index along that dimension. Set the
    ///     mask to 0 to use a common scaling factor for the whole output
    ///     tensor.
    /// @param scales Constant vector of output scaling factors. The following
    ///     equality must hold:
    ///     \f$scales.size() = \prod\limits_{d \in mask} weights.dims[d].\f$
    ///     Violations can only be detected when the attributes are used to
    ///     create a primitive descriptor.
    void get_rnn_weights_projection_qparams(
            int &mask, std::vector<float> &scales) {
        dnnl_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(
                dnnl_primitive_attr_get_rnn_weights_projection_qparams(
                        get(), &count, &c_mask, &c_scales),
                "could not get primitive RNN weights projection quantization "
                "parameters attributes");
        scales.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; c++)
            scales[c] = c_scales[c];
    }
};

/// @} dnnl_api_attributes

/// @addtogroup dnnl_api_primitives_common
/// @{

/// Base class for all primitive descriptors.
struct primitive_desc_base : public handle<dnnl_primitive_desc_t> {
    using handle<dnnl_primitive_desc_t>::handle;

    /// Default constructor. Produces an empty object.
    primitive_desc_base() = default;

    /// Returns the engine of the primitive descriptor.
    /// @returns The engine of the primitive descriptor.
    engine get_engine() const { return query_engine(query::engine); }

    /// Returns implementation name.
    /// @returns The implementation name.
    const char *impl_info_str() const {
        const char *res;
        error::wrap_c_api(dnnl_primitive_desc_query(
                                  get(), dnnl_query_impl_info_str, 0, &res),
                "could not retrieve implementation info string from a "
                "primitive descriptor");
        return res;
    }

    /// Returns a memory::dim value (same as int64_t).
    /// @param what The value to query.
    /// @returns The result of the query.
    memory::dim query_s64(query what) const {
        memory::dim res;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl::convert_to_c(what), 0, &res);
        return status == dnnl_success ? res : 0;
    }

    /// Returns strides.
    /// @returns Strides.
    /// @returns An empty #dnnl::memory::dims if the primitive does not have
    ///     a strides parameter.
    memory::dims get_strides() const { return query_dims(query::strides); }

    /// Returns dilations.
    /// @returns Dilations.
    /// @returns An empty #dnnl::memory::dims if the primitive does not have
    ///     a dilations parameter.
    memory::dims get_dilations() const { return query_dims(query::dilations); }

    /// Returns a left padding.
    /// @returns A left padding.
    /// @returns An empty #dnnl::memory::dims if the primitive does not have
    ///     a left padding parameter.
    memory::dims get_padding_l() const { return query_dims(query::padding_l); }

    /// Returns a right padding.
    /// @returns A right padding.
    /// @returns An empty #dnnl::memory::dims if the primitive does not have
    ///     a right padding parameter.
    memory::dims get_padding_r() const { return query_dims(query::padding_r); }

    /// Returns an epsilon.
    /// @returns An epsilon.
    /// @returns Zero if the primitive does not have an epsilon parameter.
    float get_epsilon() const { return query_f32(query::epsilon_f32); }

    /// Returns flags.
    /// @tparam T Flags enumeration type.
    /// @returns Flags.
    /// @returns Zero if the primitive does not have a flags parameter.
    template <typename T = unsigned>
    T get_flags() const {
        unsigned res;
        dnnl_status_t status
                = dnnl_primitive_desc_query(get(), dnnl_query_flags, 0, &res);
        return static_cast<T>(status == dnnl_success ? res : 0x0U);
    }

    /// Returns an algorithm kind.
    /// @returns An algorithm kind.
    /// @returns #dnnl::algorithm::undef if the primitive does not have an
    ///     algorithm parameter.
    dnnl::algorithm get_algorithm() const { return query_alg(query::alg_kind); }

    /// Returns an alpha.
    /// @returns An alpha.
    /// @returns Zero if the primitive does not have an alpha parameter.
    float get_alpha() const { return query_f32(query::alpha_f32); }

    /// Returns a beta.
    /// @returns A beta.
    /// @returns Zero if the primitive does not have a beta parameter.
    float get_beta() const { return query_f32(query::beta_f32); }

    /// Returns an axis.
    /// @returns An axis.
    /// @returns A negative number if the primitive does not have an axis
    ///     parameter.
    int get_axis() const {
        int res;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl_query_axis_s32, 0, &res);
        return status == dnnl_success ? res : -1;
    }

    /// Returns an LRN local size parameter.
    /// @returns An LRN local size parameter.
    /// @returns Zero if the primitive does not have an LRN local size
    ///     parameter.
    memory::dim get_local_size() const {
        return query_s64(query::local_size_s64);
    }

    /// Returns an LRN K parameter.
    /// @returns An LRN K parameter.
    /// @returns Zero if the primitive does not have an LRN K parameter.
    float get_k() const { return query_f32(query::k_f32); }

    /// Returns a reduction P parameter.
    /// @returns A reduction P parameter.
    /// @returns Zero if the primitive does not have a reduction P parameter.
    float get_p() const { return query_f32(query::p_f32); }

    /// Returns a resampling factors parameters.
    /// @returns A vector of factors.
    /// @returns An empty vector if the primitive does not have a resampling
    ///     factors parameter.
    std::vector<float> get_factors() const {
        float *factors;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl_query_factors, 0, &factors);

        const bool is_backward = get_prop_kind() != prop_kind::forward_training
                && get_prop_kind() != prop_kind::forward_inference;
        const_dnnl_memory_desc_t md = dnnl_primitive_desc_query_md(get(),
                is_backward ? dnnl_query_diff_dst_md : dnnl_query_dst_md, 0);

        int ndims;
        error::wrap_c_api(
                dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &ndims),
                "could not query ndims from a memory descriptor");

        return status == dnnl_success
                ? std::vector<float>(factors, factors + (ndims - 2))
                : std::vector<float> {};
    }

    /// Returns an RNN cell kind parameter.
    /// @returns An RNN cell kind parameter.
    /// @returns #dnnl::algorithm::undef if the primitive does not have an
    ///     RNN cell kind parameter.
    dnnl::algorithm get_cell_kind() const {
        return query_alg(query::cell_kind);
    }

    /// Returns an RNN direction parameter.
    /// @returns An RNN direction parameter.
    /// @returns #dnnl::rnn_direction::undef if the primitive does not have
    ///     an RNN direction parameter.
    dnnl::rnn_direction get_direction() const {
        dnnl_rnn_direction_t direction;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl_query_direction, 0, &direction);
        return status == dnnl_success
                ? static_cast<dnnl::rnn_direction>(direction)
                : dnnl::rnn_direction::undef;
    }

    /// Returns an RNN activation kind parameter.
    /// @returns An RNN activation kind parameter.
    /// @returns #dnnl::algorithm::undef if the primitive does not have an
    ///     RNN activation kind parameter.
    dnnl::algorithm get_activation_kind() const {
        return query_alg(query::activation_kind);
    }

    /// Returns a pooling kernel parameter.
    /// @returns A pooling kernel parameter.
    /// @returns An empty #dnnl::memory::dims if the primitive does not have
    ///     a pooling kernel parameter.
    memory::dims get_kernel() const { return query_dims(query::kernel); }

    /// Returns a group size parameter.
    /// @returns A group size parameter.
    /// @returns Zero if the primitive does not have a group size
    ///     parameter.
    memory::dim get_group_size() const {
        return query_s64(query::group_size_s64);
    }

    /// Returns a propagation kind.
    /// @returns A propagation kind.
    /// @returns #dnnl::prop_kind::undef if the primitive does not have
    ///     a propagation parameter.
    dnnl::prop_kind get_prop_kind() const {
        dnnl_prop_kind_t prop_kind;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl_query_prop_kind, 0, &prop_kind);
        return status == dnnl_success ? static_cast<dnnl::prop_kind>(prop_kind)
                                      : dnnl::prop_kind::undef;
    }

    /// Returns a memory descriptor.
    ///
    /// @note
    ///     There are also convenience methods
    ///     #dnnl::primitive_desc_base::src_desc(),
    ///     #dnnl::primitive_desc_base::dst_desc(), and others.
    ///
    /// @param what The kind of parameter to query; can be
    ///     #dnnl::query::src_md, #dnnl::query::dst_md, etc.
    /// @param idx Index of the parameter. For example, convolution bias can
    ///     be queried with what = #dnnl::query::weights_md and idx = 1.
    /// @returns The requested memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     parameter of the specified kind or index.
    memory::desc query_md(query what, int idx = 0) const {
        std::vector<query> valid_q {query::src_md, query::diff_src_md,
                query::weights_md, query::diff_weights_md, query::dst_md,
                query::diff_dst_md, query::workspace_md, query::scratchpad_md,
                query::exec_arg_md};
        if (!std::any_of(valid_q.cbegin(), valid_q.cend(),
                    [=](query q) { return what == q; }))
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "memory descriptor query is invalid");

        const_dnnl_memory_desc_t cdesc = dnnl_primitive_desc_query_md(
                get(), dnnl::convert_to_c(what), idx);
        if (!cdesc) return memory::desc();

        dnnl_memory_desc_t cloned_md = nullptr;
        error::wrap_c_api(dnnl_memory_desc_clone(&cloned_md, cdesc),
                "could not clone a memory descriptor");

        return memory::desc(cloned_md);
    }

    /// Returns a source memory descriptor.
    /// @param idx Source index.
    /// @returns Source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     source parameter with index @p idx.
    memory::desc src_desc(int idx) const {
        return query_md(query::src_md, idx);
    }

    /// Returns a destination memory descriptor.
    /// @param idx Destination index.
    /// @returns Destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     destination parameter with index @p idx.
    memory::desc dst_desc(int idx) const {
        return query_md(query::dst_md, idx);
    }

    /// Returns a weights memory descriptor.
    /// @param idx Weights index.
    /// @returns Weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     weights parameter with index @p idx.
    memory::desc weights_desc(int idx) const {
        return query_md(query::weights_md, idx);
    }

    /// Returns a diff source memory descriptor.
    /// @param idx Diff source index.
    /// @returns Diff source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff source parameter with index @p idx.
    memory::desc diff_src_desc(int idx) const {
        return query_md(query::diff_src_md, idx);
    }

    /// Returns a diff destination memory descriptor.
    /// @param idx Diff destination index.
    /// @returns Diff destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff destination parameter with index @p idx.
    memory::desc diff_dst_desc(int idx) const {
        return query_md(query::diff_dst_md, idx);
    }

    /// Returns a diff weights memory descriptor.
    /// @param idx Diff weights index.
    /// @returns Diff weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff weights parameter with index @p idx.
    memory::desc diff_weights_desc(int idx) const {
        return query_md(query::diff_weights_md, idx);
    }

    // Separate versions without the index argument for documentation
    // purposes.

    /// Returns a source memory descriptor.
    /// @returns Source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     source parameter.
    memory::desc src_desc() const { return src_desc(0); }

    /// Returns a destination memory descriptor.
    /// @returns Destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     destination parameter.
    memory::desc dst_desc() const { return dst_desc(0); }

    /// Returns a weights memory descriptor.
    /// @returns Weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     weights parameter.
    memory::desc weights_desc() const { return weights_desc(0); }

    /// Returns a diff source memory descriptor.
    /// @returns Diff source memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff source memory with.
    memory::desc diff_src_desc() const { return diff_src_desc(0); }

    /// Returns a diff destination memory descriptor.
    /// @returns Diff destination memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff destination parameter.
    memory::desc diff_dst_desc() const { return diff_dst_desc(0); }

    /// Returns a diff weights memory descriptor.
    /// @returns Diff weights memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///     diff weights parameter.
    memory::desc diff_weights_desc() const { return diff_weights_desc(0); }

    /// Returns the workspace memory descriptor.
    /// @returns Workspace memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not require
    ///     workspace parameter.
    memory::desc workspace_desc() const {
        return query_md(query::workspace_md, 0);
    }

    /// Returns the scratchpad memory descriptor.
    /// @returns scratchpad memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not require
    ///     scratchpad parameter.
    /// @sa @ref dev_guide_attributes_scratchpad
    memory::desc scratchpad_desc() const {
        return query_md(query::scratchpad_md, 0);
    }

    /// Returns the engine on which the scratchpad memory is located.
    /// @returns The engine on which the scratchpad memory is located.
    engine scratchpad_engine() const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_primitive_desc_query(get(),
                                  dnnl::convert_to_c(query::scratchpad_engine),
                                  0, &c_engine),
                "could not retrieve scratchpad engine from a primitive "
                "descriptor");
        return engine(c_engine, true);
    }

    /// Returns the primitive attributes.
    /// @returns The primitive attributes.
    primitive_attr get_primitive_attr() const {
        const_dnnl_primitive_attr_t const_c_attr;
        error::wrap_c_api(dnnl_primitive_desc_get_attr(get(), &const_c_attr),
                "could not get attributes from a primitive descriptor");
        dnnl_primitive_attr_t c_attr;
        error::wrap_c_api(dnnl_primitive_attr_clone(&c_attr, const_c_attr),
                "could not clone primitive attributes");
        return primitive_attr(c_attr);
    }

    /// Returns the kind of the primitive descriptor.
    /// @returns The kind of the primitive descriptor.
    dnnl::primitive::kind get_kind() const {
        dnnl_primitive_kind_t kind;
        error::wrap_c_api(dnnl_primitive_desc_query(get(),
                                  dnnl_query_primitive_kind, 0, (void *)&kind),
                "could not get primitive kind from a primitive descriptor");
        return static_cast<dnnl::primitive::kind>(kind);
    }

    /// Returns the cache blob ID of the primitive descriptor.
    /// @returns The cache blob ID of the primitive descriptor.
    std::vector<uint8_t> get_cache_blob_id() const {
        dnnl_dim_t count;
        const uint8_t *c_id;
        error::wrap_c_api(
                dnnl_primitive_desc_query(get(),
                        dnnl::convert_to_c(query::cache_blob_id_size_s64), 0,
                        (void *)&count),
                "could not get size of cache blob ID from a primitive "
                "descriptor");
        error::wrap_c_api(dnnl_primitive_desc_query(get(),
                                  dnnl::convert_to_c(query::cache_blob_id), 0,
                                  (void **)&c_id),
                "could not get cache blob ID from a primitive descriptor");
        std::vector<uint8_t> id(c_id, c_id + count);
        return id;
    }

protected:
    /// Returns a float value.
    /// @param what The value to query.
    /// @returns The result of the query.
    /// @returns Zero if the primitive doesn't support the query.
    float query_f32(query what) const {
        float res;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl::convert_to_c(what), 0, &res);
        return status == dnnl_success ? res : 0.0f;
    }

    /// Returns an #dnnl::algorithm value.
    /// @param what The value to query.
    /// @returns The result of the query.
    /// @returns #dnnl::algorithm::undef if the primitive doesn't support
    ///     the query.
    algorithm query_alg(query what) const {
        dnnl_alg_kind_t res;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl::convert_to_c(what), 0, &res);
        return status == dnnl_success ? static_cast<dnnl::algorithm>(res)
                                      : algorithm::undef;
    }

    /// Returns a memory::dims value.
    /// @param what The value to query.
    /// @returns The result of the query.
    /// @returns An empty #dnnl::memory::dims if the primitive doesn't support
    ///     the query.
    memory::dims query_dims(query what) const {
        const bool is_backward = get_prop_kind() != prop_kind::forward_training
                && get_prop_kind() != prop_kind::forward_inference;
        const_dnnl_memory_desc_t md = dnnl_primitive_desc_query_md(get(),
                is_backward ? dnnl_query_diff_dst_md : dnnl_query_dst_md, 0);

        int nspatial_dims = 0;
        if (md) {
            int ndims;
            error::wrap_c_api(
                    dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &ndims),
                    "could not query ndims from a memory descriptor");
            nspatial_dims = ndims - 2;
        }

        dnnl_dims_t *c_dims;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl::convert_to_c(what), 0, &c_dims);
        return status == dnnl_success
                ? memory::dims(*c_dims, *c_dims + nspatial_dims)
                : memory::dims {};
    }

    /// Returns an #dnnl::engine value.
    /// @param what The value to query.
    /// @returns The result of the query.
    /// @returns A weak handle to the engine that the primitive descriptor was
    ///     created with.
    engine query_engine(query what) const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_primitive_desc_query(get(),
                                  dnnl::convert_to_c(what), 0, &c_engine),
                "could not get an engine from a primitive_desc");
        return engine(c_engine, true);
    }

    /// Resets the value of the handle to a clone of a C API primitive
    /// descriptor.
    /// @param pd A C API primitive descriptor to clone.
    void reset_with_clone(const_dnnl_primitive_desc_t pd) {
        dnnl_primitive_desc_t new_pd;
        error::wrap_c_api(dnnl_primitive_desc_clone(&new_pd, pd),
                "could not clone a primitive descriptor");
        reset(new_pd);
    }

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     The @p prim_kind should map to a primitive that does not have
    ///     different values of propagation kind (e.g. #dnnl::binary).
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    primitive_desc_base(
            dnnl_primitive_desc_t pd, dnnl::primitive::kind prim_kind)
        : primitive_desc_base(pd, prim_kind, dnnl::prop_kind::undef) {}

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    /// @param aprop_kind Expected propagation kind.
    primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::primitive::kind prim_kind, dnnl::prop_kind aprop_kind)
        : primitive_desc_base(pd, prim_kind, aprop_kind, aprop_kind) {}

    /// Constructs a primitive descriptor base object from a clone of a C API
    /// primitive descriptor after verifying that it is what the caller
    /// expects.
    ///
    /// @note
    ///     Primitive descriptor base constructed this way does not support
    ///     next_impl() (will throw).
    ///
    /// @param pd C API primitive descriptor to clone.
    /// @param prim_kind Expected primitive kind.
    /// @param prop_kind1 Expected propagation kind (option 1).
    /// @param prop_kind2 Expected propagation kind (option 2). This value is
    ///     checked if the check with @p prop_kind1 fails.
    primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::primitive::kind prim_kind, dnnl::prop_kind prop_kind1,
            dnnl::prop_kind prop_kind2) {
        // It is OK to pass an empty primitive descriptor
        if (pd == nullptr) return;

        dnnl_status_t rc;

        dnnl_primitive_kind_t c_prim_kind = convert_to_c(prim_kind);
        dnnl_prop_kind_t c_prop_kind1 = convert_to_c(prop_kind1);
        dnnl_prop_kind_t c_prop_kind2 = convert_to_c(prop_kind2);

        // Check that primitive kind matches
        dnnl_primitive_kind_t pd_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_primitive_kind, 0, (void *)&pd_kind);
        error::wrap_c_api(
                rc, "could not get primitive kind from a primitive descriptor");
        if (pd_kind != c_prim_kind)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "primitive descriptor operation kind mismatch");

        // Check that propagation kind matches
        dnnl_prop_kind_t pd_prop_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_prop_kind, 0, (void *)&pd_prop_kind);

        // Something went wrong
        if (rc != dnnl_success && rc != dnnl_unimplemented)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "could not get propagation kind from the primitive "
                    "descriptor");

        // Everything is fine
        if ((rc == dnnl_unimplemented && c_prop_kind1 == dnnl_prop_kind_undef)
                || (rc == dnnl_success
                        && (pd_prop_kind == c_prop_kind1
                                || pd_prop_kind == c_prop_kind2))) {
            reset_with_clone(pd);
            return;
        }

        // We could get the propagation kind but there is a mismatch
        DNNL_THROW_ERROR(dnnl_invalid_arguments,
                "primitive descriptor propagation kind mismatch");
    }

    /// Returns a constant reference to a static instance of default constructed
    /// primitive attributes
    static const primitive_attr &default_attr() {
        static const primitive_attr attr;
        return attr;
    }

    const_dnnl_memory_desc_t optional_arg(const memory::desc *md) {
        return md ? md->get() : nullptr;
    }

    const dnnl_dim_t *optional_arg(const memory::dims *dims) {
        return dims ? dims->data() : nullptr;
    }

    const float *optional_arg(const std::vector<float> *arg) {
        return arg ? arg->data() : nullptr;
    }

    using base = primitive_desc_base;
};

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_reorder Reorder
///
/// A primitive to copy data between two memory objects. This primitive is
/// typically used to change the way the data is laid out in memory.
///
/// @sa @ref dev_guide_reorder in developer guide
///
/// @{

/// Reorder primitive.
struct reorder : public primitive {
    /// Primitive descriptor for a reorder primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for reorder primitive.
        ///
        /// @note
        ///     If @p allow_empty is true, the constructor does not throw if a
        ///     primitive descriptor cannot be created.
        ///
        /// @param src_engine Engine on which the source memory object will be
        ///     located.
        /// @param src_md Source memory descriptor.
        /// @param dst_engine Engine on which the destination memory object
        ///     will be located.
        /// @param dst_md Destination memory descriptor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is allowed
        ///     to fail without throwing an exception. In this case an empty
        ///     object will be produced. This flag is optional and defaults to
        ///     false.
        primitive_desc(const engine &src_engine, const memory::desc &src_md,
                const engine &dst_engine, const memory::desc &dst_md,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t result;
            dnnl_status_t status = dnnl_reorder_primitive_desc_create(&result,
                    src_md.get(), src_engine.get(), dst_md.get(),
                    dst_engine.get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a reorder "
                        "primitive");
            reset(status == dnnl_success ? result : dnnl_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for reorder primitive.
        ///
        /// @param src Source memory object. It is used to obtain the source
        ///     memory descriptor and engine.
        /// @param dst Destination memory object. It is used to obtain the
        ///     destination memory descriptor and engine.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is allowed
        ///     to fail without throwing an exception. In this case an empty
        ///     object will be produced. This flag is optional and defaults to
        ///     false.
        primitive_desc(const memory &src, const memory &dst,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t result;
            auto src_md = src.get_desc();
            auto dst_md = dst.get_desc();
            dnnl_status_t status = dnnl_reorder_primitive_desc_create(&result,
                    src_md.get(), src.get_engine().get(), dst_md.get(),
                    dst.get_engine().get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a reorder "
                        "primitive");
            reset(status == dnnl_success ? result : dnnl_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for reorder primitive from a C
        /// API primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for reorder primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::reorder) {}

        /// Returns the engine on which the source memory is allocated.
        /// @returns The engine on which the source memory is allocated.
        engine get_src_engine() const {
            return query_engine(dnnl::query::reorder_src_engine);
        }

        /// Returns the engine on which the destination memory is allocated.
        /// @returns The engine on which the destination memory is allocated.
        engine get_dst_engine() const {
            return query_engine(dnnl::query::reorder_dst_engine);
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    reorder() = default;

    /// Constructs a reorder primitive.
    /// @param pd Primitive descriptor for reorder primitive.
    reorder(const primitive_desc &pd) : primitive(pd.get()) {}

    /// Constructs a reorder primitive from a cache blob.
    /// @param pd Primitive descriptor for reorder primitive.
    /// @param cache_blob Cache blob.
    reorder(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd.get(), cache_blob) {}

    /// Constructs a reorder primitive that would reorder data between memory
    /// objects having the same memory descriptors as memory objects @p src and
    /// @p dst.
    ///
    /// @param src Source memory object.
    /// @param dst Destination memory object.
    /// @param attr Primitive attributes to use (optional).
    reorder(const memory &src, const memory &dst,
            const primitive_attr &attr = primitive_attr())
        : primitive(primitive_desc(src, dst, attr).get()) {}

    using primitive::execute;

    /// Executes the reorder primitive.
    ///
    /// @param astream Stream object. The stream must belong to the same engine
    ///     as the primitive.
    /// @param src Source memory object.
    /// @param dst Destination memory object.
    void execute(const stream &astream, memory &src, memory &dst) const {
        primitive::execute(astream, {{DNNL_ARG_FROM, src}, {DNNL_ARG_TO, dst}});
    }
};

/// @} dnnl_api_reorder

/// @addtogroup dnnl_api_concat Concat
///
/// A primitive to concatenate data by arbitrary dimension.
///
/// @sa @ref dev_guide_concat in developer guide
///
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
inline std::vector<const_dnnl_memory_desc_t> convert_to_c(
        const std::vector<memory::desc> &mds) {
    std::vector<const_dnnl_memory_desc_t> c_mds;
    c_mds.reserve(mds.size());
    for (const auto &md : mds)
        c_mds.push_back(md.get());
    return c_mds;
}
/// @endcond

/// Tensor concatenation (concat) primitive.
struct concat : public primitive {
    /// Primitive descriptor for a concat primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an out-of-place concatenation
        /// primitive.
        ///
        /// @param aengine Engine to perform the operation on.
        /// @param dst Destination memory descriptor.
        /// @param concat_dimension Source tensors will be concatenated over
        ///     dimension with this index. Note that order of dimensions does
        ///     not depend on memory format.
        /// @param srcs Vector of source memory descriptors.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &dst,
                int concat_dimension, const std::vector<memory::desc> &srcs,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            auto c_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            dnnl_status_t status = dnnl_concat_primitive_desc_create(&result,
                    aengine.get(), dst.get(), (int)c_srcs.size(),
                    concat_dimension, c_srcs.data(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a concat "
                        "primitive");
            reset(status == dnnl_success ? result : dnnl_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for an out-of-place concatenation
        /// primitive.
        ///
        /// This version derives the destination memory descriptor
        /// automatically.
        ///
        /// @param aengine Engine to perform the operation on.
        /// @param concat_dimension Source tensors will be concatenated over
        ///     dimension with this index. Note that order of dimensions does
        ///     not depend on memory format.
        /// @param srcs Vector of source memory descriptors.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, int concat_dimension,
                const std::vector<memory::desc> &srcs,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            auto c_api_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            dnnl_status_t status = dnnl_concat_primitive_desc_create(&result,
                    aengine.get(), nullptr, (int)c_api_srcs.size(),
                    concat_dimension, c_api_srcs.data(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a concat "
                        "primitive");
            reset(status == dnnl_success ? result : dnnl_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for concat primitive from a C
        /// API primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for concat primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::concat) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    concat() = default;

    /// Constructs a concatenation primitive.
    /// @param pd Primitive descriptor for concatenation primitive.
    concat(const primitive_desc &pd) : primitive(pd.get()) {}

    /// Constructs a concatenation primitive from a cache blob.
    /// @param pd Primitive descriptor for concatenation primitive.
    /// @param cache_blob Cache blob.
    concat(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd.get(), cache_blob) {}
};

/// @} dnnl_api_concat

/// @addtogroup dnnl_api_sum Sum
///
/// A primitive to sum multiple tensors.
///
/// @sa @ref dev_guide_sum in developer guide
///
/// @{

/// Out-of-place summation (sum) primitive.
struct sum : public primitive {
    /// Primitive descriptor for a sum primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a sum primitive.
        ///
        /// @param aengine Engine to perform the operation on.
        /// @param dst Destination memory descriptor.
        /// @param scales Vector of scales to multiply data in each source
        ///     memory by.
        /// @param srcs Vector of source memory descriptors.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &dst,
                const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            validate_container_size(scales,
                    "counts of scales and sources are not equal",
                    (int)srcs.size(), (int)srcs.size());

            auto c_api_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            dnnl_status_t status = dnnl_sum_primitive_desc_create(&result,
                    aengine.get(), dst.get(), (int)c_api_srcs.size(),
                    scales.data(), c_api_srcs.data(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a sum "
                        "primitive");
            reset(status == dnnl_success ? result : dnnl_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for a sum primitive.
        ///
        /// This version derives the destination memory descriptor
        /// automatically.
        ///
        /// @param aengine Engine on which to perform the operation.
        /// @param scales Vector of scales by which to multiply data in each
        ///     source memory object.
        /// @param srcs Vector of source memory descriptors.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            validate_container_size(scales,
                    "counts of scales and sources are not equal",
                    (int)srcs.size(), (int)srcs.size());

            auto c_api_srcs = convert_to_c(srcs);
            dnnl_primitive_desc_t result;
            dnnl_status_t status = dnnl_sum_primitive_desc_create(&result,
                    aengine.get(), nullptr, (int)c_api_srcs.size(),
                    scales.data(), c_api_srcs.data(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a sum "
                        "primitive");
            reset(status == dnnl_success ? result : dnnl_primitive_desc_t());
        }

        /// Constructs a primitive descriptor for sum primitive from a C API
        /// primitive descriptor which must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for sum primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::sum) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    sum() = default;

    /// Constructs a sum primitive.
    /// @param pd Primitive descriptor for sum primitive.
    sum(const primitive_desc &pd) : primitive(pd.get()) {}

    /// Constructs a sum primitive from a cache blob.
    /// @param pd Primitive descriptor for sum primitive.
    /// @param cache_blob Cache blob.
    sum(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd.get(), cache_blob) {}
};

/// @} dnnl_api_sum

/// @addtogroup dnnl_api_primitives_common
/// @{

/// A base class for descriptors of all primitives that support iteration
///     over multiple implementations.
struct primitive_desc : public primitive_desc_base {
    using primitive_desc_base::primitive_desc_base;

    primitive_desc() = default;

    /// Changes the primitive descriptor to point to the next available
    /// implementation.
    ///
    /// @returns @c true on success and @c false if the last available
    /// implementation has already been reached. In the latter case, the
    /// primitive descriptor itself is kept unchanged.
    bool next_impl() {
        dnnl_status_t status = dnnl_primitive_desc_next_impl(get());
        if (status == dnnl_last_impl_reached) return false;
        error::wrap_c_api(status, "last available implementation is reached");
        return true;
    }
};

/// @} dnnl_api_primitives_common

/// @addtogroup dnnl_api_convolution Convolution
///
/// A primitive to perform 1D, 2D or 3D convolution. Supported variants are
/// forward propagation, backward propagation, and weights gradient with or
/// without bias.
///
/// @sa @ref dev_guide_convolution in developer guide
///
/// @{

/// Convolution forward propagation primitive.
struct convolution_forward : public primitive {
    /// Primitive descriptor for a convolution forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution forward
        ///     propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &bias_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    weights_desc, &bias_desc, dst_desc, strides, nullptr,
                    padding_l, padding_r, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution forward
        ///     propagation primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    weights_desc, nullptr, dst_desc, strides, nullptr,
                    padding_l, padding_r, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution forward
        ///     propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &bias_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    weights_desc, &bias_desc, dst_desc, strides, &dilates,
                    padding_l, padding_r, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution forward
        ///     propagation primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &dilates,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    weights_desc, nullptr, dst_desc, strides, &dilates,
                    padding_l, padding_r, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// Returns the bias memory descriptor.
        /// @returns The bias memory descriptor.
        /// @returns A zero memory descriptor of the primitive does not have a
        ///     bias parameter.
        memory::desc bias_desc() const { return base::weights_desc(1); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_strides()const
        memory::dims get_strides() const { return base::get_strides(); }

        /// @copydoc dnnl::primitive_desc_base::get_dilations()const
        memory::dims get_dilations() const { return base::get_dilations(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_l()const
        memory::dims get_padding_l() const { return base::get_padding_l(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_r()const
        memory::dims get_padding_r() const { return base::get_padding_r(); }

    private:
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc *bias_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims *dilates, const memory::dims &padding_l,
                const memory::dims &padding_r, const primitive_attr &attr,
                bool allow_empty) {

            memory::validate_dims(strides, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_l, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_r, src_desc.get_ndims() - 2);

            if (dilates)
                memory::validate_dims(*dilates, src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_convolution_forward_primitive_desc_create(&pd,
                            aengine.get(), dnnl::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm), src_desc.get(),
                            weights_desc.get(), optional_arg(bias_desc),
                            dst_desc.get(), &strides[0], optional_arg(dilates),
                            &padding_l[0], &padding_r[0], attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "convolution forward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    convolution_forward() = default;

    /// Constructs a convolution forward propagation primitive.
    /// @param pd Primitive descriptor for a convolution forward propagation
    ///     primitive.
    convolution_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a convolution forward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a convolution forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    convolution_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Convolution backward propagation primitive.
struct convolution_backward_data : public primitive {
    /// Primitive descriptor for a convolution backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution backward
        ///     propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a convolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, diff_src_desc, weights_desc,
                    diff_dst_desc, strides, nullptr, padding_l, padding_r,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution backward
        ///     propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a convolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, diff_src_desc, weights_desc,
                    diff_dst_desc, strides, &dilates, padding_l, padding_r,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_strides()const
        memory::dims get_strides() const { return base::get_strides(); }

        /// @copydoc dnnl::primitive_desc_base::get_dilations()const
        memory::dims get_dilations() const { return base::get_dilations(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_l()const
        memory::dims get_padding_l() const { return base::get_padding_l(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_r()const
        memory::dims get_padding_r() const { return base::get_padding_r(); }

    private:
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims *dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr, bool allow_empty) {

            memory::validate_dims(strides, diff_src_desc.get_ndims() - 2);
            memory::validate_dims(padding_l, diff_src_desc.get_ndims() - 2);
            memory::validate_dims(padding_r, diff_src_desc.get_ndims() - 2);

            if (dilates)
                memory::validate_dims(*dilates, diff_src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_convolution_backward_data_primitive_desc_create(&pd,
                            aengine.get(), convert_to_c(aalgorithm),
                            diff_src_desc.get(), weights_desc.get(),
                            diff_dst_desc.get(), &strides[0],
                            optional_arg(dilates), &padding_l[0], &padding_r[0],
                            hint_fwd_pd.get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "convolution backward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    convolution_backward_data() = default;

    /// Constructs a convolution backward propagation primitive.
    /// @param pd Primitive descriptor for a convolution backward propagation
    ///     primitive.
    convolution_backward_data(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a convolution backward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a convolution backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    convolution_backward_data(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Convolution weights gradient primitive.
struct convolution_backward_weights : public primitive {
    /// Primitive descriptor for a convolution weights gradient primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution weights gradient
        ///     primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a convolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                    &diff_bias_desc, diff_dst_desc, strides, nullptr, padding_l,
                    padding_r, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights gradient
        ///     primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a convolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                    nullptr, diff_dst_desc, strides, nullptr, padding_l,
                    padding_r, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights
        ///     gradient primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a convolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                    &diff_bias_desc, diff_dst_desc, strides, &dilates,
                    padding_l, padding_r, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights
        ///     gradient primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Convolution algorithm. Possible values are
        ///     #dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd, and
        ///     #dnnl::algorithm::convolution_auto.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a convolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                    nullptr, diff_dst_desc, strides, &dilates, padding_l,
                    padding_r, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights gradient
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a convolution weights
        ///     gradient primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// Returns the diff bias memory descriptor.
        /// @returns The diff bias memory descriptor.
        /// @returns A zero memory descriptor of the primitive does not have a
        ///          diff bias parameter.
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_strides()const
        memory::dims get_strides() const { return base::get_strides(); }

        /// @copydoc dnnl::primitive_desc_base::get_dilations()const
        memory::dims get_dilations() const { return base::get_dilations(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_l()const
        memory::dims get_padding_l() const { return base::get_padding_l(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_r()const
        memory::dims get_padding_r() const { return base::get_padding_r(); }

    private:
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc *diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims *dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr, bool allow_empty) {

            memory::validate_dims(strides, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_l, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_r, src_desc.get_ndims() - 2);

            if (dilates)
                memory::validate_dims(*dilates, src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_convolution_backward_weights_primitive_desc_create(
                            &pd, aengine.get(), convert_to_c(aalgorithm),
                            src_desc.get(), diff_weights_desc.get(),
                            optional_arg(diff_bias_desc), diff_dst_desc.get(),
                            &strides[0], optional_arg(dilates), &padding_l[0],
                            &padding_r[0], hint_fwd_pd.get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "convolution weights update primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    convolution_backward_weights() = default;

    /// Constructs a convolution weights gradient primitive.
    /// @param pd Primitive descriptor for a convolution weights gradient
    ///     primitive.
    convolution_backward_weights(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a convolution weights gradient primitive from a cache blob.
    /// @param pd Primitive descriptor for a convolution weights gradient
    ///     primitive.
    /// @param cache_blob Cache blob.
    convolution_backward_weights(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_convolution
//
/// @addtogroup dnnl_api_deconvolution Deconvolution
///
/// A primitive to perform 1D, 2D or 3D deconvolution. Supported variants are
/// forward propagation, backward propagation, and weights gradient with or
/// without bias.
///
/// @{

/// Deconvolution forward propagation primitive.
struct deconvolution_forward : public primitive {
    /// Primitive descriptor for a deconvolution forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution forward
        ///     propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &bias_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    weights_desc, &bias_desc, dst_desc, strides, nullptr,
                    padding_l, padding_r, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        ///     propagation primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    weights_desc, nullptr, dst_desc, strides, nullptr,
                    padding_l, padding_r, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        ///     propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param bias_desc Bias memory descriptor. Passing zero memory
        ///     descriptor disables the bias term.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &bias_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    weights_desc, &bias_desc, dst_desc, strides, &dilates,
                    padding_l, padding_r, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        ///     propagation primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Deconvolution algorithm:
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &dilates,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    weights_desc, nullptr, dst_desc, strides, &dilates,
                    padding_l, padding_r, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const { return base::weights_desc(1); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_strides()const
        memory::dims get_strides() const { return base::get_strides(); }

        /// @copydoc dnnl::primitive_desc_base::get_dilations()const
        memory::dims get_dilations() const { return base::get_dilations(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_l()const
        memory::dims get_padding_l() const { return base::get_padding_l(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_r()const
        memory::dims get_padding_r() const { return base::get_padding_r(); }

    private:
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc *bias_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims *dilates, const memory::dims &padding_l,
                const memory::dims &padding_r, const primitive_attr &attr,
                bool allow_empty) {

            memory::validate_dims(strides, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_l, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_r, src_desc.get_ndims() - 2);

            if (dilates)
                memory::validate_dims(*dilates, src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_deconvolution_forward_primitive_desc_create(&pd,
                            aengine.get(), dnnl::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm), src_desc.get(),
                            weights_desc.get(), optional_arg(bias_desc),
                            dst_desc.get(), &strides[0], optional_arg(dilates),
                            &padding_l[0], &padding_r[0], attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "deconvolution forward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_forward() = default;

    /// Constructs a deconvolution forward propagation primitive.
    /// @param pd Primitive descriptor for a deconvolution forward propagation
    ///     primitive.
    deconvolution_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a deconvolution forward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a deconvolution forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    deconvolution_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Deconvolution backward propagation primitive.
struct deconvolution_backward_data : public primitive {
    /// Primitive descriptor for a deconvolution backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution backward
        ///     propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Deconvolution algorithm
        ///     (#dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, diff_src_desc, weights_desc,
                    diff_dst_desc, strides, nullptr, padding_l, padding_r,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution backward
        ///     propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Deconvolution algorithm
        ///     (#dnnl::algorithm::convolution_direct,
        ///     #dnnl::algorithm::convolution_winograd).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param weights_desc Weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, diff_src_desc, weights_desc,
                    diff_dst_desc, strides, &dilates, padding_l, padding_r,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_strides()const
        memory::dims get_strides() const { return base::get_strides(); }

        /// @copydoc dnnl::primitive_desc_base::get_dilations()const
        memory::dims get_dilations() const { return base::get_dilations(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_l()const
        memory::dims get_padding_l() const { return base::get_padding_l(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_r()const
        memory::dims get_padding_r() const { return base::get_padding_r(); }

    private:
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims *dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr, bool allow_empty) {

            memory::validate_dims(strides, diff_src_desc.get_ndims() - 2);
            memory::validate_dims(padding_l, diff_src_desc.get_ndims() - 2);
            memory::validate_dims(padding_r, diff_src_desc.get_ndims() - 2);

            if (dilates)
                memory::validate_dims(*dilates, diff_src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_deconvolution_backward_data_primitive_desc_create(
                            &pd, aengine.get(), convert_to_c(aalgorithm),
                            diff_src_desc.get(), weights_desc.get(),
                            diff_dst_desc.get(), &strides[0],
                            optional_arg(dilates), &padding_l[0], &padding_r[0],
                            hint_fwd_pd.get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "deconvolution backward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_backward_data() = default;

    /// Constructs a deconvolution backward propagation primitive.
    /// @param pd Primitive descriptor for a deconvolution backward propagation
    ///     primitive.
    deconvolution_backward_data(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a deconvolution backward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a deconvolution backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    deconvolution_backward_data(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Deconvolution weights gradient primitive.
struct deconvolution_backward_weights : public primitive {
    /// Primitive descriptor for a deconvolution weights gradient primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution weights
        ///     gradient primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                    &diff_bias_desc, diff_dst_desc, strides, nullptr, padding_l,
                    padding_r, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        ///     gradient primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p padding_l, and @p padding_r contain values
        /// for spatial dimensions only and hence must have the same number of
        /// elements as there are spatial dimensions. The order of values is
        /// the same as in the tensor: depth (for 3D tensors), height (for 3D
        /// and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                    nullptr, diff_dst_desc, strides, nullptr, padding_l,
                    padding_r, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        ///     gradient primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_bias_desc Diff bias memory descriptor. Passing zero
        ///     memory descriptor disables the bias term.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                    &diff_bias_desc, diff_dst_desc, strides, &dilates,
                    padding_l, padding_r, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        ///     gradient primitive without bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// Arrays @p strides, @p dilates, @p padding_l, and @p padding_r
        /// contain values for spatial dimensions only and hence must have the
        /// same number of elements as there are spatial dimensions. The order
        /// of values is the same as in the tensor: depth (for 3D tensors),
        /// height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Deconvolution algorithm. Possible values are
        ///     #dnnl::algorithm::deconvolution_direct, and
        ///     #dnnl::algorithm::deconvolution_winograd.
        /// @param src_desc Source memory descriptor.
        /// @param diff_weights_desc Diff weights memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Strides for each spatial dimension.
        /// @param dilates Dilations for each spatial dimension. A zero value
        ///     means no dilation in the corresponding dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, src_desc, diff_weights_desc,
                    nullptr, diff_dst_desc, strides, &dilates, padding_l,
                    padding_r, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        /// gradient primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a deconvolution weights
        ///     gradient primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::convolution_backward_weights::primitive_desc::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_strides()const
        memory::dims get_strides() const { return base::get_strides(); }

        /// @copydoc dnnl::primitive_desc_base::get_dilations()const
        memory::dims get_dilations() const { return base::get_dilations(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_l()const
        memory::dims get_padding_l() const { return base::get_padding_l(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_r()const
        memory::dims get_padding_r() const { return base::get_padding_r(); }

    private:
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc *diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims *dilates, const memory::dims &padding_l,
                const memory::dims &padding_r,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr, bool allow_empty) {

            memory::validate_dims(strides, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_l, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_r, src_desc.get_ndims() - 2);

            if (dilates)
                memory::validate_dims(*dilates, src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_deconvolution_backward_weights_primitive_desc_create(
                            &pd, aengine.get(), convert_to_c(aalgorithm),
                            src_desc.get(), diff_weights_desc.get(),
                            optional_arg(diff_bias_desc), diff_dst_desc.get(),
                            &strides[0], optional_arg(dilates), &padding_l[0],
                            &padding_r[0], hint_fwd_pd.get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "deconvolution weights update primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_backward_weights() = default;

    /// Constructs a deconvolution weights gradient primitive.
    /// @param pd Primitive descriptor for a deconvolution weights gradient
    ///     primitive.
    deconvolution_backward_weights(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a deconvolution weights gradient primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a deconvolution weights gradient
    ///     primitive.
    /// @param cache_blob Cache blob.
    deconvolution_backward_weights(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_deconvolution

/// @addtogroup dnnl_api_lrn LRN
///
/// A primitive to perform local response normalization (LRN) across or within
/// channels.
///
/// @sa @ref dev_guide_lrn in developer guide
///
/// @{

/// Local response normalization (LRN) forward propagation primitive.
struct lrn_forward : public primitive {
    /// Primitive descriptor for an LRN forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LRN forward propagation
        ///     primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm LRN algorithm kind: either
        ///     #dnnl::algorithm::lrn_across_channels, or
        ///     #dnnl::algorithm::lrn_within_channel.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param local_size Regularization local size.
        /// @param alpha The alpha regularization parameter.
        /// @param beta The beta regularization parameter.
        /// @param k The k regularization parameter.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc, memory::dim local_size,
                float alpha, float beta, float k,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_lrn_forward_primitive_desc_create(&pd,
                    aengine.get(), dnnl::convert_to_c(aprop_kind),
                    convert_to_c(aalgorithm), src_desc.get(), dst_desc.get(),
                    local_size, alpha, beta, k, attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a lrn "
                        "forward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LRN forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::lrn,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_alpha()const
        float get_alpha() const { return base::get_alpha(); }

        /// @copydoc dnnl::primitive_desc_base::get_beta()const
        float get_beta() const { return base::get_beta(); }

        /// @copydoc dnnl::primitive_desc_base::get_local_size()const
        memory::dim get_local_size() const { return base::get_local_size(); }

        /// @copydoc dnnl::primitive_desc_base::get_k()const
        float get_k() const { return base::get_k(); }
    };

    /// Default constructor. Produces an empty object.
    lrn_forward() = default;

    /// Constructs an LRN forward propagation primitive.
    /// @param pd Primitive descriptor for an LRN forward propagation
    ///     primitive.
    lrn_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LRN forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LRN forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lrn_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Local response normalization (LRN) backward propagation primitive.
struct lrn_backward : public primitive {
    /// Primitive descriptor for an LRN backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LRN backward propagation
        ///     primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm LRN algorithm kind: either
        ///     #dnnl::algorithm::lrn_across_channels, or
        ///     #dnnl::algorithm::lrn_within_channel.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param src_desc Source memory descriptor.
        /// @param local_size Regularization local size.
        /// @param alpha The alpha regularization parameter.
        /// @param beta The beta regularization parameter.
        /// @param k The k regularization parameter.
        /// @param hint_fwd_pd Primitive descriptor for an LRN forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &src_desc,
                memory::dim local_size, float alpha, float beta, float k,
                const lrn_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_lrn_backward_primitive_desc_create(&pd,
                    aengine.get(), convert_to_c(aalgorithm),
                    diff_src_desc.get(), diff_dst_desc.get(), src_desc.get(),
                    local_size, alpha, beta, k, hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a lrn "
                        "backward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LRN backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::lrn,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_alpha()const
        float get_alpha() const { return base::get_alpha(); }

        /// @copydoc dnnl::primitive_desc_base::get_beta()const
        float get_beta() const { return base::get_beta(); }

        /// @copydoc dnnl::primitive_desc_base::get_local_size()const
        memory::dim get_local_size() const { return base::get_local_size(); }

        /// @copydoc dnnl::primitive_desc_base::get_k()const
        float get_k() const { return base::get_k(); }
    };

    /// Default constructor. Produces an empty object.
    lrn_backward() = default;

    /// Constructs an LRN backward propagation primitive.
    /// @param pd Primitive descriptor for an LRN backward propagation
    ///     primitive.
    lrn_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LRN backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LRN backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lrn_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_lrn

/// @addtogroup dnnl_api_eltwise Eltwise
///
/// A primitive to perform elementwise operations such as the
/// rectifier linear unit (ReLU).
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// @warning
///     Because the original source data is required for backward propagation,
///     in-place forward propagation is not generally supported in the
///     training mode. However, for algorithms supporting destination as input
///     memory, dst can be used for the backward propagation, which makes it
///     possible to get performance benefit even in the training mode.
///
/// @sa @ref dev_guide_eltwise in developer guide
///
/// @{

/// Elementwise unary operation forward propagation primitive.
struct eltwise_forward : public primitive {
    /// Primitive descriptor for an elementwise forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise forward
        ///     propagation primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    dst_desc, nullptr, nullptr, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise forward
        ///     propagation primitive with an alpha parameter.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param alpha The alpha parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc, float alpha,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    dst_desc, &alpha, nullptr, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise forward
        ///     propagation primitive with an alpha and beta parameters.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param alpha The alpha parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param beta The beta parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc, float alpha, float beta,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, src_desc,
                    dst_desc, &alpha, &beta, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an eltwise forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an eltwise forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::eltwise,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        dnnl::algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_alpha()const
        float get_alpha() const { return base::get_alpha(); }

        /// @copydoc dnnl::primitive_desc_base::get_beta()const
        float get_beta() const { return base::get_beta(); }

    private:
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc, const float *alpha,
                const float *beta, const primitive_attr &attr,
                bool allow_empty) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_eltwise_forward_primitive_desc_create(
                    &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                    dnnl::convert_to_c(aalgorithm), src_desc.get(),
                    dst_desc.get(), alpha ? *alpha : 0.0f, beta ? *beta : 0.0f,
                    attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for an "
                        "eltwise forward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    eltwise_forward() = default;

    /// Constructs an eltwise forward propagation primitive.
    /// @param pd Primitive descriptor for an eltwise forward propagation
    ///     primitive.
    eltwise_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an eltwise forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an eltwise forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    eltwise_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Elementwise unary operation backward propagation primitive.
struct eltwise_backward : public primitive {
    /// Primitive descriptor for eltwise backward propagation.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise backward
        ///     propagation primitive with an alpha parameter.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param data_desc Destination memory descriptor if one of the
        ///     "use_dst_for_bwd" algorithms are used (such as
        ///     #dnnl_eltwise_relu_use_dst_for_bwd), source memory descriptor
        ///     otherwise.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::desc &data_desc,
                const eltwise_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, diff_src_desc, diff_dst_desc,
                    data_desc, nullptr, nullptr, hint_fwd_pd, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise backward
        ///     propagation primitive with an alpha parameter.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param data_desc Destination memory descriptor if one of the
        ///     "use_dst_for_bwd" algorithms are used (such as
        ///     #dnnl_eltwise_relu_use_dst_for_bwd), source memory descriptor
        ///     otherwise.
        /// @param alpha The alpha parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::desc &data_desc, float alpha,
                const eltwise_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, diff_src_desc, diff_dst_desc,
                    data_desc, &alpha, nullptr, hint_fwd_pd, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise backward
        ///     propagation primitive with an alpha and beta parameters.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Elementwise algorithm kind.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param data_desc Destination memory descriptor if one of the
        ///     "use_dst_for_bwd" algorithms are used (such as
        ///     #dnnl_eltwise_relu_use_dst_for_bwd), source memory descriptor
        ///     otherwise.
        /// @param alpha The alpha parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param beta The beta parameter for the elementwise operation.
        ///     Specific meaning depends on the algorithm.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::desc &data_desc, float alpha, float beta,
                const eltwise_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, diff_src_desc, diff_dst_desc,
                    data_desc, &alpha, &beta, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an eltwise backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an eltwise backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::eltwise,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        dnnl::algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_alpha()const
        float get_alpha() const { return base::get_alpha(); }

        /// @copydoc dnnl::primitive_desc_base::get_beta()const
        float get_beta() const { return base::get_beta(); }

    private:
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::desc &data_desc, const float *alpha,
                const float *beta,
                const eltwise_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr, bool allow_empty) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_eltwise_backward_primitive_desc_create(
                    &pd, aengine.get(), dnnl::convert_to_c(aalgorithm),
                    diff_src_desc.get(), diff_dst_desc.get(), data_desc.get(),
                    alpha ? *alpha : 0.0f, beta ? *beta : 0.0f,
                    hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for an "
                        "eltwise backward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    eltwise_backward() = default;

    /// Constructs an eltwise backward propagation primitive.
    /// @param pd Primitive descriptor for an eltwise backward propagation
    ///     primitive.
    eltwise_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an eltwise backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an eltwise backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    eltwise_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_eltwise

/// @addtogroup dnnl_api_softmax Softmax
///
/// A primitive to perform softmax.
///
/// @sa @ref dev_guide_softmax in developer guide
///
/// @{

/// Softmax forward propagation primitive.
struct softmax_forward : public primitive {
    /// Primitive descriptor for a softmax forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax forward propagation
        /// primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Softmax algorithm kind: either
        ///     #dnnl::algorithm::softmax_accurate,
        ///     or #dnnl::algorithm::softmax_log.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param axis Axis over which softmax is computed.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc, int axis,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_softmax_forward_primitive_desc_create(
                    &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                    dnnl::convert_to_c(aalgorithm), src_desc.get(),
                    dst_desc.get(), axis, attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a softmax "
                        "forward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a softmax forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        dnnl::algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_axis()const
        int get_axis() const { return base::get_axis(); }
    };

    /// Default constructor. Produces an empty object.
    softmax_forward() = default;

    /// Constructs a softmax forward propagation primitive.
    /// @param pd Primitive descriptor for a softmax forward propagation
    ///     primitive.
    softmax_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a softmax forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a softmax forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    softmax_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Softmax backward propagation primitive.
struct softmax_backward : public primitive {
    /// Primitive descriptor for a softmax backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax backward propagation
        /// primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Softmax algorithm kind: either
        ///     #dnnl::algorithm::softmax_accurate,
        ///     or #dnnl::algorithm::softmax_log.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param axis Axis over which softmax is computed.
        /// @param hint_fwd_pd Primitive descriptor for a softmax
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &dst_desc,
                int axis, const softmax_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_softmax_backward_primitive_desc_create(
                    &pd, aengine.get(), dnnl::convert_to_c(aalgorithm),
                    diff_src_desc.get(), diff_dst_desc.get(), dst_desc.get(),
                    axis, hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a softmax "
                        "backward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a softmax backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        dnnl::algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_axis()const
        int get_axis() const { return base::get_axis(); }
    };

    /// Default constructor. Produces an empty object.
    softmax_backward() = default;

    /// Constructs a softmax backward propagation primitive.
    /// @param pd Primitive descriptor for a softmax backward propagation
    ///     primitive.
    softmax_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a softmax backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a softmax backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    softmax_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_softmax

/// @addtogroup dnnl_api_batch_normalization Batch Normalization
///
/// A primitive to perform batch normalization.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// The batch normalization primitives computations can be controlled by
/// specifying different @ref dnnl::normalization_flags values. For example,
/// batch normalization forward propagation can be configured to either
/// compute the mean and variance or take them as arguments. It can either
/// perform scaling and shifting using gamma and beta parameters or not.
/// Optionally, it can also perform a fused ReLU, which in case of training
/// would also require a workspace.
///
/// @sa @ref dev_guide_batch_normalization in developer guide
///
/// @{

/// Batch normalization forward propagation primitive.
struct batch_normalization_forward : public primitive {
    /// Primitive descriptor for a batch normalization forward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a batch normalization forward
        /// propagation primitive.
        ///
        /// @note
        ///     In-place operation is supported: the dst can refer to the same
        ///     memory as the src.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param epsilon Batch normalization epsilon parameter.
        /// @param flags Batch normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                float epsilon, normalization_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_batch_normalization_forward_primitive_desc_create(
                            &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                            src_desc.get(), dst_desc.get(), epsilon,
                            convert_to_c(flags), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a batch "
                        "normalization forward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a batch normalization
        /// forward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a batch normalization
        ///     forward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::batch_normalization,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// Returns memory descriptor for mean.
        /// @returns Memory descriptor for mean.
        memory::desc mean_desc() const { return stat_desc(mean); }

        /// Returns memory descriptor for variance.
        /// @returns Memory descriptor for variance.
        memory::desc variance_desc() const { return stat_desc(var); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_epsilon()const
        float get_epsilon() const { return base::get_epsilon(); }

        /// Returns normalization flags.
        /// @return Normalization flags.
        normalization_flags get_flags() const {
            return base::get_flags<normalization_flags>();
        }

    private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            const bool use_global_stats
                    = (get_flags() & normalization_flags::use_global_stats)
                    != normalization_flags::none;
            return query_md(
                    use_global_stats ? query::src_md : query::dst_md, kind);
        }
    };

    /// Default constructor. Produces an empty object.
    batch_normalization_forward() = default;

    /// Constructs a batch normalization forward propagation primitive.
    /// @param pd Primitive descriptor for a batch normalization forward
    ///     propagation primitive.
    batch_normalization_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a batch normalization forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a batch normalization forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    batch_normalization_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Batch normalization backward propagation primitive.
struct batch_normalization_backward : public primitive {
    /// Primitive descriptor for a batch normalization backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a batch normalization backward
        /// propagation primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param src_desc Source memory descriptor.
        /// @param epsilon Batch normalization epsilon parameter.
        /// @param flags Batch normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param hint_fwd_pd Primitive descriptor for a batch normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &src_desc,
                float epsilon, normalization_flags flags,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_batch_normalization_backward_primitive_desc_create(
                            &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                            diff_src_desc.get(), diff_dst_desc.get(),
                            src_desc.get(), epsilon, convert_to_c(flags),
                            hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a batch "
                        "normalization backward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a batch normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a batch normalization
        ///     backward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::batch_normalization,
                    dnnl::prop_kind::backward, dnnl::prop_kind::backward_data) {
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_epsilon()const
        float get_epsilon() const { return base::get_epsilon(); }

        /// Returns normalization flags.
        /// @return Normalization flags.
        normalization_flags get_flags() const {
            return base::get_flags<normalization_flags>();
        }
    };

    /// Default constructor. Produces an empty object.
    batch_normalization_backward() = default;

    /// Constructs a batch normalization backward propagation primitive.
    /// @param pd Primitive descriptor for a batch normalization backward
    ///     propagation primitive.
    batch_normalization_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a batch normalization backward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a batch normalization backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    batch_normalization_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_batch_normalization

/// @addtogroup dnnl_api_group_normalization Group Normalization
///
/// A primitive to perform group normalization.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// The group normalization primitives computations can be controlled by
/// specifying different @ref dnnl::normalization_flags values. For example,
/// group normalization forward propagation can be configured to either
/// compute the mean and variance or take them as arguments. It can either
/// perform scaling and shifting using gamma and beta parameters or not.
///
/// @sa @ref dev_guide_group_normalization in developer guide
///
/// @{

/// Group normalization forward propagation primitive.
struct group_normalization_forward : public primitive {
    /// Primitive descriptor for a group normalization forward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a group normalization forward
        /// propagation primitive.
        ///
        /// @note
        ///     In-place operation is supported: the dst can refer to the same
        ///     memory as the src.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param groups Group normalization groups parameter.
        /// @param epsilon Group normalization epsilon parameter.
        /// @param flags Group normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                memory::dim groups, float epsilon, normalization_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_group_normalization_forward_primitive_desc_create(
                            &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                            src_desc.get(), dst_desc.get(), groups, epsilon,
                            convert_to_c(flags), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a group "
                        "normalization forward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a group normalization
        /// forward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a group normalization
        ///     forward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::group_normalization,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// Returns memory descriptor for mean.
        /// @returns Memory descriptor for mean.
        memory::desc mean_desc() const { return stat_desc(mean); }

        /// Returns memory descriptor for variance.
        /// @returns Memory descriptor for variance.
        memory::desc variance_desc() const { return stat_desc(var); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_group_size()const
        memory::dim get_group_size() const { return base::get_group_size(); }

        /// @copydoc dnnl::primitive_desc_base::get_epsilon()const
        float get_epsilon() const { return base::get_epsilon(); }

        /// Returns normalization flags.
        /// @return Normalization flags.
        normalization_flags get_flags() const {
            return base::get_flags<normalization_flags>();
        }

    private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            const bool use_global_stats
                    = (get_flags() & normalization_flags::use_global_stats)
                    != normalization_flags::none;
            return query_md(
                    use_global_stats ? query::src_md : query::dst_md, kind);
        }
    };

    /// Default constructor. Produces an empty object.
    group_normalization_forward() = default;

    /// Constructs a group normalization forward propagation primitive.
    /// @param pd Primitive descriptor for a group normalization forward
    ///     propagation primitive.
    group_normalization_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a group normalization forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a group normalization forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    group_normalization_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Group normalization backward propagation primitive.
struct group_normalization_backward : public primitive {
    /// Primitive descriptor for a group normalization backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a group normalization backward
        /// propagation primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param src_desc Source memory descriptor.
        /// @param groups Group normalization groups parameter.
        /// @param epsilon Group normalization epsilon parameter.
        /// @param flags Group normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param hint_fwd_pd Primitive descriptor for a group normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &src_desc,
                memory::dim groups, float epsilon, normalization_flags flags,
                const group_normalization_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_group_normalization_backward_primitive_desc_create(
                            &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                            diff_src_desc.get(), diff_dst_desc.get(),
                            src_desc.get(), groups, epsilon,
                            convert_to_c(flags), hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a group "
                        "normalization backward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a group normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a group normalization
        ///     backward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::group_normalization,
                    dnnl::prop_kind::backward, dnnl::prop_kind::backward_data) {
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::group_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::group_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_group_size()const
        memory::dim get_group_size() const { return base::get_group_size(); }

        /// @copydoc dnnl::primitive_desc_base::get_epsilon()const
        float get_epsilon() const { return base::get_epsilon(); }

        /// Returns normalization flags.
        /// @return Normalization flags.
        normalization_flags get_flags() const {
            return base::get_flags<normalization_flags>();
        }
    };

    /// Default constructor. Produces an empty object.
    group_normalization_backward() = default;

    /// Constructs a group normalization backward propagation primitive.
    /// @param pd Primitive descriptor for a group normalization backward
    ///     propagation primitive.
    group_normalization_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a group normalization backward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a group normalization backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    group_normalization_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_group_normalization

/// @addtogroup dnnl_api_layer_normalization Layer Normalization
///
/// A primitive to perform layer normalization. Normalization is performed
/// within the last logical dimension of data tensor.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can refer to the same memory for forward
/// propagation, and diff_dst and diff_src can refer to the same memory for
/// backward propagation.
///
/// The layer normalization primitives computations can be controlled by
/// specifying different @ref dnnl::normalization_flags values. For example,
/// layer normalization forward propagation can be configured to either
/// compute the mean and variance or take them as arguments. It can either
/// perform scaling and shifting using gamma and beta parameters or not.
///
/// @sa @ref dev_guide_layer_normalization in developer guide
///
/// @{

/// Layer normalization forward propagation primitive.
struct layer_normalization_forward : public primitive {
    /// Primitive descriptor for a layer normalization forward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param stat_desc Statistics memory descriptors.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                const memory::desc &stat_desc, float epsilon,
                normalization_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, src_desc, dst_desc,
                    &stat_desc, memory::data_type::f32, epsilon, flags, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                float epsilon, normalization_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, src_desc, dst_desc, nullptr,
                    memory::data_type::f32, epsilon, flags, attr, allow_empty) {
        }

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive with a user-provided data type for the scale
        /// and shift memory objects.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param stat_desc Statistics memory descriptors.
        /// @param scale_shift_data_type Data type of scale and shift memory.
        ///     If neither scale nor shift flag are specified the parameter
        ///     is ignored.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                const memory::desc &stat_desc,
                memory::data_type scale_shift_data_type, float epsilon,
                normalization_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, src_desc, dst_desc,
                    &stat_desc, scale_shift_data_type, epsilon, flags, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive with a user-provided data type for the scale
        /// and shift memory objects.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param scale_shift_data_type Data type of scale and shift memory.
        ///     If neither scale nor shift flag are specified the parameter
        ///     is ignored.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                memory::data_type scale_shift_data_type, float epsilon,
                normalization_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, src_desc, dst_desc, nullptr,
                    scale_shift_data_type, epsilon, flags, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization
        /// forward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a layer normalization
        ///     forward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::layer_normalization,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const { return stat_desc(mean); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const { return stat_desc(var); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_epsilon()const
        float get_epsilon() const { return base::get_epsilon(); }

        /// Returns normalization flags.
        /// @return Normalization flags.
        normalization_flags get_flags() const {
            return base::get_flags<normalization_flags>();
        }

    private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            const bool use_global_stats
                    = (get_flags() & normalization_flags::use_global_stats)
                    != normalization_flags::none;
            return query_md(
                    use_global_stats ? query::src_md : query::dst_md, kind);
        }

        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                const memory::desc *stat_desc,
                memory::data_type scale_shift_data_type, float epsilon,
                normalization_flags flags, const primitive_attr &attr,
                bool allow_empty) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_layer_normalization_forward_primitive_desc_create_v2(
                            &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                            src_desc.get(), dst_desc.get(),
                            optional_arg(stat_desc),
                            memory::convert_to_c(scale_shift_data_type),
                            epsilon, convert_to_c(flags), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a layer "
                        "normalization forward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    layer_normalization_forward() = default;

    /// Constructs a layer normalization forward propagation primitive.
    /// @param pd Primitive descriptor for a layer normalization forward
    ///     propagation primitive.
    layer_normalization_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a layer normalization forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a layer normalization forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    layer_normalization_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Layer normalization backward propagation primitive.
struct layer_normalization_backward : public primitive {
    /// Primitive descriptor for a layer normalization backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param src_desc Source memory descriptor.
        /// @param stat_desc Statistics memory descriptors.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &src_desc,
                const memory::desc &stat_desc, float epsilon,
                normalization_flags flags,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, diff_src_desc, diff_dst_desc,
                    src_desc, &stat_desc, memory::data_type::f32,
                    memory::data_type::f32, epsilon, flags, hint_fwd_pd, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param src_desc Source memory descriptor.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &src_desc,
                float epsilon, normalization_flags flags,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, diff_src_desc, diff_dst_desc,
                    src_desc, nullptr, memory::data_type::f32,
                    memory::data_type::f32, epsilon, flags, hint_fwd_pd, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive with a user-provided data type for the scale
        /// and shift memory objects.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param src_desc Source memory descriptor.
        /// @param stat_desc Statistics memory descriptors.
        /// @param diff_scale_shift_data_type Data type of diff scale and shift
        ///     memory. If neither scale nor shift flag are specified the
        ///     parameter is ignored.
        /// @param scale_shift_data_type Data type of scale and shift memory.
        ///     If neither scale nor shift flag are specified the parameter
        ///     is ignored.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &src_desc,
                const memory::desc &stat_desc,
                memory::data_type diff_scale_shift_data_type,
                memory::data_type scale_shift_data_type, float epsilon,
                normalization_flags flags,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, diff_src_desc, diff_dst_desc,
                    src_desc, &stat_desc, diff_scale_shift_data_type,
                    scale_shift_data_type, epsilon, flags, hint_fwd_pd, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive with a user-provided data type for the scale
        /// and shift memory objects.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
        ///     (diffs for all parameters are computed in this case).
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param src_desc Source memory descriptor.
        /// @param diff_scale_shift_data_type Data type of diff scale and shift
        ///     memory. If neither scale nor shift flag are specified the
        ///     parameter is ignored.
        /// @param scale_shift_data_type Data type of scale and shift memory.
        ///     If neither scale nor shift flag are specified the parameter
        ///     is ignored.
        /// @param epsilon Layer normalization epsilon parameter.
        /// @param flags Layer normalization flags (@ref
        ///     dnnl::normalization_flags).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &src_desc,
                memory::data_type diff_scale_shift_data_type,
                memory::data_type scale_shift_data_type, float epsilon,
                normalization_flags flags,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, diff_src_desc, diff_dst_desc,
                    src_desc, nullptr, diff_scale_shift_data_type,
                    scale_shift_data_type, epsilon, flags, hint_fwd_pd, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a layer normalization
        ///     backward propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::layer_normalization,
                    dnnl::prop_kind::backward, dnnl::prop_kind::backward_data) {
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()const
        memory::desc mean_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()const
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        dnnl::prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_epsilon()const
        float get_epsilon() const { return base::get_epsilon(); }

        /// Returns normalization flags.
        /// @return Normalization flags.
        normalization_flags get_flags() const {
            return base::get_flags<normalization_flags>();
        }

    private:
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::desc &src_desc,
                const memory::desc *stat_desc,
                memory::data_type diff_scale_shift_data_type,
                memory::data_type scale_shift_data_type, float epsilon,
                normalization_flags flags,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr, bool allow_empty) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_layer_normalization_backward_primitive_desc_create_v2(
                            &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                            diff_src_desc.get(), diff_dst_desc.get(),
                            src_desc.get(), optional_arg(stat_desc),
                            memory::convert_to_c(diff_scale_shift_data_type),
                            memory::convert_to_c(scale_shift_data_type),
                            epsilon, convert_to_c(flags), hint_fwd_pd.get(),
                            attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a layer "
                        "normalization backward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    layer_normalization_backward() = default;

    /// Constructs a layer normalization backward propagation primitive.
    /// @param pd Primitive descriptor for a layer normalization backward
    ///     propagation primitive.
    layer_normalization_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a layer normalization backward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a layer normalization backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    layer_normalization_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_layer_normalization

/// @addtogroup dnnl_api_inner_product Inner Product
///
/// A primitive to compute an inner product.
///
/// @sa @ref dev_guide_inner_product in developer guide
///
/// @{

/// Inner product forward propagation primitive.
struct inner_product_forward : public primitive {
    /// Primitive descriptor for an inner product forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Memory descriptor for src.
        /// @param weights_desc Memory descriptor for weights.
        /// @param bias_desc Memory descriptor for bias.
        /// @param dst_desc Memory descriptor for dst.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, src_desc, weights_desc,
                    &bias_desc, dst_desc, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Memory descriptor for src.
        /// @param weights_desc Memory descriptor for weights.
        /// @param dst_desc Memory descriptor for dst.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, src_desc, weights_desc,
                    nullptr, dst_desc, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const { return base::weights_desc(1); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

    private:
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc *bias_desc, const memory::desc &dst_desc,
                const primitive_attr &attr, bool allow_empty) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_inner_product_forward_primitive_desc_create(&pd,
                            aengine.get(), dnnl::convert_to_c(aprop_kind),
                            src_desc.get(), weights_desc.get(),
                            optional_arg(bias_desc), dst_desc.get(),
                            attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for an inner "
                        "product forward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    inner_product_forward() = default;

    /// Constructs an inner product forward propagation primitive.
    /// @param pd Primitive descriptor for an inner product forward
    ///     propagation primitive.
    inner_product_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an inner product forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for an inner product forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    inner_product_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Inner product backward propagation primitive.
struct inner_product_backward_data : public primitive {
    /// Primitive descriptor for an inner product backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param diff_src_desc Memory descriptor for diff src.
        /// @param weights_desc Memory descriptor for weights.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {
            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_inner_product_backward_data_primitive_desc_create(
                            &pd, aengine.get(), diff_src_desc.get(),
                            weights_desc.get(), diff_dst_desc.get(),
                            hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for an inner "
                        "product backward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }
    };

    /// Default constructor. Produces an empty object.
    inner_product_backward_data() = default;

    /// Constructs an inner product backward propagation primitive.
    /// @param pd Primitive descriptor for an inner product backward
    ///     propagation primitive.
    inner_product_backward_data(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an inner product backward propagation primitive from
    /// a cache blob.
    /// @param pd Primitive descriptor for an inner product backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    inner_product_backward_data(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Inner product weights gradient primitive.
struct inner_product_backward_weights : public primitive {
    /// Primitive descriptor for an inner product weights gradient primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive with bias.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param src_desc Memory descriptor for src.
        /// @param diff_weights_desc Memory descriptor for diff weights.
        /// @param diff_bias_desc Memory descriptor for diff bias.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, src_desc, diff_weights_desc,
                    &diff_bias_desc, diff_dst_desc, hint_fwd_pd, attr,
                    allow_empty) {}

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param src_desc Memory descriptor for src.
        /// @param diff_weights_desc Memory descriptor for diff weights.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, src_desc, diff_weights_desc, nullptr,
                    diff_dst_desc, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an inner product weights
        ///     gradient primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()const
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::convolution_backward_weights::primitive_desc::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

    private:
        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc *diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr, bool allow_empty) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_inner_product_backward_weights_primitive_desc_create(
                            &pd, aengine.get(), src_desc.get(),
                            diff_weights_desc.get(),
                            optional_arg(diff_bias_desc), diff_dst_desc.get(),
                            hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for an inner "
                        "product weights gradient primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    inner_product_backward_weights() = default;

    /// Constructs an inner product weights gradient primitive.
    /// @param pd Primitive descriptor for an inner product weights gradient
    ///     primitive.
    inner_product_backward_weights(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an inner product weights gradient primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for an inner product weights gradient
    ///     primitive.
    /// @param cache_blob Cache blob.
    inner_product_backward_weights(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_inner_product

/// @addtogroup dnnl_api_rnn RNN
///
/// A primitive to compute recurrent neural network layers.
///
/// @sa @ref dev_guide_rnn in developer guide
///
/// @{

/// Base class for primitive descriptors for RNN primitives.
struct rnn_primitive_desc_base : public primitive_desc {
    using primitive_desc::primitive_desc;

    /// Default constructor. Produces an empty object.
    rnn_primitive_desc_base() = default;

    /// Constructs an RNN primitive descriptor base from a C API primitive
    /// descriptor while checking that it actually describes the expected
    /// primitive by comparing propagation and primitive kinds.
    ///
    /// @param pd C API primitive descriptor.
    /// @param aprop_kind Expected propagation kind.
    /// @param cell_kind Expected cell kind.
    rnn_primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::prop_kind aprop_kind, dnnl::algorithm cell_kind)
        : rnn_primitive_desc_base(pd, aprop_kind, aprop_kind, cell_kind) {}

    /// Returns source layer memory descriptor.
    /// @returns Source layer memory descriptor.
    memory::desc src_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_SRC_LAYER);
    }

    /// Returns AUGRU attention memory descriptor.
    /// @returns AUGRU attention memory descriptor.
    memory::desc augru_attention_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_AUGRU_ATTENTION);
    }

    /// Returns source iteration memory descriptor.
    /// @returns Source iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          source iteration parameter.
    memory::desc src_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_SRC_ITER);
    }

    /// Returns source recurrent cell state memory descriptor.
    /// @returns Source recurrent cell state memory descriptor.
    memory::desc src_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_SRC_ITER_C);
    }

    /// Returns weights layer memory descriptor.
    /// @returns Weights layer memory descriptor.
    memory::desc weights_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_LAYER);
    }

    /// Returns weights iteration memory descriptor.
    /// @returns Weights iteration memory descriptor.
    memory::desc weights_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_ITER);
    }

    /// Returns weights peephole memory descriptor.
    /// @returns Weights peephole memory descriptor.
    memory::desc weights_peephole_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_PEEPHOLE);
    }

    /// Returns weights projection memory descriptor.
    /// @returns Weights projection memory descriptor.
    memory::desc weights_projection_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_PROJECTION);
    }

    /// Returns bias memory descriptor.
    /// @returns Bias memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          bias parameter.
    memory::desc bias_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_BIAS);
    }

    /// Returns destination layer memory descriptor.
    /// @returns Destination layer memory descriptor.
    memory::desc dst_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DST_LAYER);
    }

    /// Returns destination iteration memory descriptor.
    /// @returns Destination iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          destination iteration parameter.
    memory::desc dst_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DST_ITER);
    }

    /// Returns destination recurrent cell state memory descriptor.
    /// @returns Destination recurrent cell state memory descriptor.
    memory::desc dst_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DST_ITER_C);
    }

    /// Returns diff source layer memory descriptor.
    /// @returns Diff source layer memory descriptor.
    memory::desc diff_src_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC_LAYER);
    }

    /// Returns diff AUGRU attention memory descriptor.
    /// @returns Diff AUGRU attention memory descriptor.
    memory::desc diff_augru_attention_desc() const {
        return base::query_md(
                query::exec_arg_md, DNNL_ARG_DIFF_AUGRU_ATTENTION);
    }

    /// Returns diff source iteration memory descriptor.
    /// @returns Diff source iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff source iteration parameter.
    memory::desc diff_src_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC_ITER);
    }

    /// Returns diff source recurrent cell state memory descriptor.
    /// @returns Diff source recurrent cell state memory descriptor.
    memory::desc diff_src_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC_ITER_C);
    }

    /// Returns diff weights layer memory descriptor.
    /// @returns Diff weights layer memory descriptor.
    memory::desc diff_weights_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS_LAYER);
    }

    /// Returns diff weights iteration memory descriptor.
    /// @returns Diff weights iteration memory descriptor.
    memory::desc diff_weights_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS_ITER);
    }

    /// Returns diff weights peephole memory descriptor.
    /// @returns Diff weights peephole memory descriptor.
    memory::desc diff_weights_peephole_desc() const {
        return base::query_md(
                query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE);
    }

    /// Returns diff weights projection memory descriptor.
    /// @returns Diff weights projection memory descriptor.
    memory::desc diff_weights_projection_desc() const {
        return base::query_md(
                query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS_PROJECTION);
    }

    /// Returns diff bias memory descriptor.
    /// @returns Diff bias memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff bias parameter.
    memory::desc diff_bias_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_BIAS);
    }

    /// Returns diff destination layer memory descriptor.
    /// @returns Diff destination layer memory descriptor.
    memory::desc diff_dst_layer_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST_LAYER);
    }

    /// Returns diff destination iteration memory descriptor.
    /// @returns Diff destination iteration memory descriptor.
    /// @returns A zero memory descriptor if the primitive does not have a
    ///          diff destination iteration parameter.
    memory::desc diff_dst_iter_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST_ITER);
    }

    /// Returns diff destination recurrent cell state memory descriptor.
    /// @returns Diff destination recurrent cell state memory descriptor.
    memory::desc diff_dst_iter_c_desc() const {
        return base::query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST_ITER_C);
    }

protected:
    using rnn_base = rnn_primitive_desc_base;

    // (Deliberately not using doxygen comments)
    //
    // Constructs an RNN primitive descriptor base from a C API primitive
    // descriptor while checking that it actually describes the expected
    // primitive by comparing propagation and primitive kinds. Caller can
    // pass two options propagation kinds. This is typically used to check
    // that propagation kind is inference or training forward propagation.
    //
    // @param pd C API primitive descriptor.
    // @param prop_kind1 Expected propagation kind.
    // @param prop_kind2 Expected propagation kind.
    // @param cell_kind Expected cell kind.
    rnn_primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::prop_kind prop_kind1, dnnl::prop_kind prop_kind2,
            dnnl::algorithm cell_kind) {

        dnnl_status_t rc;

        dnnl_primitive_kind_t q_primitive_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_primitive_kind, 0, &q_primitive_kind);
        error::wrap_c_api(rc,
                "could not retrieve a primitive kind from a primitive "
                "descriptor for an RNN primitive");

        dnnl_prop_kind_t q_prop_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_prop_kind, 0, &q_prop_kind);
        error::wrap_c_api(rc,
                "could not retrieve a propagation kind from a primitive "
                "descriptor for an RNN primitive");

        dnnl_alg_kind_t q_cell_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_cell_kind, 0, &q_cell_kind);
        error::wrap_c_api(rc,
                "could not retrieve a cell kind from a primitive descriptor "
                "for an RNN primitive");

        dnnl_prop_kind_t c_prop_kind1 = convert_to_c(prop_kind1);
        dnnl_prop_kind_t c_prop_kind2 = convert_to_c(prop_kind2);
        dnnl_alg_kind_t c_cell_kind = convert_to_c(cell_kind);

        bool ok = q_primitive_kind == dnnl_rnn
                && (q_prop_kind == c_prop_kind1 || q_prop_kind == c_prop_kind2)
                && q_cell_kind == c_cell_kind;

        if (!ok)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "mismatch between expected and provided descriptors for an "
                    "RNN primitive");

        reset_with_clone(pd);
    }

    // Constructs an RNN forward propagation primitive descriptor base for
    // any cell kind.
    rnn_primitive_desc_base(const engine &aengine, algorithm cell_kind,
            prop_kind aprop_kind, algorithm activation, rnn_direction direction,
            const memory::desc &src_layer_desc,
            const memory::desc &src_iter_desc,
            const memory::desc *src_iter_c_desc,
            const memory::desc *attention_desc,
            const memory::desc &weights_layer_desc,
            const memory::desc &weights_iter_desc,
            const memory::desc *weights_peephole_desc,
            const memory::desc *weights_projection_desc,
            const memory::desc &bias_desc, const memory::desc &dst_layer_desc,
            const memory::desc &dst_iter_desc,
            const memory::desc *dst_iter_c_desc, rnn_flags flags, float alpha,
            float beta, const primitive_attr &attr, bool allow_empty) {

        dnnl_status_t status = dnnl_success;
        const char *msg
                = "could not create a primitive descriptor for a requested "
                  "cell kind";

        dnnl_primitive_desc_t pd = nullptr;
        switch (cell_kind) {
            case algorithm::vanilla_rnn:
                status = dnnl_vanilla_rnn_forward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(activation),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), weights_layer_desc.get(),
                        weights_iter_desc.get(), bias_desc.get(),
                        dst_layer_desc.get(), dst_iter_desc.get(),
                        convert_to_c(flags), alpha, beta, attr.get());
                msg = "could not create a primitive descriptor for a vanilla "
                      "RNN forward propagation primitive";
                break;
            case algorithm::vanilla_lstm:
                status = dnnl_lstm_forward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), optional_arg(src_iter_c_desc),
                        weights_layer_desc.get(), weights_iter_desc.get(),
                        optional_arg(weights_peephole_desc),
                        optional_arg(weights_projection_desc), bias_desc.get(),
                        dst_layer_desc.get(), dst_iter_desc.get(),
                        optional_arg(dst_iter_c_desc), convert_to_c(flags),
                        attr.get());
                msg = "could not create a primitive descriptor for an LSTM "
                      "forward propagation primitive";
                break;
            case algorithm::vanilla_gru:
                status = dnnl_gru_forward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), weights_layer_desc.get(),
                        weights_iter_desc.get(), bias_desc.get(),
                        dst_layer_desc.get(), dst_iter_desc.get(),
                        convert_to_c(flags), attr.get());
                msg = "could not create a primitive descriptor for a GRU "
                      "forward propagation primitive";
                break;
            case algorithm::lbr_gru:
                status = dnnl_lbr_gru_forward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), weights_layer_desc.get(),
                        weights_iter_desc.get(), bias_desc.get(),
                        dst_layer_desc.get(), dst_iter_desc.get(),
                        convert_to_c(flags), attr.get());
                msg = "could not create a primitive descriptor for an LBR GRU "
                      "forward propagation primitive";
                break;
            case algorithm::vanilla_augru:
                status = dnnl_augru_forward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), optional_arg(attention_desc),
                        weights_layer_desc.get(), weights_iter_desc.get(),
                        bias_desc.get(), dst_layer_desc.get(),
                        dst_iter_desc.get(), convert_to_c(flags), attr.get());
                msg = "could not create a primitive descriptor for an AUGRU "
                      "forward propagation primitive";
                break;
            case algorithm::lbr_augru:
                status = dnnl_lbr_augru_forward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), optional_arg(attention_desc),
                        weights_layer_desc.get(), weights_iter_desc.get(),
                        bias_desc.get(), dst_layer_desc.get(),
                        dst_iter_desc.get(), convert_to_c(flags), attr.get());
                msg = "could not create a primitive descriptor for an LBR "
                      "AUGRU forward propagation primitive";
                break;
            default: status = dnnl_unimplemented;
        }

        if (!allow_empty) error::wrap_c_api(status, msg);
        reset(pd);
    }

    // Constructs an RNN backward propagation primitive descriptor base for
    // any cell kind.
    rnn_primitive_desc_base(const engine &aengine, algorithm cell_kind,
            prop_kind aprop_kind, algorithm activation, rnn_direction direction,
            const memory::desc &src_layer_desc,
            const memory::desc &src_iter_desc,
            const memory::desc *src_iter_c_desc,
            const memory::desc *attention_desc,
            const memory::desc &weights_layer_desc,
            const memory::desc &weights_iter_desc,
            const memory::desc *weights_peephole_desc,
            const memory::desc *weights_projection_desc,
            const memory::desc &bias_desc, const memory::desc &dst_layer_desc,
            const memory::desc &dst_iter_desc,
            const memory::desc *dst_iter_c_desc,
            const memory::desc &diff_src_layer_desc,
            const memory::desc &diff_src_iter_desc,
            const memory::desc *diff_src_iter_c_desc,
            const memory::desc *diff_attention_desc,
            const memory::desc &diff_weights_layer_desc,
            const memory::desc &diff_weights_iter_desc,
            const memory::desc *diff_weights_peephole_desc,
            const memory::desc *diff_weights_projection_desc,
            const memory::desc &diff_bias_desc,
            const memory::desc &diff_dst_layer_desc,
            const memory::desc &diff_dst_iter_desc,
            const memory::desc *diff_dst_iter_c_desc, rnn_flags flags,
            float alpha, float beta, const rnn_primitive_desc_base &hint_fwd_pd,
            const primitive_attr &attr, bool allow_empty) {

        dnnl_status_t status = dnnl_success;
        const char *msg = "";

        dnnl_primitive_desc_t pd = nullptr;
        switch (cell_kind) {
            case algorithm::vanilla_rnn:
                status = dnnl_vanilla_rnn_backward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(activation),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), weights_layer_desc.get(),
                        weights_iter_desc.get(), bias_desc.get(),
                        dst_layer_desc.get(), dst_iter_desc.get(),
                        diff_src_layer_desc.get(), diff_src_iter_desc.get(),
                        diff_weights_layer_desc.get(),
                        diff_weights_iter_desc.get(), diff_bias_desc.get(),
                        diff_dst_layer_desc.get(), diff_dst_iter_desc.get(),
                        convert_to_c(flags), alpha, beta, hint_fwd_pd.get(),
                        attr.get());
                msg = "could not create a primitive descriptor for a vanilla "
                      "RNN backward propagation primitive";
                break;
            case algorithm::vanilla_lstm:
                status = dnnl_lstm_backward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), optional_arg(src_iter_c_desc),
                        weights_layer_desc.get(), weights_iter_desc.get(),
                        optional_arg(weights_peephole_desc),
                        optional_arg(weights_projection_desc), bias_desc.get(),
                        dst_layer_desc.get(), dst_iter_desc.get(),
                        optional_arg(dst_iter_c_desc),
                        diff_src_layer_desc.get(), diff_src_iter_desc.get(),
                        optional_arg(diff_src_iter_c_desc),
                        diff_weights_layer_desc.get(),
                        diff_weights_iter_desc.get(),
                        optional_arg(diff_weights_peephole_desc),
                        optional_arg(diff_weights_projection_desc),
                        diff_bias_desc.get(), diff_dst_layer_desc.get(),
                        diff_dst_iter_desc.get(),
                        optional_arg(diff_dst_iter_c_desc), convert_to_c(flags),
                        hint_fwd_pd.get(), attr.get());
                msg = "could not create a primitive descriptor for an LSTM "
                      "backward propagation primitive";
                break;
            case algorithm::vanilla_gru:
                status = dnnl_gru_backward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), weights_layer_desc.get(),
                        weights_iter_desc.get(), bias_desc.get(),
                        dst_layer_desc.get(), dst_iter_desc.get(),
                        diff_src_layer_desc.get(), diff_src_iter_desc.get(),
                        diff_weights_layer_desc.get(),
                        diff_weights_iter_desc.get(), diff_bias_desc.get(),
                        diff_dst_layer_desc.get(), diff_dst_iter_desc.get(),
                        convert_to_c(flags), hint_fwd_pd.get(), attr.get());
                msg = "could not create a primitive descriptor for a GRU "
                      "backward propagation primitive";
                break;
            case algorithm::lbr_gru:
                status = dnnl_lbr_gru_backward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), weights_layer_desc.get(),
                        weights_iter_desc.get(), bias_desc.get(),
                        dst_layer_desc.get(), dst_iter_desc.get(),
                        diff_src_layer_desc.get(), diff_src_iter_desc.get(),
                        diff_weights_layer_desc.get(),
                        diff_weights_iter_desc.get(), diff_bias_desc.get(),
                        diff_dst_layer_desc.get(), diff_dst_iter_desc.get(),
                        convert_to_c(flags), hint_fwd_pd.get(), attr.get());
                msg = "could not create a primitive descriptor for an LBR GRU "
                      "backward propagation primitive";
                break;
            case algorithm::vanilla_augru:
                status = dnnl_augru_backward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), optional_arg(attention_desc),
                        weights_layer_desc.get(), weights_iter_desc.get(),
                        bias_desc.get(), dst_layer_desc.get(),
                        dst_iter_desc.get(), diff_src_layer_desc.get(),
                        diff_src_iter_desc.get(),
                        optional_arg(diff_attention_desc),
                        diff_weights_layer_desc.get(),
                        diff_weights_iter_desc.get(), diff_bias_desc.get(),
                        diff_dst_layer_desc.get(), diff_dst_iter_desc.get(),
                        convert_to_c(flags), hint_fwd_pd.get(), attr.get());
                msg = "could not create a primitive descriptor for an AUGRU "
                      "backward propagation primitive";
                break;
            case algorithm::lbr_augru:
                status = dnnl_lbr_augru_backward_primitive_desc_create(&pd,
                        aengine.get(), dnnl::convert_to_c(aprop_kind),
                        dnnl::convert_to_c(direction), src_layer_desc.get(),
                        src_iter_desc.get(), optional_arg(attention_desc),
                        weights_layer_desc.get(), weights_iter_desc.get(),
                        bias_desc.get(), dst_layer_desc.get(),
                        dst_iter_desc.get(), diff_src_layer_desc.get(),
                        diff_src_iter_desc.get(),
                        optional_arg(diff_attention_desc),
                        diff_weights_layer_desc.get(),
                        diff_weights_iter_desc.get(), diff_bias_desc.get(),
                        diff_dst_layer_desc.get(), diff_dst_iter_desc.get(),
                        convert_to_c(flags), hint_fwd_pd.get(), attr.get());
                msg = "could not create a primitive descriptor for an LBR "
                      "AUGRU backward propagation primitive";
                break;
            default: status = dnnl_unimplemented;
        }
        if (!allow_empty) error::wrap_c_api(status, msg);
        reset(pd);
    }
};

/// Vanilla RNN forward propagation primitive.
struct vanilla_rnn_forward : public primitive {
    /// Primitive descriptor for a vanilla RNN forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a vanilla RNN forward
        ///     propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the RNN forward propagation primitive
        /// should not use them and should default to zero values instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param activation Activation kind. Possible values are
        ///     #dnnl::algorithm::eltwise_relu,
        ///     #dnnl::algorithm::eltwise_tanh, or
        ///     #dnnl::algorithm::eltwise_logistic.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm activation, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_rnn,
                    aprop_kind, activation, direction, src_layer_desc,
                    src_iter_desc, nullptr, nullptr, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, rnn_flags::undef,
                    0.0f, 0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN forward
        ///     propagation primitive with alpha parameter.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the RNN forward propagation primitive
        /// should not use them and should default to zero values instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param activation Activation kind. Possible values are
        ///     #dnnl::algorithm::eltwise_relu,
        ///     #dnnl::algorithm::eltwise_tanh, or
        ///     #dnnl::algorithm::eltwise_logistic.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param alpha Negative slope if activation is
        ///     #dnnl::algorithm::eltwise_relu.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm activation, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc, float alpha,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_rnn,
                    aprop_kind, activation, direction, src_layer_desc,
                    src_iter_desc, nullptr, nullptr, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, rnn_flags::undef,
                    alpha, 0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a vanilla RNN forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_rnn) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_activation_kind()const
        algorithm get_activation_kind() const {
            return base::get_activation_kind();
        }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }

        /// @copydoc dnnl::primitive_desc_base::get_alpha()const
        float get_alpha() const { return base::get_alpha(); }

        /// @copydoc dnnl::primitive_desc_base::get_beta()const
        float get_beta() const { return base::get_beta(); }
    };

    /// Default constructor. Produces an empty object.
    vanilla_rnn_forward() = default;

    /// Constructs a vanilla RNN forward propagation primitive.
    /// @param pd Primitive descriptor for a vanilla RNN forward
    ///     propagation primitive.
    vanilla_rnn_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a vanilla RNN forward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a vanilla RNN forward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    vanilla_rnn_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Vanilla RNN backward propagation primitive.
struct vanilla_rnn_backward : public primitive {
    /// Primitive descriptor for an RNN backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a vanilla RNN backward
        ///     propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the RNN backward propagation
        /// primitive should not use the respective data and should use zero
        /// values instead.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param activation Activation kind. Possible values are
        ///     #dnnl::algorithm::eltwise_relu,
        ///     #dnnl::algorithm::eltwise_tanh, or
        ///     #dnnl::algorithm::eltwise_logistic.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param hint_fwd_pd Primitive descriptor for a vanilla RNN
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm activation, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const vanilla_rnn_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_rnn,
                    aprop_kind, activation, direction, src_layer_desc,
                    src_iter_desc, nullptr, nullptr, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
                    diff_src_iter_desc, nullptr, nullptr,
                    diff_weights_layer_desc, diff_weights_iter_desc, nullptr,
                    nullptr, diff_bias_desc, diff_dst_layer_desc,
                    diff_dst_iter_desc, nullptr, rnn_flags::undef, 0.0f, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN backward
        ///     propagation primitive with an alpha parameter.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the RNN backward propagation
        /// primitive should not use the respective data and should use zero
        /// values instead.
        ///
        /// @note
        ///     All the memory descriptors may be initialized with the
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param activation Activation kind. Possible values are
        ///     #dnnl::algorithm::eltwise_relu,
        ///     #dnnl::algorithm::eltwise_tanh, or
        ///     #dnnl::algorithm::eltwise_logistic.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param alpha Negative slope if activation is
        ///     #dnnl::algorithm::eltwise_relu.
        /// @param hint_fwd_pd Primitive descriptor for a vanilla RNN
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm activation, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc, float alpha,
                const vanilla_rnn_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_rnn,
                    aprop_kind, activation, direction, src_layer_desc,
                    src_iter_desc, nullptr, nullptr, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
                    diff_src_iter_desc, nullptr, nullptr,
                    diff_weights_layer_desc, diff_weights_iter_desc, nullptr,
                    nullptr, diff_bias_desc, diff_dst_layer_desc,
                    diff_dst_iter_desc, nullptr, rnn_flags::undef, alpha, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a vanilla RNN backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_rnn) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_activation_kind()const
        algorithm get_activation_kind() const {
            return base::get_activation_kind();
        }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }

        /// @copydoc dnnl::primitive_desc_base::get_alpha()const
        float get_alpha() const { return base::get_alpha(); }

        /// @copydoc dnnl::primitive_desc_base::get_beta()const
        float get_beta() const { return base::get_beta(); }
    };

    /// Default constructor. Produces an empty object.
    vanilla_rnn_backward() = default;

    /// Constructs a vanilla RNN backward propagation primitive.
    /// @param pd Primitive descriptor for a vanilla RNN backward
    ///     propagation primitive.
    vanilla_rnn_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a vanilla RNN backward propagation primitive from
    ///     a cache blob.
    /// @param pd Primitive descriptor for a vanilla RNN backward
    ///     propagation primitive.
    /// @param cache_blob Cache blob.
    vanilla_rnn_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LSTM forward propagation primitive.
struct lstm_forward : public primitive {
    /// Primitive descriptor for an LSTM forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LSTM (with or without
        ///     peephole and with or without projection) forward propagation
        ///     primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p weights_peephole_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// The @p weights_projection_desc may point to a zero memory
        /// descriptor. This would then indicate that the LSTM doesn't have
        /// recurrent projection layer.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param weights_projection_desc Memory descriptor for the weights
        ///     applied to the hidden states to get the recurrent projection
        ///     (according to the Projection LSTM formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &weights_peephole_desc,
                const memory::desc &weights_projection_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_lstm,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, &src_iter_c_desc, nullptr,
                    weights_layer_desc, weights_iter_desc,
                    &weights_peephole_desc, &weights_projection_desc, bias_desc,
                    dst_layer_desc, dst_iter_desc, &dst_iter_c_desc,
                    rnn_flags::undef, 0.0f, 0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM (with or without
        ///     peephole) forward propagation primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p weights_peephole_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &weights_peephole_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_lstm,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, &src_iter_c_desc, nullptr,
                    weights_layer_desc, weights_iter_desc,
                    &weights_peephole_desc, nullptr, bias_desc, dst_layer_desc,
                    dst_iter_desc, &dst_iter_c_desc, rnn_flags::undef, 0.0f,
                    0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM forward propagation
        ///     primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors can be initialized with an
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_lstm,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, &src_iter_c_desc, nullptr,
                    weights_layer_desc, weights_iter_desc, nullptr, nullptr,
                    bias_desc, dst_layer_desc, dst_iter_desc, &dst_iter_c_desc,
                    rnn_flags::undef, 0.0f, 0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LSTM forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_lstm) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_c_desc() const {
            return rnn_base::src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_peephole_desc()const
        memory::desc weights_peephole_desc() const {
            return rnn_base::weights_peephole_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_projection_desc()const
        memory::desc weights_projection_desc() const {
            return rnn_base::weights_projection_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc dst_iter_c_desc() const {
            return rnn_base::dst_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    lstm_forward() = default;

    /// Constructs an LSTM forward propagation primitive.
    /// @param pd Primitive descriptor for an LSTM forward propagation
    ///     primitive.
    lstm_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LSTM forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LSTM forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lstm_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LSTM backward propagation primitive.
struct lstm_backward : public primitive {
    /// Primitive descriptor for an LSTM backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs an LSTM (with or without peephole and with or without
        ///     projection) primitive descriptor for backward propagation
        ///     using @p prop_kind, @p direction, and memory descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p weights_peephole_desc together with
        ///   @p diff_weights_peephole_desc
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// The @p weights_projection_desc together with @p
        /// diff_weights_projection_desc may point to a zero memory descriptor.
        /// This would then indicate that the LSTM doesn't have recurrent
        /// projection layer.
        ///
        /// @note
        ///     All memory descriptors can be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param weights_projection_desc Memory descriptor for the weights
        ///     applied to the hidden states to get the recurrent projection
        ///     (according to the Projection LSTM formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_weights_peephole_desc Memory descriptor for the diff of
        ///     weights applied to the cell states (according to the Peephole
        ///     LSTM formula).
        /// @param diff_weights_projection_desc Memory descriptor for the diff
        ///     of weights applied to the hidden states to get the recurrent
        ///     projection (according to the Projection LSTM formula).
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param hint_fwd_pd Primitive descriptor for an LSTM
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &weights_peephole_desc,
                const memory::desc &weights_projection_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_src_iter_c_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_weights_peephole_desc,
                const memory::desc &diff_weights_projection_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const memory::desc &diff_dst_iter_c_desc,
                const lstm_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_lstm,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, &src_iter_c_desc, nullptr,
                    weights_layer_desc, weights_iter_desc,
                    &weights_peephole_desc, &weights_projection_desc, bias_desc,
                    dst_layer_desc, dst_iter_desc, &dst_iter_c_desc,
                    diff_src_layer_desc, diff_src_iter_desc,
                    &diff_src_iter_c_desc, nullptr, diff_weights_layer_desc,
                    diff_weights_iter_desc, &diff_weights_peephole_desc,
                    &diff_weights_projection_desc, diff_bias_desc,
                    diff_dst_layer_desc, diff_dst_iter_desc,
                    &diff_dst_iter_c_desc, rnn_flags::undef, 0.0f, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs an LSTM (with or without peephole) primitive descriptor
        ///     for backward propagation using @p prop_kind, @p direction,
        ///     and memory descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p weights_peephole_desc together with
        ///   @p diff_weights_peephole_desc
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param weights_peephole_desc Memory descriptor for the weights
        ///     applied to the cell states (according to the Peephole LSTM
        ///     formula).
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_weights_peephole_desc Memory descriptor for the diff of
        ///     weights applied to the cell states (according to the Peephole
        ///     LSTM formula).
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param hint_fwd_pd Primitive descriptor for an LSTM
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &weights_peephole_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_src_iter_c_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_weights_peephole_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const memory::desc &diff_dst_iter_c_desc,
                const lstm_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_lstm,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, &src_iter_c_desc, nullptr,
                    weights_layer_desc, weights_iter_desc,
                    &weights_peephole_desc, nullptr, bias_desc, dst_layer_desc,
                    dst_iter_desc, &dst_iter_c_desc, diff_src_layer_desc,
                    diff_src_iter_desc, &diff_src_iter_c_desc, nullptr,
                    diff_weights_layer_desc, diff_weights_iter_desc,
                    &diff_weights_peephole_desc, nullptr, diff_bias_desc,
                    diff_dst_layer_desc, diff_dst_iter_desc,
                    &diff_dst_iter_c_desc, rnn_flags::undef, 0.0f, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs an LSTM primitive descriptor for backward propagation
        ///     using @p prop_kind, @p direction, and memory descriptors.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p src_iter_c_desc,
        ///   @p diff_src_iter_desc, and @p diff_src_iter_c_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p dst_iter_c_desc,
        ///   @p diff_dst_iter_desc, and @p diff_dst_iter_c_desc.
        ///
        /// This would then indicate that the LSTM backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param src_iter_c_desc Memory descriptor for the input recurrent
        ///     cell state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param dst_iter_c_desc Memory descriptor for the output recurrent
        ///     cell state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_src_iter_c_desc Memory descriptor for the diff of
        ///     input recurrent cell state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param diff_dst_iter_c_desc Memory descriptor for the diff of
        ///     output recurrent cell state vector.
        /// @param hint_fwd_pd Primitive descriptor for a convolution
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_src_iter_c_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const memory::desc &diff_dst_iter_c_desc,
                const lstm_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_lstm,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, &src_iter_c_desc, nullptr,
                    weights_layer_desc, weights_iter_desc, nullptr, nullptr,
                    bias_desc, dst_layer_desc, dst_iter_desc, &dst_iter_c_desc,
                    diff_src_layer_desc, diff_src_iter_desc,
                    &diff_src_iter_c_desc, nullptr, diff_weights_layer_desc,
                    diff_weights_iter_desc, nullptr, nullptr, diff_bias_desc,
                    diff_dst_layer_desc, diff_dst_iter_desc,
                    &diff_dst_iter_c_desc, rnn_flags::undef, 0.0f, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LSTM backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_lstm) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_c_desc() const {
            return rnn_base::src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_peephole_desc()const
        memory::desc weights_peephole_desc() const {
            return rnn_base::weights_peephole_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_projection_desc()const
        memory::desc weights_projection_desc() const {
            return rnn_base::weights_projection_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc dst_iter_c_desc() const {
            return rnn_base::dst_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_c_desc()const
        memory::desc diff_src_iter_c_desc() const {
            return rnn_base::diff_src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_peephole_desc()const
        memory::desc diff_weights_peephole_desc() const {
            return rnn_base::diff_weights_peephole_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_projection_desc()const
        memory::desc diff_weights_projection_desc() const {
            return rnn_base::diff_weights_projection_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_c_desc()const
        memory::desc diff_dst_iter_c_desc() const {
            return rnn_base::diff_dst_iter_c_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    lstm_backward() = default;

    /// Constructs an LSTM backward propagation primitive.
    /// @param pd Primitive descriptor for an LSTM backward propagation
    ///     primitive.
    lstm_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LSTM backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LSTM backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lstm_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// GRU forward propagation primitive.
struct gru_forward : public primitive {
    /// Primitive descriptor for a GRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a GRU forward propagation
        ///     primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the GRU forward propagation primitive
        /// should not use them and should default to zero values instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_gru,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, nullptr, nullptr, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, rnn_flags::undef,
                    0.0f, 0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a GRU forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    gru_forward() = default;

    /// Constructs a GRU forward propagation primitive.
    /// @param pd Primitive descriptor for a GRU forward propagation
    ///     primitive.
    gru_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a GRU forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a GRU forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    gru_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// GRU backward propagation primitive.
struct gru_backward : public primitive {
    /// Primitive descriptor for a GRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a GRU backward propagation
        ///     primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the GRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param hint_fwd_pd Primitive descriptor for a GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const gru_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_gru,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, nullptr, nullptr, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
                    diff_src_iter_desc, nullptr, nullptr,
                    diff_weights_layer_desc, diff_weights_iter_desc, nullptr,
                    nullptr, diff_bias_desc, diff_dst_layer_desc,
                    diff_dst_iter_desc, nullptr, rnn_flags::undef, 0.0f, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a GRU backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    gru_backward() = default;

    /// Constructs a GRU backward propagation primitive.
    /// @param pd Primitive descriptor for a GRU backward propagation
    ///     primitive.
    gru_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a GRU backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a GRU backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    gru_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LBR GRU forward propagation primitive.
struct lbr_gru_forward : public primitive {
    /// Primitive descriptor for an LBR GRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for LBR GRU forward propagation
        ///     primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the LBR GRU forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::lbr_gru, aprop_kind,
                    algorithm::undef, direction, src_layer_desc, src_iter_desc,
                    nullptr, nullptr, weights_layer_desc, weights_iter_desc,
                    nullptr, nullptr, bias_desc, dst_layer_desc, dst_iter_desc,
                    nullptr, rnn_flags::undef, 0.0f, 0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a LBR GRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a LBR GRU forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::lbr_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    lbr_gru_forward() = default;

    /// Constructs an LBR GRU forward propagation primitive.
    /// @param pd Primitive descriptor for an LBR GRU forward propagation
    ///     primitive.
    lbr_gru_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LBR GRU forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LBR GRU forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lbr_gru_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LBR GRU backward propagation primitive.
struct lbr_gru_backward : public primitive {
    /// Primitive descriptor for an LBR GRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for LBR GRU backward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the LBR GRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param hint_fwd_pd Primitive descriptor for an LBR GRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const lbr_gru_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::lbr_gru, aprop_kind,
                    algorithm::undef, direction, src_layer_desc, src_iter_desc,
                    nullptr, nullptr, weights_layer_desc, weights_iter_desc,
                    nullptr, nullptr, bias_desc, dst_layer_desc, dst_iter_desc,
                    nullptr, diff_src_layer_desc, diff_src_iter_desc, nullptr,
                    nullptr, diff_weights_layer_desc, diff_weights_iter_desc,
                    nullptr, nullptr, diff_bias_desc, diff_dst_layer_desc,
                    diff_dst_iter_desc, nullptr, rnn_flags::undef, 0.0f, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a LBR GRU backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a LBR GRU backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(
                    pd, dnnl::prop_kind::backward, dnnl::algorithm::lbr_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    lbr_gru_backward() = default;

    /// Constructs an LBR GRU backward propagation primitive.
    /// @param pd Primitive descriptor for an LBR GRU backward propagation
    ///     primitive.
    lbr_gru_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LBR GRU backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LBR GRU backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lbr_gru_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// AUGRU forward propagation primitive.
struct augru_forward : public primitive {
    /// Primitive descriptor for an AUGRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an AUGRU forward propagation
        ///     primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the AUGRU forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param attention_desc Memory descriptor for the attention vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &attention_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_augru,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, nullptr, &attention_desc, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, rnn_flags::undef,
                    0.0f, 0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an AUGRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an AUGRU forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_augru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::augru_attention_desc()const
        memory::desc attention_desc() const {
            return rnn_base::augru_attention_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    augru_forward() = default;

    /// Constructs an AUGRU forward propagation primitive.
    /// @param pd Primitive descriptor for an AUGRU forward propagation
    ///     primitive.
    augru_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an AUGRU forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an AUGRU forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    augru_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// AUGRU backward propagation primitive.
struct augru_backward : public primitive {
    /// Descriptor for an AUGRU backward propagation primitive.
    /// Primitive descriptor for an AUGRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an AUGRU backward propagation
        ///     primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the AUGRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param attention_desc Memory descriptor for the attention vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_attention_desc Memory descriptor for the diff of
        ///     attention vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param hint_fwd_pd Primitive descriptor for an AUGRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &attention_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_attention_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const augru_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::vanilla_augru,
                    aprop_kind, algorithm::undef, direction, src_layer_desc,
                    src_iter_desc, nullptr, &attention_desc, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
                    diff_src_iter_desc, nullptr, &diff_attention_desc,
                    diff_weights_layer_desc, diff_weights_iter_desc, nullptr,
                    nullptr, diff_bias_desc, diff_dst_layer_desc,
                    diff_dst_iter_desc, nullptr, rnn_flags::undef, 0.0f, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an AUGRU backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an AUGRU backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_augru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::augru_attention_desc()const
        memory::desc attention_desc() const {
            return rnn_base::augru_attention_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_augru_attention_desc()const
        memory::desc diff_attention_desc() const {
            return rnn_base::diff_augru_attention_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    augru_backward() = default;

    /// Constructs an AUGRU backward propagation primitive.
    /// @param pd Primitive descriptor for an AUGRU backward propagation
    ///     primitive.
    augru_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an AUGRU backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an AUGRU backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    augru_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LBR AUGRU forward propagation primitive.
struct lbr_augru_forward : public primitive {
    /// Descriptor for an LBR AUGRU forward propagation primitive.

    /// Primitive descriptor for an LBR AUGRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for LBR AUGRU forward propagation
        ///     primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc,
        /// - @p bias_desc,
        /// - @p dst_iter_desc.
        ///
        /// This would then indicate that the LBR AUGRU forward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc may be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param attention_desc Memory descriptor for the attention vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &attention_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::lbr_augru, aprop_kind,
                    algorithm::undef, direction, src_layer_desc, src_iter_desc,
                    nullptr, &attention_desc, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, rnn_flags::undef,
                    0.0f, 0.0f, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an LBR AUGRU forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for an LBR AUGRU forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::lbr_augru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::augru_attention_desc()const
        memory::desc attention_desc() const {
            return rnn_base::augru_attention_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    lbr_augru_forward() = default;

    /// Constructs an LBR AUGRU forward propagation primitive.
    /// @param pd Primitive descriptor for an LBR AUGRU forward propagation
    ///     primitive.
    lbr_augru_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LBR AUGRU forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LBR AUGRU forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lbr_augru_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// LBR AUGRU backward propagation primitive.
struct lbr_augru_backward : public primitive {
    /// Primitive descriptor for an LBR AUGRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for LBR AUGRU backward propagation
        /// primitive.
        ///
        /// The following arguments may point to a zero memory descriptor:
        /// - @p src_iter_desc together with @p diff_src_iter_desc,
        /// - @p bias_desc together with @p diff_bias_desc,
        /// - @p dst_iter_desc together with @p diff_dst_iter_desc.
        ///
        /// This would then indicate that the LBR AUGRU backward propagation
        /// primitive should not use them and should default to zero values
        /// instead.
        ///
        /// @note
        ///     All memory descriptors may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Must be
        ///     #dnnl::prop_kind::backward.
        /// @param direction RNN direction. See @ref dnnl::rnn_direction for
        ///     more info.
        /// @param src_layer_desc Memory descriptor for the input vector.
        /// @param src_iter_desc Memory descriptor for the input recurrent
        ///     hidden state vector.
        /// @param attention_desc Memory descriptor for the attention vector.
        /// @param weights_layer_desc Memory descriptor for the weights
        ///     applied to the layer input.
        /// @param weights_iter_desc Memory descriptor for the weights applied
        ///     to the recurrent input.
        /// @param bias_desc Bias memory descriptor.
        /// @param dst_layer_desc Memory descriptor for the output vector.
        /// @param dst_iter_desc Memory descriptor for the output recurrent
        ///     hidden state vector.
        /// @param diff_src_layer_desc Memory descriptor for the diff of input
        ///     vector.
        /// @param diff_src_iter_desc Memory descriptor for the diff of input
        ///     recurrent hidden state vector.
        /// @param diff_attention_desc Memory descriptor for the diff of
        ///     attention vector.
        /// @param diff_weights_layer_desc Memory descriptor for the diff of
        ///     weights applied to the layer input.
        /// @param diff_weights_iter_desc Memory descriptor for the diff of
        ///     weights applied to the recurrent input.
        /// @param diff_bias_desc Diff bias memory descriptor.
        /// @param diff_dst_layer_desc Memory descriptor for the diff of
        ///     output vector.
        /// @param diff_dst_iter_desc Memory descriptor for the diff of output
        ///     recurrent hidden state vector.
        /// @param hint_fwd_pd Primitive descriptor for an LBR AUGRU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                rnn_direction direction, const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &attention_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_attention_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const lbr_augru_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : rnn_primitive_desc_base(aengine, algorithm::lbr_augru, aprop_kind,
                    algorithm::undef, direction, src_layer_desc, src_iter_desc,
                    nullptr, &attention_desc, weights_layer_desc,
                    weights_iter_desc, nullptr, nullptr, bias_desc,
                    dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
                    diff_src_iter_desc, nullptr, &diff_attention_desc,
                    diff_weights_layer_desc, diff_weights_iter_desc, nullptr,
                    nullptr, diff_bias_desc, diff_dst_layer_desc,
                    diff_dst_iter_desc, nullptr, rnn_flags::undef, 0.0f, 0.0f,
                    hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for an LBR AUGRU backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for an LBR AUGRU backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::lbr_augru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()const
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()const
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::augru_attention_desc()const
        memory::desc attention_desc() const {
            return rnn_base::augru_attention_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()const
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()const
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()const
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()const
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()const
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()const
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()const
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_augru_attention_desc()const
        memory::desc diff_attention_desc() const {
            return rnn_base::diff_augru_attention_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()const
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()const
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()const
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()const
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()const
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc dnnl::primitive_desc_base::get_cell_kind()const
        algorithm get_cell_kind() const { return base::get_cell_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_direction()const
        rnn_direction get_direction() const { return base::get_direction(); }
    };

    /// Default constructor. Produces an empty object.
    lbr_augru_backward() = default;

    /// Constructs an LBR AUGRU backward propagation primitive.
    /// @param pd Primitive descriptor for an LBR AUGRU backward propagation
    ///     primitive.
    lbr_augru_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an LBR AUGRU backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for an LBR AUGRU backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    lbr_augru_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_rnn

/// @addtogroup dnnl_api_shuffle Shuffle
///
/// A primitive to shuffle tensor data along an axis.
///
/// @sa @ref dev_guide_shuffle in developer guide
///
/// @{

/// Shuffle forward propagation primitive.
struct shuffle_forward : public primitive {
    /// Primitive descriptor for a shuffle forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a shuffle forward propagation
        /// primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param axis The axis along which the data is shuffled.
        /// @param group_size Shuffle group size.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                int axis, int group_size,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_shuffle_forward_primitive_desc_create(
                    &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                    src_desc.get(), dst_desc.get(), axis, group_size,
                    attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a shuffle "
                        "forward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a shuffle forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a shuffle forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::shuffle,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_axis()const
        int get_axis() const { return base::get_axis(); }

        /// @copydoc dnnl::primitive_desc_base::get_group_size()const
        memory::dim get_group_size() const { return base::get_group_size(); }
    };

    /// Default constructor. Produces an empty object.
    shuffle_forward() = default;

    /// Constructs a shuffle forward propagation primitive.
    /// @param pd Primitive descriptor for a shuffle forward propagation
    ///     primitive.
    shuffle_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a shuffle forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a shuffle forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    shuffle_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Shuffle backward propagation primitive.
struct shuffle_backward : public primitive {
    /// Primitive descriptor for a shuffle backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a shuffle backward propagation
        /// primitive.
        ///
        /// @param aengine Engine to use.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param axis The axis along which the data is shuffled.
        /// @param group_size Shuffle group size.
        /// @param hint_fwd_pd Primitive descriptor for a shuffle forward
        ///     propagation primitive. It is used as a hint for deciding which
        ///     memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, int axis, int group_size,
                const shuffle_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_shuffle_backward_primitive_desc_create(
                    &pd, aengine.get(), diff_src_desc.get(),
                    diff_dst_desc.get(), axis, group_size, hint_fwd_pd.get(),
                    attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a shuffle "
                        "backward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a shuffle backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a shuffle backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::shuffle,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_axis()const
        int get_axis() const { return base::get_axis(); }

        /// @copydoc dnnl::primitive_desc_base::get_group_size()const
        memory::dim get_group_size() const { return base::get_group_size(); }
    };

    /// Default constructor. Produces an empty object.
    shuffle_backward() = default;

    /// Constructs a shuffle backward propagation primitive.
    /// @param pd Primitive descriptor for a shuffle backward propagation
    ///     primitive.
    shuffle_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a shuffle backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a shuffle backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    shuffle_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_shuffle

/// @addtogroup dnnl_api_binary Binary
///
/// A primitive to perform tensor operations over two tensors.
///
/// @sa @ref dev_guide_binary in developer guide
///
/// @{

/// Elementwise binary operator primitive.
struct binary : public primitive {
    /// Primitive descriptor for an elementwise binary operator primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Elementwise binary algorithm.
        /// @param src0 Memory descriptor for source tensor #0.
        /// @param src1 Memory descriptor for source tensor #1.
        /// @param dst Memory descriptor for destination tensor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src0, const memory::desc &src1,
                const memory::desc &dst,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_binary_primitive_desc_create(&pd,
                    aengine.get(), dnnl::convert_to_c(aalgorithm), src0.get(),
                    src1.get(), dst.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a binary "
                        "operation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a binary primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a binary primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::binary) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc(int)const
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// Returns the memory descriptor for source #0.
        memory::desc src0_desc() const { return base::src_desc(0); }

        /// Returns the memory descriptor for source #1.
        memory::desc src1_desc() const { return base::src_desc(1); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }
    };

    /// Default constructor. Produces an empty object.
    binary() = default;

    /// Constructs an elementwise binary operation primitive.
    /// @param pd Primitive descriptor for an elementwise binary operation
    ///     primitive.
    binary(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs an elementwise binary operation primitive from a cache blob.
    /// @param pd Primitive descriptor for an elementwise binary operation
    ///     primitive.
    /// @param cache_blob Cache blob.
    binary(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_binary

/// @addtogroup dnnl_api_matmul Matrix Multiplication
///
/// A primitive to perform matrix-matrix multiplication. The batched mode
/// is supported with 3D tensors.
///
/// @sa @ref dev_guide_matmul in developer guide
///
///
/// @{

/// Matrix multiplication (matmul) primitive.
struct matmul : public primitive {
    /// Primitive descriptor for a matmul primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a matmul primitive
        ///     without bias.
        ///
        /// @param aengine Engine to use.
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, src_desc, weights_desc, nullptr, dst_desc,
                    attr, allow_empty) {}

        /// Constructs a primitive descriptor for a matmul primitive with bias.
        ///
        /// @param aengine Engine to use.
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        /// @param bias_desc Memory descriptor for bias.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, src_desc, weights_desc, &bias_desc,
                    dst_desc, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a matmul primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a matmul primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::matmul) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return query_md(query::src_md, 0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()const
        memory::desc weights_desc() const {
            return query_md(query::weights_md, 0);
        }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()const
        memory::desc bias_desc() const {
            return query_md(query::weights_md, 1);
        }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return query_md(query::dst_md, 0); }

    private:
        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc *bias_desc,
                const memory::desc &dst_desc, const primitive_attr &attr,
                bool allow_empty) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_matmul_primitive_desc_create(&pd,
                    aengine.get(), src_desc.get(), weights_desc.get(),
                    optional_arg(bias_desc), dst_desc.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a matmul "
                        "primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    matmul() = default;

    /// Constructs a matmul primitive.
    /// @param pd Primitive descriptor for a matmul primitive.
    matmul(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a matmul primitive from a cache blob.
    /// @param pd Primitive descriptor for a matmul primitive.
    /// @param cache_blob Cache blob.
    matmul(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_matmul

/// @addtogroup dnnl_api_resampling Resampling
///
/// A primitive to compute resampling operation on 1D, 2D or 3D data tensor
/// using Nearest Neighbor, or Linear (Bilinear, Trilinear) interpolation
/// method.
///
/// @sa @ref dev_guide_resampling in developer guide
///
/// @{

/// Resampling forward propagation.
struct resampling_forward : public primitive {
    /// Primitive descriptor for a resampling forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a resampling forward
        ///     propagation primitive using source and destination memory
        ///     descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, nullptr, src_desc,
                    &dst_desc, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a resampling forward
        ///     propagation primitive using source memory descriptor and
        ///     factors.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param src_desc Source memory descriptor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const std::vector<float> &factors,
                const memory::desc &src_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, &factors,
                    src_desc, nullptr, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a resampling forward
        ///     propagation primitive.
        ///
        /// @note
        ///     The destination memory descriptor may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const std::vector<float> &factors,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aprop_kind, aalgorithm, &factors,
                    src_desc, &dst_desc, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a resampling forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a resampling forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::resampling,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

    private:
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const std::vector<float> *factors,
                const memory::desc &src_desc, const memory::desc *dst_desc,
                const primitive_attr &attr, bool allow_empty) {

            if (factors)
                memory::validate_dims(*factors, src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_resampling_forward_primitive_desc_create(&pd,
                            aengine.get(), dnnl::convert_to_c(aprop_kind),
                            convert_to_c(aalgorithm), optional_arg(factors),
                            src_desc.get(), optional_arg(dst_desc), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "resampling forward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    resampling_forward() = default;

    /// Constructs a resampling forward propagation primitive.
    /// @param pd Primitive descriptor for a resampling forward propagation
    ///     primitive.
    resampling_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a resampling forward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a resampling forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    resampling_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Resampling backward propagation primitive.
struct resampling_backward : public primitive {
    /// Primitive descriptor for resampling backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a resampling backward
        ///     propagation primitive using source and destination memory
        ///     descriptors.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param hint_fwd_pd Primitive descriptor for a resampling
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const resampling_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, nullptr, diff_src_desc,
                    diff_dst_desc, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for resampling backward
        ///     propagation primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm resampling algorithm kind: either
        ///     #dnnl::algorithm::resampling_nearest, or
        ///     #dnnl::algorithm::resampling_linear
        /// @param factors Vector of scaling factors for spatial dimension.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param hint_fwd_pd Primitive descriptor for a resampling
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const std::vector<float> &factors,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const resampling_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
            : primitive_desc(aengine, aalgorithm, &factors, diff_src_desc,
                    diff_dst_desc, hint_fwd_pd, attr, allow_empty) {}

        /// Constructs a primitive descriptor for a resampling backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a resampling backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::resampling,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

    private:
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const std::vector<float> *factors,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const resampling_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr, bool allow_empty) {

            if (factors)
                memory::validate_dims(*factors, diff_src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status
                    = dnnl_resampling_backward_primitive_desc_create(&pd,
                            aengine.get(), convert_to_c(aalgorithm),
                            optional_arg(factors), diff_src_desc.get(),
                            diff_dst_desc.get(), hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "resampling backward propagation primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    resampling_backward() = default;

    /// Constructs a resampling backward propagation primitive.
    /// @param pd Primitive descriptor for a resampling backward propagation
    ///     primitive.
    resampling_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a resampling backward propagation primitive from a cache
    ///     blob.
    /// @param pd Primitive descriptor for a resampling backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    resampling_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_resampling

/// @addtogroup dnnl_api_pooling Pooling
///
/// A primitive to perform max or average pooling with dilation.
///
/// @sa @ref dev_guide_pooling in developer guide
///
/// @{

/// Pooling forward propagation primitive.
struct pooling_forward : public primitive {
    /// Primitive descriptor for a pooling forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for pooling forward propagation
        ///     primitive.
        ///
        /// Arrays @p strides, @p kernel, @p dilation, @p padding_l
        /// and @p padding_r contain values for spatial dimensions only and
        /// hence must have the same number of elements as there are spatial
        /// dimensions. The order of values is the same as in the tensor:
        /// depth (for 3D tensors), height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #dnnl::algorithm::pooling_max,
        ///     #dnnl::algorithm::pooling_avg_include_padding,
        ///     or #dnnl::algorithm::pooling_avg_exclude_padding.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param dilation Array of dilations for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &kernel, const memory::dims &dilation,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            memory::validate_dims(strides, src_desc.get_ndims() - 2);
            memory::validate_dims(kernel, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_l, src_desc.get_ndims() - 2);
            memory::validate_dims(padding_r, src_desc.get_ndims() - 2);
            memory::validate_dims(dilation, src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_pooling_forward_primitive_desc_create(
                    &pd, aengine.get(), dnnl::convert_to_c(aprop_kind),
                    convert_to_c(aalgorithm), src_desc.get(), dst_desc.get(),
                    &strides[0], &kernel[0], &dilation[0], &padding_l[0],
                    &padding_r[0], attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a descriptor for a pooling forward "
                        "propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a pooling forward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::pooling,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_strides()const
        memory::dims get_strides() const { return base::get_strides(); }

        /// @copydoc dnnl::primitive_desc_base::get_kernel()const
        memory::dims get_kernel() const { return base::get_kernel(); }

        /// @copydoc dnnl::primitive_desc_base::get_dilations()const
        memory::dims get_dilations() const { return base::get_dilations(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_l()const
        memory::dims get_padding_l() const { return base::get_padding_l(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_r()const
        memory::dims get_padding_r() const { return base::get_padding_r(); }
    };

    /// Default constructor. Produces an empty object.
    pooling_forward() = default;

    /// Constructs a pooling forward propagation primitive.
    ///
    /// @param pd Primitive descriptor for a pooling forward propagation
    ///     primitive.
    pooling_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a pooling forward propagation primitive from a cache blob.
    ///
    /// @param pd Primitive descriptor for a pooling forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    pooling_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// Pooling backward propagation primitive.
struct pooling_backward : public primitive {
    /// Primitive descriptor for a pooling backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling backward propagation
        ///     primitive.
        ///
        /// Arrays @p strides, @p kernel, @p dilation, @p padding_l
        /// and @p padding_r contain values for spatial dimensions only and
        /// hence must have the same number of elements as there are spatial
        /// dimensions. The order of values is the same as in the tensor:
        /// depth (for 3D tensors), height (for 3D and 2D tensors), and width.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm Pooling algorithm kind: either
        ///     #dnnl::algorithm::pooling_max,
        ///     #dnnl::algorithm::pooling_avg_include_padding,
        ///     or #dnnl::algorithm::pooling_avg_exclude_padding.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param strides Vector of strides for spatial dimension.
        /// @param kernel Vector of kernel spatial dimensions.
        /// @param dilation Array of dilations for spatial dimension.
        /// @param padding_l Vector of padding values for low indices for each
        ///     spatial dimension `([[front,] top,] left)`.
        /// @param padding_r Vector of padding values for high indices for
        ///     each spatial dimension `([[back,] bottom,] right)`.
        /// @param hint_fwd_pd Primitive descriptor for a pooling
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &kernel, const memory::dims &dilation,
                const memory::dims &padding_l, const memory::dims &padding_r,
                const pooling_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            memory::validate_dims(strides, diff_src_desc.get_ndims() - 2);
            memory::validate_dims(kernel, diff_src_desc.get_ndims() - 2);
            memory::validate_dims(padding_l, diff_src_desc.get_ndims() - 2);
            memory::validate_dims(padding_r, diff_src_desc.get_ndims() - 2);
            memory::validate_dims(dilation, diff_src_desc.get_ndims() - 2);

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_pooling_backward_primitive_desc_create(
                    &pd, aengine.get(), convert_to_c(aalgorithm),
                    diff_src_desc.get(), diff_dst_desc.get(), &strides[0],
                    &kernel[0], &dilation[0], &padding_l[0], &padding_r[0],
                    hint_fwd_pd.get(), attr.get());
            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a descriptor for a pooling backward "
                        "propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a pooling backward propagation
        /// primitive from a C API primitive descriptor that must have a
        /// matching kind.
        ///
        /// @param pd C API primitive descriptor for a pooling backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::pooling,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()const
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }

        /// @copydoc dnnl::primitive_desc_base::get_strides()const
        memory::dims get_strides() const { return base::get_strides(); }

        /// @copydoc dnnl::primitive_desc_base::get_kernel()const
        memory::dims get_kernel() const { return base::get_kernel(); }

        /// @copydoc dnnl::primitive_desc_base::get_dilations()const
        memory::dims get_dilations() const { return base::get_dilations(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_l()const
        memory::dims get_padding_l() const { return base::get_padding_l(); }

        /// @copydoc dnnl::primitive_desc_base::get_padding_r()const
        memory::dims get_padding_r() const { return base::get_padding_r(); }
    };

    /// Default constructor. Produces an empty object.
    pooling_backward() = default;

    /// Constructs a pooling backward propagation primitive.
    ///
    /// @param pd Primitive descriptor for a pooling backward propagation
    ///     primitive.
    pooling_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a pooling backward propagation primitive from a cache blob.
    ///
    /// @param pd Primitive descriptor for a pooling backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    pooling_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_pooling

/// @addtogroup dnnl_api_prelu PReLU
///
/// PReLU primitive
/// A primitive to perform PReLU (leaky ReLU with trainable alpha parameter)
///
/// @sa @ref dev_guide_prelu in developer guide
///
/// @{

/// PReLU forward propagation primitive.
struct prelu_forward : public primitive {
    /// Primitive descriptor for a PReLU forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a PReLU forward propagation
        /// primitive.
        ///
        /// @param aengine Engine to use.
        /// @param aprop_kind Propagation kind. Possible values are
        ///     #dnnl::prop_kind::forward_training, and
        ///     #dnnl::prop_kind::forward_inference.
        /// @param src_desc Source memory descriptor.
        /// @param weight_desc Alpha parameters memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &weight_desc,
                const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_prelu_forward_primitive_desc_create(&pd,
                    aengine.get(), dnnl::convert_to_c(aprop_kind),
                    src_desc.get(), weight_desc.get(), dst_desc.get(),
                    attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a prelu "
                        "forward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a prelu forward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a prelu forward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::prelu,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }
    };

    /// Default constructor. Produces an empty object.
    prelu_forward() = default;

    /// Constructs a prelu forward propagation primitive.
    /// @param pd Primitive descriptor for a prelu forward propagation
    ///     primitive.
    prelu_forward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a prelu forward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a prelu forward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    prelu_forward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// PReLU backward propagation primitive.
struct prelu_backward : public primitive {
    /// Primitive descriptor for prelu backward propagation.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a descriptor for a PReLU backward propagation
        /// primitive.
        ///
        /// @param aengine Engine to use.
        /// @param src_desc Source memory descriptor.
        /// @param weight_desc Alpha parameters memory descriptor.
        /// @param diff_src_desc Diff source memory descriptor.
        /// @param diff_weights_desc Diff alpha parameters memory descriptor.
        /// @param diff_dst_desc Diff destination memory descriptor.
        /// @param hint_fwd_pd Primitive descriptor for a PReLU
        ///     forward propagation primitive. It is used as a hint for
        ///     deciding which memory format to use.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &weight_desc,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const prelu_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_prelu_backward_primitive_desc_create(
                    &pd, aengine.get(), src_desc.get(), weight_desc.get(),
                    diff_src_desc.get(), diff_weights_desc.get(),
                    diff_dst_desc.get(), hint_fwd_pd.get(), attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a prelu "
                        "backward propagation primitive");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a prelu backward
        /// propagation primitive from a C API primitive descriptor that must
        /// have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a prelu backward
        ///     propagation primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::prelu,
                    dnnl::prop_kind::backward) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()const
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()const
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_prop_kind()const
        prop_kind get_prop_kind() const { return base::get_prop_kind(); }
    };

    /// Default constructor. Produces an empty object.
    prelu_backward() = default;

    /// Constructs a prelu backward propagation primitive.
    /// @param pd Primitive descriptor for a prelu backward propagation
    ///     primitive.
    prelu_backward(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a prelu backward propagation primitive from a cache blob.
    /// @param pd Primitive descriptor for a prelu backward propagation
    ///     primitive.
    /// @param cache_blob Cache blob.
    prelu_backward(
            const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_prelu

/// @addtogroup dnnl_api_reduction Reduction
///
/// A primitive to compute reduction operation on data tensor
/// using min, max, mul, sum, mean and norm_lp operations.
///
/// @sa @ref dev_guide_reduction in developer guide
///
/// @{

/// Reduction.
struct reduction : public primitive {
    /// Primitive descriptor for a reduction primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a reduction primitive using
        ///     algorithm specific parameters, source and destination memory
        ///     descriptors.
        ///
        /// @note
        ///     Destination memory descriptor may be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        /// @param aengine Engine to use.
        /// @param aalgorithm reduction algorithm kind. Possible values:
        ///     #dnnl_reduction_max, #dnnl_reduction_min, #dnnl_reduction_sum,
        ///     #dnnl_reduction_mul, #dnnl_reduction_mean,
        ///     #dnnl_reduction_norm_lp_max, #dnnl_reduction_norm_lp_sum,
        ///     #dnnl_reduction_norm_lp_power_p_max,
        ///     #dnnl_reduction_norm_lp_power_p_sum.
        /// @param p algorithm specific parameter.
        /// @param eps algorithm specific parameter.
        /// @param src_desc Source memory descriptor.
        /// @param dst_desc Destination memory descriptor.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                float p, float eps, const primitive_attr &attr = default_attr(),
                bool allow_empty = false) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_reduction_primitive_desc_create(&pd,
                    aengine.get(), convert_to_c(aalgorithm), src_desc.get(),
                    dst_desc.get(), p, eps, attr.get());

            if (!allow_empty)
                error::wrap_c_api(status,
                        "could not create a primitive descriptor for a "
                        "reduction primitive descriptor");
            reset(pd);
        }

        /// Constructs a primitive descriptor for a reduction primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a reduction primitive.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::reduction) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()const
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()const
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::get_p()const
        float get_p() const { return base::get_p(); }

        /// @copydoc dnnl::primitive_desc_base::get_epsilon()const
        float get_epsilon() const { return base::get_epsilon(); }

        /// @copydoc dnnl::primitive_desc_base::get_algorithm()const
        algorithm get_algorithm() const { return base::get_algorithm(); }
    };

    /// Default constructor. Produces an empty object.
    reduction() = default;

    /// Constructs a reduction primitive.
    /// @param pd Primitive descriptor for a reduction primitive.
    reduction(const primitive_desc &pd) : primitive(pd) {}

    /// Constructs a reduction primitive from a cache blob.
    /// @param pd Primitive descriptor for a reduction primitive.
    /// @param cache_blob Cache blob.
    reduction(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
        : primitive(pd, cache_blob) {}
};

/// @} dnnl_api_reduction

/// @} dnnl_api_primitives

/// @addtogroup dnnl_api_service Service
///
/// A set of functions that aid in oneDNN debugging and profiling.
///
/// @{

/// @copydoc dnnl_version_t
using version_t = dnnl_version_t;

/// Status values returned by the library functions.
enum class status {
    /// @copydoc dnnl_success
    success = dnnl_success,
    /// @copydoc dnnl_out_of_memory
    out_of_memory = dnnl_out_of_memory,
    /// @copydoc dnnl_invalid_arguments
    invalid_arguments = dnnl_invalid_arguments,
    /// @copydoc dnnl_unimplemented
    unimplemented = dnnl_unimplemented,
    /// @copydoc dnnl_last_impl_reached
    last_impl_reached = dnnl_last_impl_reached,
    /// @copydoc dnnl_runtime_error
    runtime_error = dnnl_runtime_error,
    /// @copydoc dnnl_not_required
    not_required = dnnl_not_required,
};

/// @copydoc dnnl_set_verbose()
inline status set_verbose(int level) {
    return static_cast<status>(dnnl_set_verbose(level));
}

/// @copydoc dnnl_version()
inline const version_t *version() {
    return dnnl_version();
}

/// Returns the floating-point math mode that will be used by default
/// for all subsequently created primitives.
///
/// @returns Output FP math mode.
inline fpmath_mode get_default_fpmath_mode() {
    dnnl_fpmath_mode_t mode;
    error::wrap_c_api(dnnl_get_default_fpmath_mode(&mode),
            "could not get a default fpmath mode");
    return static_cast<fpmath_mode>(mode);
}

/// @copydoc dnnl_set_default_fpmath_mode()
inline status set_default_fpmath_mode(fpmath_mode mode) {
    return static_cast<status>(
            dnnl_set_default_fpmath_mode(convert_to_c(mode)));
}

/// @copydoc dnnl_set_jit_dump()
inline status set_jit_dump(int enable) {
    return static_cast<status>(dnnl_set_jit_dump(enable));
}

/// @copydoc dnnl_set_jit_profiling_flags()
inline status set_jit_profiling_flags(unsigned flags) {
    return static_cast<status>(dnnl_set_jit_profiling_flags(flags));
}

/// @copydoc dnnl_set_jit_profiling_jitdumpdir()
inline status set_jit_profiling_jitdumpdir(const std::string &dir) {
    return static_cast<status>(dnnl_set_jit_profiling_jitdumpdir(dir.c_str()));
}

/// @copydoc dnnl_cpu_isa_t
enum class cpu_isa {
    /// @copydoc dnnl_cpu_isa_default
    isa_default = dnnl_cpu_isa_default,
    /// @copydoc dnnl_cpu_isa_sse41
    sse41 = dnnl_cpu_isa_sse41,
    /// @copydoc dnnl_cpu_isa_avx
    avx = dnnl_cpu_isa_avx,
    /// @copydoc dnnl_cpu_isa_avx2
    avx2 = dnnl_cpu_isa_avx2,
    /// @copydoc dnnl_cpu_isa_avx2_vnni
    avx2_vnni = dnnl_cpu_isa_avx2_vnni,
    /// @copydoc dnnl_cpu_isa_avx2_vnni_2
    avx2_vnni_2 = dnnl_cpu_isa_avx2_vnni_2,
    /// @copydoc dnnl_cpu_isa_avx512_core
    avx512_core = dnnl_cpu_isa_avx512_core,
    /// @copydoc dnnl_cpu_isa_avx512_core_vnni
    avx512_core_vnni = dnnl_cpu_isa_avx512_core_vnni,
    /// @copydoc dnnl_cpu_isa_avx512_core_bf16
    avx512_core_bf16 = dnnl_cpu_isa_avx512_core_bf16,
    /// @copydoc dnnl_cpu_isa_avx10_1_512
    avx10_1_512 = dnnl_cpu_isa_avx10_1_512,
    /// @copydoc dnnl_cpu_isa_avx512_core_fp16
    avx512_core_fp16 = dnnl_cpu_isa_avx512_core_fp16,
    /// @copydoc dnnl_cpu_isa_avx10_1_512_amx
    avx10_1_512_amx = dnnl_cpu_isa_avx10_1_512_amx,
    /// @copydoc dnnl_cpu_isa_avx512_core_amx
    avx512_core_amx = dnnl_cpu_isa_avx512_core_amx,
    /// @copydoc dnnl_cpu_isa_avx10_1_512_amx_fp16
    avx10_1_512_amx_fp16 = dnnl_cpu_isa_avx10_1_512_amx_fp16,
    /// @copydoc dnnl_cpu_isa_avx512_core_amx_fp16
    avx512_core_amx_fp16 = dnnl_cpu_isa_avx512_core_amx_fp16,
};

/// @copydoc dnnl_set_max_cpu_isa()
inline status set_max_cpu_isa(cpu_isa isa) {
    return static_cast<status>(
            dnnl_set_max_cpu_isa(static_cast<dnnl_cpu_isa_t>(isa)));
}

/// @copydoc dnnl_get_effective_cpu_isa()
inline cpu_isa get_effective_cpu_isa() {
    return static_cast<cpu_isa>(dnnl_get_effective_cpu_isa());
}

/// @copydoc dnnl_cpu_isa_hints_t
enum class cpu_isa_hints {
    /// @copydoc dnnl_cpu_isa_no_hints
    no_hints = dnnl_cpu_isa_no_hints,
    /// @copydoc dnnl_cpu_isa_prefer_ymm
    prefer_ymm = dnnl_cpu_isa_prefer_ymm,
};

/// @copydoc dnnl_set_cpu_isa_hints()
inline status set_cpu_isa_hints(cpu_isa_hints isa_hints) {
    return static_cast<status>(dnnl_set_cpu_isa_hints(
            static_cast<dnnl_cpu_isa_hints_t>(isa_hints)));
}

/// @copydoc dnnl_get_cpu_isa_hints()
inline cpu_isa_hints get_cpu_isa_hints() {
    return static_cast<cpu_isa_hints>(dnnl_get_cpu_isa_hints());
}

/// @} dnnl_api_service

#ifdef DNNL_EXPERIMENTAL_PROFILING
/// @addtogroup dnnl_api_profiling Profiling
/// @{

/// Profiling data kind.
enum class profiling_data_kind {
    /// Undefined profiling data kind.
    undef = dnnl_profiling_data_kind_undef,
    /// Data kind to query an execution time in nanoseconds.
    time = dnnl_profiling_data_kind_time,
};

/// Resets a profiler's state.
///
/// @param stream Stream associated with the profiler.
inline void reset_profiling(stream &stream) {
    error::wrap_c_api(
            dnnl_reset_profiling(stream.get()), "could not reset profiling");
}

/// Returns requested profiling data. The profiling data accumulates for each
/// primitive execution. The size of the vector will be equal to the number
/// of executions since the last `dnnl::reset_profiling` call.
///
/// The profiling data can be reset by calling #dnnl::reset_profiling.
///
/// @note
///     It is required to wait for all submitted primitives to complete
///     using #dnnl::stream::wait prior to querying profiling data.
///
/// @param stream Stream that was used for executing a primitive that
///     is being profiled.
/// @param data_kind Profiling data kind to query.
///
/// @returns A vector with the requested profiling data.
inline std::vector<uint64_t> get_profiling_data(
        stream &stream, profiling_data_kind data_kind) {
    int num_entries = 0;
    error::wrap_c_api(
            dnnl_query_profiling_data(stream.get(),
                    static_cast<dnnl_profiling_data_kind_t>(data_kind),
                    &num_entries, nullptr),
            "could not get number of entries for profiling data");

    if (num_entries == 0) return {};

    std::vector<uint64_t> data(num_entries);
    error::wrap_c_api(
            dnnl_query_profiling_data(stream.get(),
                    static_cast<dnnl_profiling_data_kind_t>(data_kind),
                    &num_entries, data.data()),
            "could not get profiling data");
    return data;
}

/// @} dnnl_api_profiling
#endif

/// @addtogroup dnnl_api_primitive_cache Primitive Cache
///
/// A set of functions that provide primitive cache control.
///
/// @{

/// Returns the number of primitives that can be held in the primitive cache
/// at the same time.
inline int get_primitive_cache_capacity() {
    int result = 0;
    error::wrap_c_api(dnnl_get_primitive_cache_capacity(&result),
            "could not get primitive cache capacity");
    return result;
}

/// @copydoc dnnl_set_primitive_cache_capacity(int capacity)
inline void set_primitive_cache_capacity(int capacity) {
    error::wrap_c_api(dnnl_set_primitive_cache_capacity(capacity),
            "could not set primitive cache capacity");
}

/// @} dnnl_api_primitive_cache

/// @addtogroup dnnl_api_blas BLAS functions
///
/// A subset of Basic Linear Algebra (BLAS) functions that perform
/// matrix-matrix multiplication.
///
/// @{

/// @copydoc dnnl_sgemm()
inline status sgemm(char transa, char transb, dnnl_dim_t M, dnnl_dim_t N,
        dnnl_dim_t K, float alpha, const float *A, dnnl_dim_t lda,
        const float *B, dnnl_dim_t ldb, float beta, float *C, dnnl_dim_t ldc) {
    return static_cast<status>(dnnl_sgemm(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}

/// @copydoc dnnl_gemm_u8s8s32()
inline status gemm_u8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const uint8_t *A,
        dnnl_dim_t lda, uint8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co) {
    return static_cast<status>(dnnl_gemm_u8s8s32(transa, transb, offsetc, M, N,
            K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co));
}

/// @copydoc dnnl_gemm_s8s8s32()
inline status gemm_s8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const int8_t *A,
        dnnl_dim_t lda, int8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co) {
    return static_cast<status>(dnnl_gemm_s8s8s32(transa, transb, offsetc, M, N,
            K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co));
}

/// @} dnnl_api_blas

// implementation section

/// @cond DO_NOT_DOCUMENT_THIS
inline primitive::primitive(const_dnnl_primitive_desc_t c_pd) {
    dnnl_primitive_t result;
    error::wrap_c_api(dnnl_primitive_create(&result, c_pd),
            "could not create a primitive");
    reset(result);
}

inline primitive::primitive(const_dnnl_primitive_desc_t c_pd,
        const std::vector<uint8_t> &cache_blob) {
    dnnl_primitive_t result;
    size_t size = cache_blob.size();
    const uint8_t *cache_blob_data = cache_blob.data();
    error::wrap_c_api(dnnl_primitive_create_from_cache_blob(
                              &result, c_pd, size, cache_blob_data),
            "could not create a primitive from a cache blob");
    reset(result);
}

inline primitive::primitive(const primitive_desc &pd) : primitive(pd.get()) {}
inline primitive::primitive(
        const primitive_desc &pd, const std::vector<uint8_t> &cache_blob)
    : primitive(pd.get(), cache_blob) {}

inline void primitive::execute(const stream &astream,
        const std::unordered_map<int, memory> &args) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get(true)});

    error::wrap_c_api(dnnl_primitive_execute(get(), astream.get(),
                              (int)c_args.size(), c_args.data()),
            "could not execute a primitive");
}

/// @endcond

#undef DNNL_DEFINE_BITMASK_OPS

} // namespace dnnl

/// oneAPI namespace

/// The oneAPI namespace.
/// Contains the oneapi::dnnl namespace as an alias to the ::dnnl namespace.
namespace oneapi {
// Note: without this guard, doxygen warns of potentially recursive namespace
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// oneDNN alias namespace
namespace dnnl = ::dnnl;
#endif
} // namespace oneapi

/// @} dnnl_api

#endif /* ONEAPI_DNNL_DNNL_HPP */
