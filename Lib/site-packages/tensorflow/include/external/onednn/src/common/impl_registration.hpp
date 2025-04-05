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

#ifndef COMMON_IMPL_REGISTRATION_HPP
#define COMMON_IMPL_REGISTRATION_HPP

#include "oneapi/dnnl/dnnl_config.h"

// Workload section

// Note: REG_BWD_D_PK is a dedicated macro for deconv to enable bwd_d conv.
#if BUILD_TRAINING
#define REG_BWD_PK(...) __VA_ARGS__
#define REG_BWD_D_PK(...) __VA_ARGS__
#else
#define REG_BWD_PK(...) \
    { nullptr }
#define REG_BWD_D_PK(...) \
    { nullptr }
#endif

// Primitives section

// Note:
// `_P` is a mandatory suffix for macros. This is to avoid a conflict with
// `REG_BINARY`, Windows-defined macro.

#if BUILD_PRIMITIVE_ALL || BUILD_BATCH_NORMALIZATION
#define REG_BNORM_P(...) __VA_ARGS__
#else
#define REG_BNORM_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_BINARY
#define REG_BINARY_P(...) __VA_ARGS__
#else
#define REG_BINARY_P(...) \
    { nullptr }
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_CONCAT
#define REG_CONCAT_P(...) __VA_ARGS__
#else
#define REG_CONCAT_P(...) \
    { nullptr }
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_CONVOLUTION
#define REG_CONV_P(...) __VA_ARGS__
#else
#define REG_CONV_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_DECONVOLUTION
#define REG_DECONV_P(...) __VA_ARGS__
// This case is special, it requires handling of convolution_bwd_d internally
// since major optimizations are based on convolution implementations.
#ifndef REG_CONV_P
#error "REG_CONV_P is not defined. Check that convolution is defined prior deconvolution."
#else
#undef REG_CONV_P
#define REG_CONV_P(...) __VA_ARGS__
#endif

#ifndef REG_BWD_D_PK
#error "REG_BWD_D_PK is not defined. Dedicated macro was not enabled."
#else
#undef REG_BWD_D_PK
#define REG_BWD_D_PK(...) __VA_ARGS__
#endif

#else // BUILD_PRIMITIVE_ALL || BUILD_DECONVOLUTION
#define REG_DECONV_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_ELTWISE
#define REG_ELTWISE_P(...) __VA_ARGS__
#else
#define REG_ELTWISE_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_GROUP_NORMALIZATION
#define REG_GNORM_P(...) __VA_ARGS__
#else
#define REG_GNORM_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_INNER_PRODUCT
#define REG_IP_P(...) __VA_ARGS__
#else
#define REG_IP_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_LAYER_NORMALIZATION
#define REG_LNORM_P(...) __VA_ARGS__
#else
#define REG_LNORM_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_LRN
#define REG_LRN_P(...) __VA_ARGS__
#else
#define REG_LRN_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_MATMUL
#define REG_MATMUL_P(...) __VA_ARGS__
#else
#define REG_MATMUL_P(...) \
    { nullptr }
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_POOLING
#define REG_POOLING_P(...) __VA_ARGS__
#else
#define REG_POOLING_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_PRELU
#define REG_PRELU_P(...) __VA_ARGS__
#else
#define REG_PRELU_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_REDUCTION
#define REG_REDUCTION_P(...) __VA_ARGS__
#else
#define REG_REDUCTION_P(...) \
    { nullptr }
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_REORDER
#define REG_REORDER_P(...) __VA_ARGS__
#else
#define REG_REORDER_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_RESAMPLING
#define REG_RESAMPLING_P(...) __VA_ARGS__
#else
#define REG_RESAMPLING_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_RNN
#define REG_RNN_P(...) __VA_ARGS__
#else
#define REG_RNN_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_SHUFFLE
#define REG_SHUFFLE_P(...) __VA_ARGS__
#else
#define REG_SHUFFLE_P(...) \
    { nullptr }
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_SOFTMAX
#define REG_SOFTMAX_P(...) __VA_ARGS__
#else
#define REG_SOFTMAX_P(...) \
    {}
#endif

#if BUILD_PRIMITIVE_ALL || BUILD_SUM
#define REG_SUM_P(...) __VA_ARGS__
#else
#define REG_SUM_P(...) \
    { nullptr }
#endif

// Primitive CPU ISA section is in src/cpu/platform.hpp

#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_GEN9
#define REG_GEN9_ISA(...) __VA_ARGS__
#else
#define REG_GEN9_ISA(...)
#endif

#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_GEN11
#define REG_GEN11_ISA(...) __VA_ARGS__
#else
#define REG_GEN11_ISA(...)
#endif

#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_XELP
#define REG_XELP_ISA(...) __VA_ARGS__
#else
#define REG_XELP_ISA(...)
#endif

#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_XEHP
#define REG_XEHP_ISA(...) __VA_ARGS__
#else
#define REG_XEHP_ISA(...)
#endif

#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_XEHPG
#define REG_XEHPG_ISA(...) __VA_ARGS__
#else
#define REG_XEHPG_ISA(...)
#endif

#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_XEHPC
#define REG_XEHPC_ISA(...) __VA_ARGS__
#else
#define REG_XEHPC_ISA(...)
#endif

#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_XE2
#define REG_XE2_ISA(...) __VA_ARGS__
#else
#define REG_XE2_ISA(...)
#endif

#endif
