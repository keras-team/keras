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

#ifndef COMMON_DNNL_TRAITS_HPP
#define COMMON_DNNL_TRAITS_HPP

#include <assert.h>
#include <stdint.h>

#include "oneapi/dnnl/dnnl.h"

#include "bfloat16.hpp"
#include "c_types_map.hpp"
#include "float16.hpp"
#include "float8.hpp"
#include "nstl.hpp"
#include "opdesc.hpp"
#include "utils.hpp"
#include "z_magic.hpp"

namespace dnnl {
namespace impl {

template <data_type_t>
struct prec_traits {}; /* ::type -> float */
template <typename>
struct data_traits {}; /* ::data_type -> f32 */
template <int>
struct typesize_traits {}; /* ::data_type_size -> f32 */
template <primitive_kind_t>
struct pkind_traits {}; /* ::desc_type, ::query_d */

template <>
struct prec_traits<data_type::f8_e5m2> {
    typedef float8_e5m2_t type;
};
template <>
struct prec_traits<data_type::f8_e4m3> {
    typedef float8_e4m3_t type;
};
template <>
struct prec_traits<data_type::f16> {
    typedef float16_t type;
};
template <>
struct prec_traits<data_type::bf16> {
    typedef bfloat16_t type;
};
template <>
struct prec_traits<data_type::f32> {
    typedef float type;
};
template <>
struct prec_traits<data_type::f64> {
    typedef double type;
};
template <>
struct prec_traits<data_type::s32> {
    typedef int32_t type;
};
template <>
struct prec_traits<data_type::s8> {
    typedef int8_t type;
};
template <>
struct prec_traits<data_type::u8> {
    typedef uint8_t type;
};
template <>
struct prec_traits<data_type::s4> {
    typedef int4_t type;
};
template <>
struct prec_traits<data_type::u4> {
    typedef uint4_t type;
};
template <>
struct prec_traits<data_type::boolean> {
    typedef bool type;
};

template <>
struct data_traits<float8_e5m2_t> {
    static constexpr data_type_t data_type = data_type::f8_e5m2;
};
template <>
struct data_traits<float8_e4m3_t> {
    static constexpr data_type_t data_type = data_type::f8_e4m3;
};
template <>
struct data_traits<float16_t> {
    static constexpr data_type_t data_type = data_type::f16;
};
template <>
struct data_traits<bfloat16_t> {
    static constexpr data_type_t data_type = data_type::bf16;
};
template <>
struct data_traits<float> {
    static constexpr data_type_t data_type = data_type::f32;
};
template <>
struct data_traits<int32_t> {
    static constexpr data_type_t data_type = data_type::s32;
};
template <>
struct data_traits<int8_t> {
    static constexpr data_type_t data_type = data_type::s8;
};
template <>
struct data_traits<uint8_t> {
    static constexpr data_type_t data_type = data_type::u8;
};
template <>
struct data_traits<int4_t> {
    static constexpr data_type_t data_type = data_type::s4;
};
template <>
struct data_traits<uint4_t> {
    static constexpr data_type_t data_type = data_type::u4;
};
template <>
struct data_traits<bool> {
    static constexpr data_type_t data_type = data_type::boolean;
};

template <>
struct typesize_traits<4> {
    typedef float type;
};
template <>
struct typesize_traits<2> {
    typedef int16_t type;
};
template <>
struct typesize_traits<1> {
    typedef uint8_t type;
};

#define PKIND_TRAITS_INST(op) \
    template <> \
    struct pkind_traits<primitive_kind::op> { \
        typedef CONCAT2(op, _desc_t) desc_type; \
    }
PKIND_TRAITS_INST(convolution);
PKIND_TRAITS_INST(deconvolution);
PKIND_TRAITS_INST(shuffle);
PKIND_TRAITS_INST(eltwise);
PKIND_TRAITS_INST(softmax);
PKIND_TRAITS_INST(pooling);
PKIND_TRAITS_INST(prelu);
PKIND_TRAITS_INST(lrn);
PKIND_TRAITS_INST(batch_normalization);
PKIND_TRAITS_INST(group_normalization);
PKIND_TRAITS_INST(layer_normalization);
PKIND_TRAITS_INST(inner_product);
PKIND_TRAITS_INST(rnn);
PKIND_TRAITS_INST(gemm);
PKIND_TRAITS_INST(zero_pad);
PKIND_TRAITS_INST(binary);
PKIND_TRAITS_INST(matmul);
PKIND_TRAITS_INST(resampling);
PKIND_TRAITS_INST(reduction);
#undef PKIND_TRAITS_INST

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
