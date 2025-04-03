/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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

#ifndef CPU_SIMPLE_Q10N_HPP
#define CPU_SIMPLE_Q10N_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace q10n {

template <typename data_t, typename acc_t>
inline typename utils::enable_if<!nstl::is_integral<data_t>::value,
        typename utils::remove_reference<acc_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    return v;
}

template <typename data_t, typename acc_t>
inline typename utils::enable_if<nstl::is_integral<data_t>::value,
        typename utils::remove_reference<acc_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    acc_t lbound = (acc_t)nstl::numeric_limits<data_t>::lowest();
    // Pick up a modified version of max value when do f32 -> s32.
    acc_t ubound = types::max_value<acc_t>(data_traits<data_t>::data_type);
    if (v < lbound) v = lbound;
    if (v > ubound) v = ubound;
    return v;
}

template <>
inline uint8_t saturate<int8_t, uint8_t>(const uint8_t &x) {
    return x <= 127u ? x : 127;
}

template <>
inline int8_t saturate<uint8_t, int8_t>(const int8_t &x) {
    return x >= 0 ? x : 0;
}

template <typename out_t>
inline typename utils::enable_if<nstl::is_integral<out_t>::value,
        typename utils::remove_reference<out_t>::type>::type
out_round(float v) {
    return (out_t)math::mxcsr_cvt(v);
}

template <typename out_t>
inline typename utils::enable_if<!nstl::is_integral<out_t>::value,
        typename utils::remove_reference<out_t>::type>::type
out_round(float v) {
    return v;
}

template <typename out_t, typename acc_t = float>
inline out_t saturate_and_round(acc_t f) {
    return out_round<out_t>(saturate<out_t, acc_t>(f));
}

/* Quantization with alpha == 1 and beta == 0 */
template <typename in_t, typename out_t, typename enabled = void>
struct qz_a1b0 {
    out_t operator()(in_t in) { return saturate_and_round<out_t>((float)in); }
};

template <typename in_t, typename out_t>
struct qz_a1b0<in_t, out_t,
        typename utils::enable_if<true && nstl::is_integral<in_t>::value
                && !is_subset<in_t, out_t>::value>::type> {
    out_t operator()(in_t in) { return saturate<out_t>(in); }
};

template <typename in_t, typename out_t>
struct qz_a1b0<in_t, out_t,
        typename utils::enable_if<is_subset<in_t, out_t>::value>::type> {
    out_t operator()(in_t in) { return (out_t)in; }
};

/* Quantization with alpha == 1 */
template <typename in_t, typename out_t>
struct qz_a1 {
    out_t operator()(in_t in, out_t out, float beta) {
        return saturate_and_round<out_t>((float)in + beta * out);
    }
};

template <typename in_t>
struct qz_a1<in_t, float> {
    float operator()(in_t in, float out, float beta) {
        return (float)in + beta * out;
    }
};

/* Quantization with beta == 0 */
template <typename in_t, typename out_t>
struct qz_b0 {
    out_t operator()(in_t in, float alpha) {
        return saturate_and_round<out_t>(alpha * in);
    }
};

template <typename in_t>
struct qz_b0<in_t, float> {
    float operator()(in_t in, float alpha) { return alpha * in; }
};

/* Quantization */
template <typename in_t, typename out_t>
struct qz {
    out_t operator()(in_t in, out_t out, float alpha, float beta) {
        return saturate_and_round<out_t>(alpha * in + (beta ? beta * out : 0));
    }
};

template <typename in_t>
struct qz<in_t, float> {
    float operator()(in_t in, float out, float alpha, float beta) {
        return alpha * in + (beta ? beta * out : 0);
    }
};

template <>
struct qz<bfloat16_t, bfloat16_t> {
    float operator()(bfloat16_t in, bfloat16_t out, float alpha, float beta) {
        return (bfloat16_t)(alpha * (float)in + (beta ? beta * (float)out : 0));
    }
};

template <>
struct qz<float, bfloat16_t> {
    float operator()(float in, bfloat16_t out, float alpha, float beta) {
        return (bfloat16_t)(alpha * in + (beta ? beta * out : 0));
    }
};

template <>
struct qz<float16_t, float16_t> {
    float operator()(float16_t in, float16_t out, float alpha, float beta) {
        return (float16_t)(alpha * (float)in + (beta ? beta * (float)out : 0));
    }
};

template <>
struct qz<float, float16_t> {
    float operator()(float in, float16_t out, float alpha, float beta) {
        return (float16_t)(alpha * in + (beta ? beta * out : 0));
    }
};

} // namespace q10n
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
