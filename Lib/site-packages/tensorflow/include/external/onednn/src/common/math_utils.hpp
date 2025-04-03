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

#ifndef COMMON_MATH_UTILS_HPP
#define COMMON_MATH_UTILS_HPP

#include <type_traits>

#include <math.h>
#include <stdint.h>

#include "dnnl_traits.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace math {

// Algorithm is picked from https://en.wikipedia.org/wiki/Primality_test
template <typename T>
inline bool is_prime(T n) {
    static_assert(std::is_integral<T>::value == true, "Not an integral type");

    if (n <= 1 || n % 2 == 0 || n % 3 == 0 || n % 5 == 0) return false;

    const T sqrtn = static_cast<T>(std::sqrt(n));
    // It is enough to check prime divisors up to `sqrt(n)`.
    // All potential prime divisors are represented with `6*i + k` for k={1, 5}.
    for (T i = 1; 6 * i + 5 <= sqrtn; i++) {
        if ((n % (6 * i + 1) == 0) || (n % (6 * i + 5) == 0)) return false;
    }
    return true;
}

template <typename T>
inline T gcd(T a, T b) {
    a = impl::nstl::abs(a);
    b = impl::nstl::abs(b);
    if (a < b) {
        T x = a;
        a = b;
        b = x;
    }

    if (b == 0) return a;

    T r;
    while ((r = a % b) != 0) {
        a = b;
        b = r;
    }

    return b;
}

inline int lcm(int a, int b) {
    a = impl::nstl::abs(a);
    b = impl::nstl::abs(b);
    assert(a > 0 && b > 0);

    return a * b / gcd(a, b);
}

template <typename T>
inline bool is_pow2(const T &v) {
    return (v > 0) && ((v & (v - 1)) == 0);
}

/** returns floor(log2(v)), aka the position of the leftmost non-0 bit */
inline int ilog2q(size_t v) {
    if (v == 0) return -1;

    int p = 0;
#define CP(pw) \
    do { \
        if (v >= (1ull << pw)) { \
            v >>= pw; \
            p += pw; \
        } \
    } while (0)
    CP(32);
    CP(16);
    CP(8);
    CP(4);
    CP(2);
    CP(1);
#undef CP
    return p;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U one_m_square(T x) {
    return (U)(1 - x) * (1 + x);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U x_m_square(T x) {
    return (U)(1 - x) * x;
}

/* activation */

/** rounds @p f to an integer according to the mxcsr register */
inline float mxcsr_round(float f) ATTR_NO_MSAN {
    return nearbyintf(f);
}

/** converts @p f to an integer according to the mxcsr register */
inline int mxcsr_cvt(float f) ATTR_NO_MSAN {
    return (int)mxcsr_round(f);
}

inline float round_fwd(float s) {
    return mxcsr_round(s);
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline typename utils::enable_if<nstl::is_integral<U>::value, U>::type relu_fwd(
        T s, A alpha) {
    return s > 0 ? s : (U)mxcsr_cvt(static_cast<float>(s * alpha));
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline typename utils::enable_if<!nstl::is_integral<U>::value, U>::type
relu_fwd(T s, A alpha) ATTR_NO_MSAN {
    return s > 0 ? s : (U)(s * alpha);
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U relu_bwd(T dd, T s, A alpha) {
    return s > 0 ? dd : (U)(dd * alpha);
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U relu_bwd(T s, A alpha) {
    return s > 0 ? (U)1 : (U)alpha;
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U relu_bwd_use_dst(T dd, T d, A alpha) {
    return d > 0 ? dd : (U)(dd * alpha);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U tanh_fwd(T s) {
    const float e = tanhf((float)s);
    return (U)e;
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U tanh_bwd(T dd, T s) {
    const float e = tanh_fwd<float>((float)s);
    return (U)(dd * (1 - e) * (1 + e));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U tanh_bwd_use_dst(T dd, T d) {
    return (U)(dd * (1 - d) * (1 + d));
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U elu_fwd(T s, A alpha) {
    return s > 0 ? s : (U)(alpha * (::expm1f((float)s)));
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U elu_bwd(T dd, T s, A alpha) {
    return (U)(dd * (s > 0 ? 1 : alpha * ::expf((float)s)));
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U elu_bwd_use_dst(T dd, T d, A alpha) {
    return (U)(dd * (d > 0 ? 1 : d + alpha));
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U square_fwd(T s) {
    return s * s;
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U square_bwd(T dd, T s) {
    return dd * 2 * s;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U abs_fwd(T s) {
    return s > 0 ? s : (U)-s;
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U abs_bwd(T dd, T s) {
    return s > 0 ? dd : s < 0 ? (U)-dd : (U)0;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U sqrt_fwd(T s) {
    return (U)(::sqrtf((float)(s)));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U sqrt_bwd(T dd, T s) {
    return (U)(dd / (2 * ::sqrtf((float)(s))));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U sqrt_bwd_use_dst(T dd, T d) {
    return (U)(dd / (2 * d));
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U linear_fwd(T s, A alpha, A beta) {
    return (U)(alpha * s + beta);
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U linear_bwd(T dd, T s, A alpha, A beta) {
    (void)s;
    (void)beta;
    return (U)(dd * alpha);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U logistic_fwd(T s) {
    // Here we avoid division/inverse by infinity as some architectures have
    // non-standard behavior
    float exp_overflow_bound = 88.72283172607421875;
    float in = (float)-s;
    return in < exp_overflow_bound ? (U)(1.f / (1.f + ::expf(in))) : 0.f;
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U logistic_bwd(T dd, T s) {
    float v = logistic_fwd<float>(s);
    return (U)(dd * v * (1 - v));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U logistic_bwd_use_dst(T dd, T d) {
    return (U)(dd * d * (1 - d));
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U soft_relu_fwd(T s, A alpha) {
    float exp_overflow_bound = 88.72283172607421875;
    float in = (float)s * (float)alpha;
    float v = (in < exp_overflow_bound ? (U)(::log1pf(::expf(in))) : (U)in);
    return (U)(v / alpha);
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U soft_relu_bwd(T dd, T s, A alpha) {
    float in = (float)s * (float)alpha;
    return (U)(dd * logistic_fwd<float>(in));
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U mish_fwd(T s) {
    return s * tanh_fwd(soft_relu_fwd(s, 1.f));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U mish_bwd(T dd, T s) {
    const float tanh = tanh_fwd(soft_relu_fwd(s, 1.f));
    const float srelu_bwd = soft_relu_bwd(1.f, s, 1.f);
    const float derivative = tanh + s * srelu_bwd * (1 - ::powf(tanh, 2.0f));
    return dd * derivative;
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U swish_fwd(T s, A alpha) {
    return (U)(s * logistic_fwd<float>(alpha * s));
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U swish_bwd(T dd, T s, A alpha) {
    float v = logistic_fwd<float>(alpha * s);
    return dd * (v + s * alpha * v * (1 - v));
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U exp_fwd(T s) {
    return (U)(::expf((float)s));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U exp_bwd(T dd, T s) {
    return (U)(dd * (::expf((float)s)));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U exp_bwd_use_dst(T dd, T d) {
    return (U)(dd * d);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U gelu_tanh_fwd(T s) {
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    float v = tanh_fwd(sqrt_2_over_pi * s * (1 + fitting_const * s * s));
    return (U)(0.5 * s * (1. + v));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U gelu_tanh_bwd(T dd, T s) {
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    float g = s * sqrt_2_over_pi * (1 + fitting_const * s * s);
    float dg = sqrt_2_over_pi * (1 + 3 * fitting_const * s * s);
    float v = tanh_fwd(g);
    return (U)(dd * 0.5 * (1. + v) * (1. + s * (1 - v) * dg));
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U log_fwd(T s) {
    return (U)(::logf((float)s));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U log_bwd(T dd, T s) {
    return (U)(dd * (1.f / (float)s));
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U clip_fwd(T s, A alpha, A beta) {
    s = s > alpha ? s : (U)alpha;
    return s > beta ? (U)beta : s;
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U clip_bwd(T dd, T s, A alpha, A beta) {
    return dd * (alpha < s && s <= beta ? 1 : 0);
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U clip_v2_fwd(T s, A alpha, A beta) {
    s = s > alpha ? s : (U)alpha;
    return s < beta ? s : (U)beta;
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U clip_v2_bwd(T dd, T s, A alpha, A beta) {
    return dd * (alpha < s && s < beta ? 1 : 0);
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U clip_v2_bwd_use_dst(T dd, T d, A alpha, A beta) {
    return clip_v2_bwd(dd, d, alpha, beta);
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U pow_fwd(T s, A alpha, A beta) {
    return (U)(alpha * ::powf((float)s, beta));
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U pow_bwd(T dd, T s, A alpha, A beta) {
    if (beta == 0) return 0;

    float v = pow_fwd(s, alpha * beta, beta - 1);
    return (U)(dd * v);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U gelu_erf_fwd(T s) {
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = s * sqrt_2_over_2;
    return (U)(0.5f * s * (1.f + ::erff(v)));
}
template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U gelu_erf_bwd(T dd, T s) {
    const float two_over_sqrt_pi = 1.12837922573089599609375f;
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = s * sqrt_2_over_2;
    return (U)(dd * 0.5f
            * (1.f + ::erff(v) + v * two_over_sqrt_pi * ::expf(-v * v)));
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U hardsigmoid_fwd(T s, A alpha, A beta) {
    float v = alpha * s + beta;
    return v <= 0.f ? 0.f : v >= 1.f ? 1.f : v;
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U hardsigmoid_bwd(T dd, T s, A alpha, A beta) {
    float v = alpha * s + beta;
    return v <= 0.f ? 0.f : v >= 1.f ? 0.f : dd * alpha;
}

template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U hardswish_fwd(T s, A alpha, A beta) {
    return (U)(s * hardsigmoid_fwd(s, alpha, beta));
}
template <typename T, typename A,
        typename U = typename utils::remove_reference<T>::type>
inline U hardswish_bwd(T dd, T s, A alpha, A beta) {
    float v = alpha * s + beta;
    float w = 2.f * alpha * s + beta;
    return v <= 0.f ? 0.f : v >= 1.f ? dd : dd * w;
}

inline bool is_eltwise_ok(
        data_type_t src_dt, alg_kind_t alg, float alpha, float beta) {
    using namespace alg_kind;
    using namespace utils;

    const bool eltwise_use_src
            = one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
                      eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                      eltwise_soft_relu, eltwise_mish, eltwise_logistic,
                      eltwise_exp, eltwise_gelu_tanh, eltwise_hardsigmoid,
                      eltwise_hardswish, eltwise_swish, eltwise_log,
                      eltwise_clip, eltwise_clip_v2, eltwise_pow,
                      eltwise_gelu_erf, eltwise_round)
            && IMPLICATION(
                    one_of(alg, eltwise_clip, eltwise_clip_v2), beta >= alpha)
            && IMPLICATION(alg == eltwise_round, src_dt == dnnl_f32)
            && IMPLICATION(one_of(src_dt, dnnl_s32, dnnl_s8, dnnl_u8),
                    one_of(alg, eltwise_relu, eltwise_linear));

    const bool eltwise_use_dst
            = one_of(alg, eltwise_relu_use_dst_for_bwd,
                      eltwise_tanh_use_dst_for_bwd, eltwise_elu_use_dst_for_bwd,
                      eltwise_sqrt_use_dst_for_bwd,
                      eltwise_logistic_use_dst_for_bwd,
                      eltwise_exp_use_dst_for_bwd,
                      eltwise_clip_v2_use_dst_for_bwd)
            && IMPLICATION(one_of(alg, eltwise_relu_use_dst_for_bwd,
                                   eltwise_elu_use_dst_for_bwd),
                    alpha >= 0)
            && IMPLICATION(
                    alg == eltwise_clip_v2_use_dst_for_bwd, beta >= alpha);

    return eltwise_use_src || eltwise_use_dst;
}

} // namespace math
} // namespace impl
} // namespace dnnl

#endif
