/*******************************************************************************
* Copyright 2023 IBM Corporation
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
#ifndef CPU_S390X_KERNELS_PACK_HPP
#define CPU_S390X_KERNELS_PACK_HPP
#include "common/utils.hpp"
#include "cpu/s390x/helpers.h"
namespace dnnl {
namespace impl {
namespace cpu {
namespace s390x {

template <int G, int KK, bool accessSide, typename T, typename DT>
typename utils::enable_if<(KK == 2), void>::type pack_G_KK(int k,
        const T *__restrict src, int srcLD, DT *__restrict dst,
        DT add_val = 0) {
    auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};

    for (int p = 0; p < k / 2; p++) {
        for (int j = 0; j < G; j++) {
            dst[0] = (DT)Src(2 * p, j) + add_val;
            dst[1] = (DT)Src(2 * p + 1, j) + add_val;
            dst += 2;
        }
    }
    int k_2 = k & (-2);
    if ((k & 1) == 1) {
        for (int j = 0; j < G; j++) {
            dst[0] = (DT)Src(k_2, j) + add_val;
            dst[1] = (DT)add_val;
            dst += 2;
        }
    }
}

template <int G, int KK, bool accessSide, typename T, typename DT>
typename utils::enable_if<(KK == 1), void>::type pack_G_KK(
        int k, const T *src, int srcLD, T *dst, DT add_val) {
    auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};

    for (int p = 0; p < k; p++) {
        for (int j = 0; j < G; j++) {
            *dst++ = (DT)Src(p, j) + add_val;
        }
    }
}

template <int N, int G, int KK, bool accessSide, typename T, typename DT>
typename utils::enable_if<(N >= G), void>::type pack_KK_TAILS(const int k,
        const int n, const T *__restrict src, int srcLD, DT *__restrict dst,
        DT add_val = 0) {
    // no need to unroll
}

template <int N, int G, int KK, bool accessSide, typename T, typename DT>
typename utils::enable_if<(N < G), void>::type pack_KK_TAILS(const int k,
        const int n, const T *__restrict src, int srcLD, DT *__restrict dst,
        DT add_val = 0) {
    if (n & N) {

        // for example n&1, n&2 and et cetera
        pack_G_KK<N, KK, accessSide>(k, src, srcLD, dst, add_val);
        int kk = (k + KK - 1) & (-KK);
        auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};
        src = Src.ptr(0, N);
        dst += N * kk;
    }

    pack_KK_TAILS<N * 2, G, KK, accessSide>(k, n, src, srcLD, dst, add_val);
}

template <typename T, typename DT, int G, int KK = 4, bool trans>
void __attribute__((noinline))
pack_K(const int k, const int n, const T *__restrict src, int srcLD,
        DT *__restrict dst, DT add_val = 0) {

    // if k is not divisible by 4 , the rest will be 0-ed
    constexpr bool accessSide = trans ? (!ISLASTINDEX_FAST) : ISLASTINDEX_FAST;
    int kk = (k + KK - 1) & (-KK);
    int nn = n - n % G; // n & (-G); G should be power of 2, so let the
            // compiler to decide
    auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};
    for (int j = 0; j < nn; j += G) {
        pack_G_KK<G, KK, accessSide>(k, Src.ptr(0, j), srcLD, dst, add_val);
        // last dst will not be accessed
        dst += G * kk;
    }

    // if not padded fully
    // unroll with those conditions:
    // k&1 k&2 k&4 k&8 and so on
    pack_KK_TAILS<1, G, KK, accessSide>(
            k, n - nn, Src.ptr(0, nn), srcLD, dst, add_val);
}

} // namespace s390x
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
