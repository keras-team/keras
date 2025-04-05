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

#ifndef CPU_X64_GEMM_GEMM_INFO_HPP
#define CPU_X64_GEMM_GEMM_INFO_HPP

#include <cstdint>
#include <memory>

#include "common/c_types_map.hpp"

#include "cpu/x64/gemm/gemm_pack_storage.hpp"
#include "cpu/x64/gemm/gemm_threading.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

enum class pack_type { none, pack_a, pack_b };

enum class offset_type {
    none,
    fixed,
    column,
    row,
};

// Indices for kernel arrays. TODO Is it okay to place this here?
enum { no_sum = 0, do_sum = 1 };
enum { no_trans = 0, do_trans = 1, packed = 2 };
enum { no_beta0 = 0, do_beta0 = 1 };
enum { no_alpha1 = 0, do_alpha1 = 1 };

template <typename a_t, typename b_t, typename c_t>
struct gemm_info_t {

    // Interface arguments.
    int transa, transb;
    offset_type offsetc;
    dim_t m, n, k;
    dim_t lda, ldb, ldc;
    const a_t *a;
    const b_t *b;
    c_t *c;
    float alpha, beta;

    int32_t ao;
    int32_t bo;
    const c_t *co;

    pack_type packing;
    gemm_pack_storage_t *pack_dst;
    bool measure_only;
    std::shared_ptr<const gemm_pack_storage_t> a_packed, b_packed;

    // Kernel parameters.
    dim_t um, un, uk, bm, bn, bk;
    dim_t bn_small_k, bk_traditional, blocking_small_k;

    // Gemv parameters
    int swap = false;

    using copy_a_fptr_t = void (*)(const dim_t *m, const dim_t *n,
            const a_t *src, const dim_t *ldsrc, const float *alpha, a_t *dst,
            const dim_t *dummy1, const dim_t *dummy2, c_t *row_col_sum);

    using copy_b_fptr_t = void (*)(const dim_t *m, const dim_t *n,
            const b_t *src, const dim_t *ldsrc, const float *alpha, b_t *dst,
            const dim_t *dummy1, const dim_t *dummy2, c_t *row_col_sum);

    using gemm_fptr_t = void (*)(const dim_t *, const dim_t *, const dim_t *,
            const float *, const a_t *, const b_t *, c_t *, const dim_t,
            const c_t *, const c_t *);

    using gemv_fptr_t = void (*)(const dim_t *, const dim_t *, const float *,
            const a_t *, const dim_t *, const b_t *, const dim_t *, c_t *,
            const dim_t *);

    using gemv_s8s8s32_fptr_t
            = void (*)(const dim_t, const dim_t, const float, const int8_t *,
                    const dim_t, const int8_t *, const float, int32_t *);

    using gemv_s8u8s32_fptr_t
            = void (*)(const dim_t, const dim_t, const float, const int8_t *,
                    const dim_t, const uint8_t *, const float, int32_t *);

    using gemv_u8s8s32_fptr_t
            = void (*)(const dim_t, const dim_t, const float, const uint8_t *,
                    const dim_t, const int8_t *, const float, int32_t *);

    // gemm kernels
    copy_a_fptr_t copyA = nullptr;
    copy_b_fptr_t copyB = nullptr;
    gemm_fptr_t kernel[2][2][2] = {{{nullptr}}};

    // gemv kernels
    gemv_fptr_t gemv_kernel[2] = {nullptr};
    gemv_s8s8s32_fptr_t gemv_s8s8s32_kernel = nullptr;
    gemv_s8u8s32_fptr_t gemv_s8u8s32_kernel = nullptr;
    gemv_u8s8s32_fptr_t gemv_u8s8s32_kernel = nullptr;

    // copyA[trans][sum]
    static copy_a_fptr_t copy_a_kern[2][2];

    // copyB[trans][sum]
    static copy_b_fptr_t copy_b_kern[2][2];

    // kern[beta0][alpha1][col_off][row_off]
    static gemm_fptr_t kern[2][2][2][2];

    // gemv_kern[trans]
    static gemv_fptr_t gemv_kern[2];

    static gemv_s8s8s32_fptr_t gemv_s8s8s32_kern;
    static gemv_s8u8s32_fptr_t gemv_s8u8s32_kern;
    static gemv_u8s8s32_fptr_t gemv_u8s8s32_kern;

    template <bool is_trans>
    static void copy_a_sum_ref(const dim_t *p_k, const dim_t *p_m,
            const a_t *src, const dim_t *p_ld, const float *p_alpha, a_t *dst,
            const dim_t *dummy1, const dim_t *dummy2, c_t *a_row_sum) {

        copy_a_kern[is_trans][no_sum](
                p_k, p_m, src, p_ld, p_alpha, dst, dummy1, dummy2, a_row_sum);

        dim_t k = *p_k;
        dim_t m = *p_m;
        dim_t ld = *p_ld;

        // Calculate op(A) row sum.
        if (!is_trans) {
            PRAGMA_OMP_SIMD()
            for (dim_t i = 0; i < m; i++)
                a_row_sum[i] = 0;

            for (dim_t j = 0; j < k; j++) {
                PRAGMA_OMP_SIMD()
                for (dim_t i = 0; i < m; i++) {
                    a_row_sum[i] += src[i + j * ld];
                }
            }
        } else {
            for (dim_t i = 0; i < m; i++) {
                c_t acc = 0;

                PRAGMA_OMP_SIMD(reduction(+ : acc))
                for (dim_t j = 0; j < k; j++) {
                    acc += src[j + i * ld];
                }

                a_row_sum[i] = acc;
            }
        }
    }

    template <bool is_trans>
    static void copy_b_sum_ref(const dim_t *p_k, const dim_t *p_n,
            const b_t *src, const dim_t *p_ld, const float *alpha, b_t *dst,
            const dim_t *dummy1, const dim_t *dummy2, c_t *b_col_sum) {

        copy_b_kern[is_trans][no_sum](
                p_k, p_n, src, p_ld, alpha, dst, dummy1, dummy2, b_col_sum);

        dim_t k = *p_k;
        dim_t n = *p_n;
        dim_t ld = *p_ld;

        // Calculate op(B) column sum.
        if (!is_trans) {
            for (dim_t j = 0; j < n; j++) {
                c_t acc = 0;

                PRAGMA_OMP_SIMD(reduction(+ : acc))
                for (dim_t i = 0; i < k; i++)
                    acc += src[i + j * ld];

                b_col_sum[j] = acc;
            }
        } else {
            PRAGMA_OMP_SIMD()
            for (dim_t j = 0; j < n; j++)
                b_col_sum[j] = 0;

            for (dim_t i = 0; i < k; i++) {
                PRAGMA_OMP_SIMD()
                for (dim_t j = 0; j < n; j++)
                    b_col_sum[j] += src[j + i * ld];
            }
        }
    }

    bool force_nocopy;

    gemm_info_t(const char *transA, const char *transB, const char *offsetC,
            const dim_t *m, const dim_t *n, const dim_t *k, const float *alpha,
            const a_t *a, const dim_t *lda, const a_t *oa, const b_t *b,
            const dim_t *ldb, const b_t *ob, const float *beta, c_t *c,
            const dim_t *ldc, const c_t *oc, bool force_nocopy,
            pack_type packing, gemm_pack_storage_t *pack_dst,
            bool measure_only);

    bool hasKernels(void);

    void update_blocking(const gemm_threading_t &thread_info);

private:
    void jit_init(void);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_GEMM_INFO_HPP
