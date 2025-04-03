/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CPU_X64_GEMM_GEMM_UTILS_HPP
#define CPU_X64_GEMM_GEMM_UTILS_HPP

#include <tuple>

#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/utils.hpp"

#include "cpu/x64/gemm/gemm_pack_storage.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace gemm_utils {

static inline std::tuple<int, int> calc_nthr_2d(int nthrs, dim_t m, dim_t n,
        dim_t block_m, dim_t block_n, dim_t small_m, dim_t small_n,
        dim_t &thread_m, dim_t &thread_n) {

    int nthr_m = static_cast<int>(utils::div_up(m, block_m));
    int nthr_n = static_cast<int>(utils::div_up(n, block_n));

    if (nthr_m < 1) nthr_m = 1;
    if (nthr_n < 1) nthr_n = 1;

    float ratio_float = static_cast<float>(nthr_m) / static_cast<float>(nthr_n);

    int ratio = 0;
    if (nthr_m > nthr_n)
        ratio = (int)ratio_float;
    else
        ratio = (int)(1. / ratio_float);

    // scale down nthr_m and nthr_n if they are too large
    while (nthr_m * nthr_n > 4 * nthrs) {
        nthr_m /= 2;
        nthr_n /= 2;
    }

    if (nthr_m < 1) nthr_m = 1;
    if (nthr_n < 1) nthr_n = 1;

    // Simple partition reduction
    int counter = 0;
    while (nthr_m * nthr_n > nthrs) {
        if (nthr_m > nthr_n) {
            if (counter < ratio)
                nthr_m--;
            else {
                nthr_n--;
                counter = -1;
            }
        } else {
            if (counter < ratio)
                nthr_n--;
            else {
                nthr_m--;
                counter = -1;
            }
        }
        counter++;
    }

    // Simple partition increment
    counter = 0;
    while (nthr_m * nthr_n < 0.95 * nthrs) {
        if (nthr_m > nthr_n) {
            if (counter < ratio)
                nthr_m++;
            else {
                nthr_n++;
                counter = -1;
            }
        } else {
            if (counter < ratio)
                nthr_n++;
            else {
                nthr_m++;
                counter = -1;
            }
        }
        counter++;
    }

    // if nothing works out, then this should work
    if ((nthr_m * nthr_n > nthrs)) {

        if (nthr_m <= nthr_n) {
            nthr_m = (int)sqrt((double)nthrs);
            if (nthr_m > utils::div_up(m, small_m))
                nthr_m = static_cast<int>(utils::div_up(m, small_m));
            nthr_n = nthrs / nthr_m;

            while ((nthr_m > 1) && (nthr_m * nthr_n != nthrs)) {
                nthr_m--;
                nthr_n = nthrs / nthr_m;
            }
        } else {
            nthr_n = (int)sqrt((double)nthrs);
            if (nthr_n > utils::div_up(n, small_n))
                nthr_n = static_cast<int>(utils::div_up(n, small_n));
            nthr_m = nthrs / nthr_n;

            while ((nthr_n > 1) && (nthr_m * nthr_n != nthrs)) {
                nthr_n--;
                nthr_m = nthrs / nthr_n;
            }
        }
    }

    thread_m = utils::div_up(m, nthr_m) + small_m - 1;
    thread_n = utils::div_up(n, nthr_n) + small_n - 1;
    thread_m -= thread_m % small_m;
    thread_n -= thread_n % small_n;

    if (thread_m * nthr_m > m)
        nthr_m = static_cast<int>(utils::div_up(m, thread_m));
    if (thread_n * nthr_n > n)
        nthr_n = static_cast<int>(utils::div_up(n, thread_n));

    return std::make_tuple(nthr_m, nthr_n);
}

template <typename T>
static inline dim_t get_ld_padd(const dim_t x) {
    return x != 1 ? utils::rnd_up(x, 2048 / sizeof(T)) + (64 / sizeof(T)) : 1;
}

template <typename mat_t, typename acc_t>
void prep_gemm_pack(bool do_a, int is_trans, dim_t nrows, dim_t ncols,
        gemm_pack_storage_t *pack_dst) {

    auto ld = !is_trans ? get_ld_padd<mat_t>(nrows) : get_ld_padd<mat_t>(ncols);
    auto td = !is_trans ? ncols : nrows;

    // TODO Do we need to use only one thread?
    pack_dst->which() = do_a ? matrix_id::a : matrix_id::b;
    pack_dst->setup(1);
    pack_dst->threading().copy = copy_type::no_copy;
    pack_dst->threading().nthrs_m = 1;
    pack_dst->threading().nthrs_n = 1;
    pack_dst->threading().nthrs_k = 1;
    pack_dst->set_nocopy(0, is_trans, ld, td);
    pack_dst->finalize<mat_t, acc_t>();
}

template <typename T>
dnnl_status_t pack_no_copy(const T *src, dim_t ld_src, dim_t nrows, dim_t ncols,
        int trans_src, float alpha, gemm_pack_storage_t *dst_pack) {

    auto dst = dst_pack->matrix<T>(0);
    int trans_dst;
    dim_t nrows_dst, ncols_dst;
    dim_t ld_dst, td_dst;

    constexpr bool is_f32 = data_traits<T>::data_type == data_type::f32;

    if (!dst_pack->get_nocopy(0, trans_dst, ld_dst, td_dst))
        return dnnl_invalid_arguments;

    if (!trans_dst) {
        nrows_dst = nrows;
        ncols_dst = ncols;
    } else {
        nrows_dst = ncols;
        ncols_dst = nrows;
    }

    if (trans_src == trans_dst) {
        parallel_nd(ncols_dst, [=](dim_t j) {
            auto src_col = src + j * ld_src;
            auto dst_col = dst + j * ld_dst;

            PRAGMA_OMP_SIMD()
            for (dim_t i = 0; i < nrows_dst; i++)
                if (is_f32)
                    dst_col[i] = alpha * src_col[i];
                else
                    dst_col[i] = src_col[i];
        });
    } else {
        // Naive code for now.
        parallel_nd(ncols_dst, [=](dim_t j) {
            auto src_col = src + j;
            auto dst_col = dst + j * ld_dst;

            PRAGMA_OMP_SIMD()
            for (dim_t i = 0; i < nrows_dst; i++)
                if (is_f32)
                    dst_col[i] = alpha * src_col[i * ld_src];
                else
                    dst_col[i] = src_col[i * ld_src];
        });
    }

    return dnnl_success;
}

} // namespace gemm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_GEMM_UTILS_HPP
