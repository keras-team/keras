/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#ifndef CPU_GEMM_F32_GEMM_UTILS_F32_HPP
#define CPU_GEMM_F32_GEMM_UTILS_F32_HPP

#include <cstddef>

namespace dnnl {
namespace impl {
namespace cpu {

namespace gemm_utils {
template <typename T, bool isTransA, bool isTransB>
struct gemm_traits {};

template <bool isTransA, bool isTransB>
struct gemm_traits<double, isTransA, isTransB> {
    static constexpr dim_t m = 8;
    static constexpr dim_t n = 6;
    static constexpr dim_t BM = 4032;
    static constexpr dim_t BN = isTransA ? 96 : 192;
    static constexpr dim_t BK = isTransB ? 96 : 512;
};

template <bool isTransA, bool isTransB>
struct gemm_traits<float, isTransA, isTransB> {
    static constexpr dim_t m = 16;
    static constexpr dim_t n = 6;
    static constexpr dim_t BM = 4032;
    static constexpr dim_t BN = isTransA ? 96 : 48;
    static constexpr dim_t BK = isTransB ? 96 : 256;
};

template <bool isTransA, bool isTransB>
struct gemm_traits<bfloat16_t, isTransA, isTransB> {
    static constexpr dim_t m = 32;
    static constexpr dim_t n = 6;
    static constexpr dim_t BM = 4032;
    static constexpr dim_t BN = isTransA ? 96 : 48;
    static constexpr dim_t BK = isTransB ? 96 : 256;
};

template <typename T>
using unroll_factor = gemm_traits<T, false, false>;

template <typename data_t>
void sum_two_matrices(dim_t m, dim_t n, data_t *__restrict p_src, dim_t ld_src,
        data_t *__restrict p_dst, dim_t ld_dst);

void calc_nthr_nocopy_avx512_common(dim_t m, dim_t n, dim_t k, int nthrs,
        int *nthrs_m, int *nthrs_n, int *nthrs_k, dim_t *BM, dim_t *BN,
        dim_t *BK);

void calc_nthr_nocopy_avx(dim_t m, dim_t n, dim_t k, int nthrs, int *nthrs_m,
        int *nthrs_n, int *nthrs_k, dim_t *BM, dim_t *BN, dim_t *BK);

void partition_unit_diff(
        int ithr, int nthr, dim_t n, dim_t *t_offset, dim_t *t_block);
}; // namespace gemm_utils

} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif // CPU_GEMM_F32_GEMM_UTILS_F32_HPP
