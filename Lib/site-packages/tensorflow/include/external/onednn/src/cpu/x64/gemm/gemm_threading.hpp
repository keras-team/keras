/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef CPU_X64_GEMM_GEMM_THREADING_HPP
#define CPU_X64_GEMM_GEMM_THREADING_HPP

#include <cstdint>

#include "common/c_types_map.hpp"

#include "cpu/x64/gemm/gemm_partition.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

enum class partition_type { row_1d, col_1d, col_major_2d, mnk_3d };

enum class copy_type {
    nonshared,
    shared_a,
    no_copy,
};

struct gemm_slice_t {
    dim_t off_m, off_n, off_k;
    dim_t m, n, k;
    int ithr_m, ithr_n, ithr_k;
};

struct gemm_threading_t {
    gemm_threading_t() {};

    int nthrs_m, nthrs_n, nthrs_k;
    dim_t block_m, block_n, block_k; // Blocking sizes (-1 = default)
    dim_t thread_m, thread_n, thread_k; // Thread matrix sizes (-1 = default)
    partition_type partition;
    copy_type copy;

    int nthrs() const { return nthrs_m * nthrs_n * nthrs_k; }

    friend bool operator==(
            const gemm_threading_t &t1, const gemm_threading_t &t2) {
        return (t1.nthrs_m == t2.nthrs_m && t1.nthrs_n == t2.nthrs_n
                && t1.nthrs_k == t2.nthrs_k && t1.partition == t2.partition
                && t1.copy == t2.copy);
    }

    friend bool operator!=(
            const gemm_threading_t &t1, const gemm_threading_t &t2) {
        return !(t1 == t2);
    }

    gemm_slice_t get_thread_slice(int ithr, dim_t m, dim_t n, dim_t k) const {

        dim_t off_m = 0, off_n = 0, off_k = 0;
        dim_t size_m = m, size_n = n, size_k = k;
        int ithr_m = 0, ithr_n = 0, ithr_k = 0;

        switch (partition) {
            case partition_type::row_1d:
                ithr_m = ithr;
                partition_1d(ithr, nthrs(), m, off_m, size_m);
                break;

            case partition_type::col_1d:
                ithr_n = ithr;
                partition_1d(ithr, nthrs(), n, off_n, size_n);
                break;

            case partition_type::col_major_2d: {
                int nthr_eff = nthrs();
                ithr_m = ithr % nthrs_m;
                ithr_n = ithr / nthrs_m;

                partition_2d(ithr, &nthr_eff, ithr_m, ithr_n, nthrs_m, nthrs_n,
                        m, n, off_m, size_m, off_n, size_n);
                break;
            }

            case partition_type::mnk_3d: {
                assert(thread_m > 0 && thread_n > 0 && thread_k > 0);
                ithr_m = ithr % nthrs_m;
                ithr_n = (ithr / nthrs_m) % nthrs_n;
                ithr_k = (ithr / nthrs_m) / nthrs_n;

                off_m = ithr_m * thread_m;
                off_n = ithr_n * thread_n;
                off_k = ithr_k * thread_k;

                size_m = nstl::min(thread_m, m - off_m);
                size_n = nstl::min(thread_n, n - off_n);
                size_k = nstl::min(thread_k, k - off_k);
                break;
            }
        }

        return {off_m, off_n, off_k, size_m, size_n, size_k, ithr_m, ithr_n,
                ithr_k};
    }

    int thr_k_stride() const { return nthrs_m * nthrs_n; }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
