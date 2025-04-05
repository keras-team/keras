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

#ifndef CPU_X64_GEMM_GEMM_PARTITION_HPP
#define CPU_X64_GEMM_GEMM_PARTITION_HPP

#include <array>
#include <cstdint>
#include <tuple>

#include "common/nstl.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static inline void partition_1d(const int ithr, const int nthrs, const dim_t n,
        dim_t &t_offset, dim_t &t_block) {

    dim_t band = n / nthrs;

    dim_t tail = n - (nthrs - 1) * band;
    if (tail > (band + 1)) band++;
    tail = n - (nthrs - 1) * band;

    if (ithr < (nthrs - 1))
        t_block = band;
    else
        t_block = tail;

    t_offset = ithr * band;

    if (t_offset >= n) {
        t_block = 0;
        t_offset = 0;
    } else if ((t_offset + t_block) > n) {
        t_block = n - t_offset;
    }
}

static inline void partition_2d(const int ithr, int *nthrs, const int ithr_i,
        const int ithr_j, const int nthrs_m, const int nthrs_n, const dim_t m,
        const dim_t n, dim_t &out_m_disp, dim_t &out_m_band, dim_t &out_n_disp,
        dim_t &out_n_band) {

    dim_t m_disp = 0, n_disp = 0;
    dim_t m_band = 0, n_band = 0;

    int m_div = nthrs_m;
    int n_div = nthrs_n;

    dim_t m_bandt = m / m_div; /* size per thread */
    dim_t n_bandt = n / n_div; /* size per thread */
    int first_m_group = m_div - 1;
    int first_n_group = n_div - 1;
    dim_t first_m_val = m_bandt;
    dim_t first_n_val = n_bandt;

    int mthr_used = m_div;
    if (m - (m_div - 1) * m_bandt > m_bandt + 1) {
        if (m - (m_div - 1) * m_bandt > m_div) ++m_bandt;

        first_m_val = m_bandt + 1;
        mthr_used = (int)(m / first_m_val);

        if (mthr_used * first_m_val < m) ++mthr_used;

        first_m_group = mthr_used - 1;
    }

    int nthr_used = n_div;
    if (n - (n_div - 1) * n_bandt > n_bandt + 1) {
        first_n_val = n_bandt + 1;
        nthr_used = (int)(n / first_n_val);

        if (nthr_used * first_n_val < n) ++nthr_used;

        first_n_group = nthr_used - 1;
    }

    *nthrs = mthr_used * nthr_used;

    if (ithr < *nthrs) {
        if (ithr_i < first_m_group) {
            m_band = first_m_val;
            m_disp = ithr_i * first_m_val;
        } else if (ithr_i <= mthr_used - 2) {
            m_band = m_bandt;
            m_disp = first_m_group * first_m_val
                    + (ithr_i - first_m_group) * m_bandt;
        } else {
            m_disp = first_m_group * first_m_val
                    + (mthr_used - 1 - first_m_group) * m_bandt;
            m_band = nstl::max(dim_t(0), m - m_disp);
        }

        if (ithr_j < first_n_group) {
            n_band = first_n_val;
            n_disp = ithr_j * first_n_val;
        } else if (ithr_j <= nthr_used - 2) {
            n_band = n_bandt;
            n_disp = first_n_group * first_n_val
                    + (ithr_j - first_n_group) * n_bandt;
        } else {
            n_disp = first_n_group * first_n_val
                    + (nthr_used - 1 - first_n_group) * n_bandt;
            n_band = nstl::max(dim_t(0), n - n_disp);
        }
        m_disp = nstl::max(nstl::min(m_disp, m - 1), dim_t(0));
        n_disp = nstl::max(nstl::min(n_disp, n - 1), dim_t(0));
    }

    if (ithr < *nthrs) {
        out_m_disp = m_disp;
        out_n_disp = n_disp;
        out_m_band = m_band;
        out_n_band = n_band;
    } else {
        out_m_disp = 0;
        out_n_disp = 0;
        out_m_band = 0;
        out_n_band = 0;
    }

    return;
}

static inline std::tuple<int, int> partition_2d_minblk_with_primes(dim_t m,
        dim_t n, dim_t block_m, dim_t block_n, dim_t min_m, dim_t min_n,
        dim_t um, dim_t un, int nthr, bool use_aspect_ratio) {

    auto part_m = nstl::max(dim_t(1), m / block_m);
    auto part_n = nstl::max(dim_t(1), n / block_n);

    // Quick exit if there are enough partitions in one direction
    // and there is only 1 partition in the other one
    if (part_m == 1 && part_n >= nthr)
        return std::make_tuple(1, nstl::min((int)part_n, nthr));

    if (part_n == 1 && part_m >= nthr)
        return std::make_tuple(nstl::min((int)part_m, nthr), 1);

    auto num_parts = part_m * part_n;

    int nthr_ite = nthr;
    int nthr_m = 1, nthr_n = 1;
    dim_t band_m = m, band_n = n;

    for (auto p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}) {
        bool finished = false;

        while ((nthr_ite % p) == 0 && !finished) {
            nthr_ite /= p;
            auto nthr_m_ite = nthr_m * p;
            auto nthr_n_ite = nthr_n * p;

            auto band_m_ite = band_m / p;
            auto band_n_ite = band_n / p;
            float band_m_ite_f = static_cast<float>(band_m_ite);
            float band_n_ite_f = static_cast<float>(band_n_ite);

            // Try partitioning with block size bm x bn
            auto try_partition = [&](dim_t bm, dim_t bn, bool pick_small) {
                float ratio_m = band_m_ite_f / static_cast<float>(bm);
                float ratio_n = band_n_ite_f / static_cast<float>(bn);
                bool do_m = false, do_n = false;

                if (ratio_m < 1. && ratio_n >= 1.)
                    do_n = true;
                else if (ratio_m >= 1. && ratio_n < 1.)
                    do_m = true;
                else if (ratio_m >= 1. && ratio_n >= 1.) {
                    if (use_aspect_ratio) {
                        float ratio_goal = static_cast<float>(um)
                                / static_cast<float>(un);
                        float try_ratio_m = band_m_ite_f
                                / static_cast<float>(band_n)
                                * (1.f / ratio_goal);
                        float try_ratio_n = static_cast<float>(band_m)
                                / band_n_ite_f * (1.f / ratio_goal);
                        if (pick_small) {
                            // Pick either the smaller or larger ratio as appropriate.
                            ((ratio_m < ratio_n) ? do_m : do_n) = true;
                        } else {
                            // Pick the dimension that will keep as close as possible
                            // to best ratio between m and n.
                            ((nstl::abs(try_ratio_m - 1.)
                                     < nstl::abs(try_ratio_n - 1))
                                            ? do_m
                                            : do_n)
                                    = true;
                        }
                    } else {
                        (((ratio_m < ratio_n) == pick_small) ? do_m : do_n)
                                = true;
                    }
                }

                if (do_m) {
                    // Partition m.
                    nthr_m = nthr_m_ite;
                    band_m = band_m_ite;
                } else if (do_n) {
                    // Partition n.
                    nthr_n = nthr_n_ite;
                    band_n = band_n_ite;
                }

                return do_m || do_n;
            };

            // If we will need min based partitioning do it now
            if (num_parts < nthr) {
                num_parts *= p;
                if (try_partition(min_m, min_n, true)) continue;
            }

            if (try_partition(block_m, block_n, false)) continue;
            if (try_partition(min_m, min_n, true)) continue;

            // Both band_m/n are smaller than min_m/n
            // exit the loops, nothing to partition
            finished = true;
        }

        if (finished) break;
    }

    return std::make_tuple(nthr_m, nthr_n);
}

static inline std::tuple<int, int> partition_2d_minblk(dim_t m, dim_t n,
        dim_t block_m, dim_t block_n, dim_t min_m, dim_t min_n, dim_t um,
        dim_t un, int nthr, bool use_aspect_ratio) {

    auto part_m = nstl::max(dim_t(1), m / min_m);
    auto part_n = nstl::max(dim_t(1), n / min_n);

    // Quick exit if one of the dimensions is too small to partition.
    if (part_m == 1) {
        part_n = nstl::max(dim_t(1), utils::div_up(n, min_n));
        return std::make_tuple(1, nstl::min((int)part_n, nthr));
    }

    if (part_n == 1) {
        part_m = nstl::max(dim_t(1), utils::div_up(m, min_m));
        return std::make_tuple(nstl::min((int)part_m, nthr), 1);
    }

    int nthr_m = 0, nthr_n = 0;
    auto nthr_thresh = nstl::min(0.95 * nthr, (double)(part_m * part_n));

    for (int nthr_new = nthr; nthr_new > nthr / 2; nthr_new--) {
        if (nthr_m * nthr_n >= nthr_thresh) break;
        std::tie(nthr_m, nthr_n)
                = partition_2d_minblk_with_primes(m, n, block_m, block_n, min_m,
                        min_n, um, un, nthr_new, use_aspect_ratio);
    }

    return std::make_tuple(nthr_m, nthr_n);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
