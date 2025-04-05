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

#ifndef CPU_X64_GEMM_GEMM_PACK_STORAGE_HPP
#define CPU_X64_GEMM_GEMM_PACK_STORAGE_HPP

#include <cstdint>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/x64/gemm/gemm_threading.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

enum struct matrix_id { a, b };

struct gemm_pack_storage_t {
    gemm_threading_t &threading() { return header->threading; }
    matrix_id &which() { return header->which; }
    bool &has_row_sums() { return header->has_row_sums; }
    bool &has_col_sums() { return header->has_col_sums; }

    const gemm_threading_t &threading() const { return header->threading; }
    const matrix_id &which() const { return header->which; }
    const bool &has_row_sums() const { return header->has_row_sums; }
    const bool &has_col_sums() const { return header->has_col_sums; }

    size_t size() const { return header->size; }
    void *get() const { return static_cast<void *>(base); }
    void set(void *data) {
        base = static_cast<char *>(data);
        header = static_cast<header_t *>(data);
    }

    bool single_nocopy() const {
        return (threading().copy == copy_type::no_copy);
    }

    int nthr() const { return single_nocopy() ? 1 : threading().nthrs(); }

    int nslice() const {
        return (which() == matrix_id::a)
                ? threading().nthrs_m * threading().nthrs_k
                : threading().nthrs_n * threading().nthrs_k;
    }

    template <typename data_type>
    gemm_pack_storage_t(data_type *data_, bool header_set_ = true)
        : base(nullptr)
        , header(nullptr)
        , matrix_header(nullptr)
        , sums_header(nullptr)
        , header_set(header_set_) {
        reset((void *)data_);
    }

    gemm_pack_storage_t()
        : base(nullptr)
        , header(nullptr)
        , matrix_header(nullptr)
        , sums_header(nullptr)
        , header_set(true) {}

    std::tuple<int, int> thread_slice_info(int ithr) const {
        assert(ithr < nthr());

        bool is_a = (which() == matrix_id::a);
        auto nthr_inner = is_a ? threading().nthrs_m : threading().nthrs_n;

        auto ithr_i = ithr % threading().nthrs_m;
        auto ithr_jk = ithr / threading().nthrs_m;
        auto ithr_j = ithr_jk % threading().nthrs_n;
        auto ithr_k = ithr_jk / threading().nthrs_n;

        auto ithr_inner = is_a ? ithr_i : ithr_j;
        auto ithr_outer = ithr_k;
        auto ithr_slice = is_a ? ithr_j : ithr_i;

        auto id = ithr_outer * nthr_inner + ithr_inner;

        return std::make_tuple(id, ithr_slice);
    }

    int thread_to_slice(int ithr) const {
        return std::get<0>(thread_slice_info(ithr));
    }

    bool is_first_thread_in_slice(int ithr) const {
        return (std::get<1>(thread_slice_info(ithr)) == 0);
    }

    template <typename data_type>
    data_type *row_sums(int ithr, dim_t r0, dim_t cblock) const {
        if (!has_row_sums()) return NULL;
        auto id = thread_to_slice(ithr);
        return get_block<data_type>(sums_header->slice[id], r0, cblock);
    }

    template <typename data_type>
    data_type *col_sums(int ithr, dim_t rblock, dim_t c0) const {
        if (!has_col_sums()) return NULL;
        auto id = thread_to_slice(ithr);
        return get_block<data_type>(sums_header->slice[id], rblock, c0);
    }

    template <typename data_type>
    data_type *matrix(int ithr, dim_t r0, dim_t c0) const {
        auto id = thread_to_slice(ithr);
        return get_block<data_type>(matrix_header->slice[id], r0, c0);
    }

    template <typename data_type>
    data_type *matrix(int ithr) const {
        assert(!matrix_header->slice[thread_to_slice(ithr)].packed);
        return matrix<data_type>(ithr, 0, 0);
    }

    template <typename data_type>
    data_type *matrix() const {
        assert(single_nocopy());
        return matrix<data_type>(0);
    }

    bool get_nocopy(int ithr, int &trans, dim_t &ld, dim_t &td) const {
        auto id = thread_to_slice(ithr);
        return matrix_header->slice[id].get_nocopy(trans, ld, td);
    }

    bool get_nocopy(int &trans, dim_t &ld, dim_t &td) const {
        if (!single_nocopy()) return false;
        return get_nocopy(0, trans, ld, td);
    }

    void get_blocking(int ithr, dim_t &block_r, dim_t &block_c) const {
        auto id = thread_to_slice(ithr);
        matrix_header->slice[id].get_blocking(block_r, block_c);
    }

    void set_blocking(
            int ithr, dim_t rows, dim_t cols, dim_t block_r, dim_t block_c) {

        auto id = thread_to_slice(ithr);
        auto nblk_r = (block_r == 0) ? 0 : utils::div_up(rows, block_r);
        auto nblk_c = (block_c == 0) ? 0 : utils::div_up(cols, block_c);

        matrix_header->slice[id].set_blocking(nblk_r, nblk_c, block_r, block_c);

        if (has_row_sums())
            sums_header->slice[id].set_blocking(nblk_r, nblk_c, block_r, 1);
        else
            sums_header->slice[id].set_blocking(nblk_r, nblk_c, 1, block_c);
    }

    void set_nocopy(int ithr, int trans, dim_t ld, dim_t td) {
        auto id = thread_to_slice(ithr);
        matrix_header->slice[id].set_nocopy(trans, ld, td);
    }

    void setup(int max_nthr, bool has_row_sums = false,
            bool has_col_sums = false) {

        assert(!(has_row_sums && has_col_sums));

        auto sz_mh = matrix_header_size(max_nthr);
        auto sz_h = header_size();

        header->has_row_sums = has_row_sums;
        header->has_col_sums = has_col_sums;
        header->off_matrix = sz_h;
        header->off_sums = sz_h + sz_mh;
        total_header_size = sz_h + sz_mh * 2;

        header->size = 0;

        header_set = true;

        reset(get());

        for (int id = 0; id < max_nthr; id++) {
            matrix_header->slice[id].set_blocking(0, 0, 0, 0);
            sums_header->slice[id].set_blocking(0, 0, 0, 0);
        }
    }

    template <typename matrix_dt, typename sums_dt>
    void finalize() {
        assert(total_header_size > 0);
        size_t cur_off = total_header_size;

        matrix_header->finalize<matrix_dt>(cur_off, nslice());
        if (has_row_sums() || has_col_sums())
            sums_header->finalize<sums_dt>(cur_off, nslice());

        header->size = cur_off;

        /* Compute kernels overrun to preload data. */
        header->size += align_data;
    }

protected:
    char *base;

    struct header_t {
        matrix_id which;
        bool has_row_sums;
        bool has_col_sums;
        size_t off_matrix, off_sums;
        size_t size;
        gemm_threading_t threading; /* if packed */
    } * header;

    struct slice_header_t {
        bool packed;
        int trans;
        dim_t nblk_r, nblk_c;
        dim_t block_r, block_c;
        size_t off_data;

        template <typename data_type>
        size_t block_size() const {
            return utils::rnd_up(
                    block_r * block_c * sizeof(data_type), align_data);
        }

        template <typename data_type>
        size_t block_offset(dim_t r0, dim_t c0, bool col_major) const {
            assert((r0 % block_r) == 0);
            assert((c0 % block_c) == 0);

            auto rb = r0 / block_r;
            auto cb = c0 / block_c;
            auto mb = col_major ? rb + cb * nblk_r : cb + rb * nblk_c;

            return block_size<data_type>() * mb;
        }

        template <typename data_type>
        size_t size() const {
            return block_size<data_type>() * nblk_r * nblk_c;
        }

        void set_blocking(
                dim_t nblk_r_, dim_t nblk_c_, dim_t block_r_, dim_t block_c_) {
            packed = true;
            nblk_r = nblk_r_;
            nblk_c = nblk_c_;
            block_r = block_r_;
            block_c = block_c_;
        }

        void set_nocopy(int trans_, dim_t ld, dim_t td) {
            packed = false;
            trans = trans_;
            block_r = ld;
            block_c = td;
            nblk_r = 1;
            nblk_c = 1;
        }

        void get_blocking(dim_t &block_r_, dim_t &block_c_) const {
            block_r_ = block_r;
            block_c_ = block_c;
        }

        bool get_nocopy(int &trans_, dim_t &ld, dim_t &td) const {
            if (!packed) {
                trans_ = trans;
                ld = block_r;
                td = block_c;
            }
            return !packed;
        }

        template <typename data_type>
        void finalize(size_t &cur_off) {
            cur_off = utils::rnd_up(cur_off, align_data);
            off_data = cur_off;
            cur_off += size<data_type>();
        }
    };

    struct matrix_header_t {
        dim_t ld; /* if not packed */
        slice_header_t slice[1]; /* array of size nthr, if packed */

        template <typename data_type>
        void finalize(size_t &cur_off, int nslices) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
            // This, I hope, is a temporary workaround...
            // The reason for this special case is that in case of threadpool
            // threading this function may be called to estimate the amount of
            // memory needed when no threading information is actually
            // available. Hence, it needs to provide an upper bound.
            size_t max_off = cur_off;
            for (int id = 0; id < nslices; id++) {
                slice[id].finalize<data_type>(cur_off);
                if (id == 0) {
                    // Assume that slice[0] is the largest one.
                    size_t slice0_size = cur_off - max_off;
                    max_off += slice0_size * dnnl_get_max_threads();
                }
            }
            if (!threadpool_utils::get_active_threadpool() && nslices)
                // The std::max is a paranoid check for the case when slice[0]
                // is not actually the largest one. Probably a crash will
                // happen anyways...
                cur_off = std::max(cur_off, max_off);
#else
            for (int id = 0; id < nslices; id++)
                slice[id].finalize<data_type>(cur_off);
#endif
        }
    } * matrix_header, *sums_header;

    size_t total_header_size = 0;

    static constexpr auto align_headers = 0x20;
    static constexpr auto align_data = 0x1000;

    static size_t header_size() {
        return utils::rnd_up(sizeof(header_t), align_headers);
    }

    static size_t matrix_header_size(int max_nthr) {
        auto sz = sizeof(matrix_header_t)
                + sizeof(slice_header_t) * (max_nthr - 1);

        return utils::rnd_up(sz, align_headers);
    }

    template <typename data_type>
    data_type *get_block(
            const slice_header_t &slice, dim_t r0, dim_t c0) const {
        return reinterpret_cast<data_type *>(base + slice.off_data
                + slice.block_offset<data_type>(r0, c0, col_major()));
    }

    bool col_major() const { return (which() == matrix_id::a); }

    void reset(void *data) {
        set(data);

        if (!header_set) return;

        matrix_header = reinterpret_cast<matrix_header_t *>(
                base + header->off_matrix);
        sums_header
                = reinterpret_cast<matrix_header_t *>(base + header->off_sums);
    }

    bool header_set = true;
};

struct gemm_pack_storage_shell_t : public gemm_pack_storage_t {

    gemm_pack_storage_shell_t(int max_nthr, bool has_row_sums = false,
            bool has_col_sums = false) {
        void *ptr = malloc(shell_size(max_nthr), 64);
        if (ptr) {
            set(ptr);
            setup(max_nthr, has_row_sums, has_col_sums);
        }
    }

    ~gemm_pack_storage_shell_t() { free(get()); }

private:
    static size_t shell_size(int max_nthr) {
        return header_size() + matrix_header_size(max_nthr) * 2;
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
