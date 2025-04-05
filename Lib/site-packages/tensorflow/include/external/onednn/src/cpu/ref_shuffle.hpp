/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#ifndef CPU_REF_SHUFFLE_HPP
#define CPU_REF_SHUFFLE_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_shuffle_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_shuffle_t : public primitive_t {
    struct pd_t : public cpu_shuffle_pd_t {
        using cpu_shuffle_pd_t::cpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_shuffle_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;

            const memory_desc_wrapper src_d(
                    is_fwd() ? src_md() : diff_src_md());
            const memory_desc_wrapper dst_d(
                    is_fwd() ? dst_md() : diff_dst_md());

            VDISPATCH_SHUFFLE(src_d.data_type() == dst_d.data_type(),
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_SHUFFLE(
                    platform::has_data_type_support(src_d.data_type()),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SHUFFLE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SHUFFLE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SHUFFLE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");

            if (ndims() == 5) {
                dat_tag_ = memory_desc_matches_one_of_tag(
                        *src_d.md_, nCdhw16c, nCdhw8c, nCdhw4c, ncdhw, ndhwc);
            } else if (ndims() == 4) {
                dat_tag_ = memory_desc_matches_one_of_tag(
                        *src_d.md_, nChw16c, nChw8c, nChw4c, nchw, nhwc);
            } else
                dat_tag_ = any;

            return status::success;
        }

        format_tag_t dat_tag_;
    };

    ref_shuffle_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const dim_t axis_size = pd()->axis_size();
        const dim_t group_size = pd()->group_size();
        const dim_t transpose_row
                = pd()->is_fwd() ? group_size : axis_size / group_size;
        const dim_t transpose_col
                = pd()->is_fwd() ? axis_size / group_size : group_size;
        rev_transposed_ = (dim_t *)malloc(
                axis_size * sizeof(dim_t), platform::get_cache_line_size());
        if (rev_transposed_ == nullptr) return dnnl_out_of_memory;
        parallel_nd(transpose_col, transpose_row, [&](dim_t i, dim_t j) {
            rev_transposed_[j * transpose_col + i] = i * transpose_row + j;
        });
        return dnnl_success;
    }

    ~ref_shuffle_t() { free(rev_transposed_); }

    status_t execute(const exec_ctx_t &ctx) const override {
        const memory_desc_wrapper src_d(
                pd()->is_fwd() ? pd()->src_md() : pd()->diff_src_md());
        switch (types::data_type_size(src_d.data_type())) {
            case sizeof(float): return execute_<sizeof(float)>(ctx); break;
            case sizeof(bfloat16_t):
                return execute_<sizeof(bfloat16_t)>(ctx);
                break;
            case sizeof(int8_t): return execute_<sizeof(int8_t)>(ctx); break;
            default: assert(!"unsupported data type size");
        }
        return status::success;
    }

private:
    template <int data_type_size>
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    dim_t *rev_transposed_ = nullptr;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
