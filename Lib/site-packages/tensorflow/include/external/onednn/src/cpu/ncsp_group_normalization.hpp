/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef CPU_NCSP_GROUP_NORMALIZATION_HPP
#define CPU_NCSP_GROUP_NORMALIZATION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_group_normalization_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ncsp_group_normalization_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cpu_group_normalization_fwd_pd_t {
        using cpu_group_normalization_fwd_pd_t::
                cpu_group_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("ncsp_gnorm:any", ncsp_group_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            VDISPATCH_GNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_GNORM(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
            VDISPATCH_GNORM(
                    utils::one_of(src_md()->data_type, f32, bf16, f16, s8, u8)
                            && platform::has_data_type_support(
                                    src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(
                    utils::one_of(dst_md()->data_type, f32, bf16, f16, s8, u8)
                            && platform::has_data_type_support(
                                    dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(
                    check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(memory_desc_matches_one_of_tag(
                                    *src_md(), ncdhw, nchw, ncw, nc),
                    VERBOSE_UNSUPPORTED_TAG_S, "src");
            VDISPATCH_GNORM(memory_desc_matches_one_of_tag(
                                    *dst_md(), ncdhw, nchw, ncw, nc),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");
            VDISPATCH_GNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_GNORM(
                    attr()->has_default_values(skip_mask_t::scales_runtime)
                            && attr_scales_ok(),
                    VERBOSE_UNSUPPORTED_ATTR);
            nthr_ = dnnl_get_max_threads();

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();

            const auto src_dt = src_md()->data_type;
            const auto dst_dt = dst_md()->data_type;
            if (!utils::everyone_is(data_type::f32, src_dt, dst_dt)) {
                const size_t cvt_buf_sz = nthr_ * cvt_per_thread_size_;
                scratchpad.template book<float>(key_gnorm_cvt, cvt_buf_sz);
            }
            return status::success;
        }

        static constexpr size_t cvt_per_thread_size_ = 16;
        int nthr_; // To not exceed the limit in execute used for set up.
    };

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
