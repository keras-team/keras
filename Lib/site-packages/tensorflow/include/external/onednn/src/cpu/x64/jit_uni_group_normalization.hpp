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

#ifndef CPU_X64_JIT_UNI_GROUP_NORMALIZATION_HPP
#define CPU_X64_JIT_UNI_GROUP_NORMALIZATION_HPP

#include "common/primitive.hpp"

#include "cpu/cpu_group_normalization_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_uni_group_normalization_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cpu_group_normalization_fwd_pd_t {
        using cpu_group_normalization_fwd_pd_t::
                cpu_group_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_group_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src_d(src_md());

            VDISPATCH_GNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_GNORM(mayiuse(avx2), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_GNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_GNORM(
                    utils::one_of(src_md()->data_type, f32, bf16, f16, s8, u8)
                            && IMPLICATION(utils::one_of(src_md()->data_type,
                                                   bf16, f16),
                                    mayiuse(avx512_core)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(
                    utils::one_of(dst_md()->data_type, f32, bf16, f16, s8, u8)
                            && IMPLICATION(utils::one_of(dst_md()->data_type,
                                                   bf16, f16),
                                    mayiuse(avx512_core)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(
                    attr()->has_default_values(skip_mask_t::scales_runtime)
                            && attr_scales_ok(),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GNORM(memory_desc_matches_one_of_tag(
                                    *src_md(), ndhwc, nhwc, nwc, nc),
                    VERBOSE_UNSUPPORTED_TAG_S, "src");
            VDISPATCH_GNORM(memory_desc_matches_one_of_tag(
                                    *dst_md(), ndhwc, nhwc, nwc, nc),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");
            VDISPATCH_GNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

            const size_t C_PER_G = C() / G();
            const size_t vlen = isa_max_vlen(get_max_cpu_isa());
            const size_t simd_w
                    = vlen / types::data_type_size(stat_md()->data_type);
            VDISPATCH_GNORM(IMPLICATION(C_PER_G != 1, C_PER_G % simd_w == 0),
                    VERBOSE_INCONSISTENT_DIM, "C", (int)C(), "groups",
                    (int)desc()->groups);
            // C_PER_G should be less than simd_w * unroll_c (which is 6 for var)
            VDISPATCH_GNORM(C_PER_G / simd_w <= 6, VERBOSE_SHAPE_RESTRICTION);

            nthr_ = dnnl_get_max_threads();
            auto scratchpad = scratchpad_registry().registrar();
            if (!stats_is_src()) {
                using namespace memory_tracking::names;
                const size_t stats_size = MB() * C();
                const size_t stats_reduction_buf_sz = stats_size * nthr_;
                scratchpad.template book<float>(
                        key_gnorm_reduction, stats_reduction_buf_sz);
                if (!is_training()) {
                    scratchpad.template book<float>(
                            key_gnorm_tmp_mean, stats_size);
                    scratchpad.template book<float>(
                            key_gnorm_tmp_var, stats_size);
                }
            }

            return status::success;
        }

        int nthr_; // To not exceed the limit in execute used for set up.
    };

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_, kernel_base_t::create(pd())));
        CHECK(safe_ptr_assign(kernel_mean_, kernel_stat_base_t::create(pd())));
        CHECK(safe_ptr_assign(
                kernel_var_, kernel_stat_base_t::create(pd(), true)));
        if (kernel_) CHECK(kernel_->create_kernel());
        if (kernel_mean_) CHECK(kernel_mean_->create_kernel());
        if (kernel_var_) CHECK(kernel_var_->create_kernel());
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    struct kernel_base_t {
        virtual void operator()(const void *src, void *dst, const float *scale,
                const float *shift, float *mean, float *var,
                const float *src_scales, const float *dst_scales,
                const size_t block_size) const = 0;
        static kernel_base_t *create(const group_normalization_pd_t *pd);
        virtual status_t create_kernel() = 0;
        virtual ~kernel_base_t() = default;
    };

    struct kernel_stat_base_t {
        virtual void operator()(
                const void *src, float *mean, size_t block_size) const = 0;
        virtual void operator()(const void *src, const float *mean, float *var,
                size_t block_size) const = 0;
        static kernel_stat_base_t *create(
                const group_normalization_pd_t *pd, bool compute_var = false);
        virtual status_t create_kernel() = 0;
        virtual ~kernel_stat_base_t() = default;
    };

protected:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<kernel_base_t> kernel_;
    std::unique_ptr<kernel_stat_base_t> kernel_mean_;
    std::unique_ptr<kernel_stat_base_t> kernel_var_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
