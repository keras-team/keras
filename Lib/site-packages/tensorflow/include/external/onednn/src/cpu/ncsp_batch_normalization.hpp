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

#ifndef CPU_NCSP_BATCH_NORMALIZATION_HPP
#define CPU_NCSP_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_batch_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t d_type>
struct ncsp_batch_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        using cpu_batch_normalization_fwd_pd_t::
                cpu_batch_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("ncsp_bnorm:any", ncsp_batch_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;

            VDISPATCH_BNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_BNORM(utils::everyone_is(d_type, src_md()->data_type,
                                    dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(IMPLICATION(is_training(),
                                    platform::has_training_support(d_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");
            VDISPATCH_BNORM((attr()->has_default_values()
                                    || with_relu_post_op(is_training())),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BNORM(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_BNORM(
                    memory_desc_matches_one_of_tag(*src_md(), ncdhw, nchw, ncw),
                    VERBOSE_UNSUPPORTED_TAG_S, "src");

            // BN+Add+Relu fusion is not currently implemented
            VDISPATCH_BNORM(!fuse_norm_add_relu(), VERBOSE_UNSUPPORTED_FEATURE,
                    "sum+relu post-ops configuration is not supported");

            if (is_training() && fuse_norm_relu()) init_default_ws(8);

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();

            return status::success;
        }

        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (!stats_is_src()) {
                scratchpad.template book<acc_data_t>(
                        key_bnorm_reduction, C() * nthr_);

                if (!is_training()) {
                    scratchpad.template book<acc_data_t>(
                            key_bnorm_tmp_mean, C());
                    scratchpad.template book<acc_data_t>(
                            key_bnorm_tmp_var, C());
                }
            }

            if (utils::one_of(d_type, data_type::bf16, data_type::f16)) {
                static constexpr dim_t simd_w = 16;
                const dim_t SP = D() * H() * W();
                const int nbufs = 2;
                const size_t cvt_buf_sz
                        = nbufs * nthr_ * utils::rnd_up(SP, simd_w);
                scratchpad.template book<acc_data_t>(key_bnorm_cvt, cvt_buf_sz);
            }
        }
    };

    typedef typename prec_traits<d_type>::type data_t;
    typedef float acc_data_t;

    ncsp_batch_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    ~ncsp_batch_normalization_fwd_t() {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <data_type_t d_type>
struct ncsp_batch_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_bwd_pd_t {
        using cpu_batch_normalization_bwd_pd_t::
                cpu_batch_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("ncsp_bnorm:any", ncsp_batch_normalization_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;

            VDISPATCH_BNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_BNORM(
                    utils::everyone_is(d_type, src_md()->data_type,
                            diff_dst_md()->data_type, diff_src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(platform::has_training_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");
            VDISPATCH_BNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BNORM(memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
            VDISPATCH_BNORM(
                    memory_desc_matches_one_of_tag(*src_md(), ncdhw, nchw, ncw),
                    VERBOSE_UNSUPPORTED_TAG_S, "src");
            VDISPATCH_BNORM(memory_desc_matches_one_of_tag(
                                    *diff_src_md(), ncdhw, nchw, ncw),
                    VERBOSE_UNSUPPORTED_TAG_S, "diff_src");

            // BN+Add+Relu fusion is not currently implemented
            VDISPATCH_BNORM(!fuse_norm_add_relu(), VERBOSE_UNSUPPORTED_FEATURE,
                    "sum+relu post-ops configuration is not supported");

            if (fuse_norm_relu()) {
                init_default_ws(8);
                VDISPATCH_BNORM(compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);
            }

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();

            return status::success;
        }

        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.template book<acc_data_t>(
                    key_bnorm_reduction, 2 * C() * nthr_);
            const auto pk_is_bwd = desc()->prop_kind == prop_kind::backward;
            size_t ss_size = 0;
            if (!use_scale() || !pk_is_bwd) ss_size += C();
            if (!use_shift() || !pk_is_bwd) ss_size += C();

            if (ss_size)
                scratchpad.template book<acc_data_t>(
                        key_bnorm_tmp_diff_ss, ss_size);

            if (utils::one_of(d_type, data_type::bf16, data_type::f16)) {
                static constexpr dim_t simd_w = 16;
                const dim_t SP = D() * H() * W();
                const int nbufs = 2 + !use_global_stats();
                const size_t cvt_buf_sz
                        = nbufs * nthr_ * utils::rnd_up(SP, simd_w);
                scratchpad.template book<acc_data_t>(key_bnorm_cvt, cvt_buf_sz);
            }
        }
    };

    typedef typename prec_traits<d_type>::type data_t;
    typedef float acc_data_t;

    ncsp_batch_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    ~ncsp_batch_normalization_bwd_t() {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
