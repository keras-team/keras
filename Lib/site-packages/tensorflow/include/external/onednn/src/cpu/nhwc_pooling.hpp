/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef CPU_NHWC_POOLING_HPP
#define CPU_NHWC_POOLING_HPP

#include <assert.h>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace nhwc_pooling {
size_t strided_offset(const int _n, const size_t _sn, const int _d,
        const size_t _sd, const int _h, const size_t _sh, const int _w,
        const size_t _sw);
}

template <data_type_t d_type>
struct nhwc_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nhwc:any", nhwc_pooling_fwd_t);

        status_t init(engine_t *engine) {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

            using namespace prop_kind;
            using namespace alg_kind;
            VDISPATCH_POOLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(utils::one_of(desc()->alg_kind, pooling_max,
                                      pooling_avg_include_padding,
                                      pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(utils::everyone_is(d_type, src_md()->data_type,
                                      dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");
            VDISPATCH_POOLING(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, d_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_POOLING(
                    ref_post_ops_t::primitive_kind_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(
                    memory_desc_matches_tag(*src_md(), desired_fmt_tag),
                    VERBOSE_UNSUPPORTED_TAG_S, "src");
            VDISPATCH_POOLING(
                    memory_desc_matches_tag(*dst_md(), desired_fmt_tag),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");
            VDISPATCH_POOLING(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            const bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                init_default_ws();
            }

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();

            return status::success;
        }

        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (src_md()->data_type != data_type::f32) {
                const size_t bf16cvt_sz_ = IC() * nthr_;
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<float>(
                        key_pool_src_bf16cvt, bf16cvt_sz_);
                scratchpad.template book<float>(
                        key_pool_dst_bf16cvt, bf16cvt_sz_);
            }
        }
    };

    nhwc_pooling_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    using data_t = typename prec_traits<d_type>::type;
    using ker_data_t = typename prec_traits<data_type::f32>::type;

    status_t init(engine_t *engine) override {
        ref_post_ops_
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops_) return status::out_of_memory;
        CHECK(ref_post_ops_->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    void array_div_by_const(const int n, const ker_data_t *src,
            const size_t num, ker_data_t *dst) const;
    void array_add(const int n, const ker_data_t *src, ker_data_t *dst) const;
    void array_nhwc_max(const int n, ker_data_t *dst, const ker_data_t *src,
            unsigned char *ws, const size_t ws_offset, const data_type_t ws_dt,
            const int index) const;
    void array_nhwc_initialize(const int n, ker_data_t *dst, unsigned char *ws,
            const size_t ws_offset, const data_type_t ws_dt) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<ref_post_ops_t> ref_post_ops_;
};

template <impl::data_type_t d_type>
struct nhwc_pooling_bwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nhwc:any", nhwc_pooling_bwd_t);

        status_t init(engine_t *engine) {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

            using namespace prop_kind;
            using namespace alg_kind;
            VDISPATCH_POOLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(utils::one_of(desc()->alg_kind, pooling_max,
                                      pooling_avg_include_padding,
                                      pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(
                    utils::everyone_is(d_type, diff_dst_md()->data_type,
                            diff_src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_POOLING(
                    memory_desc_matches_tag(*diff_dst_md(), desired_fmt_tag),
                    VERBOSE_UNSUPPORTED_TAG_S, "diff_dst");
            VDISPATCH_POOLING(
                    memory_desc_matches_tag(*diff_src_md(), desired_fmt_tag),
                    VERBOSE_UNSUPPORTED_TAG_S, "diff_src");
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");

            if (desc()->alg_kind == pooling_max) {
                const auto ws_dt = hint_fwd_pd_->workspace_md()->data_type;
                init_default_ws(ws_dt);
                VDISPATCH_POOLING(
                        compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);
            }

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();

            return status::success;
        }

        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (diff_src_md()->data_type != data_type::f32) {
                size_t bf16cvt_sz_ = IC() * nthr_;
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<float>(
                        key_pool_src_bf16cvt, bf16cvt_sz_);
                scratchpad.template book<float>(
                        key_pool_dst_bf16cvt, bf16cvt_sz_);
            }
        }
    };

    nhwc_pooling_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<d_type>::type data_t;

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
