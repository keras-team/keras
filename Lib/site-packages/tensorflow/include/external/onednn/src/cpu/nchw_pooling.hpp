/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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

#ifndef CPU_NCHW_POOLING_HPP
#define CPU_NCHW_POOLING_HPP

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

template <data_type_t d_type>
struct nchw_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nchw:any", nchw_pooling_fwd_t);

        status_t init(engine_t *engine) {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            VDISPATCH_POOLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(
                    utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                            alg_kind::pooling_avg_include_padding,
                            alg_kind::pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(utils::everyone_is(d_type, src_md()->data_type,
                                      dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
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

            const bool is_training
                    = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (src_md()->data_type != data_type::f32) {
                const size_t src_sz_ = ID() * IH() * IW() * IC() * MB();
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.template book<float>(key_pool_src_bf16cvt, src_sz_);
            }
        }
    };

    nchw_pooling_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    using data_t = typename prec_traits<d_type>::type;

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
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<ref_post_ops_t> ref_post_ops_;
};

template <data_type_t d_type>
struct nchw_pooling_bwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple_nchw:any", nchw_pooling_bwd_t);

        status_t init(engine_t *engine) {
            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            using namespace prop_kind;
            using namespace alg_kind;
            VDISPATCH_POOLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(
                    utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                            alg_kind::pooling_avg_include_padding,
                            alg_kind::pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(
                    utils::everyone_is(d_type, diff_dst_md()->data_type,
                            diff_src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(platform::has_data_type_support(d_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
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
            calculate_channel_block_size();
            init_scratchpad();

            return status::success;
        }

        dim_t channel_block_size_;
        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            if (diff_dst_md()->data_type != data_type::f32) {
                size_t dst_sz_ = OD() * OH() * OW();
                size_t src_sz_ = ID() * IH() * IW();
                auto scratchpad = scratchpad_registry().registrar();

                scratchpad.template book<float>(key_pool_src_bf16cvt,
                        src_sz_ * nthr_ * channel_block_size_);
                scratchpad.template book<float>(key_pool_dst_bf16cvt,
                        dst_sz_ * nthr_ * channel_block_size_);
            }
        }

        void calculate_channel_block_size() {
            // calculate channels block size at which the data fits into half
            // of L1, it allows to improve performance for problems with small
            // spatial
            dim_t dst_sz_ = OD() * OH() * OW();
            dim_t src_sz_ = ID() * IH() * IW();
            dim_t C_per_thr = nstl::min(MB() * IC() / nthr_, IC());
            const dim_t max_block_size
                    = platform::get_per_core_cache_size(1) / 2;
            dim_t data_size_per_ch = (dst_sz_ + src_sz_) * 6; // f32 + bf16
            channel_block_size_ = nstl::max(
                    nstl::min(C_per_thr, max_block_size / data_size_per_ch),
                    (dim_t)1);
        }
    };

    nchw_pooling_bwd_t(const pd_t *apd) : primitive_t(apd) {}
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
