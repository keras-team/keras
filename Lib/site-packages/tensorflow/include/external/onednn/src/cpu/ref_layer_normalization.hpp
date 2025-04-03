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

#ifndef CPU_REF_LAYER_NORMALIZATION_HPP
#define CPU_REF_LAYER_NORMALIZATION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_layer_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_layer_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        using cpu_layer_normalization_fwd_pd_t::
                cpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_layer_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            VDISPATCH_LNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(
                    utils::one_of(src_md()->data_type, f32, bf16, f16, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    utils::one_of(dst_md()->data_type, f32, bf16, f16, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    platform::has_data_type_support(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    platform::has_data_type_support(dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");
            VDISPATCH_LNORM(
                    attr()->has_default_values(skip_mask_t::scales_runtime
                            | skip_mask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_LNORM(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

            bool ok = attr_.set_default_formats(dst_md(0)) == status::success;
            VDISPATCH_LNORM(ok, VERBOSE_UNSUPPORTED_POSTOP);

            return status::success;
        }

        bool post_ops_ok() const {
            return ref_post_ops_t::primitive_kind_ok(attr()->post_ops_);
        }
    };

    ref_layer_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));

        return status::success;
    };

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

struct ref_layer_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_bwd_pd_t {
        using cpu_layer_normalization_bwd_pd_t::
                cpu_layer_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_layer_normalization_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            VDISPATCH_LNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(utils::one_of(src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    utils::one_of(diff_dst_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    utils::one_of(diff_src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    platform::has_data_type_support(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    platform::has_data_type_support(diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    platform::has_data_type_support(diff_src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");
            VDISPATCH_LNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    ref_layer_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}

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

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
