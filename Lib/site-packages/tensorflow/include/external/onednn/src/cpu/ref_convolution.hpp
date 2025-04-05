/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#ifndef CPU_REF_CONVOLUTION_HPP
#define CPU_REF_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            bool ok = is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && platform::has_data_type_support(src_type)
                    && platform::has_data_type_support(bia_type)
                    && platform::has_data_type_support(dst_type)
                    && utils::one_of(src_type, f32, bf16, f16, f8_e5m2, f8_e4m3)
                    && src_type == wei_type
                    && utils::one_of(dst_type, src_type, f32)
                    && utils::one_of(bia_type, data_type::undef, src_type, f32)
                    && set_default_formats()
                    && attr()->has_default_values(
                            smask_t::post_ops | smask_t::sum_dt, dst_type)
                    && attr()->post_ops_.check_sum_consistency(
                            dst_type, /* is_int8 */ false)
                    && post_ops_ok()
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            return ok ? status::success : status::unimplemented;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool post_ops_ok() const {
            return ref_post_ops_t::primitive_kind_ok(attr()->post_ops_);
        }
    };

    ref_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

struct ref_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto diff_src_type = diff_src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto diff_dst_type = diff_dst_md(0)->data_type;

            bool ok = desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && platform::has_data_type_support(diff_src_type)
                    && platform::has_data_type_support(diff_dst_type)
                    && utils::one_of(diff_dst_type, f32, bf16, f16)
                    && wei_type == diff_dst_type
                    && utils::one_of(diff_src_type, f32, diff_dst_type)
                    && set_default_formats() && attr()->has_default_values();

            return ok ? status::success : status::unimplemented;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    ref_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct ref_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        using cpu_convolution_bwd_weights_pd_t::
                cpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto diff_wei_type = diff_weights_md(0)->data_type;
            const auto diff_bia_type = diff_weights_md(1)->data_type;
            const auto diff_dst_type = diff_dst_md(0)->data_type;

            bool ok = desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && platform::has_data_type_support(src_type)
                    && platform::has_data_type_support(diff_wei_type)
                    && utils::one_of(src_type, f32, bf16, f16)
                    && diff_dst_type == src_type
                    && utils::one_of(diff_wei_type, f32, src_type)
                    && utils::one_of(
                            diff_bia_type, data_type::undef, f32, src_type)
                    && set_default_formats() && attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    ref_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
