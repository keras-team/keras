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

#ifndef CPU_REF_RESAMPLING_HPP
#define CPU_REF_RESAMPLING_HPP

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_resampling_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_resampling_fwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_fwd_pd_t {
        using cpu_resampling_fwd_pd_t::cpu_resampling_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_resampling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            VDISPATCH_RESAMPLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING(
                    platform::has_data_type_support(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING(
                    platform::has_data_type_support(dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING(set_default_params() == status::success,
                    VERBOSE_BAD_PARAM, "");
            VDISPATCH_RESAMPLING(attr()->has_default_values(
                                         sm::post_ops, dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_RESAMPLING(
                    ref_post_ops_t::primitive_kind_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_RESAMPLING(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            return status::success;
        }
    };

    ref_resampling_fwd_t(const pd_t *apd);
    ~ref_resampling_fwd_t();

    status_t init(engine_t *engine) override {
        ref_post_ops_
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops_) return status::out_of_memory;
        CHECK(ref_post_ops_->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void execute_forward(const exec_ctx_t &ctx) const;

    std::unique_ptr<ref_post_ops_t> ref_post_ops_;
};

struct ref_resampling_bwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_bwd_pd_t {
        using cpu_resampling_bwd_pd_t::cpu_resampling_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_resampling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            VDISPATCH_RESAMPLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING(
                    platform::has_data_type_support(diff_src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING(
                    platform::has_data_type_support(diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING(set_default_params() == status::success,
                    VERBOSE_BAD_PARAM, "");
            VDISPATCH_RESAMPLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            return status::success;
        }
    };

    ref_resampling_bwd_t(const pd_t *apd);
    ~ref_resampling_bwd_t();

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void execute_backward(const exec_ctx_t &ctx) const;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
