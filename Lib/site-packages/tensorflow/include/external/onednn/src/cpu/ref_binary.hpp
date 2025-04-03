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

#ifndef CPU_REF_BINARY_HPP
#define CPU_REF_BINARY_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            VDISPATCH_BINARY(
                    platform::has_data_type_support(src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BINARY(
                    platform::has_data_type_support(src_md(1)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BINARY(
                    platform::has_data_type_support(dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BINARY(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY(attr()->has_default_values(
                                     sm::post_ops | sm::scales_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BINARY(IMPLICATION(!attr()->scales_.has_default_values(),
                                     check_scales_mask()),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_BINARY(
                    ref_post_ops_t::primitive_kind_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_BINARY(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            return status::success;
        }

    private:
        bool check_scales_mask() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};
            return attr_scales_ok(supported_args);
        }
    };

    ref_binary_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
