/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef CPU_REF_REDUCTION_HPP
#define CPU_REF_REDUCTION_HPP

#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_reduction_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src_type, data_type_t dst_type, data_type_t acc_type>
struct ref_reduction_t : public primitive_t {
    struct pd_t : public cpu_reduction_pd_t {
        using cpu_reduction_pd_t::cpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_reduction_t);

        status_t init(engine_t *engine) {
            using sm = primitive_attr_t::skip_mask_t;

            VDISPATCH_REDUCTION(
                    src_type == src_md()->data_type, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REDUCTION(
                    dst_type == dst_md()->data_type, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REDUCTION(acc_type
                            == types::default_accum_data_type(
                                    src_type, dst_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REDUCTION(platform::has_data_type_support(src_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REDUCTION(platform::has_data_type_support(dst_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REDUCTION(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_REDUCTION(attr()->has_default_values(sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REDUCTION(
                    ref_post_ops_t::primitive_kind_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_REDUCTION(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            return status::success;
        }
    };

    ref_reduction_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    using src_t = typename prec_traits<src_type>::type;
    using acc_t = typename prec_traits<acc_type>::type;
    using dst_t = typename prec_traits<dst_type>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;

    void accumulate(
            acc_t &acc, const src_t &src, alg_kind_t alg_kind, float p) const;
    void finalize(
            float &acc_f32, alg_kind_t alg, float p, float eps, dim_t n) const;
    void init_acc(acc_t &acc, alg_kind_t alg) const;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
