/******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2023 KNS Group LLC (YADRO)
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

#ifndef RV64_NCHW_POOLING_HPP
#define RV64_NCHW_POOLING_HPP

#include "cpu/cpu_pooling_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <data_type_t d_type>
struct riscv_nchw_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T_("RISCV64GCV", riscv_nchw_pooling_fwd_t)

        status_t init(engine_t *engine) {
            UNUSED(engine);

            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            const bool is_training
                    = desc_.prop_kind == prop_kind::forward_training;

            const bool ok = is_fwd()
                    && utils::one_of(desc()->alg_kind, alg_kind::pooling_max)
                    && memory_desc_wrapper(dst_md()).is_dense(false)
                    && utils::everyone_is(
                            d_type, src_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(d_type)
                    && !has_zero_dim_memory() && !is_dilated()
                    && attr()->has_default_values()
                    && set_default_params() == status::success
                    && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
                    && memory_desc_matches_tag(*dst_md(), desired_fmt_tag)
                    && attr_.set_default_formats(dst_md(0)) == status::success
                    && !is_training
                    && KW() < riscv_nchw_pooling_fwd_t<
                               d_type>::max_kernel_width;

            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    riscv_nchw_pooling_fwd_t(const pd_t *apd);

    using data_t = typename prec_traits<d_type>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    constexpr static int max_kernel_width = 32;

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
