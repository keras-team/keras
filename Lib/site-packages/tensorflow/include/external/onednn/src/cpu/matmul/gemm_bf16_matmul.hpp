/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef CPU_MATMUL_GEMM_BF16_MATMUL_HPP
#define CPU_MATMUL_GEMM_BF16_MATMUL_HPP

#include <assert.h>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/gemm_inner_product_utils.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/matmul/gemm_based_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

template <impl::data_type_t dst_type>
struct gemm_bf16_matmul_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("gemm:jit:bf16", gemm_bf16_matmul_t);

        status_t init(engine_t *engine);
        const gemm_based::params_t &params() const { return params_; }

        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        status_t check_and_configure_attributes(engine_t *engine);
        gemm_based::params_t params_;
    };

    gemm_bf16_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        if (pd()->params().has_pp_kernel_) {
            const bool has_runtime_dims
                    = memory_desc_wrapper(pd()->dst_md()).has_runtime_dims();
            const int nthr = pd()->nthr_;
            const dim_t batch = pd()->batch();
            const dim_t M = pd()->M();

            // mb value is calculated based on work-sharing using
            // balance211 in execute()
            dim_t mb = DNNL_RUNTIME_DIM_VAL;
            if (!has_runtime_dims && ((batch * M) % nthr == 0)) {
                const dim_t m_per_thr = nstl::max<dim_t>(1, (batch * M) / nthr);
                if (m_per_thr >= M && m_per_thr % M == 0) {
                    mb = M;
                } else if (m_per_thr < M && M % m_per_thr == 0) {
                    mb = m_per_thr;
                }
            }

            const bool skip_sum
                    = should_skip_sum_po(); // sum can be done by gemm itself
            CHECK(safe_ptr_assign(pp_kernel_,
                    inner_product_utils::pp_kernel_t::create(pd()->N(), mb,
                            pd()->ldc(), &pd()->params().pp_attr_,
                            pd()->desc()->bias_desc.data_type,
                            pd()->desc()->accum_data_type, pd()->dst_md(),
                            skip_sum)));
            return pp_kernel_->create_kernel();
        }
        return status::success;
    }

    static constexpr data_type_t src_type = data_type::bf16;
    static constexpr data_type_t weights_type = data_type::bf16;
    static constexpr data_type_t acc_type = data_type::f32;

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<weights_type>::type weights_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    bool should_skip_sum_po() const noexcept;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;

    std::unique_ptr<inner_product_utils::pp_kernel_t> pp_kernel_;
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
