/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#ifndef CPU_X64_LRN_JIT_UNI_LRN_HPP
#define CPU_X64_LRN_JIT_UNI_LRN_HPP

#include <memory>
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_lrn_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/lrn/jit_uni_lrn_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, data_type_t d_type>
struct jit_uni_lrn_fwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_fwd_pd_t {
        using cpu_lrn_fwd_pd_t::cpu_lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_lrn_fwd_t);

        status_t init(engine_t *engine);

        format_tag_t dat_tag_;
    };

    jit_uni_lrn_fwd_t(const pd_t *apd);
    ~jit_uni_lrn_fwd_t();

    using data_t = typename prec_traits<d_type>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_lrn_fwd_kernel_t<isa, d_type>> ker_, ker_first_,
            ker_last_;
    static constexpr int VECTOR_LENGTH
            = jit_uni_lrn_fwd_kernel_t<isa, d_type>::VECTOR_LENGTH;
};

template <cpu_isa_t isa, data_type_t d_type>
struct jit_uni_lrn_bwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_bwd_pd_t {
        using cpu_lrn_bwd_pd_t::cpu_lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_lrn_bwd_t);

        status_t init(engine_t *engine);

        format_tag_t dat_tag_;
    };

    jit_uni_lrn_bwd_t(const pd_t *apd);
    ~jit_uni_lrn_bwd_t();

    using data_t = typename prec_traits<d_type>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_lrn_bwd_kernel_t<isa, d_type>> ker_, ker_first_,
            ker_last_;
    static constexpr int VECTOR_LENGTH
            = jit_uni_lrn_bwd_kernel_t<isa, d_type>::VECTOR_LENGTH;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
