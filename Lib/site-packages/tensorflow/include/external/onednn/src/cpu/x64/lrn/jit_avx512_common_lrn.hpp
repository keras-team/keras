/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_HPP

#include <memory>
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_lrn_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/lrn/lrn_executor.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
template <data_type_t d_type>
struct jit_avx512_common_lrn_fwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_fwd_pd_t {
        using cpu_lrn_fwd_pd_t::cpu_lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("lrn_jit:",
                        utils::map(true, avx512_core,
                                d_type == data_type::bf16
                                        && mayiuse(avx512_core_bf16),
                                avx512_core_bf16,
                                d_type == data_type::bf16
                                        && !mayiuse(avx512_core_bf16),
                                bf16_emulation_t::get_isa(),
                                d_type == data_type::f16, avx512_core_fp16),
                        ""),
                jit_avx512_common_lrn_fwd_t);

        status_t init(engine_t *engine);
    };

    jit_avx512_common_lrn_fwd_t(const pd_t *apd);
    ~jit_avx512_common_lrn_fwd_t();

    using data_t = typename prec_traits<d_type>::type;

    status_t init(engine_t *engine) override {
        return lrn_executor_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return lrn_executor_->execute(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<lrn::i_lrn_executor_t> lrn_executor_;
};

template <data_type_t d_type>
struct jit_avx512_common_lrn_bwd_t : public primitive_t {
    struct pd_t : public cpu_lrn_bwd_pd_t {
        using cpu_lrn_bwd_pd_t::cpu_lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("lrn_jit:",
                        utils::map(true, avx512_core,
                                d_type == data_type::bf16
                                        && mayiuse(avx512_core_bf16),
                                avx512_core_bf16,
                                d_type == data_type::bf16
                                        && !mayiuse(avx512_core_bf16),
                                bf16_emulation_t::get_isa(),
                                d_type == data_type::f16, avx512_core_fp16),
                        ""),
                jit_avx512_common_lrn_bwd_t);

        status_t init(engine_t *engine);
    };

    jit_avx512_common_lrn_bwd_t(const pd_t *apd);
    ~jit_avx512_common_lrn_bwd_t();

    using data_t = typename prec_traits<d_type>::type;

    status_t init(engine_t *engine) override {
        return lrn_executor_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return lrn_executor_->execute(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<lrn::i_lrn_executor_t> lrn_executor_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
