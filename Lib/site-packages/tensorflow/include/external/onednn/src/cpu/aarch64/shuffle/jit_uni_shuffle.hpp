/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_SHUFFLE_HPP
#define CPU_AARCH64_JIT_UNI_SHUFFLE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_shuffle_pd.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/shuffle/jit_uni_shuffle_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa>
struct jit_uni_shuffle_kernel_t;

template <cpu_isa_t isa>
struct jit_uni_shuffle_t : public primitive_t {
    struct pd_t : public cpu_shuffle_pd_t {
        using cpu_shuffle_pd_t::cpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", conf_.isa, ""), jit_uni_shuffle_t);

        status_t init(engine_t *engine);

        jit_shuffle_conf_t get_conf() const { return conf_; };

    private:
        jit_shuffle_conf_t conf_;
    };

    jit_uni_shuffle_t(const pd_t *apd);

    ~jit_uni_shuffle_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t precompute_offsets();
    std::unique_ptr<jit_uni_shuffle_kernel_t<isa>> kernel_;
    unsigned *input_off_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
