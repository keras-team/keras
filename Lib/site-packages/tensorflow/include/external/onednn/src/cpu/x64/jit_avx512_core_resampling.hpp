/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_RESAMPLING_HPP
#define CPU_X64_JIT_AVX512_CORE_RESAMPLING_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_resampling_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_resampling_args_t;

struct jit_avx512_core_resampling_kernel_base_t : public jit_generator {
    jit_avx512_core_resampling_kernel_base_t(
            const resampling_pd_t *pd, const char *name);
    virtual ~jit_avx512_core_resampling_kernel_base_t() = default;

protected:
    const resampling_pd_t *pd_;

    data_type_t src_data_type() const;
    data_type_t dst_data_type() const;
};

struct jit_avx512_core_resampling_bwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_bwd_pd_t {
        using cpu_resampling_bwd_pd_t::cpu_resampling_bwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", avx512_core, ""),
                jit_avx512_core_resampling_bwd_t);

        status_t init(engine_t *engine);
    };

    jit_avx512_core_resampling_bwd_t(const pd_t *apd) : primitive_t(apd) {}
    ~jit_avx512_core_resampling_bwd_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_avx512_core_resampling_kernel_base_t> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
