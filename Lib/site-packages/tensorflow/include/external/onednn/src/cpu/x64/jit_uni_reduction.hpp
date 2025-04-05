/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef CPU_X64_UNI_REDUCTION_HPP
#define CPU_X64_UNI_REDUCTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_reduction_pd.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/jit_uni_reduction_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_uni_reduction_t : public primitive_t {
    struct pd_t : public cpu_reduction_pd_t {
        using cpu_reduction_pd_t::cpu_reduction_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", conf_.isa, ""),
                jit_uni_reduction_t);

        status_t init(engine_t *engine);

        const jit_reduction_conf_t &get_conf() const { return conf_; };

    private:
        bool fill_post_ops_conf();

        jit_reduction_conf_t conf_;
    };

    jit_uni_reduction_t(const pd_t *apd) : primitive_t(apd) {}
    virtual ~jit_uni_reduction_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t get_proper_kernel(
            const memory_desc_t *dst_md, const jit_reduction_conf_t &conf);

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_reduction_kernel_base_t> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
