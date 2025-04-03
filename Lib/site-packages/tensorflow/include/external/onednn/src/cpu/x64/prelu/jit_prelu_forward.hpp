/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef CPU_X64_PRELU_JIT_PRELU_FORWARD_HPP
#define CPU_X64_PRELU_JIT_PRELU_FORWARD_HPP

#include <memory>
#include <set>

#include "common/primitive.hpp"
#include "cpu/cpu_prelu_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_prelu_forward_kernel_t;

class jit_prelu_fwd_t : public primitive_t {
public:
    struct pd_t : public cpu_prelu_fwd_pd_t {
    public:
        using cpu_prelu_fwd_pd_t::cpu_prelu_fwd_pd_t;
        DECLARE_COMMON_PD_T("jit_uni", jit_prelu_fwd_t);
        status_t init(engine_t *engine);

    private:
        bool bcast_supported(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &weights_d,
                const memory_desc_wrapper &dst_d) const;
    };

    jit_prelu_fwd_t(const pd_t *apd);
    ~jit_prelu_fwd_t();
    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const;
    std::unique_ptr<jit_prelu_forward_kernel_t> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
