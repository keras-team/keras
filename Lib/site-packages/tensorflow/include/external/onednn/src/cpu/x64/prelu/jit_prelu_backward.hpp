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

#ifndef CPU_X64_PRELU_JIT_PRELU_BACKWARD_HPP
#define CPU_X64_PRELU_JIT_PRELU_BACKWARD_HPP

#include <memory>
#include <set>

#include "common/primitive.hpp"
#include "cpu/cpu_prelu_pd.hpp"

#include "cpu/x64/prelu/jit_prelu_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_prelu_backward_kernel_t;
class jit_prelu_reduction_kernel_t;

class jit_prelu_bwd_t : public primitive_t {
public:
    struct pd_t : public cpu_prelu_bwd_pd_t {
    public:
        using cpu_prelu_bwd_pd_t::cpu_prelu_bwd_pd_t;
        DECLARE_COMMON_PD_T("jit_uni", jit_prelu_bwd_t);
        status_t init(engine_t *engine);
        int nthr_; // To not exceed the limit in execute used for set up.

    private:
        bool dt_supported(
                const std::set<data_type_t> &tensor_data_types) const noexcept;
        bool bcast_supported(const prelu::bcast &bcast,
                const memory_desc_wrapper &src_diff_d,
                const memory_desc_wrapper &weights_diff_d, int simd_w) const;
    };

    jit_prelu_bwd_t(const pd_t *apd);
    ~jit_prelu_bwd_t();
    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    using byte = unsigned char;
    void fill_scratchpad_zeros(float *const scratchpad,
            size_t thread_scratchpad_size, int nthr) const;
    void scratchpad_to_diff_weights_reduction(float *scratchpad,
            byte *weights_diff, size_t weights_diff_dt, dim_t C,
            size_t reduction_blocks) const;
    const pd_t *pd() const;
    std::unique_ptr<jit_prelu_backward_kernel_t> kernel_;
    std::unique_ptr<jit_prelu_reduction_kernel_t> reduction_kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
