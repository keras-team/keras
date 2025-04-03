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

#ifndef CPU_X64_PRELU_JIT_PRELU_BASE_KERNEL_HPP_
#define CPU_X64_PRELU_JIT_PRELU_BASE_KERNEL_HPP_

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/prelu/jit_prelu_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_prelu_base_kernel_t : public jit_generator {
public:
    jit_prelu_base_kernel_t(const cpu_isa_t &isa, const int vlen,
            const prelu::bcast &bcast, const memory_desc_wrapper &tensor_md,
            const size_t number_vmm_single_compute, const char *name);

    size_t simd_w() const noexcept;
    prelu::bcast get_bcast() const noexcept;

protected:
    int reserve_vmm();
    int get_compute_vmm(size_t base_idx, size_t unroll_group) const;

    size_t get_number_reserved_vmms() const noexcept;

    const cpu_isa_t isa_;
    const size_t simd_w_ = 0;
    const prelu::bcast bcast_ = prelu::bcast::unsupported;
    const size_t tail_size_ = 0u;
    const Xbyak::Reg64 &reg_data_size_ = r8;
    const Xbyak::Reg64 &reg_offset_ = r9;

private:
    void generate() override;
    virtual bool any_tensor_bf16() const = 0;
    virtual void load_kernel_call_params() = 0;
    virtual void prepare_kernel_const_vars() = 0;
    virtual void compute_dst(size_t unrolling_factor, bool tail) = 0;
    virtual void finalize() = 0;
    size_t calc_unrolling_factor() const noexcept;
    size_t calc_tail_size(const memory_desc_wrapper &tensor_md) const noexcept;
    const memory_desc_wrapper tensor_md_;
    const size_t number_vmm_single_compute_ = 0;
    size_t number_reserved_vmms_ = 0;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
