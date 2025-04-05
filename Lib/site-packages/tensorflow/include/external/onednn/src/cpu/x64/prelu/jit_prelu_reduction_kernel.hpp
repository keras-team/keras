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

#ifndef CPU_X64_PRELU_JIT_PRELU_REDUCTION_HPP
#define CPU_X64_PRELU_JIT_PRELU_REDUCTION_HPP

#include <memory>

#include "cpu/cpu_prelu_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_prelu_reduction_kernel_t : public jit_generator {
public:
    static jit_prelu_reduction_kernel_t *create(const cpu_prelu_bwd_pd_t *pd);

    struct call_params_t {
        size_t reduction_blocks = 0;
        const void *weights_diff_scratch = nullptr;
        void *weights_diff = nullptr;
        bool tail = false;
        bool is_last_c_blk = false;
    };

    void generate() override;
    size_t simd_w() const;
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_prelu_reduction_kernel_t)

    void operator()(jit_prelu_reduction_kernel_t::call_params_t *params) {
        jit_generator::operator()(params);
    }

private:
    void load_kernel_call_params();
    virtual size_t get_unrolling_factor(bool tail) const = 0;
    virtual void compute_dst(int unrolling_factor, bool tail) = 0;
    virtual void prepare_kernel_const_vars(bool tail) = 0;
    virtual void finalize(bool tail) = 0;
    void generate(bool tail);

    const Xbyak::Reg64 &reg_reduction_blocks_ = r8;
    const Xbyak::Reg64 &reg_weights_diff_scratch_ = r10;
    const Xbyak::Reg8 &reg_tail_ = r12b;

    const size_t scratchpad_c_block_offset_ = 0;

protected:
    jit_prelu_reduction_kernel_t(const cpu_prelu_bwd_pd_t *pd, int simd_w);
    Xbyak::Address diff_scratch_ptr(int unrolling_group) const;
    int reserve_vmm();

    const size_t simd_w_ = 0;
    const data_type_t data_type_;
    const size_t tail_size_ = 0;
    const Xbyak::Reg64 &reg_offset_ = r9;
    const Xbyak::Reg64 &reg_weights_diff_ = r11;
    const Xbyak::Reg8 &reg_last_c_blk_byte_ = r13b;
    size_t number_reserved_vmms_ = 0;
    size_t tail_block_size_ = 0;
    size_t c_blk_nelems_ = 0;
};

template <typename Vmm>
class jit_uni_prelu_reduction_kernel_t : public jit_prelu_reduction_kernel_t {
public:
    jit_uni_prelu_reduction_kernel_t(
            const cpu_prelu_bwd_pd_t *pd, const cpu_isa_t &isa);

private:
    size_t get_unrolling_factor(bool tail) const override;
    void prepare_kernel_const_vars(bool tail) override;
    void finalize(bool tail) override;
    void compute_dst(int unrolling_factor, bool tail) override;

    const cpu_isa_t isa_;
    const bool saturation_needed_;
    const Vmm tail_vmm_mask_; // Keep it higher to preserve idx=0 tail register
    const Vmm accumulator_;
    const Vmm saturation_lower_bound_;
    const Vmm saturation_upper_bound_;

    const Xbyak::Opmask &tail_opmask_ = k1;
    const Xbyak::Reg64 &reg_tmp_ = r15;

    io::jit_io_helper_t<Vmm> io_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
