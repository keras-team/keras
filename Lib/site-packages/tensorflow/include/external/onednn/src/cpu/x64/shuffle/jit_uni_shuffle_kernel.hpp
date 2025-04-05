/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_SHUFFLE_KERNEL_HPP
#define CPU_X64_JIT_UNI_SHUFFLE_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_shuffle_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/shuffle/jit_uni_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_shuffle_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_shuffle_kernel_t)

    jit_uni_shuffle_kernel_t(const jit_shuffle_conf_t &conf);

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    constexpr int vmm_idx(int idx) const {
        return (cpu_isa_traits<isa>::n_vregs - 1) - idx;
    }

    /*
     * Prepare the mask to be used during tail processing.
     * vmm_tail_mask_ is filled if it is avx and
     * if it is avx512_core at least then k_tail_mask_ is filled.
     */
    void prepare_mask();

    /*
     * Emulates the behavior of vgatherdps for architectures
     * that do not support this instruction.
     */
    void emu_gather_data(const Reg64 &reg_src_addr, const int indices_idx,
            const int data_idx, const bool is_tail = false);

    void gather_data(const Reg64 &reg_src_addr, const int indices_idx,
            const int data_idx, const bool is_tail = false);

    void store_data(const int data_idx, const Reg64 &reg_dst_addr,
            const int offset = 0, const bool is_tail = false);

    void shuffle_blocked_format();

    void append_zero_padding(
            const Reg64 &reg_dst_addr, const bool zero_extend_write);

    void generate() override;

    const Vmm vmm_tail_mask_ = Vmm(0);
    // Used only for avx
    // Vgatherdps always gets data using a conditional mask
    // This register contains all bits set to 1, allowing
    // to get the maximum number of values available to the register
    const Vmm vmm_full_mask_ = Vmm(1);
    const Vmm vmm_src_ = Vmm(2);
    const Vmm vmm_tmp_ = Vmm(3);
    const Vmm vmm_indices_ = Vmm(4);
    const Vmm vmm_zero_ = Vmm(11);

    const Opmask k_tail_mask_ = k1;
    const Opmask k_full_mask_ = k2;

    const Reg64 &reg_tmp_ = rax;
    const Reg64 &reg_dst_ = rbx;
    const Reg64 &reg_indices_ = rcx;
    const Reg64 &reg_work_ = rdx;
    // Always mimic the Unix ABI
    const Reg64 &reg_param = rdi;
    const Reg64 &reg_src_ = rsi;
    const Reg64 &reg_tmp1_ = r8;
    const Reg64 &reg_tmp2_ = r9;
    const Reg64 &reg_tmp3_ = r10;
    const Reg64 &reg_tmp4_ = r11;
    const Reg64 &reg_tmp5_ = r12;
    const Reg64 &reg_tmp6_ = r13;
    const Reg8 &reg_padded_block = r14b;

    const jit_shuffle_conf_t conf_;
    const size_t padding_size_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
