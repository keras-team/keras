/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_COMP_PAD_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_CONV_COMP_PAD_KERNEL_HPP

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace jit_uni_brgemm_conv_comp_pad_kernel {
struct jit_brgemm_conv_comp_pad_call_s {
    const void *ptr_in;
    void *ptr_zp_out;
    void *ptr_cp_out;
    size_t use_inversion;
    size_t kw_l;
    size_t kh_l;
    size_t kd_l;
    size_t ker_l {1};
    size_t last_ocb {1};
};

// Kernel is unified to work with fwd and bwd_d conv
// Variables with "ic" and "oc" are named from perspective of fwd
// For bwd_d "ic" and "oc" are swapped
template <typename Vmm>
struct jit_uni_brgemm_conv_comp_pad_kernel_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_brgemm_conv_comp_pad_kernel_t)

    using reg64_t = const Xbyak::Reg64;

    jit_uni_brgemm_conv_comp_pad_kernel_t(const jit_brgemm_conv_conf_t &ajcp);

    ~jit_uni_brgemm_conv_comp_pad_kernel_t() = default;

protected:
    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;

    jit_brgemm_conv_conf_t jcp_;
    const int inp_dsz_;
    const int out_dsz_;
    const size_t nb_ic_;
    const size_t inp_ic_sz_;
    const size_t inp_kw_sz_;
    const size_t inp_kh_sz_;
    const size_t inp_kd_sz_;
    const size_t out_ow_sz_;
    const size_t out_ker_sz_;
    const int isa_max_regs;

    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_in = r15;
    const reg64_t reg_comp_out = r14;
    const reg64_t reg_zp_comp_out = r13;
    const reg64_t reg_use_inversion = rax;

    const reg64_t reg_kd_l = r12;
    const reg64_t reg_kh_l = r11;
    const reg64_t reg_kw_l = r10;
    const reg64_t reg_icb = r9;
    const reg64_t reg_ker_l = rdx;

    const reg64_t reg_aux_comp_out = r9;
    const reg64_t reg_aux_zp_comp_out = r10;
    const reg64_t reg_aux_in = r8;
    const reg64_t reg_aux_kh_in = rbx;
    const reg64_t reg_aux_kd_in = rsi;
    const reg64_t reg_tmp = rax;

    Vmm vmm_tmp = Vmm(isa_max_regs - 1);
    Vmm vmm_one_bytes = Vmm(isa_max_regs - 2);
    Vmm vmm_zp_shift = Vmm(isa_max_regs - 3);
    Vmm vmm_cp_shift = Vmm(isa_max_regs - 4);

    Xbyak::Zmm zmm_one_words = Xbyak::Zmm(27);
    Xbyak::Zmm zmm_int8_temp = Xbyak::Zmm(26);

    const int last_ic_block_ = 4;
    const int m_block2_ = vreg_traits<Vmm>::vlen / sizeof(int32_t);
    static constexpr int max_oc_block_ = 64;
    const int n_max_regs_ = max_oc_block_ / m_block2_;

    const Vmm &vmm_tmp_1() const noexcept { return vmm_tmp; }

    Vmm accum(const int n_block, const int m, const int n) const;
    size_t out_oc_offset(const int n, const int w) const;
    size_t inp_ic_offset(
            const int m_block, const int icb, const int m, const int n) const;
    int compute_ic_step(
            const int m_max_regs, const int m_block, const int n_block) const;

    void store_accumulators(const int m_block, const int n_block,
            const int ow_b, const int ow_e);
    void zero_accumulators(const int m_block, const int n_block);
    void compute(const int ic_step, const int m_block, const int n_block,
            const int m_tail, const bool is_mb_tail);
    void icb_loop(const int icb, const int icb_tail, const int ic_step,
            const int m_block, const int mb_tail, const int n_block);
    void kdh_loop(const int icb, const int icb_tail, const int ic_step,
            const int m_block, const int mb_tail, const int n_block);
    void copy_ow(const int m_block, const int n_block, const int ow_b,
            const int ow_e);
    void copy_ow_body(const int n_block, const int ow_b, const int ow_e);
    void kw_loop(const int icb, const int icb_tail, const int ic_step,
            const int m_block, const int mb_tail, const int n_block,
            const bool use_inversion);
    void kw_loop_base(const int icb, const int icb_tail, const int ic_step,
            const int m_block, const int mb_tail, const int n_block);
    void fwd_kw_ow_loop(const int icb, const int icb_tail, const int ic_step,
            const int m_block, const int mb_tail, const int n_block,
            const bool use_inversion);
    void bwd_kw_iw_loop(const int icb, const int icb_tail, const int ic_step,
            const int m_block, const int mb_tail, const int n_block);

    void load_params();
    void generate() override;
};

template <typename Vmm>
struct jit_uni_brgemm_conv_relo_comp_pad_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_brgemm_conv_relo_comp_pad_kernel_t)
    using reg64_t = const Xbyak::Reg64;

    jit_uni_brgemm_conv_relo_comp_pad_kernel_t(
            const jit_brgemm_conv_conf_t &ajcp);
    ~jit_uni_brgemm_conv_relo_comp_pad_kernel_t() = default;

protected:
    jit_brgemm_conv_conf_t jcp_;
    const int inp_dsz_;
    const int out_dsz_;
    const int inp_oc_block_;
    const size_t inp_ic_sz_;
    const size_t inp_kw_sz_;
    const size_t inp_kh_sz_;
    const size_t inp_oc_sz_;
    const size_t out_ow_sz_;
    const size_t out_ker_sz_;
    const int isa_max_regs_;

    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_in = r15;
    const reg64_t reg_comp_out = r14;
    const reg64_t reg_zp_comp_out = r13;
    const reg64_t reg_kh_l = r12;

    const reg64_t reg_aux_zp_comp_out = r11;
    const reg64_t reg_aux_comp_out = r10;
    const reg64_t reg_aux_in = r9;
    const reg64_t reg_ker_l = rsi;
    const reg64_t reg_last_ocb = rbx;
    const reg64_t reg_tmp = rax;

    Vmm vmm_tmp = Vmm(isa_max_regs_ - 1);
    Vmm vmm_cp_shift = Vmm(isa_max_regs_ - 2);

    const int n_max_regs_ = 4;

    Vmm accum(const int n, const bool has_s8s8_shift = false) const;
    size_t out_oc_offset(const int n, const int w) const;
    size_t inp_ic_offset(const int kw, const int ic, const int m) const;
    void zero_accumulators(const int n_block);
    void store_accumulators(const int n_block, const int ow_b, const int ow_e);
    void store(const int n_block, const int ow_b, const int ow_e);
    void kw_loop(const int n_block);
    void compute(const int n_block, const int kw_b, const int kw_e);
    void load_params();

    void generate() override;
};
} // namespace jit_uni_brgemm_conv_comp_pad_kernel

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
