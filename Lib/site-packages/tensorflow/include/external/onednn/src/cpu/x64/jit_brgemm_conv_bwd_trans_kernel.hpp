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

#ifndef CPU_X64_JIT_BRGEMM_CONV_BWD_TRANS_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_CONV_BWD_TRANS_KERNEL_HPP

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace jit_avx512_core_brgemm_conv_bwd_trans_kernel {
struct jit_brgemm_conv_bwd_trans_kernel_call_s {
    const void *src;
    const void *dst;
    size_t iwb;
    size_t oc;
    size_t t_pad;
    size_t h_count;
    size_t b_pad;
};

template <typename Vmm>
struct jit_avx512_core_brgemm_conv_bwd_trans_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_brgemm_conv_bwd_trans_kernel_t)

    using reg64_t = const Xbyak::Reg64;

    jit_avx512_core_brgemm_conv_bwd_trans_kernel_t(
            const jit_brgemm_conv_conf_t &ajcp, const char *name = jit_name());

protected:
    static constexpr bool is_zmm_ = std::is_same<Vmm, Xbyak::Zmm>::value;
    jit_brgemm_conv_conf_t jcp;
    dim_t inp_dsz;
    dim_t oc_block_sz;
    dim_t ow_size, dst_w_block, dst_stride;
    dim_t dst_w_offset, dst_h_offset;
    dim_t VL, n_vec, n_tail_vec;
    const reg64_t inp_ptr = r15;
    const reg64_t dst_ptr = r14;

    const reg64_t aux_inp_ptr = r13;
    const reg64_t aux_dst_ptr = r12;

    const reg64_t reg_hc = r10;

    const reg64_t reg_oc = r9;

    const reg64_t reg_iwb = rdx;

    const reg64_t kh_over = r8;
    const reg64_t reg_t_pad = rax;
    const reg64_t reg_b_pad = rbx;

    const reg64_t reg_tmp = rsi;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask kblock_tail_mask = Xbyak::Opmask(3);

    const Vmm vmm_tmp = Vmm(0);
    const Vmm vmm_zero = Vmm(1);

    void load(
            const Vmm &x, const Xbyak::Address &addr, const int load_size = 0);

    void store(
            const Xbyak::Address &addr, const Vmm &x, const int store_size = 0);

    void zero_oc_block(bool is_oc_tail, dim_t dst_off);
    void copy_oc_block(bool is_oc_tail, dim_t inp_off = 0, dim_t dst_off = 0,
            bool do_load = true);
    void generate() override;
    void copy_iw_block(bool is_oc_tail);
    void copy_iw_block_body(int lpad, int iw_len, int ow_len, bool is_oc_tail);

    int inp_w(int out_w) const;
    int inp_w_start(int iwb) const;
};

} // namespace jit_avx512_core_brgemm_conv_bwd_trans_kernel
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
