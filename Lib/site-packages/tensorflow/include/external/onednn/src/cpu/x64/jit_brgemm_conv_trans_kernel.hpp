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

#ifndef CPU_X64_JIT_BRGEMM_CONV_TRANS_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_CONV_TRANS_KERNEL_HPP

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace jit_avx512_core_brgemm_conv_trans_kernel {
struct jit_brgemm_conv_trans_kernel_call_s {
    const void *src;
    const void *dst;
    size_t owb;
    size_t ic;
    size_t t_pad;
    size_t h_count;
    size_t b_pad;
};

struct jit_avx512_core_brgemm_conv_trans_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_brgemm_conv_trans_kernel_t)

    using reg64_t = const Xbyak::Reg64;

    jit_avx512_core_brgemm_conv_trans_kernel_t(
            const jit_brgemm_conv_conf_t &ajcp, const char *name = jit_name());

    static int dst_w(const jit_brgemm_conv_conf_t &ajcp, int out_w);

protected:
    jit_brgemm_conv_conf_t jcp;
    dim_t inp_dsz;
    dim_t ic_block_sz;
    dim_t ic_block_offset;
    dim_t iw_size, dst_w_block, dst_stride;
    dim_t dst_h_offset, dst_w_offset;
    dim_t VL, n_vec, n_tail_vec;
    const reg64_t inp_ptr = r15;
    const reg64_t dst_ptr = r14;

    const reg64_t aux_inp_ptr = r13;
    const reg64_t aux_dst_ptr = r12;

    const reg64_t reg_hc = r10;

    const reg64_t reg_ic = r9;

    const reg64_t reg_owb = rdx;

    const reg64_t kh_over = r8;
    const reg64_t reg_t_pad = rax;
    const reg64_t reg_b_pad = rbx;

    const reg64_t reg_tmp = rsi;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask kblock_tail_mask = Xbyak::Opmask(3);

    const Xbyak::Zmm zmm_zero = Xbyak::Zmm(0);
    const Xbyak::Zmm zmm_s8s8_shift = Xbyak::Zmm(0);

    void load(const Xbyak::Xmm &x, const Xbyak::Address &addr);

    void store(const Xbyak::Address &addr, const Xbyak::Xmm &x);

    void zero_ic_block(bool is_ic_tail, dim_t dst_off);
    void copy_ic_block(dim_t zidx, bool is_ic_tail, dim_t inp_off,
            dim_t dst_off, bool do_load);
    Xbyak::Zmm get_zmm(dim_t idx) const { return Xbyak::Zmm(1 + (idx % 31)); }
    void generate() override;
    void copy_ow_block(bool is_ic_tail);
    void copy_ow_block_body(int lpad, int ow_len, int iw_len, bool is_ic_tail);

    int inp_w(int out_w) const;
    int inp_w(int out_w, int kw) const;
    int inp_w_start(int owb) const;
};

struct jit_avx512_core_brgemm_conv_rtus_kernel_t
    : jit_avx512_core_brgemm_conv_trans_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_brgemm_conv_rtus_kernel_t)

    jit_avx512_core_brgemm_conv_rtus_kernel_t(
            const jit_brgemm_conv_conf_t &ajcp);

private:
    void generate() override;
};

} // namespace jit_avx512_core_brgemm_conv_trans_kernel

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
