/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_BF16CVT_HPP
#define CPU_X64_JIT_AVX512_CORE_BF16CVT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

#include "common/bfloat16.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace bf16_support {
struct jit_call_t {
    void *inp;
    void *out;
    void *add;
    size_t nelems;
    int mask;
};
} // namespace bf16_support

#define GET_OFF(field) offsetof(bf16_support::jit_call_t, field)

struct bf16_emulation_t {
    using opmask_t = const Xbyak::Opmask;
    using Zmm_t = const Xbyak::Zmm;
    using Ymm_t = const Xbyak::Ymm;
    using Xmm_t = const Xbyak::Xmm;
    using reg64_t = const Xbyak::Reg64;

    bf16_emulation_t(jit_generator *host, Zmm_t one, Zmm_t even, Zmm_t selector,
            reg64_t scratch, Zmm_t tr0, Zmm_t tr1)
        : host_(host)
        , one_(one)
        , even_(even)
        , selector_(selector)
        , scratch_(scratch)
        , tr0_(tr0)
        , tr1_(tr1) {}

    bf16_emulation_t(jit_generator *host, Zmm_t one, Zmm_t even, Zmm_t selector,
            reg64_t scratch, Zmm_t tr0)
        : bf16_emulation_t(host, one, even, selector, scratch, tr0, tr0) {}

    void vdpbf16ps(Zmm_t &acc, Zmm_t wei, Zmm_t inp) {
        host_->vpsrad(tr0_, wei, 16);
        host_->vpslld(tr0_, tr0_, 16);

        host_->vpsrad(tr1_, inp, 16);
        host_->vpslld(tr1_, tr1_, 16);

        host_->vfmadd231ps(acc, tr1_, tr0_);

        host_->vpslld(tr0_, wei, 16);
        host_->vpslld(tr1_, inp, 16);

        host_->vfmadd231ps(acc, tr1_, tr0_);
    }

    void vcvtneps2bf16(Xmm_t &out, Xmm_t &in) {
        const bool input_is_zmm = in.isZMM() && out.isYMM();
        const bool input_is_ymm = in.isYMM() && out.isXMM();
        assert((input_is_zmm || input_is_ymm)
                && "Incorrect usage of vcvtneps2bf16 instruction.");

        if (input_is_zmm)
            vcvtneps2bf16(out, in, tr0_, one_, even_, selector_);
        else if (input_is_ymm) {
            const Ymm_t tr0_y {tr0_.getIdx()};
            const Ymm_t even_y {even_.getIdx()};
            const Ymm_t selector_y {selector_.getIdx()};
            const Ymm_t one_y {one_.getIdx()};

            vcvtneps2bf16(out, in, tr0_y, one_y, even_y, selector_y);
        }
    }

private:
    void vcvtneps2bf16(const Xbyak::Operand &out, const Xmm_t &in,
            const Xmm_t &tr0, const Xbyak::Operand &one, const Xmm_t &even,
            const Xbyak::Operand &selector) {
        host_->vpsrld(tr0, in, 16);
        host_->vpandd(tr0, tr0, one);

        host_->vpaddd(tr0, even, tr0);

        host_->vpaddd(tr0, in, tr0);
        host_->vfixupimmps(tr0, in, selector, 0);

        host_->vpsrad(tr0, tr0, 16);
        host_->vpmovdw(out, tr0);
    }

public:
    void init_vcvtneps2bf16() {
        const int selector_int32 =
                /* qnan input to qnan output (presenrving input bits 0..21) */
                encode_fixup_selector(
                        fixup_input_code_snan, fixup_output_code_qnan_input)
                |
                /* snan input to qnan output (presenrving input bits 0..21) */
                encode_fixup_selector(
                        fixup_input_code_qnan, fixup_output_code_qnan_input)
                |
                /* neg inf input copied to output */
                encode_fixup_selector(
                        fixup_input_code_ninf, fixup_output_code_copy_input)
                |
                /* pos inf input copied to output */
                encode_fixup_selector(
                        fixup_input_code_pinf, fixup_output_code_copy_input);

        host_->xor_(scratch_, scratch_);
        host_->mov(scratch_.cvt32(), 0x1);
        host_->vpbroadcastd(one_, scratch_.cvt32());

        host_->xor_(scratch_, scratch_);
        host_->mov(scratch_.cvt32(), 0x7fff);
        host_->vpbroadcastd(even_, scratch_.cvt32());

        host_->xor_(scratch_, scratch_);
        host_->mov(scratch_.cvt32(), selector_int32);
        host_->vpbroadcastd(selector_, scratch_.cvt32());
    }

    static cpu_isa_t get_isa() { return avx512_core; }

private:
    jit_generator *const host_;
    Zmm_t one_;
    Zmm_t even_;
    Zmm_t selector_;
    reg64_t scratch_;
    Zmm_t tr0_;
    Zmm_t tr1_;

    int encode_fixup_selector(int input, int output) {
        return ((output) << (4 * (input)));
    }

    enum {
        fixup_input_code_qnan = 0,
        fixup_input_code_snan = 1,
        fixup_input_code_ninf = 4,
        fixup_input_code_pinf = 5,
        fixup_output_code_copy_input = 1,
        fixup_output_code_qnan_input = 2,
    };
};

// performs element-by-element sum of inp and add float arrays and stores
// result to bfloat16 out array with downconversion
struct jit_avx512_core_add_cvt_ps_to_bf16_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_add_cvt_ps_to_bf16)

    jit_avx512_core_add_cvt_ps_to_bf16_t()
        : jit_generator(jit_name()), simd_w_(16) {
        bf16_emu_ = new bf16_emulation_t(
                this, one, even, selector, scratch, fp32_tmp, fp32_tmp);

        UNUSED_STATUS(create_kernel());
    }

    ~jit_avx512_core_add_cvt_ps_to_bf16_t() { delete bf16_emu_; }

    void generate() override {
        preamble();

        bool use_bf16_emu = !mayiuse(avx512_core_bf16);

        auto add_cvt = [&](size_t idx, Xbyak::Opmask ktail_mask) {
            vmovups(fp32_inp | ktail_mask | T_z,
                    ptr[reg_inp + sizeof(float) * (idx)]);
            vaddps(fp32_inp | ktail_mask | T_z, fp32_inp,
                    ptr[reg_add + sizeof(float) * (idx)]);
            if (use_bf16_emu)
                bf16_emu_->vcvtneps2bf16(bf16_out, fp32_inp);
            else
                vcvtneps2bf16(bf16_out, fp32_inp);

            vmovdqu16(yword[reg_out + sizeof(bfloat16_t) * (idx)] | ktail_mask,
                    bf16_out);
        };

        mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        mov(reg_add, ptr[abi_param1 + GET_OFF(add)]);
        mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
        mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);

        if (use_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

        mov(reg32_tail, 0xffff);
        kmovw(ktail_mask, reg32_tail);

        constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
        Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
        for (int i = n_unroll; i >= 0; i--) {
            const int unroll = 1 << i; // 4, 2, 1
            L(l_simd_loop[i + 1]);
            {
                cmp(reg_nelems, simd_w_ * unroll);
                jl(l_simd_loop[i], T_NEAR);
                for (int j = 0; j < simd_w_ * unroll; j += simd_w_) {
                    add_cvt(j, ktail_mask);
                }
                add(reg_inp, simd_w_ * unroll * sizeof(float));
                add(reg_add, simd_w_ * unroll * sizeof(float));
                add(reg_out, simd_w_ * unroll * sizeof(bfloat16_t));

                sub(reg_nelems, simd_w_ * unroll);
                jmp(l_simd_loop[i + 1], T_NEAR);
            }
        }
        L(l_simd_loop[0]);
        test(reg_nelems, reg_nelems);
        jz(l_simd_notail);
        // JIT of `tail_mask_ = (1 << (nelems_ % simd_w_)) - 1;`
        mov(reg32_mask, 1);
        mov(reg64_tail, reg_nelems);
        shl(reg32_mask, reg8_mask_shift);
        sub(reg32_mask, 1);
        kmovd(ktail_mask, reg32_mask);
        add_cvt(0, ktail_mask);
        L(l_simd_notail);

        postamble();
    }

    void operator()(bf16_support::jit_call_t *params) const {
        jit_generator::operator()(params);
        msan_unpoison(params->out, params->nelems * sizeof(bfloat16_t));
    }

private:
    int simd_w_;

    bf16_emulation_t *bf16_emu_;

    Xbyak::Opmask ktail_mask = k2;
    Xbyak::Zmm fp32_inp = Xbyak::Zmm(0);
    Xbyak::Zmm fp32_tmp = Xbyak::Zmm(1);

    Xbyak::Zmm one = Xbyak::Zmm(2);
    Xbyak::Zmm even = Xbyak::Zmm(3);
    Xbyak::Zmm selector = Xbyak::Zmm(4);
    Xbyak::Reg64 scratch = r15;

    Xbyak::Ymm bf16_out = Xbyak::Ymm(5);

    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_add = r11;
    Xbyak::Reg64 reg_nelems = rdx;

    Xbyak::Reg64 reg64_tail = rcx;
    Xbyak::Reg32 reg32_tail = ecx;
    Xbyak::Reg8 reg8_mask_shift = cl;
    Xbyak::Reg32 reg32_mask = r8d;
};

// implementation of reorder of part of tensor [s][16c] -> [S][16c][2s]
// it is required for quick implementation of 1x1 bf16 bwd_w jit kernel
// w/o using permw instruction inside
// TODO: consider modification/replacement for outer transformation jit kernel
struct jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_reorder_s16c_to_S16c2s)

    jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t()
        : jit_generator(jit_name()), simd_w_(16), in_stride_(16) {}

    jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t(int in_stride)
        : jit_generator(jit_name()), simd_w_(16), in_stride_(in_stride) {}

    ~jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t() {}

    void generate() override {
        preamble();

        mov(reg32_tail, ptr[abi_param1 + GET_OFF(mask)]);
        mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
        mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);

        auto zmm_reg = [=](int idx) {
            assert(idx < 31);
            return Xbyak::Zmm(idx);
        };

        kmovd(ktail_mask_lo, reg32_tail);
        kshiftld(ktail_mask_hi, ktail_mask_lo, 16);

        Xbyak::Label dst_prm_table;
        mov(reg_prm, dst_prm_table);
        vmovups(zmm_prm, ptr[reg_prm]);

        constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
        int sizeofcacheline = 2 * simd_w_ * sizeof(bfloat16_t);
        int in_stride_bytes = in_stride_ * sizeof(bfloat16_t);
        Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
        for (int i = n_unroll; i >= 0; i--) {
            const int unroll = 1 << i; // 4, 2, 1
            L(l_simd_loop[i + 1]);
            {
                cmp(reg_nelems, 2 * unroll);
                jl(l_simd_loop[i], T_NEAR);
                for (int j = 0; j < unroll; j++) {
                    auto zmm_inp = zmm_reg(j);
                    if (in_stride_ == 16)
                        vmovups(zmm_inp, zword[reg_inp + j * sizeofcacheline]);
                    else {
                        vmovdqu16(zmm_inp | ktail_mask_lo | T_z,
                                zword[reg_inp + 2 * j * in_stride_bytes]);
                        vmovdqu16(zmm_inp | ktail_mask_hi,
                                zword[reg_inp + (2 * j + 1) * in_stride_bytes
                                        - 32]);
                    }
                    vpermw(zmm_inp, zmm_prm, zmm_inp);
                    vmovups(zword[reg_out + j * sizeofcacheline], zmm_inp);
                }
                add(reg_inp,
                        unroll
                                * (in_stride_ == 16 ? sizeofcacheline
                                                    : 2 * in_stride_bytes));
                add(reg_out, unroll * sizeofcacheline);

                sub(reg_nelems, 2 * unroll);
                jmp(l_simd_loop[i + 1], T_NEAR);
            }
        }
        L(l_simd_loop[0]);

        test(reg_nelems, reg_nelems);
        jz(l_simd_notail);

        auto zmm_inp = zmm_reg(0);
        vpxord(zmm_inp, zmm_inp, zmm_inp);
        vmovdqu16(zmm_inp | ktail_mask_lo | T_z, ptr[reg_inp]);
        vpermw(zmm_inp, zmm_prm, zmm_inp);
        vmovups(zword[reg_out], zmm_inp);

        L(l_simd_notail);

        postamble();

        const uint16_t dst_prm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20,
                5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13,
                29, 14, 30, 15, 31};

        align(64);
        L(dst_prm_table);
        for (size_t i = 0; i < 32; ++i)
            dw(dst_prm_array[i]);
    }

    void operator()(bf16_support::jit_call_t *params) const {
        jit_generator::operator()(params);
        msan_unpoison(params->out, params->nelems * sizeof(bfloat16_t));
    }

private:
    int simd_w_;
    int in_stride_;

    Xbyak::Opmask ktail_mask_lo = k2;
    Xbyak::Opmask ktail_mask_hi = k3;
    Xbyak::Zmm zmm_prm = Xbyak::Zmm(31);

    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_prm = r11;
    Xbyak::Reg64 reg_nelems = rdx;

    Xbyak::Reg32 reg32_tail = abi_not_param1.cvt32();
};

#undef GET_OFF
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
