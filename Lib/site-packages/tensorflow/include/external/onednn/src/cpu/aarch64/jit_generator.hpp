/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_GENERATOR_HPP
#define CPU_AARCH64_JIT_GENERATOR_HPP

#include <limits.h>

#include "common/bit_cast.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

#include "cpu/jit_utils/jit_utils.hpp"

#if defined(_WIN32) && !defined(__GNUC__)
#define STRUCT_ALIGN(al, ...) __declspec(align(al)) __VA_ARGS__
#else
#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))
#endif

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { return STRINGIFY(jit_name); } \
    const char *source_file() const override { return __FILE__; }

static const size_t CSIZE = sizeof(uint32_t);

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// TODO: move this to jit_generator class?
namespace {

typedef enum {
    MAX_CODE_SIZE = 256 * 1024,
} max_code_size_t;

// Callee-saved registers
constexpr Xbyak_aarch64::Operand::Code abi_save_gpr_regs[]
        = {Xbyak_aarch64::Operand::X16, Xbyak_aarch64::Operand::X17,
                Xbyak_aarch64::Operand::X19, Xbyak_aarch64::Operand::X20,
                Xbyak_aarch64::Operand::X21, Xbyak_aarch64::Operand::X22,
                Xbyak_aarch64::Operand::X23, Xbyak_aarch64::Operand::X24,
                Xbyak_aarch64::Operand::X25, Xbyak_aarch64::Operand::X26,
                Xbyak_aarch64::Operand::X27, Xbyak_aarch64::Operand::X28};

// See "Procedure Call Standsard for the ARM 64-bit Architecture (AArch64)"
static const Xbyak_aarch64::XReg abi_param1(Xbyak_aarch64::Operand::X0),
        abi_param2(Xbyak_aarch64::Operand::X1),
        abi_param3(Xbyak_aarch64::Operand::X2),
        abi_param4(Xbyak_aarch64::Operand::X3),
        abi_param5(Xbyak_aarch64::Operand::X4),
        abi_param6(Xbyak_aarch64::Operand::X5),
        abi_param7(Xbyak_aarch64::Operand::X6),
        abi_param8(Xbyak_aarch64::Operand::X7),
        abi_not_param1(Xbyak_aarch64::Operand::X15);
} // namespace

class jit_generator : public Xbyak_aarch64::CodeGenerator, public c_compatible {
public:
    using c_compatible::operator new;
    using c_compatible::operator new[];
    using c_compatible::operator delete;
    using c_compatible::operator delete[];

private:
    const size_t xreg_len = 8;
    const size_t vreg_len_preserve = 8; // Only bottom 8byte must be preserved.
    const size_t vreg_to_preserve = 8; // VREG8 - VREG15

    const size_t num_abi_save_gpr_regs
            = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    const size_t preserved_stack_size = xreg_len * (2 + num_abi_save_gpr_regs)
            + vreg_len_preserve * vreg_to_preserve;

    const size_t size_of_abi_save_regs = num_abi_save_gpr_regs * x0.getBit() / 8
            + vreg_to_preserve * vreg_len_preserve;

public:
    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    const uint64_t cpu_sveLen = get_sve_length();

    const Xbyak_aarch64::WReg W_TMP_0 = w23;
    const Xbyak_aarch64::WReg W_TMP_1 = w24;
    const Xbyak_aarch64::WReg W_TMP_2 = w25;
    const Xbyak_aarch64::WReg W_TMP_3 = w26;
    const Xbyak_aarch64::WReg W_TMP_4 = w27;
    const Xbyak_aarch64::XReg X_TMP_0 = x23;
    const Xbyak_aarch64::XReg X_TMP_1 = x24;
    const Xbyak_aarch64::XReg X_TMP_2 = x25;
    const Xbyak_aarch64::XReg X_TMP_3 = x26;
    const Xbyak_aarch64::XReg X_TMP_4 = x27;
    const Xbyak_aarch64::XReg X_DEFAULT_ADDR = x28;
    const Xbyak_aarch64::XReg X_SP = x21;
    const Xbyak_aarch64::XReg X_TRANSLATOR_STACK = x22;
    const Xbyak_aarch64::PReg P_TMP = p7;
    const Xbyak_aarch64::PReg P_TMP_0 = p11;
    const Xbyak_aarch64::PReg P_TMP_1 = p12;
    const Xbyak_aarch64::PReg P_ALL_ZERO = p10;
    const Xbyak_aarch64::PReg P_NOT_256 = p13;
    const Xbyak_aarch64::PReg P_NOT_128 = p14;
    const Xbyak_aarch64::PReg P_ALL_ONE = p0;

    const std::vector<Xbyak_aarch64::XReg> x_tmp_vec
            = {X_TMP_0, X_TMP_1, X_TMP_2, X_TMP_3, X_TMP_4};
    const int x_tmp_vec_size = x_tmp_vec.size();

    const Xbyak_aarch64::XReg param1 = abi_param1;
    constexpr static size_t translator_stack_offset = 1024 * 128;
    constexpr static uint32_t DUMMY_IDX = 99;

    inline size_t get_size_of_abi_save_regs() { return size_of_abi_save_regs; }

    void preamble() {
        using namespace Xbyak_aarch64::util;
        uint64_t sveLen = get_sve_length();

        stp(x29, x30, pre_ptr(sp, -16));
        /* x29 is a frame pointer. */
        mov(x29, sp);
        sub(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);

        /* x9 can be used as a temporal register. */
        mov(x9, sp);

        if (vreg_to_preserve) {
            st4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve * 4));
            st4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve * 4));
        }
        for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
            stp(Xbyak_aarch64::XReg(abi_save_gpr_regs[i]),
                    Xbyak_aarch64::XReg(abi_save_gpr_regs[i + 1]),
                    post_ptr(x9, xreg_len * 2));
        }

        if (sveLen) { /* SVE is available. */
            ptrue(P_ALL_ONE.b);
            pfalse(P_ALL_ZERO.b);
        }
        if (sveLen >= SVE_256) {
            ptrue(P_NOT_128.b, Xbyak_aarch64::VL16);
            not_(P_NOT_128.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_128.b);
        }
        if (sveLen >= SVE_512) {
            ptrue(P_NOT_256.b, Xbyak_aarch64::VL32);
            not_(P_NOT_256.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_256.b);
        }

        mov(X_SP, sp);
        sub_imm(X_TRANSLATOR_STACK, X_SP, translator_stack_offset, X_TMP_0);
    }

    void postamble() {
        using namespace Xbyak_aarch64::util;

        mov(x9, sp);

        if (vreg_to_preserve) {
            ld4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve * 4));
            ld4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve * 4));
        }

        for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
            ldp(Xbyak_aarch64::XReg(abi_save_gpr_regs[i]),
                    Xbyak_aarch64::XReg(abi_save_gpr_regs[i + 1]),
                    post_ptr(x9, xreg_len * 2));
        }

        add(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);
        ldp(x29, x30, post_ptr(sp, 16));
        ret();
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak_aarch64::Label &label) {
        Xbyak_aarch64::CodeGenerator::L(label);
    }

    void L_aligned(Xbyak_aarch64::Label &label, int alignment = 16) {
        align(alignment);
        L(label);
    }

    template <typename T>
    Xbyak_aarch64::XReg addr_off(const Xbyak_aarch64::XReg &base, const T off,
            const Xbyak_aarch64::XReg &addr, const Xbyak_aarch64::XReg &x_tmp) {
        if (off == 0) return base;

        add_imm(addr, base, off, x_tmp);
        return addr;
    }

    template <typename PRegBHSD, typename T>
    void set_preg(const PRegBHSD &p, T tail_size,
            const Xbyak_aarch64::XReg x_tmp0 = Xbyak_aarch64::XReg(DUMMY_IDX),
            const Xbyak_aarch64::XReg x_tmp1 = Xbyak_aarch64::XReg(DUMMY_IDX)) {
        using namespace Xbyak_aarch64;

        assert(tail_size <= 64); // Implemented only for "SVE size <=  512"

        switch (tail_size) {
            case 0: pfalse(PRegB(p.getIdx())); return;
            case 1: ptrue(p, VL1); return;
            case 2: ptrue(p, VL2); return;
            case 3: ptrue(p, VL3); return;
            case 4: ptrue(p, VL4); return;
            case 5: ptrue(p, VL5); return;
            case 6: ptrue(p, VL6); return;
            case 7: ptrue(p, VL7); return;
            case 8: ptrue(p, VL8); return;
            case 16: ptrue(p, VL16); return;
            case 32: ptrue(p, VL32); return;
            case 64: ptrue(p, VL64); return;
        }

        assert(x_tmp0.getIdx() != DUMMY_IDX && x_tmp1.getIdx() != DUMMY_IDX);
        mov_imm(x_tmp0, 0);
        mov_imm(x_tmp1, tail_size);
        whilelt(p, x_tmp0, x_tmp1);
    }

    template <typename T>
    void uni_add(const T &x1, const T &x2, const T &op) {
        add(x1, x2, op);
    }

    void uni_add(const Xbyak_aarch64::VReg4S &x1,
            const Xbyak_aarch64::VReg4S &x2, const Xbyak_aarch64::VReg4S &op) {
        add(x1, x2, op);
    }

    void uni_add(const Xbyak_aarch64::ZReg &x1, const Xbyak_aarch64::ZReg &x2,
            const Xbyak_aarch64::ZReg &op) {
        add(Xbyak_aarch64::ZRegS(x1.getIdx()),
                Xbyak_aarch64::ZRegS(x2.getIdx()),
                Xbyak_aarch64::ZRegS(op.getIdx()));
    }

    template <typename T>
    void udiv_mod(const T &q, const T &r, const T &divend, const T &divisor) {
        assert(q.getIdx() != divisor.getIdx());
        assert(q.getIdx() != divend.getIdx());
        assert(r.getIdx() != divend.getIdx());

        udiv(q, divend, divisor);
        mul(r, q, divisor);
        sub(r, divend, r);
    }

    template <typename T>
    void umod(const T &r, const T &divend, const T &divisor) {
        assert(r.getIdx() != divend.getIdx());
        assert(r.getIdx() != divisor.getIdx());

        udiv(r, divend, divisor);
        mul(r, r, divisor);
        sub(r, divend, r);
    }

    void uni_clear(const Xbyak_aarch64::VReg &dst) { eor(dst.b, dst.b, dst.b); }

    void uni_clear(const Xbyak_aarch64::ZReg &dst) { eor(dst.d, dst.d, dst.d); }

    template <typename T>
    void uni_fadd(const T &dst, const T &src, const T &src2) {
        fadd(dst, src, src2);
    }

    void uni_fcvtzs(
            const Xbyak_aarch64::VReg4S &d, const Xbyak_aarch64::VReg4S &s) {
        fcvtzs(d, s);
    }

    void uni_fcvtzs(
            const Xbyak_aarch64::ZRegS &d, const Xbyak_aarch64::ZRegS &s) {
        fcvtzs(d, P_ALL_ONE / Xbyak_aarch64::T_z, s);
    }

    template <typename TReg>
    void uni_fdiv(const TReg &dst, const TReg &src, const TReg &src2,
            const TReg &tmp, const Xbyak_aarch64::PReg &pred) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        uint32_t src2Idx = src2.getIdx();
        uint32_t tmpIdx = tmp.getIdx();

        if (dstIdx == src2Idx) {
            assert(tmpIdx != srcIdx && tmpIdx != src2Idx);

            mov(Xbyak_aarch64::ZRegD(tmpIdx), Xbyak_aarch64::ZRegD(src2Idx));
            mov(dst, pred / Xbyak_aarch64::T_m, src);
            fdiv(dst, pred / Xbyak_aarch64::T_m, tmp);
        } else if (dstIdx == srcIdx) {
            fdiv(dst, pred / Xbyak_aarch64::T_m, src2);
        } else {
            mov(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src);
            fdiv(dst, pred / Xbyak_aarch64::T_m, src2);
        }
    }

    template <typename TReg>
    void uni_fdiv(const TReg &dst, const TReg &src, const TReg &src2) {
        fdiv(dst, src, src2);
    }

    void uni_fdiv(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src, const Xbyak_aarch64::VReg4S &src2,
            const Xbyak_aarch64::VReg4S &tmp, const Xbyak_aarch64::PReg &pred) {
        UNUSED(tmp);
        UNUSED(pred);
        fdiv(dst, src, src2);
    }

    template <typename T>
    void uni_fmad(const T &dst, const T &src, const T &src2) {
        fmad(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src, src2);
    }

    void uni_fmad(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src,
            const Xbyak_aarch64::VReg4S &src2) {
        fmul(dst, dst, src);
        fadd(dst, dst, src2);
    }

    template <typename T>
    void uni_fmax(const T &dst, const T &src, const T &src2) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        if (dstIdx != srcIdx)
            mov(Xbyak_aarch64::ZRegD(dstIdx), Xbyak_aarch64::ZRegD(srcIdx));
        fmax(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src2);
    }

    template <typename T>
    void uni_fmaxnm(const T &dst, const T &src, const T &src2) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        if (dstIdx != srcIdx)
            mov(Xbyak_aarch64::ZRegD(dstIdx), Xbyak_aarch64::ZRegD(srcIdx));
        fmaxnm(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src2);
    }

    void uni_fmaxnm(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src,
            const Xbyak_aarch64::VReg4S &src2) {
        fmaxnm(dst, src, src2);
    }

    template <typename T>
    void uni_fmin(const T &dst, const T &src, const T &src2) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        if (dstIdx != srcIdx)
            mov(Xbyak_aarch64::ZRegD(dstIdx), Xbyak_aarch64::ZRegD(srcIdx));
        fmin(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src2);
    }

    template <typename T>
    void uni_fmul(const T &dst, const T &src, const T &src2) {
        fmul(dst, src, src2);
    }

    void uni_frinti(
            const Xbyak_aarch64::VReg4S &d, const Xbyak_aarch64::VReg4S &s) {
        frinti(d, s);
    }

    void uni_frinti(
            const Xbyak_aarch64::ZRegS &d, const Xbyak_aarch64::ZRegS &s) {
        frinti(d, P_ALL_ONE / Xbyak_aarch64::T_m, s);
    }

    template <typename T>
    void uni_fsqrt(const T &dst, const T &src) {
        fsqrt(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src);
    }

    void uni_fsqrt(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src) {
        fsqrt(dst, src);
    }

    void uni_fsub(const Xbyak_aarch64::VReg4S &v1,
            const Xbyak_aarch64::VReg4S &v2, const Xbyak_aarch64::VReg4S &v3) {
        fsub(v1, v2, v3);
    }

    template <typename T>
    void uni_fsub(const T &dst, const T &src, const T &src2) {
        fsub(dst, src, src2);
    }

    void uni_fsub(const Xbyak_aarch64::ZRegS &z1,
            const Xbyak_aarch64::ZRegS &z2, const Xbyak_aarch64::ZRegS &z3) {
        fsub(z1, z2, z3);
    }

    void uni_eor(const Xbyak_aarch64::VReg &v1, const Xbyak_aarch64::VReg &v2,
            const Xbyak_aarch64::VReg &v3) {
        eor(Xbyak_aarch64::VReg16B(v1.getIdx()),
                Xbyak_aarch64::VReg16B(v2.getIdx()),
                Xbyak_aarch64::VReg16B(v3.getIdx()));
    }

    void uni_eor(const Xbyak_aarch64::ZReg &z1, const Xbyak_aarch64::ZReg &z2,
            const Xbyak_aarch64::ZReg &z3) {
        eor(Xbyak_aarch64::ZRegD(z1.getIdx()),
                Xbyak_aarch64::ZRegD(z2.getIdx()),
                Xbyak_aarch64::ZRegD(z3.getIdx()));
    }

    void uni_ld1rw(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::XReg &base, const int64_t off) {
        if (off == 0) {
            ld1r(dst, ptr(base));
        } else {
            add_imm(X_DEFAULT_ADDR, base, off, X_TMP_0);
            ld1r(dst, ptr(X_DEFAULT_ADDR));
        }
    }

    void uni_ld1rw(const Xbyak_aarch64::ZRegS &dst,
            const Xbyak_aarch64::XReg &base, const int64_t off) {
        if (-32 <= off && off < 32) {
            ld1rw(dst, P_ALL_ONE / Xbyak_aarch64::T_z, ptr(base, (int)off));
        } else {
            add_imm(X_DEFAULT_ADDR, base, off, X_TMP_0);
            ld1rw(dst, P_ALL_ONE / Xbyak_aarch64::T_z, ptr(X_DEFAULT_ADDR));
        }
    }

    void uni_ldr(
            const Xbyak_aarch64::VReg &dst, const Xbyak_aarch64::XReg &addr) {
        ldr(Xbyak_aarch64::QReg(dst.getIdx()), ptr(addr));
    }

    void uni_ldr(
            const Xbyak_aarch64::ZReg &dst, const Xbyak_aarch64::XReg &addr) {
        ldr(dst, ptr(addr));
    }

    template <typename T>
    void uni_ldr(const Xbyak_aarch64::ZReg &r, const Xbyak_aarch64::XReg &base,
            const T off) {
        const int off_mod = off % cpu_sveLen;
        const int off_mul_vl = off / cpu_sveLen;

        if (off_mod == 0 && -256 <= off_mul_vl && off_mul_vl <= 255) {
            ldr(r, Xbyak_aarch64::ptr(base, off_mul_vl, Xbyak_aarch64::MUL_VL));
        } else {
            const int offset = off_mod * 0x10 * (cpu_sveLen / 16);
            add_imm(X_DEFAULT_ADDR, base, offset, X_TMP_0);
            ldr(r, Xbyak_aarch64::ptr(X_DEFAULT_ADDR));
        }
    }

    template <typename T>
    void uni_ldr(const Xbyak_aarch64::VReg &r, const Xbyak_aarch64::XReg &base,
            const T off) {
        const int off_mod = off % 16;
        const int off_mul_vl = off / 16;

        if (off_mod == 0 && 0 <= off_mul_vl && off_mul_vl <= 65520) {
            ldr(Xbyak_aarch64::QReg(r.getIdx()), Xbyak_aarch64::ptr(base, off));
        } else {
            add_imm(X_DEFAULT_ADDR, base, off_mod * 4, X_TMP_0);
            ldr(Xbyak_aarch64::QReg(r.getIdx()),
                    Xbyak_aarch64::ptr(X_DEFAULT_ADDR));
        }
    }

    void uni_orr(const Xbyak_aarch64::VReg &d, const Xbyak_aarch64::VReg &s0,
            const Xbyak_aarch64::VReg &s1) {
        orr(d.b16, s0.b16, s1.b16);
    }

    void uni_orr(const Xbyak_aarch64::ZReg &d, const Xbyak_aarch64::ZReg &s0,
            const Xbyak_aarch64::ZReg &s1) {
        orr(d.d, s0.d, s1.d);
    }

    void uni_scvtf(
            const Xbyak_aarch64::VReg4S &t0, const Xbyak_aarch64::VReg4S &t1) {
        scvtf(t0, t1);
    }

    void uni_scvtf(
            const Xbyak_aarch64::ZRegS &t0, const Xbyak_aarch64::ZRegS &t1) {
        scvtf(t0, P_ALL_ONE / Xbyak_aarch64::T_m, t1);
    }

    void uni_str(
            const Xbyak_aarch64::VReg &src, const Xbyak_aarch64::XReg &addr) {
        str(Xbyak_aarch64::QReg(src.getIdx()), ptr(addr));
    }

    void uni_str(
            const Xbyak_aarch64::ZReg &src, const Xbyak_aarch64::XReg &addr) {
        str(src, ptr(addr));
    }

    template <typename T>
    void uni_sub(const T &x1, const T &x2, const T &op) {
        sub(x1, x2, op);
    }

    /*
      Saturation facility functions. enable to prepare the register
      holding the saturation upperbound and apply the saturation on
      the floating point register
     */
    template <typename Vmm>
    void init_vmm(Vmm vmm, Xbyak_aarch64::XReg reg_tmp, float value) {
        using namespace data_type;
        bool isSVE = get_sve_length() ? true : false;
        Xbyak_aarch64::ZRegS z_tmp(vmm.getIdx());
        Xbyak_aarch64::VReg4S v_tmp(vmm.getIdx());
        Xbyak_aarch64::WReg w_tmp(reg_tmp.getIdx());
        mov_imm(w_tmp, float2int(value));
        if (isSVE) /* SVE is available. */
            dup(z_tmp, w_tmp);
        else
            dup(v_tmp, w_tmp);
    }

    template <typename Vmm>
    void init_saturate_f32(Vmm vmm_lbound, Vmm vmm_ubound,
            Xbyak_aarch64::XReg reg_tmp, data_type_t idt, data_type_t odt,
            bool force_lbound = false) {
        using namespace data_type;
        bool isSVE = get_sve_length() ? true : false;

        if (!((idt == f32) && utils::one_of(odt, u8, data_type::s8, s32)))
            return;

        assert(IMPLICATION(
                idt == u8, vmm_lbound.getIdx() != vmm_ubound.getIdx()));
        // No need to saturate on lower bound for signed integer types, as
        // the conversion to int would return INT_MIN, and then proper
        // saturation will happen in store_data. The param force_lbound, will
        // force saturate values unconditionally to lbound.
        if (odt == u8) {
            if (isSVE) /* SVE is available. */
                dup(Xbyak_aarch64::ZRegS(vmm_lbound.getIdx()), 0);
            else if (mayiuse(asimd))
                movi(Xbyak_aarch64::VReg4S(vmm_lbound.getIdx()), 0);
            else
                assert(!"unreachable");
        } else if (force_lbound) {
            const float saturation_lbound
                    = odt == data_type::s8 ? INT8_MIN : INT32_MIN;
            init_vmm(vmm_lbound, reg_tmp, saturation_lbound);
        }

        float saturation_ubound = types::max_value<float>(odt);
        init_vmm(vmm_ubound, reg_tmp, saturation_ubound);
    }

    template <typename Vmm>
    void saturate_f32(const Vmm &vmm, const Vmm &vmm_lbound,
            const Vmm &vmm_ubound, data_type_t odt,
            const Xbyak_aarch64::PReg &p_true, bool force_lbound = false) {
        // This function is used to saturate to odt in f32 before converting
        // to s32 in order to avoid bad saturation due to cvtps2dq
        // behavior (it returns INT_MIN if the f32 is out of the
        // s32 range)
        using namespace data_type;
        bool isSVE = get_sve_length() ? true : false;

        if (!utils::one_of(odt, u8, data_type::s8, s32)) return;

        Xbyak_aarch64::VReg4S v_tmp(vmm.getIdx());
        Xbyak_aarch64::VReg4S v_lbound(vmm_lbound.getIdx());
        Xbyak_aarch64::VReg4S v_ubound(vmm_ubound.getIdx());
        Xbyak_aarch64::ZRegS z_tmp(vmm.getIdx());
        Xbyak_aarch64::ZRegS z_lbound(vmm_lbound.getIdx());
        Xbyak_aarch64::ZRegS z_ubound(vmm_ubound.getIdx());

        // no need to apply lower saturation bound when odt is
        // signed, as cvtps2dq will return MIN_INT if the value
        // does not fit. The param force_lbound, will force saturate values
        // unconditionally to lbound.
        if (odt == u8 || force_lbound) {
            if (isSVE) /* SVE is available. */
                fmax(z_tmp, p_true / Xbyak_aarch64::T_m, z_lbound);
            else if (mayiuse(asimd))
                fmax(v_tmp, v_tmp, v_lbound);
            else
                assert(!"unreachable");
        }
        if (isSVE) /* SVE is available. */
            fmin(z_tmp, p_true / Xbyak_aarch64::T_m, z_ubound);
        else if (mayiuse(asimd))
            fmin(v_tmp, v_tmp, v_ubound);
        else
            assert(!"unreachable");
    }

    /* A utility function to process f32 tail (load, store or other) depending
     * on tail size, stored in Reg64. Tail size must be value from 0 to 3/7
     * (Xmm/Ymm). Tail process functions require integer as argument to specify
     * behavior for each tail size.
     *
     * Only supported for Xmm and Ymm.
     */
    template <cpu_isa_t isa>
    void runtime_tail_process(const Xbyak_aarch64::XReg &reg_tail,
            const Xbyak_aarch64::XReg &reg_tmp,
            const std::function<void(int)> &tail_process) {
        constexpr int simd_w_ymm = 8;
        constexpr int f32_bits = sizeof(float) * 8;
        const auto simd_w = cpu_isa_traits<isa>::vlen * 8 / f32_bits;
        assert(simd_w != cpu_isa_traits<isa>::vlen * 8 / f32_bits);

        Xbyak_aarch64::Label label_tbl, label_tbl_end;
        Xbyak_aarch64::Label l_case[simd_w_ymm];

        adr(reg_tmp, label_tbl);
        mov_imm(X_TMP_0, sizeof(void *));
        madd(X_DEFAULT_ADDR, reg_tail, X_TMP_0, reg_tmp);
        br(X_DEFAULT_ADDR);

        // create jump table
        L(label_tbl);
        for (size_t i = 0; i < simd_w; i++)
            putL(l_case[i]);

        // cases for each tail size - from 0 to 3/7
        L(l_case[0]);
        b(label_tbl_end);
        for (size_t i = 1; i < simd_w; i++) {
            L(l_case[i]);
            tail_process(i);
            b(label_tbl_end);
        }
        L(label_tbl_end);
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator);

public:
    jit_generator(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true, cpu_isa_t max_cpu_isa = isa_all)
        : Xbyak_aarch64::CodeGenerator(code_size,
                (code_ptr == nullptr && use_autogrow) ? Xbyak_aarch64::AutoGrow
                                                      : code_ptr)
        , max_cpu_isa_(max_cpu_isa) {}
    virtual ~jit_generator() {}

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    void register_jit_code(const uint8_t *code, size_t code_size) const {
        jit_utils::register_jit_code(code, code_size, name(), source_file());
    }

    const uint8_t *jit_ker() const { return jit_ker_; }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual status_t create_kernel() {
        generate();
        jit_ker_ = getCode();
        return (jit_ker_) ? status::success : status::runtime_error;
    }

private:
    const cpu_isa_t max_cpu_isa_;
    const uint8_t *getCode() {
        this->ready();
        if (!is_initialized()) return nullptr;
        const uint8_t *code
                = reinterpret_cast<const uint8_t *>(CodeGenerator::getCode());
        register_jit_code(code, getSize() * CSIZE);
        return code;
    }

    inline bool is_valid_isa(cpu_isa_t isa) {
        return is_subset(isa, max_cpu_isa_) && mayiuse(isa);
    }

    static inline bool is_initialized() {
        /* At the moment, Xbyak_aarch64 does not have GetError()\
         so that return dummy result. */
        return true;
    }

protected:
    virtual void generate() = 0;
    const uint8_t *jit_ker_ = nullptr;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
