/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#ifndef CPU_X64_JIT_UNI_POOL_KERNEL_HPP
#define CPU_X64_JIT_UNI_POOL_KERNEL_HPP

#include <cfloat>
#include <functional>
#include <memory>

#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct bf16_emulation_t;

template <cpu_isa_t isa>
struct jit_uni_pool_kernel : public jit_generator {

    jit_uni_pool_kernel(
            const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md);
    jit_pool_conf_t jpp;
    ~jit_uni_pool_kernel();

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_kernel)

    static status_t init_conf(jit_pool_conf_t &jbp,
            memory_tracking::registrar_t &scratchpad, primitive_attr_t &attr,
            const pooling_pd_t *ppd);

private:
    using Xmm = Xbyak::Xmm;
    using Ymm = Xbyak::Ymm;
    using Zmm = Xbyak::Zmm;
    using Opmask = Xbyak::Opmask;
    using Reg32 = Xbyak::Reg32;
    using Reg64 = Xbyak::Reg64;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    int vmm_idx_upper_bound() const noexcept {
        return is_superset(isa, avx512_core) ? 31 : 15;
    }

    int reg_idx(int idx) const noexcept { return vmm_idx_upper_bound() - idx; }

    Xmm xreg(int idx) const noexcept { return Xmm(reg_idx(idx)); }
    Ymm yreg(int idx) const noexcept { return Ymm(reg_idx(idx)); }
    Zmm zreg(int idx) const noexcept { return Zmm(reg_idx(idx)); }
    Vmm vreg(int idx) const noexcept { return Vmm(reg_idx(idx)); }

    const Xbyak::AddressFrame &vmmword = (isa == sse41)  ? xword
            : utils::one_of(isa, avx, avx2, avx2_vnni_2) ? yword
                                                         : zword;

    Xmm vmm_mask = Xmm(0);
    Xmm xmm_tmp_1 = Xmm(0);
    Ymm ymm_tmp_1 = Ymm(0);
    Vmm vmm_tmp_1 = Vmm(0);

    // Used only for avx and if c tail is present
    Vmm vmm_c_tail_mask = Vmm(2);
    Xmm xmm_c_tail_mask = Xmm(2);

    Xmm xmm_tmp = Xmm(3);

    Vmm vmm_ker_area_h = Vmm(2);
    Vmm vmm_one = Vmm(2);
    Vmm vmm_tmp = Vmm(3);
    Ymm ymm_tmp = Ymm(3);

    Vmm vmm_k_offset = Vmm(1);

    // Used only for avx512 when bf16 is present
    inline Vmm vmm_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? Vmm(4) : Vmm(1);
        } else
            return Vmm(4);
    }

    Zmm bf16_emu_reserv_1 = Zmm(5);
    Zmm bf16_emu_reserv_2 = Zmm(6);
    Zmm bf16_emu_reserv_3 = Zmm(7);
    Reg64 bf16_emu_reserv_4 = r11;
    Zmm bf16_emu_reserv_5 = Zmm(8);

    Opmask k_c_tail_mask = Opmask(4);
    Opmask k_mask_cvt = Opmask(5);
    Opmask k_store_mask = Opmask(6);

    // Here be some (tame) dragons. This kernel does not follow the regular
    // OS-agnostic ABI pattern because when isa is sse41 it uses maskmovdqu
    // instruction which has its destination hardcoded in rdi. Therefore:
    // - all registers are hardcoded
    // - on Windows rdi and rcx are swapped to mimic the Unix x86_64 ABI
    //
    // While this is only required by the backward pass, the quirk above
    // is applied to the forward pass as well to keep things simpler.

    using reg64_t = const Reg64;
    reg64_t reg_param = rdi; // Always mimic the Unix ABI
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t reg_index = r10;
    reg64_t reg_output = r12;
    reg64_t reg_kd_pad_shift = r13;
    reg64_t dst_ptr = rdi; // Must be rdi due to maskmovdqu

    reg64_t kj = r14;
    reg64_t oi_iter = r15;
    reg64_t reg_kh = rax;
    reg64_t reg_k_shift = rbx;
    reg64_t tmp_gpr = rcx; // Must be rcx because rdi is used above
    reg64_t reg_ker_area_h = rdx;
    reg64_t reg_nbc = rsi;

    reg64_t reg_zero_ptr = r9;
    reg64_t reg_zero_id = r13;
    reg64_t reg_zero_ih = r14;
    reg64_t aux_reg_zero_ih = r15;
    reg64_t ki = r12;
    reg64_t aux_reg_input_d = r8;

    Reg32 reg_shuf_mask = esi;

    bool sse_high_half = false;
    bool disable_postops_when_sse_high_half_processed_ = false;

    int prev_kw;

    void prepare_tail_mask();
    void put_one_in_vmm();
    void uni_broadcast_reg_val(const int reg_idx, const int vmm_idx);
    void push_vmm_val(const int idx);
    void pop_vmm_val(const int idx);
    void load(const int idx, const reg64_t &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);
    void store(const int idx, const reg64_t &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void avg_step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_fwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_bwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);

    void zero_diff_src(int ur_bc, bool with_c_tail_proccessing);

    void step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing) {
        if (jpp.alg == alg_kind::pooling_max) {
            if (jpp.is_backward)
                max_step_bwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
            else
                max_step_fwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
        } else
            avg_step(ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
    }

    void step_high_half(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_processing) {
        add(reg_input, sizeof(float) * 4);
        add(reg_output, sizeof(float) * 4);
        if (jpp.alg == alg_kind::pooling_max
                && (jpp.is_training || jpp.is_backward))
            add(reg_index, types::data_type_size(jpp.ind_dt) * 4);

        step(ur_w, ur_bc, pad_l, pad_r, with_c_tail_processing);
    }

    void generate() override;

    void avx_vpadd1(const Ymm &y0, const Xmm &x1, const Xmm &xtmp) {
        assert(y0.getIdx() != x1.getIdx());
        vextractf128(xtmp, y0, 0);
        vpaddd(xtmp, xtmp, x1);
        vinsertf128(y0, y0, xtmp, 0);
        vextractf128(xtmp, y0, 1);
        vpaddd(xtmp, xtmp, x1);
        vinsertf128(y0, y0, xtmp, 1);
    }

    void avx_vpadd1(const Xmm &x0, const Xmm &x1, const Xmm &) {
        assert(false /*function should not be used*/);
        paddd(x0, x1);
    }

    void avx_pmovzxbd(const Ymm &y0, const Xmm &x1, const Xmm &xtmp) {
        Xmm x0(y0.getIdx());
        pshufd(xmm_tmp, x1, 1);
        pmovzxbd(x0, x1);
        pmovzxbd(xmm_tmp, xmm_tmp);
        vinsertf128(y0, y0, xmm_tmp, 1);
    }

    void avx_pmovzxbd(const Xmm &x0, const Xmm &x1, const Xmm &) {
        assert(false /*function should not be used*/);
        pmovzxbd(x0, x1);
    }

    void avx_pcmpeqd(
            const Ymm &y0, const Ymm &y1, const Ymm &y2, const Xmm &xtmp) {
        assert(y0.getIdx() != y1.getIdx());
        assert(y0.getIdx() != y2.getIdx());
        Xmm x0(y0.getIdx());
        Xmm x2(y2.getIdx());
        vextractf128(x0, y1, 1);
        vextractf128(xtmp, y2, 1);
        pcmpeqd(xtmp, x0);
        vextractf128(x0, y1, 0);
        pcmpeqd(x0, x2);
        vinsertf128(y0, y0, xtmp, 1);
    }

    void avx_pcmpeqd(const Xmm &x0, const Xmm &x1, const Xmm &, const Xmm &) {
        assert(false /*function should not be used*/);
        pcmpeqd(x0, x1);
    }

    void apply_postops(int ur_bc, int ur_w, int c_block,
            const std::function<bool(int, bool)> &is_tail_predicate);

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    inline bool use_bf16_emulation() const {
        return jpp.is_bf16 && !isa_has_bf16(jpp.isa) && isa != avx2_vnni_2;
    }

    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
