/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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
#ifndef CPU_AARCH64_JIT_BRDGMM_KERNEL_HPP
#define CPU_AARCH64_JIT_BRDGMM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
struct jit_brdgmm_kernel_base_t : public jit_generator {
    jit_brdgmm_kernel_base_t(const brgemm_t &abrd);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brdgmm_kernel_base_t)

    brgemm_t brg;

    static bool is_fast_vnni_int8(const brgemm_t &brg) {
        return brg.is_dgmm && brg.is_int8 && brg.ldb_tail /*n_vlen_tail*/ == 0;
    }

private:
    using po_injector_t = injector::jit_uni_postops_injector_t<sve_512>;
    std::unique_ptr<po_injector_t> postops_injector_;

    Xbyak_aarch64::Label permute_index_table;

    // Register decomposition
    const Xbyak_aarch64::XReg param1 = x0;
    const Xbyak_aarch64::XReg reg_A = x1;
    const Xbyak_aarch64::XReg reg_B = x8;
    const Xbyak_aarch64::XReg reg_aux_batch_addr = x15;
    const Xbyak_aarch64::XReg reg_BS = x6;

    // loop variables
    const Xbyak_aarch64::XReg reg_BS_loop = x12;
    const Xbyak_aarch64::XReg reg_aux_M = x13;
    const Xbyak_aarch64::XReg reg_aux_D = x3;
    const Xbyak_aarch64::XReg reg_aux_C = x2;
    const Xbyak_aarch64::XReg reg_aux_A = x10;
    const Xbyak_aarch64::XReg reg_aux_B = x7;
    const Xbyak_aarch64::XReg reg_aux1_A = reg_A; // brgemm_strd
    const Xbyak_aarch64::XReg reg_aux1_B = reg_B; // brgemm_strd
    const Xbyak_aarch64::XReg reg_a_offset = x9;
    const Xbyak_aarch64::XReg reg_aux_N = x11;

    const Xbyak_aarch64::XReg reg_aux_A_vpad_top = x14;
    const Xbyak_aarch64::XReg reg_aux_A_vpad_bottom = x5;

    const Xbyak_aarch64::XReg reg_table_base = x4;
    const Xbyak_aarch64::XReg reg_tmp = reg_table_base;
    const Xbyak_aarch64::XReg reg_total_padding = reg_table_base;
    const Xbyak_aarch64::XReg reg_aux_bias = reg_table_base;
    const Xbyak_aarch64::XReg reg_aux_scales = reg_table_base;
    const Xbyak_aarch64::XReg reg_aux_dst_scales = reg_table_base;
    const Xbyak_aarch64::XReg reg_binary_params = x7; // default for binary ops
    const Xbyak_aarch64::XReg reg_ptr_sum_scale = reg_aux_A_vpad_top;
    const Xbyak_aarch64::XReg reg_ptr_sum_zp = reg_aux_A_vpad_bottom;

    Xbyak_aarch64::PReg k_mask = Xbyak_aarch64::PReg(2);
    Xbyak_aarch64::PReg k_tail_mask = Xbyak_aarch64::PReg(3);
    Xbyak_aarch64::PReg kblend_mask = Xbyak_aarch64::PReg(4);

    const int simd_w_;
    constexpr static int max_vmms_ = 32;
    constexpr static int reg_batch0_addr_offs_ = 0;
    constexpr static int reg_bias_offs_ = 8;
    constexpr static int reg_scales_offs_ = 16;
    constexpr static int reg_A_offs_ = 24; // brgemm_strd
    constexpr static int reg_B_offs_ = 32; // brgemm_strd
    constexpr static int abi_param1_offs_ = 40;
    constexpr static int reg_dst_scales_offs_ = 48;
    constexpr static int stack_space_needed_ = 56;

    bool with_binary_non_scalar_bcast_ = false;

    inline int M() { return brg.bcast_dim; }
    inline int N() { return brg.load_dim; }
    inline int m_block1() { return brg.bd_block; }
    inline int nb_m_block1() { return brg.bdb; }
    inline int m_block1_tail() { return brg.bdb_tail; }
    inline int m_block2() { return brg.bd_block2; }
    inline int nb_m_block2() { return brg.bdb2; }
    inline int m_block2_tail() { return brg.bdb2_tail; }

    inline int n_block1() { return brg.ld_block; }
    inline int nb_n_block1() { return brg.ldb; }
    inline int n_block1_tail() { return brg.ldb_tail; }
    inline int n_block2() { return brg.ld_block2; }
    inline int nb_n_block2() { return brg.ldb2; }
    inline int n_block2_tail() { return brg.ldb2_tail; }

    int tail_length() { return n_block1_tail() % simd_w_; }
    bool is_fma_embd() { return brg.is_f32; }
    bool is_fast_vnni_int8() { return is_fast_vnni_int8(brg); }
    int vnni_substep() {
        return brg.isa_impl == sve_256 && (brg.is_bf16 || brg.is_f16) ? 2 : 1;
    }
    int get_substep_simd(int n_i, int v_i, bool has_n_tail) {
        const int last_n_block_sz
                = n_block2_tail() > 0 ? n_block2_tail() : n_block2();
        if (has_n_tail && n_i + 1 == last_n_block_sz) {
            return nstl::min(simd_w_, n_block1_tail() - v_i * simd_w_);
        } else {
            return simd_w_;
        }
    }
    Xbyak_aarch64::ZReg vmm_permute() {
        return Xbyak_aarch64::ZReg(0);
    } // used in fast_vnni_int8
    Xbyak_aarch64::ZReg vmm_a() {
        return Xbyak_aarch64::ZReg(is_fast_vnni_int8());
    }
    Xbyak_aarch64::ZReg vmm_b(int bi = 0) {
        return Xbyak_aarch64::ZReg(is_fast_vnni_int8() + !is_fma_embd() + bi);
    }
    Xbyak_aarch64::ZReg accm(
            int m_blocks, int n_blocks, int m, int n, int vnni_idx) {
        assert(m_blocks <= m_block2() && m < m_blocks);
        assert(n_blocks <= n_block2() && n < n_blocks);
        const int accm_start = max_vmms_ - m_blocks * n_blocks * vnni_substep();
        const int accm_rel_idx
                = m * n_blocks * vnni_substep() + n * vnni_substep() + vnni_idx;
        const uint32_t idx = accm_start + accm_rel_idx;
        assert(idx < max_vmms_ && idx > vmm_b(0).getIdx());
        return Xbyak_aarch64::ZReg(idx);
    }
    Xbyak_aarch64::ZReg vmm_tmp(int i) {
        const int idx
                = max_vmms_ - m_block2() * n_block2() * vnni_substep() - 1 - i;
        assert(idx > (is_fast_vnni_int8() - 1));
        return Xbyak_aarch64::ZReg(idx);
    }
    int ld_idx = 0;
    bool is_push = false;
    Xbyak_aarch64::ZReg push_z_tmp(
            Xbyak_aarch64::ZReg dst_zreg, Xbyak_aarch64::ZReg src_zreg) {
        int dsc_idx = dst_zreg.getIdx();
        int src_idx = src_zreg.getIdx();
        int idx;
        for (idx = ld_idx; idx < max_vmms_; idx++) {
            if ((idx != dsc_idx) && (idx != src_idx)) break;
        }
        assert(idx < max_vmms_);
        Xbyak_aarch64::ZReg z_tmp = Xbyak_aarch64::ZReg(idx);
        if ((ld_idx < idx) && (idx <= dsc_idx)) {
            is_push = false;
        } else {
            str(z_tmp, ptr(X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
            is_push = true;
        }
        return z_tmp;
    }
    void pop_z_tmp(Xbyak_aarch64::ZReg z_tmp) {
        if (is_push) {
            ldr(z_tmp, ptr(X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
        }
    }

    void init_masks();
    void read_params();
    void load_accumulators(int m_blocks, int n_blocks);
    void restore_A_B_matrices();
    void set_A_B_matrices();
    void advance_A_B_matrices();
    void load_a(Xbyak_aarch64::ZReg vmma, int m_i, int n_i, int v_i,
            bool has_n_tail);
    void load_b(Xbyak_aarch64::ZReg vmmb, int n_i, int v_i, bool has_n_tail);
    void brdgmm_microkernel(int m_blocks, int n_blocks, bool has_top_padding,
            bool has_bottom_padding, bool has_tail = false);
    void compute_loop();
    void batch_loop(const int m_blocks, const int n_blocks, bool has_n_tail);
    void cvt2ps(data_type_t type_in, const Xbyak_aarch64::ZReg vmm_in,
            const Xbyak_aarch64::AdrNoOfs &op, bool mask_flag,
            bool store); //for only memory operand
    void apply_post_ops(int m_blocks, int n_blocks, bool has_n_tail);
    void maybe_transpose_interleaved_vnni_to_plain(
            int m_blocks, int n_blocks, bool has_n_tail);
    void store_accumulators(int m_blocks, int n_blocks, bool has_n_tail);
    void store_accumulators_without_post_ops(
            int m_blocks, int n_blocks, bool has_n_tail);
    void store_accumulators_apply_post_ops(
            int m_blocks, int n_blocks, bool has_n_tail);

    bool has_vpad() {
        return brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0;
    }
    bool check_effective_padding() { return has_vpad() && M() > m_block2(); }

    int oc_logical_offset(int n) { return n * n_block1(); }
    int A_offset(int m, int n) {
        return brg.typesize_A * (m * brg.LDA + n * n_block1());
    }
    int B_offset(int n) { return brg.typesize_B * n * n_block1(); }
    int C_offset(int m, int n, int v) {
        return brg.typesize_C * (m * brg.LDC + n * n_block1() + v * simd_w_);
    }
    int D_offset(int m, int n, int v) {
        return brg.typesize_D * (m * brg.LDD + n * n_block1() + v * simd_w_);
    }
    int bias_offset(int n, int v) {
        return brg.typesize_bias * (n * n_block1() + v * simd_w_);
    }
    int scales_offset(int n, int v) {
        return sizeof(float) * brg.is_oc_scale * (n * n_block1() + v * simd_w_);
    }

    void generate() override;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
