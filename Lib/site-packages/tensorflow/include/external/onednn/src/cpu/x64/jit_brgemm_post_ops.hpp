/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_POST_OPS_HPP
#define CPU_X64_JIT_BRGEMM_POST_OPS_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_brgemm_primitive_conf.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct brgemm_kernel_diff_bias_t {
    brgemm_kernel_diff_bias_t()
        : ptr_diff_dst(nullptr)
        , ptr_diff_bias_acc(nullptr)
        , ptr_diff_bias(nullptr)
        , flags(0) {};

    void *ptr_diff_dst;
    void *ptr_diff_bias_acc;
    void *ptr_diff_bias;
    int flags;
};

#define GET_OFF(field) offsetof(brgemm_kernel_diff_bias_t, field)
template <typename Vmm>
struct jit_brgemm_kernel_diff_bias_t : public jit_generator {
    jit_brgemm_kernel_diff_bias_t(
            const jit_brgemm_primitive_conf_t &ajbgp, const brgemm_desc_t &abrg)
        : jit_generator(jit_name())
        , brg_(abrg)
        , ddst_dt_(ajbgp.dst_dt)
        , bia_dt_(ajbgp.bia_dt)
        , acc_dt_(ajbgp.acc_dt)
        , bia_typesize_(types::data_type_size(bia_dt_))
        , acc_typesize_(types::data_type_size(acc_dt_)) {

        ddst_dt_ = (ajbgp.isa == avx512_core_fp16 && ajbgp.use_buffer_b)
                ? data_type::f32
                : ajbgp.dst_dt;
        ddst_typesize_ = types::data_type_size(ddst_dt_);
        mult_ = data_type_vnni_granularity(ddst_dt_);
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_diff_bias_t)

private:
    brgemm_desc_t brg_;
    data_type_t ddst_dt_;
    data_type_t bia_dt_;
    data_type_t acc_dt_;

    int ddst_typesize_;
    int bia_typesize_;
    int acc_typesize_;
    int mult_;

    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;
    using reg64_t = const Xbyak::Reg64;
    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_ddst = r15;
    const reg64_t reg_bias = r14;
    const reg64_t reg_bias_acc = r13;
    const reg64_t aux_reg_ddst = r12;
    const reg64_t reg_k_iter = r11;
    const reg64_t reg_flag = r10;
    const reg64_t reg_mask = rax;

    Xbyak::Label f16_perm_table_;
    Xbyak::Label mask_label_;
    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);
    Xbyak::Opmask k_f16_perm_mask = Xbyak::Opmask(4);
    Vmm vreg_unit = Vmm(31);
    Vmm vreg_perm = Vmm(30);
    Vmm vmm_tail_mask = Vmm(15); // use for avx tail loads

    const int n_max_regs_ = 4;

    const Vmm vmm_mask(const Vmm vmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) {
        return mask_flag && isa_has_masks(brg_.isa_impl)
                ? (store ? vmm_in | ktail_mask : vmm_in | ktail_mask | T_z)
                : vmm_in;
    }

    Vmm get_bias_reg(int n) const { return Vmm(n); }
    Vmm_lower_t get_bias_reg_lower(int n) const { return Vmm_lower_t(n); }
    Vmm get_ddst_reg(int n) const { return Vmm(n + n_max_regs_); }

    void accumulate_bias(int idx, bool mask_flag) {
        auto vddst = get_ddst_reg(idx);
        auto vddst_load = vmm_mask(vddst, mask_flag, false, k_tail_mask);
        auto vbias = get_bias_reg(idx);
        if (ddst_dt_ == data_type::f16) {
            // As we do not have fp16_vnni, we add twice to accumulate
            // adjacent elements.
            for (int i = 0; i < 2; ++i) {
                auto addr = ptr[aux_reg_ddst
                        + ddst_typesize_ * mult_ * idx * brg_.ld_block + i * 2];
                vmovups(vddst_load, addr);
                vpermw(vddst | k_f16_perm_mask | T_z, vreg_perm, vddst);
                vcvtph2psx(vddst, Vmm_lower_t(vddst.getIdx()));
                vaddps(vbias, vbias, vddst);
            }
        } else {
            auto addr = ptr[aux_reg_ddst
                    + ddst_typesize_ * mult_ * idx * brg_.ld_block];
            if (IMPLICATION(mask_flag, isa_has_masks(brg_.isa_impl)))
                vmovups(vddst_load, addr);
            else
                vmaskmovps(vddst_load, vmm_tail_mask, addr);
            if (ddst_dt_ == data_type::bf16)
                vdpbf16ps(vbias, vreg_unit, vddst);
            else
                vaddps(vbias, vbias, vddst);
        }
    }

    void store(int idx, bool mask_flag) {
        auto addr = ptr[reg_bias + bia_typesize_ * idx * brg_.ld_block];
        auto vbias = get_bias_reg(idx);
        auto vbias_lower = get_bias_reg_lower(idx);
        switch (bia_dt_) {
            case data_type::bf16:
                vcvtneps2bf16(vbias_lower, vbias);
                if (mask_flag) {
                    vmovdqu16(addr,
                            vmm_mask(vbias, mask_flag, true, k_tail_mask));
                } else {
                    vmovups(addr, vbias_lower);
                }
                break;
            case data_type::f16:
                vcvtps2ph(vbias_lower, vbias, 0x4);
                if (mask_flag) {
                    vmovdqu16(addr,
                            vmm_mask(vbias, mask_flag, true, k_tail_mask));
                } else {
                    vmovups(addr, vbias_lower);
                }
                break;
            case data_type::f32:
                if (IMPLICATION(mask_flag, isa_has_masks(brg_.isa_impl)))
                    vmovups(addr,
                            vmm_mask(vbias, mask_flag, true, k_tail_mask));
                else
                    vmaskmovps(addr, vmm_tail_mask, vbias);
                break;
            default: assert("Unsupported bias data type");
        }
    }

    void loop_by_N(int n_loop, int nb_tail) {

        mov(aux_reg_ddst, reg_ddst);

        int n_iters = n_loop;
        if (nb_tail > 0) n_iters--;
        Xbyak::Label k_loop, init_zero, init_done;
        int n_ = 0;

        test(reg_flag, FLAG_REDUCE_FIRST);
        jnz(init_zero, T_NEAR); // FLAG_REDUCE_FIRST is set

        for (; n_ < n_iters; n_++) {
            auto vbias = get_bias_reg(n_);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            vmovups(vbias, addr);
        }
        if (nb_tail > 0) {
            auto vbias = vmm_mask(get_bias_reg(n_), true, false, k_tail_mask);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            if (isa_has_masks(brg_.isa_impl))
                vmovups(vbias, addr);
            else
                vmaskmovps(vbias, vmm_tail_mask, addr);
        }
        jmp(init_done, T_NEAR);
        L(init_zero);

        for (int n_ = 0; n_ < n_loop; n_++) {
            uni_vxorps(get_bias_reg(n_), get_bias_reg(n_), get_bias_reg(n_));
        }
        L(init_done);

        mov(reg_k_iter, utils::div_up(brg_.reduce_dim, mult_));
        L(k_loop);
        {
            int n_ = 0;
            for (; n_ < n_iters; n_++)
                accumulate_bias(n_, false);

            if (nb_tail > 0) accumulate_bias(n_, true);

            add(aux_reg_ddst, ddst_typesize_ * mult_ * brg_.LDB);

            sub(reg_k_iter, 1);
            jnz(k_loop, T_NEAR);
        }

        Xbyak::Label store_final, store_done;
        test(reg_flag, FLAG_REDUCE_LAST);
        jnz(store_final, T_NEAR); // FLAG_REDUCE_LAST is set

        n_ = 0;
        for (; n_ < n_iters; n_++) {
            auto vbias = get_bias_reg(n_);
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            vmovups(addr, vbias);
        }
        if (nb_tail > 0) {
            auto addr = ptr[reg_bias_acc + acc_typesize_ * n_ * brg_.ld_block];
            auto vbias = get_bias_reg(n_);
            if (isa_has_masks(brg_.isa_impl)) {
                vbias = vmm_mask(vbias, true, true, k_tail_mask);
                vmovups(addr, vbias);
            } else {
                vmaskmovps(addr, vmm_tail_mask, vbias);
            }
        }
        jmp(store_done, T_NEAR);

        L(store_final);
        n_ = 0;

        for (; n_ < n_iters; n_++)
            store(n_, false);

        if (nb_tail > 0) store(n_, true);

        L(store_done);
    }

    void init_masks(int tail_length) {
        if (ddst_dt_ == data_type::f16) {
            const auto half_mask = size_t((1 << 16) - 1);
            mov(reg_mask, half_mask);
            kmovq(k_f16_perm_mask, reg_mask);

            vmovups(vreg_perm | k_f16_perm_mask | T_z,
                    ptr[rip + f16_perm_table_]);
        }

        if (tail_length == 0) return;
        if (isa_has_masks(brg_.isa_impl)) {
            const auto full_mask = size_t {0xffffffffffffffff};
            const auto tail_mask = size_t((1 << tail_length) - 1);
            mov(reg_mask, full_mask);
            kmovq(k_full_mask, reg_mask);
            mov(reg_mask, tail_mask);
            kmovq(k_tail_mask, reg_mask);

        } else {
            vmovups(vmm_tail_mask, ptr[rip + mask_label_]);
        }
    }

    void generate() override {
        preamble();

        int nb = utils::div_up(brg_.load_dim, brg_.ld_block);
        int nb_tail = brg_.load_dim % brg_.ld_block;

        int n_loop = nb / n_max_regs_;
        int n_loop_tail = nb % n_max_regs_;
        if (n_loop_tail == 0 && nb_tail > 0) {
            n_loop--;
            n_loop_tail = n_max_regs_;
        }

        init_masks(nb_tail);

        if (ddst_dt_ == data_type::bf16) {
            auto reg_tmp = rax;
            auto reg_unit_val = reg_tmp.cvt16();
            mov(reg_unit_val, 0x3f80); // bf16 value of 1.
            vpbroadcastw(vreg_unit, reg_unit_val);
        }

        mov(reg_ddst, ptr[param1 + GET_OFF(ptr_diff_dst)]);
        mov(reg_bias_acc, ptr[param1 + GET_OFF(ptr_diff_bias_acc)]);
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_diff_bias)]);
        mov(reg_flag, ptr[param1 + GET_OFF(flags)]);

        for (int nb_ = 0; nb_ < n_loop; nb_++) {
            loop_by_N(n_max_regs_, 0);

            add(reg_ddst, ddst_typesize_ * mult_ * n_max_regs_ * brg_.ld_block);
            add(reg_bias, bia_typesize_ * n_max_regs_ * brg_.ld_block);
            add(reg_bias_acc, acc_typesize_ * n_max_regs_ * brg_.ld_block);
        }

        if (n_loop_tail > 0) loop_by_N(n_loop_tail, nb_tail);
        postamble();

        if (ddst_dt_ == data_type::f16) {
            // convert interleaved vnni data with holes to packed.
            const uint16_t f16_prm_array[16] = {
                    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
            align(64);
            L(f16_perm_table_);
            for (int i = 0; i < 16; ++i)
                dw(f16_prm_array[i]);
        }

        if (!isa_has_masks(brg_.isa_impl) && nb_tail > 0) {
            align(32);
            L(mask_label_);
            for (int i = 0; i < nb_tail; ++i)
                dd(~uint32_t(0));
            for (int i = nb_tail; i < brg_.ld_block; ++i)
                dd(0);
        }
    }
};

#undef GET_OFF

#define GET_OFF(field) offsetof(brgemm_kernel_post_ops_t, field)

struct brgemm_kernel_post_ops_t {
    void *ptr_in;
    void *ptr_out;
    void *ptr_bias;
    void *ptr_scales;
    const void *ptr_binary_post_ops_rhs;
    size_t apply_comp = 0;
    int32_t a_comp_val = 1;
    int32_t *a_zp_compensation;
    int32_t *c_zp_values;
    int32_t *s8s8_compensation;
    const void *dst_orig;
    void *ptr_dst_scales;
};

template <cpu_isa_t isa>
struct jit_brgemm_kernel_post_ops : public jit_generator {

    jit_brgemm_kernel_post_ops(const jit_brgemm_conv_conf_t &ajcp,
            const brgemm_desc_t &abrg, const primitive_attr_t &aattr)
        : jit_generator(jit_name(), abrg.isa_impl)
        , brg(abrg)
        , jcp(ajcp)
        , attr(aattr)
        , postops_injector_(nullptr)
        , with_binary_non_scalar_bcast_(brg.with_binary
                  && binary_injector::
                          any_binary_postop_rhs_non_scalar_broadcast(
                                  brg.attr()->post_ops_,
                                  memory_desc_wrapper(brg.dst_md()))) {

        if (brg.beta != 0) {
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(vmm_tmp(4).getIdx()), this->r14,
                    this->r15, this->r13, preserve_gpr, preserve_vmm,
                    GET_OFF(ptr_binary_post_ops_rhs), GET_OFF(dst_orig),
                    memory_desc_wrapper(brg.dst_md()),
                    static_cast<size_t>(brg.load_dim % brg.ld_block),
                    k_tail_mask, use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {this->param1, rhs_sp};

            const bool save_state = jcp.with_eltwise;
            const auto &reserved_eltwise_gpr = reg_reserved_eltwise;
            const auto reserved_eltwise_maskr = Xbyak::Opmask(1);

            const eltwise_injector::static_params_t esp {
                    save_state, reserved_eltwise_gpr, reserved_eltwise_maskr};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<po_isa_t>>(
                    this, attr.post_ops_, bsp, esp);
        }
        if (brg.is_bf16_emu)
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                    bf16_emu_scratch, bf16_emu_reserv_4, bf16_emu_reserv_4);

        const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
        // per_oc: conv: 1 << 0, (1 << 1) + (1 << 0) (with groups)
        // per_oc: ip: 1 << 0
        is_oc_scale_
                = utils::one_of(wei_scales.mask_, 1 << 0, (1 << 1) + (1 << 0));

        LDD_ = brg.LDD;
        inp_dt_ = brg.dt_c;
        out_dt_ = brg.dt_d;
        bia_dt_ = jcp.bia_dt;
        inp_typesize_ = types::data_type_size(inp_dt_);
        out_typesize_ = types::data_type_size(out_dt_);
        bia_typesize_ = (jcp.with_bias) ? types::data_type_size(bia_dt_) : 0;
    }

    ~jit_brgemm_kernel_post_ops() = default;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_post_ops)

    brgemm_desc_t brg;
    jit_brgemm_conv_conf_t jcp;
    const primitive_attr_t &attr;

private:
    int LDD_;

    data_type_t inp_dt_;
    data_type_t out_dt_;
    data_type_t bia_dt_;
    static constexpr cpu_isa_t po_isa_t = utils::map(isa, avx512_core, avx2,
            avx2, avx2_vnni, avx2, avx2_vnni_2, avx2_vnni_2, avx512_core_fp16,
            avx512_core_fp16);
    std::unique_ptr<injector::jit_uni_postops_injector_t<po_isa_t>>
            postops_injector_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    const bool with_binary_non_scalar_bcast_;

    int inp_typesize_;
    int out_typesize_;
    int bia_typesize_;

    int is_oc_scale_;
    constexpr static int max_vregs_ = cpu_isa_traits<po_isa_t>::n_vregs;

    using reg64_t = const Xbyak::Reg64;
    using Vmm = typename utils::conditional<utils::one_of(isa, avx2, avx2_vnni,
                                                    avx2_vnni_2),
            Xbyak::Ymm, Xbyak::Zmm>::type;
    using Vmm_lower_t = typename vreg_traits<Vmm>::Vmm_lower_t;

    // Register decomposition
    const reg64_t reg_reserved_eltwise = rax;
    const reg64_t param1 = abi_param1;
    const reg64_t reg_in = r15;
    const reg64_t reg_out = r14;
    const reg64_t aux_reg_in = r13;
    const reg64_t aux_reg_out = r12;

    const reg64_t reg_bias = r11;
    const reg64_t aux_reg_bias = r10;

    const reg64_t reg_scales = r9;
    const reg64_t aux_reg_scales = r8;

    const reg64_t reg_ptr_sum_scale = rdx;
    const reg64_t reg_ptr_sum_zp = rsi;

    const reg64_t reg_zp_c_values = rbx;
    const reg64_t aux_reg_zp_c_values = rbx;
    const reg64_t reg_zp_a_comp = rbx;
    const reg64_t aux_reg_zp_a_comp = rbx;
    const reg64_t reg_s8s8_comp = rbx;
    const reg64_t aux_reg_s8s8_comp = rbx;
    const reg64_t reg_zp_a_val = rbx;
    const reg64_t reg_apply_comp = rbx;
    const reg64_t reg_dst_scales = rbx;
    const reg64_t aux_reg_dst_scales = rbx;
    const reg64_t reg_tmp = abi_not_param1;

    constexpr static int reg_zp_c_values_offs_ = 0;
    constexpr static int aux_reg_zp_c_values_offs_ = 8;
    constexpr static int reg_zp_a_comp_offs_ = 16;
    constexpr static int aux_reg_zp_a_comp_offs_ = 24;
    constexpr static int reg_s8s8_comp_offs_ = 32;
    constexpr static int aux_reg_s8s8_comp_offs_ = 40;
    constexpr static int reg_zp_a_val_offs_ = 48;
    constexpr static int reg_apply_comp_offs_ = 56;
    constexpr static int reg_dst_scales_offs_ = 64;
    constexpr static int stack_space_needed_ = 72;

    /* bf16 emulation */
    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(24);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(25);
    Xbyak::Zmm bf16_emu_reserv_4 = Xbyak::Zmm(26);
    reg64_t bf16_emu_scratch = reg_tmp;

    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);

    const int n_block2_ = 4;

    Vmm vmm_tmp(int i) const { return Vmm(max_vregs_ - 1 - i); }

    int zp_c_values_offset(int n, bool is_tail = false) const noexcept {
        if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
            return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                             : sizeof(int32_t) * n * brg.ld_block;
        }

        return 0;
    }
    int zp_comp_a_vpad_offset(
            int n, int m, bool is_tail = false) const noexcept {
        return (is_tail) ? sizeof(int32_t) * (brg.ldb_tail + m * brg.LDB)
                         : sizeof(int32_t) * (n * brg.ld_block + m * brg.LDB);
    }
    int mb_zp_comp_a_offset(int m_block) const noexcept {
        return sizeof(int32_t) * m_block * brg.LDB;
    }
    int compensation_vpad_offset(
            int n, int m, bool is_tail = false) const noexcept {
        return (is_tail) ? sizeof(int32_t) * (brg.ldb_tail + m * brg.LDB)
                         : sizeof(int32_t) * (n * brg.ld_block + m * brg.LDB);
    }
    int mb_compensation_offset(int m_block) const noexcept {
        return sizeof(int32_t) * m_block * brg.LDB;
    }

    template <typename T>
    const T maybe_mask(const T vmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) {
        assert(IMPLICATION(mask_flag, isa_has_masks(isa)));
        return mask_flag
                ? (store ? vmm_in | ktail_mask : vmm_in | ktail_mask | T_z)
                : vmm_in;
    }

    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
            int tail_size, bool store, Xbyak::Opmask ktail_mask,
            bool skip_cvt2ps = false) {
        const bool is_tail = op.isMEM()
                && tail_size != vreg_traits<Vmm>::vlen / sizeof(float)
                // The current kernel is written such that tail_size = 0 implies
                // no tail and full vmm must be processed.
                && tail_size > 0;

        if (IMPLICATION(is_tail, isa_has_masks(isa))) {
            const Vmm vmm = maybe_mask(vmm_in, is_tail, store, ktail_mask);
            switch (type_in) {
                case data_type::f32:
                case data_type::s32: vmovups(vmm, op); break;
                case data_type::s8: vpmovsxbd(vmm, op); break;
                case data_type::u8: vpmovzxbd(vmm, op); break;
                case data_type::bf16:
                    vpmovzxwd(vmm, op);
                    vpslld(vmm, vmm, 16);
                    break;
                case data_type::f16: vcvtph2ps(vmm, op); break;
                default: assert(!"unsupported data type");
            }
        } else {
            load_data(type_in, vmm_in, op.getAddress(), tail_size);
        }
        if (!skip_cvt2ps && types::is_integral_dt(type_in))
            uni_vcvtdq2ps(vmm_in, vmm_in);
    }

    Vmm vector(int m, int n, int n_block) { return Vmm(m * n_block + n); };

    void inject_attr_postops(int m_block, int n_block, int tail = 0) {
        const auto &p = attr.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const auto k_mask = tail == 0 ? k_full_mask : k_tail_mask;
        const auto sum_dt = p.get_sum_dt(out_dt_);

        const auto sum_injector = [&] {
            const float *p_sum_scale = &p.entry_[sum_idx].sum.scale;
            const int32_t *p_sum_zp = &p.entry_[sum_idx].sum.zero_point;
            if (*p_sum_scale != 1.f)
                mov(reg_ptr_sum_scale, (size_t)p_sum_scale);
            auto vmm_sum_zp = vmm_tmp(1);
            if (*p_sum_zp != 0) {
                mov(reg_ptr_sum_zp, (size_t)p_sum_zp);
                if (is_superset(isa, avx512_core)) {
                    vcvtdq2ps(vmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
                } else {
                    vpbroadcastd(vmm_sum_zp, ptr[reg_ptr_sum_zp]);
                    uni_vcvtdq2ps(vmm_sum_zp, vmm_sum_zp);
                }
            }

            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto vmm = vector(m, n, n_block);
                const auto addr = ptr[aux_reg_out
                        + out_typesize_ * (m * LDD_ + n * brg.ld_block)];

                const auto vmm_prev_dst = vmm_tmp(0);
                cvt2ps(sum_dt, vmm_prev_dst, addr, tail, false, k_mask);
                if (*p_sum_zp != 0)
                    uni_vsubps(vmm_prev_dst, vmm_prev_dst, vmm_sum_zp);
                if (*p_sum_scale == 1.f)
                    uni_vaddps(vmm, vmm, vmm_prev_dst);
                else {
                    if (is_superset(isa, avx512_core)) {
                        vfmadd231ps(
                                vmm, vmm_prev_dst, ptr_b[reg_ptr_sum_scale]);
                    } else {
                        auto vmm_sum_scale = vmm_tmp(2);
                        vpbroadcastd(vmm_sum_scale, ptr[reg_ptr_sum_scale]);
                        vfmadd231ps(vmm, vmm_prev_dst, vmm_sum_scale);
                    }
                }
            }
        };

        if (jcp.with_sum) {
            postops_injector_->set_lambda_injector(
                    primitive_kind::sum, sum_injector);
        }

        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

        if (with_binary_non_scalar_bcast_) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto vmm_idx = vector(m, n, n_block).getIdx();
                const size_t aux_output_offset
                        = out_typesize_ * (m * LDD_ + n * brg.ld_block);

                rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, aux_reg_out);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, aux_output_offset);
                if (tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }

        postops_injector_->compute_vector_range(
                0, m_block * n_block, rhs_arg_params);
    }
    void apply_comp(int m_block, int n_block, int tail = 0) {
        auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
        const bool has_tail = tail > 0;
        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            auto vmm_zp_a_val = vmm_tmp(1);
            mov(reg_zp_a_val, ptr[rsp + reg_zp_a_val_offs_]);
            uni_vpbroadcastd(vmm_zp_a_val, reg_zp_a_val.cvt32());

            mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
            const auto vmm_zp_comp_a = vmm_tmp(0);
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const size_t zp_comp_offset
                        = sizeof(int32_t) * (n * brg.ld_block + m * brg.LDB);
                auto zp_comp_a_addr = is_superset(isa, avx512_core)
                        ? EVEX_compress_addr(aux_reg_zp_a_comp, zp_comp_offset)
                        : ptr[aux_reg_zp_a_comp + zp_comp_offset];
                if (IMPLICATION(has_tail, isa_has_masks(isa))) {
                    auto vmm_zp_comp_a_masked = maybe_mask(
                            vmm_zp_comp_a, has_tail, false, k_mask);
                    vmovups(vmm_zp_comp_a_masked, zp_comp_a_addr);
                } else {
                    load_data(data_type::s32, vmm_zp_comp_a, zp_comp_a_addr,
                            tail);
                }
                uni_vpmulld(vmm_zp_comp_a, vmm_zp_a_val, zp_comp_a_addr);

                auto vmm = vector(m, n, n_block);
                uni_vpaddd(vmm, vmm, vmm_zp_comp_a);
            }
        }

        if (brg.req_s8s8_compensation) {
            mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
            const auto vmm_comp = vmm_tmp(0);
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const size_t s8s8_comp_offset
                        = sizeof(int32_t) * (n * brg.ld_block + m * brg.LDB);

                auto comp_addr = is_superset(isa, avx512_core)
                        ? EVEX_compress_addr(
                                aux_reg_s8s8_comp, s8s8_comp_offset)
                        : ptr[aux_reg_s8s8_comp + s8s8_comp_offset];
                if (IMPLICATION(tail > 0, isa_has_masks(isa))) {
                    auto vmm_comp_masked
                            = maybe_mask(vmm_comp, tail > 0, false, k_mask);
                    vmovups(vmm_comp_masked, comp_addr);
                } else
                    load_data(data_type::s32, vmm_comp, comp_addr, tail);

                auto vmm = vector(m, n, n_block);
                uni_vpaddd(vmm, vmm, vmm_comp);
            }
        }
    }
    void maybe_apply_comp(int m_block, int n_block, int tail = 0) {
        Xbyak::Label label_apply_without_comp;
        mov(reg_apply_comp, ptr[rsp + reg_apply_comp_offs_]);
        cmp(reg_apply_comp, 0);
        je(label_apply_without_comp, T_NEAR);
        apply_comp(m_block, n_block, tail);
        L_aligned(label_apply_without_comp);

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            uni_vcvtdq2ps(vector(m, n, n_block), vector(m, n, n_block));
        }
    }

    void apply_post_ops(int m_block, int n_block, int tail = 0) {
        const auto vector = [=](int m, int n) { return Vmm(m * n_block + n); };
        auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
        const auto req_comp = brg.is_int8 && brg.beta != 0
                && (brg.req_s8s8_compensation
                        || brg.zp_type_a != brgemm_broadcast_t::none);

        // brg.alpha == 0 means initialize registers, 1 means read from input
        // brg.beta == 0 means skip postwork, 1 means do postwork
        // req_comp == true -> convert accumulated values to f32 after applying
        // compensation to avoid the loss of accuracy when converting s32 to f32
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            if (brg.alpha == 0 && brg.beta != 0) {
                // if postwork then have to init vmm each time
                uni_vpxor(vector(m, n), vector(m, n), vector(m, n));
            } else if (brg.alpha != 0) {
                auto inp_addr = ptr[aux_reg_in
                        + inp_typesize_ * (m * brg.LDC + n * brg.ld_block)];
                cvt2ps(inp_dt_, vector(m, n), inp_addr, tail, false, k_mask,
                        req_comp);
            }
        }

        if (req_comp) maybe_apply_comp(m_block, n_block, tail);

        if (brg.beta != 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                const auto addr = ptr[aux_reg_scales
                        + is_oc_scale_ * sizeof(float) * (n * brg.ld_block)];
                auto vmm = vector(m, n);
                if (IMPLICATION(tail > 0, isa_has_masks(isa))) {
                    vmm = maybe_mask(vector(m, n), tail > 0, false, k_mask);
                    vmulps(vmm, vmm, addr);
                } else {
                    auto vmm_scales = vmm_tmp(0);
                    load_data(data_type::f32, vmm_scales, addr, tail);
                    vmulps(vmm, vmm, vmm_scales);
                }
            }
        }

        if (brg.beta != 0 && jcp.with_bias) {
            for (int n = 0; n < n_block; n++) {
                auto vmm_bias = vmm_tmp(0);
                auto bias_addr = ptr[aux_reg_bias
                        + bia_typesize_ * (n * brg.ld_block)];
                cvt2ps(bia_dt_, vmm_bias, bias_addr, tail, false, k_mask);
                for (int m = 0; m < m_block; m++) {
                    vaddps(vector(m, n), vmm_bias);
                }
            }
        }

        if (postops_injector_) inject_attr_postops(m_block, n_block, tail);

        if (brg.beta != 0 && brg.with_dst_scales) {
            mov(aux_reg_dst_scales, ptr[rsp + reg_dst_scales_offs_]);
            const auto addr = ptr[aux_reg_dst_scales];
            auto vmm_scales = vmm_tmp(0);
            if (!isa_has_masks(isa)) vmovups(vmm_scales, addr);

            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto vmm = vector(m, n);
                if (isa_has_masks(isa)) {
                    vmm = maybe_mask(vector(m, n), tail > 0, false, k_mask);
                    vmulps(vmm, vmm, addr);
                } else {
                    vmulps(vmm, vmm, vmm_scales);
                }
            }
        }

        if (brg.beta != 0 && brg.zp_type_c != brgemm_broadcast_t::none) {
            mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
            auto vmm_zp_c = vmm_tmp(0);
            if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
                if (is_superset(isa, avx512_core))
                    vcvtdq2ps(vmm_zp_c,
                            EVEX_compress_addr(aux_reg_zp_c_values, 0, true));
                else {
                    uni_vbroadcastss(vmm_zp_c, ptr[aux_reg_zp_c_values]);
                    uni_vcvtdq2ps(vmm_zp_c, vmm_zp_c);
                }
            }
            for (int n = 0; n < n_block; n++) {
                if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
                    int zp_c_off = zp_c_values_offset(n);
                    auto zp_c_addr = is_superset(isa, avx512_core)
                            ? EVEX_compress_addr(aux_reg_zp_c_values, zp_c_off)
                            : ptr[aux_reg_zp_c_values + zp_c_off];
                    cvt2ps(data_type::s32, vmm_zp_c, zp_c_addr, tail, false,
                            k_mask);
                }
                for (int m = 0; m < m_block; m++) {
                    const auto vmm = vector(m, n);
                    uni_vaddps(vmm, vmm, vmm_zp_c);
                }
            }
        }

        const bool dt_requires_saturation = types::is_integral_dt(out_dt_);

        const reg64_t reg_tmp_gpr = reg_tmp;
        auto vmm_lbound = vmm_tmp(0);
        auto vmm_ubound = vmm_tmp(1);
        if (dt_requires_saturation) {
            init_saturate_f32(vmm_lbound, vmm_ubound, reg_tmp_gpr,
                    data_type::f32, out_dt_);
        }

        if (brg.is_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            // incase of tail, stores are unconditionally masked, regardless
            // of `n`, implying n_block must be equal to `1`.
            assert(IMPLICATION(tail > 0, n_block == 1));
            auto vmm = vector(m, n);
            const size_t offset = out_typesize_ * (m * LDD_ + n * brg.ld_block);
            const auto addr = ptr[aux_reg_out + offset];

            if (dt_requires_saturation) {
                saturate_cvt_f32(vmm, vmm_lbound, vmm_ubound, out_dt_);
            }

            if (is_superset(isa, avx512_core)) {
                auto vmm_masked = maybe_mask(vmm, tail > 0, true, k_mask);
                Vmm_lower_t vmm_low = Vmm_lower_t(vmm.getIdx());
                auto vmm_low_masked
                        = maybe_mask(vmm_low, tail > 0, true, k_mask);
                switch (out_dt_) {
                    case data_type::f32:
                    case data_type::s32: uni_vmovups(addr, vmm_masked); break;
                    case data_type::bf16:
                        if (brg.is_bf16_emu) {
                            bf16_emu_->vcvtneps2bf16(vmm_low, vmm);
                            vmovdqu16(addr, vmm_low_masked);
                        } else {
                            vcvtneps2bf16(vmm_low, vmm);
                            vmovdqu16(addr, vmm_low_masked);
                        }
                        break;
                    case data_type::f16:
                        vcvtps2ph(vmm_low, vmm, _op_mxcsr);
                        vmovdqu16(addr, vmm_low_masked);
                        break;
                    case data_type::s8: vpmovsdb(addr, vmm_masked); break;
                    case data_type::u8: vpmovusdb(addr, vmm_masked); break;
                    default: assert(!"unknown dst_dt");
                }
            } else {
                const int simd_w = vreg_traits<Vmm>::vlen / sizeof(float);
                const int nelems = tail > 0 ? tail : simd_w;
                store_data(out_dt_, vmm, aux_reg_out, offset, nelems);
            }
        }
    }

    void loop_by_N(int m_block, int nb2, int nb2_tail, int nb_tail) {

        if (brg.alpha) { mov(aux_reg_in, reg_in); }
        if (brg.beta != 0) {
            if (jcp.with_bias) mov(aux_reg_bias, reg_bias);
            if (brg.zp_type_c != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_c_values, ptr[rsp + reg_zp_c_values_offs_]);
                mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
            }
            if (brg.zp_type_a != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_a_comp, ptr[rsp + reg_zp_a_comp_offs_]);
                mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
            }
            if (brg.req_s8s8_compensation) {
                mov(aux_reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
                mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
            }
            mov(aux_reg_scales, reg_scales);
        }
        mov(aux_reg_out, reg_out);

        for (int n_loop_ = 0; n_loop_ < nb2; n_loop_++) {
            apply_post_ops(m_block, n_block2_);

            const auto oc_l_offset = n_block2_ * brg.ld_block;

            add(aux_reg_out, out_typesize_ * oc_l_offset);
            if (brg.alpha != 0) {
                add(aux_reg_in, inp_typesize_ * oc_l_offset);
            }
            if (brg.beta != 0) {
                if (jcp.with_bias)
                    add(aux_reg_bias, bia_typesize_ * oc_l_offset);
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_c_values,
                            ptr[rsp + aux_reg_zp_c_values_offs_]);
                    add(aux_reg_zp_c_values, zp_c_values_offset(n_block2_));
                    mov(ptr[rsp + aux_reg_zp_c_values_offs_],
                            aux_reg_zp_c_values);
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                    add(aux_reg_zp_a_comp, sizeof(int32_t) * oc_l_offset);
                    mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
                }
                if (brg.req_s8s8_compensation) {
                    mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                    add(aux_reg_s8s8_comp, sizeof(int32_t) * oc_l_offset);
                    mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
                }

                add(aux_reg_scales, is_oc_scale_ * sizeof(float) * oc_l_offset);
            }
        }
        if (nb2_tail > 0) {
            apply_post_ops(m_block, nb2_tail);
            const auto oc_l_offset = nb2_tail * brg.ld_block;

            add(aux_reg_out, out_typesize_ * oc_l_offset);
            if (brg.alpha != 0) {
                add(aux_reg_in, inp_typesize_ * oc_l_offset);
            }
            if (brg.beta != 0) {
                if (jcp.with_bias)
                    add(aux_reg_bias, bia_typesize_ * oc_l_offset);
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_c_values,
                            ptr[rsp + aux_reg_zp_c_values_offs_]);
                    add(aux_reg_zp_c_values, zp_c_values_offset(nb2_tail));
                    mov(ptr[rsp + aux_reg_zp_c_values_offs_],
                            aux_reg_zp_c_values);
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                    add(aux_reg_zp_a_comp, sizeof(int32_t) * oc_l_offset);
                    mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
                }
                if (brg.req_s8s8_compensation) {
                    mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                    add(aux_reg_s8s8_comp, sizeof(int32_t) * oc_l_offset);
                    mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
                }

                add(aux_reg_scales, is_oc_scale_ * sizeof(float) * oc_l_offset);
            }
        }
        if (nb_tail > 0) {
            apply_post_ops(m_block, 1, nb_tail);

            if (brg.alpha != 0) { add(aux_reg_in, inp_typesize_ * (nb_tail)); }
            if (brg.beta != 0) {
                if (jcp.with_bias) add(aux_reg_bias, bia_typesize_ * (nb_tail));
                if (brg.zp_type_c != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_c_values,
                            ptr[rsp + aux_reg_zp_c_values_offs_]);
                    add(aux_reg_zp_c_values, zp_c_values_offset(1, nb_tail));
                    mov(ptr[rsp + aux_reg_zp_c_values_offs_],
                            aux_reg_zp_c_values);
                }
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                    add(aux_reg_zp_a_comp, sizeof(int32_t) * nb_tail);
                    mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
                }
                if (brg.req_s8s8_compensation) {
                    mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                    add(aux_reg_s8s8_comp, sizeof(int32_t) * nb_tail);
                    mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
                }
                add(aux_reg_scales, is_oc_scale_ * bia_typesize_ * (nb_tail));
            }
            add(aux_reg_out, out_typesize_ * (nb_tail));
        }
    }

    void generate() override {
        preamble();

        sub(rsp, stack_space_needed_);

        int nb = brg.load_dim / brg.ld_block;
        int nb_tail = brg.load_dim % brg.ld_block;

        int nb2 = nb / n_block2_;
        int nb2_tail = nb % n_block2_;
        int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : n_block2_;

        int m_max_regs = (brg.is_bf16_emu ? 24 : max_vregs_ - 4) / n_block;
        int m_block = nstl::min(brg.bcast_dim, m_max_regs);

        int mb = brg.bcast_dim / m_block;
        int mb_tail = brg.bcast_dim % m_block;

        if (isa_has_masks(isa)) {
            const auto full_mask = size_t {0xffffffffffffffff};
            const auto tail_mask = size_t((1 << nb_tail) - 1);

            reg64_t reg_mask = reg_tmp;
            mov(reg_mask, full_mask);
            kmovq(k_full_mask, reg_mask);
            mov(reg_mask, tail_mask);
            kmovq(k_tail_mask, reg_mask);
        }

        if (brg.alpha != 0) { mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]); }
        if (brg.beta != 0) {
            mov(reg_scales, ptr[param1 + GET_OFF(ptr_scales)]);
            mov(reg_apply_comp, ptr[param1 + GET_OFF(apply_comp)]);
            mov(ptr[rsp + reg_apply_comp_offs_], reg_apply_comp);

            if (jcp.with_bias) mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
            if (brg.zp_type_c != brgemm_broadcast_t::none) {
                mov(reg_zp_c_values, ptr[param1 + GET_OFF(c_zp_values)]);
                mov(ptr[rsp + reg_zp_c_values_offs_], reg_zp_c_values);
            }
            if (brg.zp_type_a != brgemm_broadcast_t::none) {
                mov(reg_zp_a_comp, ptr[param1 + GET_OFF(a_zp_compensation)]);
                mov(ptr[rsp + reg_zp_a_comp_offs_], reg_zp_a_comp);

                mov(reg_zp_a_val, ptr[param1 + GET_OFF(a_comp_val)]);
                mov(ptr[rsp + reg_zp_a_val_offs_], reg_zp_a_val);
            }
            if (brg.req_s8s8_compensation) {
                mov(reg_s8s8_comp, ptr[param1 + GET_OFF(s8s8_compensation)]);
                mov(ptr[rsp + reg_s8s8_comp_offs_], reg_s8s8_comp);
            }
            if (brg.with_dst_scales) {
                mov(reg_dst_scales, ptr[param1 + GET_OFF(ptr_dst_scales)]);
                mov(ptr[rsp + reg_dst_scales_offs_], reg_dst_scales);
            }
        }
        mov(reg_out, ptr[param1 + GET_OFF(ptr_out)]);

        // brg.alpha == 0 means initialize registers, 1 means read from input
        // brg.beta == 0 means skip postwork, 1 means do postwork
        if (brg.alpha == 0 && brg.beta == 0) {
            for_(int m = 0; m < m_block; m++)
            for (int n = 0; n < n_block; n++) {
                auto vmm = Vmm(m * n_block + n);
                uni_vpxor(vmm, vmm, vmm);
            }
        }

        for (int mb_ = 0; mb_ < mb; mb_++) {
            loop_by_N(m_block, nb2, nb2_tail, nb_tail);

            if (brg.alpha != 0)
                add(reg_in, inp_typesize_ * (m_block * brg.LDC));
            if (brg.beta != 0) {
                if (brg.zp_type_a != brgemm_broadcast_t::none) {
                    mov(reg_zp_a_comp, ptr[rsp + reg_zp_a_comp_offs_]);
                    add(reg_zp_a_comp, mb_zp_comp_a_offset(m_block));
                    mov(ptr[rsp + reg_zp_a_comp_offs_], reg_zp_a_comp);
                }
                if (brg.req_s8s8_compensation) {
                    mov(reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
                    add(reg_s8s8_comp, mb_compensation_offset(m_block));
                    mov(ptr[rsp + reg_s8s8_comp_offs_], reg_s8s8_comp);
                }
            }
            add(reg_out, out_typesize_ * (m_block * LDD_));
        }
        if (mb_tail > 0) loop_by_N(mb_tail, nb2, nb2_tail, nb_tail);

        add(rsp, stack_space_needed_);

        postamble();

        if (postops_injector_)
            postops_injector_->prepare_table(/* generate = */ true);
    }
};

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
