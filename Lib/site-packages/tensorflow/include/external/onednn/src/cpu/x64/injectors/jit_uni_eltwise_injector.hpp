/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_ELTWISE_INJECTOR_HPP
#define CPU_X64_JIT_UNI_ELTWISE_INJECTOR_HPP

#include <assert.h>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace eltwise_injector {
struct static_params_t {

    static_params_t(bool save_state = true,
            Xbyak::Reg64 p_table = Xbyak::Reg64(Xbyak::Operand::RAX),
            Xbyak::Opmask k_mask = Xbyak::Opmask(1), bool is_fwd = true,
            bool use_dst = false, bool preserve_vmm = true,
            bool preserve_p_table = true)
        : save_state(save_state)
        , p_table_(p_table)
        , k_mask_(k_mask)
        , is_fwd(is_fwd)
        , use_dst(use_dst)
        , preserve_vmm(preserve_vmm)
        , preserve_p_table(preserve_p_table) {}

    bool save_state;
    Xbyak::Reg64 p_table_;
    Xbyak::Opmask k_mask_;
    bool is_fwd;
    bool use_dst;
    bool preserve_vmm;
    bool preserve_p_table;
};

/*
 * Checks if isa is supported by eltwise injector.
 */
bool is_isa_supported(cpu_isa_t isa);

/*
 * Checks if eltwise algorithm is supported by eltwise injector.
 */
bool is_alg_supported(alg_kind_t alg);

/*
 * Checks if eltwise injection for given args is supported.
 */
bool is_supported(cpu_isa_t isa, alg_kind_t alg);

} // namespace eltwise_injector

template <cpu_isa_t isa, typename Wmm = typename cpu_isa_traits<isa>::Vmm>
struct jit_uni_eltwise_injector_f32 {
    using Vmm = Wmm;

    // Arguments description:
    // host - jit generator which is filled with instructions
    // alg, alpha, beta, scale - user eltwise arguments
    // save_state - when true, preserves on stack vmm_aux registers preventing
    //   results spoiling. Restores them when done in injector_postamble().
    // p_table - GPR where table label is stored to get access for pre-defined
    //   constants used in alg codes.
    // k_mask - k_register to operate with masks in alg codes.
    // is_fwd - when true, computes d = alg(s), otherwise, computes ds = alg'(s)
    //   - algorithm derivative.
    // use_dst - defines whether source or destination point is passed to alg
    //   code. Depends on algorithm. See `_use_dst_for_bwd` algs definition.
    jit_uni_eltwise_injector_f32(jit_generator *host, alg_kind_t alg,
            float alpha, float beta, float scale, bool save_state = true,
            Xbyak::Reg64 p_table = Xbyak::Reg64(Xbyak::Operand::RAX),
            Xbyak::Opmask k_mask = Xbyak::Opmask(1), bool is_fwd = true,
            bool use_dst = false, bool preserve_vmm = true,
            bool preserve_p_table = true)
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , h(host)
        , save_state_(save_state)
        , p_table_(p_table)
        , k_mask_(k_mask)
        , is_fwd_(is_fwd)
        , use_dst_(use_dst)
        , preserve_vmm_(preserve_vmm)
        , preserve_p_table_(preserve_p_table)
        , n_vregs_to_preserve_(aux_vecs_count(alg_, is_fwd_, alpha_)) {
        assert(eltwise_injector::is_supported(isa, alg_));

        register_table_entries();
    }

    jit_uni_eltwise_injector_f32(jit_generator *host,
            const post_ops_t::entry_t::eltwise_t &eltwise,
            bool save_state = true,
            Xbyak::Reg64 p_table = Xbyak::Reg64(Xbyak::Operand::RAX),
            Xbyak::Opmask k_mask = Xbyak::Opmask(1), bool is_fwd = true,
            bool use_dst = false, bool preserve_vmm = true,
            bool preserve_p_table = true)
        : jit_uni_eltwise_injector_f32(host, eltwise.alg, eltwise.alpha,
                eltwise.beta, eltwise.scale, save_state, p_table, k_mask,
                is_fwd, use_dst, preserve_vmm, preserve_p_table) {}

    void compute_vector_range(size_t start_compute_idx, size_t end_compute_idx,
            const injector_utils::vmm_index_set_t &vmm_aux_indices = {});
    void compute_vector_range(
            const injector_utils::vmm_index_set_t &vmm_compute_idxs,
            const injector_utils::vmm_index_set_t &vmm_aux_indices = {});
    void compute_vector(size_t compute_idx,
            const injector_utils::vmm_index_set_t &vmm_aux_indices = {}) {
        compute_vector_range({compute_idx}, vmm_aux_indices);
    }
    void prepare_table(bool gen_table = true);
    void load_table_addr() { h->lea(p_table_, h->ptr[h->rip + l_table_]); }

    // This call is `static` and `public` to make a decision on the injector's
    // saving state if the caller can supply the necessary number of vmms. The
    // decision must be made BEFORE constructing the injector, thus, `static`.
    static size_t aux_vecs_count(alg_kind_t alg, bool is_fwd, float alpha);

private:
    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;

    jit_generator *const h;

    const bool save_state_;
    const Xbyak::Reg64 p_table_;
    Xbyak::Reg64 reg_vmm_stack_ptr_;
    const Xbyak::Opmask k_mask_;
    const bool is_fwd_;
    const bool use_dst_;
    const bool preserve_vmm_;
    const bool preserve_p_table_;

    Xbyak::Label l_table_;

    // if only the injector was inherited from jit_generator...
    enum {
        _cmp_eq_oq = jit_generator::_cmp_eq_oq,
        _cmp_neq_uq = jit_generator::_cmp_neq_uq,
        _cmp_lt_os = jit_generator::_cmp_lt_os,
        _cmp_le_os = jit_generator::_cmp_le_os,
        _cmp_ge_os = jit_generator::_cmp_nlt_us,
        _cmp_gt_os = jit_generator::_cmp_nle_us,
        _op_floor = jit_generator::_op_floor,
        _op_mxcsr = jit_generator::_op_mxcsr
    };

    const bool is_avx512_ = is_superset(isa, avx512_core);

    static constexpr size_t vlen_ = vreg_traits<Vmm>::vlen;
    static constexpr size_t preserved_vecs_max_ = 6;
    static constexpr size_t preserved_gprs_max_ = 5;
    static constexpr size_t n_vregs_ = cpu_isa_traits<isa>::n_vregs;
    static constexpr int n_mantissa_bits_ = 23;

    const size_t n_vregs_to_preserve_;
    size_t n_vregs_preserved_ = 0;
    bool need_vmm_mask_register_ = false;
    // Default initialization will put zeros. Putting any value to trigger a
    // potential error to Xbyak is not working as Xbyak cycles vmm indices
    // over 32 value.
    size_t preserved_vmm_indices_[preserved_vecs_max_] = {};
    size_t preserved_vmm_tail_indices_[preserved_vecs_max_] = {};
    size_t preserved_gpr_indices_[preserved_gprs_max_] = {};

    Vmm vmm_mask_;
    Vmm vmm_tmp_;
    Xbyak::Ymm ymm_tmp_;
    Xbyak::Xmm xmm_tmp_;

    static bool need_mask_register(alg_kind_t alg, bool is_fwd, float alpha);
    static size_t aux_gprs_count(alg_kind_t alg, bool is_fwd, float alpha);
    static bool need_vmm_stack_ptr(alg_kind_t alg, bool is_fwd, float alpha);
    static size_t op_vecs_count(alg_kind_t alg, bool is_fwd);
    size_t get_stack_vmm_space();

    void compute_body(
            const injector_utils::vmm_index_set_iterator_t &start_idx_it,
            const injector_utils::vmm_index_set_iterator_t &end_idx_it);
    void injector_preamble(const injector_utils::vmm_index_set_t &vmm_idxs,
            injector_utils::vmm_index_set_iterator_t &start_idx_tail_it,
            const injector_utils::vmm_index_set_t &vmm_aux_indices);
    void injector_preamble_tail(size_t n_vregs_not_preserved);
    void injector_postamble();
    void assign_regs();
    Wmm vmm_aux(size_t idx);
    void vec_shift(const Vmm &vmm_dst, const Vmm &vmm_src, bool shift_left,
            const int imm);
    void compute_cmp_mask(const Vmm &vmm_src,
            const Xbyak::Operand &compare_operand, int cmp_predicate);
    void blend_with_mask(const Vmm &vmm_dst, const Xbyak::Operand &src);
    void test_mask();

    void exp_compute_vector_fwd(const Vmm &vmm_src);
    void relu_compute_vector_fwd(const Vmm &vmm_src);
    void relu_zero_ns_compute_vector_fwd(const Vmm &vmm_src);
    void elu_compute_vector_fwd(const Vmm &vmm_src);
    void tanh_compute_vector_fwd(const Vmm &vmm_src);
    void square_compute_vector_fwd(const Vmm &vmm_src);
    void abs_compute_vector_fwd(const Vmm &vmm_src);
    void sqrt_compute_vector_fwd(const Vmm &vmm_src);
    void linear_compute_vector_fwd(const Vmm &vmm_src);
    void soft_relu_compute_vector_fwd(const Vmm &vmm_src);
    void mish_compute_vector_fwd(const Vmm &vmm_src);
    void logistic_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_tanh_compute_vector_fwd(const Vmm &vmm_src);
    void swish_compute_vector_fwd(const Vmm &vmm_src);
    void log_compute_vector_fwd(const Vmm &vmm_src);
    void clip_compute_vector_fwd(const Vmm &vmm_src);
    void pow_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_erf_compute_vector_fwd(const Vmm &vmm_src);
    void gelu_erf_minimax_approx_compute_vector_fwd(const Vmm &vmm_src);
    void round_compute_vector_fwd(const Vmm &vmm_src);
    void hardswish_compute_vector_fwd(const Vmm &vmm_src);
    void hardsigmoid_compute_vector_fwd(const Vmm &vmm_src);

    void exp_compute_vector_bwd(const Vmm &vmm_src);
    void relu_compute_vector_bwd(const Vmm &vmm_src);
    void elu_compute_vector_bwd(const Vmm &vmm_src);
    void tanh_compute_vector_bwd(const Vmm &vmm_src);
    void square_compute_vector_bwd(const Vmm &vmm_src);
    void abs_compute_vector_bwd(const Vmm &vmm_src);
    void sqrt_compute_vector_bwd(const Vmm &vmm_src);
    void linear_compute_vector_bwd(const Vmm &vmm_src);
    void soft_relu_compute_vector_bwd(const Vmm &vmm_src);
    void logistic_compute_vector_bwd(const Vmm &vmm_src);
    void mish_compute_vector_bwd(const Vmm &vmm_src);
    void gelu_tanh_compute_vector_bwd(const Vmm &vmm_src);
    void swish_compute_vector_bwd(const Vmm &vmm_src);
    void log_compute_vector_bwd(const Vmm &vmm_src);
    void clip_compute_vector_bwd(const Vmm &vmm_src);
    void pow_compute_vector_bwd(const Vmm &vmm_src);
    void gelu_erf_compute_vector_bwd(const Vmm &vmm_src);
    void hardswish_compute_vector_bwd(const Vmm &vmm_src);
    void hardsigmoid_compute_vector_bwd(const Vmm &vmm_src);

    enum key_t {
        scale = 0, // scale argument
        alpha, // alpha argument
        beta, // beta argument
        zero, // 0.f
        half, // 0.5f
        one, // 1.f  or  mask for exponent bits
        two, // 2.f
        three, // 3.f
        six, // 6.f
        minus_one, // -1.f  or  changes sign to opposite
        minus_two, // -2.f
        minus_three, // -3.f
        ln2f, // 0.69314718f
        positive_mask, // changes sign to positive
        sign_mask, // gets sign value
        exponent_bias, // (127 = 2^7 - 1), gets exponent bits
        exp_log2ef, // 1.44269502f - formula-based for approx
        exp_ln_flt_max_f, // logf(FLT_MAX) - max normal value
        exp_ln_flt_min_f, // logf(FLT_MIN) - min normal value
        exp_pol, // see correspondent table for float values
        // e^(2*x)+2*e^x+2 = FLT_MAX; x =~ 44.36141952603634
        fwd_mish_max_x_for_equation_f,
        // e^x(e^3x+4e^2x+e^x*(6+4*x)+4*(1+x)) = FLT_MAX; x =~ 22.18070976278534
        bwd_mish_max_x_for_equation_f,
        tanh_idx_bias, // bias applied during index computation
        tanh_idx_mask, // mask applied to extract index
        tanh_linear_ubound, // arg below which tanh(x) = x
        tanh_saturation_lbound, // arg after which tanh(x) = 1.f
        tanh_pol_table, // table of polynomial coefficients
        soft_relu_one_twenty_six, // 126.f
        soft_relu_mantissa_sign_mask, // mask for mantissa bits and sign
        soft_relu_pol, // see correspondent table for float values
        gelu_tanh_fitting_const, // 0.044715f
        gelu_tanh_fitting_const_times_three, // 0.134145f
        gelu_tanh_sqrt_two_over_pi, // sqrtf(2.f/pi) = 0.797884f
        // 0.3275911f - implementation based for approx
        gelu_erf_Abramowitz_Stegun_approx_const,
        gelu_erf_Abramowitz_Stegun_one_over_sqrt_two, // 1.f / sqrtf(2.f)
        // 1.f / sqrtf(pi) = 0.564190f
        gelu_erf_Abramowitz_Stegun_one_over_sqrt_pi,
        // see correspondent table for float values
        gelu_erf_Abramowitz_Stegun_pol,
        gelu_erf_minimax_pol, // see correspondent table for float values
        gelu_erf_idx_bias, // bias applied to compute table index
        gelu_erf_rbound, // upper bound at which we clamp erf at 1
        gelu_erf_one, // just the integer value 1, used for index clamping
        gelu_erf_twenty_three, // just the integer value 23, used for index clamping
        gelu_erf_twenty_four, // just the integer value 24, used for index clamping
        log_inf, // inf
        log_minus_inf, // -inf
        log_qnan, // qnan
        log_mantissa_mask, // gets mantissa bits
        log_full_k_reg_mask, // sets k_register with all bits of 1
        log_full_vector_reg_mask, // sets vector register will all bits of 1
        log_five_bit_offset, // 5 bits off (31 = 2^5 - 1)
        log_pol, // see correspondent table for float values
        log_predefined_vals, // see correspondent table for float values
        undef_key,
    };

    size_t table_off(key_t key, size_t key_off_val_shift = 0) {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        // TODO: enforce through data structure
        const auto it = entry_map_.find(key); // search an entry for a key
        if (it == entry_map_.end()) {
            assert(!"Non-existent key");
            return 0;
        }
        const auto &te = (*it).second;
        const auto scale = te.bcast ? vlen_ : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }
    Xbyak::Address table_val(key_t key, size_t key_off_val_shift = 0) {
        auto off = table_off(key, key_off_val_shift);
        return h->ptr[p_table_ + off];
    }

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t; // offsets are in bytes wrt p_table_
    using table_entry_bcast_t = bool; // true => bcast value

    struct table_entry_t {
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };
    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };

    using table_t = std::multimap<key_t, table_entry_t>;
    using mapped_table_t = std::multimap<key_t, mapped_table_entry_t>;

    void register_table_entries();
    mapped_table_t entry_map_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
