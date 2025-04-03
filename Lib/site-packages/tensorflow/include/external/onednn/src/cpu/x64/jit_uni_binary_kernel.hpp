/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef CPU_X64_UNI_BINARY_KERNEL_HPP
#define CPU_X64_UNI_BINARY_KERNEL_HPP

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

struct binary_kernel_t : public jit_generator {
    using op_t = binary_op_t;
    using bcast_t = binary_bcast_t;

    binary_kernel_t(const size_t vlen, const binary_pd_t *pd,
            const jit_binary_conf_t conf, const char *name,
            bool tail_kernel = false);
    ~binary_kernel_t() override = default;

    void operator()(jit_binary_call_s *p) { jit_generator::operator()(p); }

    size_t simd_w() const noexcept { return simd_w_; }
    size_t vlen() const noexcept { return vlen_; }

protected:
    size_t get_tail_size() const;

    const size_t vlen_;
    const size_t simd_w_;
    constexpr static int vmm_start_idx_ = 1;
    const binary_pd_t *pd_;
    const jit_binary_conf_t conf_;
    const bool is_tail_kernel_;
    const bool is_src1_outer_dims_tail_;
    const size_t tail_size_;
    const size_t padding_tail_size_;
};

template <cpu_isa_t isa, typename Vmm>
struct jit_uni_binary_kernel_t : public binary_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binary_kernel_t)

    const AddressFrame &vmmword = (std::is_same<Vmm, Xmm>::value)
            ? xword
            : ((std::is_same<Vmm, Ymm>::value) ? yword : zword);

    const bool is_avx512 = is_superset(isa, avx512_core);

    const Reg64 reg_param_ = abi_param1;
    const Reg64 reg_src0_ = r8;
    const Reg64 reg_src1_ = r9;
    const Reg64 reg_dst_ = r10;
    const Reg64 reg_offt_src0_ = r11;
    const Reg64 reg_outer_dims_range_ = r12;
    const Reg64 reg_offt_src1_ = rax;
    const Reg64 reg_src1_stride_range_ = r15;
    const Reg64 reg_reverse_src1_stride_range_ = rax;
    const Reg64 reg_reverse_spat_offt_ = r13;
    const Reg64 reg_tmp_ = r14;
    const Reg64 reg_tmp1_ = abi_not_param1;
    const Reg64 reg_elt_inj_table_ = r15;
    const Reg64 reg_off_rhs_postops_ = rdx;
    const Reg64 reg_scales_src0_ = rbx;
    const Reg64 reg_scales_src1_ = rbp;
    const Reg64 reg_offt_dst_ = rdx;
    const Opmask tail_opmask_ = k2;
    const Opmask cmp_mask = k3;
    const Opmask full_mask_ = k4;
    const Vmm vmm_tail_vmask_ = Vmm(0);
    const Vmm vreg_sum_scale_ = Vmm(is_avx512 ? 17 : 9);
    const Xmm xreg_sum_scale_ = Xmm(9);
    const Vmm vreg_zero_ = Vmm(is_avx512 ? 18 : 10);
    const Vmm vreg_one_ = Vmm(is_avx512 ? 19 : 11);
    const Vmm vreg_saturation_ubound_ = Vmm(is_avx512 ? 20 : 12);
    const Vmm vreg_bcast_src1_ = Vmm(is_avx512 ? 21 : 13);
    const Xmm xreg_bcast_src1_ = Xmm(13);
    const Vmm vreg_scales_src0_ = Vmm(is_avx512 ? 22 : 14);
    const Vmm vreg_scales_src1_ = Vmm(is_avx512 ? 23 : 15);

    const Zmm vreg_bf16_emu_1_ = Zmm(26);
    const Zmm vreg_bf16_emu_2_ = Zmm(27);
    const Zmm vreg_bf16_emu_3_ = Zmm(28);
    const Zmm vreg_bf16_emu_4_ = Zmm(29);

    const Vmm vmm_full_mask_ = Vmm(is_avx512 ? 24 : 5);
    const Vmm vmm_tmp_gather_ = Vmm(is_avx512 ? 25 : 6);
    const Vmm vmm_indices_ = Vmm(is_avx512 ? 30 : 7);
    const Vmm vmm_gathered_src_ = Vmm(is_avx512 ? 31 : 8);

    const size_t unroll_regs_ = is_avx512 ? 8 : 4;
    const size_t offt_src0_;
    const size_t offt_src1_;

    static constexpr cpu_isa_t inject_isa
            = isa == avx512_core_bf16 ? avx512_core : isa;
    io::jit_io_multi_dt_helper_t<Vmm> io_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<inject_isa, Vmm>>
            postops_injector_;
    const Opmask elt_inj_opmask_ = k1;

    void init();
    void init_post_ops_injector();
    void apply_postops(int unroll, bool tail);
    void load_kernel_params();
    Address src0_ptr(size_t offt = 0);
    Address src1_ptr(size_t offt = 0);
    Address dst_ptr(size_t offt = 0);
    unsigned int cmp_predicate(alg_kind_t alg);
    void perform_op(
            const Vmm &v0, const Vmm &v1, const Vmm &s_src0, const Vmm &s_src1);
    void prepare_isa_kernel();
    void compute_bcast(bool tail);
    void load_src1(const Vmm &vreg_src1, const int offt, bool tail);
    void store(int unroll, bool tail);
    void compute_ne_xf16_dst_body(int unroll, bool tail);
    void compute_dst_body(int unroll, bool tail);
    void compute_dst(int unroll, bool tail);
    void forward();
    void forward_over_outer_dims();
    void generate() override;

    jit_uni_binary_kernel_t(const binary_pd_t *pd, const jit_binary_conf_t conf,
            bool tail_kernel = false);
    ~jit_uni_binary_kernel_t() override = default;

    std::map<data_type_t, io::io_saturation_conf_t>
    create_saturation_vmm_map() const;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
