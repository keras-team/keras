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

#ifndef CPU_X64_UNI_REDUCTION_KERNEL_HPP
#define CPU_X64_UNI_REDUCTION_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_resampling_pd.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_uni_reduction_kernel_base_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reduction)

    jit_uni_reduction_kernel_base_t(const jit_reduction_conf_t &conf)
        : jit_generator(jit_name(), conf.isa)
        , conf_(conf)
        , sum_scales_(conf_.sum_scales) {}
    virtual ~jit_uni_reduction_kernel_base_t() = default;

    virtual std::size_t get_simd_w() = 0;

protected:
    const jit_reduction_conf_t &conf_;
    std::queue<float> sum_scales_;
};

template <cpu_isa_t isa, typename Vmm = typename cpu_isa_traits<isa>::Vmm>
struct jit_uni_reduction_kernel_t : public jit_uni_reduction_kernel_base_t {
    jit_uni_reduction_kernel_t(
            const jit_reduction_conf_t &conf, const memory_desc_t *dst_md);

    virtual ~jit_uni_reduction_kernel_t() = default;

    std::size_t get_simd_w() override { return simd_w_; }

private:
    using compute_fn_t = std::function<void(
            const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc)>;

    void init_acc();
    void init_compute_op();
    void init_compute_scalar_op();
    void init_post_ops_injector(const memory_desc_t *dst_md);

    void reduce_ymm_to_xmm(const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp);
    void reduce_xmm_to_scalar(const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp,
            const std::size_t number_of_values_to_reduce
            = number_of_f32_in_xmm_);
    void reduce_zmm_to_ymm(const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp);
    void reduce_ymm_to_scalar(const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp1,
            const Xbyak::Xmm &tmp2,
            const std::size_t number_of_values_to_reduce
            = number_of_f32_in_ymm_);
    void reduce_vmm_to_scalar(const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp1,
            const Xbyak::Xmm &tmp2, const Xbyak::Xmm &tmp3,
            const std::size_t number_of_values_to_reduce
            = number_of_f32_in_zmm_);

    void reduce();
    void reduce_base();
    void reduce_ne_convert_xf16();

    void load_params();
    void apply_sum(const int data_idx);
    void apply_postops(const int data_idx);
    void finalize();
    void generate() override;

    const Vmm vmm_tail_load_mask_ = Vmm(0);
    const Vmm vmm_tail_store_mask_ = Vmm(1);
    const Vmm vmm_zero_saturation_ = Vmm(2);
    const Vmm vmm_saturation_ubound_ = Vmm(3);
    const Vmm vmm_acc_ = Vmm(4);
    const Vmm vmm_tmp1_ = Vmm(5);
    const Vmm vmm_tmp2_ = Vmm(6);
    const Vmm vmm_tmp3_ = Vmm(7);
    const Vmm vmm_tmp4_ = Vmm(8);
    const Vmm vmm_sum_scale_ = Vmm(9);
    const Vmm rhs_dt_helper_vmm_ = Vmm(10);
    const Xbyak::Zmm vmm_bf16_emu_1_ = Xbyak::Zmm(28);
    const Xbyak::Zmm vmm_bf16_emu_2_ = Xbyak::Zmm(29);
    const Xbyak::Zmm vmm_bf16_emu_3_ = Xbyak::Zmm(30);
    const Xbyak::Zmm vmm_bf16_emu_4_ = Xbyak::Zmm(31);

    const Xbyak::Opmask k_tail_load_mask_ = k3;
    const Xbyak::Opmask k_tail_store_mask_ = k4;

    const Xbyak::Reg64 reg_work_ = rax;
    const Xbyak::Reg64 reg_src_ = rbx;
    const Xbyak::Reg64 reg_dst_ = rdx;
    const Xbyak::Reg64 reg_param_ = abi_param1;
    const Xbyak::Reg64 reg_tmp_ = abi_not_param1;
    const Xbyak::Reg64 reg_tmp1_ = r13;

    static constexpr bool is_zmm_ = std::is_same<Vmm, Xbyak::Zmm>::value;
    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;
    static constexpr std::size_t vlen_ = is_zmm_ ? 64 : is_ymm_ ? 32 : 16;
    static constexpr std::size_t simd_w_ = vlen_ / sizeof(float);
    static constexpr std::size_t number_of_f32_in_xmm_ = 4;
    static constexpr std::size_t number_of_f32_in_ymm_ = 8;
    static constexpr std::size_t number_of_f32_in_zmm_ = 16;
    const std::size_t load_tail_size_;
    static constexpr std::size_t store_tail_size_ = 1;

    io::jit_io_helper_t<Vmm> io_load_;
    io::jit_io_helper_t<Vmm> io_store_;

    compute_fn_t compute_op_;
    compute_fn_t compute_scalar_op_;

    const Xbyak::Opmask elt_inj_opmask_ = k1;
    const Xbyak::Reg64 reg_po_injector_helper_1_ = r14;
    const Xbyak::Reg64 reg_po_injector_helper_2_ = r15;
    const Xbyak::Reg64 reg_po_injector_helper_3_ = r12;

    // post-ops injector does not use avx512_core_bf16 instructions
    static constexpr cpu_isa_t inject_isa_
            = isa == avx512_core_bf16 ? avx512_core : isa;
    std::unique_ptr<injector::jit_uni_postops_injector_t<inject_isa_, Vmm>>
            postops_injector_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
