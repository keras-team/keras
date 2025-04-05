/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_X8S8S32X_DECONVOLUTION_HPP
#define CPU_X64_JIT_UNI_X8S8S32X_DECONVOLUTION_HPP

#include <functional>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_deconvolution_pd.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace zp {
class jit_uni_deconv_zp_pad_str_kernel_base_t;
} // namespace zp

namespace injector {
template <cpu_isa_t isa, typename Vmm>
class jit_uni_postops_injector_t;
} // namespace injector

using namespace Xbyak;

template <cpu_isa_t isa, typename Vmm>
struct _jit_uni_x8s8s32x_deconv_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_uni_x8s8s32x_deconv_fwd_kernel);

    _jit_uni_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_wrapper &dst_d);
    ~_jit_uni_x8s8s32x_deconv_fwd_kernel();

    const jit_conv_conf_t jcp_;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa, Vmm>>
            postops_injector_;
    using reg64_t = const Xbyak::Reg64;

    static constexpr dim_t IC_SUB_STEP = 4;
    const int ker_max_regs_ = -1;

    enum ker_block_t {
        no_last_block = 0x1U,
        last_ic_block = 0x2U,
        last_sp_block = 0x4U,
    };

    /* data regs */
    const reg64_t reg_src_ = r8;
    const reg64_t reg_filt_ = r9;
    const reg64_t reg_dst_ = r10;
    const reg64_t param1_ = abi_param1;
    const reg64_t reg_kh_ = abi_not_param1;
    const reg64_t reg_ki_ = r14;

    const reg64_t reg_nur_w_ = rbx;
    const reg64_t reg_bias_ = rdx;
    const reg64_t reg_icb_ = reg_bias_;
    const reg64_t reg_ptr_scales_ = rax;
    const reg64_t reg_ptr_dst_scales_ = abi_not_param1;
    const reg64_t reg_ptr_saturation_ubound_ = rax;
    const reg64_t reg_oc_blocks_ = rsi;

    const reg64_t aux_reg_src_ = r11;
    const reg64_t aux_reg_filt_ = r12;

    const reg64_t aux_reg_src_d_ = r13;
    const reg64_t aux_reg_filt_d_ = r15;

    const reg64_t reg_compensation_ = r14;
    const reg64_t reg_scratch_ = r14;
    const reg64_t reg_ptr_sum_scale_ = r11;
    const reg64_t reg_ptr_sum_zp_ = r15;
    const reg64_t reg_overflow_ = rax;
    const reg64_t reg_comp_strides_ = reg_overflow_;
    const reg64_t reg_ker_long_offt_ = r15;
    const reg64_t reg_zp_dst_ = r15;
    const reg64_t reg_zp_src_ = r15;
    const reg64_t reg_zp_compensation_ = r11;
    const Xbyak::Address zp_src_pad_comp_addr_ = ptr[rsp];
    const Xbyak::Address reg_scratch_preserved_ = ptr[rsp + 8];
    static constexpr int reserved_stack_size_ = 16;

    const Vmm vmm_tmp_ = Vmm(3);
    const Vmm vmm_one_ = Vmm(2);
    /* used during write-out section of store_output */
    const Vmm vmm_zero_ = Vmm(0);
    const Vmm vmm_saturation_ = vmm_zero_;
    const Vmm vmm_wei_ = vmm_zero_;
    const Vmm vmm_scale_ = vmm_zero_;
    const Vmm vmm_dst_scale_ = vmm_zero_;
    /* signed input */
    const Vmm vmm_shift_ = Vmm(1);
    const Vmm vmm_comp_ = Vmm(1);
    const Vmm vmm_bias_ = vmm_zero_;
    const Vmm vmm_prev_dst_ = vmm_zero_;
    const Vmm vmm_sum_zp_ = vmm_tmp_;

    Vmm vmm_out(int i_ur, int i_oc) const;
    Vmm vmm_inp(int i_ic, int nb_x_blocking) const;

    int get_ow_start(int ki, int l_overflow) const noexcept;
    int get_ow_end(int ur_w, int ki, int r_overflow) const noexcept;
    int get_blocking_size() const noexcept;
    int get_tail_size() const noexcept;

    void prepare_output(int ur_w);
    void apply_postops(int ur_w, bool last_oc_block, const float *p_sum_scale,
            const int32_t *p_sum_zp);
    void store_output(int ur_w, bool last_oc_block);
    void compute_ker(int ur_w, int l_overflow, int r_overflow,
            ker_block_t last_ic_block_flag, bool h_padded = false);
    void compute(const Vmm vreg_acc, const Vmm vreg_wei, const Vmm vreg_src);
    std::function<Vmm()> prepare_round_robin_vmm_inp_generator(
            int ur_w) const noexcept;
    void apply_zp_src_pad_str_comp(
            int ur_w, int l_overflow, int r_overflow, bool h_padded);
    void append_zp_src_pad_str_comp(int ur_w, int l_overflow, int r_overflow,
            bool h_padded, bool last_oc_block);
    void kh_loop(int ur_w, int pad_l, int pad_r, ker_block_t last_ker_block);
    void icb_loop(int ur_w, int pad_l, int pad_r, bool last_block);
    void generate() override;
    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Reg64 reg,
            int offset, int load_size);
};

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_deconv_fwd_kernel {

    jit_uni_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_wrapper &dst_d);

    status_t create_kernel() {
        if (kernel_) return kernel_->create_kernel();
        return status::out_of_memory;
    }

    ~jit_uni_x8s8s32x_deconv_fwd_kernel();

    void operator()(const jit_deconv_call_s *p) const { (*kernel_)(p); }

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const deconvolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            const bool with_bias, memory_desc_t &bias_md,
            primitive_attr_t &attr, int nthreads);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    using _jit_avx2_x8s8s32x_deconv_fwd_kernel
            = _jit_uni_x8s8s32x_deconv_fwd_kernel<avx2, Xbyak::Ymm>;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_uni_x8s8s32x_deconv_fwd_kernel);
    std::unique_ptr<jit_generator> kernel_;
};

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_uni_int8:",
                        isa == avx2 && jcp_.has_vnni ? avx2_vnni : isa, ""),
                jit_uni_x8s8s32x_deconvolution_fwd_t);

        status_t init(engine_t *engine);
        jit_conv_conf_t jcp_;
    };

    jit_uni_x8s8s32x_deconvolution_fwd_t(const pd_t *apd);
    ~jit_uni_x8s8s32x_deconvolution_fwd_t();

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t execute_forward_1d(const exec_ctx_t &ctx) const;
    status_t execute_forward_2d(const exec_ctx_t &ctx) const;
    status_t execute_forward_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    const float *adjust_oscales(const memory_tracking::grantor_t &scratchpad,
            const float *src_scales, const float *wei_scales) const;
    std::unique_ptr<jit_uni_x8s8s32x_deconv_fwd_kernel<isa>> kernel_;
    std::unique_ptr<zp::jit_uni_deconv_zp_pad_str_kernel_base_t>
            zp_src_pad_comp_kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
