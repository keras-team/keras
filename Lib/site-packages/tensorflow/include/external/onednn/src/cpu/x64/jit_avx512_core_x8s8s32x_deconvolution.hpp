/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_X8S8S32X_DECONVOLUTION_HPP
#define CPU_X64_JIT_AVX512_CORE_X8S8S32X_DECONVOLUTION_HPP

#include <functional>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_deconvolution_pd.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/jit_uni_deconv_zp_pad_str_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

typedef enum {
    no_last_block = 0x1U,
    last_ic_block = 0x2U,
    last_sp_block = 0x4U,
} ker_block_t;

struct ur_w_blks_params_t {
    struct single_ur_w_blk_params_t {
        single_ur_w_blk_params_t(
                int l_overflow, int r_overflow, bool process_sp_carefully)
            : l_overflow(l_overflow)
            , r_overflow(r_overflow)
            , process_sp_carefully(process_sp_carefully) {}

        // l_overflow - no. of spatial elements of weights standing out of
        // src spatial when computing the 1st output pixel in the current blk
        int l_overflow;
        // r_overflow - no. of spatial elements of weights standing out of
        // src spatial when computing the lst output pixel in the current blk
        int r_overflow;
        // process_sp_carefully - indicates if loading the last src sp
        // for computation of the last dst sp of the block can't be done
        // by fetching 4 src sp at once
        bool process_sp_carefully;
    };
    std::vector<single_ur_w_blk_params_t> blks_params;
    int num_pre_blks; // num of blocks with l_overflow>0
    int num_post_blks; // num of blocks with r_overflow>0 or that need to be
            // processed carefully
};

template <typename Vmm>
struct jit_avx512_core_x8s8s32x_deconv_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_x8s8s32x_deconv_fwd_ker_t);

    jit_avx512_core_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);
    ~jit_avx512_core_x8s8s32x_deconv_fwd_kernel();

    const jit_conv_conf_t &jcp;
    const primitive_attr_t &attr_;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core, Vmm>>
            postops_injector_;

    const int ic_sub_step = 4;

    /* data regs */
    const Xbyak::Reg64 reg_src = r8;
    const Xbyak::Reg64 reg_filt = r9;
    const Xbyak::Reg64 reg_dst = r10;
    const Xbyak::Reg64 param1 = abi_param1;
    const Xbyak::Reg64 reg_kh = abi_not_param1;
    const Xbyak::Reg64 reg_ki = r14;

    const Xbyak::Reg64 reg_nur_w = rbx;
    const Xbyak::Reg64 reg_bias = rdx;
    const Xbyak::Reg64 reg_icb = reg_bias;
    const Xbyak::Reg64 reg_ptr_scales = rax;
    const Xbyak::Reg64 reg_ptr_dst_scales = rax;
    const Xbyak::Reg64 reg_ptr_saturation_ubound = rax;
    const Xbyak::Reg64 reg_oc_blocks = rsi;

    const Xbyak::Reg64 aux_reg_src = r11;
    const Xbyak::Reg64 aux_reg_filt = r12;

    const Xbyak::Reg64 aux_reg_src_d = r13;
    const Xbyak::Reg64 aux_reg_filt_d = r15;

    const Xbyak::Reg64 reg_compensation = r14;
    const Xbyak::Reg64 reg_scratch = r14;
    const Xbyak::Reg64 reg_ptr_sum_scale = r11;
    const Xbyak::Reg64 reg_overflow = rax;
    const Xbyak::Reg64 reg_comp_strides = reg_overflow;
    const Xbyak::Reg64 reg_ker_long_offt = r15;
    const Xbyak::Reg64 &reg_zp_dst_ = r15;
    const Xbyak::Reg64 &reg_zp_src_ = r15;
    const Xbyak::Reg64 &reg_zp_compensation = r11;
    static constexpr int reserved_stack_size_ = 16;
    const Xbyak::Address zp_src_pad_comp_addr = ptr[rsp];
    const Xbyak::Address reg_scratch_preserved = ptr[rsp + 8];

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    const Vmm vmm_tmp = Vmm(28);
    const Vmm vmm_one = Vmm(29);
    /* used during write-out section of store_output */
    const Vmm vmm_zero = Vmm(31);
    const Vmm vmm_saturation = Vmm(31);
    const Vmm vmm_wei = Vmm(31);

    /* signed input */
    const Vmm vmm_shift = Vmm(30);
    const Vmm vmm_comp = Vmm(30);
    const Vmm vmm_bias = Vmm(31);
    const Vmm vmm_dst_scale = Vmm(31);
    const Vmm vmm_prev_dst = Vmm(31);

    Vmm vmm_out(int i_ur, int i_oc) {
        int idx = i_ur * jcp.nb_oc_blocking + i_oc;
        assert(idx < 31);
        return Vmm(idx);
    }
    Vmm vmm_inp(int i_ic, int nb_x_blocking) const {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Vmm(idx);
    }

    int get_ow_start(int ki, int l_overflow) {
        int res = (jcp.ow - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;
        return res;
    }

    int get_ow_end(int ur_w, int ki, int r_overflow) {
        if (utils::one_of(ur_w, jcp.ow, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;
        return ur_w - res;
    }

    int get_blocking_size() const noexcept;
    int get_tail_size() const noexcept;
    void prepare_output(int ur_w);
    void store_output(int ur_w, bool last_oc_block);
    void compute(const Vmm &vreg_acc, const Vmm &vreg_wei, const Vmm &vreg_src);
    std::function<Vmm()> prepare_round_robin_vmm_inp_generator(
            int ur_w) const noexcept;
    void apply_zp_src_pad_str_comp(
            int ur_w, int l_overflow, int r_overflow, bool h_padded);
    void append_zp_src_pad_str_comp(int ur_w, int l_overflow, int r_overflow,
            bool h_padded, bool last_oc_block);
    void compute_ker(int ur_w, int l_overflow, int r_overflow,
            ker_block_t last_ic_block_flag, bool h_padded = false);
    void kh_loop(int ur_w, int pad_l, int pad_r, ker_block_t last_ker_block);
    void icb_loop(int ur_w, int pad_l, int pad_r, bool last_block);

    ur_w_blks_params_t get_ur_w_blks_params();

    void generate() override;
    void cvt2ps(data_type_t type_in, Vmm vmm_in, const Xbyak::Operand &op,
            bool mask_flag);
};

struct _jit_avx512_core_x8s8s32x_deconv_fwd_kernel {
    _jit_avx512_core_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md)
        : kernel_(nullptr) {

        int ch_block = ajcp.is_depthwise ? ajcp.ch_block : ajcp.ic_block;
        switch (ch_block) {
            case 16:
                kernel_ = new jit_avx512_core_x8s8s32x_deconv_fwd_kernel<
                        Xbyak::Zmm>(ajcp, attr, dst_md);
                return;
            case 8:
                kernel_ = new jit_avx512_core_x8s8s32x_deconv_fwd_kernel<
                        Xbyak::Ymm>(ajcp, attr, dst_md);
                return;
            case 4:
                kernel_ = new jit_avx512_core_x8s8s32x_deconv_fwd_kernel<
                        Xbyak::Xmm>(ajcp, attr, dst_md);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() {
        if (kernel_) return kernel_->create_kernel();
        return status::out_of_memory;
    }

    ~_jit_avx512_core_x8s8s32x_deconv_fwd_kernel() { delete kernel_; }

    void operator()(const jit_deconv_call_s *p) const { (*kernel_)(p); }

    static bool post_ops_ok(jit_conv_conf_t &jcp, primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const deconvolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            const bool with_bias, memory_desc_t &bias_md,
            primitive_attr_t &attr, int nthreads);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(_jit_avx512_core_x8s8s32x_deconv_fwd_kernel);
    jit_generator *kernel_;
};

struct jit_avx512_core_x8s8s32x_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_deconvolution:",
                        ((jcp_.has_vnni) ? avx512_core_vnni : avx512_core), ""),
                jit_avx512_core_x8s8s32x_deconvolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;
            VDISPATCH_DECONVOLUTION(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(
                    (desc()->alg_kind & alg_kind::deconvolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(utils::one_of(src_md(0)->data_type, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    weights_md(0)->data_type == s8, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    IMPLICATION(with_bias(),
                            utils::one_of(weights_md(1)->data_type, f32, s32,
                                    s8, u8)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    utils::one_of(dst_md(0)->data_type, f32, s32, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    desc()->accum_data_type == s32, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    attr()->has_default_values(skip_mask_t::scales_runtime
                            | skip_mask_t::post_ops
                            | skip_mask_t::zero_points_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_DECONVOLUTION(
                    attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);

            CHECK(_jit_avx512_core_x8s8s32x_deconv_fwd_kernel::init_conf(jcp_,
                    *desc(), src_md_, weights_md_, dst_md_, with_bias(),
                    bias_md_, attr_, dnnl_get_max_threads()));

            auto scratchpad = scratchpad_registry().registrar();
            _jit_avx512_core_x8s8s32x_deconv_fwd_kernel::init_scratchpad(
                    scratchpad, jcp_, *attr());

            return status::success;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_core_x8s8s32x_deconvolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new _jit_avx512_core_x8s8s32x_deconv_fwd_kernel(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));

        if (zp::should_calculate_deconv_zp_src_pad_str_comp(pd()->jcp_)) {
            CHECK(safe_ptr_assign(zp_src_pad_comp_kernel_,
                    zp::create_deconv_zp_pad_str_comp_ker<avx512_core>(
                            pd()->jcp_)));
            const auto zp_kernel_status
                    = zp_src_pad_comp_kernel_->create_kernel();
            if (zp_kernel_status != status::success) return zp_kernel_status;
        }

        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto ndims = pd()->ndims();
        if (ndims == 3)
            return execute_forward_1d(ctx);
        else if (ndims == 4)
            return execute_forward_2d(ctx);
        else if (ndims == 5)
            return execute_forward_3d(ctx);
        return status::runtime_error;
    }

private:
    status_t execute_forward_1d(const exec_ctx_t &ctx) const;
    status_t execute_forward_2d(const exec_ctx_t &ctx) const;
    status_t execute_forward_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<_jit_avx512_core_x8s8s32x_deconv_fwd_kernel> kernel_;
    std::unique_ptr<zp::jit_uni_deconv_zp_pad_str_kernel_base_t>
            zp_src_pad_comp_kernel_;
    const float *adjust_oscales(const memory_tracking::grantor_t &scratchpad,
            const float *src_scales, const float *wei_scales) const;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
