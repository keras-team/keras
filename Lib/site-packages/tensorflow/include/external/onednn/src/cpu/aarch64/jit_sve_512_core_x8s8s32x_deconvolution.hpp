/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_SVE_512_CORE_X8S8S32X_DECONVOLUTION_HPP
#define CPU_AARCH64_JIT_SVE_512_CORE_X8S8S32X_DECONVOLUTION_HPP

#include <functional>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include "cpu/aarch64/jit_uni_deconv_zp_pad_str_kernel.hpp"
#include "cpu/cpu_deconvolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

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

template <cpu_isa_t isa>
struct jit_sve_512_core_x8s8s32x_deconv_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sve_512_core_x8s8s32x_deconv_fwd_ker_t);

    jit_sve_512_core_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);
    ~jit_sve_512_core_x8s8s32x_deconv_fwd_kernel();

    const jit_conv_conf_t &jcp;
    const primitive_attr_t &attr_;

private:
    const int ic_sub_step = 4;
    const uint64_t isa_sveLen = cpu_isa_traits<isa>::vlen;

    /* data regs */
    const XReg reg_src = x8;
    const XReg reg_filt = x9;
    const XReg reg_dst = x10;
    const XReg param1 = abi_param1;
    const XReg reg_kh = abi_param2;
    const XReg reg_ki = x14;

    const XReg reg_nur_w = x3;
    const XReg reg_bias = x2;
    const XReg reg_icb = reg_bias;
    const XReg reg_ptr_scales = x16;
    const WReg reg_ptr_saturation_ubound = w16;
    const XReg reg_oc_blocks = x6;

    const XReg aux_reg_src = x11;
    const XReg aux_reg_filt = x12;

    const XReg aux_reg_src_d = x13;
    const XReg aux_reg_filt_d = x17;

    const XReg reg_compensation = x14;
    const XReg reg_bias_alpha = abi_not_param1;
    const XReg reg_overflow = x16;
    const XReg reg_comp_strides = reg_overflow;
    const XReg reg_ker_long_offt = x17;
    const XReg &reg_zp_dst_ = x17;
    const XReg &reg_zp_src_ = x17;
    const XReg &reg_zp_compensation = x11;
    const XReg reg_zp_src_pad_comp = x20;

    PReg ktail_mask = p2;
    const PReg mask_tmp = p4;

    const ZReg vmm_tmp0 = z28;
    const ZReg vmm_tmp1 = z29;
    /* used during write-out section of store_output */
    const ZReg vmm_saturation = z31;
    const ZReg vmm_wei = z31;

    /* unsigned input */
    const ZReg vmm_shift = z30;
    const ZReg vmm_comp = z30;
    ///

    ZReg vmm_out(int i_ur, int i_oc) {
        int idx = i_ur * jcp.nb_oc_blocking + i_oc;
        assert(idx < 31);
        return ZReg(idx);
    }

    ZReg vmm_inp(int i_ic, int nb_x_blocking) const {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return ZReg(idx);
    }
    ZReg vmm_bias_alpha() { return ZReg(jcp.nb_oc_blocking * jcp.ur_w); }

    template <typename T>
    ZReg compress_addr(const XReg &base, T offt, const bool mask_flag = false,
            const bool bcast = false) {
        assert(bcast == false);
        PReg mask = P_ALL_ONE;
        XReg addr = base;

        if (offt) {
            add_imm(X_DEFAULT_ADDR, base, offt, X_TMP_0);
            addr = X_DEFAULT_ADDR;
        }

        if (mask_flag) mask = PReg(ktail_mask.getIdx());

        mov(X_TMP_0, 64);
        ld1w(vmm_tmp0.s, mask / T_z, ptr(addr));
        return vmm_tmp0;
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
    void compute(
            const ZReg &vreg_acc, const ZReg &vreg_wei, const ZReg &vreg_src);
    std::function<uint32_t()> prepare_round_robin_vmm_inp_generator(
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
    void cvt2ps(const data_type_t type_in, const ZReg &vmm_in, const ZReg &op,
            const bool mask_flag);

    void uni_ld1r(const VReg4S &d, const XReg &addr) { ld1r(d, ptr(addr)); }

    void uni_ld1r(const ZRegS &d, const XReg &addr) {
        ld1rw(d, P_ALL_ONE / T_z, ptr(addr));
    }

    void uni_fmin(const VReg4S &d, const VReg4S &s) { fmin(d, d, s); }

    void uni_fmin(const ZRegS &d, const ZRegS &s) {
        fmin(d, P_ALL_ONE / T_m, s);
    }
};

struct _jit_sve_512_core_x8s8s32x_deconv_fwd_kernel {
    _jit_sve_512_core_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md)
        : kernel_(nullptr) {

        int ch_block = ajcp.is_depthwise ? ajcp.ch_block : ajcp.ic_block;
        switch (ch_block) {
            case 16:
                kernel_ = new jit_sve_512_core_x8s8s32x_deconv_fwd_kernel<
                        sve_512>(ajcp, attr, dst_md);
                return;
            case 8:
                kernel_ = new jit_sve_512_core_x8s8s32x_deconv_fwd_kernel<
                        sve_256>(ajcp, attr, dst_md);
                return;
            case 4:
                kernel_ = new jit_sve_512_core_x8s8s32x_deconv_fwd_kernel<
                        sve_128>(ajcp, attr, dst_md);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() { return kernel_->create_kernel(); }

    ~_jit_sve_512_core_x8s8s32x_deconv_fwd_kernel() { delete kernel_; }

    void operator()(const jit_deconv_call_s *p) const { (*kernel_)(p); }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const deconvolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            const bool with_bias, memory_desc_t &bias_md,
            primitive_attr_t &attr, int nthreads);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(_jit_sve_512_core_x8s8s32x_deconv_fwd_kernel);
    jit_generator *kernel_;
};

struct jit_sve_512_core_x8s8s32x_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_deconvolution:", sve_512, ""),
                jit_sve_512_core_x8s8s32x_deconvolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;
            const bool ok = mayiuse(sve_512) && is_fwd()
                    && (desc()->alg_kind & alg_kind::deconvolution_direct)
                    && utils::one_of(src_md(0)->data_type, s8, u8)
                    && weights_md(0)->data_type == s8
                    && IMPLICATION(with_bias(),
                            utils::one_of(
                                    weights_md(1)->data_type, f32, s32, s8, u8))
                    && utils::one_of(dst_md(0)->data_type, f32, s32, s8, u8)
                    && desc()->accum_data_type == s32
                    && attr()->has_default_values(skip_mask_t::oscale_runtime
                            | skip_mask_t::post_ops
                            | skip_mask_t::zero_points_runtime);
            if (!ok) return status::unimplemented;

            CHECK(_jit_sve_512_core_x8s8s32x_deconv_fwd_kernel::init_conf(jcp_,
                    *desc(), src_md_, weights_md_, dst_md_, with_bias(),
                    bias_md_, attr_, dnnl_get_max_threads()));

            auto scratchpad = scratchpad_registry().registrar();
            _jit_sve_512_core_x8s8s32x_deconv_fwd_kernel::init_scratchpad(
                    scratchpad, jcp_, *attr());

            return status::success;
        }

        jit_conv_conf_t jcp_;
    };

    jit_sve_512_core_x8s8s32x_deconvolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new _jit_sve_512_core_x8s8s32x_deconv_fwd_kernel(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));

        if (zp::should_calculate_deconv_zp_src_pad_str_comp(pd()->jcp_)) {
            CHECK(safe_ptr_assign(zp_src_pad_comp_kernel_,
                    zp::create_deconv_zp_pad_str_comp_ker<sve_512>(
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
    std::unique_ptr<_jit_sve_512_core_x8s8s32x_deconv_fwd_kernel> kernel_;
    std::unique_ptr<zp::jit_uni_deconv_zp_pad_str_kernel_base_t>
            zp_src_pad_comp_kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
