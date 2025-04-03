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

#ifndef CPU_X64_JIT_AVX512_CORE_BF16_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_BF16_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename Vmm>
struct _jit_avx512_core_bf16_fwd_kernel : public jit_generator {
    _jit_avx512_core_bf16_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_core_bf16_fwd_kernel)

    const jit_conv_conf_t &jcp;
    const primitive_attr_t &attr_;

private:
    constexpr static int isa_simd_width_
            = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    using Vmm_down_t =
            typename utils::conditional<std::is_same<Vmm, Xbyak::Zmm>::value,
                    Xbyak::Ymm, Xbyak::Xmm>::type;
    using reg64_t = const Xbyak::Reg64;
    enum {
        ker_reg_base_idx = 28,
    };

    reg64_t param = abi_param1; //L: RDI, W: RCX

    reg64_t reg_src = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_dst = r10;
    reg64_t reg_owb = r11;

    reg64_t aux_reg_src = r12;
    reg64_t aux_reg_ker = r13;

    reg64_t reg_ic = rax;
    reg64_t reg_oc = r15;
    reg64_t reg_bias = rbx;

    reg64_t reg_kj = abi_not_param1;
    reg64_t reg_ki = reg_bias;
    reg64_t reg_oi = rdx;
    reg64_t reg_kh = rsi;

    reg64_t reg_long_offt = r14;

    /* binary post-ops operand */
    reg64_t temp_offset_reg = r12;

    int vmm_dst_idx(const int i_ur, const int i_oc) const;
    Vmm vmm_dst(const int i_ur, const int i_oc) const;

    Vmm vmm_src(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Vmm(idx);
    }

    Vmm_down_t vmm_src_down(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Vmm_down_t(idx);
    }

    inline Vmm may_be_mask_vmm(Vmm vmm, bool mask_flag, bool zero_mask,
            bool use_extended_mask = false) {
        if (mask_flag) {
            vmm = vmm
                    | (use_extended_mask ? k_oc_tail_mask_extended
                                         : k_oc_tail_mask);
            if (zero_mask) vmm = vmm | T_z;
        }
        return vmm;
    }

    inline Vmm_down_t may_be_mask_vmm(Vmm_down_t vmm, bool mask_flag) {
        return (mask_flag) ? vmm | k_oc_tail_mask : vmm;
    }

    Vmm vmm_wei = Vmm(31);
    Vmm vmm_prev_dst = Vmm(31);
    Vmm vmm_bias = Vmm(31);

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_scratch = reg_ic;
    Xbyak::Zmm bf16_emu_reserv_4 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(30);

    Xbyak::Opmask odd_load_mask = Xbyak::Opmask(2);
    Xbyak::Opmask even_load_mask = Xbyak::Opmask(3);
    Xbyak::Opmask k_oc_tail_mask = Xbyak::Opmask(4);
    Xbyak::Opmask k_oc_tail_mask_extended = Xbyak::Opmask(5);
    const Xbyak::Opmask postops_mask = Xbyak::Opmask(6);

    constexpr static int off_reg_src_ = 0;
    constexpr static int off_reg_ker_ = 8;
    constexpr static int stack_space_needed_ = 16;

    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core, Vmm>>
            postops_injector_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    inline void prepare_dst(int ur_w);
    void apply_postops(int ur_w);
    inline void store_dst(int ur_w);
    inline void compute_loop(int ur_w, int pad_l, int pad_r);

    void generate() override;

    inline dim_t get_dst_offset(dim_t sp_idx, int ocb) {
        const bool is_layout_nxc = is_dst_layout_nxc();
        dim_t sp_str = is_layout_nxc ? static_cast<dim_t>(jcp.ngroups) * jcp.oc
                                     : jcp.oc_block;
        dim_t ocb_str = jcp.oc_block
                * (is_layout_nxc ? 1 : (dim_t)jcp.od * jcp.oh * jcp.ow);
        return jcp.typesize_out * (ocb_str * ocb + sp_str * sp_idx);
    }

    inline dim_t filter_w_to_src(int kw, int ow = 0, int pad_l = 0) {
        return static_cast<dim_t>(kw) * (jcp.dilate_w + 1) + ow * jcp.stride_w
                - pad_l;
    }
    inline dim_t filter_h_to_src(int kh) {
        return static_cast<dim_t>(kh) * (jcp.dilate_h + 1) * jcp.iw;
    }
    inline dim_t filter_d_to_src(int kd) {
        return static_cast<dim_t>(kd) * (jcp.dilate_d + 1) * jcp.iw * jcp.ih;
    }

    inline dim_t get_src_offset(dim_t ic_idx, dim_t isp) {
        int icb = ic_idx / jcp.ic_block;
        int ic = ic_idx % jcp.ic_block;
        dim_t isp_str = is_src_layout_nxc()
                ? static_cast<dim_t>(jcp.ngroups) * jcp.ic
                : (jcp.is_1stconv ? 1 : jcp.ic_block);
        dim_t full_spatial_size = (dim_t)jcp.iw * jcp.ih * jcp.id;
        dim_t ic_str = jcp.is_1stconv && !is_src_layout_nxc()
                ? full_spatial_size
                : 1;
        dim_t icb_str
                = (is_src_layout_nxc() ? 1 : full_spatial_size) * jcp.ic_block;
        return jcp.typesize_in * (isp_str * isp + icb_str * icb + ic_str * ic);
    }

    inline dim_t get_kernel_offset(
            int ocb, int ic_idx, int kw, int kh = 0, int kd = 0) {
        int scale = 2; //bf16 vnni is used
        int rnd_ic_block = utils::rnd_up(jcp.ic_block, scale);
        int icb = ic_idx / jcp.ic_block;
        int ic = ic_idx % jcp.ic_block;
        dim_t ksp_str = static_cast<dim_t>(rnd_ic_block) * jcp.oc_block;
        dim_t ksp_idx
                = static_cast<dim_t>(kd) * jcp.kh * jcp.kw + kh * jcp.kw + kw;

        dim_t icb_str = static_cast<dim_t>(jcp.kd) * jcp.kh * jcp.kw * ksp_str;
        dim_t ocb_str = static_cast<dim_t>(jcp.nb_ic) * icb_str;
        dim_t ic_off = static_cast<dim_t>(ic / scale) * jcp.oc_block * scale
                + (ic % scale);
        return static_cast<dim_t>(jcp.typesize_in)
                * (ocb * ocb_str + icb * icb_str + ksp_idx * ksp_str + ic_off);
    }

    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w
                - nstl::max(0,
                        utils::div_up(
                                pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1),
                                jcp.stride_w));
    }
    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_dst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

struct jit_avx512_core_bf16_fwd_kernel {
    jit_avx512_core_bf16_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md)
        : kernel_(nullptr) {
        switch (ajcp.oc_block) {
            case 16:
                kernel_ = new _jit_avx512_core_bf16_fwd_kernel<Xbyak::Zmm>(
                        ajcp, attr, dst_md);
                return;
            case 8:
                kernel_ = new _jit_avx512_core_bf16_fwd_kernel<Xbyak::Ymm>(
                        ajcp, attr, dst_md);
                return;
            case 4:
                kernel_ = new _jit_avx512_core_bf16_fwd_kernel<Xbyak::Xmm>(
                        ajcp, attr, dst_md);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() {
        if (kernel_) return kernel_->create_kernel();
        return status::out_of_memory;
    }

    ~jit_avx512_core_bf16_fwd_kernel() { delete kernel_; }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void operator()(const jit_conv_call_s *p) const { (*kernel_)(p); }
    const Xbyak::uint8 *jit_ker() const { return kernel_->jit_ker(); }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_avx512_core_bf16_fwd_kernel);
    jit_generator *kernel_;
};

template <typename Vmm>
struct _jit_avx512_core_bf16_bwd_data_kernel : public jit_generator {

    _jit_avx512_core_bf16_bwd_data_kernel(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name(), avx512_core_bf16)
        , jcp(ajcp)
        , bf16_emu_(nullptr) {
        if (!isa_has_bf16(jcp.isa))
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                    bf16_emu_scratch, bf16_emu_reserv_4, bf16_emu_reserv_5);
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_core_bf16_bwd_data_kernel_f32)

    const jit_conv_conf_t &jcp;

private:
    using Vmm_down_t =
            typename utils::conditional<std::is_same<Vmm, Xbyak::Zmm>::value,
                    Xbyak::Ymm, Xbyak::Xmm>::type;
    using reg64_t = const Xbyak::Reg64;
    enum {
        ker_reg_base_idx = 31,
    };

    reg64_t param = abi_param1;
    reg64_t reg_dst = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_src = r10;

    reg64_t reg_iwb = rdx;

    reg64_t aux_reg_dst = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_dst_d = r12;
    reg64_t aux_reg_ker_d = r13;
    reg64_t reg_ki = rsi;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_oc = r11;
    reg64_t reg_ic = aux_reg_ker_d;

    Xbyak::Opmask k_ic_tail_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_ic_tail_mask_extended = Xbyak::Opmask(3);

    Vmm vmm_ddst(int i_ic) {
        int idx = i_ic + jcp.nb_ic_blocking * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Vmm(idx);
    }

    Vmm_down_t vmm_ddst_down(int i_ic) {
        int idx = i_ic + jcp.nb_ic_blocking * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Vmm_down_t(idx);
    }

    Vmm vmm_dsrc(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Vmm(idx);
    }

    inline Vmm may_be_mask_vmm(Vmm vmm, bool mask_flag, bool zero_mask,
            bool use_extended_mask = false) {
        if (mask_flag) {
            vmm = vmm
                    | (use_extended_mask ? k_ic_tail_mask_extended
                                         : k_ic_tail_mask);
            if (zero_mask) vmm = vmm | T_z;
        }
        return vmm;
    }

    inline Vmm_down_t may_be_mask_vmm(Vmm_down_t vmm, bool mask_flag) {
        return mask_flag ? vmm | k_ic_tail_mask : vmm;
    }

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_scratch = reg_kj;
    Xbyak::Zmm bf16_emu_reserv_4 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(30);

    Vmm vmm_wei = Vmm(31);
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop(int ur_w, int l_overflow, int r_overflow);
    void generate() override;

    int get_iw_start(int ki, int l_overflow) {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    int get_iw_end(int ur_w, int ki, int r_overflow) {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }

    inline int filter_h_to_dst(int kh) {
        return kh * (jcp.dilate_h + 1) * jcp.ow;
    }
    inline int filter_d_to_dst(int kd) {
        return kd * (jcp.dilate_d + 1) * jcp.ow * jcp.oh;
    }

    inline size_t get_diff_src_offset(int iw_idx, int n_ic_block) {
        const bool is_nxc_layout = is_dsrc_layout_nxc();
        size_t iw_str = is_nxc_layout ? jcp.ngroups * jcp.ic : jcp.ic_block;
        size_t icb_str = jcp.ic_block
                * (is_nxc_layout ? 1 : (size_t)jcp.id * jcp.ih * jcp.iw);
        return jcp.typesize_out * (iw_str * iw_idx + icb_str * n_ic_block);
    }

    inline size_t get_diff_dst_offset(
            int osp_idx, int oc_within_block_idx, int oc_block_idx = 0) {
        const bool is_nxc_layout = is_ddst_layout_nxc();
        size_t osp_str = is_nxc_layout ? jcp.ngroups * jcp.oc : jcp.oc_block;
        size_t ocb_str = jcp.oc_block
                * (is_nxc_layout ? 1 : (size_t)jcp.od * jcp.oh * jcp.ow);
        return jcp.typesize_in
                * (osp_str * osp_idx + ocb_str * oc_block_idx
                        + oc_within_block_idx);
    }

    inline size_t get_kernel_offset(
            int icb, int oc_idx, int kw, int kh = 0, int kd = 0) {
        int scale = 2; //bf16 vnni is used
        int ocb = oc_idx / jcp.oc_block;
        int oc = oc_idx % jcp.oc_block;
        size_t ksp_str = jcp.ic_block * jcp.oc_block;
        size_t ksp_idx = kd * jcp.kh * jcp.kw + kh * jcp.kw + kw;

        size_t icb_str = jcp.kd * jcp.kh * jcp.kw * ksp_str;
        size_t ocb_str = jcp.nb_ic * icb_str;
        size_t oc_off = (oc / scale) * jcp.ic_block * scale + (oc % scale);
        return jcp.typesize_in
                * (ocb * ocb_str + icb * icb_str + ksp_idx * ksp_str + oc_off);
    }

    inline bool is_dsrc_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

struct jit_avx512_core_bf16_bwd_data_kernel {

    jit_avx512_core_bf16_bwd_data_kernel(const jit_conv_conf_t &ajcp)
        : kernel_(nullptr) {
        switch (ajcp.ic_block) {
            case 16:
                kernel_ = new _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Zmm>(
                        ajcp);
                return;
            case 8:
                kernel_ = new _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Ymm>(
                        ajcp);
                return;
            case 4:
                kernel_ = new _jit_avx512_core_bf16_bwd_data_kernel<Xbyak::Xmm>(
                        ajcp);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() {
        if (kernel_) return kernel_->create_kernel();
        return status::out_of_memory;
    }

    ~jit_avx512_core_bf16_bwd_data_kernel() { delete kernel_; }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &diff_src_md,
            memory_desc_t &weights_md, memory_desc_t &diff_dst_md,
            int nthreads);
    void operator()(const jit_conv_call_s *p) const { (*kernel_)(p); }
    const Xbyak::uint8 *jit_ker() const { return kernel_->jit_ker(); }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_avx512_core_bf16_bwd_data_kernel);
    jit_generator *kernel_;
};

struct jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 : public jit_generator {

    jit_avx512_core_bf16_conv_bwd_weights_kernel_f32(
            const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name(), avx512_core_bf16)
        , jcp(ajcp)
        , bf16_emu_(nullptr) {
        if (!isa_has_bf16(jcp.isa)) {
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(
                    this, one, even, selector, scratch, tmp0, tmp1);
        }
    }

    ~jit_avx512_core_bf16_conv_bwd_weights_kernel_f32() = default;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_bf16_conv_bwd_weights_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
            memory_desc_t &diff_dst_md, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    const jit_conv_conf_t &jcp;

private:
    Xbyak::Label dst_prm_table;
    // Used by compute_ic_block_step_{vpermw, interleave}
    Xbyak::Opmask m_ffffffff = Xbyak::Opmask(1);
    // Used by compute_ic_block_step_vpermw
    Xbyak::Opmask m_0000ffff = Xbyak::Opmask(2);
    Xbyak::Opmask m_ffff0000 = Xbyak::Opmask(3);
    Xbyak::Opmask m_0000_oc_tail = Xbyak::Opmask(4);
    Xbyak::Opmask m_oc_tail_0000 = Xbyak::Opmask(5);
    Xbyak::Opmask m_0000_ic_tail = Xbyak::Opmask(6);
    Xbyak::Opmask m_ic_tail_0000 = Xbyak::Opmask(7);
    // Used by compute_ic_block_step_extern (1st_conv only)
    Xbyak::Opmask everyother_mask = Xbyak::Opmask(6);
    Xbyak::Opmask everyother_shift_mask = Xbyak::Opmask(7);
    // Used by compute_ic_block_step_interleave (1st_conv only)
    Xbyak::Opmask underflow_mask = Xbyak::Opmask(4);
    Xbyak::Opmask overflow_mask = Xbyak::Opmask(5);
    Xbyak::Opmask underflow_stride_mask = Xbyak::Opmask(6);
    Xbyak::Opmask overflow_stride_mask = Xbyak::Opmask(7);

    using reg64_t = const Xbyak::Reg64;
    enum {
        sizeof_cacheline = 64,
        full_spat_opt_working_set_size = 48 * 1024,
        full_spat_max_working_set_size = 128 * 1024,
    };
    static const int max_ur_w;

    reg64_t param = abi_param1;
    reg64_t reg_src = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_ddst = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_ur_w_trips = r10;
    reg64_t reg_oj = r15;
    reg64_t reg_tmp = r14;
    reg64_t reg_ih_shift = reg_tmp;
    reg64_t reg_long_offt = r14;
    reg64_t reg_icb = rbx;

    reg64_t ki = r11;
    reg64_t reg_oj_setup = r11;
    reg64_t reg_kd_count = r12;
    reg64_t reg_oi = r12;
    reg64_t reg_d_index = r13;
    reg64_t reg_src_d = r15;
    reg64_t reg_ddst_d = rbx;
    reg64_t aux_reg_src = r12;
    reg64_t aux_reg_kernel = r13;

    Xbyak::Zmm vreg_bias_acc = Xbyak::Zmm(0);
    Xbyak::Zmm vreg_bias_unit = Xbyak::Zmm(1);
    Xbyak::Zmm vreg_bias_ddst = Xbyak::Zmm(2);

    Xbyak::Zmm one = Xbyak::Zmm(27);
    Xbyak::Zmm even = Xbyak::Zmm(28);
    Xbyak::Zmm selector = Xbyak::Zmm(29);
    Xbyak::Zmm tmp0 = Xbyak::Zmm(30);
    Xbyak::Zmm tmp1 = Xbyak::Zmm(31);
    reg64_t scratch = r11;

    inline void maybe_zero_kernel();
    inline void get_ur_w(int &ur_w, int &ur_w_tail, int &ur_w_trips);
    inline void compute_oh_step_unroll_ow_icblock(int ic_block_step);
    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_oh_step_unroll_ow(int ic_block_step);
    inline void compute_ic_block_step(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int src_offset, int kernel_offset,
            int ddst_offset, bool is_tail = false);
    inline void compute_ic_block_step_extern(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int src_offset, int kernel_offset,
            int ddst_offset, bool is_tail = false);
    inline void compute_ic_block_step_interleave(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int src_offset, int kernel_offset,
            int ddst_offset, bool is_tail = false);
    inline void compute_ic_block_step_vpermw(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int src_offset, int kernel_offset,
            int ddst_offset, bool is_tail = false);
    inline void compute_oh_step_common(int ic_block_step);
    inline void compute_oh_step_disp();
    inline void compute_loop();
    inline void compute_oh_loop_common(bool partial = false);
    inline void compute_od_loop_common(bool partial = false);
    void compute_full_spat_loop();
    void compute_diff_bias_init();
    void compute_diff_bias_row(bool is_partial = true);
    void maybe_compute_diff_bias();
    void convert_src_to_vnni_format(
            int ur_w, int pad_l, int pad_r, int src_offset);
    void may_be_set_oc_tail_mask();
    void may_be_reset_oc_tail_mask();
    inline void compute_ic_block_step_vpermw_expl(int ur_w, int pad_l,
            int pad_r, int ic_block_step, int src_offset, int kernel_offset,
            int ddst_offset, bool is_tail = false);
    inline bool is_src_layout_nxc() {
        return jcp.uses_permw_transposition
                && utils::one_of(jcp.src_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return jcp.uses_permw_transposition
                && utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                        format_tag::nhwc, format_tag::nwc);
    }

    void generate() override;

    static void balance(const jit_conv_conf_t &j, int &nthr, int &nthr_mb,
            int &nthr_g, int &nthr_oc_b, int &nthr_ic_b);

    void get_w_positions(int ur_w, int pad_l, int pad_r, int i_ur, int i_kw,
            int &iw_0, int &iw_1) {
        auto get_w_position = [&](int idx) {
            int iw = i_ur + idx;
            if (iw >= ur_w) return -1;
            iw += i_kw;
            if (iw - pad_l < 0 || iw > (ur_w - 1) + (jcp.kw - 1) - pad_r)
                return -1;
            return iw - pad_l;
        };
        iw_0 = get_w_position(0);
        iw_1 = get_w_position(1);
    }
    bool check_borders(int ur_w, int pad_l, int pad_r, int i_ur, int i_kw) {
        int iw_1, iw_2;
        get_w_positions(ur_w, pad_l, pad_r, i_ur, i_kw, iw_1, iw_2);

        return (iw_1 == -1 && iw_2 == -1) ? false : true;
    }
    bool get_load_mask(int ur_w, int pad_l, int pad_r, int i_ur, int i_kw,
            Xbyak::Opmask &load_mask) {
        int iw_1, iw_2;
        get_w_positions(ur_w, pad_l, pad_r, i_ur, i_kw, iw_1, iw_2);

        bool rt = true;
        if (iw_1 != -1 && iw_2 != -1)
            load_mask = m_ffffffff;
        else if (iw_1 != -1 && iw_2 == -1)
            load_mask = m_0000ffff;
        else if (iw_1 == -1 && iw_2 != -1)
            load_mask = m_ffff0000;
        else
            rt = false;

        return rt;
    }

    inline dim_t filter_w_to_src(int kw, int ow = 0, int pad_l = 0) {
        int stride_w = jcp.transpose_src ? 1 : jcp.stride_w;
        return static_cast<dim_t>(kw) * (jcp.dilate_w + 1) + ow * stride_w
                - pad_l;
    }
    inline dim_t filter_h_to_src(int kh) {
        return static_cast<dim_t>(kh) * (jcp.dilate_h + 1);
    }
    inline dim_t filter_d_to_src(int kd) {
        return static_cast<dim_t>(kd) * (jcp.dilate_d + 1) * jcp.ih;
    }

    inline dim_t get_src_offset(dim_t ic_idx, dim_t w_idx, dim_t hd_idx = 0) {
        // For is_src_layout_nxc() the ic_idx index inside the block
        // is supported only ic_idx == jcp.ic_block is considered as a shift
        // within one block and not as moving to the next ic block.
        assert(IMPLICATION(!is_src_layout_nxc(), ic_idx <= jcp.ic_block));
        dim_t icb = is_src_layout_nxc() ? ic_idx / jcp.ic_block : 0;
        dim_t ic = is_src_layout_nxc() ? ic_idx % jcp.ic_block : ic_idx;
        dim_t iw_str = jcp.is_1stconv || jcp.transpose_src
                ? 1
                : (is_src_layout_nxc()
                                ? static_cast<dim_t>(jcp.ngroups) * jcp.ic
                                : jcp.ic_block);
        dim_t ihid_str = static_cast<dim_t>(jcp.tr_iw)
                * (jcp.transpose_src ? jcp.ic_block : iw_str);
        // jcp.transpose_src w_idx might be greater than jcp.tr_iw as right zero
        // padding memory is shared with left zero padding of the next block
        dim_t isp_off = hd_idx * ihid_str + w_idx * iw_str;
        dim_t full_spatial_size = (dim_t)jcp.tr_iw * jcp.ih * jcp.id;
        dim_t ic_str = jcp.transpose_src
                ? jcp.tr_iw
                : (jcp.is_1stconv ? full_spatial_size : 1);
        dim_t icb_str = static_cast<dim_t>(jcp.ic_block)
                * (is_src_layout_nxc() ? 1 : full_spatial_size);
        return static_cast<dim_t>(jcp.typesize_in)
                * (isp_off + icb_str * icb + ic_str * ic);
    }

    inline dim_t get_ddst_offset(dim_t w_idx, dim_t hd_idx = 0) {
        int ow_per_oc = jcp.transpose_dst ? 2 : 1;
        int ch_mult
                = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
        dim_t hd_off = jcp.tr_ow * ch_mult * hd_idx;
        dim_t w_off
                = w_idx / ow_per_oc * ow_per_oc * ch_mult + w_idx % ow_per_oc;
        return jcp.typesize_in * (w_off + hd_off);
    }

    inline dim_t get_kernel_offset(int ic_idx, dim_t ksp_idx) {
        // Only the ic_idx index inside the block is supported,
        // ic_idx == jcp.ic_block is considered as a shift inside one block
        // and not as moving to the next ic block.
        // Negative values are supported for negative shift.
        assert(nstl::abs(ic_idx) <= jcp.ic_block);
        return jcp.typesize_out * jcp.oc_block
                * (ksp_idx * jcp.ic_block + ic_idx);
    }

    Xbyak::Zmm get_perm_reg() {
        int idx = !(jcp.uses_permw_transposition
                          && jcp.kernel_kind == expl_bcast)
                ? 24
                : ((!isa_has_bf16(jcp.isa)) ? 26 : 31);
        return Xbyak::Zmm(idx);
    }
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    inline int interleave_w_reorder_size(int ur_w) const;
    inline int interleave_w_reorder_bytes(int ur_w);
    inline int interleave_stack_size(int ur_w, int ic_block_step);
    inline int permw_stack_size(int ur_w) {
        return (ur_w + jcp.kw - 1) * sizeof_cacheline;
    }

    inline void setup_stack_space();
    static const int extern_ic_block_step_stack_size = 0;
    int ic_block_step_stack_size = 0;
    int stack_space_needed = 0;
    int permw_buffer_start = 0;
    int kd_count_offset = 0;
    int src_d_offset = 0;
    int ddst_d_offset = 0;
    int d_index_offset = 0;
    int trans_tmp_offset = 0;
    int ih_dilate_shift = 0;
    int icb_loop_ker_ptr = 0;
    int icb_loop_src_ptr = 0;
};
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
