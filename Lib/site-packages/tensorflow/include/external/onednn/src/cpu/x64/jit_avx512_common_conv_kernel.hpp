/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_COMMON_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_COMMON_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename Vmm>
struct _jit_avx512_common_conv_fwd_kernel : public jit_generator {

    _jit_avx512_common_conv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_common_conv_fwd_kernel)

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    constexpr static int isa_simd_width_
            = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    using reg64_t = const Xbyak::Reg64;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
    };

    reg64_t param = abi_param1;
    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;

    reg64_t reg_owb = r12;

    reg64_t aux_reg_inp = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t reg_channel = rsi;
    reg64_t reg_bias = rdx;

    reg64_t aux_reg_ker_d = r9;
    reg64_t aux_reg_inp_d = rbx;
    reg64_t reg_ki = r10;

    reg64_t reg_kj = rax;
    reg64_t reg_relu_ns = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_tmp = rbp;

    reg64_t reg_long_offt = r11;
    reg64_t reg_out_long_offt = r14;
    reg64_t reg_ker_long_offt = r11;
    reg64_t reg_tail = aux_reg_ker;
    reg64_t reg_load_work = reg_tail;

    /* binary post-ops operand */
    reg64_t temp_offset_reg = r12;

    Xbyak::Opmask k_oc_tail_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask postops_mask = Xbyak::Opmask(3);

    inline Vmm vmm_ker(int i_ic) {
        assert(i_ic < 4);
        return Vmm(ker_reg_base_idx + i_ic);
    }

    inline int vmm_out_idx(int i_ur, int i_oc) {
        const int idx = i_ur * jcp.nb_oc_blocking + i_oc;
        assert(idx < ker_reg_base_idx);
        return idx;
    }

    inline Vmm vmm_out(int i_ur, int i_oc) {
        return Vmm(vmm_out_idx(i_ur, i_oc));
    }

    inline Vmm vmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Vmm(idx);
    }

    Xbyak::Reg64 imm_addr64 = r15;
    Vmm vmm_wei = Vmm(31);

    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;

    inline void prepare_output(int ur_w);
    inline void apply_postops(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma(int ur_w, int pad_l, int pad_r);
    inline void compute_loop_fma_core(int ur_w, int pad_l, int pad_r);
    inline void compute_loop(int ur_w, int pad_l, int pad_r);

    void generate() override;

    inline size_t get_output_offset(int oi, int n_oc_block) {
        const bool is_nxc_layout = is_dst_layout_nxc();
        size_t ow_str = is_nxc_layout ? jcp.ngroups * jcp.oc : jcp.oc_block;
        size_t ocb_str = is_nxc_layout
                ? jcp.oc_block
                : (size_t)jcp.od * jcp.oh * jcp.ow * jcp.oc_block;

        return jcp.typesize_out * (n_oc_block * ocb_str + oi * ow_str);
    }

    inline size_t get_input_offset(int ki, int ic, int oi, int pad_l) {
        const bool is_nxc_layout = is_src_layout_nxc();
        size_t iw_str = is_nxc_layout ? jcp.ngroups * jcp.ic
                                      : (!jcp.is_1stconv ? jcp.ic_block : 1);
        size_t ic_str = !jcp.is_1stconv || is_nxc_layout
                ? 1
                : (size_t)jcp.iw * jcp.ih * jcp.id;
        size_t iw_idx = ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l;

        return jcp.typesize_in * (iw_idx * iw_str + ic * ic_str);
    }

    inline int get_kernel_offset(
            int ki, int ic, int n_oc_block, int ker_number) {
        return jcp.typesize_in * jcp.oc_block
                * (n_oc_block * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw
                                * jcp.kd
                        + (ic + ker_number) + ki * jcp.ic_block);
    }

    inline int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    inline int get_ow_end(int ur_w, int ki, int pad_r) {
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

struct jit_avx512_common_conv_fwd_kernel {

    jit_avx512_common_conv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md)
        : kernel_(nullptr) {
        switch (ajcp.oc_block) {
            case 16:
                kernel_ = new _jit_avx512_common_conv_fwd_kernel<Xbyak::Zmm>(
                        ajcp, attr, dst_md);
                return;
            case 8:
                kernel_ = new _jit_avx512_common_conv_fwd_kernel<Xbyak::Ymm>(
                        ajcp, attr, dst_md);
                return;
            case 4:
                kernel_ = new _jit_avx512_common_conv_fwd_kernel<Xbyak::Xmm>(
                        ajcp, attr, dst_md);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() {
        if (kernel_) return kernel_->create_kernel();
        return status::out_of_memory;
    }

    ~jit_avx512_common_conv_fwd_kernel() { delete kernel_; }

    enum { typesize = sizeof(float) };

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void operator()(jit_conv_call_s *p) const { (*kernel_)(p); }

    const Xbyak::uint8 *jit_ker() const { return kernel_->jit_ker(); }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_avx512_common_conv_fwd_kernel);
    jit_generator *kernel_;
};

template <typename Vmm>
struct _jit_avx512_common_conv_bwd_data_kernel_f32 : public jit_generator {

    _jit_avx512_common_conv_bwd_data_kernel_f32(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name()), jcp(ajcp) {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_common_conv_bwd_data_kernel_f32)
    jit_conv_conf_t jcp;

private:
    using reg64_t = const Xbyak::Reg64;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
    };

    reg64_t param = abi_param1;
    reg64_t reg_dst = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_src = r10;

    reg64_t reg_iwb = r14;

    reg64_t aux_reg_dst = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_dst_d = rbx;
    reg64_t aux_reg_ker_d = r9;
    reg64_t reg_ki = r10;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_channel = rsi;

    reg64_t reg_tmp = rbp;
    reg64_t reg_long_offt = r14;

    reg64_t reg_tail = aux_reg_ker;
    reg64_t reg_load_work = reg_tail;

    Xbyak::Opmask k_ic_tail_mask = Xbyak::Opmask(1);

    inline Vmm vmm_ker(int i_ic) {
        assert(i_ic < 4);
        return Vmm(ker_reg_base_idx + i_ic);
    }
    inline Vmm vmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Vmm(idx);
    }
    inline Vmm vmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Vmm(idx);
    }

    Vmm vmm_wei = Vmm(31);

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop_fma_core(
            int ur_w, int l_overflow, int r_overflow, int k_offset);
    inline void compute_loop(
            int ur_w, int l_overflow, int r_overflow, int k_offset = 0);
    void generate() override;

    inline int get_iw_start(int ki, int l_overflow) {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    inline int get_iw_end(int ur_w, int ki, int r_overflow) {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }

    inline size_t get_diff_src_offset(int iw, int icb) {
        const bool is_nxc_layout = is_dsrc_layout_nxc();
        size_t iw_str = is_nxc_layout ? jcp.ngroups * jcp.ic : jcp.ic_block;
        size_t icb_str = is_nxc_layout
                ? jcp.ic_block
                : (size_t)jcp.id * jcp.ih * jcp.iw * jcp.ic_block;

        return typesize * (icb * icb_str + iw * iw_str);
    }

    inline ptrdiff_t get_dst_offset(int iw, int oc, int kw) {
        ptrdiff_t ow
                = (iw + jcp.l_pad - kw * (jcp.dilate_w + 1)) / jcp.stride_w;
        ptrdiff_t ow_str = is_ddst_layout_nxc()
                ? static_cast<ptrdiff_t>(jcp.ngroups) * jcp.oc
                : jcp.oc_block;

        return typesize * (ow * ow_str + oc);
    };

    inline bool is_dsrc_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

struct jit_avx512_common_conv_bwd_data_kernel_f32 {

    jit_avx512_common_conv_bwd_data_kernel_f32(const jit_conv_conf_t &ajcp)
        : kernel_(nullptr) {
        switch (ajcp.ic_block) {
            case 16:
                kernel_ = new _jit_avx512_common_conv_bwd_data_kernel_f32<
                        Xbyak::Zmm>(ajcp);
                return;
            case 8:
                kernel_ = new _jit_avx512_common_conv_bwd_data_kernel_f32<
                        Xbyak::Ymm>(ajcp);
                return;
            case 4:
                kernel_ = new _jit_avx512_common_conv_bwd_data_kernel_f32<
                        Xbyak::Xmm>(ajcp);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() {
        if (kernel_) return kernel_->create_kernel();
        return status::out_of_memory;
    }

    ~jit_avx512_common_conv_bwd_data_kernel_f32() { delete kernel_; }

    enum { typesize = sizeof(float) };

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &diff_src_d,
            memory_desc_t &weights_d, memory_desc_t &diff_dst_d, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void operator()(const jit_conv_call_s *p) const { (*kernel_)(p); }
    const Xbyak::uint8 *jit_ker() const { return kernel_->jit_ker(); }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_avx512_common_conv_bwd_data_kernel_f32);
    jit_generator *kernel_;
};

struct jit_avx512_common_conv_bwd_weights_kernel_f32 : public jit_generator {

    jit_avx512_common_conv_bwd_weights_kernel_f32(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name()), jcp(ajcp) {}

    void generate() override {
        if (jcp.harness != harness_nxc)
            generate_kernel();
        else
            generate_microkernel();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_conv_bwd_weights_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
            memory_desc_t &diff_dst_md, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;

private:
    using reg64_t = const Xbyak::Reg64;
    enum { typesize = sizeof(float) };
    static const int max_ur_w;
    static const int min_oh_reduce;

    reg64_t param = abi_param1;
    reg64_t reg_input = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_output = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_ur_w_trips = r10;
    reg64_t reg_oj = r15;
    reg64_t reg_tmp = r14;
    reg64_t reg_long_offt = r14;
    reg64_t reg_icb = rbx;

    reg64_t ki = r11;
    reg64_t reg_kd_count = r12;
    reg64_t reg_oi = r12;
    reg64_t reg_d_index = r13;
    reg64_t reg_input_d = r15;
    reg64_t reg_output_d = rbx;
    reg64_t aux_reg_input = r12;
    reg64_t aux_reg_kernel = r13;
    reg64_t reg_bias = rbx;
    reg64_t reg_oc_tail = r14;

    Xbyak::Opmask k_oc_mask = Xbyak::Opmask(2);

    inline void bias_kernel_2d();
    inline void bias_kernel_3d();
    inline void maybe_zero_kernel();
    inline void compute_oh_step_unroll_ow_icblock(
            int ic_block_step, int max_ur_w);
    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w);
    inline void compute_ic_block_step(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool input_wraparound = false);
    inline void compute_ic_block_step_fma(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool input_wraparound);
    inline void compute_ic_block_step_fma_expl(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool input_wraparound);
    inline void compute_oh_step_common(int ic_block_step, int max_ur_w);
    inline void compute_oh_step_disp();
    inline void compute_oh_loop_common();
    inline void compute_oh_loop_partial();
    inline void compute_od_loop_partial();

    inline void compute_loop();
    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }

    inline ptrdiff_t get_full_src_offset(
            int i_iw, int i_ic, ptrdiff_t input_offset) {
        const bool is_nxc_layout = is_src_layout_nxc();
        const size_t w_shift_st = (jcp.is_hw_transp ? jcp.iw : 1)
                * (jcp.is_1stconv ? 1 : jcp.ic_block);
        ptrdiff_t w_shift = is_nxc_layout ? jcp.ngroups * jcp.ic : w_shift_st;
        ptrdiff_t ic_shift = jcp.is_1stconv && !is_nxc_layout
                ? (ptrdiff_t)jcp.ih * jcp.iw * jcp.id
                : 1;

        ptrdiff_t local_input_offset = i_iw * w_shift + i_ic * ic_shift;
        return input_offset + typesize * local_input_offset;
    };

    inline int get_iw_idx(int ow, int kw, int l_pad) {
        return ow * jcp.stride_w + kw * (jcp.dilate_w + 1) - l_pad;
    }

    void generate_kernel();
    void generate_microkernel();

    static void balance(const jit_conv_conf_t &j, int &nthr, int &nthr_mb,
            int &nthr_g, int &nthr_oc_b, int &nthr_ic_b, int nthreads);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
