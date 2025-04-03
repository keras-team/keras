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

#ifndef CPU_X64_JIT_AVX2_CONV_KERNEL_F32_HPP
#define CPU_X64_JIT_AVX2_CONV_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx2_conv_fwd_kernel_f32 : public jit_generator {
    jit_avx2_conv_fwd_kernel_f32(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_fwd_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx2>>
            postops_injector_;

    constexpr static int isa_simd_width_
            = cpu_isa_traits<avx2>::vlen / sizeof(float);
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output = rsi;
    reg64_t reg_bias = rbx;

    reg64_t aux_reg_inp_d = r11;
    reg64_t aux_reg_ker_d = abi_not_param1;

    reg64_t reg_ki = rsi;
    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t ki_iter = r12;
    reg64_t reg_channel = ki_iter;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_oc_blocks = r14;
    reg64_t imm_addr64 = r15;
    reg64_t reg_long_offt = r15;
    Xbyak::Reg32 reg_ci_flag = r13d;
    Xbyak::Reg32 reg_oc_flag = r14d;

    /* binary post-ops operand */
    reg64_t temp_offset_reg = r12;

    Xbyak::Ymm ytmp = Xbyak::Ymm(14);

    inline void oh_step_unroll_kw(
            int ur_w, int pad_l, int pad_r, int oc_blocks);
    inline void oh_step_nopad(int ur_w, int pad_l, int pad_r, int oc_blocks);
    void apply_postops(const int oc_blocks, const int ur_w, const int oc_tail);
    inline void width_blk_step(int ur_w, int pad_l, int pad_r, int oc_blocks);
    inline void solve_common(int oc_blocks);

    inline dim_t filter_w_to_input(int ki, int oi = 0, int pad_l = 0) {
        return static_cast<dim_t>(ki) * (jcp.dilate_w + 1) + oi * jcp.stride_w
                - pad_l;
    };
    inline dim_t filter_h_to_input(int ki) {
        return static_cast<dim_t>(ki) * (jcp.dilate_h + 1) * jcp.iw;
    };
    inline dim_t filter_d_to_input(int ki) {
        return static_cast<dim_t>(ki) * (jcp.dilate_d + 1) * jcp.iw * jcp.ih;
    };

    inline dim_t get_input_offset(int i_ic, int i_iw) {
        dim_t offset;
        if (utils::one_of(jcp.src_tag, format_tag::ncw, format_tag::nchw,
                    format_tag::ncdhw)) {
            offset = static_cast<dim_t>(i_ic) * jcp.id * jcp.ih * jcp.iw + i_iw;
        } else if (utils::one_of(jcp.src_tag, format_tag::nwc, format_tag::nhwc,
                           format_tag::ndhwc)) {
            offset = static_cast<dim_t>(i_iw) * jcp.ic * jcp.ngroups + i_ic;
        } else {
            offset = static_cast<dim_t>(i_iw) * jcp.ic_block + i_ic;
        }
        return sizeof(float) * offset;
    }

    inline dim_t get_output_offset(int i_oc_block, int i_ow) {
        dim_t offset;
        if (utils::one_of(jcp.dst_tag, format_tag::nwc, format_tag::nhwc,
                    format_tag::ndhwc)) {
            offset = static_cast<dim_t>(i_ow) * jcp.oc * jcp.ngroups
                    + i_oc_block * jcp.oc_block;
        } else {
            offset = static_cast<dim_t>(i_oc_block) * jcp.od * jcp.oh * jcp.ow
                            * jcp.oc_block
                    + i_ow * jcp.oc_block;
        }
        return sizeof(float) * offset;
    }

    inline dim_t get_kernel_offset(int i_oc_block, int ki, int i_ic) {
        dim_t block_step_size = jcp.ic_block * jcp.oc_block;
        dim_t ic_block_step_size = static_cast<dim_t>(jcp.kd) * jcp.kh * jcp.kw
                * block_step_size;
        dim_t oc_block_step_size
                = static_cast<dim_t>(jcp.nb_ic) * ic_block_step_size;
        dim_t offset = static_cast<dim_t>(i_oc_block) * oc_block_step_size
                + ki * block_step_size + i_ic * jcp.oc_block;
        return sizeof(float) * offset;
    }

    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }

    void generate() override;
};

struct jit_avx2_conv_bwd_data_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_bwd_data_kernel_f32)

    jit_avx2_conv_bwd_data_kernel_f32(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name()), jcp(ajcp) {}

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;

private:
    using reg64_t = const Xbyak::Reg64;

    reg64_t reg_ddst = rax;
    reg64_t aux_reg_ddst = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r10;
    reg64_t reg_dsrc = rsi;
    reg64_t aux_reg_ddst_oc_loop = rbx; // used in ndims < 5 case only
    reg64_t aux_reg_kernel_oc_loop = abi_not_param1; /* used in ndims < 5
                                                        case only */

    reg64_t aux_reg_dst_d = r12; // used in ndims == 5 case only
    reg64_t aux_reg_ker_d = r14; // used in ndims == 5 case only

    reg64_t reg_ki = abi_not_param1; // used in ndims == 5 case only
    reg64_t kj = r11;
    reg64_t oi_iter = r12;
    reg64_t reg_kh = r14;
    reg64_t reg_channel = r13; // used in ndims < 5 case only
    reg64_t reg_channel_work = r9; // used in ndims < 5 case only
    reg64_t reg_long_offt = r15;
    reg64_t reg_reduce_work = reg_long_offt;
    Xbyak::Reg32 reg_ci_flag = r13d; // used for nxc tails

    inline void compute_loop(int ur_w, int l_overflow, int r_overflow);

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

    inline dim_t filter_w_to_ddst(int ki, int oi = 0, int pad_l = 0) {
        return (oi + pad_l - ki * (jcp.dilate_w + 1)) / jcp.stride_w;
    }

    inline dim_t get_ddst_offset(int i_oc_block, int i_ow, int i_oc) {
        dim_t offset;
        if (utils::one_of(jcp.dst_tag, format_tag::nwc, format_tag::nhwc,
                    format_tag::ndhwc)) {
            offset = static_cast<dim_t>(i_ow) * jcp.oc * jcp.ngroups
                    + i_oc_block * jcp.oc_block + i_oc;
        } else {
            offset = static_cast<dim_t>(i_oc_block) * jcp.od * jcp.oh * jcp.ow
                            * jcp.oc_block
                    + i_ow * jcp.oc_block + i_oc;
        }
        return sizeof(float) * offset;
    }

    inline dim_t get_dsrc_offset(int i_ic_block, int i_iw) {
        dim_t offset;
        if (utils::one_of(jcp.src_tag, format_tag::nwc, format_tag::nhwc,
                    format_tag::ndhwc)) {
            offset = static_cast<dim_t>(i_iw) * jcp.ic * jcp.ngroups
                    + i_ic_block * jcp.ic_block;
        } else {
            offset = static_cast<dim_t>(i_ic_block) * jcp.id * jcp.ih * jcp.iw
                            * jcp.ic_block
                    + i_iw * jcp.ic_block;
        }
        return sizeof(float) * offset;
    }

    inline dim_t get_kernel_offset(
            int i_oc_block, int i_ic_block, int ki, int i_oc) {
        dim_t block_step_size = jcp.ic_block * jcp.oc_block;
        dim_t ic_block_step_size = static_cast<dim_t>(jcp.kd) * jcp.kh * jcp.kw
                * block_step_size;
        dim_t oc_block_step_size
                = static_cast<dim_t>(jcp.nb_ic) * ic_block_step_size;
        dim_t offset = static_cast<dim_t>(i_oc_block) * oc_block_step_size
                + i_ic_block * ic_block_step_size + ki * block_step_size
                + i_oc * jcp.ic_block;
        return sizeof(float) * offset;
    }
};

struct jit_avx2_conv_bwd_weights_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_bwd_weights_kernel_f32)

    jit_avx2_conv_bwd_weights_kernel_f32(const jit_conv_conf_t &ajcp)
        : jit_generator(jit_name()), jcp(ajcp) {}

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_weights_d,
            const memory_desc_wrapper &diff_dst_d);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_output = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_ur_w_trips = r10;
    reg64_t reg_tmp = r11;
    reg64_t reg_oj = r15;
    reg64_t reg_ih_count = rbx;
    reg64_t aux_reg_input = r12;
    reg64_t aux_reg_kernel = r13;
    reg64_t ki = r14;
    reg64_t reg_long_offt = r11;
    reg64_t reg_channel = reg_ih_count; // used for nxc tails
    Xbyak::Reg32 reg_ci_flag = r9d; // used for nxc tails

    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_ic_block_step(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset);
    inline void compute_oh_step_disp();
    inline void compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w);
    inline void compute_oh_step_common(int ic_block_step, int max_ur_w);
    inline void compute_oh_loop_common();

    inline dim_t get_input_offset(int i_ic, int i_iw) {
        dim_t offset;
        if (utils::one_of(jcp.src_tag, format_tag::ncw, format_tag::nchw,
                    format_tag::ncdhw)) {
            offset = static_cast<dim_t>(i_ic) * jcp.id * jcp.ih * jcp.iw + i_iw;
        } else if (utils::one_of(jcp.src_tag, format_tag::nwc, format_tag::nhwc,
                           format_tag::ndhwc)) {
            offset = static_cast<dim_t>(i_iw) * jcp.ic * jcp.ngroups + i_ic;
        } else {
            offset = static_cast<dim_t>(i_iw) * jcp.ic_block + i_ic;
        }
        return sizeof(float) * offset;
    }

    inline dim_t get_output_offset(int i_oc_block, int i_ow) {
        dim_t offset;
        if (utils::one_of(jcp.dst_tag, format_tag::nwc, format_tag::nhwc,
                    format_tag::ndhwc)) {
            offset = static_cast<dim_t>(i_ow) * jcp.oc * jcp.ngroups
                    + i_oc_block * jcp.oc_block;
        } else {
            offset = static_cast<dim_t>(i_oc_block) * jcp.od * jcp.oh * jcp.ow
                            * jcp.oc_block
                    + i_ow * jcp.oc_block;
        }
        return sizeof(float) * offset;
    }

    inline dim_t get_kernel_offset(int ki, int i_ic) {
        dim_t block_step_size = jcp.ic_block * jcp.oc_block;
        dim_t offset = static_cast<dim_t>(ki) * block_step_size
                + i_ic * jcp.oc_block;
        return sizeof(float) * offset;
    }
    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
