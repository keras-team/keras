/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_SSE41_CONV_KERNEL_F32_HPP
#define CPU_X64_JIT_SSE41_CONV_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_sse41_conv_fwd_kernel_f32 : public jit_generator {
    jit_sse41_conv_fwd_kernel_f32(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
            int nthreads);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_conv_fwd_kernel_f32)
    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    static constexpr auto simd_w_ = cpu_isa_traits<sse41>::vlen / sizeof(float);
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output = rsi;
    reg64_t reg_bias = rbx;

    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t ki_iter = r12;
    reg64_t reg_kh = abi_not_param1;
    reg64_t simd_iter = r15;
    reg64_t reg_oc_blocks = r14;
    reg64_t imm_addr64 = reg_oc_blocks;

    Xbyak::Reg32 reg_ci_flag = r13d;

    std::unique_ptr<injector::jit_uni_postops_injector_t<sse41>>
            postops_injector_;

    inline void oh_step_unroll_kw(
            int ur_w, int pad_l, int pad_r, int oc_blocks);
    inline void oh_step_nopad(int ur_w, int pad_l, int pad_r, int oc_blocks);
    inline void width_blk_step(int ur_w, int pad_l, int pad_r, int oc_blocks);
    inline void solve_common(int oc_blocks);

    inline dim_t filter_w_to_input(int ki, int oi = 0, int pad_l = 0) {
        return ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l;
    }

    inline dim_t filter_h_to_input(int ki) {
        return static_cast<dim_t>(ki) * (jcp.dilate_h + 1) * jcp.iw;
    }

    inline dim_t get_input_offset(int i_ic, int i_iw) {
        dim_t offset;
        if (utils::one_of(jcp.src_tag, format_tag::ncw, format_tag::nchw,
                    format_tag::ncdhw)) {
            offset = static_cast<dim_t>(i_ic) * jcp.ih * jcp.iw + i_iw;
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
            offset = (static_cast<dim_t>(i_oc_block) * jcp.oh * jcp.ow + i_ow)
                    * jcp.oc_block;
        }
        return sizeof(float) * offset;
    }

    inline dim_t get_kernel_offset(int i_oc_block, int ki, int i_ic) {
        dim_t block_step_size = jcp.ic_block * jcp.oc_block;
        dim_t ic_block_step_size = jcp.kh * jcp.kw * block_step_size;
        dim_t oc_block_step_size = jcp.nb_ic * ic_block_step_size;
        dim_t offset = i_oc_block * oc_block_step_size + ki * block_step_size
                + i_ic * jcp.oc_block;
        return sizeof(float) * offset;
    }

    void apply_postops(const int oc_blocks, const int ur_w);

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
