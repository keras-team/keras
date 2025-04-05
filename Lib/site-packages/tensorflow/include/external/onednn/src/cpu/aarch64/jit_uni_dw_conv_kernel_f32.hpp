/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
* Copyright 2021-2022 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_DW_CONV_KERNEL_F32_HPP
#define CPU_AARCH64_JIT_UNI_DW_CONV_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa>
struct jit_uni_dw_conv_fwd_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_fwd_kernel_f32)

    jit_uni_dw_conv_fwd_kernel_f32(jit_conv_conf_t ajcp)
        : jcp(ajcp), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<sve_512>(
                    this, jcp.eltwise);
    }

    ~jit_uni_dw_conv_fwd_kernel_f32() { delete eltwise_injector_; }

    jit_conv_conf_t jcp;

private:
    using reg64_t = const XReg;
    const int vlen = cpu_isa_traits<isa>::vlen;

    // dw convolution
    reg64_t reg_input = x1;
    reg64_t aux_reg_input = x2;
    reg64_t reg_kernel = x3;
    reg64_t aux_reg_kernel = x5;
    reg64_t reg_ch_blocks = x6;
    reg64_t reg_output = x7;
    reg64_t reg_bias = x8;
    reg64_t reg_kh = x9;
    reg64_t iter_kh = x10;
    reg64_t reg_oi = x11;
    reg64_t aux_reg_ch_blocks = x12;
    // fused convolution
    reg64_t reg_input_buffer_ptr = x13;
    reg64_t aux_reg_input_buffer_ptr = x14;
    reg64_t reg_iw_offset = reg_input;

    /* Temprary regs */
    reg64_t reg_tmp_imm = x15;
    reg64_t reg_kernel_stack = x16;
    reg64_t reg_input_stack = x17;
    reg64_t reg_output_stack = x27;
    reg64_t reg_bias_stack = x19;
    reg64_t reg_tmp_addr = x20;

    inline void load_src(int ur_ch_blocks, int ur_w);
    inline void compute_loop(int ur_w, int ur_ch_blocks, int pad_l, int pad_r);
    inline void ow_loop(int ur_ch_blocks);
    inline void apply_filter_unrolled(
            int ur_ch_blocks, int ur_w, int pad_l, int pad_r);
    inline void apply_activation(int ur_ch_blocks, int ur_w);
    inline void store_dst(int ur_ch_blocks, int ur_w);

    inline ZReg get_ker_reg(int idx) {
        assert(idx <= 31);
        return ZReg(idx + 0);
    }
    inline ZRegS get_ker_reg_s(int idx) {
        assert(idx <= 31);
        return ZRegS(idx + 0);
    }
    inline ZReg get_src_reg(int idx) {
        assert((idx + 1) <= 31);
        return ZReg(idx + 1);
    }
    inline ZRegS get_src_reg_s(int idx) {
        assert((idx + 1) <= 31);
        return ZRegS(idx + 1);
    }

    inline ZReg get_acc_reg(int idx) {
        assert((idx + 4) <= 31);
        return ZReg(idx + 4);
    }
    inline ZRegS get_acc_reg_s(int idx) {
        assert((idx + 4) <= 31);
        return ZRegS(idx + 4);
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

    jit_uni_eltwise_injector_f32<sve_512> *eltwise_injector_;
    void generate() override;
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_data_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_bwd_data_kernel_f32)

    jit_uni_dw_conv_bwd_data_kernel_f32(jit_conv_conf_t ajcp) : jcp(ajcp) {}
    jit_conv_conf_t jcp;

private:
    using reg64_t = const XReg;

    inline ZReg get_ker_reg(int idx) { return ZReg(idx + 0); }
    inline ZReg get_src_reg(int idx) { return ZReg(idx + 1); }
    inline ZReg get_acc_reg(int idx) { return ZReg(idx + 4); }
    inline ZRegS get_ker_reg_s(int idx) { return ZRegS(idx + 0); }
    inline ZRegS get_src_reg_s(int idx) { return ZRegS(idx + 1); }
    inline ZRegS get_acc_reg_s(int idx) { return ZRegS(idx + 4); }

    reg64_t reg_ddst = x1;
    reg64_t aux_reg_ddst = x2;
    reg64_t aux1_reg_ddst = x3;
    reg64_t reg_kernel = x5;
    reg64_t aux_reg_kernel = x6;
    reg64_t aux1_reg_kernel = x7;
    reg64_t reg_dsrc = x8;

    reg64_t reg_ur_str_w = x9;
    reg64_t reg_ch_blocks = x10;

    reg64_t iter_kh = x11;
    reg64_t iter_kw = x12;
    reg64_t reg_kh = x13;
    reg64_t reg_kw = x14;

    /* Temprary regs */
    reg64_t reg_tmp_imm = x15;
    reg64_t reg_tmp_addr = x16;

    inline void loop_body(int ur_ch_blocks);
    inline void load_ddst(int ur_ch_blocks, int ur_str_w);
    inline void apply_filter(int ur_ch_blocks, int ur_str_w);
    inline void store_dsrc(int ur_ch_blocks, int ur_str_w);

    void generate() override;
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_weights_kernel_f32 : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_bwd_weights_kernel_f32)

    jit_uni_dw_conv_bwd_weights_kernel_f32(jit_conv_conf_t ajcp) : jcp(ajcp) {}

    jit_conv_conf_t jcp;

private:
    using reg64_t = const XReg;
    int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);

    inline ZReg get_bias_reg(int idx = 0) { return ZReg(idx); }
    inline ZReg get_output_reg(int idx) { return ZReg(idx + 1); }
    inline ZReg get_input_reg(int idx) { return ZReg(idx + 5); }
    inline ZReg get_acc_reg(int idx) { return ZReg(idx + 2); }
    inline ZReg get_aux_reg() { return ZReg(0); }
    inline ZRegS get_bias_reg_s(int idx = 0) { return ZRegS(idx); }
    inline ZRegS get_output_reg_s(int idx) { return ZRegS(idx + 1); }
    inline ZRegS get_input_reg_s(int idx) { return ZRegS(idx + 5); }
    inline ZRegS get_acc_reg_s(int idx) { return ZRegS(idx + 2); }
    inline ZRegS get_aux_reg_s() { return ZRegS(0); }

    reg64_t reg_tmp_input = x1;
    reg64_t reg_tmp_output = x2;
    reg64_t reg_tmp_filter = x3;
    reg64_t reg_kh_offset = x5;

    /* parameter passed by driver into kernel */
    reg64_t reg_exec_flags = x14;

    reg64_t reg_oh_worksize = x6;
    reg64_t reg_oh = x5;

    reg64_t reg_iter_ow_blk = x7;

    reg64_t reg_kh = x8;
    reg64_t reg_kh_count = x9;

    /* Base addresses for convolution parameters. */
    reg64_t reg_input_baddr = x10;
    reg64_t reg_output_baddr = x11;
    reg64_t reg_filter_baddr = x12;
    reg64_t reg_bias_baddr = x13;

    /* Temporary regs */
    reg64_t reg_tmp_imm = x15;
    reg64_t reg_tmp_addr = x16;

    /* Micro-kernel JIT'ing, fusing 'kw' and 'ow_block' loops into unrolled FMAs
     */
    inline void compute_ow_step_unroll(
            int unroll_w, int l_pad, int pad_offset, int ow_block);

    /* JIT'ing the outer loops for the micro-kernel -> {kh, oh_block} */
    inline void compute_h_step(
            int unroll_w, int l_pad, int pad_offset, int ow_block);
    inline void compute_h_loop(
            int unroll_w, int l_pad, int pad_offset, int ow_block);

    /* Write 'width' micro-kernel JITs; depending on the padding and convolution
     * size, write a micro-kernel for the left ow-block, middle ow-block(s), and
     * right ow-block.*/
    inline void compute_ow_block_unroll();

    inline void compute_zero_filter();
    inline void load_filter();
    inline void zero_filter();
    inline void load_bias();
    inline void zero_bias();
    inline void compute_bias_step_unroll(const int unroll_w);
    inline void compute_bias_loop(const int block_size);
    inline void store_filter();
    inline void store_bias();

    void generate() override;
};
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
