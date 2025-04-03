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

#ifndef CPU_X64_JIT_AVX512_COMMON_1X1_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_COMMON_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_common_1x1_conv_kernel : public jit_generator {
    jit_avx512_common_1x1_conv_kernel(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_1x1_conv_kernel)

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
            int nthreads, bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;

    constexpr static int isa_simd_width_
            = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;

    reg64_t reg_bcast_data = r8;
    reg64_t reg_load_data = r10;
    reg64_t reg_output_data = r9;
    reg64_t aux_reg_bcast_data = r14;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t aux_reg_load_data = r15;
    reg64_t imm_addr64 = aux_reg_load_data;
    reg64_t aux_reg_output_data = abi_not_param1;
    reg64_t reg_load_loop_work = rsi;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t reg_bcast_loop_iter = rdx;
    reg64_t reduce_loop_iter = abi_param1;
    reg64_t reg_reduce_pos_flag = rax;
    reg64_t reg_output_stride = r13;
    reg64_t reg_bias_data = r12;
    reg64_t reg_relu_ns = r13;
    reg64_t reg_bcast_loop_work = aux1_reg_bcast_data;
    reg64_t reg_load_dim_tail_mask = aux_reg_load_data;
    reg64_t reg_long_offt = reg_bcast_data;

    Xbyak::Zmm vreg_bcast = Xbyak::Zmm(31);
    Xbyak::Opmask k_load_dim_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_load_dim_tail_mask = Xbyak::Opmask(3);

    constexpr static int reg64_size_ = sizeof(int64_t);
    constexpr static int reg_bcast_loop_work_offt = 0;
    constexpr static int reg_abi_param1_backup = 1 * reg64_size_;
    constexpr static int reg_bcast_data_off = 2 * reg64_size_;
    constexpr static int reg_dw_binary_output_off = 3 * reg64_size_;
    constexpr static int stack_space_needed = 4 * reg64_size_;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    inline size_t get_output_offset(const bool is_out_layout_nxc,
            const int i_load, const int i_ur, bool ignore_dw_conv = false) {
        const size_t i_load_shift = is_out_layout_nxc
                ? jcp.load_block
                : (!ignore_dw_conv && jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim)
                        * jcp.load_block;
        const size_t i_ur_shift
                = is_out_layout_nxc ? jcp.load_dim : jcp.load_block;
        return jcp.typesize_out * (i_load * i_load_shift + i_ur * i_ur_shift);
    }

    Xbyak::Address output_ptr(
            const bool out_layout_nxc, const int i_load, const int i_ur);
    void apply_postops(const bool is_out_layout_nxc, const int load_loop_blk,
            const int ur);
    void generate() override;
    static void balance(jit_1x1_conv_conf_t &jcp);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
