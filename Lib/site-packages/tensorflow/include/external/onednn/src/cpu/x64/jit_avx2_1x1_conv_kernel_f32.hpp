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

#ifndef CPU_X64_JIT_AVX2_1X1_CONV_KERNEL_F32_HPP
#define CPU_X64_JIT_AVX2_1X1_CONV_KERNEL_F32_HPP

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

struct jit_avx2_1x1_conv_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_1x1_conv_kernel_f32)

    jit_avx2_1x1_conv_kernel_f32(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx2>>
            postops_injector_;

    constexpr static int isa_simd_width_
            = cpu_isa_traits<avx2>::vlen / sizeof(float);
    using reg64_t = const Xbyak::Reg64;
    using ymm_t = const Xbyak::Ymm;

    reg64_t reg_bcast_data = rax;
    reg64_t reg_load_data = rsi;
    reg64_t reg_output_data = rbx;
    reg64_t aux_reg_bcast_data = rdx;
    reg64_t aux1_reg_bcast_data = abi_not_param1;
    reg64_t aux_reg_load_data = abi_param1;
    reg64_t aux_reg_output_data = rbp;
    reg64_t reg_load_loop_work = r9;
    reg64_t reg_bcast_loop_work = r10;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t load_loop_iter = r13;
    reg64_t bcast_loop_iter = r14;
    reg64_t reduce_loop_iter = r15;
    reg64_t reg_reduce_pos_flag = r8;
    reg64_t reg_output_stride = r12;
    reg64_t reg_bias_data = r12;
    reg64_t reg_diff_bias_data = bcast_loop_iter;
    reg64_t reg_tmp_output_stride = reg_bcast_data;
    reg64_t reg_tmp = aux_reg_bcast_data;
    reg64_t reg_output_stride_scale = load_loop_iter;
    reg64_t reg_long_offt = reg_bcast_data;

    constexpr static int reg64_size_ = sizeof(int64_t);
    constexpr static int reg_diff_bias_data_stack_offt = 0;
    constexpr static int reg_abi_param1_backup = 1 * reg64_size_;
    constexpr static int reg_bcast_data_off = 2 * reg64_size_;
    constexpr static int reg_dw_binary_output_off = 3 * reg64_size_;
    constexpr static int stack_space_needed = 4 * reg64_size_;

    ymm_t vreg_bcast = ymm_t(15);
    ymm_t vtmp = ymm_t(14);

    void apply_postops(
            const int load_loop_blk, const int ur, const int load_dim_tail);
    void generate_bcast_loop(int load_loop_blk);
    void generate_reduce_loop(int load_loop_blk, int ur);
    void generate_diff_bias_loop(int load_loop_blk);

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
