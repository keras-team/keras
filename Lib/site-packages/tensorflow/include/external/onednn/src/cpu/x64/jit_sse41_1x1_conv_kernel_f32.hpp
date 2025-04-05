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

#ifndef CPU_X64_JIT_SSE41_1X1_CONV_KERNEL_F32_HPP
#define CPU_X64_JIT_SSE41_1X1_CONV_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_sse41_1x1_conv_kernel_f32 : public jit_generator {
    jit_sse41_1x1_conv_kernel_f32(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
            int nthreads);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse41_1x1_conv_kernel_f32)

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    static constexpr auto simd_w_ = cpu_isa_traits<sse41>::vlen / sizeof(float);
    using reg64_t = const Xbyak::Reg64;
    using xmm_t = const Xbyak::Xmm;

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
    reg64_t imm_addr64 = load_loop_iter;
    reg64_t bcast_loop_iter = r14;
    reg64_t reduce_loop_iter = r15;
    reg64_t reg_reduce_pos_flag = r8;
    reg64_t reg_output_stride = r12;
    reg64_t reg_bias_data = r12;
    reg64_t reg_diff_bias_data = bcast_loop_iter;

    constexpr static int reg64_size_ = sizeof(int64_t);
    constexpr static int reg_diff_bias_data_stack_offt = 0;
    constexpr static int reg_abi_param1_backup = 1 * reg64_size_;
    constexpr static int reg_dw_binary_output_off = 2 * reg64_size_;
    constexpr static int stack_space_needed = 3 * reg64_size_;

    xmm_t reg_bcast = xmm_t(15);

    std::unique_ptr<injector::jit_uni_postops_injector_t<sse41>>
            postops_injector_;

    void generate_bcast_loop(int load_loop_blk);
    void generate_reduce_loop(int load_loop_blk, int ur);
    void generate_diff_bias_loop(int load_loop_blk);

    void generate() override;

    void apply_postops(const int load_loop_blk, const int ur);
    size_t get_fwd_output_ptr_l_off(
            int i, int j, int n, bool ignore_dw_conv) const;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
