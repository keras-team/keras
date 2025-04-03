/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_X8S8S32X_1X1_CONV_KERNEL_HPP
#define CPU_X64_JIT_UNI_X8S8S32X_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, typename Vmm>
struct _jit_uni_x8s8s32x_1x1_conv_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_uni_x8s8s32x_1x1_conv_kernel)
    _jit_uni_x8s8s32x_1x1_conv_kernel(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    int get_tail_size() { return jcp.oc_without_padding % jcp.oc_block; }

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;

    enum {
        ker_max_reg_idx = 13,
    };
    const Xbyak::Reg64 reg_bcast_data = r8;
    const Xbyak::Reg64 reg_ptr_scales = r8;
    const Xbyak::Reg64 reg_output_data = r9;
    const Xbyak::Reg64 reg_load_data = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r10;
    const Xbyak::Reg64 reg_ptr_sum_zp = rdx;
    const Xbyak::Reg64 reg_reduce_loop_work = r11;
    const Xbyak::Reg64 reg_bias_data = r12;
    const Xbyak::Reg64 reg_comp_data = r12;
    const Xbyak::Reg64 reg_ptr_dst_scale = r12;
    const Xbyak::Reg64 reg_init_bcast = r13;
    const Xbyak::Reg64 reg_store_bcast = r13;
    const Xbyak::Reg64 reg_reduce_loop_iter = r13;
    const Xbyak::Reg64 aux_reg_bcast_data = r14;
    const Xbyak::Reg64 aux_reg_load_data = r15;
    const Xbyak::Reg64 aux_reg_saturation = r15;
    const Xbyak::Reg64 reg_reduce_pos_flag = rax;
    const Xbyak::Reg64 aux1_reg_bcast_data = rbx;
    const Xbyak::Reg64 reg_bcast_loop_work = rbx;
    const Xbyak::Reg64 reg_bcast_loop_iter = rdx;
    const Xbyak::Reg64 reg_load_loop_work = rsi;
    const Xbyak::Reg64 aux_reg_output_data = abi_not_param1;
    // zero-point computation
    const Xbyak::Reg64 reg_zp_compensation = aux_reg_load_data; // r15
    const Xbyak::Reg64 reg_src_zero_point = aux_reg_bcast_data; // r14
    const Xbyak::Reg64 reg_dst_zero_point = reg_src_zero_point;

    const Vmm vmm_tmp = Vmm(3);
    const Vmm vmm_one = Vmm(2);
    const Vmm vmm_zero = Vmm(1);
    const Vmm vmm_shift = Vmm(1);
    const Vmm vmm_bcast = Vmm(0);
    const Vmm vmm_saturation = Vmm(0);
    /* used during scale section of store_output */
    const Vmm vmm_scale = Vmm(1);
    /* used during post_op sum section of store_output */
    const Vmm vmm_prev_dst = Vmm(1);
    /* used during bias section of store_output */
    const Vmm vmm_comp = Vmm(0); // only for signed input
    const Vmm vmm_bias = Vmm(3);
    /* zero-point */
    const Vmm vmm_zp = Vmm(1);
    const Vmm vmm_zp_comp = Vmm(2);
    /* dst scale */
    const Vmm vmm_dst_scale = Vmm(1);

    constexpr static int simd_w = isa == avx2 ? 8 : 4;
    constexpr static int reg64_size = sizeof(int64_t);
    constexpr static int bcast_loop_work_off = 0;
    constexpr static int reg_bias_data_off = 1 * reg64_size;
    constexpr static int reg_bcast_data_off = 2 * reg64_size;
    constexpr static int reg_load_data_off = 3 * reg64_size;
    constexpr static int reg_ptr_sum_scale_off = 4 * reg64_size;
    constexpr static int reg_bcast_loop_iter_off = 5 * reg64_size;
    constexpr static int reg_comp_data_off = 6 * reg64_size;
    constexpr static int reg_zp_compensation_off = 7 * reg64_size;
    constexpr static int reg_src_zero_point_off = 8 * reg64_size;
    constexpr static int reg_dst_zero_point_off = 9 * reg64_size;
    constexpr static int reg_dst_scale_off = 10 * reg64_size;
    constexpr static int stack_space_needed = 11 * reg64_size;

    int vreg_accum_idx(
            const int load_loop_blk, const int i_load, const int i_ur);
    Vmm vreg_accum(const int load_loop_blk, const int i_load, const int i_ur);
    int output_ptr(const int i_load, const int i_ur);
    void bcast_loop(int load_loop_blk);
    void apply_sum(const int ur, const int load_loop_blk,
            const bool mask_flag_in, const float *p_sum_scale,
            const int32_t *p_sum_zp);
    void apply_postops(const int ur, const int load_loop_blk,
            const bool mask_flag_in, const float *p_sum_scale,
            const int32_t *p_sum_zp);
    void reduce_loop(int load_loop_blk, int ur, bool wraparound);

    void generate() override;
    void cvt2ps(data_type_t type_in, const Vmm &vmm_in, const Xbyak::Reg64 &reg,
            int offset, int load_size);
};

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_1x1_conv_kernel {

    jit_uni_x8s8s32x_1x1_conv_kernel(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md)
        : kernel_(nullptr) {

        switch (isa) {
            case avx2:
                kernel_ = new jit_avx2_x8s8s32x_1x1_conv_kernel(
                        ajcp, attr, dst_md);
                return;
            case sse41:
                kernel_ = new jit_sse41_x8s8s32x_1x1_conv_kernel(
                        ajcp, attr, dst_md);
                return;
            default: assert(!"Current ISA is not supported!");
        }
    }

    status_t create_kernel() {
        if (kernel_) return kernel_->create_kernel();
        return status::out_of_memory;
    }

    ~jit_uni_x8s8s32x_1x1_conv_kernel() { delete kernel_; }

    void operator()(const jit_1x1_conv_call_s *p) const { (*kernel_)(p); }

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
            primitive_attr_t &attr, int nthreads, bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    using jit_sse41_x8s8s32x_1x1_conv_kernel
            = _jit_uni_x8s8s32x_1x1_conv_kernel<sse41, Xbyak::Xmm>;
    using jit_avx2_x8s8s32x_1x1_conv_kernel
            = _jit_uni_x8s8s32x_1x1_conv_kernel<avx2, Xbyak::Ymm>;

    constexpr static int simd_w = isa == avx2 ? 8 : 4;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_uni_x8s8s32x_1x1_conv_kernel);
    jit_generator *kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
