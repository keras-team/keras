/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_X8S8S32X_1X1_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_X8S8S32X_1X1_CONV_KERNEL_HPP

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
struct _jit_avx512_core_x8s8s32x_1x1_conv_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_core_x8s8s32x_1x1_conv_fwd_ker_t)
    _jit_avx512_core_x8s8s32x_1x1_conv_kernel(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    constexpr static int isa_simd_width_
            = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    using Vmm_down_t =
            typename utils::conditional<std::is_same<Vmm, Xbyak::Zmm>::value,
                    Xbyak::Ymm, Xbyak::Xmm>::type;
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core, Vmm>>
            postops_injector_;

    /* register mapping */
    const Xbyak::Reg64 reg_last_load = r8;
    const Xbyak::Reg64 reg_bcast_data = r8;
    const Xbyak::Reg64 reg_ptr_scales = r8;
    const Xbyak::Reg64 reg_ptr_saturation_ubound = r8;
    const Xbyak::Reg64 reg_output_data = r9;
    const Xbyak::Reg64 reg_load_data = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r10;
    const Xbyak::Reg64 reg_reduce_loop_work = r11;
    const Xbyak::Reg64 reg_bias_data = r12;
    const Xbyak::Reg64 reg_comp_data = r12;
    const Xbyak::Reg64 reg_ptr_dst_scale = r12;
    const Xbyak::Reg64 reg_scratch = r13;
    const Xbyak::Reg64 aux_reg_bcast_data = r14;
    const Xbyak::Reg64 aux_reg_load_data = r15;
    const Xbyak::Reg64 imm_addr64 = r15;
    const Xbyak::Reg64 reg_reduce_pos_flag = rax;
    const Xbyak::Reg64 aux1_reg_bcast_data = rbx;
    const Xbyak::Reg64 reg_bcast_loop_work = rbx;
    const Xbyak::Reg64 bcast_loop_iter = rdx; // Note: Fix me
    const Xbyak::Reg64 reg_load_loop_work = rsi;
    const Xbyak::Reg64 aux_reg_output_data = abi_not_param1;
    const Xbyak::Reg64 reduce_loop_iter = abi_param1;
    // zero-point computation
    const Xbyak::Reg64 reg_zp_compensation = aux_reg_load_data; // r15
    const Xbyak::Reg64 reg_src_zero_point = aux_reg_bcast_data; // r14
    const Xbyak::Reg64 reg_dst_zero_point = reg_src_zero_point;
    const Xbyak::Reg64 reg_load_dim_tail_mask = reg_scratch;

    const Xbyak::Opmask k_load_dim_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask k_load_dim_mask_extended = Xbyak::Opmask(3);
    const Xbyak::Opmask k_load_dim_tail_mask = Xbyak::Opmask(4);
    const Xbyak::Opmask k_load_dim_tail_mask_extended = Xbyak::Opmask(5);
    const Xbyak::Opmask postops_mask = Xbyak::Opmask(6);
    const Xbyak::Opmask vmask = k7;

    const Vmm vmm_tmp = Vmm(28);
    const Vmm vmm_saturation = Vmm(28);
    const Vmm vmm_one = Vmm(29);
    const Vmm vmm_zero = Vmm(30);
    const Vmm vmm_prev_dst = Vmm(30);
    const Vmm vmm_shift = Vmm(30);
    const Vmm vmm_bcast = Vmm(31);
    /* zero-point */
    const Vmm vmm_zp = Vmm(30);
    const Vmm vmm_zp_tmp = vmm_zp;

    const Vmm vmm_dst_scale = Vmm(30);

    /* bfloat16 */
    const Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(25);
    const Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(26);
    const Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(27);
    const Xbyak::Reg64 bf16_emu_reserv_4 = imm_addr64;
    const Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(28);
    const Xbyak::Ymm ymm_store = Xbyak::Ymm(31);

    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    constexpr static int reg64_size_ = sizeof(int64_t);
    constexpr static int bcast_loop_work_off = 0;
    constexpr static int reg_bias_data_off = 1 * reg64_size_;
    constexpr static int reg_bcast_data_off = 2 * reg64_size_;
    constexpr static int reg_load_data_off = 3 * reg64_size_;
    constexpr static int reg_ptr_sum_scale_off = 4 * reg64_size_;
    constexpr static int reg_ptr_sum_zp_off = 5 * reg64_size_;
    constexpr static int reg_comp_data_off = 6 * reg64_size_;
    constexpr static int reg_zp_compensation_off = 7 * reg64_size_;
    constexpr static int reg_src_zero_point_off = 8 * reg64_size_;
    constexpr static int reg_dst_zero_point_off = 9 * reg64_size_;
    constexpr static int reg_dst_scale_off = 10 * reg64_size_;
    constexpr static int reg_abi_param1_backup = 11 * reg64_size_;
    constexpr static int stack_space_needed = 12 * reg64_size_;

    inline Vmm maybe_mask_vmm(Vmm vmm, bool mask_flag) {
        return mask_flag ? vmm | k_load_dim_mask_extended : vmm;
    }
    inline Vmm_down_t maybe_mask_vmm_down(Vmm_down_t vmm_down, bool mask_flag) {
        return mask_flag ? vmm_down | k_load_dim_mask : vmm_down;
    }
    inline Vmm_down_t vmm_store() { return Vmm_down_t(ymm_store.getIdx()); };

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, bool wraparound);

    Xbyak::Address output_ptr(const int i_load, const int i_ur);
    int vreg_accum_idx(const int load_loop_blk, int i_load, int i_ur) const;
    Vmm vreg_accum(const int load_loop_blk, int i_load, int i_ur) const;
    void apply_sum(const int load_loop_blk, const int ur,
            const bool mask_flag_in, const float *p_sum_scale,
            const int32_t *p_sum_zp);
    void apply_postops(const int load_loop_blk, const int ur,
            const bool mask_flag_in, const float *p_sum_scale,
            const int32_t *p_sum_zp);
    void generate() override;
    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
            bool mask_flag);
};

struct jit_avx512_core_x8s8s32x_1x1_conv_kernel {
    jit_avx512_core_x8s8s32x_1x1_conv_kernel(const jit_1x1_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md)
        : kernel_(nullptr) {
        int ch_block = ajcp.ic_block;
        switch (ch_block) {
            case 16:
                kernel_ = new _jit_avx512_core_x8s8s32x_1x1_conv_kernel<
                        Xbyak::Zmm>(ajcp, attr, dst_md);
                return;
            case 8:
                kernel_ = new _jit_avx512_core_x8s8s32x_1x1_conv_kernel<
                        Xbyak::Ymm>(ajcp, attr, dst_md);
                return;
            case 4:
                kernel_ = new _jit_avx512_core_x8s8s32x_1x1_conv_kernel<
                        Xbyak::Xmm>(ajcp, attr, dst_md);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() {
        if (kernel_) return kernel_->create_kernel();
        return status::out_of_memory;
    }

    ~jit_avx512_core_x8s8s32x_1x1_conv_kernel() { delete kernel_; }

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t *&src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads,
            bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    void operator()(const jit_1x1_conv_call_s *p) const { (*kernel_)(p); }
    const Xbyak::uint8 *jit_ker() const { return kernel_->jit_ker(); }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_avx512_core_x8s8s32x_1x1_conv_kernel);
    jit_generator *kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
