/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_AMX_1X1_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_AMX_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_amx_1x1_fwd_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_1x1_fwd_kernel_t)

    jit_avx512_core_amx_1x1_fwd_kernel_t(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_t &dst_md);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    // Tile-registers decomposition
    enum { C_BASE = 0, W_BASE = 6, I_BASE = 4 };

    void tile_configure(char *tcgf_buff);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    constexpr static int isa_simd_width_
            = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;

    enum {
        zmm_idx_limit_bf16 = 29,
        zmm_idx_limit_int8 = 27,
    };

    int row_count_ = 0;
    int buf_count_ = 0;
    bool is_store_done_ = false;
    bool is_buffer_empty_ = true;
    bool check_last_sb_ = false;
    bool last_oc_block_flag_ = false;

    /* data regs */
    const Xbyak::Reg64 inp_ptr = r15;
    const Xbyak::Reg64 wei_ptr = r14;
    const Xbyak::Reg64 out_ptr = r13;
    const Xbyak::Reg64 wsp_ptr = r12;

    const Xbyak::Reg64 reg_bias = r11;
    const Xbyak::Reg64 reg_ptr_scales = r10;
    const Xbyak::Reg64 reg_ptr_sum_scale = r9;
    const Xbyak::Reg64 reg_ptr_sum_zp = rax;
    const Xbyak::Reg64 aux_reg_saturation = reg_ptr_sum_scale;
    const Xbyak::Reg64 reg_last_h = r8;

    const Xbyak::Reg64 stride_seq = rbx;
    const Xbyak::Reg64 stride_nhwc = rsi;
    const Xbyak::Reg64 reg_tmp = abi_not_param1;

    const Xbyak::Reg64 reg_oc_blocks = rdx;
    const Xbyak::Reg64 reg_is_osb = rsi;
    const Xbyak::Reg64 reg_postop = abi_not_param1;
    const Xbyak::Reg64 reg_scratch = reg_bias;
    const Xbyak::Reg64 reg_tilebuff = reg_ptr_scales;
    /* zero-point */
    const Xbyak::Reg64 reg_zp_compensation = reg_last_h;
    const Xbyak::Reg64 reg_src_zero_point = reg_oc_blocks;
    const Xbyak::Reg64 reg_dst_zero_point = rax;

    /* scale */
    const Xbyak::Reg64 reg_ptr_dst_scale = reg_ptr_scales;

    const Xbyak::Zmm zmm_bias = zmm31;
    const Xbyak::Zmm zmm_saturation = zmm_bias;
    const Xbyak::Zmm zmm_zero = zmm30;
    const Xbyak::Zmm zmm_prev_dst = zmm29;
    const Xbyak::Zmm zmm_sum_zp = zmm26;
    /* zero-point */
    const Xbyak::Zmm zmm_zp = zmm29;
    const Xbyak::Zmm zmm_src_zp = zmm28;
    const Xbyak::Zmm zmm_dst_zp = zmm27;

    const Xbyak::Reg64 bin_injector_helper_reg_1 = r14;
    const Xbyak::Reg64 bin_injector_helper_reg_2 = r15;
    const Xbyak::Reg64 bin_injector_helper_reg_3 = r11;

    const Xbyak::Opmask ktail_mask = k2;

    bool is_bf16() const;

    void init_runtime_counters();

    int get_out_tensor(int h, int i) const;
    int get_inp_tensor(int h) const;
    int get_wei_tensor(int i) const;
    int get_ic_tail() const;

    size_t out_h_shift() const;
    size_t out_w_shift() const;
    size_t inp_offset(int ih, int iw, int icb) const;
    size_t out_row_offset(int h, int w, int ocb) const;

    void prepare_output();

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm ymm_in,
            const Xbyak::Operand &op, bool mask_flag);
    Xbyak::Zmm zmm_out(const int idx) {
        const int upper_limit
                = is_bf16() ? zmm_idx_limit_bf16 : zmm_idx_limit_int8;
        assert(upper_limit > idx);
        MAYBE_UNUSED(upper_limit);
        return Xbyak::Zmm(idx);
    }
    Xbyak::Zmm zmm_mask(
            const Xbyak::Zmm zmm_in, bool mask_flag, bool store = false);
    Xbyak::Ymm ymm_mask(
            const Xbyak::Ymm ymm_in, bool mask_flag, bool store = false);

    void update_buffer_pointers();
    void interleave_store();
    void apply_sum(const Xbyak::Zmm zmm_out, const float *p_sum_scale,
            const int32_t *p_sum_zp, const Xbyak::Address &addr,
            const bool mask_flag);
    void apply_postops(const Xbyak::Zmm zmm_out, const float *p_sum_scale,
            const int32_t *p_sum_zp, const Xbyak::Address &addr,
            const size_t off, const bool mask_flag);
    static bool is_fast_postops(const jit_conv_conf_t &jcp);
    void store_output_vectors_int8(int ocb, int osb);
    void store_output_vector_int8(
            const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    inline void store_output_ymm_bf16(
            const int idx, const Xbyak::Address &addr, const bool mask_flag);
    void store_output_vectors_bf16(int ocb, int osb);
    void store_output_vector_bf16(
            const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    void store_output_vectors(int ocb, int osb);
    void store_output_vector(const Xbyak::Zmm zmm_out, int ocb, int h, int w);
    void store_output(bool do_store, bool is_tail);
    void icb_loop(bool do_store);
    void osb_loop(int nb_os = 1);

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
