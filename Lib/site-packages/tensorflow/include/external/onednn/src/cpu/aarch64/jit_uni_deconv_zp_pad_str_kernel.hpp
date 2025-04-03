/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_DECONV_ZP_PAD_STR_KERNEL_HPP
#define CPU_AARCH64_JIT_UNI_DECONV_ZP_PAD_STR_KERNEL_HPP

#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct jit_conv_conf_t;

namespace zp {

struct jit_uni_deconv_zp_pad_str_call_params_t {
    const int8_t *wei;
    const int32_t *src_zero_point;
    int32_t *dst_scratchpad;
    bool last_oc_block;
};

/*
 * Compute zero point source compensation applied during filter application on
 * the padding as well as stride holes.
 *
 * zp_pad_str_compensation = conv(1, weights_s8) * zero_point_source
 *
 * output_format - dhwc
 */
class jit_uni_deconv_zp_pad_str_kernel_base_t : public jit_generator {
public:
    jit_uni_deconv_zp_pad_str_kernel_base_t(const jit_conv_conf_t &jcp);

    void operator()(const jit_uni_deconv_zp_pad_str_call_params_t *params) {
        jit_generator::operator()(params);
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_deconv_zp_pad_str_kernel_base_t);

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_uni_deconv_zp_pad_str_kernel_base_t);

    void generate() override;
    void load_addresses();
    void compute();
    virtual void compute_step(const dim_t icb_offset) = 0;
    virtual void apply_zero_point() = 0;
    virtual void store_result() = 0;
    virtual void init() = 0;

protected:
    size_t number_reserved_vmms_ = 0;
    size_t reserve_vmm();

    const jit_conv_conf_t &jcp_;

    const Xbyak_aarch64::XReg &reg_src_zp_ = x8;
    const Xbyak_aarch64::XReg &reg_wei_ = x9;
    const Xbyak_aarch64::XReg &reg_dst_ = x10;
    const Xbyak_aarch64::WReg &reg_last_oc_block_ = w12;
    const size_t tail_size_;
};

template <cpu_isa_t isa>
class jit_uni_deconv_zp_pad_str_kernel_t
    : public jit_uni_deconv_zp_pad_str_kernel_base_t {
public:
    jit_uni_deconv_zp_pad_str_kernel_t(const jit_conv_conf_t &jcp);

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;

    void init() override;
    void compute_step(const dim_t icb_offset) override;
    void apply_zero_point() override;
    void store_result() override;

    uint32_t get_next_vmm_idx();

    const TReg result_acc_;
    const TReg vmm_tmp_;
    const TReg vmm_one_bytes_;
    const TReg vmm_one_words_;

    const Xbyak_aarch64::PReg &ktail_mask_ = p2;
    dim_t current_vmm_;
};

bool should_calculate_deconv_zp_src_pad_str_comp(
        const jit_conv_conf_t &jcp) noexcept;

template <cpu_isa_t isa>
jit_uni_deconv_zp_pad_str_kernel_base_t *create_deconv_zp_pad_str_comp_ker(
        const jit_conv_conf_t &jcp);

void compute_deconv_zp_pad_str_comp_ker(const jit_conv_conf_t &jcp_,
        const bool with_groups, const memory_desc_wrapper &wei_d,
        const int8_t *wei, const int32_t *src_zp, int32_t *dst,
        jit_uni_deconv_zp_pad_str_kernel_base_t *ker);

} // namespace zp
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
