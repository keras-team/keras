/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_JIT_GEMM_X8S8S32X_CONV_ZP_SRC_PAD_COMP_HPP
#define CPU_X64_JIT_GEMM_X8S8S32X_CONV_ZP_SRC_PAD_COMP_HPP

#include <functional>

#include "common/c_types_map.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct conv_gemm_conf_t;

namespace x64 {

class jit_generator;

namespace gemm_x8s8s32x_convolution_utils {

struct jit_gemm_x8s8s32x_zp_pad_comp_helper {
    jit_gemm_x8s8s32x_zp_pad_comp_helper(jit_generator *host,
            const conv_gemm_conf_t &jcp, const Xbyak::Reg64 &reg_zp_pad_comp,
            const Xbyak::Reg64 &reg_zp_pad_comp_temp,
            const Xbyak::Reg8 &should_apply_zp_src_pad, const dim_t ndims);

public:
    void init(const dim_t off_w, const dim_t off_h, const dim_t off_w_size,
            const dim_t off_w_off, const dim_t off_zp_pad_com_base_off,
            const dim_t off_g_oc_offset_prologue, const dim_t off_g_oc_offset,
            const dim_t off_zp_src_pad_com_d_offset,
            const dim_t off_should_apply_zp_src_pad_comp_d);
    void fin();
    void load_next_point_zp_src_comp_pad_addr();
    void zp_src_comp_pad_operation(
            const std::function<void(const Xbyak::Reg64 &)> &op);
    struct zp_src_pad_com_d {
        bool should_apply_pad_comp_d;
        dim_t offset;
    };

    zp_src_pad_com_d calculate_zp_src_pad_com_d(const dim_t d_off) const;

private:
    enum bound { upper, lower };

    dim_t calculate_lower_bound_dim(const dim_t begin_comp_pad) const noexcept;
    dim_t calculate_upper_bound_dim(
            const dim_t output_size, const dim_t end_comp_pad) const noexcept;

    void set_up_initial_args(const dim_t off_w, const dim_t off_h,
            const dim_t off_w_size, const dim_t off_w_off,
            const dim_t off_zp_pad_com_base_off,
            const dim_t off_g_oc_offset_prologue, const dim_t off_g_oc_offset,
            const dim_t off_zp_src_pad_com_d_offset,
            const dim_t off_should_apply_zp_src_pad_comp_d);
    void check_bound(const Xbyak::Reg64 &reg_dim,
            const Xbyak::Address &result_addr, const dim_t bound_value,
            const bound bound_kind);
    void load_zp_src_comp_pad_addr_if_needed(const Xbyak::Address &g_oc_offset);
    void calculate_zp_src_comp_pad_effective_addr(
            const Xbyak::Address &g_oc_offset);
    void get_zp_pad_com_dim(const Xbyak::Address &dim_under_lower_bound,
            const Xbyak::Address &dim_over_eq_upper_bound,
            const dim_t begin_pad, dim_t mid_pad, const dim_t end_pad,
            const dim_t out_dim_size, const Xbyak::Address &out_point_dim,
            const Xbyak::Address &result);
    void should_apply_zp_src_pad();
    void next_point();

    jit_generator *const host_;
    const conv_gemm_conf_t &jcp_;
    const Xbyak::Address w_addr_;
    const Xbyak::Address h_addr_;
    const Xbyak::Address w_size_addr_;
    const Xbyak::Address w_off_addr_;
    const Xbyak::Address zp_pad_com_h_;
    const Xbyak::Address zp_pad_com_w_;
    const Xbyak::Address zp_pad_com_base_;
    const Xbyak::Address g_oc_offset_prologue_;
    const Xbyak::Address g_oc_offset_;
    const Xbyak::Address zp_pad_com_d_offset_;

    const Xbyak::Address h_under_lower_bound_;
    const Xbyak::Address h_over_eq_upper_bound_;
    const Xbyak::Address w_under_lower_bound_;
    const Xbyak::Address w_over_eq_upper_bound_;
    const Xbyak::Address should_apply_zp_src_pad_comp_d_;
    const Xbyak::Reg8 &should_apply_zp_src_pad_;

    const dim_t lower_h_bound_;
    const dim_t upper_h_bound_;
    const dim_t lower_w_bound_;
    const dim_t upper_w_bound_;
    const dim_t lower_d_bound_;
    const dim_t upper_d_bound_;

    const bool with_zp_pad_com_d_;
    const bool with_zp_pad_com_h_;

    const Xbyak::Reg64 &reg_zp_pad_comp_;
    const Xbyak::Reg64 &reg_zp_pad_comp_tmp_;
    // 10 * 4 (qword_size) + 5 * 1 (byte size) = 85
    // 85 aligned to 4 = 88
    static constexpr dim_t reserved_stack_size_ = 88;
};

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
