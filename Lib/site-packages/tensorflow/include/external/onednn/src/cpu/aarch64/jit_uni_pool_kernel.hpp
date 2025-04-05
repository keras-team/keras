/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
* Copyright 2018 YANDEX LLC
* Copyright 2020-2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_POOL_KERNEL_HPP
#define CPU_AARCH64_JIT_UNI_POOL_KERNEL_HPP

#include <cfloat>
#include <functional>
#include <memory>

#include "common/memory_tracking.hpp"

#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa>
struct jit_uni_pool_kernel : public jit_generator {

    jit_uni_pool_kernel(
            const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md);
    jit_pool_conf_t jpp;
    ~jit_uni_pool_kernel();

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_kernel)

    static status_t init_conf(jit_pool_conf_t &jbp,
            memory_tracking::registrar_t &scratchpad, primitive_attr_t &attr,
            const pooling_pd_t *ppd);

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    int vmm_idx_upper_bound() const noexcept { return 31; }

    int reg_idx(int idx) const noexcept { return vmm_idx_upper_bound() - idx; }

    VReg xreg(int idx) const noexcept { return VReg(reg_idx(idx)); }
    ZReg yreg(int idx) const noexcept { return ZReg(reg_idx(idx)); }
    ZReg zreg(int idx) const noexcept { return ZReg(reg_idx(idx)); }
    TReg vreg(int idx) const noexcept { return TReg(reg_idx(idx)); }

    VReg vmm_mask = VReg(0);
    ZReg ymm_tmp_1 = ZReg(0);
    TRegS vmm_tmp_1 = TRegS(0);

    TReg vmm_c_tail_mask = TReg(2);

    VReg xmm_ker_area_h = VReg(2);
    VReg xmm_one = VReg(2);
    VReg xmm_tmp = VReg(3);

    TRegS vmm_ker_area_h = TRegS(2);
    TRegS vmm_one = TRegS(2);
    TReg vmm_tmp = TReg(3);
    ZReg ymm_tmp = ZReg(3);

    TRegS vmm_k_offset = TRegS(1);

    inline uint32_t reg_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? 4 : 1;
        } else
            return 4;
    }

    ZReg z_tmp0 = z4;

    PReg k_c_tail_mask_s = p4;
    PReg k_c_tail_mask_s_not = p2;
    PReg k_c_tail_mask_b = p7;
    PReg k_mask_cvt = p5;
    PReg k_store_mask = p6;

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_all_zero = p1;
    PReg p_tmp0 = p3;

    using xreg_t = const XReg;
    xreg_t reg_param = x0;
    xreg_t reg_input = x4;
    xreg_t aux_reg_input = x5;
    xreg_t reg_index = x10;
    xreg_t reg_output = x12;
    xreg_t reg_kd_pad_shift = x13;

    xreg_t kj = x14;
    xreg_t oi_iter = x15;
    xreg_t reg_kh = x7;
    const WReg reg_k_shift = w3;
    xreg_t tmp_gpr = x6;
    xreg_t reg_ker_area_h = x2;
    xreg_t reg_nbc = x1;

    xreg_t reg_zero_ptr = x5;
    xreg_t reg_zero_id = x13;
    xreg_t reg_zero_ih = x14;
    xreg_t aux_reg_zero_ih = x15;
    xreg_t ki = x12;
    xreg_t aux_reg_input_d = x4;

    int prev_kw;

    void prepare_tail_mask();
    void push_vmm_val(const int idx);
    void pop_vmm_val(const int idx);
    void load(const int idx, const xreg_t &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);
    void store(const int idx, const xreg_t &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void avg_step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_fwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_bwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);

    void zero_diff_src(int ur_bc, bool with_c_tail_proccessing);

    void step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing) {
        if (jpp.alg == alg_kind::pooling_max) {
            if (jpp.is_backward)
                max_step_bwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
            else
                max_step_fwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
        } else
            avg_step(ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
    }

    void generate() override;

    void apply_postops(int ur_bc, int ur_w, int c_block,
            const std::function<bool(int)> &is_tail_predicate);

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
