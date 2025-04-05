/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_SCALE_PRECOMPUTE_HPP
#define CPU_X64_JIT_AVX512_CORE_SCALE_PRECOMPUTE_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/scale_utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "oneapi/dnnl/dnnl_debug.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_scale_precompute_t;

namespace scale_utils {
struct jit_call_t {
    jit_call_t(const float *src_scales, const float *wei_scales, float *scales,
            size_t nelems, size_t stride_per_groups)
        : src_scales_(src_scales)
        , wei_scales_(wei_scales)
        , scales_(scales)
        , nelems_(nelems)
        , stride_per_groups_(stride_per_groups) {}

    const void *src_scales_;
    const void *wei_scales_;
    float *scales_;
    size_t nelems_;
    size_t stride_per_groups_;
};

const float *precompute_scales(const memory_tracking::grantor_t &scratchpad,
        const float *src_scales, const float *wei_scales, dim_t IC, dim_t OC,
        const bool wei_scale_per_ic, const bool wei_scale_per_oc,
        const primitive_attr_t *attr,
        const jit_avx512_core_scale_precompute_t *const jit_scale_precompute,
        float scale_adjust_factor = 1.0f, bool req_transpose = false);
} // namespace scale_utils

struct jit_avx512_core_scale_precompute_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_scale_precompute_t)

    jit_avx512_core_scale_precompute_t(
            const primitive_attr_t *attr, const float scale_adjust_factor = 1)
        : jit_generator(jit_name())
        , attr_(attr)
        , with_wei_scales_(
                  !attr_->scales_.get(DNNL_ARG_WEIGHTS).has_default_values())
        , wei_scales_dt_(with_wei_scales_
                          ? attr_->scales_.get(DNNL_ARG_WEIGHTS).data_type_
                          : data_type::f32)
        , wei_scales_dsz_(types::data_type_size(wei_scales_dt_))
        , wei_groups_ic_(with_wei_scales_
                                  && attr_->scales_.get(DNNL_ARG_WEIGHTS).ndims_
                                          > 0
                          ? attr_->scales_.get(DNNL_ARG_WEIGHTS).group_dims_[0]
                          : 1)
        , scale_adjust_factor_(scale_adjust_factor)
        , compute_scale_factor_(scale_adjust_factor_ != 1) {}

    void generate() override;

    void operator()(scale_utils::jit_call_t *params) const {
        jit_generator::operator()(params);
        msan_unpoison(params->scales_, params->nelems_ * sizeof(float));
    }

private:
    constexpr static int simd_w_
            = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    using Vmm = typename cpu_isa_traits<avx512_core>::Vmm;

    const primitive_attr_t *attr_;
    const bool with_wei_scales_;
    const data_type_t wei_scales_dt_;
    const size_t wei_scales_dsz_;
    const size_t wei_groups_ic_;
    const float scale_adjust_factor_;
    const bool compute_scale_factor_;

    Xbyak::Reg64 reg_src_scales_ = r15;
    Xbyak::Reg64 reg_wei_scales_ = r14;
    Xbyak::Reg64 reg_dst_scales_ = r13;
    Xbyak::Reg64 reg_scale_factor_ = r12;
    Xbyak::Reg64 reg_nelems_ = r11;
    Xbyak::Reg64 reg_groups_stride_ = r10;
    Xbyak::Reg64 reg_aux_dst_scales_ = r9;
    Xbyak::Reg64 reg_tail_ = rcx;
    Xbyak::Reg32 reg_mask_ = eax;

    const Xbyak::Opmask ktail_f32_mask_ = Xbyak::Opmask(1);

    const Vmm vmm_dst_ = Vmm(0);
    const Vmm vmm_wei_scales_ = Vmm(1);
    const Vmm vmm_scale_factor_ = Vmm(2);

    void setup_mask();
    void store(const int offset_base, const bool compute_tail);
    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
            const bool mask_flag);
    void compute_scale(const int offset_base, const bool compute_tail);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_JIT_AVX512_CORE_SCALE_PRECOMPUTE_HPP
