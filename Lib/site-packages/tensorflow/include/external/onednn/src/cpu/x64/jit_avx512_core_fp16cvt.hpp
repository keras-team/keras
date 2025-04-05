/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_FP16CVT_HPP
#define CPU_X64_JIT_AVX512_CORE_FP16CVT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/float16.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "oneapi/dnnl/dnnl_debug.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace f16_support {
struct jit_call_t {
    void *inp;
    void *out;
    void *add;
    size_t nelems;
};
} // namespace f16_support

// performs element-by-element sum of inp and add float arrays and stores
// result to float16 out array with downconversion
struct jit_avx512_core_fp16_add_cvt_ps_to_f16_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_add_cvt_ps_to_f16)

    jit_avx512_core_fp16_add_cvt_ps_to_f16_t()
        : jit_generator(jit_name()), simd_w_(16) {
        create_kernel();
    }

    void generate() override;

    void operator()(f16_support::jit_call_t *params) const {
        jit_generator::operator()(params);
        msan_unpoison(params->out, params->nelems * sizeof(float16_t));
    }

private:
    int simd_w_;

    Xbyak::Opmask ktail_mask = k2;
    Xbyak::Zmm fp32_inp = Xbyak::Zmm(0);
    Xbyak::Zmm fp32_tmp = Xbyak::Zmm(1);

    Xbyak::Zmm one = Xbyak::Zmm(2);
    Xbyak::Zmm even = Xbyak::Zmm(3);
    Xbyak::Zmm selector = Xbyak::Zmm(4);
    Xbyak::Reg64 scratch = r15;

    Xbyak::Ymm f16_out = Xbyak::Ymm(5);

    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_add = r11;
    Xbyak::Reg64 reg_nelems = rdx;

    Xbyak::Reg64 reg64_tail = rcx;
    Xbyak::Reg32 reg32_tail = ecx;
    Xbyak::Reg8 reg8_mask_shift = cl;
    Xbyak::Reg32 reg32_mask = r8d;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
