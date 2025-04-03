/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef CPU_X64_GEMM_AMX_JIT_AVX512_CORE_AMX_GEMM_KERN_HPP
#define CPU_X64_GEMM_AMX_JIT_AVX512_CORE_AMX_GEMM_KERN_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_avx512_core_amx_gemm_kern : public jit_generator {
public:
    jit_avx512_core_amx_gemm_kern(
            int typea, int typeb, int typec, int betaZero);
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_gemm_kern);

protected:
    void generate() override ATTRIBUTE_OPTIMIZE;
    const int typea;
    const int typeb;
    const int typec;
    const int isBetaZero;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_AMX_JIT_AVX512_CORE_AMX_GEMM_KERN_HPP
