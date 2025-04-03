/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_X64_GEMM_S8X8S32_JIT_AVX512_CORE_GEMV_S8X8S32_HPP
#define CPU_X64_GEMM_S8X8S32_JIT_AVX512_CORE_GEMV_S8X8S32_HPP

#include <cstdint>

#include "cpu/x64/gemm/gemm_info.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename T>
int jump_to_gemv_s8x8s32(T *arg);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_S8X8S32_JIT_AVX512_CORE_GEMV_S8X8S32_HPP
