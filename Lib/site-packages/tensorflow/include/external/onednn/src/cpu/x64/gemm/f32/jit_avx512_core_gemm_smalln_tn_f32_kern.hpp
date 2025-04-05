/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_GEMM_F32_JIT_AVX512_CORE_GEMM_SMALLN_TN_F32_KERN_HPP
#define CPU_X64_GEMM_F32_JIT_AVX512_CORE_GEMM_SMALLN_TN_F32_KERN_HPP

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename a_type, typename b_type, typename c_type>
struct gemm_info_t;

template <typename a_type, typename b_type, typename c_type>
dnnl_status_t jump_to_gemm_smalln_tn(
        const gemm_info_t<a_type, b_type, c_type> *arg);

dnnl_status_t jit_avx512_core_gemm_smalln_tn_f32(const char *transa,
        const char *transb, const dim_t *p_m, const dim_t *p_n,
        const dim_t *p_k, const float *p_alpha, const float *A,
        const dim_t *p_lda, const float *B, const dim_t *p_ldb,
        const float *p_beta, float *C, const dim_t *p_ldc);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_GEMM_F32_JIT_AVX512_CORE_GEMM_SMALLN_TN_F32_KERN_HPP
