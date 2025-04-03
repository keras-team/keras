/*******************************************************************************
* Copyright 2022 IBM Corporation
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

#ifndef CPU_PPC64_GEMM_GEMM_DRIVER_HPP
#define CPU_PPC64_GEMM_GEMM_DRIVER_HPP

namespace dnnl {
namespace impl {
namespace cpu {
namespace ppc64 {

dnnl_status_t cblas_gemm_s8x8s32_ppc64(int, int, char const *, dim_t, dim_t,
        dim_t, float, int8_t const *, dim_t, int8_t const *, uint8_t const *,
        dim_t, uint8_t const *, int *, float, dim_t, int const *, int);

} // namespace ppc64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif // CPU_PPC64_GEMM_GEMM_DRIVER_HPP
