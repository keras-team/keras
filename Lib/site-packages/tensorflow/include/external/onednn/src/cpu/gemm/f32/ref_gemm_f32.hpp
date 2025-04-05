/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_GEMM_F32_REF_GEMM_F32_HPP
#define CPU_GEMM_F32_REF_GEMM_F32_HPP

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <typename data_t>
dnnl_status_t ref_gemm(const char *transa, const char *transb, const dim_t *M,
        const dim_t *N, const dim_t *K, const data_t *alpha, const data_t *A,
        const dim_t *lda, const data_t *B, const dim_t *ldb, const data_t *beta,
        data_t *C, const dim_t *ldc, const data_t *bias);
}
} // namespace impl
} // namespace dnnl

#endif // CPU_GEMM_F32_REF_GEMM_F32_HPP
