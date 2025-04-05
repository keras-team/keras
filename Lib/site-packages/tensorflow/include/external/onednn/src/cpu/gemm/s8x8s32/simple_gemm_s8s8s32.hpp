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

#ifndef CPU_GEMM_S8X8S32_SIMPLE_GEMM_S8S8S32_HPP
#define CPU_GEMM_S8X8S32_SIMPLE_GEMM_S8S8S32_HPP

#include <cstdint>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

dnnl_status_t simple_gemm_s8s8s32(const char *transA, const char *transB,
        const char *offsetC, const dim_t *m, const dim_t *n, const dim_t *k,
        const float *alpha, const int8_t *a, const dim_t *lda, const int8_t *oa,
        const int8_t *b, const dim_t *ldb, const int8_t *ob, const float *beta,
        int32_t *c, const dim_t *ldc, const int32_t *oc);
}
} // namespace impl
} // namespace dnnl

#endif // CPU_GEMM_S8X8S32_SIMPLE_GEMM_S8S8S32_HPP
