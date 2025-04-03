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

#ifndef CPU_GEMM_GEMM_PACK_HPP
#define CPU_GEMM_GEMM_PACK_HPP

#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_types.h"

#include "common/bfloat16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

bool pack_sgemm_supported();
bool pack_gemm_bf16bf16f32_supported();

dnnl_status_t DNNL_API sgemm_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack = nullptr);

dnnl_status_t DNNL_API gemm_bf16bf16f32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack = nullptr);

dnnl_status_t DNNL_API gemm_s8u8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack = nullptr);

dnnl_status_t DNNL_API gemm_s8s8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack = nullptr);

dnnl_status_t DNNL_API sgemm_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const float *src, float *dst);

dnnl_status_t DNNL_API gemm_bf16bf16f32_pack(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb,
        const bfloat16_t *src, bfloat16_t *dst);

dnnl_status_t DNNL_API gemm_s8u8s32_pack(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, const void *src,
        void *dst);

dnnl_status_t DNNL_API gemm_s8s8s32_pack(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, const void *src,
        void *dst);

dnnl_status_t DNNL_API sgemm_compute(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *A,
        const dim_t *lda, const float *B, const dim_t *ldb, const float *beta,
        float *C, const dim_t *ldc);

dnnl_status_t DNNL_API gemm_bf16bf16f32_compute(const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const bfloat16_t *A, const dim_t *lda, const bfloat16_t *B,
        const dim_t *ldb, const float *beta, float *C, const dim_t *ldc);

dnnl_status_t DNNL_API gemm_s8u8s32_compute(const char *transa,
        const char *transb, const char *offsetc, const dim_t *M, const dim_t *N,
        const dim_t *K, const int8_t *A, const dim_t *lda, const uint8_t *B,
        const dim_t *ldb, const float *beta, int32_t *C, const dim_t *ldc,
        const int32_t *co);

dnnl_status_t DNNL_API gemm_s8s8s32_compute(const char *transa,
        const char *transb, const char *offsetc, const dim_t *M, const dim_t *N,
        const dim_t *K, const int8_t *A, const dim_t *lda, const int8_t *B,
        const dim_t *ldb, const float *beta, int32_t *C, const dim_t *ldc,
        const int32_t *co);

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_GEMM_GEMM_PACK_HPP
