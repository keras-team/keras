/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef ONEAPI_DNNL_DNNL_THREADPOOL_H
#define ONEAPI_DNNL_DNNL_THREADPOOL_H

#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_threadpool_interop
/// @{

/// Creates an execution stream with specified threadpool.
///
/// @sa @ref dev_guide_threadpool
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param threadpool Pointer to an instance of a C++ class that implements
///     dnnl::threapdool_iface interface.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_threadpool_interop_stream_create(
        dnnl_stream_t *stream, dnnl_engine_t engine, void *threadpool);

/// Returns a threadpool to be used by the execution stream.
///
/// @sa @ref dev_guide_threadpool
///
/// @param astream Execution stream.
/// @param threadpool Output pointer to an instance of a C++ class that
///     implements dnnl::threapdool_iface interface. Set to NULL if the
///     stream was created without threadpool.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_threadpool_interop_stream_get_threadpool(
        dnnl_stream_t astream, void **threadpool);

/// Sets the maximum concurrency assumed by oneDNN when outside a
/// parallel call.
///
/// @param max_concurrency The maximum concurrency assumed by oneDNN
/// when outside a parallel call. This is a threadlocal setting.
/// @returns #dnnl_success on success and a status describing the
/// error otherwise.
dnnl_status_t DNNL_API dnnl_threadpool_interop_set_max_concurrency(
        int max_concurrency);

/// Gets the maximum concurrency assumed by oneDNN when outside a
/// parallel call.
///
/// @param max_concurrency The maximum concurrency assumed by oneDNN
/// when outside a parallel call. This is a threadlocal setting.
/// @returns #dnnl_success on success and a status describing the
/// error otherwise.
dnnl_status_t DNNL_API dnnl_threadpool_interop_get_max_concurrency(
        int *max_concurrency);

/// @copydoc dnnl_sgemm()
/// @param threadpool A pointer to a threadpool interface (only when built with
///     the THREADPOOL CPU runtime).
dnnl_status_t DNNL_API dnnl_threadpool_interop_sgemm(char transa, char transb,
        dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, float alpha, const float *A,
        dnnl_dim_t lda, const float *B, dnnl_dim_t ldb, float beta, float *C,
        dnnl_dim_t ldc, void *threadpool);

/// @copydoc dnnl_gemm_u8s8s32()
/// @param threadpool A pointer to a threadpool interface (only when built with
///     the THREADPOOL CPU runtime).
dnnl_status_t DNNL_API dnnl_threadpool_interop_gemm_u8s8s32(char transa,
        char transb, char offsetc, dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K,
        float alpha, const uint8_t *A, dnnl_dim_t lda, uint8_t ao,
        const int8_t *B, dnnl_dim_t ldb, int8_t bo, float beta, int32_t *C,
        dnnl_dim_t ldc, const int32_t *co, void *threadpool);

/// @copydoc dnnl_gemm_s8s8s32()
/// @param threadpool A pointer to a threadpool interface (only when built with
///     the THREADPOOL CPU runtime).
dnnl_status_t DNNL_API dnnl_threadpool_interop_gemm_s8s8s32(char transa,
        char transb, char offsetc, dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K,
        float alpha, const int8_t *A, dnnl_dim_t lda, int8_t ao,
        const int8_t *B, dnnl_dim_t ldb, int8_t bo, float beta, int32_t *C,
        dnnl_dim_t ldc, const int32_t *co, void *threadpool);

/// @} dnnl_api_threadpool_interop

/// @} dnnl_api_interop

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif
