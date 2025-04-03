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

#ifndef ONEAPI_DNNL_DNNL_SYCL_TYPES_H
#define ONEAPI_DNNL_DNNL_SYCL_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_sycl_interop
/// @{

/// Memory allocation kind.
typedef enum {
    /// USM (device, shared, host, or unknown) memory allocation kind - default.
    dnnl_sycl_interop_usm,
    /// Buffer memory allocation kind.
    dnnl_sycl_interop_buffer,
} dnnl_sycl_interop_memory_kind_t;

/// @} dnnl_api_sycl_interop

/// @} dnnl_api_interop

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif
