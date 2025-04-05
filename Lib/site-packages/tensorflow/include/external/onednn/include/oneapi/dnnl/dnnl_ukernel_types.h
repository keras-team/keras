/*******************************************************************************
* Copyright 2024 Intel Corporation
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

/// @file
/// ukernel C API types definitions

#ifndef ONEAPI_DNNL_DNNL_UKERNEL_TYPES_H
#define ONEAPI_DNNL_DNNL_UKERNEL_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "oneapi/dnnl/dnnl_types.h"

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_ukernel
/// @{

#ifdef DNNL_EXPERIMENTAL_UKERNEL
/// @addtogroup dnnl_api_ukernel_brgemm
/// @{

/// @struct dnnl_brgemm
/// An opaque structure to describe a brgemm ukernel.
struct dnnl_brgemm;

/// A brgemm ukernel handle.
typedef struct dnnl_brgemm *dnnl_brgemm_t;

/// A constant brgemm ukernel handle.
typedef const struct dnnl_brgemm *const_dnnl_brgemm_t;

/// @struct dnnl_brgemm_pack_B
/// An opaque structure to describe a brgemm ukernel packing B routine.
struct dnnl_brgemm_pack_B;

/// A brgemm ukernel packing B routine handle.
typedef struct dnnl_brgemm_pack_B *dnnl_brgemm_pack_B_t;

/// A constant brgemm ukernel packing B routine handle.
typedef const struct dnnl_brgemm_pack_B *const_dnnl_brgemm_pack_B_t;

/// @} dnnl_api_ukernel_brgemm
#endif

/// @} dnnl_api_ukernel

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif /* ONEAPI_DNNL_DNNL_UKERNEL_TYPES_H */
