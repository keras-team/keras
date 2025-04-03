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

#ifndef ONEAPI_DNNL_DNNL_GRAPH_SYCL_H
#define ONEAPI_DNNL_DNNL_GRAPH_SYCL_H

#include "oneapi/dnnl/dnnl_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_graph_api
/// @{

/// @addtogroup dnnl_graph_api_interop
/// @{

/// @addtogroup dnnl_graph_api_sycl_interop
/// @{

/// Allocation call-back function interface for SYCL. SYCL allocator should be
/// used for SYCL runtime and host allocator should be used for non-SYCL. The
/// call-back should return a USM device memory pointer.
typedef void *(*dnnl_graph_sycl_allocate_f)(
        size_t size, size_t alignment, const void *dev, const void *context);

/// Deallocation call-back function interface for SYCL. SYCL allocator should be
/// used for SYCL runtime and host allocator should be used for non-SYCL. The
/// call-back should deallocate a USM device memory returned by
/// #dnnl_graph_sycl_allocate_f.
typedef void (*dnnl_graph_sycl_deallocate_f)(
        void *buf, const void *dev, const void *context, void *event);

/// Creates an allocator with the given allocation and deallocation call-back
/// function pointers.
///
/// @param allocator Output allocator
/// @param sycl_malloc A pointer to SYCL malloc function
/// @param sycl_free A pointer to SYCL free function
/// @returns #dnnl_success on success and a status describing the
///     error otherwise.
dnnl_status_t DNNL_API dnnl_graph_sycl_interop_allocator_create(
        dnnl_graph_allocator_t *allocator,
        dnnl_graph_sycl_allocate_f sycl_malloc,
        dnnl_graph_sycl_deallocate_f sycl_free);

/// This API is a supplement for existing onednn engine API.
dnnl_status_t DNNL_API dnnl_graph_sycl_interop_make_engine_with_allocator(
        dnnl_engine_t *engine, const void *device, const void *context,
        const_dnnl_graph_allocator_t alloc);

/// Execute a compiled partition with sycl runtime.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param stream The stream used for execution
/// @param num_inputs The number of input tensors
/// @param inputs A list of input tensors
/// @param num_outputs The number of output tensors
/// @param outputs A non-empty list of output tensors
/// @param deps Optional handle of list with `sycl::event` dependencies.
/// @param sycl_event The handle of sycl event.
/// @returns #dnnl_success on success and a status describing the
///     error otherwise.
dnnl_status_t DNNL_API dnnl_graph_sycl_interop_compiled_partition_execute(
        const_dnnl_graph_compiled_partition_t compiled_partition,
        dnnl_stream_t stream, size_t num_inputs,
        const_dnnl_graph_tensor_t *inputs, size_t num_outputs,
        const_dnnl_graph_tensor_t *outputs, const void *deps, void *sycl_event);

/// @} dnnl_graph_api_sycl_interop

/// @} dnnl_graph_api_interop

/// @} dnnl_graph_api

#ifdef __cplusplus
}
#endif

#endif
