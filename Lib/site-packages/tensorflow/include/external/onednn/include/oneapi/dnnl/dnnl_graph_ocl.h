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

#ifndef ONEAPI_DNNL_DNNL_GRAPH_OCL_H
#define ONEAPI_DNNL_DNNL_GRAPH_OCL_H

#include "oneapi/dnnl/dnnl_graph.h"

/// @cond DO_NOT_DOCUMENT_THIS
// Set target version for OpenCL explicitly to suppress a compiler warning.
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include <CL/cl.h>
/// @endcond

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_graph_api
/// @{

/// @addtogroup dnnl_graph_api_interop
/// @{

/// @addtogroup dnnl_graph_api_ocl_interop
/// @{

/// Allocation call-back function interface for OpenCL. OpenCL allocator should
/// be used for OpenCL GPU runtime. The call-back should return a USM device
/// memory pointer.
///
/// @param size Memory size in bytes for requested allocation
/// @param alignment The minimum alignment in bytes for the requested allocation
/// @param device A valid OpenCL device used to allocate
/// @param context A valid OpenCL context used to allocate
/// @returns The memory address of the requested USM allocation.
typedef void *(*dnnl_graph_ocl_allocate_f)(
        size_t size, size_t alignment, cl_device_id device, cl_context context);

/// Deallocation call-back function interface for OpenCL. OpenCL allocator
/// should be used for OpenCL runtime. The call-back should deallocate a USM
/// device memory returned by #dnnl_graph_ocl_allocate_f. The event should be
/// completed before deallocate the USM.
///
/// @param buf The USM allocation to be released
/// @param device A valid OpenCL device the USM associated with
/// @param context A valid OpenCL context used to free the USM allocation
/// @param event A event which the USM deallocation depends on
typedef void (*dnnl_graph_ocl_deallocate_f)(
        void *buf, cl_device_id device, cl_context context, cl_event event);

/// Creates an allocator with the given allocation and deallocation call-back
/// function pointers.
///
/// @param allocator Output allocator
/// @param ocl_malloc A pointer to OpenCL malloc function
/// @param ocl_free A pointer to OpenCL free function
/// @returns #dnnl_success on success and a status describing the
///     error otherwise.
dnnl_status_t DNNL_API dnnl_graph_ocl_interop_allocator_create(
        dnnl_graph_allocator_t *allocator, dnnl_graph_ocl_allocate_f ocl_malloc,
        dnnl_graph_ocl_deallocate_f ocl_free);

/// This API is a supplement for existing oneDNN engine API:
/// dnnl_status_t DNNL_API dnnl_ocl_interop_engine_create(
///     dnnl_engine_t *engine, cl_device_id device, cl_context context);
///
/// @param engine Output engine.
/// @param device Underlying OpenCL device to use for the engine.
/// @param context Underlying OpenCL context to use for the engine.
/// @param alloc Underlying allocator to use for the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_ocl_interop_make_engine_with_allocator(
        dnnl_engine_t *engine, cl_device_id device, cl_context context,
        const_dnnl_graph_allocator_t alloc);

/// This API is a supplement for existing oneDNN engine API:
/// dnnl_status_t DNNL_API dnnl_ocl_interop_engine_create_from_cache_blob(
///     dnnl_engine_t *engine, cl_device_id device, cl_context context,
///     size_t size, const uint8_t *cache_blob);
///
/// @param engine Output engine.
/// @param device The OpenCL device that this engine will encapsulate.
/// @param context The OpenCL context (containing the device) that this
///     engine will use for all operations.
/// @param alloc Underlying allocator to use for the engine.
/// @param size Size of the cache blob in bytes.
/// @param cache_blob Cache blob of size @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API
dnnl_graph_ocl_interop_make_engine_from_cache_blob_with_allocator(
        dnnl_engine_t *engine, cl_device_id device, cl_context context,
        const_dnnl_graph_allocator_t alloc, size_t size,
        const uint8_t *cache_blob);

/// Execute a compiled partition with OpenCL runtime.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param stream The stream used for execution
/// @param num_inputs The number of input tensors
/// @param inputs A list of input tensors
/// @param num_outputs The number of output tensors
/// @param outputs A non-empty list of output tensors
/// @param deps Optional handle of list with `cl_event` dependencies.
/// @param ndeps Number of dependencies.
/// @param return_event The handle of cl_event.
/// @returns #dnnl_success on success and a status describing the
///     error otherwise.
dnnl_status_t DNNL_API dnnl_graph_ocl_interop_compiled_partition_execute(
        const_dnnl_graph_compiled_partition_t compiled_partition,
        dnnl_stream_t stream, size_t num_inputs,
        const_dnnl_graph_tensor_t *inputs, size_t num_outputs,
        const_dnnl_graph_tensor_t *outputs, const cl_event *deps, int ndeps,
        cl_event *return_event);

/// @} dnnl_graph_api_ocl_interop

/// @} dnnl_graph_api_interop

/// @} dnnl_graph_api

#ifdef __cplusplus
}
#endif

#endif
