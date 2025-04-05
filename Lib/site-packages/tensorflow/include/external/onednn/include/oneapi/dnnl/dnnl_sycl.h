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

#ifndef ONEAPI_DNNL_DNNL_SYCL_H
#define ONEAPI_DNNL_DNNL_SYCL_H

#include "oneapi/dnnl/dnnl.h"

#include "oneapi/dnnl/dnnl_sycl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_sycl_interop
/// @{

/// Creates an engine associated with a SYCL device and a SYCL context.
///
/// @param engine Output engine.
/// @param device Pointer to the SYCL device to use for the engine.
/// @param context Pointer to the SYCL context to use for the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_engine_create(
        dnnl_engine_t *engine, const void *device, const void *context);

/// Returns the SYCL context associated with an engine.
///
/// @param engine Engine to query.
/// @param context Pointer to the underlying SYCL context of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_engine_get_context(
        dnnl_engine_t engine, void **context);

/// Returns the SYCL device associated with an engine.
///
/// @param engine Engine to query.
/// @param device Pointer to the underlying SYCL device of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_engine_get_device(
        dnnl_engine_t engine, void **device);

/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying buffer set. In this
/// case, the buffer will be initialized as if:
/// - dnnl_memory_set_data_handle() had been called, if @p memory_kind is equal
///   to dnnl_sycl_interop_usm, or
/// - dnnl_sycl_interop_memory_set_buffer() has been called, if @p memory_kind
///   is equal to dnnl_sycl_interop_buffer.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param memory_kind Memory allocation kind to specify the type of handle.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl_sycl_interop_usm.
///     - A pointer to SYCL buffer. In this case the library doesn't own the
///       buffer. Requires @p memory_kind be equal to be equal to
///       dnnl_sycl_interop_buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer that corresponds to the memory allocation kind
///       @p memory_kind for the memory object. In this case the library
///       owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_memory_create(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        dnnl_sycl_interop_memory_kind_t memory_kind, void *handle);

/// Returns the memory allocation kind associated with a memory object.
///
/// @param memory Memory to query.
/// @param memory_kind Output underlying memory allocation kind of the memory
///     object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_memory_get_memory_kind(
        const_dnnl_memory_t memory,
        dnnl_sycl_interop_memory_kind_t *memory_kind);

/// Sets a SYCL buffer for a memory object.
///
/// @param memory Memory object.
/// @param buffer SYCL buffer to be set in the memory object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_memory_set_buffer(
        dnnl_memory_t memory, void *buffer);

/// Creates an execution stream for a given engine associated with a SYCL
/// queue.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param queue SYCL queue to use.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_stream_create(
        dnnl_stream_t *stream, dnnl_engine_t engine, void *queue);

/// Returns the SYCL queue associated with an execution stream.
///
/// @param stream Execution stream to query.
/// @param queue Output SYCL command queue.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_stream_get_queue(
        dnnl_stream_t stream, void **queue);

/// Executes computations specified by the primitive in a specified stream and
/// returns a SYCL event.
///
/// @param primitive Primitive to execute.
/// @param stream Stream to use.
/// @param nargs Number of arguments.
/// @param args Array of arguments. Each argument is an
///     <index, #dnnl_memory_t> pair. The index is one of the `DNNL_ARG_*`
///     values such as `DNNL_ARG_SRC`. Unless runtime shapes are used (see
///     #DNNL_RUNTIME_DIM_VAL), the memory object must have the same memory
///     descriptor as that returned by
///     #dnnl_primitive_desc_query_md(#dnnl_query_exec_arg_md, index).
/// @param deps A pointer to std::vector<sycl::event> that contains
///     dependencies.
/// @param return_event Output event.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_sycl_interop_primitive_execute(
        const_dnnl_primitive_t primitive, dnnl_stream_t stream, int nargs,
        const dnnl_exec_arg_t *args, const void *deps, void *return_event);

/// @} dnnl_api_sycl_interop

/// @} dnnl_api_interop

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif
