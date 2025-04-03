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

#ifndef ONEAPI_DNNL_DNNL_OCL_HPP
#define ONEAPI_DNNL_DNNL_OCL_HPP

#include "oneapi/dnnl/dnnl.hpp"

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_ocl.h"

#include <CL/cl.h>
/// @endcond

/// @addtogroup dnnl_api
/// @{

namespace dnnl {

/// @addtogroup dnnl_api_interop Runtime interoperability API
/// API extensions to interact with the underlying run-time.
/// @{

/// @addtogroup dnnl_api_ocl_interop OpenCL interoperability API
/// API extensions to interact with the underlying OpenCL run-time.
///
/// @sa @ref dev_guide_opencl_interoperability in developer guide
/// @{

/// OpenCL interoperability namespace
namespace ocl_interop {

/// Memory allocation kind.
enum class memory_kind {
    /// USM (device, shared, host, or unknown) memory allocation kind.
    usm = dnnl_ocl_interop_usm,
    /// Buffer memory allocation kind - default.
    buffer = dnnl_ocl_interop_buffer,
};

/// Converts a memory allocation kind enum value from C++ API to C API type.
///
/// @param akind C++ API memory allocation kind enum value.
/// @returns Corresponding C API memory allocation kind enum value.
inline dnnl_ocl_interop_memory_kind_t convert_to_c(memory_kind akind) {
    return static_cast<dnnl_ocl_interop_memory_kind_t>(akind);
}

/// Returns the cache blob ID of the OpenCL device.
///
/// @warning
///     This API is intended to be used with
///     #dnnl::ocl_interop::get_engine_cache_blob() and
///     #dnnl::ocl_interop::make_engine(cl_device_id, cl_context, const std::vector<uint8_t> &).
///     The returned cache blob ID can only be used as an ID of the cache blob
///     returned by #dnnl::ocl_interop::get_engine_cache_blob().
///
/// @note The cache blob ID can be empty (@p size will be 0 and
///     @p cache_blob_id will be nullptr) if oneDNN doesn't have anything to
///     put in the cache blob. (#dnnl_ocl_interop_engine_get_cache_blob will
///     return an empty cache blob).
///
/// @param device An OpenCL device.
/// @returns A vector containing the cache blob ID.
inline std::vector<uint8_t> get_engine_cache_blob_id(cl_device_id device) {
    size_t size = 0;
    error::wrap_c_api(
            dnnl_ocl_interop_engine_get_cache_blob_id(device, &size, nullptr),
            "could not get an engine cache blob id size");

    std::vector<uint8_t> cache_blob_id(size);
    error::wrap_c_api(dnnl_ocl_interop_engine_get_cache_blob_id(
                              device, &size, cache_blob_id.data()),
            "could not get an engine cache blob id");
    return cache_blob_id;
}

/// Returns a cache blob for the engine.
///
/// @note The cache blob vector can be empty if oneDNN doesn't have anything
///     to put in the cache blob. It's the user's responsibility to check
///     whether it's empty prior to passing it to
///     #dnnl::ocl_interop::make_engine(cl_device_id, cl_context, const std::vector<uint8_t> &)
///
/// @param aengine Engine to query for the cache blob.
/// @returns Vector containing the cache blob.
inline std::vector<uint8_t> get_engine_cache_blob(const engine &aengine) {
    size_t size = 0;
    error::wrap_c_api(dnnl_ocl_interop_engine_get_cache_blob(
                              aengine.get(), &size, nullptr),
            "could not get an engine cache blob size");

    std::vector<uint8_t> cache_blob(size);
    error::wrap_c_api(dnnl_ocl_interop_engine_get_cache_blob(
                              aengine.get(), &size, cache_blob.data()),
            "could not get an engine cache blob");
    return cache_blob;
}

/// Constructs an engine from the given cache blob.
///
/// @param device The OpenCL device that this engine will encapsulate.
/// @param context The OpenCL context (containing the device) that this
///     engine will use for all operations.
/// @param cache_blob Cache blob.
/// @returns An engine.
inline engine make_engine(cl_device_id device, cl_context context,
        const std::vector<uint8_t> &cache_blob) {
    dnnl_engine_t c_engine;
    error::wrap_c_api(
            dnnl_ocl_interop_engine_create_from_cache_blob(&c_engine, device,
                    context, cache_blob.size(), cache_blob.data()),
            "could not create an engine from cache blob");
    return engine(c_engine);
}

/// Constructs an engine from OpenCL device and context objects.
///
/// @param device The OpenCL device that this engine will encapsulate.
/// @param context The OpenCL context (containing the device) that this
///     engine will use for all operations.
/// @returns An engine.
inline engine make_engine(cl_device_id device, cl_context context) {
    dnnl_engine_t c_engine;
    error::wrap_c_api(
            dnnl_ocl_interop_engine_create(&c_engine, device, context),
            "could not create an engine");
    return engine(c_engine);
}

/// Returns OpenCL context associated with the engine.
///
/// @param aengine An engine.
/// @returns Underlying OpenCL context.
inline cl_context get_context(const engine &aengine) {
    cl_context context = nullptr;
    error::wrap_c_api(
            dnnl_ocl_interop_engine_get_context(aengine.get(), &context),
            "could not get an OpenCL context from an engine");
    return context;
}

/// Returns OpenCL device associated with the engine.
///
/// @param aengine An engine.
/// @returns Underlying OpenCL device.
inline cl_device_id get_device(const engine &aengine) {
    cl_device_id device = nullptr;
    error::wrap_c_api(dnnl_ocl_interop_get_device(aengine.get(), &device),
            "could not get an OpenCL device from an engine");
    return device;
}

/// Constructs an execution stream for the specified engine and OpenCL queue.
///
/// @param aengine Engine to create the stream on.
/// @param queue OpenCL queue to use for the stream.
/// @returns An execution stream.
inline stream make_stream(const engine &aengine, cl_command_queue queue) {
    dnnl_stream_t c_stream;
    error::wrap_c_api(
            dnnl_ocl_interop_stream_create(&c_stream, aengine.get(), queue),
            "could not create a stream");
    return stream(c_stream);
}

/// Returns OpenCL queue object associated with the execution stream.
///
/// @param astream An execution stream.
/// @returns Underlying OpenCL queue.
inline cl_command_queue get_command_queue(const stream &astream) {
    cl_command_queue queue = nullptr;
    error::wrap_c_api(
            dnnl_ocl_interop_stream_get_command_queue(astream.get(), &queue),
            "could not get an OpenCL command queue from a stream");
    return queue;
}

/// Returns the OpenCL memory object associated with the memory object.
///
/// @param amemory A memory object.
/// @returns Underlying OpenCL memory object.
inline cl_mem get_mem_object(const memory &amemory) {
    cl_mem mem_object;
    error::wrap_c_api(
            dnnl_ocl_interop_memory_get_mem_object(amemory.get(), &mem_object),
            "could not get OpenCL buffer object from a memory object");
    return mem_object;
}

/// Sets the OpenCL memory object associated with the memory object.
///
/// For behavioral details see memory::set_data_handle().
///
/// @param amemory A memory object.
/// @param mem_object OpenCL cl_mem object to use as the underlying
///     storage. It must have at least get_desc().get_size() bytes
///     allocated.
inline void set_mem_object(memory &amemory, cl_mem mem_object) {
    error::wrap_c_api(
            dnnl_ocl_interop_memory_set_mem_object(amemory.get(), mem_object),
            "could not set OpenCL buffer object from a memory object");
}

/// Returns the memory allocation kind associated with a memory object.
///
/// @param amemory A memory object.
///
/// @returns The underlying memory allocation kind of the memory object.
inline memory_kind get_memory_kind(const memory &amemory) {
    dnnl_ocl_interop_memory_kind_t ckind;
    error::wrap_c_api(
            dnnl_ocl_interop_memory_get_memory_kind(amemory.get(), &ckind),
            "could not get memory kind");
    return static_cast<memory_kind>(ckind);
}

/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying buffer set. In this
/// case, the buffer will be initialized as if:
/// - dnnl::memory::set_data_handle() had been called, if @p memory_kind is
///   equal to dnnl::ocl_interop::memory_kind::usm, or
/// - dnnl::ocl_interop::set_mem_object() has been called, if @p memory_kind is
///   equal to dnnl::ocl_interop::memory_kind::buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param kind Memory allocation kind to specify the type of handle.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl::ocl_interop::memory_kind::usm.
///     - An OpenCL buffer. In this case the library doesn't own the buffer.
///       Requires @p memory_kind be equal to be equal to
///       dnnl::ocl_interop::memory_kind::buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer that corresponds to the memory allocation kind
///       @p memory_kind for the memory object. In this case the library
///       owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
///
/// @returns Created memory object.
inline memory make_memory(const memory::desc &memory_desc,
        const engine &aengine, memory_kind kind,
        void *handle = DNNL_MEMORY_ALLOCATE) {
    dnnl_memory_t c_memory;
    error::wrap_c_api(
            dnnl_ocl_interop_memory_create(&c_memory, memory_desc.get(),
                    aengine.get(), convert_to_c(kind), handle),
            "could not create a memory");
    return memory(c_memory);
}

/// Constructs a memory object from an OpenCL buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param mem_object An OpenCL buffer to use.
///
/// @returns Created memory object.
inline memory make_memory(const memory::desc &memory_desc,
        const engine &aengine, cl_mem mem_object) {
    memory amemory(memory_desc, aengine, DNNL_MEMORY_NONE);
    set_mem_object(amemory, mem_object);
    return amemory;
}

/// Executes computations specified by the primitive in a specified stream and
/// returns a SYCL event.
///
/// Arguments are passed via an arguments map containing
/// <index, memory object> pairs. The index must be one of the `DNNL_ARG_*`
/// values such as `DNNL_ARG_SRC`, and the memory must have a memory descriptor
/// matching the one returned by
/// #dnnl::primitive_desc::query_md(#query::exec_arg_md, index) unless using
/// dynamic shapes (see #DNNL_RUNTIME_DIM_VAL).
///
/// @param aprimitive Primitive to execute.
/// @param astream Stream object. The stream must belong to the same engine
///     as the primitive.
/// @param args Arguments map.
/// @param deps Optional vector with `cl_event` dependencies.
///
/// @returns Output event. It's the user's responsibility to manage lifetime
///     of the event.
inline cl_event execute(const dnnl::primitive &aprimitive,
        const stream &astream, const std::unordered_map<int, memory> &args,
        const std::vector<cl_event> &deps = {}) {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    const cl_event *c_deps = deps.empty() ? nullptr : deps.data();

    cl_event return_event;
    error::wrap_c_api(dnnl_ocl_interop_primitive_execute(aprimitive.get(),
                              astream.get(), (int)c_args.size(), c_args.data(),
                              c_deps, (int)deps.size(), &return_event),
            "could not execute a primitive");
    return return_event;
}

} // namespace ocl_interop

/// @} dnnl_api_ocl_interop

/// @} dnnl_api_interop

} // namespace dnnl

/// @} dnnl_api

#endif
