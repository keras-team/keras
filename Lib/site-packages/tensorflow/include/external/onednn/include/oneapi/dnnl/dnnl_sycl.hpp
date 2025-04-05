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

#ifndef ONEAPI_DNNL_DNNL_SYCL_HPP
#define ONEAPI_DNNL_DNNL_SYCL_HPP

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.h"

/// @endcond

/// @addtogroup dnnl_api
/// @{

namespace dnnl {

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_sycl_interop SYCL interoperability API
/// API extensions to interact with the underlying SYCL run-time.
///
/// @sa @ref dev_guide_dpcpp_interoperability in developer guide
/// @{

/// SYCL interoperability namespace
namespace sycl_interop {

/// Memory allocation kind.
enum class memory_kind {
    /// USM (device, shared, host, or unknown) memory allocation kind - default.
    usm = dnnl_sycl_interop_usm,
    /// Buffer memory allocation kind.
    buffer = dnnl_sycl_interop_buffer,
};

/// Converts a memory allocation kind enum value from C++ API to C API type.
///
/// @param akind C++ API memory allocation kind enum value.
/// @returns Corresponding C API memory allocation kind enum value.
inline dnnl_sycl_interop_memory_kind_t convert_to_c(memory_kind akind) {
    return static_cast<dnnl_sycl_interop_memory_kind_t>(akind);
}

/// Constructs an engine from SYCL device and context objects.
///
/// @param adevice SYCL device.
/// @param acontext SYCL context.
///
/// @returns Created engine.
inline engine make_engine(
        const sycl::device &adevice, const sycl::context &acontext) {
    dnnl_engine_t aengine;
    error::wrap_c_api(dnnl_sycl_interop_engine_create(&aengine,
                              static_cast<const void *>(&adevice),
                              static_cast<const void *>(&acontext)),
            "could not create an engine");
    return engine(aengine);
}

/// Returns the SYCL context associated with an engine.
///
/// @param aengine Engine to query.
///
/// @returns The underlying SYCL device of the engine.
inline sycl::context get_context(const engine &aengine) {
    void *ctx_ptr;
    error::wrap_c_api(
            dnnl_sycl_interop_engine_get_context(aengine.get(), &ctx_ptr),
            "could not get a context handle");
    auto ctx = *static_cast<sycl::context *>(ctx_ptr);
    return ctx;
}

/// Returns the SYCL device associated with an engine.
///
/// @param aengine Engine to query.
///
/// @returns The underlying SYCL context of the engine.
inline sycl::device get_device(const engine &aengine) {
    void *dev_ptr;
    error::wrap_c_api(
            dnnl_sycl_interop_engine_get_device(aengine.get(), &dev_ptr),
            "could not get a device handle");
    auto dev = *static_cast<sycl::device *>(dev_ptr);
    return dev;
}

/// Creates an execution stream for a given engine associated with a SYCL
/// queue.
///
/// @param aengine Engine object to use for the stream.
/// @param aqueue SYCL queue to use for the stream.
///
/// @returns An execution stream.
inline stream make_stream(const engine &aengine, sycl::queue &aqueue) {
    dnnl_stream_t astream;
    error::wrap_c_api(
            dnnl_sycl_interop_stream_create(&astream, aengine.get(), &aqueue),
            "could not create a stream");
    return stream(astream);
}

/// Returns the SYCL queue associated with an execution stream.
///
/// @param astream Execution stream to query.
///
/// @returns SYCL queue object.
inline sycl::queue get_queue(const stream &astream) {
    void *queue_ptr;
    error::wrap_c_api(
            dnnl_sycl_interop_stream_get_queue(astream.get(), &queue_ptr),
            "could not get a stream handle");
    auto queue = *static_cast<sycl::queue *>(queue_ptr);
    return queue;
}

/// Returns the SYCL buffer associated with a memory object.
///
/// Throws an exception if the memory allocation kind associated with the
/// memory object is not equal to dnnl::sycl_interop::memory_kind::buffer.
///
/// @tparam T Type of the requested buffer.
/// @tparam ndims Number of dimensions of the requested buffer.
/// @param amemory Memory object.
///
/// @returns SYCL buffer associated with the memory object.
template <typename T, int ndims = 1>
sycl::buffer<T, ndims> get_buffer(const memory &amemory) {
    static_assert(ndims == 1, "only 1D buffers supported");

    // XXX: workaround: when CPU runtime is not SYCL and amemory was created
    // for CPU engine `get_buffer` should return an error. Use interop API to
    // implement the check.
    dnnl_sycl_interop_memory_kind_t ckind;
    error::wrap_c_api(
            dnnl_sycl_interop_memory_get_memory_kind(amemory.get(), &ckind),
            "could not get SYCL buffer object");

    void *handle_ptr;
    error::wrap_c_api(dnnl_memory_get_data_handle(amemory.get(), &handle_ptr),
            "could not get SYCL buffer object");

    // XXX: workaround: zero-range buffer cannot be constructed.
    if (!handle_ptr) return sycl::buffer<T, ndims>(sycl::range<1>(1));

    auto &buf_u8 = *static_cast<sycl::buffer<uint8_t, 1> *>(handle_ptr);

    auto range = sycl::range<1>(buf_u8.byte_size() / sizeof(T));
    return buf_u8.reinterpret<T, 1>(range);
}

/// Sets SYCL buffer associated with a memory object.
///
/// @tparam T Type of the buffer.
/// @tparam ndims Number of dimensions of the buffer.
/// @param amemory Memory object to change.
/// @param abuffer SYCL buffer.
template <typename T, int ndims>
void set_buffer(memory &amemory, sycl::buffer<T, ndims> &abuffer) {
    auto range = sycl::range<1>(abuffer.byte_size());
    auto buf_u8 = abuffer.template reinterpret<uint8_t, 1>(range);
    error::wrap_c_api(dnnl_sycl_interop_memory_set_buffer(
                              amemory.get(), static_cast<void *>(&buf_u8)),
            "could not set SYCL buffer object");
}

/// Returns the memory allocation kind associated with a memory object.
///
/// @param amemory A memory object.
///
/// @returns The underlying memory allocation kind of the memory object.
inline memory_kind get_memory_kind(const memory &amemory) {
    dnnl_sycl_interop_memory_kind_t ckind;
    error::wrap_c_api(
            dnnl_sycl_interop_memory_get_memory_kind(amemory.get(), &ckind),
            "could not get memory kind");
    return static_cast<memory_kind>(ckind);
}

/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying buffer set. In this
/// case, the buffer will be initialized as if:
/// - dnnl::memory::set_data_handle() had been called, if @p memory_kind is
///   equal to dnnl::sycl_interop::memory_kind::usm, or
/// - dnnl::sycl_interop::set_buffer() has been called, if @p memory_kind is
///   equal to dnnl::sycl_interop::memory_kind::buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param kind Memory allocation kind to specify the type of handle.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl::sycl_interop::memory_kind::usm.
///     - A pointer to SYCL buffer. In this case the library doesn't own the
///       buffer. Requires @p memory_kind be equal to be equal to
///       dnnl::sycl_interop::memory_kind::buffer.
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
            dnnl_sycl_interop_memory_create(&c_memory, memory_desc.get(),
                    aengine.get(), convert_to_c(kind), handle),
            "could not create a memory");
    return memory(c_memory);
}

/// Constructs a memory object from a SYCL buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param abuffer A SYCL buffer to use.
///
/// @returns Created memory object.
template <typename T, int ndims = 1>
memory make_memory(const memory::desc &memory_desc, const engine &aengine,
        sycl::buffer<T, ndims> &abuffer) {
    memory amemory(memory_desc, aengine, DNNL_MEMORY_NONE);
    set_buffer(amemory, abuffer);
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
/// @param deps Optional vector with `sycl::event` dependencies.
///
/// @returns Output event.
inline sycl::event execute(const dnnl::primitive &aprimitive,
        const stream &astream, const std::unordered_map<int, memory> &args,
        const std::vector<sycl::event> &deps = {}) {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    sycl::event return_event;
    error::wrap_c_api(
            dnnl_sycl_interop_primitive_execute(aprimitive.get(), astream.get(),
                    (int)c_args.size(), c_args.data(), &deps, &return_event),
            "could not execute a primitive");
    return return_event;
}

} // namespace sycl_interop

/// @} dnnl_api_sycl_interop

/// @} dnnl_api_interop

} // namespace dnnl

/// @} dnnl_api

#endif // DNNL_SYCL_HPP
