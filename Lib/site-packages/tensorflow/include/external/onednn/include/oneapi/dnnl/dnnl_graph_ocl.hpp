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

#ifndef ONEAPI_DNNL_DNNL_GRAPH_OCL_HPP
#define ONEAPI_DNNL_DNNL_GRAPH_OCL_HPP

/// @cond DO_NOT_DOCUMENT_THIS
#include <vector>

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_ocl.h"
#include "oneapi/dnnl/dnnl_ocl.hpp"
/// @endcond

/// @addtogroup dnnl_graph_api
/// @{

namespace dnnl {
namespace graph {

/// @addtogroup dnnl_graph_api_interop Runtime interoperability API
/// API extensions to interact with the underlying run-time.
/// @{

/// @addtogroup dnnl_graph_api_ocl_interop OpenCL interoperability API API
/// extensions to interact with the underlying OpenCL run-time.
///
/// @{

/// OpenCL interoperability namespace
namespace ocl_interop {

/// Constructs an allocator from OpenCL malloc and free function pointer. OpenCL
/// allocator  should be used for OpenCL GPU runtime. Currently, only device USM
/// allocator is supported.
///
/// @param ocl_malloc The pointer to OpenCL malloc function
/// @param ocl_free The pointer to OpenCL free function
/// @returns Created allocator
inline allocator make_allocator(dnnl_graph_ocl_allocate_f ocl_malloc,
        dnnl_graph_ocl_deallocate_f ocl_free) {
    dnnl_graph_allocator_t c_allocator = nullptr;
    error::wrap_c_api(dnnl_graph_ocl_interop_allocator_create(
                              &c_allocator, ocl_malloc, ocl_free),
            "could not create allocator for opencl device");
    return allocator(c_allocator);
}

/// Constructs an engine from an OpenCL device, an OpenCL context, and an
/// allocator.
///
/// @param device A valid OpenCL device to construct the engine
/// @param context A valid OpenCL context to construct the engine
/// @param alloc An allocator to associate with the engine
/// @returns Created engine
inline engine make_engine_with_allocator(
        cl_device_id device, cl_context context, const allocator &alloc) {
    dnnl_engine_t c_engine;
    error::wrap_c_api(dnnl_graph_ocl_interop_make_engine_with_allocator(
                              &c_engine, device, context, alloc.get()),
            "could not make an engine with allocator");
    return engine(c_engine);
}

/// Constructs an engine from an OpenCL device, an OpenCL context, an
/// allocator, and a serialized engine cache blob.
///
/// @param device A valid OpenCL device to construct the engine
/// @param context A valid OpenCL context to construct the engine
/// @param alloc An allocator to associate with the engine
/// @param cache_blob Cache blob serialized beforehand
/// @returns Created engine
inline engine make_engine_with_allocator(cl_device_id device,
        cl_context context, const allocator &alloc,
        const std::vector<uint8_t> &cache_blob) {
    dnnl_engine_t c_engine;
    error::wrap_c_api(
            dnnl_graph_ocl_interop_make_engine_from_cache_blob_with_allocator(
                    &c_engine, device, context, alloc.get(), cache_blob.size(),
                    cache_blob.data()),
            "could not make an engine with allocator from cache blob");
    return engine(c_engine);
}

/// Executes a compiled partition in a specified stream and returns a OpenCL
/// event.
///
/// @param c_partition Compiled partition to execute.
/// @param astream Stream object to run over
/// @param inputs Arguments map.
/// @param outputs Arguments map.
/// @param deps Optional vector with `cl_event` dependencies.
/// @returns Output event.
inline cl_event execute(compiled_partition &c_partition, stream &astream,
        const std::vector<tensor> &inputs, std::vector<tensor> &outputs,
        const std::vector<cl_event> &deps = {}) {
    std::vector<const_dnnl_graph_tensor_t> c_inputs;
    c_inputs.reserve(inputs.size());
    for (auto &in : inputs) {
        c_inputs.push_back(in.get());
    }
    std::vector<const_dnnl_graph_tensor_t> c_outputs;
    c_outputs.reserve(outputs.size());
    for (auto &out : outputs) {
        c_outputs.push_back(out.get());
    }

    const cl_event *c_deps = deps.empty() ? nullptr : deps.data();

    cl_event ocl_event;
    error::wrap_c_api(
            dnnl_graph_ocl_interop_compiled_partition_execute(c_partition.get(),
                    astream.get(), c_inputs.size(), c_inputs.data(),
                    c_outputs.size(), c_outputs.data(), c_deps,
                    (int)deps.size(), &ocl_event),
            "could not execute the compiled_partition on a specified opencl "
            "stream");
    return ocl_event;
}

} // namespace ocl_interop

/// @} dnnl_graph_api_ocl_interop

/// @} dnnl_graph_api_interop

} // namespace graph
} // namespace dnnl

/// @} dnnl_graph_api

#endif
