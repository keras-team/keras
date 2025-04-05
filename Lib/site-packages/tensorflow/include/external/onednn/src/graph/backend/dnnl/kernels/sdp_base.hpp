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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_BASE_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_BASE_HPP

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"
#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/thread_local_cache.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "graph/backend/dnnl/kernels/sdp.hpp"
#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"

#include "common/primitive_hashing.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// The second template param dt is used to indicate the internal data type of
// int8 sdp pattern. It doesn't take any effect if quantized param is false.
template <bool quantized = false, memory::data_type dt = memory::data_type::f32>
struct sdp_base_t : public kernel_base_t {
private:
    std::shared_ptr<kernel_base_t> kernel;

public:
    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override {
        const engine_kind_t ekind = g_engine->kind();
        const bool enable_decomp
                = ekind == engine_kind::cpu && enable_decomp_kernel();
        status_t sdp_decomp_status = status::success;
        if (enable_decomp) {
            kernel = std::make_shared<sdp_decomp_kernel_t<quantized, dt>>();
            sdp_decomp_status
                    = kernel->compile_impl(part, g_engine, inputs, outputs);
        }

        if (!enable_decomp || sdp_decomp_status != status::success) {
            kernel = std::make_shared<larger_partition_kernel_t>();
            return kernel->compile_impl(part, g_engine, inputs, outputs);
        }
        return sdp_decomp_status;
    }

    // The fuction is used to check if enable the decompostion kernel based on
    // user's env and params. Currently, we restrict the library’s CPU runtime
    // to openmp. We also only enable the decompose kernel in the machine with
    // specific ISA support. There is also an internal env var to decide if use
    // the kernel.
    // TODO: Remove CPU runtime check when we extend the support for others
    bool enable_decomp_kernel() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP \
        || DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        return graph::utils::getenv_int_internal("ENABLE_SDP_DECOMP", 1) > 0;
#else
        return false;
#endif
    }

    status_t execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) override {
        return kernel->execute_impl(g_stream, inputs, outputs);
    }

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        return kernel->sycl_execute_impl(
                g_stream, inputs, outputs, sycl_deps, sycl_event);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(const stream_t *g_stream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &deps, cl_event *event) override {
        return kernel->ocl_execute_impl(g_stream, inputs, outputs, deps, event);
    }
#endif
};
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
