/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GRAPH_UTILS_ALLOC_HPP
#define GRAPH_UTILS_ALLOC_HPP

#include "graph/interface/c_types_map.hpp"
#include "graph/utils/utils.hpp"

#ifdef DNNL_WITH_SYCL
#include "graph/utils/sycl_check.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "graph/utils/ocl_check.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

/// Default allocator for CPU
class cpu_allocator_t {
public:
    constexpr static size_t DEFAULT_ALIGNMENT = 64;
    static void *malloc(size_t size, size_t alignment);
    static void free(void *p);
};

#ifdef DNNL_WITH_SYCL
/// Default allocator for SYCL device
class sycl_allocator_t {
public:
    constexpr static size_t DEFAULT_ALIGNMENT = 64;
    static void *malloc(
            size_t size, size_t alignment, const void *dev, const void *ctx);
    static void free(void *ptr, const void *dev, const void *ctx, void *event);
};
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
class ocl_allocator_t {
public:
    static void *malloc(
            size_t size, size_t alignment, cl_device_id dev, cl_context ctx);
    static void free(
            void *ptr, cl_device_id dev, cl_context ctx, cl_event event);
};
#endif

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
