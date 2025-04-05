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

#ifndef GRAPH_UTILS_OCL_USM_UTILS_HPP
#define GRAPH_UTILS_OCL_USM_UTILS_HPP

#include "oneapi/dnnl/dnnl_config.h"

#include "graph/utils/ocl_check.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
namespace ocl {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
void *malloc_shared(const cl_device_id dev, const cl_context ctx, size_t size,
        size_t alignment = 0);

void free(void *ptr, const cl_device_id dev, const cl_context ctx);
#endif

} // namespace ocl
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
