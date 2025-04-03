/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CPU_JIT_UTILS_LINUX_PERF_LINUX_PERF_HPP
#define CPU_JIT_UTILS_LINUX_PERF_LINUX_PERF_HPP

#ifdef __linux__
#include <cstddef>

namespace dnnl {
namespace impl {
namespace cpu {
namespace jit_utils {

void linux_perf_jitdump_record_code_load(
        const void *code, size_t code_size, const char *code_name);

void linux_perf_perfmap_record_code_load(
        const void *code, size_t code_size, const char *code_name);
} // namespace jit_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

#endif
