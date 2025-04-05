/*******************************************************************************
* Copyright 2022-2023 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_THREAD_HPP
#define CPU_AARCH64_ACL_THREAD_HPP

#include "common/dnnl_thread.hpp"
#include "common/verbose.hpp"

#include "arm_compute/runtime/Scheduler.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_thread_utils {

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
// Number of threads in Compute Library is set by OMP_NUM_THREADS
// dnnl_get_max_threads() == OMP_NUM_THREADS
void acl_thread_bind();
// Swap BenchmarkScheduler for default ACL scheduler builds (i.e. CPPScheduler, OMPScheduler)
// for DNNL_VERBOSE=profile,profile_externals
void acl_set_benchmark_scheduler_default();
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
// Retrieve threadpool size during primitive execution and set ThreadpoolScheduler num_threads
void acl_set_tp_scheduler();
void acl_set_threadpool_num_threads();
// Swap BenchmarkScheduler for custom scheduler builds (i.e. ThreadPoolScheduler) for DNNL_VERBOSE=profile,profile_externals
void acl_set_tp_benchmark_scheduler();
#endif
// Set threading for ACL
void set_acl_threading();
} // namespace acl_thread_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_THREAD_HPP
