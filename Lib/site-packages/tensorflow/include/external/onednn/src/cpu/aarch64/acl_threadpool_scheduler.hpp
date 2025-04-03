/*******************************************************************************
* Copyright 2022 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_THREADPOOL_SCHEDULER_HPP
#define CPU_AARCH64_ACL_THREADPOOL_SCHEDULER_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL

#include "arm_compute/runtime/IScheduler.h"
#include "support/Mutex.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

class ThreadpoolScheduler final : public arm_compute::IScheduler {
public:
    ThreadpoolScheduler();
    ~ThreadpoolScheduler();

    /// Sets the number of threads the scheduler will use to run the kernels.
    void set_num_threads(unsigned int num_threads) override;
    /// Returns the number of threads that the ThreadpoolScheduler has in its pool.
    unsigned int num_threads() const override;

    /// Multithread the execution of the passed kernel if possible.
    void schedule(arm_compute::ICPPKernel *kernel,
            const arm_compute::IScheduler::Hints &hints) override;

    /// Multithread the execution of the passed kernel if possible.
    void schedule_op(arm_compute::ICPPKernel *kernel,
            const arm_compute::IScheduler::Hints &hints,
            const arm_compute::Window &window,
            arm_compute::ITensorPack &tensors) override;

protected:
    /// Execute workloads in parallel using num_threads
    void run_workloads(std::vector<Workload> &workloads) override;

private:
    uint _num_threads {};
    arm_compute::Mutex _run_workloads_mutex {};
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_THREADPOOL_SCHEDULER_HPP

#endif // DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
