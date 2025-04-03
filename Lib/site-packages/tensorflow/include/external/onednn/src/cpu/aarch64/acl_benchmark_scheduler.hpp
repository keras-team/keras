/*******************************************************************************
* Copyright 2023 Arm Ltd. and affiliates
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
#ifndef CPU_AARCH64_ACL_BENCHMARK_SCHEDULER_HPP
#define CPU_AARCH64_ACL_BENCHMARK_SCHEDULER_HPP

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/runtime/IScheduler.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
// BenchmarkScheduler implement's ACL IScheduler interface and acts as an interceptor scheduler
// when DNNL_VERBOSE=profile,profile_externals. It intercepts calls made by the actual scheduler used by ACL and adds
// timers to benchmark execution time of ACL kernels and store kernel information.
class BenchmarkScheduler final : public arm_compute::IScheduler {
public:
    BenchmarkScheduler(IScheduler &real_scheduler);

    ~BenchmarkScheduler();

    void set_num_threads(unsigned int num_threads) override;
    unsigned int num_threads() const override;
    void set_num_threads_with_affinity(
            unsigned int num_threads, BindFunc func) override;
    void schedule(arm_compute::ICPPKernel *kernel,
            const arm_compute::IScheduler::Hints &hints) override;
    void schedule_op(arm_compute::ICPPKernel *kernel,
            const arm_compute::IScheduler::Hints &hints,
            const arm_compute::Window &window,
            arm_compute::ITensorPack &tensors) override;

protected:
    void run_workloads(std::vector<Workload> &workloads) override;
    void run_tagged_workloads(
            std::vector<Workload> &workloads, const char *tag) override;

private:
    IScheduler &_real_scheduler;
};

#endif // CPU_AARCH64_ACL_BENCHMARK_SCHEDULER_HPP

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
