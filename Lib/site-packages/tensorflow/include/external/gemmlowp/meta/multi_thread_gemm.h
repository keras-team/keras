// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GEMMLOWP_META_MULTI_THREAD_GEMM_H_
#define GEMMLOWP_META_MULTI_THREAD_GEMM_H_

#include "multi_thread_common.h"
#include "single_thread_gemm.h"

namespace gemmlowp {
namespace meta {
namespace internal {

const std::int32_t kMinGemmTaskSize = 16000;
const std::int32_t kMinGemmTaskDimension = 4;

template <typename Executor, typename Params>
std::uint8_t* PrepareGemmTask(const Params& params, int kernel_m, int kernel_n,
                              int kernel_k, std::uint8_t* scratch, int m_start,
                              int m, int n_start, int n,
                              std::vector<Params>* tasks) {
  tasks->push_back(params);
  Params& task = tasks->back();
  task.scratch = scratch;

  task.m = m;
  task.lhs =
      StreamUtil<typename Params::InType, typename Params::LeftStream>::Offset(
          params.left_stream, params.lhs, m_start, 0);

  task.n = n;
  task.rhs =
      StreamUtil<typename Params::InType, typename Params::RightStream>::Offset(
          params.right_stream, params.rhs, n_start, 0);

  task.result =
      StreamUtil<typename Params::OutType, typename Params::OutputStream>::
          Offset(params.fused_kernel.output_stream, params.result, m_start,
                 n_start);

  return scratch + Executor::template EstimateScratchSize<Params>(
                       task, kernel_m, kernel_n, kernel_k);
}

template <typename MultiThreadingContext, typename Executor, typename Params>
bool PrepareGemmTasks(MultiThreadingContext* context, const Params& params,
                      int kernel_m, int kernel_n, int kernel_k,
                      std::vector<Params>* task_params) {
  const int max_threads = ResolveMaxThreads(context->max_num_threads());
  const int max_tasks_by_size =
      (params.m * params.n * params.k) / kMinGemmTaskSize;
  const int max_tasks_m = params.m / kMinGemmTaskDimension;
  const int max_tasks_n = params.n / kMinGemmTaskDimension;
  const int max_tasks_dimension = std::max(max_tasks_m, max_tasks_n);

  const int real_tasks = std::max(
      1,
      std::min(max_threads, std::min(max_tasks_by_size, max_tasks_dimension)));

  if (real_tasks == 1) {
    return false;
  }

  std::uint8_t* scratch = params.scratch;

  if (max_tasks_m > max_tasks_n) {
    const int m_chunk = params.m / real_tasks;
    for (int i = 0; i < real_tasks - 1; ++i) {
      scratch = PrepareGemmTask<Executor, Params>(
          params, kernel_m, kernel_n, kernel_k, scratch, i * m_chunk, m_chunk,
          0, params.n, task_params);
    }
    const int sum_m = (real_tasks - 1) * m_chunk;
    PrepareGemmTask<Executor, Params>(params, kernel_m, kernel_n, kernel_k,
                                      scratch, sum_m, params.m - sum_m, 0,
                                      params.n, task_params);
  } else {
    const int n_chunk = params.n / real_tasks;
    for (int i = 0; i < real_tasks - 1; ++i) {
      scratch = PrepareGemmTask<Executor, Params>(
          params, kernel_m, kernel_n, kernel_k, scratch, 0, params.m,
          i * n_chunk, n_chunk, task_params);
    }
    int sum_n = (real_tasks - 1) * n_chunk;
    PrepareGemmTask<Executor, Params>(params, kernel_m, kernel_n, kernel_k,
                                      scratch, 0, params.m, sum_n,
                                      params.n - sum_n, task_params);
  }

  return true;
}

template <typename Executor, typename Params, int kernel_m, int kernel_n,
          int kernel_k>
struct GemmTaskRunner : gemmlowp::Task {
  GemmTaskRunner(const Params& params) : params(params) {}

  void Run() override {
    Gemm<Executor, Params, kernel_m, kernel_n, kernel_k>(params);
  }

  Params params;
};

}  // namespace internal

template <typename MultiThreadingContext, typename Executor, typename Params,
          int kernel_m, int kernel_n, int kernel_k>
inline void MultiThreadGemm(MultiThreadingContext* context,
                            const Params& params) {
  typedef internal::GemmTaskRunner<Executor, Params, kernel_m, kernel_n,
                                   kernel_k>
      TaskRunnerType;

  std::vector<Params> task_params;
  if (!internal::PrepareGemmTasks<MultiThreadingContext, Executor, Params>(
          context, params, kernel_m, kernel_n, kernel_k, &task_params)) {
    Gemm<Executor, Params, kernel_m, kernel_n, kernel_k>(params);
    return;
  }

  auto workers_pool = context->workers_pool();
  std::vector<Task*> tasks;
  for (auto& task_param : task_params) {
    tasks.push_back(new TaskRunnerType(task_param));
  };
  workers_pool->Execute(tasks);
}

}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_MULTI_THREAD_GEMM_H_
