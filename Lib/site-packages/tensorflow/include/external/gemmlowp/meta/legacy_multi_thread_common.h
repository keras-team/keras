// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// multi_thread_common.h: Multithreading code shared by different meta gemm
// versions.

#ifndef GEMMLOWP_META_MULTI_THREAD_COMMON_H_
#define GEMMLOWP_META_MULTI_THREAD_COMMON_H_

#include "../internal/multi_thread_gemm.h"

namespace gemmlowp {
namespace meta {
namespace internal {

const std::int32_t kMinTaskSize = 16000;
const std::int32_t kMinTaskDimension = 4;

struct TaskRect {
  std::int32_t m_offset;
  std::int32_t m;
  std::int32_t n_offset;
  std::int32_t n;

  TaskRect(std::int32_t m_offset, std::int32_t m, std::int32_t n_offset,
           std::int32_t n)
      : m_offset(m_offset), m(m), n_offset(n_offset), n(n) {}
};

template <typename IN_TYPE, typename OUT_TYPE, typename F>
struct MetaTask : gemmlowp::Task {
  std::uint8_t* scratch;
  const IN_TYPE* lhs;
  const IN_TYPE* rhs;
  TaskRect task_rect;
  std::int32_t k;
  OUT_TYPE* result;
  std::int32_t result_stride;
  const F& operation;

  MetaTask(std::uint8_t* scratch, const IN_TYPE* lhs, const IN_TYPE* rhs,
           const TaskRect& task_rect, std::int32_t k, OUT_TYPE* result,
           std::int32_t result_stride, const F& operation)
      : scratch(scratch),
        lhs(lhs),
        rhs(rhs),
        task_rect(task_rect),
        k(k),
        result(result),
        result_stride(result_stride),
        operation(operation) {}

  void Run() override {
    const IN_TYPE* task_lhs = lhs + task_rect.m_offset * k;
    const IN_TYPE* task_rhs = rhs + task_rect.n_offset * k;
    OUT_TYPE* task_result =
        result + task_rect.m_offset * result_stride + task_rect.n_offset;
    operation.ExecuteMatrixMatrix(scratch, task_lhs, task_rhs, task_rect.m,
                                  task_rect.n, k, task_result, result_stride);
  }
};

std::int32_t ResolveMaxThreads(std::int32_t max_threads) {
  if (max_threads == 0) {
    static const int hardware_threads_count =
        static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    return hardware_threads_count;
  }
  return max_threads;
}

void PrepareTasks(std::int32_t max_tasks, std::int32_t m, std::int32_t n,
                  std::int32_t k, std::vector<internal::TaskRect>* tasks) {
  const std::int32_t max_tasks_by_size = (m * n * k) / kMinTaskSize;
  const std::int32_t max_tasks_m = m / kMinTaskDimension;
  const std::int32_t max_tasks_n = n / kMinTaskDimension;
  const std::int32_t max_tasks_dimension = std::max(max_tasks_m, max_tasks_n);

  std::int32_t real_tasks = std::max(
      1, std::min(max_tasks, std::min(max_tasks_by_size, max_tasks_dimension)));

  if (real_tasks == 1) {
    tasks->push_back(TaskRect(0, m, 0, n));
    return;
  }

  if (max_tasks_m > max_tasks_n) {
    const std::int32_t m_chunk = m / real_tasks;
    for (int i = 0; i < real_tasks - 1; ++i) {
      tasks->push_back(TaskRect(i * m_chunk, m_chunk, 0, n));
    }
    const std::int32_t last_m_offset = (real_tasks - 1) * m_chunk;
    tasks->push_back(TaskRect(last_m_offset, m - last_m_offset, 0, n));
  } else {
    const std::int32_t n_chunk = n / real_tasks;
    for (int i = 0; i < real_tasks - 1; ++i) {
      tasks->push_back(TaskRect(0, m, i * n_chunk, n_chunk));
    }
    const std::int32_t last_n_offset = (real_tasks - 1) * n_chunk;
    tasks->push_back(TaskRect(0, m, last_n_offset, n - last_n_offset));
  }
}

template <typename IN_TYPE, typename OUT_TYPE, typename F>
void MultiThreadedMatrixMatrix(gemmlowp::WorkersPool* pool,
                               std::int32_t max_threads, std::uint8_t* scratch,
                               const IN_TYPE* lhs, const IN_TYPE* rhs,
                               std::int32_t m, std::int32_t n, std::int32_t k,
                               OUT_TYPE* result, std::int32_t result_stride,
                               const F& operation) {
  max_threads = internal::ResolveMaxThreads(max_threads);

  std::vector<internal::TaskRect> task_rects;
  internal::PrepareTasks(max_threads, m, n, k, &task_rects);

  if (task_rects.size() == 1) {
    operation.ExecuteMatrixMatrix(scratch, lhs, rhs, m, n, k, result,
                                  result_stride);
    return;
  }

  std::uint8_t* task_scratch = scratch;
  std::int32_t scratch_per_thread = operation.ScratchPerThread(m, n, k);
  std::vector<Task*> tasks;
  std::for_each(
      task_rects.begin(), task_rects.end(),
      [&tasks, &task_scratch, lhs, rhs, k, result, result_stride, operation,
       scratch_per_thread](internal::TaskRect& rect) {
        tasks.push_back(new internal::MetaTask<IN_TYPE, OUT_TYPE, F>(
            task_scratch, lhs, rhs, rect, k, result, result_stride, operation));
        task_scratch += scratch_per_thread;
      });
  pool->Execute(tasks);
}

}  // namespace internal
}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_MULTI_THREAD_COMMON_H_
