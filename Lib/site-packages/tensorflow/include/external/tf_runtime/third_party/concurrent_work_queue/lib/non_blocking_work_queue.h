// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Work queue implementation based on non-blocking concurrency primitives
// optimized for CPU intensive non-blocking compute tasks.
//
// This work queue uses TaskDeque for storing pending tasks. Thread tries to
// pop a task from the front of its own queue, and in a steal loop it tries to
// steal a task from the back of another thread pending tasks queue. This gives
// mostly LIFO task execution order, which is optimal for cache locality for
// compute intensive tasks.
//
// Work stealing algorithm is based on:
//
//   "Thread Scheduling for Multiprogrammed Multiprocessors"
//   Nimar S. Arora, Robert D. Blumofe, C. Greg Plaxton

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_NON_BLOCKING_WORK_QUEUE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_NON_BLOCKING_WORK_QUEUE_H_

#include <optional>
#include <string_view>

#include "task_deque.h"
#include "tfrt/host_context/task_function.h"
#include "work_queue_base.h"

namespace tfrt {
namespace internal {

template <typename ThreadingEnvironment>
class NonBlockingWorkQueue;

template <typename ThreadingEnvironmentTy>
struct WorkQueueTraits<NonBlockingWorkQueue<ThreadingEnvironmentTy>> {
  using ThreadingEnvironment = ThreadingEnvironmentTy;
  using Thread = typename ThreadingEnvironment::Thread;
  using Queue = ::tfrt::internal::TaskDeque;
};

template <typename ThreadingEnvironment>
class NonBlockingWorkQueue
    : public WorkQueueBase<NonBlockingWorkQueue<ThreadingEnvironment>> {
  using Base = WorkQueueBase<NonBlockingWorkQueue<ThreadingEnvironment>>;

  using Queue = typename Base::Queue;
  using Thread = typename Base::Thread;
  using PerThread = typename Base::PerThread;
  using ThreadData = typename Base::ThreadData;

 public:
  explicit NonBlockingWorkQueue(QuiescingState* quiescing_state,
                                int num_threads,
                                std::string_view thread_name_prefix = "");
  ~NonBlockingWorkQueue() = default;

  void AddTask(TaskFunction task);

  using Base::Steal;

 private:
  static constexpr char const* kThreadNamePrefix = "tfrt-non-blocking-queue";

  template <typename WorkQueue>
  friend class WorkQueueBase;

  using Base::GetPerThread;
  using Base::IsNotifyParkedThreadRequired;
  using Base::IsQuiescing;
  using Base::WithPendingTaskCounter;

  using Base::coprimes_;
  using Base::event_count_;
  using Base::num_threads_;
  using Base::thread_data_;

  [[nodiscard]] std::optional<TaskFunction> NextTask(Queue* queue);
  [[nodiscard]] std::optional<TaskFunction> Steal(Queue* queue);
  [[nodiscard]] bool Empty(Queue* queue);
};

template <typename ThreadingEnvironment>
NonBlockingWorkQueue<ThreadingEnvironment>::NonBlockingWorkQueue(
    QuiescingState* quiescing_state, int num_threads,
    std::string_view thread_name_prefix)
    : WorkQueueBase<NonBlockingWorkQueue>(
          quiescing_state,
          thread_name_prefix.empty() ? kThreadNamePrefix : thread_name_prefix,
          num_threads) {}

template <typename ThreadingEnvironment>
void NonBlockingWorkQueue<ThreadingEnvironment>::AddTask(TaskFunction task) {
  // Keep track of the number of pending tasks.
  if (IsQuiescing()) task = WithPendingTaskCounter(std::move(task));

  // If the worker queue is full, we will execute `task` in the current thread.
  std::optional<TaskFunction> inline_task;

  // If a caller thread is managed by `this` we push the new task into the front
  // of thread own queue (LIFO execution order). PushFront is completely lock
  // free (PushBack requires a mutex lock), and improves data locality (in
  // practice tasks submitted together share some data).
  //
  // If a caller is a free-standing thread (or worker of another pool), we push
  // the new task into a random queue (FIFO execution order). Tasks still could
  // be executed in LIFO order, if they would be stolen by other workers.

  PerThread* pt = GetPerThread();
  if (pt->parent == this) {
    // Worker thread of this pool, push onto the thread's queue.
    Queue& q = thread_data_[pt->thread_id].queue;
    inline_task = q.PushFront(std::move(task));
  } else {
    // A free-standing thread (or worker of another pool).
    unsigned rnd = FastReduce(pt->rng(), num_threads_);
    Queue& q = thread_data_[rnd].queue;
    inline_task = q.PushBack(std::move(task));
  }
  // Note: below we touch `*this` after making `task` available to worker
  // threads. Strictly speaking, this can lead to a racy-use-after-free.
  // Consider that Schedule is called from a thread that is neither main thread
  // nor a worker thread of this pool. Then, execution of `task` directly or
  // indirectly completes overall computations, which in turn leads to
  // destruction of this. We expect that such a scenario is prevented by the
  // program, that is, this is kept alive while any threads can potentially be
  // in Schedule.
  if (!inline_task.has_value()) {
    if (IsNotifyParkedThreadRequired())
      event_count_.Notify(/*notify_all=*/false);
  } else {
    (*inline_task)();  // Push failed, execute directly.
  }
}

template <typename ThreadingEnvironment>
[[nodiscard]] std::optional<TaskFunction>
NonBlockingWorkQueue<ThreadingEnvironment>::NextTask(Queue* queue) {
  return queue->PopFront();
}

template <typename ThreadingEnvironment>
[[nodiscard]] std::optional<TaskFunction>
NonBlockingWorkQueue<ThreadingEnvironment>::Steal(Queue* queue) {
  return queue->PopBack();
}

template <typename ThreadingEnvironment>
[[nodiscard]] bool NonBlockingWorkQueue<ThreadingEnvironment>::Empty(
    Queue* queue) {
  return queue->Empty();
}

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_NON_BLOCKING_WORK_QUEUE_H_
