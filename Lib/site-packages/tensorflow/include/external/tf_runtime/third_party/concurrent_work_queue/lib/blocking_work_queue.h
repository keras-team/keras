// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Work queue implementation based on non-blocking concurrency primitives
// optimized for IO and mostly blocking tasks.
//
// This work queue uses TaskQueue for storing pending tasks. Tasks executed
// in mostly FIFO order, which is optimal for IO tasks.

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_

#include <cstdint>
#include <limits>
#include <list>
#include <optional>
#include <queue>
#include <ratio>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/Support/Compiler.h"
#include "task_queue.h"
#include "tfrt/host_context/task_function.h"
#include "work_queue_base.h"

namespace tfrt {
namespace internal {

template <typename ThreadingEnvironment>
class BlockingWorkQueue;

template <typename ThreadingEnvironmentTy>
struct WorkQueueTraits<BlockingWorkQueue<ThreadingEnvironmentTy>> {
  using ThreadingEnvironment = ThreadingEnvironmentTy;
  using Thread = typename ThreadingEnvironment::Thread;
  using Queue = ::tfrt::internal::TaskQueue;
};

template <typename ThreadingEnvironment>
class BlockingWorkQueue
    : public WorkQueueBase<BlockingWorkQueue<ThreadingEnvironment>> {
  using Base = WorkQueueBase<BlockingWorkQueue<ThreadingEnvironment>>;

  using Queue = typename Base::Queue;
  using Thread = typename Base::Thread;
  using PerThread = typename Base::PerThread;
  using ThreadData = typename Base::ThreadData;

 public:
  explicit BlockingWorkQueue(
      QuiescingState* quiescing_state, int num_threads,
      std::string_view thread_name_prefix = "",
      std::string_view dynamic_thread_name_prefix = "",
      int max_num_dynamic_threads = std::numeric_limits<int>::max(),
      std::chrono::nanoseconds idle_wait_time = std::chrono::seconds(1));
  ~BlockingWorkQueue() { Quiesce(); }

  // Enqueues `task` for execution by one of the statically allocated thread.
  // Return task wrapped in optional if all per-thread queues are full.
  std::optional<TaskFunction> EnqueueBlockingTask(TaskFunction task);

  // Runs `task` in one of the dynamically started threads. Returns task
  // wrapped in optional if can't assign it to a worker thread.
  std::optional<TaskFunction> RunBlockingTask(TaskFunction task);

  void Quiesce();

 private:
  static constexpr char const* kThreadNamePrefix = "tfrt-blocking-queue";
  static constexpr char const* kDynamicThreadNamePrefix = "tfrt-dynamic-queue";

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

  // If the blocking task does not allow queuing, it is executed in one of the
  // dynamically spawned threads. These threads have 1-to-1 task-to-thread
  // relationship, and can guarantee that tasks with inter-dependencies
  // will all make progress together.

  // Waits for the next available task. Returns empty optional if the task was
  // not found.
  std::optional<TaskFunction> WaitNextTask(mutex_lock* lock)
      TFRT_REQUIRES(mutex_);

  std::string dynamic_thread_name_prefix_;

  // Maximum number of dynamically started threads.
  const int max_num_dynamic_threads_;

  // For how long dynamically started thread waits for the next task before
  // stopping.
  const std::chrono::nanoseconds idle_wait_time_;

  // All operations with dynamic threads are done holding this mutex.
  mutex mutex_;
  condition_variable wake_do_work_cv_;
  condition_variable thread_exited_cv_;

  // Number of started dynamic threads.
  int num_dynamic_threads_ TFRT_GUARDED_BY(mutex_) = 0;

  // Number of dynamic threads waiting for the next task.
  int num_idle_dynamic_threads_ TFRT_GUARDED_BY(mutex_) = 0;

  // This queue is a temporary storage to transfer task ownership to one of the
  // idle threads. It does not keep more tasks than there are idle threads.
  std::queue<TaskFunction> idle_task_queue_ TFRT_GUARDED_BY(mutex_);

  // Unique pointer owning a dynamic thread, and an active flag.
  using DynamicThread = std::pair<std::unique_ptr<Thread>, bool>;

  // Container for dynamically started threads. Some of the threads might be
  // already terminated. Terminated threads lazily removed from the
  // `dynamic_threads_` on each call to `RunBlockingTask`.
  std::list<DynamicThread> dynamic_threads_ TFRT_GUARDED_BY(mutex_);

  // Idle threads must stop waiting for the next task in the `idle_task_queue_`.
  bool stop_waiting_ TFRT_GUARDED_BY(mutex_) = false;
};

template <typename ThreadingEnvironment>
BlockingWorkQueue<ThreadingEnvironment>::BlockingWorkQueue(
    QuiescingState* quiescing_state, int num_threads,
    std::string_view thread_name_prefix,
    std::string_view dynamic_thread_name_prefix, int max_num_dynamic_threads,
    std::chrono::nanoseconds idle_wait_time)
    : WorkQueueBase<BlockingWorkQueue>(
          quiescing_state,
          thread_name_prefix.empty() ? kThreadNamePrefix : thread_name_prefix,
          num_threads),
      dynamic_thread_name_prefix_(dynamic_thread_name_prefix),
      max_num_dynamic_threads_(max_num_dynamic_threads),
      idle_wait_time_(idle_wait_time) {}

template <typename ThreadingEnvironment>
std::optional<TaskFunction>
BlockingWorkQueue<ThreadingEnvironment>::EnqueueBlockingTask(
    TaskFunction task) {
  // In quiescing mode we count the number of pending tasks, and are allowed to
  // execute tasks in the caller thread.
  const bool is_quiescing = IsQuiescing();
  if (is_quiescing) task = WithPendingTaskCounter(std::move(task));

  // If the worker queue is full, we will return `task` to the caller.
  std::optional<TaskFunction> inline_task = {std::move(task)};

  PerThread* pt = GetPerThread();
  if (pt->parent == this) {
    // Worker thread of this pool, push onto the thread's queue.
    Queue& q = thread_data_[pt->thread_id].queue;
    inline_task = q.PushFront(std::move(*inline_task));
  } else {
    // A random free-standing thread (or worker of another pool).
    unsigned r = pt->rng();
    unsigned victim = FastReduce(r, num_threads_);
    unsigned inc = coprimes_[FastReduce(r, coprimes_.size())];

    for (unsigned i = 0; i < num_threads_ && inline_task.has_value(); i++) {
      inline_task =
          thread_data_[victim].queue.PushFront(std::move(*inline_task));
      if ((victim += inc) >= num_threads_) victim -= num_threads_;
    }
  }

  // Failed to push task into one of the worker threads queues.
  if (inline_task.has_value()) {
    // If we are in quiescing mode, we can always execute the submitted task in
    // the caller thread, because the system is anyway going to shutdown soon,
    // and even if we are running inside a non-blocking work queue, a single
    // potential context switch won't negatively impact system performance.
    if (is_quiescing) {
      (*inline_task)();
      return std::nullopt;
    } else {
      return inline_task;
    }
  }

  // Note: below we touch `*this` after making `task` available to worker
  // threads. Strictly speaking, this can lead to a racy-use-after-free.
  // Consider that Schedule is called from a thread that is neither main thread
  // nor a worker thread of this pool. Then, execution of `task` directly or
  // indirectly completes overall computations, which in turn leads to
  // destruction of this. We expect that such a scenario is prevented by the
  // program, that is, this is kept alive while any threads can potentially be
  // in Schedule.
  if (IsNotifyParkedThreadRequired()) event_count_.Notify(false);

  return std::nullopt;
}

template <typename ThreadingEnvironment>
std::optional<TaskFunction>
BlockingWorkQueue<ThreadingEnvironment>::RunBlockingTask(TaskFunction task) {
  mutex_lock lock(mutex_);

  // Attach a PendingTask counter only if we were able to submit the task
  // to one of the worker threads. It's unsafe to return the task with
  // a counter to the caller, because we don't know when/if it will be
  // destructed and the counter decremented.
  auto wrap = [&](TaskFunction task) -> TaskFunction {
    return IsQuiescing() ? WithPendingTaskCounter(std::move(task))
                         : std::move(task);
  };

  // There are idle threads. We enqueue the task to the queue and then notify
  // one of the idle threads.
  if (idle_task_queue_.size() < num_idle_dynamic_threads_) {
    idle_task_queue_.emplace(wrap(std::move(task)));
    wake_do_work_cv_.notify_one();

    return std::nullopt;
  }

  // Cleanup dynamic threads that are already terminated.
  dynamic_threads_.remove_if(
      [](DynamicThread& thread) -> bool { return thread.second == false; });

  // There are no idle threads and we are not at the thread limit. We
  // start a new thread to run the task.
  if (num_dynamic_threads_ < max_num_dynamic_threads_) {
    // Prepare an entry to hold a new dynamic thread.
    //
    // NOTE: We rely on std::list pointer stability for passing a reference to
    // the container element to the `do_work` lambda.
    dynamic_threads_.emplace_back();
    DynamicThread& dynamic_thread = dynamic_threads_.back();

    auto do_work = [this, &dynamic_thread,
                    task = wrap(std::move(task))]() mutable {
      task();
      // Reset executed task to call destructor without holding the lock,
      // because it might be expensive. Also we want to call it before
      // notifying quiescing thread, because destructor potentially could
      // drop the last references on captured async values.
      task = nullptr;

      mutex_lock lock(mutex_);

      // Try to get the next task. If one is found, run it. If there is no
      // task to execute, GetNextTask will return None that converts to
      // false.
      while (std::optional<TaskFunction> task = WaitNextTask(&lock)) {
        mutex_.unlock();
        // Do not hold the lock while executing and destructing the task.
        (*task)();
        task = nullptr;
        mutex_.lock();
      }

      // No more work to do or shutdown occurred. Exit the thread.
      dynamic_thread.second = false;
      --num_dynamic_threads_;
      if (stop_waiting_) thread_exited_cv_.notify_one();
    };

    // Start a new dynamic thread.
    dynamic_thread.second = true;  // is active
    dynamic_thread.first = ThreadingEnvironment::StartThread(
        dynamic_thread_name_prefix_.empty() ? kDynamicThreadNamePrefix
                                            : dynamic_thread_name_prefix_,
        std::move(do_work));
    ++num_dynamic_threads_;

    return std::nullopt;
  }

  // There are no idle threads and we are at the thread limit. Return task
  // to the caller.
  return {std::move(task)};
}

template <typename ThreadingEnvironment>
std::optional<TaskFunction>
BlockingWorkQueue<ThreadingEnvironment>::WaitNextTask(mutex_lock* lock) {
  ++num_idle_dynamic_threads_;

  const auto timeout = std::chrono::system_clock::now() + idle_wait_time_;
  wake_do_work_cv_.wait_until(*lock, timeout, [this]() TFRT_REQUIRES(mutex_) {
    return !idle_task_queue_.empty() || stop_waiting_;
  });
  --num_idle_dynamic_threads_;

  // Found something in the queue. Return the task.
  if (!idle_task_queue_.empty()) {
    TaskFunction task = std::move(idle_task_queue_.front());
    idle_task_queue_.pop();
    return {std::move(task)};
  }

  // Shutdown occurred. Return empty optional.
  return std::nullopt;
}

template <typename ThreadingEnvironment>
void BlockingWorkQueue<ThreadingEnvironment>::Quiesce() {
  Base::Quiesce();

  // WARN: This function provides only best-effort work queue emptyness
  // guarantees. Tasks running inside a dynamically allocated threads
  // potentially could submit new tasks to statically allocated threads, and
  // current implementaton will miss them. Clients must rely on
  // MultiThreadedWorkQueue::Quiesce() for strong emptyness guarantees.

  // Wait for the completion of all tasks in the dynamicly part of a queue.
  mutex_lock lock(mutex_);

  // Wake up all idle threads.
  stop_waiting_ = true;
  wake_do_work_cv_.notify_all();

  // Wait until all dynamicaly started threads stopped.
  thread_exited_cv_.wait(lock, [this]() TFRT_REQUIRES(mutex_) {
    return num_dynamic_threads_ == 0;
  });
  assert(idle_task_queue_.empty());

  // Prepare for the next call to Quiesce.
  stop_waiting_ = false;
}

template <typename ThreadingEnvironment>
[[nodiscard]] std::optional<TaskFunction>
BlockingWorkQueue<ThreadingEnvironment>::NextTask(Queue* queue) {
  return queue->PopBack();
}

template <typename ThreadingEnvironment>
[[nodiscard]] std::optional<TaskFunction>
BlockingWorkQueue<ThreadingEnvironment>::Steal(Queue* queue) {
  return queue->PopBack();
}

template <typename ThreadingEnvironment>
[[nodiscard]] bool BlockingWorkQueue<ThreadingEnvironment>::Empty(
    Queue* queue) {
  return queue->Empty();
}

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_BLOCKING_WORK_QUEUE_H_
