// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Base class for concurrent work queue implementation. Derived work queues
// define how pending tasks are stored and how stealing algorithm works.
//
// All threads managed by this work queue have their own pending tasks queue of
// a fixed size. Thread worker loop tries to get the next task from its own
// queue first, if the queue is empty it's going into a steal loop, and tries to
// steal the next task from the other thread pending tasks queue.
//
// If a thread was not able to find the next task function to execute, it's
// parked on a conditional variable, and waits for the notification from the
// new task added to the queue.
//
// Before parking on a conditional variable, thread might go into a spin loop
// (controlled by `kMaxSpinningThreads` constant), and execute steal loop for a
// fixed number of iterations. This allows to skip expensive park/unpark
// operations, and reduces latency. Increasing `kMaxSpinningThreads` improves
// latency at the cost of burned CPU cycles.
//
// See derived work queue implementation for more details about work stealing.
//
// -------------------------------------------------------------------------- //
// Work queue implementations are parametrized by `ThreadingEnvironment` that
// allows to provide custom thread implementation:
//
//  struct ThreadingEnvironment {
//    // Type alias for the underlying thread implementation.
//    using Thread = ...
//
//    // Starts a new thread running function `f` with arguments `arg`.
//    template <class Function, class... Args>
//    std::unique_ptr<Thread> StartThread(string_view name_prefix,
//         Function&& f, Args&&... args) { ... }
//
//    // Returns current thread id hash code. Must have characteristics of a
//    // good hash function and generate uniformly distributed values. Values
//    // are used as an initial seed for per-thread random number generation.
//    static uint64_t ThisThreadIdHash() {... }
//  }

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_WORK_QUEUE_BASE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_WORK_QUEUE_BASE_H_

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "event_count.h"
#include "llvm/Support/Compiler.h"
#include "task_queue.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace internal {

struct FastRng {
  constexpr explicit FastRng(uint64_t state) : state(state) {}

  unsigned operator()() {
    uint64_t current = state;
    // Update the internal state
    state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22u)) >>
                                 (22 + (current >> 61u)));
  }

  uint64_t state;
};

// Reduce `x` into [0, size) range (compute `x % size`).
// https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction
inline uint32_t FastReduce(uint32_t x, uint32_t size) {
  return (static_cast<uint64_t>(x) * static_cast<uint64_t>(size)) >> 32u;
}

template <typename Derived>
struct WorkQueueTraits;

//===----------------------------------------------------------------------===//
// Quiescing enables pending tasks counter to implement strong work queue
// emptiness check in the MultiThreadedWorkQueue::Quiesce() implementation.
//===----------------------------------------------------------------------===//
struct QuiescingState {
  std::atomic<int64_t> num_quiescing;
  std::atomic<int64_t> num_pending_tasks;
};

//===----------------------------------------------------------------------===//
// MultithreadedWorkQueue::Quiesce() requires strong guarantees for queue
// emptyness check. If quiescing mode is enabled (via creating an instance of
// Quiescing), work queue tracks the number of pending tasks.
//===----------------------------------------------------------------------===//
class Quiescing {
 public:
  static Quiescing Start(QuiescingState* state) { return Quiescing(state); }

  ~Quiescing() {
    if (state_ == nullptr) return;  // in moved-out state
    state_->num_quiescing.fetch_sub(1, std::memory_order_relaxed);
  }

  Quiescing(Quiescing&& other) : state_(other.state_) {
    other.state_ = nullptr;
  }

  Quiescing& operator=(Quiescing&& other) {
    state_ = other.state_;
    other.state_ = nullptr;
    return *this;
  }

  Quiescing(const Quiescing&) = delete;
  Quiescing& operator=(const Quiescing&) = delete;

  // HasPendingTasks() returns true if some of the tasks added to the owning
  // queue after `*this` was created are not completed.
  bool HasPendingTasks() const {
    return state_->num_pending_tasks.load(std::memory_order_relaxed) != 0;
  }

 private:
  explicit Quiescing(QuiescingState* state) : state_(state) {
    assert(state != nullptr);
    state_->num_quiescing.fetch_add(1, std::memory_order_relaxed);
  }

  QuiescingState* state_;
};

//===----------------------------------------------------------------------===//
// RAII helper for keeping track of the number of pending tasks.
//===----------------------------------------------------------------------===//
class PendingTask {
 public:
  explicit PendingTask(QuiescingState* state) : state_(state) {
    assert(state != nullptr);
    state_->num_pending_tasks.fetch_add(1, std::memory_order_relaxed);
  }

  ~PendingTask() {
    if (state_ == nullptr) return;  // in moved-out state
    state_->num_pending_tasks.fetch_sub(1, std::memory_order_relaxed);
  }

  PendingTask(PendingTask&& other) : state_(other.state_) {
    other.state_ = nullptr;
  }

  PendingTask& operator=(PendingTask&& other) {
    state_ = other.state_;
    other.state_ = nullptr;
    return *this;
  }

  PendingTask(const PendingTask&) = delete;
  PendingTask& operator=(const PendingTask&) = delete;

 private:
  QuiescingState* state_;
};

//===----------------------------------------------------------------------===//
// Work queue base class (derived by non-blocking and blocking work queues).
//===----------------------------------------------------------------------===//
template <typename Derived>
class WorkQueueBase {
  using Traits = WorkQueueTraits<Derived>;

  using Queue = typename Traits::Queue;
  using Thread = typename Traits::Thread;
  using ThreadingEnvironment = typename Traits::ThreadingEnvironment;

 public:
  // ------------------------------------------------------------------------ //
  // Derived work queue must implement these methods:

  // NextTask() returns the next task for a worker thread from its own queue.
  // Returns None if per thread queue of pending tasks is empty. This method
  // should be called only from a thread that owns `queue`.
  //
  // std::optional<TaskFunction> Derived::NextTask(Queue* queue);

  // Steal() tries to steal task from a specified queue of pending tasks from
  // another thread managed by `this`. Return None if was not able to steal a
  // task (queue is empty, or steal spuriosly failed under contention).
  //
  // std::optional<TaskFunction> Derived::Steal(Queue* queue);

  // Empty() returns true if the queue is empty. It must reliably detect if
  // the queue is not empty, because it's critical for keeping worker threads
  // alive and avoiding deadlocks.
  //
  // Empty() returns true if the `queue` is empty.
  //
  // bool Derived::Empty(Queue* queue);

  // ------------------------------------------------------------------------ //

  bool IsQuiescing() const {
    return quiescing_state_->num_quiescing.load(std::memory_order_relaxed) > 0;
  }

  // Quiesce blocks caller thread until all submitted tasks are completed and
  // all worker threads are in the parked state.
  void Quiesce();

  // Steal() tries to steal task from any worker managed by this queue. Returns
  // std::nullopt if it was not able to find task to steal.
  [[nodiscard]] std::optional<TaskFunction> Steal();

  // Returns `true` if all worker threads are parked. This is a weak signal of
  // work queue emptyness, because worker thread might be notified, but not
  // yet unparked and running. For strong guarantee must use use Quiesce.
  bool AllBlocked() const { return NumBlockedThreads() == num_threads_; }

  // CheckCallerThread() will abort the program or issue warning message if the
  // caller thread is managed by `*this`. This is required to prevent deadlocks
  // from calling `Quiesce` from a thread managed by the current worker queue.
  void CheckCallerThread(const char* function_name, bool is_fatal) const;

  // Returns true if the caller thread is managed by this work queue.
  bool IsInWorkerThread() const {
    PerThread* per_thread = GetPerThread();
    return per_thread->parent == &derived_;
  }

  // Stop all threads managed by this work queue.
  void Cancel();

 private:
  template <typename ThreadingEnvironment>
  friend class BlockingWorkQueue;

  template <typename ThreadingEnvironment>
  friend class NonBlockingWorkQueue;

  struct PerThread {
    constexpr PerThread() : parent(nullptr), rng(0), thread_id(-1) {}
    Derived* parent;
    FastRng rng;    // Random number generator
    int thread_id;  // Worker thread index in the workers queue
  };

  struct ThreadData {
    ThreadData() : thread(), queue() {}
    std::unique_ptr<Thread> thread;
    Queue queue;
  };

  // Returns a TaskFunction with an attached pending tasks counter, if the
  // quiescing mode is on.
  TaskFunction WithPendingTaskCounter(TaskFunction task) {
    return TaskFunction(
        [task = std::move(task), p = PendingTask(quiescing_state_)]() mutable {
          task();
        });
  }

  // TODO(ezhulenev): Make this a runtime parameter? More spinning threads help
  // to reduce latency at the cost of wasted CPU cycles.
  static constexpr int kMaxSpinningThreads = 1;

  // The number of steal loop spin iterations before parking (this number is
  // divided by the number of threads, to get spin count for each thread).
  static constexpr int kSpinCount = 5000;

  // If there are enough active threads with an empty pending task queues, there
  // is no need for spinning before parking a thread that is out of work to do,
  // because these active threads will go into a steal loop after finishing with
  // their current tasks.
  //
  // In the worst case when all active threads are executing long/expensive
  // tasks, the next AddTask() will have to wait until one of the parked threads
  // will be unparked, however this should be very rare in practice.
  static constexpr int kMinActiveThreadsToStartSpinning = 4;

  explicit WorkQueueBase(QuiescingState* quiescing_state,
                         string_view name_prefix, int num_threads);
  ~WorkQueueBase();

  // Main worker thread loop.
  void WorkerLoop(int thread_id);

  // WaitForWork() blocks until new work is available (returns true), or if it
  // is time to exit (returns false). Can optionally return a task to execute in
  // `task` (in such case `task.has_value() == true` on return).
  [[nodiscard]] bool WaitForWork(EventCount::Waiter* waiter,
                                 std::optional<TaskFunction>* task);

  // StartSpinning() checks if the number of threads in the spin loop is less
  // than the allowed maximum, if so increments the number of spinning threads
  // by one and returns true (caller must enter the spin loop). Otherwise
  // returns false, and the caller must not enter the spin loop.
  [[nodiscard]] bool StartSpinning();

  // StopSpinning() decrements the number of spinning threads by one. It also
  // checks if there were any tasks submitted into the pool without notifying
  // parked threads, and decrements the count by one. Returns true if the number
  // of tasks submitted without notification was decremented, in this case
  // caller thread might have to call Steal() one more time.
  [[nodiscard]] bool StopSpinning();

  // IsNotifyParkedThreadRequired() returns true if parked thread must be
  // notified about new added task. If there are threads spinning in the steal
  // loop, there is no need to unpark any of the waiting threads, the task will
  // be picked up by one of the spinning threads.
  [[nodiscard]] bool IsNotifyParkedThreadRequired();

  void Notify() { event_count_.Notify(false); }

  // Returns current thread id if the caller thread is managed by `this`,
  // returns `-1` otherwise.
  int CurrentThreadId() const;

  // NonEmptyQueueIndex() returns the index of a non-empty worker queue, or `-1`
  // if all queues are empty.
  [[nodiscard]] int NonEmptyQueueIndex();

  [[nodiscard]] static PerThread* GetPerThread() {
    static thread_local PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
  }

  unsigned NumBlockedThreads() const { return blocked_.load(); }
  unsigned NumActiveThreads() const { return num_threads_ - blocked_.load(); }

  const int num_threads_;

  std::vector<ThreadData> thread_data_;
  std::vector<unsigned> coprimes_;

  std::atomic<unsigned> blocked_;
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;

  // Use a conditional variable to notify waiters when all worker threads are
  // blocked. This is used to park caller thread in Quiesce() when there is no
  // work to steal, but not all threads are parked.
  mutex all_blocked_mu_;
  condition_variable all_blocked_cv_;

  // All work queues composed together in a single logical work queue, must
  // share a quiescing state to guarantee correct emptyness check.
  QuiescingState* quiescing_state_;

  // Spinning state layout:
  // - Low 32 bits encode the number of threads that are spinning in steal loop.
  //
  // - High 32 bits encode the number of tasks that were submitted to the pool
  //   without a call to event_count_.Notify(). This number can't be larger than
  //   the number of spinning threads. Each spinning thread, when it exits the
  //   spin loop must check if this number is greater than zero, and maybe make
  //   another attempt to steal a task and decrement it by one.
  static constexpr uint64_t kNumSpinningBits = 32;
  static constexpr uint64_t kNumSpinningMask = (1ull << kNumSpinningBits) - 1;
  static constexpr uint64_t kNumNoNotifyBits = 32;
  static constexpr uint64_t kNumNoNotifyShift = 32;
  static constexpr uint64_t kNumNoNotifyMask = ((1ull << kNumNoNotifyBits) - 1)
                                               << kNumNoNotifyShift;
  std::atomic<uint64_t> spinning_state_;

  struct SpinningState {
    uint64_t num_spinning;         // number of spinning threads
    uint64_t num_no_notification;  // number of tasks submitted without
                                   // notifying waiting threads

    // Decode `spinning_state_` value.
    static SpinningState Decode(uint64_t state) {
      uint64_t num_spinning = (state & kNumSpinningMask);
      uint64_t num_no_notification =
          (state & kNumNoNotifyMask) >> kNumNoNotifyShift;

      assert(num_no_notification <= num_spinning);

      return {num_spinning, num_no_notification};
    }

    // Encode as `spinning_state_` value.
    uint64_t Encode() const {
      return (num_no_notification << kNumNoNotifyShift) | num_spinning;
    }
  };

  EventCount event_count_;
  Derived& derived_;
};

// Calculate coprimes of all numbers [1, n].
//
// Coprimes are used for random walks over all threads in Steal
// and NonEmptyQueueIndex. Iteration is based on the fact that if we take a
// random starting thread index `t` and calculate `num_threads - 1` subsequent
// indices as `(t + coprime) % num_threads`, we will cover all threads without
// repetitions (effectively getting a pseudo-random permutation of thread
// indices).
inline std::vector<unsigned> ComputeCoprimes(int n) {
  std::vector<unsigned> coprimes;
  for (unsigned i = 1; i <= n; i++) {
    unsigned a = n;
    unsigned b = i;
    // If GCD(a, b) == 1, then a and b are coprimes.
    while (b != 0) {
      unsigned tmp = a;
      a = b;
      b = tmp % b;
    }
    if (a == 1) coprimes.push_back(i);
  }
  return coprimes;
}

template <typename Derived>
WorkQueueBase<Derived>::WorkQueueBase(QuiescingState* quiescing_state,
                                      string_view name_prefix, int num_threads)
    : num_threads_(num_threads),
      thread_data_(num_threads),
      coprimes_(ComputeCoprimes(num_threads)),
      blocked_(0),
      done_(false),
      cancelled_(false),
      quiescing_state_(quiescing_state),
      spinning_state_(0),
      event_count_(num_threads),
      derived_(static_cast<Derived&>(*this)) {
  assert(num_threads >= 1);
  for (int i = 0; i < num_threads; i++) {
    thread_data_[i].thread = ThreadingEnvironment::StartThread(
        name_prefix, [this, i]() { WorkerLoop(i); });
  }
}

template <typename Derived>
WorkQueueBase<Derived>::~WorkQueueBase() {
  done_ = true;

  // Now if all threads block without work, they will start exiting.
  // But note that threads can continue to work arbitrary long,
  // block, submit new work, unblock and otherwise live full life.
  if (!cancelled_) {
    event_count_.Notify(true);
  } else {
    // Since we were cancelled, there might be entries in the queues.
    // Empty them to prevent their destructor from asserting.
    for (ThreadData& thread_data : thread_data_) {
      thread_data.queue.Flush();
    }
  }
  // All worker threads joined in destructors.
  for (ThreadData& thread_data : thread_data_) {
    thread_data.thread.reset();
  }
}

template <typename Derived>
void WorkQueueBase<Derived>::CheckCallerThread(const char* function_name,
                                               bool is_fatal) const {
  PerThread* pt = GetPerThread();
  if (is_fatal) {
    TFRT_LOG_IF(FATAL, pt->parent == this)
        << "Error at " << __FILE__ << ":" << __LINE__ << ": " << function_name
        << " should not be called by a work thread already managed by the "
           "queue.";
  } else {
    TFRT_DLOG_IF(WARNING, pt->parent == this)
        << "Warning at " << __FILE__ << ":" << __LINE__ << ": " << function_name
        << " should not be called by a work thread already managed by the "
           "queue.";
  }
}

template <typename Derived>
void WorkQueueBase<Derived>::Quiesce() {
  CheckCallerThread("WorkQueueBase::Quiesce", /*is_fatal=*/true);

  // Keep stealing tasks until we reach a point when we have nothing to steal
  // and all worker threads are in blocked state.
  std::optional<TaskFunction> task = Steal();

  while (task.has_value()) {
    // Execute stolen task in the caller thread.
    (*task)();

    // Try to steal the next task.
    task = Steal();
  }

  // If we didn't find a task to execute, and there are still worker threads
  // running, park current thread on a conditional variable until all worker
  // threads are blocked.
  if (!AllBlocked()) {
    mutex_lock lock(all_blocked_mu_);
    all_blocked_cv_.wait(
        lock, [this]() TFRT_REQUIRES(all_blocked_mu_) { return AllBlocked(); });
  }
}

template <typename Derived>
[[nodiscard]] std::optional<TaskFunction> WorkQueueBase<Derived>::Steal() {
  PerThread* pt = GetPerThread();
  unsigned r = pt->rng();
  unsigned victim = FastReduce(r, num_threads_);
  unsigned inc = coprimes_[FastReduce(r, coprimes_.size())];

  for (unsigned i = 0; i < num_threads_; i++) {
    std::optional<TaskFunction> t =
        derived_.Steal(&(thread_data_[victim].queue));
    if (t.has_value()) return t;

    victim += inc;
    if (victim >= num_threads_) {
      victim -= num_threads_;
    }
  }
  return std::nullopt;
}

template <typename Derived>
void WorkQueueBase<Derived>::WorkerLoop(int thread_id) {
  PerThread* pt = GetPerThread();
  pt->parent = &derived_;
  pt->rng = FastRng(ThreadingEnvironment::ThisThreadIdHash());
  pt->thread_id = thread_id;

  Queue* q = &(thread_data_[thread_id].queue);
  EventCount::Waiter* waiter = event_count_.waiter(thread_id);

  // TODO(dvyukov,rmlarsen): The time spent in NonEmptyQueueIndex() is
  // proportional to num_threads_ and we assume that new work is scheduled at
  // a constant rate, so we set spin_count to 5000 / num_threads_. The
  // constant was picked based on a fair dice roll, tune it.
  const int spin_count = num_threads_ > 0 ? kSpinCount / num_threads_ : 0;

  while (!cancelled_) {
    std::optional<TaskFunction> t = derived_.NextTask(q);
    if (!t.has_value()) {
      t = Steal();
      if (!t.has_value()) {
        // Maybe leave thread spinning. This reduces latency.
        const bool start_spinning = StartSpinning();
        if (start_spinning) {
          for (int i = 0; i < spin_count && !t.has_value(); ++i) {
            t = Steal();
          }

          const bool stopped_spinning = StopSpinning();
          // If a task was submitted to the queue without a call to
          // `event_count_.Notify()`, and we didn't steal anything above, we
          // must try to steal one more time, to make sure that this task will
          // be executed. We will not necessarily find it, because it might have
          // been already stolen by some other thread.
          if (stopped_spinning && !t.has_value()) {
            t = Steal();
          }
        }

        if (!t.has_value()) {
          if (!WaitForWork(waiter, &t)) {
            return;
          }
          if (!t.has_value()) {
            continue;
          }
        }
      }
    }
    // When reaching here, the task is always available:
    // Case 0: there are pending tasks in the queue, NextTask returns a task;
    // Case 1: no pending task in the queue, stealed a task from others';
    // Case 2: no pending task in the queue and cannot steal a task, WaitForWork
    //         returns a new enqueued task.
    assert(t.has_value());
    (*t)();  // Execute a task.
  }
}

template <typename Derived>
bool WorkQueueBase<Derived>::WaitForWork(EventCount::Waiter* waiter,
                                         std::optional<TaskFunction>* task) {
  assert(!task->has_value());
  // We already did best-effort emptiness check in Steal, so prepare for
  // blocking.
  event_count_.Prewait();
  // Now do a reliable emptiness check.
  int victim = NonEmptyQueueIndex();
  if (victim != -1) {
    event_count_.CancelWait();
    if (cancelled_) {
      return false;
    } else {
      *task = derived_.Steal(&(thread_data_[victim].queue));
      return true;
    }
  }
  // Number of blocked threads is used as termination condition.
  // If we are shutting down and all worker threads blocked without work,
  // that's we are done.
  blocked_.fetch_add(1);

  // Notify threads that are waiting for "all blocked" event.
  if (blocked_.load() == static_cast<unsigned>(num_threads_)) {
    mutex_lock lock(all_blocked_mu_);
    all_blocked_cv_.notify_all();
  }

  // Prepare to shutdown worker thread if done.
  if (done_ && blocked_.load() == static_cast<unsigned>(num_threads_)) {
    event_count_.CancelWait();
    // Almost done, but need to re-check queues.
    // Consider that all queues are empty and all worker threads are preempted
    // right after incrementing blocked_ above. Now a free-standing thread
    // submits work and calls destructor (which sets done_). If we don't
    // re-check queues, we will exit leaving the work unexecuted.
    if (NonEmptyQueueIndex() != -1) {
      // Note: we must not pop from queues before we decrement blocked_,
      // otherwise the following scenario is possible. Consider that instead
      // of checking for emptiness we popped the only element from queues.
      // Now other worker threads can start exiting, which is bad if the
      // work item submits other work. So we just check emptiness here,
      // which ensures that all worker threads exit at the same time.
      blocked_.fetch_sub(1);
      return true;
    }
    // Reached stable termination state.
    event_count_.Notify(true);
    return false;
  }

  event_count_.CommitWait(waiter);
  blocked_.fetch_sub(1);
  return true;
}

template <typename Derived>
bool WorkQueueBase<Derived>::StartSpinning() {
  if (NumActiveThreads() > kMinActiveThreadsToStartSpinning) return false;

  uint64_t spinning = spinning_state_.load(std::memory_order_relaxed);
  for (;;) {
    SpinningState state = SpinningState::Decode(spinning);

    if ((state.num_spinning - state.num_no_notification) >= kMaxSpinningThreads)
      return false;

    // Increment the number of spinning threads.
    ++state.num_spinning;

    if (spinning_state_.compare_exchange_weak(spinning, state.Encode(),
                                              std::memory_order_relaxed)) {
      return true;
    }
  }
}

template <typename Derived>
bool WorkQueueBase<Derived>::StopSpinning() {
  uint64_t spinning = spinning_state_.load(std::memory_order_relaxed);
  for (;;) {
    SpinningState state = SpinningState::Decode(spinning);

    // Decrement the number of spinning threads.
    --state.num_spinning;

    // Maybe decrement the number of tasks submitted without notification.
    bool has_no_notify_task = state.num_no_notification > 0;
    if (has_no_notify_task) --state.num_no_notification;

    if (spinning_state_.compare_exchange_weak(spinning, state.Encode(),
                                              std::memory_order_relaxed)) {
      return has_no_notify_task;
    }
  }
}

template <typename Derived>
bool WorkQueueBase<Derived>::IsNotifyParkedThreadRequired() {
  uint64_t spinning = spinning_state_.load(std::memory_order_relaxed);
  for (;;) {
    SpinningState state = SpinningState::Decode(spinning);

    // If the number of tasks submitted without notifying parked threads is
    // equal to the number of spinning threads, we must wake up one of the
    // parked threads.
    if (state.num_no_notification == state.num_spinning) return true;

    // Increment the number of tasks submitted without notification.
    ++state.num_no_notification;

    if (spinning_state_.compare_exchange_weak(spinning, state.Encode(),
                                              std::memory_order_relaxed)) {
      return false;
    }
  }
}

template <typename Derived>
int WorkQueueBase<Derived>::NonEmptyQueueIndex() {
  PerThread* pt = GetPerThread();
  unsigned r = pt->rng();
  unsigned inc = num_threads_ == 1 ? 1 : coprimes_[r % coprimes_.size()];
  unsigned victim = FastReduce(r, num_threads_);
  for (unsigned i = 0; i < num_threads_; i++) {
    if (!derived_.Empty(&(thread_data_[victim].queue))) {
      return static_cast<int>(victim);
    }
    victim += inc;
    if (victim >= num_threads_) {
      victim -= num_threads_;
    }
  }
  return -1;
}

template <typename Derived>
int WorkQueueBase<Derived>::CurrentThreadId() const {
  const PerThread* pt = GetPerThread();
  if (pt->parent == this) {
    return pt->thread_id;
  } else {
    return -1;
  }
}

template <typename Derived>
void WorkQueueBase<Derived>::Cancel() {
  cancelled_ = true;
  done_ = true;

  // Wake up the threads without work to let them exit on their own.
  event_count_.Notify(true);
}

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_WORK_QUEUE_BASE_H_
