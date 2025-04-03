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

// multi_thread_gemm.h: Multi-threaded GEMM entry point.
// Readers note: To understand this file, it is useful to first
// read and understand the much simpler single_thread_gemm.h.

#ifndef GEMMLOWP_INTERNAL_MULTI_THREAD_GEMM_H_
#define GEMMLOWP_INTERNAL_MULTI_THREAD_GEMM_H_

#include <atomic>  // NOLINT
#include <chrono>  // NOLINT
#include <thread>  // NOLINT
#include <vector>

#include "single_thread_gemm.h"

namespace gemmlowp {

// This value was empirically derived on an end-to-end application benchmark.
// That this number of cycles means that we may be sleeping substantially longer
// than a scheduler timeslice's duration is not necessarily surprising. The
// idea is to pick up quickly new work after having finished the previous
// workload. When it's new work within the same GEMM as the previous work, the
// time interval that we might be busy-waiting is very small, so for that
// purpose it would be more than enough to sleep for 1 million cycles.
// That is all what we would observe on a GEMM benchmark. However, in a real
// application, after having finished a GEMM, we might do unrelated work for
// a little while, then start on a new GEMM. Think of a neural network
// application performing inference, where many but not all layers are
// implemented by a GEMM. In such cases, our worker threads might be idle for
// longer periods of time before having work again. If we let them passively
// wait, on a mobile device, the CPU scheduler might aggressively clock down
// or even turn off the CPU cores that they were running on. That would result
// in a long delay the next time these need to be turned back on for the next
// GEMM. So we need to strike a balance that reflects typical time intervals
// between consecutive GEMM invokations, not just intra-GEMM considerations.
// Of course, we need to balance keeping CPUs spinning longer to resume work
// faster, versus passively waiting to conserve power.
const int kMaxBusyWaitNOPs = 4 * 1000 * 1000;

// On X86 and ARM platforms we may use NOP instructions to know how long we
// are busy-waiting.

#if defined(GEMMLOWP_ALLOW_INLINE_ASM) && !defined(GEMMLOWP_NO_BUSYWAIT) && \
    (defined(GEMMLOWP_ARM) || defined(GEMMLOWP_X86))

#define GEMMLOWP_NOP "nop\n"

#define GEMMLOWP_STRING_CONCAT_4(X) X X X X
#define GEMMLOWP_NOP4 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP)
#define GEMMLOWP_NOP16 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP4)
#define GEMMLOWP_NOP64 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP16)

inline int DoSomeNOPs() {
  asm volatile(GEMMLOWP_NOP64);
  return 64;
}

#undef GEMMLOWP_STRING_CONCAT_4
#undef GEMMLOWP_NOP64
#undef GEMMLOWP_NOP16
#undef GEMMLOWP_NOP4
#undef GEMMLOWP_NOP

#else  // May not use asm NOP.

// If we can't use NOPs, let's use a non-inline function call as a basic
// thing that has some vaguely known, nonzero cost.
GEMMLOWP_NOINLINE
inline int DoSomeNOPs() {
  // Pretend that calling an empty function takes as long as 16 NOPs...
  return 16;
}
#endif

// Waits until *var != initial_value.
//
// Returns the new value of *var. The guarantee here is that
// the return value is different from initial_value, and that that
// new value has been taken by *var at some point during the
// execution of this function. There is no guarantee that this is
// still the value of *var when this function returns, since *var is
// not assumed to be guarded by any lock.
//
// First does some busy-waiting for a fixed number of no-op cycles,
// then falls back to passive waiting for the given condvar, guarded
// by the given mutex.
//
// The idea of doing some initial busy-waiting is to help get
// better and more consistent multithreading benefits for small GEMM sizes.
// Busy-waiting help ensuring that if we need to wake up soon after having
// started waiting, then we can wake up quickly (as opposed to, say,
// having to wait to be scheduled again by the OS). On the other hand,
// we must still eventually revert to passive waiting for longer waits
// (e.g. worker threads having finished a GEMM and waiting until the next GEMM)
// so as to avoid permanently spinning.
//
template <typename T>
T WaitForVariableChange(std::atomic<T>* var, T initial_value,
                        pthread_cond_t* cond, pthread_mutex_t* mutex) {
  // First, trivial case where the variable already changed value.
  T new_value = var->load(std::memory_order_acquire);
  if (new_value != initial_value) {
    return new_value;
  }
  // Then try busy-waiting.
  int nops = 0;
  while (nops < kMaxBusyWaitNOPs) {
    nops += DoSomeNOPs();
    new_value = var->load(std::memory_order_acquire);
    if (new_value != initial_value) {
      return new_value;
    }
  }

  // Finally, do real passive waiting.
  pthread_mutex_lock(mutex);
  new_value = var->load(std::memory_order_acquire);
  while (new_value == initial_value) {
    pthread_cond_wait(cond, mutex);
    new_value = var->load(std::memory_order_acquire);
  }
  pthread_mutex_unlock(mutex);
  return new_value;
}

// A BlockingCounter lets one thread to wait for N events to occur.
// This is how the master thread waits for all the worker threads
// to have finished working.
// The waiting is done using a naive spinlock waiting for the atomic
// count_ to hit the value 0. This is acceptable because in our usage
// pattern, BlockingCounter is used only to synchronize threads after
// short-lived tasks (performing parts of the same GEMM). It is not used
// for synchronizing longer waits (resuming work on the next GEMM).
class BlockingCounter {
 public:
  BlockingCounter() : count_(0) {}

  // Sets/resets the counter; initial_count is the number of
  // decrementing events that the Wait() call will be waiting for.
  void Reset(std::size_t initial_count) {
    std::size_t old_count_value = count_.load(std::memory_order_relaxed);
    assert(old_count_value == 0);
    (void)old_count_value;
    count_.store(initial_count, std::memory_order_release);
  }

  // Decrements the counter; if the counter hits zero, signals
  // the threads that were waiting for that, and returns true.
  // Otherwise (if the decremented count is still nonzero),
  // returns false.
  bool DecrementCount() {
    std::size_t old_count_value =
        count_.fetch_sub(1, std::memory_order_acq_rel);
    assert(old_count_value > 0);
    std::size_t count_value = old_count_value - 1;
    return count_value == 0;
  }

  // Waits for the N other threads (N having been set by Reset())
  // to hit the BlockingCounter.
  void Wait() {
    ScopedProfilingLabel label("BlockingCounter::Wait");
    // Busy-wait until the count value is 0.
    int nops = 0;
    while (count_.load(std::memory_order_acquire)) {
      nops += DoSomeNOPs();
      if (nops > kMaxBusyWaitNOPs) {
        nops = 0;
        // If we are unlucky, the blocking thread (that calls DecrementCount)
        // and the blocked thread (here, calling Wait) may be scheduled on
        // the same CPU, so the busy-waiting of the present thread may prevent
        // the blocking thread from resuming and unblocking.
        // If we are even unluckier, the priorities of the present thread
        // might be higher than that of the blocking thread, so just yielding
        // wouldn't allow the blocking thread to resume. So we sleep for
        // a substantial amount of time in that case. Notice that we only
        // do so after having busy-waited for kMaxBusyWaitNOPs, which is
        // typically several milliseconds, so sleeping 1 more millisecond
        // isn't terrible at that point.
        //
        // How this is mitigated in practice:
        // In practice, it is well known that the application should be
        // conservative in choosing how many threads to tell gemmlowp to use,
        // as it's hard to know how many CPU cores it will get to run on,
        // on typical mobile devices.
        // It seems impossible for gemmlowp to make this choice automatically,
        // which is why gemmlowp's default is to use only 1 thread, and
        // applications may override that if they know that they can count on
        // using more than that.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

 private:
  std::atomic<std::size_t> count_;
};

// A workload for a worker.
struct Task {
  Task() : local_allocator(nullptr) {}
  virtual ~Task() {}
  virtual void Run() = 0;
  Allocator* local_allocator;
};

// A worker thread.
class Worker {
 public:
  enum class State {
    ThreadStartup,  // The initial state before the thread main loop runs.
    Ready,          // Is not working, has not yet received new work to do.
    HasWork,        // Has work to do.
    ExitAsSoonAsPossible  // Should exit at earliest convenience.
  };

  explicit Worker(BlockingCounter* counter_to_decrement_when_ready)
      : task_(nullptr),
        state_(State::ThreadStartup),
        counter_to_decrement_when_ready_(counter_to_decrement_when_ready) {
    pthread_cond_init(&state_cond_, nullptr);
    pthread_mutex_init(&state_mutex_, nullptr);
    pthread_create(&thread_, nullptr, ThreadFunc, this);
  }

  ~Worker() {
    ChangeState(State::ExitAsSoonAsPossible);
    pthread_join(thread_, nullptr);
    pthread_cond_destroy(&state_cond_);
    pthread_mutex_destroy(&state_mutex_);
  }

  // Changes State; may be called from either the worker thread
  // or the master thread; however, not all state transitions are legal,
  // which is guarded by assertions.
  //
  // The Task argument is to be used only with new_state==HasWork.
  // It specifies the Task being handed to this Worker.
  void ChangeState(State new_state, Task* task = nullptr) {
    ScopedProfilingLabel label("Worker::ChangeState");
    pthread_mutex_lock(&state_mutex_);
    State old_state = state_.load(std::memory_order_relaxed);
    assert(old_state != new_state);
    switch (old_state) {
      case State::ThreadStartup:
        assert(new_state == State::Ready);
        break;
      case State::Ready:
        assert(new_state == State::HasWork ||
               new_state == State::ExitAsSoonAsPossible);
        break;
      case State::HasWork:
        assert(new_state == State::Ready ||
               new_state == State::ExitAsSoonAsPossible);
        break;
      default:
        abort();
    }
    switch (new_state) {
      case State::Ready:
        if (task_) {
          // Doing work is part of reverting to 'ready' state.
          task_->Run();
          task_ = nullptr;
        }
        break;
      case State::HasWork:
        assert(!task_);
        task->local_allocator = &local_allocator_;
        task_ = task;
        break;
      default:
        break;
    }
    state_.store(new_state, std::memory_order_relaxed);
    pthread_cond_broadcast(&state_cond_);
    pthread_mutex_unlock(&state_mutex_);
    if (new_state == State::Ready) {
      counter_to_decrement_when_ready_->DecrementCount();
    }
  }

  // Thread entry point.
  void ThreadFunc() {
    ScopedProfilingLabel label("Worker::ThreadFunc");

    ChangeState(State::Ready);

    // Thread main loop
    while (true) {
      // Get a state to act on
      // In the 'Ready' state, we have nothing to do but to wait until
      // we switch to another state.
      State state_to_act_upon = WaitForVariableChange(
          &state_, State::Ready, &state_cond_, &state_mutex_);

      // We now have a state to act on, so act.
      switch (state_to_act_upon) {
        case State::HasWork:
          // Got work to do! So do it, and then revert to 'Ready' state.
          ChangeState(State::Ready);
          break;
        case State::ExitAsSoonAsPossible:
          return;
        default:
          abort();
      }
    }
  }

  static void* ThreadFunc(void* arg) {
    static_cast<Worker*>(arg)->ThreadFunc();
    return nullptr;
  }

  // Called by the master thead to give this worker work to do.
  void StartWork(Task* task) { ChangeState(State::HasWork, task); }

 private:
  // The underlying thread.
  pthread_t thread_;

  // The task to be worked on.
  Task* task_;

  // The condition variable and mutex guarding state changes.
  pthread_cond_t state_cond_;
  pthread_mutex_t state_mutex_;

  // The state enum tells if we're currently working, waiting for work, etc.
  // Its concurrent accesses by the worker and main threads are guarded by
  // state_mutex_, and can thus use memory_order_relaxed. This still needs
  // to be a std::atomic because we use WaitForVariableChange.
  std::atomic<State> state_;

  // Each thread had a local allocator so they can allocate temporary
  // buffers without blocking each other.
  Allocator local_allocator_;

  // pointer to the master's thread BlockingCounter object, to notify the
  // master thread of when this worker switches to the 'Ready' state.
  BlockingCounter* const counter_to_decrement_when_ready_;
};

// A very simple pool of workers, that only allows the very
// specific parallelization pattern that we use here:
// a fixed number of workers can be given work, and one then
// waits for all of them to finish.
//
// See MultiThreadGemmContextBase for how other WorkersPool implementations can
// be used.
class WorkersPool {
 public:
  WorkersPool() {}

  ~WorkersPool() {
    for (auto w : workers_) {
      delete w;
    }
  }

  // Just executes the tasks. Does not destroy them. Similar to
  // ruy::ThreadPool::Execute.
  template <typename TaskType>
  void Execute(int tasks_count, TaskType* tasks) {
    assert(tasks_count >= 1);
    // One of the tasks will be run on the current thread.
    std::size_t workers_count = tasks_count - 1;
    CreateWorkers(workers_count);
    assert(workers_count <= workers_.size());
    counter_to_decrement_when_ready_.Reset(workers_count);
    for (std::size_t i = 0; i < tasks_count - 1; i++) {
      workers_[i]->StartWork(&tasks[i]);
    }
    // Execute the remaining workload immediately on the current thread.
    Task* task = &tasks[tasks_count - 1];
    task->local_allocator = &main_thread_task_allocator_;
    task->Run();
    // Wait for the workers submitted above to finish.
    counter_to_decrement_when_ready_.Wait();
  }

  // Legacy: executes the tasks and destroys them
  void LegacyExecuteAndDestroyTasks(const std::vector<Task*>& tasks) {
    std::size_t tasks_count = tasks.size();
    assert(tasks_count >= 1);
    // One of the tasks will be run on the current thread.
    std::size_t workers_count = tasks_count - 1;
    CreateWorkers(workers_count);
    assert(workers_count <= workers_.size());
    counter_to_decrement_when_ready_.Reset(workers_count);
    for (int i = 0; i < tasks_count - 1; i++) {
      workers_[i]->StartWork(tasks[i]);
    }
    // Execute the remaining workload immediately on the current thread.
    Task* task = tasks[tasks_count - 1];
    task->local_allocator = &main_thread_task_allocator_;
    task->Run();
    // Wait for the workers submitted above to finish.
    counter_to_decrement_when_ready_.Wait();
    // Cleanup tasks (best to do this from the same thread that allocated
    // the memory).
    std::for_each(tasks.begin(), tasks.end(), [](Task* task) { delete task; });
  }

  // Legacy old name of LegacyExecuteAndDestroyTasks
  void Execute(const std::vector<Task*>& tasks) {
    LegacyExecuteAndDestroyTasks(tasks);
  }

 private:
  // Ensures that the pool has at least the given count of workers.
  // If any new worker has to be created, this function waits for it to
  // be ready.
  void CreateWorkers(std::size_t workers_count) {
    if (workers_.size() >= workers_count) {
      return;
    }
    counter_to_decrement_when_ready_.Reset(workers_count - workers_.size());
    while (workers_.size() < workers_count) {
      workers_.push_back(new Worker(&counter_to_decrement_when_ready_));
    }
    counter_to_decrement_when_ready_.Wait();
  }

  // copy construction disallowed
  WorkersPool(const WorkersPool&) = delete;

  // The workers in this pool. They are owned by the pool:
  // the pool creates workers and destroys them in its destructor.
  std::vector<Worker*> workers_;

  // The BlockingCounter used to wait for the workers.
  BlockingCounter counter_to_decrement_when_ready_;

  // For N-threaded operations, we will use only N-1 worker threads
  // while the last task will be run directly on the main thread.
  // It will then use this main_thread_task_allocator_; having a
  // dedicated allocator for that (separate from the base allocator_)
  // allows to use the same code for all tasks regardless of which
  // thread they run on.
  Allocator main_thread_task_allocator_;
};

// The task we use to implement a multi-threaded Gemm: a block of the
// RHS has been packed by the master thread; each worker thread
// then has to pack a block of the LHS and accumulate the Gemm of these
// packed LHS and RHS blocks.
template <typename KernelFormat, typename InputScalar, typename OutputScalar,
          typename BitDepthParams, MapOrder LhsOrder, MapOrder RhsOrder,
          MapOrder ResultOrder, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType, typename GemmContextType>
struct GemmWithPackedRhsTask : Task {
  typedef PackedSideBlock<typename KernelFormat::Lhs> PackedLhs;
  typedef PackedSideBlock<typename KernelFormat::Rhs> PackedRhs;
  GemmWithPackedRhsTask(GemmContextType* _context, const KernelBase& _kernel,
                        const MatrixMap<const InputScalar, LhsOrder>& _lhs,
                        const PackedRhs& _packed_rhs,
                        MatrixMap<OutputScalar, ResultOrder>* _result,
                        const MatrixBlockBounds& _result_block,
                        const LhsOffset& _lhs_offset,
                        const RhsOffset& _rhs_offset,
                        const BlockParams& _block_params,
                        const OutputPipelineType& _output_pipeline)
      : context(_context),
        kernel(_kernel),
        lhs(_lhs),
        packed_rhs(_packed_rhs),
        result(*_result),
        result_block(_result_block),
        lhs_offset(_lhs_offset),
        rhs_offset(_rhs_offset),
        block_params(_block_params),
        output_pipeline(_output_pipeline) {}

  void Run() override {
    ScopedProfilingLabel label("GemmWithPackedRhsTask");

    const int rows = result_block.rows;
    const int cols = result_block.cols;
    const int depth = lhs.cols();

    PackedLhs packed_lhs(Side::Lhs, local_allocator, block_params);

    PackedResult packed_result(local_allocator, block_params);

    local_allocator->Commit();

    for (int c = 0; c < cols; c += block_params.l2_cols) {
      int cs = std::min(block_params.l2_cols, cols - c);

      for (int r = 0; r < rows; r += block_params.l2_rows) {
        int rs = std::min(block_params.l2_rows, rows - r);

        PackLhs(&packed_lhs, lhs.block(r, 0, rs, depth));

        Compute(kernel, block_params, &packed_result, packed_lhs, packed_rhs,
                depth);

        auto curr_result_block = MatrixBlockBounds(
            result_block.start_row + r, result_block.start_col + c, rs, cs);
        UnpackResult<KernelFormat>(
            &result, curr_result_block, packed_result, depth,
            packed_lhs.sums_of_each_slice(), packed_rhs.sums_of_each_slice(),
            lhs_offset.block(curr_result_block.start_row, rs),
            rhs_offset.block(curr_result_block.start_col, cs), output_pipeline);
      }
    }

    local_allocator->Decommit();
  }

  const GemmContextType* context;
  const KernelBase& kernel;
  const MatrixMap<const InputScalar, LhsOrder> lhs;
  const PackedRhs packed_rhs;
  MatrixMap<OutputScalar, ResultOrder> result;
  const MatrixBlockBounds result_block;
  const LhsOffset& lhs_offset;
  const RhsOffset& rhs_offset;
  const BlockParams& block_params;
  const OutputPipelineType& output_pipeline;
};

// This base class for multi-threading allows subclasses to implement their own
// workers_pool() method.  See MultiThreadGemmContext below for an example;
// any other implementation of workers_pool() must return an object with the
// same public methods as WorkersPool.
class MultiThreadGemmContextBase : public SingleThreadGemmContext {
 public:
  void set_max_num_threads(int n) { max_num_threads_ = n; }

  int max_num_threads() const { return max_num_threads_; }

 protected:
  // The maximum number of worker threads to use (including
  // the master thread).
  // The default value 1 means single-threading. That is the default
  // because gemmlowp's primary target is mobile hardware, where thermal
  // constraints usually mean that it may not be realistic to use more
  // than 1 CPU core even if multiple cores are present.
  // The special value 0 means try to detect the number of hardware threads.
  // Note: this assumes that all CPU cores are equivalent. That assumption
  // is defeated on big.LITTLE ARM devices, where we have no API to query
  // the number of big cores (which is typically what we would want to use,
  // leaving aside above-mentioned thermal issues). That is the other reason
  // why the best compromise here is to let max_num_threads_ default to 1,
  // so users who want multi-threading have to make the decision of how many
  // threads to use by themselves.
  int max_num_threads_ = 1;
};

class MultiThreadGemmContext : public MultiThreadGemmContextBase {
 public:
  WorkersPool* workers_pool() { return &workers_pool_; }

 private:
  // The workers pool used by MultiThreadGemm. Making
  // this part of the context allows it to be persistent,
  // avoiding recreating threads on every Gemm.
  WorkersPool workers_pool_;
};

// Determines how many threads should be used for a given Gemm
// operation.
template <int KernelRows>
inline int HowManyThreads(int max_num_threads, int rows, int cols, int depth) {
  // Early-exit in the default case where multi-threading is disabled.
  if (max_num_threads == 1) {
    return 1;
  }

  // Determine the maximum number of threads.
  int max_count = GetHardwareConcurrency(max_num_threads);

  // Basic calculation: take into account max pool size, and
  // how many rows we have to feed our kernel.
  // The motivation for an absolute minimum number of rows per thread,
  // potentially higher than KernelRows, is that very thin thread workload
  // currently defeat assumptions of the AddMod generator, resulting
  // in substantial bias in TestWithRealData on 24 threads.
  // Ideally, the AddMod generator should be aware of global (r,c) coordinates
  // so as to be independent of the number of threads.
  static const int AbsoluteMinRowsPerThread = 16;
  static const int MinRowsPerThread = KernelRows > AbsoluteMinRowsPerThread
                                          ? KernelRows
                                          : AbsoluteMinRowsPerThread;
  int thread_count = std::min(max_count, CeilQuotient(rows, MinRowsPerThread));

  // At this point for small products we already have thread_count==1 so
  // we can avoid doing more work; otherwise, we still want to check
  // that the cubic size (rows*cols*depth) is big enough to keep
  // workers_ busy.
  if (thread_count > 1) {
    // Empirically determined value.
    static const std::uint64_t min_cubic_size_per_thread = 64 * 1024;

    // We can only multiply two out of three sizes without risking overflow
    const std::uint64_t cubic_size =
        std::uint64_t(rows) * std::uint64_t(cols) * std::uint64_t(depth);

    thread_count =
        std::min(thread_count, int(cubic_size / min_cubic_size_per_thread));

    if (thread_count < 1) {
      thread_count = 1;
    }
  }

  assert(thread_count > 0 && thread_count <= max_count);
  return thread_count;
}

// The main multi-threaded Gemm function.
// To understand it, first read the code of SingleThreadGemm().
// The parallelization scheme used here is to have this master function
// pack a block of RHS and then start worker threads to pack a block of LHS
// each, and accumulate the corresponding products.
template <typename KernelFormat, typename InputScalar, typename OutputScalar,
          typename BitDepthParams, MapOrder LhsOrder, MapOrder RhsOrder,
          MapOrder ResultOrder, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType, typename GemmContextType>
void MultiThreadGemm(GemmContextType* context, const KernelBase& kernel,
                     const MatrixMap<const InputScalar, LhsOrder>& lhs,
                     const MatrixMap<const InputScalar, RhsOrder>& rhs,
                     MatrixMap<OutputScalar, ResultOrder>* result,
                     const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                     const OutputPipelineType& output_pipeline) {
  ScopedProfilingLabel label("gemmlowp::MultiThreadGemm");

  assert(lhs.cols() == rhs.rows());

  int rows = result->rows();
  int cols = result->cols();
  int depth = lhs.cols();

  // zero sizes should have been caught earlier and early-returned.
  assert(rows > 0);
  assert(cols > 0);
  assert(depth > 0);

  // The case of rows<cols should have been caught earlier and transposed.
  assert(rows >= cols);

  const int thread_count = HowManyThreads<KernelFormat::kRows>(
      context->max_num_threads(), rows, cols, depth);
  if (thread_count == 1) {
    return SingleThreadGemm<KernelFormat, InputScalar, OutputScalar,
                            BitDepthParams>(context, kernel, lhs, rhs, result,
                                            lhs_offset, rhs_offset,
                                            output_pipeline);
  }
  assert(thread_count > 1);

  // Simple 1:1 mapping of tasks to physical cores, which is very important
  // to getting good multithreaded performance, specially for not-very-large
  // GEMMs, and especially on Android.
  const int task_count = thread_count;

  Allocator* allocator = context->allocator();
  auto* workers_pool = context->workers_pool();

  BlockParams block_params;
  block_params.Init<KernelFormat>(
      rows, cols, depth, task_count, context->l1_bytes_to_use(),
      context->l2_bytes_to_use(), context->l2_rhs_factor());

  PackedSideBlock<typename KernelFormat::Rhs> packed_rhs(Side::Rhs, allocator,
                                                         block_params);
  allocator->Commit();

  // We loop over large blocks of the RHS.
  for (int c = 0; c < cols; c += block_params.l2_cols) {
    int cs = std::min(block_params.l2_cols, cols - c);

    // Pack a large block of the RHS.
    PackRhs(&packed_rhs, rhs.block(0, c, depth, cs));

    // Give work to each worker.
    std::vector<Task*> tasks;
    int next_start_row = 0;
    for (int n = 0; n < task_count; ++n) {
      int start_row = next_start_row;
      next_start_row = std::min(
          rows, RoundUp<KernelFormat::kRows>(rows * (n + 1) / task_count));

      int block_rows = next_start_row - start_row;
      auto lhs_block = lhs.block(start_row, 0, block_rows, depth);
      typedef GemmWithPackedRhsTask<KernelFormat, InputScalar, OutputScalar,
                                    BitDepthParams, LhsOrder, RhsOrder,
                                    ResultOrder, LhsOffset, RhsOffset,
                                    OutputPipelineType, GemmContextType>
          TaskType;
      tasks.push_back(
          new TaskType(context, kernel, lhs_block, packed_rhs, result,
                       MatrixBlockBounds(start_row, c, block_rows, cs),
                       lhs_offset, rhs_offset, block_params, output_pipeline));
    }
    // Execute the work on the workers (and partially on this thread).
    workers_pool->Execute(tasks);
  }

  allocator->Decommit();
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_MULTI_THREAD_GEMM_H_
