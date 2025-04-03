/*
 *
 * Copyright 2019 gRPC authors.
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
 *
 */

#ifndef GRPC_CORE_LIB_IOMGR_EXECUTOR_THREADPOOL_H
#define GRPC_CORE_LIB_IOMGR_EXECUTOR_THREADPOOL_H

#include <grpc/support/port_platform.h>

#include <grpc/grpc.h>

#include "src/core/lib/gprpp/thd.h"
#include "src/core/lib/iomgr/executor/mpmcqueue.h"

namespace grpc_core {

// A base abstract base class for threadpool.
// Threadpool is an executor that maintains a pool of threads sitting around
// and waiting for closures. A threadpool also maintains a queue of pending
// closures, when closures appearing in the queue, the threads in pool will
// pull them out and execute them.
class ThreadPoolInterface {
 public:
  // Waits for all pending closures to complete, then shuts down thread pool.
  virtual ~ThreadPoolInterface() {}

  // Schedules a given closure for execution later.
  // Depending on specific subclass implementation, this routine might cause
  // current thread to be blocked (in case of unable to schedule).
  // Closure should contain a function pointer and arguments it will take, more
  // details for closure struct at /grpc/include/grpc/impl/codegen/grpc_types.h
  virtual void Add(grpc_experimental_completion_queue_functor* closure) = 0;

  // Returns the current number of pending closures
  virtual int num_pending_closures() const = 0;

  // Returns the capacity of pool (number of worker threads in pool)
  virtual int pool_capacity() const = 0;

  // Thread option accessor
  virtual const Thread::Options& thread_options() const = 0;

  // Returns the thread name for threads in this ThreadPool.
  virtual const char* thread_name() const = 0;
};

// Worker thread for threadpool. Executes closures in the queue, until getting a
// NULL closure.
class ThreadPoolWorker {
 public:
  ThreadPoolWorker(const char* thd_name, MPMCQueueInterface* queue,
                   Thread::Options& options, int index)
      : queue_(queue), thd_name_(thd_name), index_(index) {
    thd_ = Thread(thd_name,
                  [](void* th) { static_cast<ThreadPoolWorker*>(th)->Run(); },
                  this, nullptr, options);
  }

  ~ThreadPoolWorker() {}

  void Start() { thd_.Start(); }
  void Join() { thd_.Join(); }

 private:
  // struct for tracking stats of thread
  struct Stats {
    gpr_timespec sleep_time;
    Stats() { sleep_time = gpr_time_0(GPR_TIMESPAN); }
  };

  void Run();  // Pulls closures from queue and executes them

  MPMCQueueInterface* queue_;  // Queue in thread pool to pull closures from
  Thread thd_;                 // Thread wrapped in
  Stats stats_;                // Stats to be collected in run time
  const char* thd_name_;       // Name of thread
  int index_;                  // Index in thread pool
};

// A fixed size thread pool implementation of abstract thread pool interface.
// In this implementation, the number of threads in pool is fixed, but the
// capacity of closure queue is unlimited.
class ThreadPool : public ThreadPoolInterface {
 public:
  // Creates a thread pool with size of "num_threads", with default thread name
  // "ThreadPoolWorker" and all thread options set to default. If the given size
  // is 0 or less, there will be 1 worker thread created inside pool.
  ThreadPool(int num_threads);

  // Same as ThreadPool(int num_threads) constructor, except
  // that it also sets "thd_name" as the name of all threads in the thread pool.
  ThreadPool(int num_threads, const char* thd_name);

  // Same as ThreadPool(const char *thd_name, int num_threads) constructor,
  // except that is also set thread_options for threads.
  // Notes for stack size:
  // If the stack size field of the passed in Thread::Options is set to default
  // value 0, default ThreadPool stack size will be used. The current default
  // stack size of this implementation is 1952K for mobile platform and 64K for
  // all others.
  ThreadPool(int num_threads, const char* thd_name,
             const Thread::Options& thread_options);

  // Waits for all pending closures to complete, then shuts down thread pool.
  ~ThreadPool() override;

  // Adds given closure into pending queue immediately. Since closure queue has
  // infinite length, this routine will not block.
  void Add(grpc_experimental_completion_queue_functor* closure) override;

  int num_pending_closures() const override;
  int pool_capacity() const override;
  const Thread::Options& thread_options() const override;
  const char* thread_name() const override;

 private:
  int num_threads_ = 0;
  const char* thd_name_ = nullptr;
  Thread::Options thread_options_;
  ThreadPoolWorker** threads_ = nullptr;  // Array of worker threads
  MPMCQueueInterface* queue_ = nullptr;   // Closure queue

  Atomic<bool> shut_down_{false};  // Destructor has been called if set to true

  void SharedThreadPoolConstructor();
  // For ThreadPool, default stack size for mobile platform is 1952K. for other
  // platforms is 64K.
  size_t DefaultStackSize();
  // Internal Use Only for debug checking.
  void AssertHasNotBeenShutDown();
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_IOMGR_EXECUTOR_THREADPOOL_H */
