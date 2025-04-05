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

#ifndef GRPC_CORE_LIB_IOMGR_EXECUTOR_MPMCQUEUE_H
#define GRPC_CORE_LIB_IOMGR_EXECUTOR_MPMCQUEUE_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/debug/stats.h"
#include "src/core/lib/gprpp/atomic.h"
#include "src/core/lib/gprpp/sync.h"

namespace grpc_core {

extern DebugOnlyTraceFlag grpc_thread_pool_trace;

// Abstract base class of a Multiple-Producer-Multiple-Consumer(MPMC) queue
// interface
class MPMCQueueInterface {
 public:
  virtual ~MPMCQueueInterface() {}

  // Puts elem into queue immediately at the end of queue.
  // This might cause to block on full queue depending on implementation.
  virtual void Put(void* elem) = 0;

  // Removes the oldest element from the queue and return it.
  // This might cause to block on empty queue depending on implementation.
  // Optional argument for collecting stats purpose.
  virtual void* Get(gpr_timespec* wait_time = nullptr) = 0;

  // Returns number of elements in the queue currently
  virtual int count() const = 0;
};

class InfLenFIFOQueue : public MPMCQueueInterface {
 public:
  // Creates a new MPMC Queue. The queue created will have infinite length.
  InfLenFIFOQueue();

  // Releases all resources held by the queue. The queue must be empty, and no
  // one waits on conditional variables.
  ~InfLenFIFOQueue();

  // Puts elem into queue immediately at the end of queue. Since the queue has
  // infinite length, this routine will never block and should never fail.
  void Put(void* elem);

  // Removes the oldest element from the queue and returns it.
  // This routine will cause the thread to block if queue is currently empty.
  // Argument wait_time should be passed in when trace flag turning on (for
  // collecting stats info purpose.)
  void* Get(gpr_timespec* wait_time = nullptr);

  // Returns number of elements in queue currently.
  // There might be concurrently add/remove on queue, so count might change
  // quickly.
  int count() const { return count_.Load(MemoryOrder::RELAXED); }

  struct Node {
    Node* next;  // Linking
    Node* prev;
    void* content;             // Points to actual element
    gpr_timespec insert_time;  // Time for stats

    Node() {
      next = prev = nullptr;
      content = nullptr;
    }
  };

  // For test purpose only. Returns number of nodes allocated in queue.
  // Any allocated node will be alive until the destruction of the queue.
  int num_nodes() const { return num_nodes_; }

  // For test purpose only. Returns the initial number of nodes in queue.
  int init_num_nodes() const { return kQueueInitNumNodes; }

 private:
  // For Internal Use Only.
  // Removes the oldest element from the queue and returns it. This routine
  // will NOT check whether queue is empty, and it will NOT acquire mutex.
  // Caller MUST check that queue is not empty and must acquire mutex before
  // callling.
  void* PopFront();

  // Stats of queue. This will only be collect when debug trace mode is on.
  // All printed stats info will have time measurement in microsecond.
  struct Stats {
    uint64_t num_started;    // Number of elements have been added to queue
    uint64_t num_completed;  // Number of elements have been removed from
                             // the queue
    gpr_timespec total_queue_time;  // Total waiting time that all the
                                    // removed elements have spent in queue
    gpr_timespec max_queue_time;    // Max waiting time among all removed
                                    // elements
    gpr_timespec busy_queue_time;   // Accumulated amount of time that queue
                                    // was not empty

    Stats() {
      num_started = 0;
      num_completed = 0;
      total_queue_time = gpr_time_0(GPR_TIMESPAN);
      max_queue_time = gpr_time_0(GPR_TIMESPAN);
      busy_queue_time = gpr_time_0(GPR_TIMESPAN);
    }
  };

  // Node for waiting thread queue. Stands for one waiting thread, should have
  // exact one thread waiting on its CondVar.
  // Using a doubly linked list for waiting thread queue to wake up waiting
  // threads in LIFO order to reduce cache misses.
  struct Waiter {
    CondVar cv;
    Waiter* next;
    Waiter* prev;
  };

  // Pushs waiter to the front of queue, require caller held mutex
  void PushWaiter(Waiter* waiter);

  // Removes waiter from queue, require caller held mutex
  void RemoveWaiter(Waiter* waiter);

  // Returns pointer to the waiter that should be waken up next, should be the
  // last added waiter.
  Waiter* TopWaiter();

  Mutex mu_;        // Protecting lock
  Waiter waiters_;  // Head of waiting thread queue

  // Initial size for delete list
  static const int kDeleteListInitSize = 1024;
  // Initial number of nodes allocated
  static const int kQueueInitNumNodes = 1024;

  Node** delete_list_ = nullptr;  // Keeps track of all allocated array entries
                                  // for deleting on destruction
  size_t delete_list_count_ = 0;  // Number of entries in list
  size_t delete_list_size_ = 0;   // Size of the list. List will be expanded to
                                  // double size on full

  Node* queue_head_ = nullptr;  // Head of the queue, remove position
  Node* queue_tail_ = nullptr;  // End of queue, insert position
  Atomic<int> count_{0};        // Number of elements in queue
  int num_nodes_ = 0;           // Number of nodes allocated

  Stats stats_;            // Stats info
  gpr_timespec busy_time;  // Start time of busy queue

  // Internal Helper.
  // Allocates an array of nodes of size "num", links all nodes together except
  // the first node's prev and last node's next. They should be set by caller
  // manually afterward.
  Node* AllocateNodes(int num);
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_IOMGR_EXECUTOR_MPMCQUEUE_H */
