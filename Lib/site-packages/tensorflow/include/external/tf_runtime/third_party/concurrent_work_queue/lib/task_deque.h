// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// TaskDeque is a fixed-size, partially non-blocking deque of Task items.
// Operations on front of the queue must be done by a single thread (owner),
// operations on back of the queue can be done by multiple threads concurrently.
//
// Algorithm outline:
// All remote threads operating on the queue back are serialized by a mutex.
// This ensures that at most two threads access state: owner and one remote
// thread (Size aside). The algorithm ensures that the occupied region of the
// underlying array is logically continuous (can wraparound, but no stray
// occupied elements). Owner operates on one end of this region, remote thread
// operates on the other end. Synchronization between these threads
// (potential consumption of the last element and take up of the last empty
// element) happens by means of state variable in each element. States are:
// empty, busy (in process of insertion of removal) and ready. Threads claim
// elements (empty->busy and ready->busy transitions) by means of a CAS
// operation. The finishing transition (busy->empty and busy->ready) are done
// with plain store as the element is exclusively owned by the current thread.
//
// Non blocking Push/Pop implementation is based on:
//
//   (1) "Thread Scheduling for Multiprogrammed Multiprocessors"
//       Nimar S. Arora, Robert D. Blumofe, C. Greg Plaxton
//
//   (2) "Non-Blocking Steal-Half Work Queues"
//       Danny Hendler, Nir Shavit

#ifndef TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_TASK_DEQUE_H_
#define TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_TASK_DEQUE_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Compiler.h"
#include "tfrt/host_context/task_function.h"
#include "tfrt/support/mutex.h"

namespace tfrt {
namespace internal {

class TaskDeque {
 public:
  static constexpr unsigned kCapacity = 1024;

  static_assert((kCapacity > 2) && (kCapacity <= (64u << 10u)),
                "TaskDeque capacity must be in [4, 65536] range");
  static_assert((kCapacity & (kCapacity - 1)) == 0,
                "TaskDeque capacity must be a power of two for fast masking");

  TaskDeque() : front_(0), back_(0) {
    for (unsigned i = 0; i < kCapacity; i++) {
      array_[i].state.store(kEmpty, std::memory_order_relaxed);
    }
  }
  TaskDeque(const TaskDeque&) = delete;
  void operator=(const TaskDeque&) = delete;

  ~TaskDeque() { assert(Size() == 0); }

  // PushFront() inserts task at the beginning of the queue.
  //
  // If the queue is full, returns passed in task wrapped in optional, otherwise
  // returns empty optional.
  [[nodiscard]] std::optional<TaskFunction> PushFront(TaskFunction task) {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e = &array_[front & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kEmpty || !e->state.compare_exchange_strong(
                           s, kBusy, std::memory_order_acquire)) {
      return std::optional<TaskFunction>(std::move(task));
    }
    front_.store(front + kIncrement, std::memory_order_relaxed);
    e->task = std::move(task);
    e->state.store(kReady, std::memory_order_release);
    return std::nullopt;
  }

  // PopFront() removes and returns the first element in the queue.
  //
  // If the queue is empty returns empty optional.
  [[nodiscard]] std::optional<TaskFunction> PopFront() {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e = &array_[(front - 1) & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kReady || !e->state.compare_exchange_strong(
                           s, kBusy, std::memory_order_acquire)) {
      return std::nullopt;
    }
    TaskFunction task = std::move(e->task);
    e->state.store(kEmpty, std::memory_order_release);
    front = ((front - 1) & kMask2) | (front & ~kMask2);
    front_.store(front, std::memory_order_relaxed);
    return std::optional<TaskFunction>(std::move(task));
  }

  // PushBack() inserts task `w` at the end of the queue.
  //
  // If the queue is full, returns passed in task wrapped in optional, otherwise
  // returns empty optional.
  [[nodiscard]] std::optional<TaskFunction> PushBack(TaskFunction task) {
    mutex_lock lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem* e = &array_[(back - 1) & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kEmpty || !e->state.compare_exchange_strong(
                           s, kBusy, std::memory_order_acquire)) {
      return std::optional<TaskFunction>(std::move(task));
    }
    back = ((back - 1) & kMask2) | (back & ~kMask2);
    back_.store(back, std::memory_order_relaxed);
    e->task = std::move(task);
    e->state.store(kReady, std::memory_order_release);
    return std::nullopt;
  }

  // PopBack() removes and returns the last elements in the queue.
  //
  // If the queue is empty returns empty optional.
  [[nodiscard]] std::optional<TaskFunction> PopBack() {
    if (Empty()) return std::nullopt;

    mutex_lock lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem* e = &array_[back & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kReady || !e->state.compare_exchange_strong(
                           s, kBusy, std::memory_order_acquire)) {
      return std::nullopt;
    }
    TaskFunction task = std::move(e->task);
    e->state.store(kEmpty, std::memory_order_release);
    back_.store(back + kIncrement, std::memory_order_relaxed);
    return std::optional<TaskFunction>(std::move(task));
  }

  // PopBackHalf removes and returns half last elements in the queue.
  // Returns number of elements removed.
  unsigned PopBackHalf(std::vector<TaskFunction>* result) {
    if (Empty()) return 0;

    mutex_lock lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    unsigned size = Size();
    unsigned mid = back;
    if (size > 1) mid = back + (size - 1) / 2;
    unsigned n = 0;
    unsigned start = 0;
    for (; static_cast<int>(mid - back) >= 0; mid--) {
      Elem* e = &array_[mid & kMask];
      uint8_t s = e->state.load(std::memory_order_relaxed);
      if (n == 0) {
        if (s != kReady || !e->state.compare_exchange_strong(
                               s, kBusy, std::memory_order_acquire))
          continue;
        start = mid;
      } else {
        // Note: no need to store temporal kBusy, we exclusively own these
        // elements.
        assert(s == kReady);
      }
      result->push_back(std::move(e->task));
      e->state.store(kEmpty, std::memory_order_release);
      n++;
    }
    if (n != 0) back_.store(start + kIncrement, std::memory_order_relaxed);
    return n;
  }

  // Size returns current queue size.
  // Can be called by any thread at any time.
  unsigned Size() const { return SizeOrNotEmpty<true>(); }

  // Empty tests whether container is empty.
  // Can be called by any thread at any time.
  bool Empty() const { return SizeOrNotEmpty<false>() == 0; }

  // Delete all the elements from the queue.
  void Flush() {
    while (!Empty()) {
      std::optional<TaskFunction> task = PopFront();
      assert(task.has_value());
    }
  }

 private:
  // Constant used to move front/back pointers in Push/Pop functions above.
  static constexpr unsigned kIncrement = (kCapacity << 1u) + 1;

  static constexpr unsigned kMask = kCapacity - 1;
  static constexpr unsigned kMask2 = (kCapacity << 1u) - 1;

  enum : uint8_t {
    kEmpty = 0,
    kBusy = 1,
    kReady = 2,
  };

  struct Elem {
    std::atomic<uint8_t> state;
    TaskFunction task;
  };

  mutex mutex_;

  // Low log(kCapacity) + 1 bits in front_ and back_ contain rolling index of
  // front/back, respectively. The remaining bits contain modification counters
  // that are incremented on Push operations. This allows us to
  //
  // (1) Distinguish between empty and full conditions (if we would use
  //     log(kCapacity) bits for position, these conditions would be
  //     indistinguishable)
  //
  // (2) Obtain consistent snapshot of front_/back_ for Size operation using the
  //     modification counters.

  // Align to 128 byte boundary to prevent false sharing between front and back
  // pointers, they are accessed from different threads.
  alignas(128) std::atomic<unsigned> front_;
  alignas(128) std::atomic<unsigned> back_;

  std::array<Elem, kCapacity> array_;

  // SizeOrNotEmpty() returns current queue size; if `need_size_estimate` is
  // false, only whether the size is 0 is guaranteed to be correct. Can be
  // called by any thread at any time.
  template <bool need_size_estimate>
  unsigned SizeOrNotEmpty() const {
    // Emptiness plays critical role in thread pool blocking. So we go to great
    // effort to not produce false positives (claim non-empty queue as empty).
    unsigned front = front_.load(std::memory_order_acquire);
    for (;;) {
      // Capture a consistent snapshot of front/tail.
      unsigned back = back_.load(std::memory_order_acquire);
      unsigned front1 = front_.load(std::memory_order_relaxed);
      if (front != front1) {
        front = front1;
        std::atomic_thread_fence(std::memory_order_acquire);
        continue;
      }
      if (need_size_estimate) {
        return CalculateSize(front, back);
      } else {
        // This value will be 0 if the queue is empty, and undefined otherwise.
        unsigned maybe_zero = ((front ^ back) & kMask2);
        // Queue size estimate must agree with maybe zero check on the queue
        // empty/non-empty state.
        assert((CalculateSize(front, back) == 0) == (maybe_zero == 0));
        return maybe_zero;
      }
    }
  }

  unsigned CalculateSize(unsigned front, unsigned back) const {
    int size = (front & kMask2) - (back & kMask2);
    // Fix overflow.
    if (size < 0) size += 2 * kCapacity;
    // Order of modification in push/pop is crafted to make the queue look
    // larger than it is during concurrent modifications. E.g. push can
    // increment size before the corresponding pop has decremented it.
    // So the computed size can be up to kCapacity + 1, fix it.
    assert(size <= kCapacity + 1);
    if (size > static_cast<int>(kCapacity)) size = kCapacity;
    return static_cast<unsigned>(size);
  }
};

}  // namespace internal
}  // namespace tfrt

#endif  // TFRT_THIRD_PARTY_CONCURRENT_WORK_QUEUE_TASK_DEQUE_H_
