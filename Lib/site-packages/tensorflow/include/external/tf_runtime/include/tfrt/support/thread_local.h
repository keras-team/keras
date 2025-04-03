/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Thread local container that does not depend on `thread_local` storage.
//
// Unlike thread local storage, objects managed by a ThreadLocal instance are
// not bound to a thread's lifetime, but are destructed together with the
// ThreadLocal instance that created them. It is also possible to create
// multiple instances in the same function, unlike "real" thread_locals that are
// initialized the first time control passes through their declaration.
//
// When a thread requests a unique element from the ThreadLocal instance, it
// gets an element designated for that thread. It is possible to specify custom
// element constructor by providing the `Constructor` type parameter. By default
// elements are default constructed the first time they are accessed.
//
// As long as the number of unique threads accessing an intance of a ThreadLocal
// is smaller than `capacity`, it is lock-free and wait-free. After the number
// of stored elements reaches `capacity` it is using mutex for synchronization.
//
// WARNING: ThreadLocal uses the OS-specific value returned by
// std::this_thread::get_id() to identify threads. This value is not guaranteed
// to be unique except for the life of the thread. A newly created thread may
// get an OS-specific ID equal to that of an already destroyed thread. However
// this should not be a concern in practice, because it would still mean that
// only one thread can access an element, and because thread construction
// synchronizes with thread destruction it will be data race free. To the user
// program different threads sharing the same thread should be indistinguishable
// (unless it uses native thread_local storage).
//
// Example:
//
//   ThreadLocal<T> container = ...
//
//   std::thread([&]() { T& value = container.Local(); }).join(); // thread #1
//   std::thread([&]() { T& value = container.Local(); }).join(); // thread #2
//
// If threads #1 and #2 happened to have the same thread id they will get a
// reference to the same element. However it is guaranteed that the element
// will not be shared between threads running concurrently.

#ifndef TFRT_SUPPORT_THREAD_LOCAL_H_
#define TFRT_SUPPORT_THREAD_LOCAL_H_

#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {

namespace internal {

template <typename T>
struct DefaultConstructor {
  T Construct() { return T(); }
};

}  // namespace internal

template <typename T, typename Constructor = internal::DefaultConstructor<T>>
class ThreadLocal {
 public:
  // Computes optimal capacity for the number of threads that are expected to
  // access an intance of a ThreadLocal. Lookup operation has a constant
  // expected time, as long as load factor is less than one.
  // Reference: https://en.wikipedia.org/wiki/Linear_probing#Analysis
  static size_t Capacity(size_t num_threads) { return num_threads * 2; }

  template <typename... Args>
  explicit ThreadLocal(size_t capacity, Args... args)
      : capacity_(capacity),
        constructor_(std::forward<Args>(args)...),
        num_lock_free_entries_(0) {
    assert(capacity_ >= 0);
    data_.resize(capacity);
    ptrs_ = new std::atomic<Entry*>[capacity_];
    for (int i = 0; i < capacity_; ++i) {
      ptrs_[i].store(nullptr, std::memory_order_relaxed);
    }
  }

  ~ThreadLocal() { delete[] ptrs_; }

  T& Local() {
    std::thread::id this_thread = std::this_thread::get_id();
    if (capacity_ == 0) return SpilledLocal(this_thread);

    size_t h = std::hash<std::thread::id>()(this_thread);
    const int start_idx = h % capacity_;

    // NOTE: From the definition of `std::this_thread::get_id()` it is
    // guaranteed that we never can have concurrent calls to this functions
    // with the same thread id. If we didn't find an entry during the initial
    // traversal, it is guaranteed that no one else could have inserted it
    // concurrently.

    // Check if we already have an entry for `this_thread`.
    int idx = start_idx;
    while (ptrs_[idx].load(std::memory_order_acquire) != nullptr) {
      Entry& entry = *(ptrs_[idx].load());
      if (entry.thread_id == this_thread) return entry.value;

      idx += 1;
      if (idx >= capacity_) idx -= capacity_;
      if (idx == start_idx) break;
    }

    // If we are here, it means that we found an insertion point in lookup
    // table at `idx`, or we did a full traversal and table is full.

    // If lock-free storage is full, fallback on mutex.
    if (num_lock_free_entries_.load(std::memory_order_relaxed) >= capacity_)
      return SpilledLocal(this_thread);

    // We double check that we still have space to insert an element into a lock
    // free storage. If old value in `num_lock_free_entries_` is larger than
    // capacity, it means that some other thread added an element while
    // we were traversing lookup table.
    int insertion_index =
        num_lock_free_entries_.fetch_add(1, std::memory_order_relaxed);
    if (insertion_index >= capacity_) return SpilledLocal(this_thread);

    // At this point it's guaranteed that we can access
    // data_[insertion_index_] without a data race.
    data_[insertion_index] = {this_thread, constructor_.Construct()};

    // That's the pointer we'll put into the lookup table.
    Entry* inserted = &*data_[insertion_index];

    // We'll use nullptr pointer to Entry in a compare-and-swap loop.
    Entry* empty = nullptr;

    // Now we have to find an insertion point into the lookup table. We start
    // from the `idx` that was identified as an insertion point above, it's
    // guaranteed that we will have an empty entry somewhere in a lookup table
    // (because we created an entry in the `data_`).
    const int insertion_idx = idx;

    do {
      // Always start search from the original insertion candidate.
      idx = insertion_idx;
      while (ptrs_[idx].load(std::memory_order_relaxed) != nullptr) {
        idx += 1;
        if (idx >= capacity_) idx -= capacity_;
        // If we did a full loop, it means that we don't have any free entries
        // in the lookup table, and this means that something is terribly wrong.
        assert(idx != insertion_idx);
      }
      // Atomic store of the pointer guarantees that any other thread, that will
      // follow this pointer will see all the mutations in the `data_`.
    } while (!ptrs_[idx].compare_exchange_weak(empty, inserted,
                                               std::memory_order_release));

    return inserted->value;
  }

  // WARN: It's not thread safe to call it concurrently with `local()`.
  void ForEach(llvm::function_ref<void(std::thread::id, T&)> f) {
    // Reading directly from `data_` is unsafe, because only store to the
    // entry in `ptrs_` makes all changes visible to other threads.
    for (int i = 0; i < capacity_; ++i) {
      Entry* entry = ptrs_[i].load(std::memory_order_acquire);
      if (entry == nullptr) continue;
      f(entry->thread_id, entry->value);
    }

    // We did not spill into the map based storage.
    if (num_lock_free_entries_.load(std::memory_order_relaxed) < capacity_)
      return;

    mutex_lock lock(mu_);
    for (auto& kv : spilled_) {
      f(kv.first, kv.second);
    }
  }

 private:
  struct Entry {
    std::thread::id thread_id;
    T value;
  };

  // Use synchronized unordered_map when lock-free storage is full.
  T& SpilledLocal(std::thread::id this_thread) {
    mutex_lock lock(mu_);

    auto it = spilled_.find(this_thread);
    if (it != spilled_.end()) return it->second;

    auto inserted = spilled_.emplace(this_thread, constructor_.Construct());
    assert(inserted.second);
    return (*inserted.first).second;
  }

  const size_t capacity_;
  Constructor constructor_;

  // Storage that backs lock-free lookup table `ptrs_`. Entries stored
  // contiguously starting from index 0. Entries stored as optional so that
  // `T` can be non-default-constructible.
  std::vector<Optional<Entry>> data_;

  // Atomic pointers to the data stored in `data_`. Used as a lookup table for
  // linear probing hash map (https://en.wikipedia.org/wiki/Linear_probing).
  std::atomic<Entry*>* ptrs_;

  // Number of entries stored in the lock free storage.
  std::atomic<int> num_lock_free_entries_;

  // When the lock-free storage is full we spill to the unordered map
  // synchronized with a mutex. In practice this should never happen, if
  // `capacity_` is a reasonable estimate of the number of unique threads that
  // are expected to access this instance of ThreadLocal.
  mutex mu_;
  std::unordered_map<std::thread::id, T> spilled_ TFRT_GUARDED_BY(mu_);
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_THREAD_LOCAL_H_
