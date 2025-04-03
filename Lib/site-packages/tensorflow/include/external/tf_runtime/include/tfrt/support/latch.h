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

// Downward counter of type ptrdiff_t which can be used for thread
// synchronization. It has an API compatible with std::latch (C++ 20).

#ifndef TFRT_HOST_CONTEXT_LATCH_H_
#define TFRT_HOST_CONTEXT_LATCH_H_

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "tfrt/support/mutex.h"

namespace tfrt {

// A latch is a thread coordination mechanism that allows any number of threads
// to block until an expected number of threads arrive at the latch (via the
// count_down function). The expected count is set when the latch is created. An
// individual latch is a single-use object; once the expected count has been
// reached, the latch cannot be reused (c++20 [thread.latch]).
//
// Reference: https://en.cppreference.com/w/cpp/thread/latch
class latch {
 public:
  explicit latch(ptrdiff_t count)
      : state_(static_cast<uint64_t>(count) << 1), notified_(false) {
    assert(count >= 0);
    assert(static_cast<uint64_t>(count) < (1ull << 63));
  }

  ~latch() { assert(try_wait()); }

  latch(const latch&) = delete;
  latch& operator=(const latch&) = delete;

  // Decrements the counter by `n`.
  void count_down(ptrdiff_t n = 1);
  // Returns true if the internal counter equals zero.
  bool try_wait() const noexcept;
  // Blocks until the counter reaches zero.
  void wait() const;
  // Decrements the counter and blocks until it reaches zero.
  void arrive_and_wait(ptrdiff_t n = 1);

 private:
  mutable mutex mu_;
  mutable condition_variable cv_;

  // State layout:
  // - lowest bit is 1 if latch has a waiter parked on `cv_`.
  // - higher bits store the counter value
  mutable std::atomic<uint64_t> state_;
  bool notified_ TFRT_GUARDED_BY(mu_);
};

inline void latch::count_down(ptrdiff_t n) {
  assert(n >= 0);

  uint64_t state = state_.fetch_sub(n * 2);
  assert((state >> 1) >= n);

  // Counter dropped to zero and latch has a waiting thread(s).
  if ((state >> 1) == n && (state & 1) == 1) {
    mutex_lock lock(mu_);
    cv_.notify_all();
    assert(!notified_);
    notified_ = true;
  }
}

inline bool latch::try_wait() const noexcept {
  uint64_t state = state_.load();
  return (state >> 1) == 0;
}

inline void latch::wait() const {
  // Set the waiter bit to 1.
  uint64_t state = state_.fetch_or(1);

  // Counter already dropped to zero.
  if ((state >> 1) == 0) return;

  // Block until the counter reaches zero.
  mutex_lock lock(mu_);
  cv_.wait(lock, [this]() TFRT_REQUIRES(mu_) { return this->notified_; });
}

inline void latch::arrive_and_wait(ptrdiff_t n) {
  count_down(n);
  wait();
}

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_LATCH_H_
