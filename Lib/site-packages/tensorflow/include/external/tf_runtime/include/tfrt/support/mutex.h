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

// Wrappers around std::{mutex,unique_lock,condition_variable} with support for
// thread safety annotations.

#ifndef TFRT_SUPPORT_STD_MUTEX_H_
#define TFRT_SUPPORT_STD_MUTEX_H_

#include <chrono>
#include <condition_variable>
#include <mutex>

#include "tfrt/support/thread_annotations.h"

// Avoid conflict with @org_tensorflow/core/platform/mutex.h:181
// TODO(tfrt-devs): remove the macro in @org_tensorflow/core/platform/mutex.h
// and replace it with the [[nodiscard] attribute when c++17 is allowed.
#undef mutex_lock

namespace tfrt {

// Wrap std::mutex with support for thread annotations.
//
// Note that std::mutex's destructor is nontrivial, so it's not safe to declare
// tfrt::mutex with static storage class.
class TFRT_CAPABILITY("mutex") mutex {
 public:
  constexpr mutex() = default;
  ~mutex() = default;

  mutex(const mutex&) = delete;
  mutex& operator=(const mutex&) = delete;

  void lock() TFRT_ACQUIRE() { mu_.lock(); }
  void unlock() TFRT_RELEASE() { mu_.unlock(); }

 private:
  friend class mutex_lock;
  std::mutex mu_{};
};

// Wrap std::unique_lock<std::mutex> with support for thread annotations.
class TFRT_SCOPED_CAPABILITY mutex_lock {
 public:
  explicit mutex_lock(mutex& mu) TFRT_ACQUIRE(mu) : unique_lock_(mu.mu_) {}
  ~mutex_lock() TFRT_RELEASE() {}

  mutex_lock(const mutex_lock&) = delete;
  mutex_lock& operator=(const mutex_lock&) = delete;

 private:
  friend class condition_variable;
  std::unique_lock<std::mutex> unique_lock_;
};

// Wraps std::condition_variable with support for mutex_lock.
class condition_variable {
 public:
  condition_variable() = default;
  ~condition_variable() = default;

  condition_variable(const condition_variable&) = delete;
  condition_variable& operator=(const condition_variable&) = delete;

  void wait(mutex_lock& mu) { cv_.wait(mu.unique_lock_); }

  template <class Predicate>
  void wait(mutex_lock& mu, Predicate pred) {
    cv_.wait(mu.unique_lock_, pred);
  }
  template <class Clock, class Duration, class Predicate>
  bool wait_until(mutex_lock& mu,
                  const std::chrono::time_point<Clock, Duration>& timeout_time,
                  Predicate pred) {
    return cv_.wait_until(mu.unique_lock_, timeout_time, pred);
  }

  template <class Clock, class Duration>
  bool wait_until(
      mutex_lock& mu,
      const std::chrono::time_point<Clock, Duration>& timeout_time) {
    return cv_.wait_until(mu.unique_lock_, timeout_time) ==
           std::cv_status::timeout;
  }

  void notify_one() { cv_.notify_one(); }
  void notify_all() { cv_.notify_all(); }

 private:
  std::condition_variable cv_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_STD_MUTEX_H_
