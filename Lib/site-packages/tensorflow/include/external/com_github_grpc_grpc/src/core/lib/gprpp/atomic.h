/*
 *
 * Copyright 2017 gRPC authors.
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

#ifndef GRPC_CORE_LIB_GPRPP_ATOMIC_H
#define GRPC_CORE_LIB_GPRPP_ATOMIC_H

#include <grpc/support/port_platform.h>

#include <atomic>

#include <grpc/support/atm.h>

namespace grpc_core {

enum class MemoryOrder {
  RELAXED = std::memory_order_relaxed,
  CONSUME = std::memory_order_consume,
  ACQUIRE = std::memory_order_acquire,
  RELEASE = std::memory_order_release,
  ACQ_REL = std::memory_order_acq_rel,
  SEQ_CST = std::memory_order_seq_cst
};

template <typename T>
class Atomic {
 public:
  explicit Atomic(T val = T()) : storage_(val) {}

  T Load(MemoryOrder order) const {
    return storage_.load(static_cast<std::memory_order>(order));
  }

  void Store(T val, MemoryOrder order) {
    storage_.store(val, static_cast<std::memory_order>(order));
  }

  T Exchange(T desired, MemoryOrder order) {
    return storage_.exchange(desired, static_cast<std::memory_order>(order));
  }

  bool CompareExchangeWeak(T* expected, T desired, MemoryOrder success,
                           MemoryOrder failure) {
    return GPR_ATM_INC_CAS_THEN(storage_.compare_exchange_weak(
        *expected, desired, static_cast<std::memory_order>(success),
        static_cast<std::memory_order>(failure)));
  }

  bool CompareExchangeStrong(T* expected, T desired, MemoryOrder success,
                             MemoryOrder failure) {
    return GPR_ATM_INC_CAS_THEN(storage_.compare_exchange_strong(
        *expected, desired, static_cast<std::memory_order>(success),
        static_cast<std::memory_order>(failure)));
  }

  template <typename Arg>
  T FetchAdd(Arg arg, MemoryOrder order = MemoryOrder::SEQ_CST) {
    return GPR_ATM_INC_ADD_THEN(storage_.fetch_add(
        static_cast<Arg>(arg), static_cast<std::memory_order>(order)));
  }

  template <typename Arg>
  T FetchSub(Arg arg, MemoryOrder order = MemoryOrder::SEQ_CST) {
    return GPR_ATM_INC_ADD_THEN(storage_.fetch_sub(
        static_cast<Arg>(arg), static_cast<std::memory_order>(order)));
  }

  // Atomically increment a counter only if the counter value is not zero.
  // Returns true if increment took place; false if counter is zero.
  bool IncrementIfNonzero(MemoryOrder load_order = MemoryOrder::ACQUIRE) {
    T count = storage_.load(static_cast<std::memory_order>(load_order));
    do {
      // If zero, we are done (without an increment). If not, we must do a CAS
      // to maintain the contract: do not increment the counter if it is already
      // zero
      if (count == 0) {
        return false;
      }
    } while (!CompareExchangeWeak(&count, count + 1, MemoryOrder::ACQ_REL,
                                  load_order));
    return true;
  }

 private:
  std::atomic<T> storage_;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_GPRPP_ATOMIC_H */
