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

#ifndef GRPCPP_IMPL_CODEGEN_SYNC_H
#define GRPCPP_IMPL_CODEGEN_SYNC_H

#include <grpc/impl/codegen/port_platform.h>

#ifdef GPR_HAS_PTHREAD_H
#include <pthread.h>
#endif

#include <mutex>

#include <grpc/impl/codegen/log.h>
#include <grpc/impl/codegen/sync.h>

#include <grpcpp/impl/codegen/core_codegen_interface.h>

// The core library is not accessible in C++ codegen headers, and vice versa.
// Thus, we need to have duplicate headers with similar functionality.
// Make sure any change to this file is also reflected in
// src/core/lib/gprpp/sync.h too.
//
// Whenever possible, prefer "src/core/lib/gprpp/sync.h" over this file,
// since in core we do not rely on g_core_codegen_interface and hence do not
// pay the costs of virtual function calls.

namespace grpc {
namespace internal {

class Mutex {
 public:
  Mutex() { g_core_codegen_interface->gpr_mu_init(&mu_); }
  ~Mutex() { g_core_codegen_interface->gpr_mu_destroy(&mu_); }

  Mutex(const Mutex&) = delete;
  Mutex& operator=(const Mutex&) = delete;

  gpr_mu* get() { return &mu_; }
  const gpr_mu* get() const { return &mu_; }

 private:
  union {
    gpr_mu mu_;
    std::mutex do_not_use_sth_;
#ifdef GPR_HAS_PTHREAD_H
    pthread_mutex_t do_not_use_pth_;
#endif
  };
};

// MutexLock is a std::
class MutexLock {
 public:
  explicit MutexLock(Mutex* mu) : mu_(mu->get()) {
    g_core_codegen_interface->gpr_mu_lock(mu_);
  }
  explicit MutexLock(gpr_mu* mu) : mu_(mu) {
    g_core_codegen_interface->gpr_mu_lock(mu_);
  }
  ~MutexLock() { g_core_codegen_interface->gpr_mu_unlock(mu_); }

  MutexLock(const MutexLock&) = delete;
  MutexLock& operator=(const MutexLock&) = delete;

 private:
  gpr_mu* const mu_;
};

class ReleasableMutexLock {
 public:
  explicit ReleasableMutexLock(Mutex* mu) : mu_(mu->get()) {
    g_core_codegen_interface->gpr_mu_lock(mu_);
  }
  explicit ReleasableMutexLock(gpr_mu* mu) : mu_(mu) {
    g_core_codegen_interface->gpr_mu_lock(mu_);
  }
  ~ReleasableMutexLock() {
    if (!released_) g_core_codegen_interface->gpr_mu_unlock(mu_);
  }

  ReleasableMutexLock(const ReleasableMutexLock&) = delete;
  ReleasableMutexLock& operator=(const ReleasableMutexLock&) = delete;

  void Lock() {
    GPR_DEBUG_ASSERT(released_);
    g_core_codegen_interface->gpr_mu_lock(mu_);
    released_ = false;
  }

  void Unlock() {
    GPR_DEBUG_ASSERT(!released_);
    released_ = true;
    g_core_codegen_interface->gpr_mu_unlock(mu_);
  }

 private:
  gpr_mu* const mu_;
  bool released_ = false;
};

class CondVar {
 public:
  CondVar() { g_core_codegen_interface->gpr_cv_init(&cv_); }
  ~CondVar() { g_core_codegen_interface->gpr_cv_destroy(&cv_); }

  CondVar(const CondVar&) = delete;
  CondVar& operator=(const CondVar&) = delete;

  void Signal() { g_core_codegen_interface->gpr_cv_signal(&cv_); }
  void Broadcast() { g_core_codegen_interface->gpr_cv_broadcast(&cv_); }

  int Wait(Mutex* mu) {
    return Wait(mu,
                g_core_codegen_interface->gpr_inf_future(GPR_CLOCK_REALTIME));
  }
  int Wait(Mutex* mu, const gpr_timespec& deadline) {
    return g_core_codegen_interface->gpr_cv_wait(&cv_, mu->get(), deadline);
  }

  template <typename Predicate>
  void WaitUntil(Mutex* mu, Predicate pred) {
    while (!pred()) {
      Wait(mu, g_core_codegen_interface->gpr_inf_future(GPR_CLOCK_REALTIME));
    }
  }

 private:
  gpr_cv cv_;
};

}  // namespace internal
}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_SYNC_H
