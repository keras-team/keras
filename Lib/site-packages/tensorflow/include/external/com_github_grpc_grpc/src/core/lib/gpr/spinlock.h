/*
 *
 * Copyright 2015 gRPC authors.
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

#ifndef GRPC_CORE_LIB_GPR_SPINLOCK_H
#define GRPC_CORE_LIB_GPR_SPINLOCK_H

#include <grpc/support/port_platform.h>

#include <grpc/support/atm.h>

/* Simple spinlock. No backoff strategy, gpr_spinlock_lock is almost always
   a concurrency code smell. */
typedef struct {
  gpr_atm atm;
} gpr_spinlock;

#ifdef __cplusplus
#define GPR_SPINLOCK_INITIALIZER (gpr_spinlock{0})
#else
#define GPR_SPINLOCK_INITIALIZER ((gpr_spinlock){0})
#endif
#define GPR_SPINLOCK_STATIC_INITIALIZER \
  { 0 }

#define gpr_spinlock_trylock(lock) (gpr_atm_acq_cas(&(lock)->atm, 0, 1))
#define gpr_spinlock_unlock(lock) (gpr_atm_rel_store(&(lock)->atm, 0))
#define gpr_spinlock_lock(lock) \
  do {                          \
  } while (!gpr_spinlock_trylock((lock)))

#endif /* GRPC_CORE_LIB_GPR_SPINLOCK_H */
