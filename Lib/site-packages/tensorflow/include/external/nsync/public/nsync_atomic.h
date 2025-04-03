/* Copyright 2016 Google Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. */

#ifndef NSYNC_PUBLIC_NSYNC_ATOMIC_H_
#define NSYNC_PUBLIC_NSYNC_ATOMIC_H_

#include "nsync_cpp.h"

/* This file is not to be included directly by the client.  It exists because
   on some platforms, one cannot use a simple uint32_t with atomic operations.
   */
#if NSYNC_ATOMIC_TYPECHECK
#include <inttypes.h>
NSYNC_CPP_START_
typedef struct { uint32_t value; } nsync_atomic_uint32_;
NSYNC_CPP_END_
#define NSYNC_ATOMIC_UINT32_INIT_ { 0 }
#define NSYNC_ATOMIC_UINT32_LOAD_(p) ((p)->value)
#define NSYNC_ATOMIC_UINT32_STORE_(p,v) ((p)->value = (v))
#define NSYNC_ATOMIC_UINT32_PTR_(p) (&(p)->value)

#elif NSYNC_ATOMIC_C11
#include <stdatomic.h>
NSYNC_CPP_START_
typedef atomic_uint_least32_t nsync_atomic_uint32_;
NSYNC_CPP_END_
#define NSYNC_ATOMIC_UINT32_INIT_ 0
#define NSYNC_ATOMIC_UINT32_LOAD_(p) (*(p))
#define NSYNC_ATOMIC_UINT32_STORE_(p,v) (*(p) = (v))
#define NSYNC_ATOMIC_UINT32_PTR_(p) (p)

#elif NSYNC_ATOMIC_CPP11
#include <atomic>
NSYNC_CPP_START_
typedef std::atomic<uint32_t> nsync_atomic_uint32_;
NSYNC_CPP_END_
#define NSYNC_ATOMIC_UINT32_INIT_ ATOMIC_VAR_INIT (0)
#define NSYNC_ATOMIC_UINT32_LOAD_(p) (std::atomic_load (p))
#define NSYNC_ATOMIC_UINT32_STORE_(p,v) (std::atomic_store ((p), (uint32_t) (v)))
#define NSYNC_ATOMIC_UINT32_PTR_(p) (p)

#else
#include <inttypes.h>
NSYNC_CPP_START_
typedef uint32_t nsync_atomic_uint32_;
NSYNC_CPP_END_
#define NSYNC_ATOMIC_UINT32_INIT_ 0
#define NSYNC_ATOMIC_UINT32_LOAD_(p) (*(p))
#define NSYNC_ATOMIC_UINT32_STORE_(p,v) (*(p) = (v))
#define NSYNC_ATOMIC_UINT32_PTR_(p) (p)
#endif

#endif /*NSYNC_PUBLIC_NSYNC_ATOMIC_H_*/
