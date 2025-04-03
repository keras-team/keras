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

#ifndef GRPC_CORE_LIB_IOMGR_CLOSURE_H
#define GRPC_CORE_LIB_IOMGR_CLOSURE_H

#include <grpc/support/port_platform.h>

#include <assert.h>
#include <stdbool.h>

#include <grpc/support/alloc.h>
#include <grpc/support/log.h>

#include "src/core/lib/gprpp/debug_location.h"
#include "src/core/lib/gprpp/manual_constructor.h"
#include "src/core/lib/gprpp/mpscq.h"
#include "src/core/lib/iomgr/error.h"
#include "src/core/lib/profiling/timers.h"

struct grpc_closure;
typedef struct grpc_closure grpc_closure;

extern grpc_core::DebugOnlyTraceFlag grpc_trace_closure;

typedef struct grpc_closure_list {
  grpc_closure* head;
  grpc_closure* tail;
} grpc_closure_list;

/** gRPC Callback definition.
 *
 * \param arg Arbitrary input.
 * \param error GRPC_ERROR_NONE if no error occurred, otherwise some grpc_error
 *              describing what went wrong.
 *              Error contract: it is not the cb's job to unref this error;
 *              the closure scheduler will do that after the cb returns */
typedef void (*grpc_iomgr_cb_func)(void* arg, grpc_error* error);

/** A closure over a grpc_iomgr_cb_func. */
struct grpc_closure {
  /** Once queued, next indicates the next queued closure; before then, scratch
   *  space */
  union {
    grpc_closure* next;
    grpc_core::ManualConstructor<
        grpc_core::MultiProducerSingleConsumerQueue::Node>
        mpscq_node;
    uintptr_t scratch;
  } next_data;

  /** Bound callback. */
  grpc_iomgr_cb_func cb;

  /** Arguments to be passed to "cb". */
  void* cb_arg;

  /** Once queued, the result of the closure. Before then: scratch space */
  union {
    grpc_error* error;
    uintptr_t scratch;
  } error_data;

// extra tracing and debugging for grpc_closure. This incurs a decent amount of
// overhead per closure, so it must be enabled at compile time.
#ifndef NDEBUG
  bool scheduled;
  bool run;  // true = run, false = scheduled
  const char* file_created;
  int line_created;
  const char* file_initiated;
  int line_initiated;
#endif
};

#ifndef NDEBUG
inline grpc_closure* grpc_closure_init(const char* file, int line,
                                       grpc_closure* closure,
                                       grpc_iomgr_cb_func cb, void* cb_arg) {
#else
inline grpc_closure* grpc_closure_init(grpc_closure* closure,
                                       grpc_iomgr_cb_func cb, void* cb_arg) {
#endif
  closure->cb = cb;
  closure->cb_arg = cb_arg;
  closure->error_data.error = GRPC_ERROR_NONE;
#ifndef NDEBUG
  closure->scheduled = false;
  closure->file_initiated = nullptr;
  closure->line_initiated = 0;
  closure->run = false;
  closure->file_created = file;
  closure->line_created = line;
#endif
  return closure;
}

/** Initializes \a closure with \a cb and \a cb_arg. Returns \a closure. */
#ifndef NDEBUG
#define GRPC_CLOSURE_INIT(closure, cb, cb_arg, scheduler) \
  grpc_closure_init(__FILE__, __LINE__, closure, cb, cb_arg)
#else
#define GRPC_CLOSURE_INIT(closure, cb, cb_arg, scheduler) \
  grpc_closure_init(closure, cb, cb_arg)
#endif

namespace closure_impl {

typedef struct {
  grpc_iomgr_cb_func cb;
  void* cb_arg;
  grpc_closure wrapper;
} wrapped_closure;

inline void closure_wrapper(void* arg, grpc_error* error) {
  wrapped_closure* wc = static_cast<wrapped_closure*>(arg);
  grpc_iomgr_cb_func cb = wc->cb;
  void* cb_arg = wc->cb_arg;
  gpr_free(wc);
  cb(cb_arg, error);
}

}  // namespace closure_impl

#ifndef NDEBUG
inline grpc_closure* grpc_closure_create(const char* file, int line,
                                         grpc_iomgr_cb_func cb, void* cb_arg) {
#else
inline grpc_closure* grpc_closure_create(grpc_iomgr_cb_func cb, void* cb_arg) {
#endif
  closure_impl::wrapped_closure* wc =
      static_cast<closure_impl::wrapped_closure*>(gpr_malloc(sizeof(*wc)));
  wc->cb = cb;
  wc->cb_arg = cb_arg;
#ifndef NDEBUG
  grpc_closure_init(file, line, &wc->wrapper, closure_impl::closure_wrapper,
                    wc);
#else
  grpc_closure_init(&wc->wrapper, closure_impl::closure_wrapper, wc);
#endif
  return &wc->wrapper;
}

/* Create a heap allocated closure: try to avoid except for very rare events */
#ifndef NDEBUG
#define GRPC_CLOSURE_CREATE(cb, cb_arg, scheduler) \
  grpc_closure_create(__FILE__, __LINE__, cb, cb_arg)
#else
#define GRPC_CLOSURE_CREATE(cb, cb_arg, scheduler) \
  grpc_closure_create(cb, cb_arg)
#endif

#define GRPC_CLOSURE_LIST_INIT \
  { nullptr, nullptr }

inline void grpc_closure_list_init(grpc_closure_list* closure_list) {
  closure_list->head = closure_list->tail = nullptr;
}

/** add \a closure to the end of \a list
    and set \a closure's result to \a error
    Returns true if \a list becomes non-empty */
inline bool grpc_closure_list_append(grpc_closure_list* closure_list,
                                     grpc_closure* closure, grpc_error* error) {
  if (closure == nullptr) {
    GRPC_ERROR_UNREF(error);
    return false;
  }
  closure->error_data.error = error;
  closure->next_data.next = nullptr;
  bool was_empty = (closure_list->head == nullptr);
  if (was_empty) {
    closure_list->head = closure;
  } else {
    closure_list->tail->next_data.next = closure;
  }
  closure_list->tail = closure;
  return was_empty;
}

/** force all success bits in \a list to false */
inline void grpc_closure_list_fail_all(grpc_closure_list* list,
                                       grpc_error* forced_failure) {
  for (grpc_closure* c = list->head; c != nullptr; c = c->next_data.next) {
    if (c->error_data.error == GRPC_ERROR_NONE) {
      c->error_data.error = GRPC_ERROR_REF(forced_failure);
    }
  }
  GRPC_ERROR_UNREF(forced_failure);
}

/** append all closures from \a src to \a dst and empty \a src. */
inline void grpc_closure_list_move(grpc_closure_list* src,
                                   grpc_closure_list* dst) {
  if (src->head == nullptr) {
    return;
  }
  if (dst->head == nullptr) {
    *dst = *src;
  } else {
    dst->tail->next_data.next = src->head;
    dst->tail = src->tail;
  }
  src->head = src->tail = nullptr;
}

/** return whether \a list is empty. */
inline bool grpc_closure_list_empty(grpc_closure_list closure_list) {
  return closure_list.head == nullptr;
}

namespace grpc_core {
class Closure {
 public:
  static void Run(const DebugLocation& location, grpc_closure* closure,
                  grpc_error* error) {
    (void)location;
    if (closure == nullptr) {
      GRPC_ERROR_UNREF(error);
      return;
    }
#ifndef NDEBUG
    if (grpc_trace_closure.enabled()) {
      gpr_log(GPR_DEBUG, "running closure %p: created [%s:%d]: run [%s:%d]",
              closure, closure->file_created, closure->line_created,
              location.file(), location.line());
    }
    GPR_ASSERT(closure->cb != nullptr);
#endif
    closure->cb(closure->cb_arg, error);
#ifndef NDEBUG
    if (grpc_trace_closure.enabled()) {
      gpr_log(GPR_DEBUG, "closure %p finished", closure);
    }
#endif
    GRPC_ERROR_UNREF(error);
  }
};
}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_IOMGR_CLOSURE_H */
