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

#ifndef GRPC_CORE_LIB_IOMGR_EXEC_CTX_H
#define GRPC_CORE_LIB_IOMGR_EXEC_CTX_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/grpc_types.h>
#include <grpc/support/atm.h>
#include <grpc/support/cpu.h>
#include <grpc/support/log.h>

#include "src/core/lib/gpr/time_precise.h"
#include "src/core/lib/gpr/tls.h"
#include "src/core/lib/gprpp/debug_location.h"
#include "src/core/lib/gprpp/fork.h"
#include "src/core/lib/iomgr/closure.h"

typedef int64_t grpc_millis;

#define GRPC_MILLIS_INF_FUTURE INT64_MAX
#define GRPC_MILLIS_INF_PAST INT64_MIN

/** A combiner represents a list of work to be executed later.
    Forward declared here to avoid a circular dependency with combiner.h. */
typedef struct grpc_combiner grpc_combiner;

/* This exec_ctx is ready to return: either pre-populated, or cached as soon as
   the finish_check returns true */
#define GRPC_EXEC_CTX_FLAG_IS_FINISHED 1
/* The exec_ctx's thread is (potentially) owned by a call or channel: care
   should be given to not delete said call/channel from this exec_ctx */
#define GRPC_EXEC_CTX_FLAG_THREAD_RESOURCE_LOOP 2
/* This exec ctx was initialized by an internal thread, and should not
   be counted by fork handlers */
#define GRPC_EXEC_CTX_FLAG_IS_INTERNAL_THREAD 4

/* This application callback exec ctx was initialized by an internal thread, and
   should not be counted by fork handlers */
#define GRPC_APP_CALLBACK_EXEC_CTX_FLAG_IS_INTERNAL_THREAD 1

gpr_timespec grpc_millis_to_timespec(grpc_millis millis, gpr_clock_type clock);
grpc_millis grpc_timespec_to_millis_round_down(gpr_timespec timespec);
grpc_millis grpc_timespec_to_millis_round_up(gpr_timespec timespec);
grpc_millis grpc_cycle_counter_to_millis_round_down(gpr_cycle_counter cycles);
grpc_millis grpc_cycle_counter_to_millis_round_up(gpr_cycle_counter cycles);

namespace grpc_core {
class Combiner;
/** Execution context.
 *  A bag of data that collects information along a callstack.
 *  It is created on the stack at core entry points (public API or iomgr), and
 *  stored internally as a thread-local variable.
 *
 *  Generally, to create an exec_ctx instance, add the following line at the top
 *  of the public API entry point or at the start of a thread's work function :
 *
 *  grpc_core::ExecCtx exec_ctx;
 *
 *  Access the created ExecCtx instance using :
 *  grpc_core::ExecCtx::Get()
 *
 *  Specific responsibilities (this may grow in the future):
 *  - track a list of core work that needs to be delayed until the base of the
 *    call stack (this provides a convenient mechanism to run callbacks
 *    without worrying about locking issues)
 *  - provide a decision maker (via IsReadyToFinish) that provides a
 *    signal as to whether a borrowed thread should continue to do work or
 *    should actively try to finish up and get this thread back to its owner
 *
 *  CONVENTIONS:
 *  - Instance of this must ALWAYS be constructed on the stack, never
 *    heap allocated.
 *  - Do not pass exec_ctx as a parameter to a function. Always access it using
 *    grpc_core::ExecCtx::Get().
 *  - NOTE: In the future, the convention is likely to change to allow only one
 *          ExecCtx on a thread's stack at the same time. The TODO below
 *          discusses this plan in more detail.
 *
 * TODO(yashykt): Only allow one "active" ExecCtx on a thread at the same time.
 *                Stage 1: If a new one is created on the stack, it should just
 *                pass-through to the underlying ExecCtx deeper in the thread's
 *                stack.
 *                Stage 2: Assert if a 2nd one is ever created on the stack
 *                since that implies a core re-entry outside of application
 *                callbacks.
 */
class ExecCtx {
 public:
  /** Default Constructor */

  ExecCtx() : flags_(GRPC_EXEC_CTX_FLAG_IS_FINISHED) {
    grpc_core::Fork::IncExecCtxCount();
    Set(this);
  }

  /** Parameterised Constructor */
  ExecCtx(uintptr_t fl) : flags_(fl) {
    if (!(GRPC_EXEC_CTX_FLAG_IS_INTERNAL_THREAD & flags_)) {
      grpc_core::Fork::IncExecCtxCount();
    }
    Set(this);
  }

  /** Destructor */
  virtual ~ExecCtx() {
    flags_ |= GRPC_EXEC_CTX_FLAG_IS_FINISHED;
    Flush();
    Set(last_exec_ctx_);
    if (!(GRPC_EXEC_CTX_FLAG_IS_INTERNAL_THREAD & flags_)) {
      grpc_core::Fork::DecExecCtxCount();
    }
  }

  /** Disallow copy and assignment operators */
  ExecCtx(const ExecCtx&) = delete;
  ExecCtx& operator=(const ExecCtx&) = delete;

  unsigned starting_cpu() const { return starting_cpu_; }

  struct CombinerData {
    /* currently active combiner: updated only via combiner.c */
    Combiner* active_combiner;
    /* last active combiner in the active combiner list */
    Combiner* last_combiner;
  };

  /** Only to be used by grpc-combiner code */
  CombinerData* combiner_data() { return &combiner_data_; }

  /** Return pointer to grpc_closure_list */
  grpc_closure_list* closure_list() { return &closure_list_; }

  /** Return flags */
  uintptr_t flags() { return flags_; }

  /** Checks if there is work to be done */
  bool HasWork() {
    return combiner_data_.active_combiner != nullptr ||
           !grpc_closure_list_empty(closure_list_);
  }

  /** Flush any work that has been enqueued onto this grpc_exec_ctx.
   *  Caller must guarantee that no interfering locks are held.
   *  Returns true if work was performed, false otherwise.
   */
  bool Flush();

  /** Returns true if we'd like to leave this execution context as soon as
   *  possible: useful for deciding whether to do something more or not
   *  depending on outside context.
   */
  bool IsReadyToFinish() {
    if ((flags_ & GRPC_EXEC_CTX_FLAG_IS_FINISHED) == 0) {
      if (CheckReadyToFinish()) {
        flags_ |= GRPC_EXEC_CTX_FLAG_IS_FINISHED;
        return true;
      }
      return false;
    } else {
      return true;
    }
  }

  /** Returns the stored current time relative to start if valid,
   *  otherwise refreshes the stored time, sets it valid and returns the new
   *  value.
   */
  grpc_millis Now();

  /** Invalidates the stored time value. A new time value will be set on calling
   *  Now().
   */
  void InvalidateNow() { now_is_valid_ = false; }

  /** To be used only by shutdown code in iomgr */
  void SetNowIomgrShutdown() {
    now_ = GRPC_MILLIS_INF_FUTURE;
    now_is_valid_ = true;
  }

  /** To be used only for testing.
   *  Sets the now value.
   */
  void TestOnlySetNow(grpc_millis new_val) {
    now_ = new_val;
    now_is_valid_ = true;
  }

  static void TestOnlyGlobalInit(gpr_timespec new_val);

  /** Global initialization for ExecCtx. Called by iomgr. */
  static void GlobalInit(void);

  /** Global shutdown for ExecCtx. Called by iomgr. */
  static void GlobalShutdown(void) { gpr_tls_destroy(&exec_ctx_); }

  /** Gets pointer to current exec_ctx. */
  static ExecCtx* Get() {
    return reinterpret_cast<ExecCtx*>(gpr_tls_get(&exec_ctx_));
  }

  static void Set(ExecCtx* exec_ctx) {
    gpr_tls_set(&exec_ctx_, reinterpret_cast<intptr_t>(exec_ctx));
  }

  static void Run(const DebugLocation& location, grpc_closure* closure,
                  grpc_error* error);

  static void RunList(const DebugLocation& location, grpc_closure_list* list);

 protected:
  /** Check if ready to finish. */
  virtual bool CheckReadyToFinish() { return false; }

  /** Disallow delete on ExecCtx. */
  static void operator delete(void* /* p */) { abort(); }

 private:
  /** Set exec_ctx_ to exec_ctx. */

  grpc_closure_list closure_list_ = GRPC_CLOSURE_LIST_INIT;
  CombinerData combiner_data_ = {nullptr, nullptr};
  uintptr_t flags_;

  unsigned starting_cpu_ = gpr_cpu_current_cpu();

  bool now_is_valid_ = false;
  grpc_millis now_ = 0;

  GPR_TLS_CLASS_DECL(exec_ctx_);
  ExecCtx* last_exec_ctx_ = Get();
};

/** Application-callback execution context.
 *  A bag of data that collects information along a callstack.
 *  It is created on the stack at core entry points, and stored internally
 *  as a thread-local variable.
 *
 *  There are three key differences between this structure and ExecCtx:
 *    1. ApplicationCallbackExecCtx builds a list of application-level
 *       callbacks, but ExecCtx builds a list of internal callbacks to invoke.
 *    2. ApplicationCallbackExecCtx invokes its callbacks only at destruction;
 *       there is no explicit Flush method.
 *    3. If more than one ApplicationCallbackExecCtx is created on the thread's
 *       stack, only the one closest to the base of the stack is actually
 *       active and this is the only one that enqueues application callbacks.
 *       (Unlike ExecCtx, it is not feasible to prevent multiple of these on the
 *       stack since the executing application callback may itself enter core.
 *       However, the new one created will just pass callbacks through to the
 *       base one and those will not be executed until the return to the
 *       destructor of the base one, preventing unlimited stack growth.)
 *
 *  This structure exists because application callbacks may themselves cause a
 *  core re-entry (e.g., through a public API call) and if that call in turn
 *  causes another application-callback, there could be arbitrarily growing
 *  stacks of core re-entries. Instead, any application callbacks instead should
 *  not be invoked until other core work is done and other application callbacks
 *  have completed. To accomplish this, any application callback should be
 *  enqueued using grpc_core::ApplicationCallbackExecCtx::Enqueue .
 *
 *  CONVENTIONS:
 *  - Instances of this must ALWAYS be constructed on the stack, never
 *    heap allocated.
 *  - Instances of this are generally constructed before ExecCtx when needed.
 *    The only exception is for ExecCtx's that are explicitly flushed and
 *    that survive beyond the scope of the function that can cause application
 *    callbacks to be invoked (e.g., in the timer thread).
 *
 *  Generally, core entry points that may trigger application-level callbacks
 *  will have the following declarations:
 *
 *  grpc_core::ApplicationCallbackExecCtx callback_exec_ctx;
 *  grpc_core::ExecCtx exec_ctx;
 *
 *  This ordering is important to make sure that the ApplicationCallbackExecCtx
 *  is destroyed after the ExecCtx (to prevent the re-entry problem described
 *  above, as well as making sure that ExecCtx core callbacks are invoked first)
 *
 */

class ApplicationCallbackExecCtx {
 public:
  /** Default Constructor */
  ApplicationCallbackExecCtx() { Set(this, flags_); }

  /** Parameterised Constructor */
  ApplicationCallbackExecCtx(uintptr_t fl) : flags_(fl) { Set(this, flags_); }

  ~ApplicationCallbackExecCtx() {
    if (reinterpret_cast<ApplicationCallbackExecCtx*>(
            gpr_tls_get(&callback_exec_ctx_)) == this) {
      while (head_ != nullptr) {
        auto* f = head_;
        head_ = f->internal_next;
        if (f->internal_next == nullptr) {
          tail_ = nullptr;
        }
        (*f->functor_run)(f, f->internal_success);
      }
      gpr_tls_set(&callback_exec_ctx_, reinterpret_cast<intptr_t>(nullptr));
      if (!(GRPC_APP_CALLBACK_EXEC_CTX_FLAG_IS_INTERNAL_THREAD & flags_)) {
        grpc_core::Fork::DecExecCtxCount();
      }
    } else {
      GPR_DEBUG_ASSERT(head_ == nullptr);
      GPR_DEBUG_ASSERT(tail_ == nullptr);
    }
  }

  static void Set(ApplicationCallbackExecCtx* exec_ctx, uintptr_t flags) {
    if (reinterpret_cast<ApplicationCallbackExecCtx*>(
            gpr_tls_get(&callback_exec_ctx_)) == nullptr) {
      if (!(GRPC_APP_CALLBACK_EXEC_CTX_FLAG_IS_INTERNAL_THREAD & flags)) {
        grpc_core::Fork::IncExecCtxCount();
      }
      gpr_tls_set(&callback_exec_ctx_, reinterpret_cast<intptr_t>(exec_ctx));
    }
  }

  static void Enqueue(grpc_experimental_completion_queue_functor* functor,
                      int is_success) {
    functor->internal_success = is_success;
    functor->internal_next = nullptr;

    auto* ctx = reinterpret_cast<ApplicationCallbackExecCtx*>(
        gpr_tls_get(&callback_exec_ctx_));

    if (ctx->head_ == nullptr) {
      ctx->head_ = functor;
    }
    if (ctx->tail_ != nullptr) {
      ctx->tail_->internal_next = functor;
    }
    ctx->tail_ = functor;
  }

  /** Global initialization for ApplicationCallbackExecCtx. Called by init. */
  static void GlobalInit(void) { gpr_tls_init(&callback_exec_ctx_); }

  /** Global shutdown for ApplicationCallbackExecCtx. Called by init. */
  static void GlobalShutdown(void) { gpr_tls_destroy(&callback_exec_ctx_); }

 private:
  uintptr_t flags_{0u};
  grpc_experimental_completion_queue_functor* head_{nullptr};
  grpc_experimental_completion_queue_functor* tail_{nullptr};
  GPR_TLS_CLASS_DECL(callback_exec_ctx_);
};
}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_IOMGR_EXEC_CTX_H */
