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

#ifndef GRPC_CORE_LIB_CHANNEL_CHANNEL_STACK_H
#define GRPC_CORE_LIB_CHANNEL_CHANNEL_STACK_H

//////////////////////////////////////////////////////////////////////////////
// IMPORTANT NOTE:
//
// When you update this API, please make the corresponding changes to
// the C++ API in src/cpp/common/channel_filter.{h,cc}
//////////////////////////////////////////////////////////////////////////////

/* A channel filter defines how operations on a channel are implemented.
   Channel filters are chained together to create full channels, and if those
   chains are linear, then channel stacks provide a mechanism to minimize
   allocations for that chain.
   Call stacks are created by channel stacks and represent the per-call data
   for that stack. */

#include <grpc/support/port_platform.h>

#include <stddef.h>

#include <grpc/grpc.h>
#include <grpc/support/log.h>
#include <grpc/support/time.h>

#include "src/core/lib/debug/trace.h"
#include "src/core/lib/gpr/time_precise.h"
#include "src/core/lib/gprpp/arena.h"
#include "src/core/lib/iomgr/call_combiner.h"
#include "src/core/lib/iomgr/polling_entity.h"
#include "src/core/lib/transport/transport.h"

typedef struct grpc_channel_element grpc_channel_element;
typedef struct grpc_call_element grpc_call_element;

typedef struct grpc_channel_stack grpc_channel_stack;
typedef struct grpc_call_stack grpc_call_stack;

typedef struct {
  grpc_channel_stack* channel_stack;
  const grpc_channel_args* channel_args;
  /** Transport, iff it is known */
  grpc_transport* optional_transport;
  int is_first;
  int is_last;
} grpc_channel_element_args;

typedef struct {
  grpc_call_stack* call_stack;
  const void* server_transport_data;
  grpc_call_context_element* context;
  const grpc_slice& path;
  gpr_cycle_counter start_time;
  grpc_millis deadline;
  grpc_core::Arena* arena;
  grpc_core::CallCombiner* call_combiner;
} grpc_call_element_args;

typedef struct {
  grpc_transport_stream_stats transport_stream_stats;
  gpr_timespec latency; /* From call creating to enqueing of received status */
} grpc_call_stats;

/** Information about the call upon completion. */
struct grpc_call_final_info {
  grpc_call_stats stats;
  grpc_status_code final_status = GRPC_STATUS_OK;
  const char* error_string = nullptr;
};

/* Channel filters specify:
   1. the amount of memory needed in the channel & call (via the sizeof_XXX
      members)
   2. functions to initialize and destroy channel & call data
      (init_XXX, destroy_XXX)
   3. functions to implement call operations and channel operations (call_op,
      channel_op)
   4. a name, which is useful when debugging

   Members are laid out in approximate frequency of use order. */
typedef struct {
  /* Called to eg. send/receive data on a call.
     See grpc_call_next_op on how to call the next element in the stack */
  void (*start_transport_stream_op_batch)(grpc_call_element* elem,
                                          grpc_transport_stream_op_batch* op);
  /* Called to handle channel level operations - e.g. new calls, or transport
     closure.
     See grpc_channel_next_op on how to call the next element in the stack */
  void (*start_transport_op)(grpc_channel_element* elem, grpc_transport_op* op);

  /* sizeof(per call data) */
  size_t sizeof_call_data;
  /* Initialize per call data.
     elem is initialized at the start of the call, and elem->call_data is what
     needs initializing.
     The filter does not need to do any chaining.
     server_transport_data is an opaque pointer. If it is NULL, this call is
     on a client; if it is non-NULL, then it points to memory owned by the
     transport and is on the server. Most filters want to ignore this
     argument.
     Implementations may assume that elem->call_data is all zeros. */
  grpc_error* (*init_call_elem)(grpc_call_element* elem,
                                const grpc_call_element_args* args);
  void (*set_pollset_or_pollset_set)(grpc_call_element* elem,
                                     grpc_polling_entity* pollent);
  /* Destroy per call data.
     The filter does not need to do any chaining.
     The bottom filter of a stack will be passed a non-NULL pointer to
     \a then_schedule_closure that should be passed to GRPC_CLOSURE_SCHED when
     destruction is complete. \a final_info contains data about the completed
     call, mainly for reporting purposes. */
  void (*destroy_call_elem)(grpc_call_element* elem,
                            const grpc_call_final_info* final_info,
                            grpc_closure* then_schedule_closure);

  /* sizeof(per channel data) */
  size_t sizeof_channel_data;
  /* Initialize per-channel data.
     elem is initialized at the creating of the channel, and elem->channel_data
     is what needs initializing.
     is_first, is_last designate this elements position in the stack, and are
     useful for asserting correct configuration by upper layer code.
     The filter does not need to do any chaining.
     Implementations may assume that elem->channel_data is all zeros. */
  grpc_error* (*init_channel_elem)(grpc_channel_element* elem,
                                   grpc_channel_element_args* args);
  /* Destroy per channel data.
     The filter does not need to do any chaining */
  void (*destroy_channel_elem)(grpc_channel_element* elem);

  /* Implement grpc_channel_get_info() */
  void (*get_channel_info)(grpc_channel_element* elem,
                           const grpc_channel_info* channel_info);

  /* The name of this filter */
  const char* name;
} grpc_channel_filter;

/* A channel_element tracks its filter and the filter requested memory within
   a channel allocation */
struct grpc_channel_element {
  const grpc_channel_filter* filter;
  void* channel_data;
};

/* A call_element tracks its filter, the filter requested memory within
   a channel allocation, and the filter requested memory within a call
   allocation */
struct grpc_call_element {
  const grpc_channel_filter* filter;
  void* channel_data;
  void* call_data;
};

/* A channel stack tracks a set of related filters for one channel, and
   guarantees they live within a single malloc() allocation */
struct grpc_channel_stack {
  grpc_stream_refcount refcount;
  size_t count;
  /* Memory required for a call stack (computed at channel stack
     initialization) */
  size_t call_stack_size;
};

/* A call stack tracks a set of related filters for one call, and guarantees
   they live within a single malloc() allocation */
struct grpc_call_stack {
  /* shared refcount for this channel stack.
     MUST be the first element: the underlying code calls destroy
     with the address of the refcount, but higher layers prefer to think
     about the address of the call stack itself. */
  grpc_stream_refcount refcount;
  size_t count;
};

/* Get a channel element given a channel stack and its index */
grpc_channel_element* grpc_channel_stack_element(grpc_channel_stack* stack,
                                                 size_t i);
/* Get the last channel element in a channel stack */
grpc_channel_element* grpc_channel_stack_last_element(
    grpc_channel_stack* stack);
/* Get a call stack element given a call stack and an index */
grpc_call_element* grpc_call_stack_element(grpc_call_stack* stack, size_t i);

/* Determine memory required for a channel stack containing a set of filters */
size_t grpc_channel_stack_size(const grpc_channel_filter** filters,
                               size_t filter_count);
/* Initialize a channel stack given some filters */
grpc_error* grpc_channel_stack_init(
    int initial_refs, grpc_iomgr_cb_func destroy, void* destroy_arg,
    const grpc_channel_filter** filters, size_t filter_count,
    const grpc_channel_args* args, grpc_transport* optional_transport,
    const char* name, grpc_channel_stack* stack);
/* Destroy a channel stack */
void grpc_channel_stack_destroy(grpc_channel_stack* stack);

/* Initialize a call stack given a channel stack. transport_server_data is
   expected to be NULL on a client, or an opaque transport owned pointer on the
   server. */
grpc_error* grpc_call_stack_init(grpc_channel_stack* channel_stack,
                                 int initial_refs, grpc_iomgr_cb_func destroy,
                                 void* destroy_arg,
                                 const grpc_call_element_args* elem_args);
/* Set a pollset or a pollset_set for a call stack: must occur before the first
 * op is started */
void grpc_call_stack_set_pollset_or_pollset_set(grpc_call_stack* call_stack,
                                                grpc_polling_entity* pollent);

#ifndef NDEBUG
#define GRPC_CALL_STACK_REF(call_stack, reason) \
  grpc_stream_ref(&(call_stack)->refcount, reason)
#define GRPC_CALL_STACK_UNREF(call_stack, reason) \
  grpc_stream_unref(&(call_stack)->refcount, reason)
#define GRPC_CHANNEL_STACK_REF(channel_stack, reason) \
  grpc_stream_ref(&(channel_stack)->refcount, reason)
#define GRPC_CHANNEL_STACK_UNREF(channel_stack, reason) \
  grpc_stream_unref(&(channel_stack)->refcount, reason)
#else
#define GRPC_CALL_STACK_REF(call_stack, reason) \
  do {                                          \
    grpc_stream_ref(&(call_stack)->refcount);   \
    (void)(reason);                             \
  } while (0);
#define GRPC_CALL_STACK_UNREF(call_stack, reason) \
  do {                                            \
    grpc_stream_unref(&(call_stack)->refcount);   \
    (void)(reason);                               \
  } while (0);
#define GRPC_CHANNEL_STACK_REF(channel_stack, reason) \
  do {                                                \
    grpc_stream_ref(&(channel_stack)->refcount);      \
    (void)(reason);                                   \
  } while (0);
#define GRPC_CHANNEL_STACK_UNREF(channel_stack, reason) \
  do {                                                  \
    grpc_stream_unref(&(channel_stack)->refcount);      \
    (void)(reason);                                     \
  } while (0);
#endif

/* Destroy a call stack */
void grpc_call_stack_destroy(grpc_call_stack* stack,
                             const grpc_call_final_info* final_info,
                             grpc_closure* then_schedule_closure);

/* Ignore set pollset{_set} - used by filters if they don't care about pollsets
 * at all. Does nothing. */
void grpc_call_stack_ignore_set_pollset_or_pollset_set(
    grpc_call_element* elem, grpc_polling_entity* pollent);
/* Call the next operation in a call stack */
void grpc_call_next_op(grpc_call_element* elem,
                       grpc_transport_stream_op_batch* op);
/* Call the next operation (depending on call directionality) in a channel
   stack */
void grpc_channel_next_op(grpc_channel_element* elem, grpc_transport_op* op);
/* Pass through a request to get_channel_info() to the next child element */
void grpc_channel_next_get_info(grpc_channel_element* elem,
                                const grpc_channel_info* channel_info);

/* Given the top element of a channel stack, get the channel stack itself */
grpc_channel_stack* grpc_channel_stack_from_top_element(
    grpc_channel_element* elem);
/* Given the top element of a call stack, get the call stack itself */
grpc_call_stack* grpc_call_stack_from_top_element(grpc_call_element* elem);

void grpc_call_log_op(const char* file, int line, gpr_log_severity severity,
                      grpc_call_element* elem,
                      grpc_transport_stream_op_batch* op);

extern grpc_core::TraceFlag grpc_trace_channel;

#define GRPC_CALL_LOG_OP(sev, elem, op)                \
  do {                                                 \
    if (GRPC_TRACE_FLAG_ENABLED(grpc_trace_channel)) { \
      grpc_call_log_op(sev, elem, op);                 \
    }                                                  \
  } while (0)

#endif /* GRPC_CORE_LIB_CHANNEL_CHANNEL_STACK_H */
