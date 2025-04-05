/*
 *
 * Copyright 2016 gRPC authors.
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

#ifndef GRPC_CORE_LIB_IOMGR_RESOURCE_QUOTA_H
#define GRPC_CORE_LIB_IOMGR_RESOURCE_QUOTA_H

#include <grpc/support/port_platform.h>

#include <grpc/grpc.h>

#include "src/core/lib/debug/trace.h"
#include "src/core/lib/iomgr/closure.h"

/** \file Tracks resource usage against a pool.

    The current implementation tracks only memory usage, but in the future
    this may be extended to (for example) threads and file descriptors.

    A grpc_resource_quota represents the pooled resources, and
    grpc_resource_user instances attach to the quota and consume those
    resources. They also offer a vector for reclamation: if we become
    resource constrained, grpc_resource_user instances are asked (in turn) to
    free up whatever they can so that the system as a whole can make progress.

    There are three kinds of reclamation that take place, in order of increasing
    invasiveness:
    - an internal reclamation, where cached resource at the resource user level
      is returned to the quota
    - a benign reclamation phase, whereby resources that are in use but are not
      helping anything make progress are reclaimed
    - a destructive reclamation, whereby resources that are helping something
      make progress may be enacted so that at least one part of the system can
      complete.

    Only one reclamation will be outstanding for a given quota at a given time.
    On each reclamation attempt, the kinds of reclamation are tried in order of
    increasing invasiveness, stopping at the first one that succeeds. Thus, on a
    given reclamation attempt, if internal and benign reclamation both fail, it
    will wind up doing a destructive reclamation. However, the next reclamation
    attempt may then be able to get what it needs via internal or benign
    reclamation, due to resources that may have been freed up by the destructive
    reclamation in the previous attempt.

    Future work will be to expose the current resource pressure so that back
    pressure can be applied to avoid reclamation phases starting.

    Resource users own references to resource quotas, and resource quotas
    maintain lists of users (which users arrange to leave before they are
    destroyed) */

extern grpc_core::TraceFlag grpc_resource_quota_trace;

// TODO(juanlishen): This is a hack. We need to do real accounting instead of
// hard coding.
constexpr size_t GRPC_RESOURCE_QUOTA_CALL_SIZE = 15 * 1024;
constexpr size_t GRPC_RESOURCE_QUOTA_CHANNEL_SIZE = 50 * 1024;

grpc_resource_quota* grpc_resource_quota_ref_internal(
    grpc_resource_quota* resource_quota);
void grpc_resource_quota_unref_internal(grpc_resource_quota* resource_quota);
grpc_resource_quota* grpc_resource_quota_from_channel_args(
    const grpc_channel_args* channel_args, bool create = true);

/* Return a number indicating current memory pressure:
   0.0 ==> no memory usage
   1.0 ==> maximum memory usage */
double grpc_resource_quota_get_memory_pressure(
    grpc_resource_quota* resource_quota);

size_t grpc_resource_quota_peek_size(grpc_resource_quota* resource_quota);

typedef struct grpc_resource_user grpc_resource_user;

grpc_resource_user* grpc_resource_user_create(
    grpc_resource_quota* resource_quota, const char* name);

/* Returns a borrowed reference to the underlying resource quota for this
   resource user. */
grpc_resource_quota* grpc_resource_user_quota(
    grpc_resource_user* resource_user);

void grpc_resource_user_ref(grpc_resource_user* resource_user);
void grpc_resource_user_unref(grpc_resource_user* resource_user);
void grpc_resource_user_shutdown(grpc_resource_user* resource_user);

/* Attempts to get quota from the resource_user to create 'thread_count' number
 * of threads. Returns true if successful (i.e the caller is now free to create
 * 'thread_count' number of threads) or false if quota is not available */
bool grpc_resource_user_allocate_threads(grpc_resource_user* resource_user,
                                         int thread_count);
/* Releases 'thread_count' worth of quota back to the resource user. The quota
 * should have been previously obtained successfully by calling
 * grpc_resource_user_allocate_threads().
 *
 * Note: There need not be an exact one-to-one correspondence between
 * grpc_resource_user_allocate_threads() and grpc_resource_user_free_threads()
 * calls. The only requirement is that the number of threads allocated should
 * all be eventually released */
void grpc_resource_user_free_threads(grpc_resource_user* resource_user,
                                     int thread_count);

/* Allocates from the resource user 'size' worth of memory if this won't exceed
 * the resource quota's total size. Returns whether the allocation is done
 * successfully. If allocated successfully, the memory should be freed by the
 * caller eventually. */
bool grpc_resource_user_safe_alloc(grpc_resource_user* resource_user,
                                   size_t size);
/* Allocates from the resource user 'size' worth of memory.
 * If optional_on_done is NULL, then allocate immediately. This may push the
 * quota over-limit, at which point reclamation will kick in. The caller is
 * always responsible to free the memory eventually.
 * Returns true if the allocation was successful. Otherwise, if optional_on_done
 * is non-NULL, it will be scheduled without error when the allocation has been
 * granted by the quota, and the caller is responsible to free the memory
 * eventually. Or it may be scheduled with an error, in which case the caller
 * fails to allocate the memory and shouldn't free the memory.
 */
bool grpc_resource_user_alloc(grpc_resource_user* resource_user, size_t size,
                              grpc_closure* optional_on_done)
    GRPC_MUST_USE_RESULT;
/* Release memory back to the quota */
void grpc_resource_user_free(grpc_resource_user* resource_user, size_t size);
/* Post a memory reclaimer to the resource user. Only one benign and one
   destructive reclaimer can be posted at once. When executed, the reclaimer
   MUST call grpc_resource_user_finish_reclamation before it completes, to
   return control to the resource quota. */
void grpc_resource_user_post_reclaimer(grpc_resource_user* resource_user,
                                       bool destructive, grpc_closure* closure);
/* Finish a reclamation step */
void grpc_resource_user_finish_reclamation(grpc_resource_user* resource_user);

/* Helper to allocate slices from a resource user */
typedef struct grpc_resource_user_slice_allocator {
  /* Closure for when a resource user allocation completes */
  grpc_closure on_allocated;
  /* Closure to call when slices have been allocated */
  grpc_closure on_done;
  /* Length of slices to allocate on the current request */
  size_t length;
  /* Number of slices to allocate on the current request */
  size_t count;
  /* Destination for slices to allocate on the current request */
  grpc_slice_buffer* dest;
  /* Parent resource user */
  grpc_resource_user* resource_user;
} grpc_resource_user_slice_allocator;

/* Initialize a slice allocator.
   When an allocation is completed, calls \a cb with arg \p. */
void grpc_resource_user_slice_allocator_init(
    grpc_resource_user_slice_allocator* slice_allocator,
    grpc_resource_user* resource_user, grpc_iomgr_cb_func cb, void* p);

/* Allocate \a count slices of length \a length into \a dest. Only one request
   can be outstanding at a time.
   Returns whether the slice was allocated inline in the function. If true,
   the \a slice_allocator->on_allocated callback will not be called. */
bool grpc_resource_user_alloc_slices(
    grpc_resource_user_slice_allocator* slice_allocator, size_t length,
    size_t count, grpc_slice_buffer* dest) GRPC_MUST_USE_RESULT;

#endif /* GRPC_CORE_LIB_IOMGR_RESOURCE_QUOTA_H */
