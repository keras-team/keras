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

#ifndef GRPC_CORE_LIB_IOMGR_ERROR_H
#define GRPC_CORE_LIB_IOMGR_ERROR_H

#include <grpc/support/port_platform.h>

#include <inttypes.h>
#include <stdbool.h>

#include <grpc/slice.h>
#include <grpc/status.h>
#include <grpc/support/log.h>
#include <grpc/support/time.h>

#include "src/core/lib/debug/trace.h"
#include "src/core/lib/gprpp/inlined_vector.h"

/// Opaque representation of an error.
/// See https://github.com/grpc/grpc/blob/master/doc/core/grpc-error.md for a
/// full write up of this object.

typedef struct grpc_error grpc_error;

extern grpc_core::DebugOnlyTraceFlag grpc_trace_error_refcount;

typedef enum {
  /// 'errno' from the operating system
  GRPC_ERROR_INT_ERRNO,
  /// __LINE__ from the call site creating the error
  GRPC_ERROR_INT_FILE_LINE,
  /// stream identifier: for errors that are associated with an individual
  /// wire stream
  GRPC_ERROR_INT_STREAM_ID,
  /// grpc status code representing this error
  GRPC_ERROR_INT_GRPC_STATUS,
  /// offset into some binary blob (usually represented by
  /// GRPC_ERROR_STR_RAW_BYTES) where the error occurred
  GRPC_ERROR_INT_OFFSET,
  /// context sensitive index associated with the error
  GRPC_ERROR_INT_INDEX,
  /// context sensitive size associated with the error
  GRPC_ERROR_INT_SIZE,
  /// http2 error code associated with the error (see the HTTP2 RFC)
  GRPC_ERROR_INT_HTTP2_ERROR,
  /// TSI status code associated with the error
  GRPC_ERROR_INT_TSI_CODE,
  /// grpc_security_status associated with the error
  GRPC_ERROR_INT_SECURITY_STATUS,
  /// WSAGetLastError() reported when this error occurred
  GRPC_ERROR_INT_WSA_ERROR,
  /// File descriptor associated with this error
  GRPC_ERROR_INT_FD,
  /// HTTP status (i.e. 404)
  GRPC_ERROR_INT_HTTP_STATUS,
  /// context sensitive limit associated with the error
  GRPC_ERROR_INT_LIMIT,
  /// chttp2: did the error occur while a write was in progress
  GRPC_ERROR_INT_OCCURRED_DURING_WRITE,
  /// channel connectivity state associated with the error
  GRPC_ERROR_INT_CHANNEL_CONNECTIVITY_STATE,

  /// Must always be last
  GRPC_ERROR_INT_MAX,
} grpc_error_ints;

typedef enum {
  /// top-level textual description of this error
  GRPC_ERROR_STR_DESCRIPTION,
  /// source file in which this error occurred
  GRPC_ERROR_STR_FILE,
  /// operating system description of this error
  GRPC_ERROR_STR_OS_ERROR,
  /// syscall that generated this error
  GRPC_ERROR_STR_SYSCALL,
  /// peer that we were trying to communicate when this error occurred
  GRPC_ERROR_STR_TARGET_ADDRESS,
  /// grpc status message associated with this error
  GRPC_ERROR_STR_GRPC_MESSAGE,
  /// hex dump (or similar) with the data that generated this error
  GRPC_ERROR_STR_RAW_BYTES,
  /// tsi error string associated with this error
  GRPC_ERROR_STR_TSI_ERROR,
  /// filename that we were trying to read/write when this error occurred
  GRPC_ERROR_STR_FILENAME,
  /// which data was queued for writing when the error occurred
  GRPC_ERROR_STR_QUEUED_BUFFERS,
  /// key associated with the error
  GRPC_ERROR_STR_KEY,
  /// value associated with the error
  GRPC_ERROR_STR_VALUE,

  /// Must always be last
  GRPC_ERROR_STR_MAX,
} grpc_error_strs;

typedef enum {
  /// timestamp of error creation
  GRPC_ERROR_TIME_CREATED,

  /// Must always be last
  GRPC_ERROR_TIME_MAX,
} grpc_error_times;

/// The following "special" errors can be propagated without allocating memory.
/// They are always even so that other code (particularly combiner locks,
/// polling engines) can safely use the lower bit for themselves.

#define GRPC_ERROR_NONE ((grpc_error*)NULL)
#define GRPC_ERROR_RESERVED_1 ((grpc_error*)1)
#define GRPC_ERROR_OOM ((grpc_error*)2)
#define GRPC_ERROR_RESERVED_2 ((grpc_error*)3)
#define GRPC_ERROR_CANCELLED ((grpc_error*)4)
#define GRPC_ERROR_SPECIAL_MAX GRPC_ERROR_CANCELLED

inline bool grpc_error_is_special(struct grpc_error* err) {
  return err <= GRPC_ERROR_SPECIAL_MAX;
}

// debug only toggles that allow for a sanity to check that ensures we will
// never create any errors in the per-RPC hotpath.
void grpc_disable_error_creation();
void grpc_enable_error_creation();

const char* grpc_error_string(grpc_error* error);

/// Create an error - but use GRPC_ERROR_CREATE instead
grpc_error* grpc_error_create(const char* file, int line,
                              const grpc_slice& desc, grpc_error** referencing,
                              size_t num_referencing);
/// Create an error (this is the preferred way of generating an error that is
///   not due to a system call - for system calls, use GRPC_OS_ERROR or
///   GRPC_WSA_ERROR as appropriate)
/// \a referencing is an array of num_referencing elements indicating one or
/// more errors that are believed to have contributed to this one
/// err = grpc_error_create(x, y, z, r, nr) is equivalent to:
///   err = grpc_error_create(x, y, z, NULL, 0);
///   for (i=0; i<nr; i++) err = grpc_error_add_child(err, r[i]);
#define GRPC_ERROR_CREATE_FROM_STATIC_STRING(desc)                           \
  grpc_error_create(__FILE__, __LINE__, grpc_slice_from_static_string(desc), \
                    NULL, 0)
#define GRPC_ERROR_CREATE_FROM_COPIED_STRING(desc)                           \
  grpc_error_create(__FILE__, __LINE__, grpc_slice_from_copied_string(desc), \
                    NULL, 0)

// Create an error that references some other errors. This function adds a
// reference to each error in errs - it does not consume an existing reference
#define GRPC_ERROR_CREATE_REFERENCING_FROM_STATIC_STRING(desc, errs, count)  \
  grpc_error_create(__FILE__, __LINE__, grpc_slice_from_static_string(desc), \
                    errs, count)
#define GRPC_ERROR_CREATE_REFERENCING_FROM_COPIED_STRING(desc, errs, count)  \
  grpc_error_create(__FILE__, __LINE__, grpc_slice_from_copied_string(desc), \
                    errs, count)

#define GRPC_ERROR_CREATE_FROM_VECTOR(desc, error_list) \
  grpc_error_create_from_vector(__FILE__, __LINE__, desc, error_list)

#ifndef NDEBUG
grpc_error* grpc_error_do_ref(grpc_error* err, const char* file, int line);
void grpc_error_do_unref(grpc_error* err, const char* file, int line);
inline grpc_error* grpc_error_ref(grpc_error* err, const char* file, int line) {
  if (grpc_error_is_special(err)) return err;
  return grpc_error_do_ref(err, file, line);
}
inline void grpc_error_unref(grpc_error* err, const char* file, int line) {
  if (grpc_error_is_special(err)) return;
  grpc_error_do_unref(err, file, line);
}
#define GRPC_ERROR_REF(err) grpc_error_ref(err, __FILE__, __LINE__)
#define GRPC_ERROR_UNREF(err) grpc_error_unref(err, __FILE__, __LINE__)
#else
grpc_error* grpc_error_do_ref(grpc_error* err);
void grpc_error_do_unref(grpc_error* err);
inline grpc_error* grpc_error_ref(grpc_error* err) {
  if (grpc_error_is_special(err)) return err;
  return grpc_error_do_ref(err);
}
inline void grpc_error_unref(grpc_error* err) {
  if (grpc_error_is_special(err)) return;
  grpc_error_do_unref(err);
}
#define GRPC_ERROR_REF(err) grpc_error_ref(err)
#define GRPC_ERROR_UNREF(err) grpc_error_unref(err)
#endif

// Consumes all the errors in the vector and forms a referencing error from
// them. If the vector is empty, return GRPC_ERROR_NONE.
template <size_t N>
static grpc_error* grpc_error_create_from_vector(
    const char* file, int line, const char* desc,
    grpc_core::InlinedVector<grpc_error*, N>* error_list) {
  grpc_error* error = GRPC_ERROR_NONE;
  if (error_list->size() != 0) {
    error = grpc_error_create(file, line, grpc_slice_from_static_string(desc),
                              error_list->data(), error_list->size());
    // Remove refs to all errors in error_list.
    for (size_t i = 0; i < error_list->size(); i++) {
      GRPC_ERROR_UNREF((*error_list)[i]);
    }
    error_list->clear();
  }
  return error;
}

grpc_error* grpc_error_set_int(grpc_error* src, grpc_error_ints which,
                               intptr_t value) GRPC_MUST_USE_RESULT;
/// It is an error to pass nullptr as `p`. Caller should allocate a dummy
/// intptr_t for `p`, even if the value of `p` is not used.
bool grpc_error_get_int(grpc_error* error, grpc_error_ints which, intptr_t* p);
/// This call takes ownership of the slice; the error is responsible for
/// eventually unref-ing it.
grpc_error* grpc_error_set_str(grpc_error* src, grpc_error_strs which,
                               const grpc_slice& str) GRPC_MUST_USE_RESULT;
/// Returns false if the specified string is not set.
/// Caller does NOT own the slice.
bool grpc_error_get_str(grpc_error* error, grpc_error_strs which,
                        grpc_slice* s);

/// Add a child error: an error that is believed to have contributed to this
/// error occurring. Allows root causing high level errors from lower level
/// errors that contributed to them. The src error takes ownership of the
/// child error.
///
/// Edge Conditions -
/// 1) If either of \a src or \a child is GRPC_ERROR_NONE, returns a reference
/// to the other argument. 2) If both \a src and \a child are GRPC_ERROR_NONE,
/// returns GRPC_ERROR_NONE. 3) If \a src and \a child point to the same error,
/// returns a single reference. (Note that, 2 references should have been
/// received to the error in this case.)
grpc_error* grpc_error_add_child(grpc_error* src,
                                 grpc_error* child) GRPC_MUST_USE_RESULT;

grpc_error* grpc_os_error(const char* file, int line, int err,
                          const char* call_name) GRPC_MUST_USE_RESULT;

inline grpc_error* grpc_assert_never_ok(grpc_error* error) {
  GPR_ASSERT(error != GRPC_ERROR_NONE);
  return error;
}

/// create an error associated with errno!=0 (an 'operating system' error)
#define GRPC_OS_ERROR(err, call_name) \
  grpc_assert_never_ok(grpc_os_error(__FILE__, __LINE__, err, call_name))
grpc_error* grpc_wsa_error(const char* file, int line, int err,
                           const char* call_name) GRPC_MUST_USE_RESULT;
/// windows only: create an error associated with WSAGetLastError()!=0
#define GRPC_WSA_ERROR(err, call_name) \
  grpc_wsa_error(__FILE__, __LINE__, err, call_name)

bool grpc_log_error(const char* what, grpc_error* error, const char* file,
                    int line);
inline bool grpc_log_if_error(const char* what, grpc_error* error,
                              const char* file, int line) {
  return error == GRPC_ERROR_NONE ? true
                                  : grpc_log_error(what, error, file, line);
}

#define GRPC_LOG_IF_ERROR(what, error) \
  (grpc_log_if_error((what), (error), __FILE__, __LINE__))

#endif /* GRPC_CORE_LIB_IOMGR_ERROR_H */
