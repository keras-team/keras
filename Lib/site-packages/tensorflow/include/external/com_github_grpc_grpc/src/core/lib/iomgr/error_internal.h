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

#ifndef GRPC_CORE_LIB_IOMGR_ERROR_INTERNAL_H
#define GRPC_CORE_LIB_IOMGR_ERROR_INTERNAL_H

#include <grpc/support/port_platform.h>

#include <inttypes.h>
#include <stdbool.h>  // TODO, do we need this?

#include <grpc/support/sync.h>
#include "src/core/lib/iomgr/error.h"

typedef struct grpc_linked_error grpc_linked_error;

struct grpc_linked_error {
  grpc_error* err;
  uint8_t next;
};

// c core representation of an error. See error.h for high level description of
// this object.
struct grpc_error {
  // All atomics in grpc_error must be stored in this nested struct. The rest of
  // the object is memcpy-ed in bulk in copy_and_unref.
  struct atomics {
    gpr_refcount refs;
    gpr_atm error_string;
  } atomics;
  // These arrays index into dynamic arena at the bottom of the struct.
  // UINT8_MAX is used as a sentinel value.
  uint8_t ints[GRPC_ERROR_INT_MAX];
  uint8_t strs[GRPC_ERROR_STR_MAX];
  uint8_t times[GRPC_ERROR_TIME_MAX];
  // The child errors are stored in the arena, but are effectively a linked list
  // structure, since they are contained within grpc_linked_error objects.
  uint8_t first_err;
  uint8_t last_err;
  // The arena is dynamically reallocated with a grow factor of 1.5.
  uint8_t arena_size;
  uint8_t arena_capacity;
  intptr_t arena[0];
};

#endif /* GRPC_CORE_LIB_IOMGR_ERROR_INTERNAL_H */
