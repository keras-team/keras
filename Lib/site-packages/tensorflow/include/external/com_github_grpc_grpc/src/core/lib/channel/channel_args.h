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

#ifndef GRPC_CORE_LIB_CHANNEL_CHANNEL_ARGS_H
#define GRPC_CORE_LIB_CHANNEL_CHANNEL_ARGS_H

#include <grpc/support/port_platform.h>

#include <grpc/grpc.h>

#include "src/core/lib/surface/channel_stack_type.h"

// Channel args are intentionally immutable, to avoid the need for locking.

/** Copy the arguments in \a src into a new instance */
grpc_channel_args* grpc_channel_args_copy(const grpc_channel_args* src);

/** Copy the arguments in \a src into a new instance, stably sorting keys */
grpc_channel_args* grpc_channel_args_normalize(const grpc_channel_args* src);

/** Copy the arguments in \a src and append \a to_add. If \a to_add is NULL, it
 * is equivalent to calling \a grpc_channel_args_copy. */
grpc_channel_args* grpc_channel_args_copy_and_add(const grpc_channel_args* src,
                                                  const grpc_arg* to_add,
                                                  size_t num_to_add);

/** Copies the arguments in \a src except for those whose keys are in
    \a to_remove. */
grpc_channel_args* grpc_channel_args_copy_and_remove(
    const grpc_channel_args* src, const char** to_remove, size_t num_to_remove);

/** Copies the arguments from \a src except for those whose keys are in
    \a to_remove and appends the arguments in \a to_add. */
grpc_channel_args* grpc_channel_args_copy_and_add_and_remove(
    const grpc_channel_args* src, const char** to_remove, size_t num_to_remove,
    const grpc_arg* to_add, size_t num_to_add);

/** Perform the union of \a a and \a b, prioritizing \a a entries */
grpc_channel_args* grpc_channel_args_union(const grpc_channel_args* a,
                                           const grpc_channel_args* b);

/** Destroy arguments created by \a grpc_channel_args_copy */
void grpc_channel_args_destroy(grpc_channel_args* a);
inline void grpc_channel_args_destroy(const grpc_channel_args* a) {
  grpc_channel_args_destroy(const_cast<grpc_channel_args*>(a));
}

int grpc_channel_args_compare(const grpc_channel_args* a,
                              const grpc_channel_args* b);

/** Returns the value of argument \a name from \a args, or NULL if not found. */
const grpc_arg* grpc_channel_args_find(const grpc_channel_args* args,
                                       const char* name);

bool grpc_channel_args_want_minimal_stack(const grpc_channel_args* args);

typedef struct grpc_integer_options {
  int default_value;  // Return this if value is outside of expected bounds.
  int min_value;
  int max_value;
} grpc_integer_options;

/** Returns the value of \a arg, subject to the constraints in \a options. */
int grpc_channel_arg_get_integer(const grpc_arg* arg,
                                 const grpc_integer_options options);
/** Similar to the above, but needs to find the arg from \a args by the name
 * first. */
int grpc_channel_args_find_integer(const grpc_channel_args* args,
                                   const char* name,
                                   const grpc_integer_options options);

/** Returns the value of \a arg if \a arg is of type GRPC_ARG_STRING.
    Otherwise, emits a warning log, and returns nullptr.
    If arg is nullptr, returns nullptr, and does not emit a warning. */
char* grpc_channel_arg_get_string(const grpc_arg* arg);
/** Similar to the above, but needs to find the arg from \a args by the name
 * first. */
char* grpc_channel_args_find_string(const grpc_channel_args* args,
                                    const char* name);
/** If \a arg is of type GRPC_ARG_INTEGER, returns true if it's non-zero.
 * Returns \a default_value if \a arg is of other types. */
bool grpc_channel_arg_get_bool(const grpc_arg* arg, bool default_value);
/** Similar to the above, but needs to find the arg from \a args by the name
 * first. */
bool grpc_channel_args_find_bool(const grpc_channel_args* args,
                                 const char* name, bool default_value);

template <typename T>
T* grpc_channel_args_find_pointer(const grpc_channel_args* args,
                                  const char* name) {
  const grpc_arg* arg = grpc_channel_args_find(args, name);
  if (arg == nullptr || arg->type != GRPC_ARG_POINTER) return nullptr;
  return static_cast<T*>(arg->value.pointer.p);
}

// Helpers for creating channel args.
grpc_arg grpc_channel_arg_string_create(char* name, char* value);
grpc_arg grpc_channel_arg_integer_create(char* name, int value);
grpc_arg grpc_channel_arg_pointer_create(char* name, void* value,
                                         const grpc_arg_pointer_vtable* vtable);

// Returns a string representing channel args in human-readable form.
// Callers takes ownership of result.
char* grpc_channel_args_string(const grpc_channel_args* args);

// Takes ownership of the old_args
typedef grpc_channel_args* (*grpc_channel_args_client_channel_creation_mutator)(
    const char* target, grpc_channel_args* old_args,
    grpc_channel_stack_type type);

// Should be called only once globaly before grpc is init'ed.
void grpc_channel_args_set_client_channel_creation_mutator(
    grpc_channel_args_client_channel_creation_mutator cb);
// This will be called at the creation of each channel.
grpc_channel_args_client_channel_creation_mutator
grpc_channel_args_get_client_channel_creation_mutator();

#endif /* GRPC_CORE_LIB_CHANNEL_CHANNEL_ARGS_H */
