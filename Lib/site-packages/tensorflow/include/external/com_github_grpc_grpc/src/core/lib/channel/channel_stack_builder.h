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

#ifndef GRPC_CORE_LIB_CHANNEL_CHANNEL_STACK_BUILDER_H
#define GRPC_CORE_LIB_CHANNEL_CHANNEL_STACK_BUILDER_H

#include <grpc/support/port_platform.h>

#include <stdbool.h>

#include "src/core/lib/channel/channel_args.h"
#include "src/core/lib/channel/channel_stack.h"

/// grpc_channel_stack_builder offers a programmatic interface to selected
/// and order channel filters
typedef struct grpc_channel_stack_builder grpc_channel_stack_builder;
typedef struct grpc_channel_stack_builder_iterator
    grpc_channel_stack_builder_iterator;

/// Create a new channel stack builder
grpc_channel_stack_builder* grpc_channel_stack_builder_create(void);

/// Assign a name to the channel stack: \a name must be statically allocated
void grpc_channel_stack_builder_set_name(grpc_channel_stack_builder* builder,
                                         const char* name);

/// Set the target uri
void grpc_channel_stack_builder_set_target(grpc_channel_stack_builder* b,
                                           const char* target);

const char* grpc_channel_stack_builder_get_target(
    grpc_channel_stack_builder* b);

/// Attach \a transport to the builder (does not take ownership)
void grpc_channel_stack_builder_set_transport(
    grpc_channel_stack_builder* builder, grpc_transport* transport);

/// Fetch attached transport
grpc_transport* grpc_channel_stack_builder_get_transport(
    grpc_channel_stack_builder* builder);

/// Attach \a resource_user to the builder (does not take ownership)
void grpc_channel_stack_builder_set_resource_user(
    grpc_channel_stack_builder* builder, grpc_resource_user* resource_user);

/// Fetch attached resource user
grpc_resource_user* grpc_channel_stack_builder_get_resource_user(
    grpc_channel_stack_builder* builder);

/// Set channel arguments: copies args
void grpc_channel_stack_builder_set_channel_arguments(
    grpc_channel_stack_builder* builder, const grpc_channel_args* args);

/// Return a borrowed pointer to the channel arguments
const grpc_channel_args* grpc_channel_stack_builder_get_channel_arguments(
    grpc_channel_stack_builder* builder);

/// Begin iterating over already defined filters in the builder at the beginning
grpc_channel_stack_builder_iterator*
grpc_channel_stack_builder_create_iterator_at_first(
    grpc_channel_stack_builder* builder);

/// Begin iterating over already defined filters in the builder at the end
grpc_channel_stack_builder_iterator*
grpc_channel_stack_builder_create_iterator_at_last(
    grpc_channel_stack_builder* builder);

/// Is an iterator at the first element?
bool grpc_channel_stack_builder_iterator_is_first(
    grpc_channel_stack_builder_iterator* iterator);

/// Is an iterator at the end?
bool grpc_channel_stack_builder_iterator_is_end(
    grpc_channel_stack_builder_iterator* iterator);

/// What is the name of the filter at this iterator position?
const char* grpc_channel_stack_builder_iterator_filter_name(
    grpc_channel_stack_builder_iterator* iterator);

/// Move an iterator to the next item
bool grpc_channel_stack_builder_move_next(
    grpc_channel_stack_builder_iterator* iterator);

/// Move an iterator to the previous item
bool grpc_channel_stack_builder_move_prev(
    grpc_channel_stack_builder_iterator* iterator);

/// Return an iterator at \a filter_name, or at the end of the list if not
/// found.
grpc_channel_stack_builder_iterator* grpc_channel_stack_builder_iterator_find(
    grpc_channel_stack_builder* builder, const char* filter_name);

typedef void (*grpc_post_filter_create_init_func)(
    grpc_channel_stack* channel_stack, grpc_channel_element* elem, void* arg);

/// Add \a filter to the stack, after \a iterator.
/// Call \a post_init_func(..., \a user_data) once the channel stack is
/// created.
bool grpc_channel_stack_builder_add_filter_after(
    grpc_channel_stack_builder_iterator* iterator,
    const grpc_channel_filter* filter,
    grpc_post_filter_create_init_func post_init_func,
    void* user_data) GRPC_MUST_USE_RESULT;

/// Add \a filter to the stack, before \a iterator.
/// Call \a post_init_func(..., \a user_data) once the channel stack is
/// created.
bool grpc_channel_stack_builder_add_filter_before(
    grpc_channel_stack_builder_iterator* iterator,
    const grpc_channel_filter* filter,
    grpc_post_filter_create_init_func post_init_func,
    void* user_data) GRPC_MUST_USE_RESULT;

/// Add \a filter to the beginning of the filter list.
/// Call \a post_init_func(..., \a user_data) once the channel stack is
/// created.
bool grpc_channel_stack_builder_prepend_filter(
    grpc_channel_stack_builder* builder, const grpc_channel_filter* filter,
    grpc_post_filter_create_init_func post_init_func,
    void* user_data) GRPC_MUST_USE_RESULT;

/// Add \a filter to the end of the filter list.
/// Call \a post_init_func(..., \a user_data) once the channel stack is
/// created.
bool grpc_channel_stack_builder_append_filter(
    grpc_channel_stack_builder* builder, const grpc_channel_filter* filter,
    grpc_post_filter_create_init_func post_init_func,
    void* user_data) GRPC_MUST_USE_RESULT;

/// Remove any filter whose name is \a filter_name from \a builder. Returns true
/// if \a filter_name was not found.
bool grpc_channel_stack_builder_remove_filter(
    grpc_channel_stack_builder* builder, const char* filter_name);

/// Terminate iteration and destroy \a iterator
void grpc_channel_stack_builder_iterator_destroy(
    grpc_channel_stack_builder_iterator* iterator);

/// Destroy the builder, return the freshly minted channel stack in \a result.
/// Allocates \a prefix_bytes bytes before the channel stack
/// Returns the base pointer of the allocated block
/// \a initial_refs, \a destroy, \a destroy_arg are as per
/// grpc_channel_stack_init
grpc_error* grpc_channel_stack_builder_finish(
    grpc_channel_stack_builder* builder, size_t prefix_bytes, int initial_refs,
    grpc_iomgr_cb_func destroy, void* destroy_arg, void** result);

/// Destroy the builder without creating a channel stack
void grpc_channel_stack_builder_destroy(grpc_channel_stack_builder* builder);

#endif /* GRPC_CORE_LIB_CHANNEL_CHANNEL_STACK_BUILDER_H */
