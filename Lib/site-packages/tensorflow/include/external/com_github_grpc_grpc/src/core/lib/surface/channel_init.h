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

#ifndef GRPC_CORE_LIB_SURFACE_CHANNEL_INIT_H
#define GRPC_CORE_LIB_SURFACE_CHANNEL_INIT_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/channel/channel_stack_builder.h"
#include "src/core/lib/surface/channel_stack_type.h"
#include "src/core/lib/transport/transport.h"

#define GRPC_CHANNEL_INIT_BUILTIN_PRIORITY 10000

/// This module provides a way for plugins (and the grpc core library itself)
/// to register mutators for channel stacks.
/// It also provides a universal entry path to run those mutators to build
/// a channel stack for various subsystems.

/// One stage of mutation: call functions against \a builder to influence the
/// finally constructed channel stack
typedef bool (*grpc_channel_init_stage)(grpc_channel_stack_builder* builder,
                                        void* arg);

/// Global initialization of the system
void grpc_channel_init_init(void);

/// Register one stage of mutators.
/// Stages are run in priority order (lowest to highest), and then in
/// registration order (in the case of a tie).
/// Stages are registered against one of the pre-determined channel stack
/// types.
/// If the channel stack type is GRPC_CLIENT_SUBCHANNEL, the caller should
/// ensure that subchannels with different filter lists will always have
/// different channel args. This requires setting a channel arg in case the
/// registration function relies on some condition other than channel args to
/// decide whether to add a filter or not.
void grpc_channel_init_register_stage(grpc_channel_stack_type type,
                                      int priority,
                                      grpc_channel_init_stage stage_fn,
                                      void* stage_arg);

/// Finalize registration. No more calls to grpc_channel_init_register_stage are
/// allowed.
void grpc_channel_init_finalize(void);
/// Shutdown the channel init system
void grpc_channel_init_shutdown(void);

/// Construct a channel stack of some sort: see channel_stack.h for details
/// \a type is the type of channel stack to create
/// \a prefix_bytes is the number of bytes before the channel stack to allocate
/// \a args are configuration arguments for the channel stack
/// \a initial_refs is the initial refcount to give the channel stack
/// \a destroy and \a destroy_arg specify how to destroy the channel stack
///    if destroy_arg is NULL, the returned value from this function will be
///    substituted
/// \a optional_transport is either NULL or a constructed transport object
/// Returns a pointer to the base of the memory allocated (the actual channel
/// stack object will be prefix_bytes past that pointer)
bool grpc_channel_init_create_stack(grpc_channel_stack_builder* builder,
                                    grpc_channel_stack_type type);

#endif /* GRPC_CORE_LIB_SURFACE_CHANNEL_INIT_H */
