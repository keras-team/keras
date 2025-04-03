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

#ifndef GRPC_CORE_LIB_SURFACE_CHANNEL_H
#define GRPC_CORE_LIB_SURFACE_CHANNEL_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/channel/channel_stack.h"
#include "src/core/lib/channel/channel_stack_builder.h"
#include "src/core/lib/channel/channelz.h"
#include "src/core/lib/surface/channel_stack_type.h"

grpc_channel* grpc_channel_create(const char* target,
                                  const grpc_channel_args* args,
                                  grpc_channel_stack_type channel_stack_type,
                                  grpc_transport* optional_transport,
                                  grpc_resource_user* resource_user = nullptr);

/** The same as grpc_channel_destroy, but doesn't create an ExecCtx, and so
 * is safe to use from within core. */
void grpc_channel_destroy_internal(grpc_channel* channel);

grpc_channel* grpc_channel_create_with_builder(
    grpc_channel_stack_builder* builder,
    grpc_channel_stack_type channel_stack_type);

/** Create a call given a grpc_channel, in order to call \a method.
    Progress is tied to activity on \a pollset_set. The returned call object is
    meant to be used with \a grpc_call_start_batch_and_execute, which relies on
    callbacks to signal completions. \a method and \a host need
    only live through the invocation of this function. If \a parent_call is
    non-NULL, it must be a server-side call. It will be used to propagate
    properties from the server call to this new client call, depending on the
    value of \a propagation_mask (see propagation_bits.h for possible values) */
grpc_call* grpc_channel_create_pollset_set_call(
    grpc_channel* channel, grpc_call* parent_call, uint32_t propagation_mask,
    grpc_pollset_set* pollset_set, const grpc_slice& method,
    const grpc_slice* host, grpc_millis deadline, void* reserved);

/** Get a (borrowed) pointer to this channels underlying channel stack */
grpc_channel_stack* grpc_channel_get_channel_stack(grpc_channel* channel);

grpc_core::channelz::ChannelNode* grpc_channel_get_channelz_node(
    grpc_channel* channel);

size_t grpc_channel_get_call_size_estimate(grpc_channel* channel);
void grpc_channel_update_call_size_estimate(grpc_channel* channel, size_t size);

struct registered_call;
struct grpc_channel {
  int is_client;
  grpc_compression_options compression_options;

  gpr_atm call_size_estimate;
  grpc_resource_user* resource_user;

  gpr_mu registered_call_mu;
  registered_call* registered_calls;

  grpc_core::RefCountedPtr<grpc_core::channelz::ChannelNode> channelz_node;

  char* target;
};
#define CHANNEL_STACK_FROM_CHANNEL(c) ((grpc_channel_stack*)((c) + 1))

inline grpc_compression_options grpc_channel_compression_options(
    const grpc_channel* channel) {
  return channel->compression_options;
}

inline grpc_channel_stack* grpc_channel_get_channel_stack(
    grpc_channel* channel) {
  return CHANNEL_STACK_FROM_CHANNEL(channel);
}

inline grpc_core::channelz::ChannelNode* grpc_channel_get_channelz_node(
    grpc_channel* channel) {
  return channel->channelz_node.get();
}

#ifndef NDEBUG
inline void grpc_channel_internal_ref(grpc_channel* channel,
                                      const char* reason) {
  GRPC_CHANNEL_STACK_REF(CHANNEL_STACK_FROM_CHANNEL(channel), reason);
}
inline void grpc_channel_internal_unref(grpc_channel* channel,
                                        const char* reason) {
  GRPC_CHANNEL_STACK_UNREF(CHANNEL_STACK_FROM_CHANNEL(channel), reason);
}
#define GRPC_CHANNEL_INTERNAL_REF(channel, reason) \
  grpc_channel_internal_ref(channel, reason)
#define GRPC_CHANNEL_INTERNAL_UNREF(channel, reason) \
  grpc_channel_internal_unref(channel, reason)
#else
inline void grpc_channel_internal_ref(grpc_channel* channel) {
  GRPC_CHANNEL_STACK_REF(CHANNEL_STACK_FROM_CHANNEL(channel), "unused");
}
inline void grpc_channel_internal_unref(grpc_channel* channel) {
  GRPC_CHANNEL_STACK_UNREF(CHANNEL_STACK_FROM_CHANNEL(channel), "unused");
}
#define GRPC_CHANNEL_INTERNAL_REF(channel, reason) \
  grpc_channel_internal_ref(channel)
#define GRPC_CHANNEL_INTERNAL_UNREF(channel, reason) \
  grpc_channel_internal_unref(channel)
#endif

/** Return the channel's compression options. */
grpc_compression_options grpc_channel_compression_options(
    const grpc_channel* channel);

#endif /* GRPC_CORE_LIB_SURFACE_CHANNEL_H */
