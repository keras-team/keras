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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_CONNECTOR_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_CONNECTOR_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/channel/channel_stack.h"
#include "src/core/lib/channel/channelz.h"
#include "src/core/lib/gprpp/orphanable.h"
#include "src/core/lib/iomgr/resolve_address.h"
#include "src/core/lib/transport/transport.h"

namespace grpc_core {

// Interface for connection-establishment functionality.
// Each transport that supports client channels (e.g., not inproc) must
// supply an implementation of this.
class SubchannelConnector : public InternallyRefCounted<SubchannelConnector> {
 public:
  struct Args {
    // Set of pollsets interested in this connection.
    grpc_pollset_set* interested_parties;
    // Deadline for connection.
    grpc_millis deadline;
    // Channel args to be passed to handshakers and transport.
    const grpc_channel_args* channel_args;
  };

  struct Result {
    // The connected transport.
    grpc_transport* transport = nullptr;
    // Channel args to be passed to filters.
    const grpc_channel_args* channel_args = nullptr;
    // Channelz socket node of the connected transport, if any.
    RefCountedPtr<channelz::SocketNode> socket_node;

    void Reset() {
      transport = nullptr;
      channel_args = nullptr;
      socket_node.reset();
    }
  };

  // Attempts to connect.
  // When complete, populates *result and invokes notify.
  // Only one connection attempt may be in progress at any one time.
  virtual void Connect(const Args& args, Result* result,
                       grpc_closure* notify) = 0;

  // Cancels any in-flight connection attempt and shuts down the
  // connector.
  virtual void Shutdown(grpc_error* error) = 0;

  void Orphan() override {
    Shutdown(GRPC_ERROR_CREATE_FROM_STATIC_STRING("Subchannel disconnected"));
    Unref();
  }
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_CONNECTOR_H */
