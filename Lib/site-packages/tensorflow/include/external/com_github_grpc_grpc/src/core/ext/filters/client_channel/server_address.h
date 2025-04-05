/*
 *
 * Copyright 2018 gRPC authors.
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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SERVER_ADDRESS_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SERVER_ADDRESS_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/channel/channel_args.h"
#include "src/core/lib/gprpp/inlined_vector.h"
#include "src/core/lib/iomgr/resolve_address.h"

// Channel arg key for a bool indicating whether an address is a grpclb
// load balancer (as opposed to a backend).
#define GRPC_ARG_ADDRESS_IS_BALANCER "grpc.address_is_balancer"

// Channel arg key for a string indicating an address's balancer name.
#define GRPC_ARG_ADDRESS_BALANCER_NAME "grpc.address_balancer_name"

namespace grpc_core {

//
// ServerAddress
//

// A server address is a grpc_resolved_address with an associated set of
// channel args.  Any args present here will be merged into the channel
// args when a subchannel is created for this address.
class ServerAddress {
 public:
  // Takes ownership of args.
  ServerAddress(const grpc_resolved_address& address, grpc_channel_args* args);
  ServerAddress(const void* address, size_t address_len,
                grpc_channel_args* args);

  ~ServerAddress() { grpc_channel_args_destroy(args_); }

  // Copyable.
  ServerAddress(const ServerAddress& other)
      : address_(other.address_), args_(grpc_channel_args_copy(other.args_)) {}
  ServerAddress& operator=(const ServerAddress& other) {
    address_ = other.address_;
    grpc_channel_args_destroy(args_);
    args_ = grpc_channel_args_copy(other.args_);
    return *this;
  }

  // Movable.
  ServerAddress(ServerAddress&& other)
      : address_(other.address_), args_(other.args_) {
    other.args_ = nullptr;
  }
  ServerAddress& operator=(ServerAddress&& other) {
    address_ = other.address_;
    grpc_channel_args_destroy(args_);
    args_ = other.args_;
    other.args_ = nullptr;
    return *this;
  }

  bool operator==(const ServerAddress& other) const;

  const grpc_resolved_address& address() const { return address_; }
  const grpc_channel_args* args() const { return args_; }

  bool IsBalancer() const;

 private:
  grpc_resolved_address address_;
  grpc_channel_args* args_;
};

//
// ServerAddressList
//

typedef InlinedVector<ServerAddress, 1> ServerAddressList;

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SERVER_ADDRESS_H */
