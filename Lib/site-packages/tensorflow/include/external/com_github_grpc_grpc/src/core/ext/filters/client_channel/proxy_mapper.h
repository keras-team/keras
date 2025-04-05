/*
 *
 * Copyright 2017 gRPC authors.
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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_PROXY_MAPPER_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_PROXY_MAPPER_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/grpc_types.h>

#include "src/core/lib/iomgr/resolve_address.h"

namespace grpc_core {

class ProxyMapperInterface {
 public:
  virtual ~ProxyMapperInterface() = default;

  /// Determines the proxy name to resolve for \a server_uri.
  /// If no proxy is needed, returns false.
  /// Otherwise, sets \a name_to_resolve, optionally sets \a new_args,
  /// and returns true.
  virtual bool MapName(const char* server_uri, const grpc_channel_args* args,
                       char** name_to_resolve,
                       grpc_channel_args** new_args) = 0;

  /// Determines the proxy address to use to contact \a address.
  /// If no proxy is needed, returns false.
  /// Otherwise, sets \a new_address, optionally sets \a new_args, and
  /// returns true.
  virtual bool MapAddress(const grpc_resolved_address& address,
                          const grpc_channel_args* args,
                          grpc_resolved_address** new_address,
                          grpc_channel_args** new_args) = 0;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_PROXY_MAPPER_H */
