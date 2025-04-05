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

#ifndef GRPC_CORE_LIB_CHANNEL_HANDSHAKER_REGISTRY_H
#define GRPC_CORE_LIB_CHANNEL_HANDSHAKER_REGISTRY_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/grpc_types.h>

#include "src/core/lib/channel/handshaker_factory.h"

namespace grpc_core {

typedef enum {
  HANDSHAKER_CLIENT = 0,
  HANDSHAKER_SERVER,
  NUM_HANDSHAKER_TYPES,  // Must be last.
} HandshakerType;

class HandshakerRegistry {
 public:
  /// Registers a new handshaker factory.  Takes ownership.
  /// If \a at_start is true, the new handshaker will be at the beginning of
  /// the list.  Otherwise, it will be added to the end.
  static void RegisterHandshakerFactory(
      bool at_start, HandshakerType handshaker_type,
      std::unique_ptr<HandshakerFactory> factory);
  static void AddHandshakers(HandshakerType handshaker_type,
                             const grpc_channel_args* args,
                             grpc_pollset_set* interested_parties,
                             HandshakeManager* handshake_mgr);
  static void Init();
  static void Shutdown();
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_CHANNEL_HANDSHAKER_REGISTRY_H */
