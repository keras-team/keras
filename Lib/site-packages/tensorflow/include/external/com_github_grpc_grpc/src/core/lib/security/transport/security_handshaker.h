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

#ifndef GRPC_CORE_LIB_SECURITY_TRANSPORT_SECURITY_HANDSHAKER_H
#define GRPC_CORE_LIB_SECURITY_TRANSPORT_SECURITY_HANDSHAKER_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/channel/handshaker.h"
#include "src/core/lib/security/security_connector/security_connector.h"

namespace grpc_core {

/// Creates a security handshaker using \a handshaker.
RefCountedPtr<Handshaker> SecurityHandshakerCreate(
    tsi_handshaker* handshaker, grpc_security_connector* connector,
    const grpc_channel_args* args);

/// Registers security handshaker factories.
void SecurityRegisterHandshakerFactories();

}  // namespace grpc_core

// TODO(arjunroy): This is transitional to account for the new handshaker API
// and will eventually be removed entirely.
grpc_handshaker* grpc_security_handshaker_create(
    tsi_handshaker* handshaker, grpc_security_connector* connector,
    const grpc_channel_args* args);

#endif /* GRPC_CORE_LIB_SECURITY_TRANSPORT_SECURITY_HANDSHAKER_H */
