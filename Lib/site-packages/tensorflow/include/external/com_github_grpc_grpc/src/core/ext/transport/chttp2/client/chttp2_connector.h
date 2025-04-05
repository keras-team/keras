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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_CLIENT_CHTTP2_CONNECTOR_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_CLIENT_CHTTP2_CONNECTOR_H

#include <grpc/support/port_platform.h>

#include "src/core/ext/filters/client_channel/connector.h"
#include "src/core/lib/channel/handshaker.h"
#include "src/core/lib/channel/handshaker_registry.h"

namespace grpc_core {

class Chttp2Connector : public SubchannelConnector {
 public:
  Chttp2Connector();
  ~Chttp2Connector();

  void Connect(const Args& args, Result* result, grpc_closure* notify) override;
  void Shutdown(grpc_error* error) override;

 private:
  static void Connected(void* arg, grpc_error* error);
  void StartHandshakeLocked();
  static void OnHandshakeDone(void* arg, grpc_error* error);

  Mutex mu_;
  Args args_;
  Result* result_ = nullptr;
  grpc_closure* notify_ = nullptr;
  bool shutdown_ = false;
  bool connecting_ = false;
  // Holds the endpoint when first created before being handed off to
  // the handshake manager.
  grpc_endpoint* endpoint_ = nullptr;
  grpc_closure connected_;
  RefCountedPtr<HandshakeManager> handshake_mgr_;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_CLIENT_CHTTP2_CONNECTOR_H */
