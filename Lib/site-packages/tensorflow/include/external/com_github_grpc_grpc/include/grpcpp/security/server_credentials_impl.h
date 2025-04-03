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

#ifndef GRPCPP_SECURITY_SERVER_CREDENTIALS_IMPL_H
#define GRPCPP_SECURITY_SERVER_CREDENTIALS_IMPL_H

#include <memory>
#include <vector>

#include <grpc/grpc_security_constants.h>
#include <grpcpp/security/auth_metadata_processor.h>
#include <grpcpp/security/tls_credentials_options.h>
#include <grpcpp/support/config.h>

struct grpc_server;

namespace grpc {

struct SslServerCredentialsOptions;
}  // namespace grpc
namespace grpc_impl {
class Server;

/// Wrapper around \a grpc_server_credentials, a way to authenticate a server.
class ServerCredentials {
 public:
  virtual ~ServerCredentials();

  /// This method is not thread-safe and has to be called before the server is
  /// started. The last call to this function wins.
  virtual void SetAuthMetadataProcessor(
      const std::shared_ptr<grpc::AuthMetadataProcessor>& processor) = 0;

 private:
  friend class ::grpc_impl::Server;

  /// Tries to bind \a server to the given \a addr (eg, localhost:1234,
  /// 192.168.1.1:31416, [::1]:27182, etc.)
  ///
  /// \return bound port number on success, 0 on failure.
  // TODO(dgq): the "port" part seems to be a misnomer.
  virtual int AddPortToServer(const grpc::string& addr,
                              grpc_server* server) = 0;
};

/// Builds SSL ServerCredentials given SSL specific options
std::shared_ptr<ServerCredentials> SslServerCredentials(
    const grpc::SslServerCredentialsOptions& options);

/// Builds insecure server credentials.
std::shared_ptr<ServerCredentials> InsecureServerCredentials();

namespace experimental {

/// Options to create ServerCredentials with ALTS
struct AltsServerCredentialsOptions {
  /// Add fields if needed.
};

/// Builds ALTS ServerCredentials given ALTS specific options
std::shared_ptr<ServerCredentials> AltsServerCredentials(
    const AltsServerCredentialsOptions& options);

/// Builds Local ServerCredentials.
std::shared_ptr<ServerCredentials> LocalServerCredentials(
    grpc_local_connect_type type);

/// Builds TLS ServerCredentials given TLS options.
std::shared_ptr<ServerCredentials> TlsServerCredentials(
    const TlsCredentialsOptions& options);

}  // namespace experimental
}  // namespace grpc_impl

#endif  // GRPCPP_SECURITY_SERVER_CREDENTIALS_IMPL_H
