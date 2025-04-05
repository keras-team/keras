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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_LOCAL_LOCAL_CREDENTIALS_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_LOCAL_LOCAL_CREDENTIALS_H

#include <grpc/support/port_platform.h>

#include <grpc/grpc_security.h>

#include "src/core/lib/security/credentials/credentials.h"

/* Main class for grpc local channel credential. */
class grpc_local_credentials final : public grpc_channel_credentials {
 public:
  explicit grpc_local_credentials(grpc_local_connect_type connect_type);
  ~grpc_local_credentials() override = default;

  grpc_core::RefCountedPtr<grpc_channel_security_connector>
  create_security_connector(
      grpc_core::RefCountedPtr<grpc_call_credentials> request_metadata_creds,
      const char* target_name, const grpc_channel_args* args,
      grpc_channel_args** new_args) override;

  grpc_local_connect_type connect_type() const { return connect_type_; }

 private:
  grpc_local_connect_type connect_type_;
};

/* Main class for grpc local server credential. */
class grpc_local_server_credentials final : public grpc_server_credentials {
 public:
  explicit grpc_local_server_credentials(grpc_local_connect_type connect_type);
  ~grpc_local_server_credentials() override = default;

  grpc_core::RefCountedPtr<grpc_server_security_connector>
  create_security_connector() override;

  grpc_local_connect_type connect_type() const { return connect_type_; }

 private:
  grpc_local_connect_type connect_type_;
};

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_LOCAL_LOCAL_CREDENTIALS_H */
