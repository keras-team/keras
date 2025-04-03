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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_PLUGIN_PLUGIN_CREDENTIALS_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_PLUGIN_PLUGIN_CREDENTIALS_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/security/credentials/credentials.h"

extern grpc_core::TraceFlag grpc_plugin_credentials_trace;

// This type is forward declared as a C struct and we cannot define it as a
// class. Otherwise, compiler will complain about type mismatch due to
// -Wmismatched-tags.
struct grpc_plugin_credentials final : public grpc_call_credentials {
 public:
  struct pending_request {
    bool cancelled;
    struct grpc_plugin_credentials* creds;
    grpc_credentials_mdelem_array* md_array;
    grpc_closure* on_request_metadata;
    struct pending_request* prev;
    struct pending_request* next;
  };

  explicit grpc_plugin_credentials(grpc_metadata_credentials_plugin plugin,
                                   grpc_security_level min_security_level);
  ~grpc_plugin_credentials() override;

  bool get_request_metadata(grpc_polling_entity* pollent,
                            grpc_auth_metadata_context context,
                            grpc_credentials_mdelem_array* md_array,
                            grpc_closure* on_request_metadata,
                            grpc_error** error) override;

  void cancel_get_request_metadata(grpc_credentials_mdelem_array* md_array,
                                   grpc_error* error) override;

  // Checks if the request has been cancelled.
  // If not, removes it from the pending list, so that it cannot be
  // cancelled out from under us.
  // When this returns, r->cancelled indicates whether the request was
  // cancelled before completion.
  void pending_request_complete(pending_request* r);

 private:
  void pending_request_remove_locked(pending_request* pending_request);

  grpc_metadata_credentials_plugin plugin_;
  gpr_mu mu_;
  pending_request* pending_requests_ = nullptr;
};

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_PLUGIN_PLUGIN_CREDENTIALS_H */
