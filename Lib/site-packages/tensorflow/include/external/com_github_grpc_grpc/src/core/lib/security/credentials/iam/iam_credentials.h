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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_IAM_IAM_CREDENTIALS_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_IAM_IAM_CREDENTIALS_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/security/credentials/credentials.h"

class grpc_google_iam_credentials : public grpc_call_credentials {
 public:
  grpc_google_iam_credentials(const char* token,
                              const char* authority_selector);
  ~grpc_google_iam_credentials() override;

  bool get_request_metadata(grpc_polling_entity* pollent,
                            grpc_auth_metadata_context context,
                            grpc_credentials_mdelem_array* md_array,
                            grpc_closure* on_request_metadata,
                            grpc_error** error) override;

  void cancel_get_request_metadata(grpc_credentials_mdelem_array* md_array,
                                   grpc_error* error) override;

 private:
  grpc_credentials_mdelem_array md_array_;
};

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_IAM_IAM_CREDENTIALS_H */
