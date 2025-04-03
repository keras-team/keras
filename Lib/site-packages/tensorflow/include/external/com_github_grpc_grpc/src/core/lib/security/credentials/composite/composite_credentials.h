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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_COMPOSITE_COMPOSITE_CREDENTIALS_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_COMPOSITE_COMPOSITE_CREDENTIALS_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/inlined_vector.h"
#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/security/credentials/credentials.h"

/* -- Composite channel credentials. -- */

class grpc_composite_channel_credentials : public grpc_channel_credentials {
 public:
  grpc_composite_channel_credentials(
      grpc_core::RefCountedPtr<grpc_channel_credentials> channel_creds,
      grpc_core::RefCountedPtr<grpc_call_credentials> call_creds)
      : grpc_channel_credentials(channel_creds->type()),
        inner_creds_(std::move(channel_creds)),
        call_creds_(std::move(call_creds)) {}

  ~grpc_composite_channel_credentials() override = default;

  grpc_core::RefCountedPtr<grpc_channel_credentials>
  duplicate_without_call_credentials() override {
    return inner_creds_;
  }

  grpc_core::RefCountedPtr<grpc_channel_security_connector>
  create_security_connector(
      grpc_core::RefCountedPtr<grpc_call_credentials> call_creds,
      const char* target, const grpc_channel_args* args,
      grpc_channel_args** new_args) override;

  grpc_channel_args* update_arguments(grpc_channel_args* args) override {
    return inner_creds_->update_arguments(args);
  }

  const grpc_channel_credentials* inner_creds() const {
    return inner_creds_.get();
  }
  const grpc_call_credentials* call_creds() const { return call_creds_.get(); }
  grpc_call_credentials* mutable_call_creds() { return call_creds_.get(); }

 private:
  grpc_core::RefCountedPtr<grpc_channel_credentials> inner_creds_;
  grpc_core::RefCountedPtr<grpc_call_credentials> call_creds_;
};

/* -- Composite call credentials. -- */

class grpc_composite_call_credentials : public grpc_call_credentials {
 public:
  using CallCredentialsList =
      grpc_core::InlinedVector<grpc_core::RefCountedPtr<grpc_call_credentials>,
                               2>;

  grpc_composite_call_credentials(
      grpc_core::RefCountedPtr<grpc_call_credentials> creds1,
      grpc_core::RefCountedPtr<grpc_call_credentials> creds2);
  ~grpc_composite_call_credentials() override = default;

  bool get_request_metadata(grpc_polling_entity* pollent,
                            grpc_auth_metadata_context context,
                            grpc_credentials_mdelem_array* md_array,
                            grpc_closure* on_request_metadata,
                            grpc_error** error) override;

  void cancel_get_request_metadata(grpc_credentials_mdelem_array* md_array,
                                   grpc_error* error) override;

  grpc_security_level min_security_level() const override {
    return min_security_level_;
  }

  const CallCredentialsList& inner() const { return inner_; }

 private:
  void push_to_inner(grpc_core::RefCountedPtr<grpc_call_credentials> creds,
                     bool is_composite);
  grpc_security_level min_security_level_;
  CallCredentialsList inner_;
};

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_COMPOSITE_COMPOSITE_CREDENTIALS_H \
        */
