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

#ifndef GRPC_INTERNAL_CPP_COMMON_SECURE_AUTH_CONTEXT_H
#define GRPC_INTERNAL_CPP_COMMON_SECURE_AUTH_CONTEXT_H

#include <grpcpp/security/auth_context.h>

#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/security/context/security_context.h"

namespace grpc {

class SecureAuthContext final : public AuthContext {
 public:
  explicit SecureAuthContext(grpc_auth_context* ctx)
      : ctx_(ctx != nullptr ? ctx->Ref() : nullptr) {}

  ~SecureAuthContext() override = default;

  bool IsPeerAuthenticated() const override;

  std::vector<grpc::string_ref> GetPeerIdentity() const override;

  grpc::string GetPeerIdentityPropertyName() const override;

  std::vector<grpc::string_ref> FindPropertyValues(
      const grpc::string& name) const override;

  AuthPropertyIterator begin() const override;

  AuthPropertyIterator end() const override;

  void AddProperty(const grpc::string& key,
                   const grpc::string_ref& value) override;

  virtual bool SetPeerIdentityPropertyName(const grpc::string& name) override;

 private:
  grpc_core::RefCountedPtr<grpc_auth_context> ctx_;
};

}  // namespace grpc

#endif  // GRPC_INTERNAL_CPP_COMMON_SECURE_AUTH_CONTEXT_H
