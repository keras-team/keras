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

#ifndef GRPC_CORE_LIB_SECURITY_CONTEXT_SECURITY_CONTEXT_H
#define GRPC_CORE_LIB_SECURITY_CONTEXT_SECURITY_CONTEXT_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/arena.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/iomgr/pollset.h"
#include "src/core/lib/security/credentials/credentials.h"

extern grpc_core::DebugOnlyTraceFlag grpc_trace_auth_context_refcount;

/* --- grpc_auth_context ---

   High level authentication context object. Can optionally be chained. */

/* Property names are always NULL terminated. */

struct grpc_auth_property_array {
  grpc_auth_property* array = nullptr;
  size_t count = 0;
  size_t capacity = 0;
};

void grpc_auth_property_reset(grpc_auth_property* property);

// This type is forward declared as a C struct and we cannot define it as a
// class. Otherwise, compiler will complain about type mismatch due to
// -Wmismatched-tags.
struct grpc_auth_context
    : public grpc_core::RefCounted<grpc_auth_context,
                                   grpc_core::NonPolymorphicRefCount> {
 public:
  explicit grpc_auth_context(
      grpc_core::RefCountedPtr<grpc_auth_context> chained)
      : grpc_core::RefCounted<grpc_auth_context,
                              grpc_core::NonPolymorphicRefCount>(
            &grpc_trace_auth_context_refcount),
        chained_(std::move(chained)) {
    if (chained_ != nullptr) {
      peer_identity_property_name_ = chained_->peer_identity_property_name_;
    }
  }

  ~grpc_auth_context() {
    chained_.reset(DEBUG_LOCATION, "chained");
    if (properties_.array != nullptr) {
      for (size_t i = 0; i < properties_.count; i++) {
        grpc_auth_property_reset(&properties_.array[i]);
      }
      gpr_free(properties_.array);
    }
  }

  const grpc_auth_context* chained() const { return chained_.get(); }
  const grpc_auth_property_array& properties() const { return properties_; }

  bool is_authenticated() const {
    return peer_identity_property_name_ != nullptr;
  }
  const char* peer_identity_property_name() const {
    return peer_identity_property_name_;
  }
  void set_peer_identity_property_name(const char* name) {
    peer_identity_property_name_ = name;
  }

  void ensure_capacity();
  void add_property(const char* name, const char* value, size_t value_length);
  void add_cstring_property(const char* name, const char* value);

 private:
  grpc_core::RefCountedPtr<grpc_auth_context> chained_;
  grpc_auth_property_array properties_;
  const char* peer_identity_property_name_ = nullptr;
};

/* --- grpc_security_context_extension ---

   Extension to the security context that may be set in a filter and accessed
   later by a higher level method on a grpc_call object. */

struct grpc_security_context_extension {
  void* instance = nullptr;
  void (*destroy)(void*) = nullptr;
};

/* --- grpc_client_security_context ---

   Internal client-side security context. */

struct grpc_client_security_context {
  explicit grpc_client_security_context(
      grpc_core::RefCountedPtr<grpc_call_credentials> creds)
      : creds(std::move(creds)) {}
  ~grpc_client_security_context();

  grpc_core::RefCountedPtr<grpc_call_credentials> creds;
  grpc_core::RefCountedPtr<grpc_auth_context> auth_context;
  grpc_security_context_extension extension;
};

grpc_client_security_context* grpc_client_security_context_create(
    grpc_core::Arena* arena, grpc_call_credentials* creds);
void grpc_client_security_context_destroy(void* ctx);

/* --- grpc_server_security_context ---

   Internal server-side security context. */

struct grpc_server_security_context {
  grpc_server_security_context() = default;
  ~grpc_server_security_context();

  grpc_core::RefCountedPtr<grpc_auth_context> auth_context;
  grpc_security_context_extension extension;
};

grpc_server_security_context* grpc_server_security_context_create(
    grpc_core::Arena* arena);
void grpc_server_security_context_destroy(void* ctx);

/* --- Channel args for auth context --- */
#define GRPC_AUTH_CONTEXT_ARG "grpc.auth_context"

grpc_arg grpc_auth_context_to_arg(grpc_auth_context* c);
grpc_auth_context* grpc_auth_context_from_arg(const grpc_arg* arg);
grpc_auth_context* grpc_find_auth_context_in_args(
    const grpc_channel_args* args);

#endif /* GRPC_CORE_LIB_SECURITY_CONTEXT_SECURITY_CONTEXT_H */
