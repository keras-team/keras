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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_CREDENTIALS_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_CREDENTIALS_H

#include <grpc/support/port_platform.h>

#include <grpc/grpc.h>
#include <grpc/grpc_security.h>
#include <grpc/support/sync.h>
#include "src/core/lib/transport/metadata_batch.h"

#include "src/core/lib/gprpp/map.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/gprpp/sync.h"
#include "src/core/lib/http/httpcli.h"
#include "src/core/lib/http/parser.h"
#include "src/core/lib/iomgr/polling_entity.h"
#include "src/core/lib/security/security_connector/security_connector.h"

struct grpc_http_response;

/* --- Constants. --- */

typedef enum {
  GRPC_CREDENTIALS_OK = 0,
  GRPC_CREDENTIALS_ERROR
} grpc_credentials_status;

#define GRPC_FAKE_TRANSPORT_SECURITY_TYPE "fake"

#define GRPC_CHANNEL_CREDENTIALS_TYPE_SSL "Ssl"
#define GRPC_CHANNEL_CREDENTIALS_TYPE_FAKE_TRANSPORT_SECURITY \
  "FakeTransportSecurity"
#define GRPC_CHANNEL_CREDENTIALS_TYPE_GOOGLE_DEFAULT "GoogleDefault"

#define GRPC_CALL_CREDENTIALS_TYPE_OAUTH2 "Oauth2"
#define GRPC_CALL_CREDENTIALS_TYPE_JWT "Jwt"
#define GRPC_CALL_CREDENTIALS_TYPE_IAM "Iam"
#define GRPC_CALL_CREDENTIALS_TYPE_COMPOSITE "Composite"

#define GRPC_AUTHORIZATION_METADATA_KEY "authorization"
#define GRPC_IAM_AUTHORIZATION_TOKEN_METADATA_KEY \
  "x-goog-iam-authorization-token"
#define GRPC_IAM_AUTHORITY_SELECTOR_METADATA_KEY "x-goog-iam-authority-selector"

#define GRPC_SECURE_TOKEN_REFRESH_THRESHOLD_SECS 60

#define GRPC_COMPUTE_ENGINE_METADATA_HOST "metadata.google.internal."
#define GRPC_COMPUTE_ENGINE_METADATA_TOKEN_PATH \
  "/computeMetadata/v1/instance/service-accounts/default/token"

#define GRPC_GOOGLE_OAUTH2_SERVICE_HOST "oauth2.googleapis.com"
#define GRPC_GOOGLE_OAUTH2_SERVICE_TOKEN_PATH "/token"

#define GRPC_SERVICE_ACCOUNT_POST_BODY_PREFIX                         \
  "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&" \
  "assertion="

#define GRPC_REFRESH_TOKEN_POST_BODY_FORMAT_STRING \
  "client_id=%s&client_secret=%s&refresh_token=%s&grant_type=refresh_token"

/* --- Google utils --- */

/* It is the caller's responsibility to gpr_free the result if not NULL. */
char* grpc_get_well_known_google_credentials_file_path(void);

/* Implementation function for the different platforms. */
char* grpc_get_well_known_google_credentials_file_path_impl(void);

/* Override for testing only. Not thread-safe */
typedef char* (*grpc_well_known_credentials_path_getter)(void);
void grpc_override_well_known_credentials_path_getter(
    grpc_well_known_credentials_path_getter getter);

/* --- grpc_channel_credentials. --- */

#define GRPC_ARG_CHANNEL_CREDENTIALS "grpc.channel_credentials"

// This type is forward declared as a C struct and we cannot define it as a
// class. Otherwise, compiler will complain about type mismatch due to
// -Wmismatched-tags.
struct grpc_channel_credentials
    : grpc_core::RefCounted<grpc_channel_credentials> {
 public:
  explicit grpc_channel_credentials(const char* type) : type_(type) {}
  virtual ~grpc_channel_credentials() = default;

  // Creates a security connector for the channel. May also create new channel
  // args for the channel to be used in place of the passed in const args if
  // returned non NULL. In that case the caller is responsible for destroying
  // new_args after channel creation.
  virtual grpc_core::RefCountedPtr<grpc_channel_security_connector>
  create_security_connector(
      grpc_core::RefCountedPtr<grpc_call_credentials> call_creds,
      const char* target, const grpc_channel_args* args,
      grpc_channel_args** new_args) = 0;

  // Creates a version of the channel credentials without any attached call
  // credentials. This can be used in order to open a channel to a non-trusted
  // gRPC load balancer.
  virtual grpc_core::RefCountedPtr<grpc_channel_credentials>
  duplicate_without_call_credentials() {
    // By default we just increment the refcount.
    return Ref();
  }

  // Allows credentials to optionally modify a parent channel's args.
  // By default, leave channel args as is. The callee takes ownership
  // of the passed-in channel args, and the caller takes ownership
  // of the returned channel args.
  virtual grpc_channel_args* update_arguments(grpc_channel_args* args) {
    return args;
  }

  // Attaches control_plane_creds to the local registry, under authority,
  // if no other creds are currently registered under authority. Returns
  // true if registered successfully and false if not.
  bool attach_credentials(
      const char* authority,
      grpc_core::RefCountedPtr<grpc_channel_credentials> control_plane_creds);

  // Gets the control plane credentials registered under authority. This
  // prefers the local control plane creds registry but falls back to the
  // global registry. Lastly, this returns self but with any attached
  // call credentials stripped off, in the case that neither the local
  // registry nor the global registry have an entry for authority.
  grpc_core::RefCountedPtr<grpc_channel_credentials>
  get_control_plane_credentials(const char* authority);

  const char* type() const { return type_; }

 private:
  const char* type_;
  std::map<grpc_core::UniquePtr<char>,
           grpc_core::RefCountedPtr<grpc_channel_credentials>,
           grpc_core::StringLess>
      local_control_plane_creds_;
};

/* Util to encapsulate the channel credentials in a channel arg. */
grpc_arg grpc_channel_credentials_to_arg(grpc_channel_credentials* credentials);

/* Util to get the channel credentials from a channel arg. */
grpc_channel_credentials* grpc_channel_credentials_from_arg(
    const grpc_arg* arg);

/* Util to find the channel credentials from channel args. */
grpc_channel_credentials* grpc_channel_credentials_find_in_args(
    const grpc_channel_args* args);

/** EXPERIMENTAL.  API MAY CHANGE IN THE FUTURE.
    Attaches \a control_plane_creds to \a credentials
    under the key \a authority. Returns false if \a authority
    is already present, in which case no changes are made.
    Note that this API is not thread safe. Only one thread may
    attach control plane creds to a given credentials object at
    any one time, and new control plane creds must not be
    attached after \a credentials has been used to create a channel. */
bool grpc_channel_credentials_attach_credentials(
    grpc_channel_credentials* credentials, const char* authority,
    grpc_channel_credentials* control_plane_creds);

/** EXPERIMENTAL.  API MAY CHANGE IN THE FUTURE.
    Registers \a control_plane_creds in the global registry
    under the key \a authority. Returns false if \a authority
    is already present, in which case no changes are made. */
bool grpc_control_plane_credentials_register(
    const char* authority, grpc_channel_credentials* control_plane_creds);

/* Initializes global control plane credentials data. */
void grpc_control_plane_credentials_init();

/* Test only: destroy global control plane credentials data.
 * This API is meant for use by a few tests that need to
 * satisdy grpc_core::LeakDetector. */
void grpc_test_only_control_plane_credentials_destroy();

/* Test only: force re-initialization of global control
 * plane credentials data if it was previously destroyed.
 * This API is meant to be used in
 * tandem with the
 * grpc_test_only_control_plane_credentials_destroy, for
 * the few tests that need it. */
void grpc_test_only_control_plane_credentials_force_init();

/* --- grpc_credentials_mdelem_array. --- */

typedef struct {
  grpc_mdelem* md = nullptr;
  size_t size = 0;
} grpc_credentials_mdelem_array;

/// Takes a new ref to \a md.
void grpc_credentials_mdelem_array_add(grpc_credentials_mdelem_array* list,
                                       grpc_mdelem md);

/// Appends all elements from \a src to \a dst, taking a new ref to each one.
void grpc_credentials_mdelem_array_append(grpc_credentials_mdelem_array* dst,
                                          grpc_credentials_mdelem_array* src);

void grpc_credentials_mdelem_array_destroy(grpc_credentials_mdelem_array* list);

/* --- grpc_call_credentials. --- */

// This type is forward declared as a C struct and we cannot define it as a
// class. Otherwise, compiler will complain about type mismatch due to
// -Wmismatched-tags.
struct grpc_call_credentials
    : public grpc_core::RefCounted<grpc_call_credentials> {
 public:
  explicit grpc_call_credentials(
      const char* type,
      grpc_security_level min_security_level = GRPC_PRIVACY_AND_INTEGRITY)
      : type_(type), min_security_level_(min_security_level) {}

  virtual ~grpc_call_credentials() = default;

  // Returns true if completed synchronously, in which case \a error will
  // be set to indicate the result.  Otherwise, \a on_request_metadata will
  // be invoked asynchronously when complete.  \a md_array will be populated
  // with the resulting metadata once complete.
  virtual bool get_request_metadata(grpc_polling_entity* pollent,
                                    grpc_auth_metadata_context context,
                                    grpc_credentials_mdelem_array* md_array,
                                    grpc_closure* on_request_metadata,
                                    grpc_error** error) = 0;

  // Cancels a pending asynchronous operation started by
  // grpc_call_credentials_get_request_metadata() with the corresponding
  // value of \a md_array.
  virtual void cancel_get_request_metadata(
      grpc_credentials_mdelem_array* md_array, grpc_error* error) = 0;

  virtual grpc_security_level min_security_level() const {
    return min_security_level_;
  }

  const char* type() const { return type_; }

 private:
  const char* type_;
  const grpc_security_level min_security_level_;
};

/* Metadata-only credentials with the specified key and value where
   asynchronicity can be simulated for testing. */
grpc_call_credentials* grpc_md_only_test_credentials_create(
    const char* md_key, const char* md_value, bool is_async);

/* --- grpc_server_credentials. --- */

// This type is forward declared as a C struct and we cannot define it as a
// class. Otherwise, compiler will complain about type mismatch due to
// -Wmismatched-tags.
struct grpc_server_credentials
    : public grpc_core::RefCounted<grpc_server_credentials> {
 public:
  explicit grpc_server_credentials(const char* type) : type_(type) {}

  virtual ~grpc_server_credentials() { DestroyProcessor(); }

  virtual grpc_core::RefCountedPtr<grpc_server_security_connector>
  create_security_connector() = 0;

  const char* type() const { return type_; }

  const grpc_auth_metadata_processor& auth_metadata_processor() const {
    return processor_;
  }
  void set_auth_metadata_processor(
      const grpc_auth_metadata_processor& processor);

 private:
  void DestroyProcessor() {
    if (processor_.destroy != nullptr && processor_.state != nullptr) {
      processor_.destroy(processor_.state);
    }
  }

  const char* type_;
  grpc_auth_metadata_processor processor_ =
      grpc_auth_metadata_processor();  // Zero-initialize the C struct.
};

#define GRPC_SERVER_CREDENTIALS_ARG "grpc.server_credentials"

grpc_arg grpc_server_credentials_to_arg(grpc_server_credentials* c);
grpc_server_credentials* grpc_server_credentials_from_arg(const grpc_arg* arg);
grpc_server_credentials* grpc_find_server_credentials_in_args(
    const grpc_channel_args* args);

/* -- Credentials Metadata Request. -- */

struct grpc_credentials_metadata_request {
  explicit grpc_credentials_metadata_request(
      grpc_core::RefCountedPtr<grpc_call_credentials> creds)
      : creds(std::move(creds)) {}
  ~grpc_credentials_metadata_request() {
    grpc_http_response_destroy(&response);
  }

  grpc_core::RefCountedPtr<grpc_call_credentials> creds;
  grpc_http_response response;
};

inline grpc_credentials_metadata_request*
grpc_credentials_metadata_request_create(
    grpc_core::RefCountedPtr<grpc_call_credentials> creds) {
  return new grpc_credentials_metadata_request(std::move(creds));
}

inline void grpc_credentials_metadata_request_destroy(
    grpc_credentials_metadata_request* r) {
  delete r;
}

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_CREDENTIALS_H */
