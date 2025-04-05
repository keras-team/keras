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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_OAUTH2_OAUTH2_CREDENTIALS_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_OAUTH2_OAUTH2_CREDENTIALS_H

#include <grpc/support/port_platform.h>

#include <grpc/grpc_security.h>
#include "src/core/lib/json/json.h"
#include "src/core/lib/security/credentials/credentials.h"
#include "src/core/lib/uri/uri_parser.h"

// Constants.
#define GRPC_STS_POST_MINIMAL_BODY_FORMAT_STRING                               \
  "grant_type=urn:ietf:params:oauth:grant-type:token-exchange&subject_token=%" \
  "s&subject_token_type=%s"

// auth_refresh_token parsing.
typedef struct {
  const char* type;
  char* client_id;
  char* client_secret;
  char* refresh_token;
} grpc_auth_refresh_token;

/// Returns 1 if the object is valid, 0 otherwise.
int grpc_auth_refresh_token_is_valid(
    const grpc_auth_refresh_token* refresh_token);

/// Creates a refresh token object from string. Returns an invalid object if a
/// parsing error has been encountered.
grpc_auth_refresh_token grpc_auth_refresh_token_create_from_string(
    const char* json_string);

/// Creates a refresh token object from parsed json. Returns an invalid object
/// if a parsing error has been encountered.
grpc_auth_refresh_token grpc_auth_refresh_token_create_from_json(
    const grpc_json* json);

/// Destructs the object.
void grpc_auth_refresh_token_destruct(grpc_auth_refresh_token* refresh_token);

// -- Oauth2 Token Fetcher credentials --
//
//  This object is a base for credentials that need to acquire an oauth2 token
//  from an http service.

struct grpc_oauth2_pending_get_request_metadata {
  grpc_credentials_mdelem_array* md_array;
  grpc_closure* on_request_metadata;
  grpc_polling_entity* pollent;
  struct grpc_oauth2_pending_get_request_metadata* next;
};

class grpc_oauth2_token_fetcher_credentials : public grpc_call_credentials {
 public:
  grpc_oauth2_token_fetcher_credentials();
  ~grpc_oauth2_token_fetcher_credentials() override;

  bool get_request_metadata(grpc_polling_entity* pollent,
                            grpc_auth_metadata_context context,
                            grpc_credentials_mdelem_array* md_array,
                            grpc_closure* on_request_metadata,
                            grpc_error** error) override;

  void cancel_get_request_metadata(grpc_credentials_mdelem_array* md_array,
                                   grpc_error* error) override;

  void on_http_response(grpc_credentials_metadata_request* r,
                        grpc_error* error);

 protected:
  virtual void fetch_oauth2(grpc_credentials_metadata_request* req,
                            grpc_httpcli_context* httpcli_context,
                            grpc_polling_entity* pollent, grpc_iomgr_cb_func cb,
                            grpc_millis deadline) = 0;

 private:
  gpr_mu mu_;
  grpc_mdelem access_token_md_ = GRPC_MDNULL;
  gpr_timespec token_expiration_;
  bool token_fetch_pending_ = false;
  grpc_oauth2_pending_get_request_metadata* pending_requests_ = nullptr;
  grpc_httpcli_context httpcli_context_;
  grpc_polling_entity pollent_;
};

// Google refresh token credentials.
class grpc_google_refresh_token_credentials final
    : public grpc_oauth2_token_fetcher_credentials {
 public:
  grpc_google_refresh_token_credentials(grpc_auth_refresh_token refresh_token);
  ~grpc_google_refresh_token_credentials() override;

  const grpc_auth_refresh_token& refresh_token() const {
    return refresh_token_;
  }

 protected:
  void fetch_oauth2(grpc_credentials_metadata_request* req,
                    grpc_httpcli_context* httpcli_context,
                    grpc_polling_entity* pollent, grpc_iomgr_cb_func cb,
                    grpc_millis deadline) override;

 private:
  grpc_auth_refresh_token refresh_token_;
  grpc_closure http_post_cb_closure_;
};

// Access token credentials.
class grpc_access_token_credentials final : public grpc_call_credentials {
 public:
  grpc_access_token_credentials(const char* access_token);
  ~grpc_access_token_credentials() override;

  bool get_request_metadata(grpc_polling_entity* pollent,
                            grpc_auth_metadata_context context,
                            grpc_credentials_mdelem_array* md_array,
                            grpc_closure* on_request_metadata,
                            grpc_error** error) override;

  void cancel_get_request_metadata(grpc_credentials_mdelem_array* md_array,
                                   grpc_error* error) override;

 private:
  grpc_mdelem access_token_md_;
};

// Private constructor for refresh token credentials from an already parsed
// refresh token. Takes ownership of the refresh token.
grpc_core::RefCountedPtr<grpc_call_credentials>
grpc_refresh_token_credentials_create_from_auth_refresh_token(
    grpc_auth_refresh_token token);

// Exposed for testing only.
grpc_credentials_status
grpc_oauth2_token_fetcher_credentials_parse_server_response(
    const struct grpc_http_response* response, grpc_mdelem* token_md,
    grpc_millis* token_lifetime);

namespace grpc_core {
// Exposed for testing only. This function validates the options, ensuring that
// the required fields are set, and outputs the parsed URL of the STS token
// exchanged service.
grpc_error* ValidateStsCredentialsOptions(
    const grpc_sts_credentials_options* options, grpc_uri** sts_url);
}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_OAUTH2_OAUTH2_CREDENTIALS_H */
