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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_GOOGLE_DEFAULT_GOOGLE_DEFAULT_CREDENTIALS_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_GOOGLE_DEFAULT_GOOGLE_DEFAULT_CREDENTIALS_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/security/credentials/credentials.h"

#define GRPC_GOOGLE_CLOUD_SDK_CONFIG_DIRECTORY "gcloud"
#define GRPC_GOOGLE_WELL_KNOWN_CREDENTIALS_FILE \
  "application_default_credentials.json"

#ifdef GPR_WINDOWS
#define GRPC_GOOGLE_CREDENTIALS_PATH_ENV_VAR "APPDATA"
#define GRPC_GOOGLE_CREDENTIALS_PATH_SUFFIX \
  GRPC_GOOGLE_CLOUD_SDK_CONFIG_DIRECTORY    \
  "/" GRPC_GOOGLE_WELL_KNOWN_CREDENTIALS_FILE
#else
#define GRPC_GOOGLE_CREDENTIALS_PATH_ENV_VAR "HOME"
#define GRPC_GOOGLE_CREDENTIALS_PATH_SUFFIX         \
  ".config/" GRPC_GOOGLE_CLOUD_SDK_CONFIG_DIRECTORY \
  "/" GRPC_GOOGLE_WELL_KNOWN_CREDENTIALS_FILE
#endif

class grpc_google_default_channel_credentials
    : public grpc_channel_credentials {
 public:
  grpc_google_default_channel_credentials(
      grpc_core::RefCountedPtr<grpc_channel_credentials> alts_creds,
      grpc_core::RefCountedPtr<grpc_channel_credentials> ssl_creds)
      : grpc_channel_credentials(GRPC_CHANNEL_CREDENTIALS_TYPE_GOOGLE_DEFAULT),
        alts_creds_(std::move(alts_creds)),
        ssl_creds_(std::move(ssl_creds)) {}

  ~grpc_google_default_channel_credentials() override = default;

  grpc_core::RefCountedPtr<grpc_channel_security_connector>
  create_security_connector(
      grpc_core::RefCountedPtr<grpc_call_credentials> call_creds,
      const char* target, const grpc_channel_args* args,
      grpc_channel_args** new_args) override;

  grpc_channel_args* update_arguments(grpc_channel_args* args) override;

  const grpc_channel_credentials* alts_creds() const {
    return alts_creds_.get();
  }
  const grpc_channel_credentials* ssl_creds() const { return ssl_creds_.get(); }

 private:
  grpc_core::RefCountedPtr<grpc_channel_credentials> alts_creds_;
  grpc_core::RefCountedPtr<grpc_channel_credentials> ssl_creds_;
};

namespace grpc_core {
namespace internal {

typedef bool (*grpc_gce_tenancy_checker)(void);

void set_gce_tenancy_checker_for_testing(grpc_gce_tenancy_checker checker);

// TEST-ONLY. Reset the internal global state.
void grpc_flush_cached_google_default_credentials(void);

}  // namespace internal
}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_GOOGLE_DEFAULT_GOOGLE_DEFAULT_CREDENTIALS_H \
        */
