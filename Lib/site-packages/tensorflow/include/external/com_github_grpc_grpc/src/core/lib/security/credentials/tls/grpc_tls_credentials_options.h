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

#ifndef GRPC_CORE_LIB_SECURITY_CREDENTIALS_TLS_GRPC_TLS_CREDENTIALS_OPTIONS_H
#define GRPC_CORE_LIB_SECURITY_CREDENTIALS_TLS_GRPC_TLS_CREDENTIALS_OPTIONS_H

#include <grpc/support/port_platform.h>

#include <grpc/grpc_security.h>

#include "src/core/lib/gprpp/inlined_vector.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/security/security_connector/ssl_utils.h"

/** TLS key materials config. **/
struct grpc_tls_key_materials_config
    : public grpc_core::RefCounted<grpc_tls_key_materials_config> {
 public:
  typedef grpc_core::InlinedVector<grpc_core::PemKeyCertPair, 1>
      PemKeyCertPairList;

  /** Getters for member fields. **/
  const char* pem_root_certs() const { return pem_root_certs_.get(); }
  const PemKeyCertPairList& pem_key_cert_pair_list() const {
    return pem_key_cert_pair_list_;
  }
  int version() const { return version_; }

  /** Setters for member fields. **/
  void set_pem_root_certs(grpc_core::UniquePtr<char> pem_root_certs) {
    pem_root_certs_ = std::move(pem_root_certs);
  }
  void add_pem_key_cert_pair(grpc_core::PemKeyCertPair pem_key_cert_pair) {
    pem_key_cert_pair_list_.push_back(pem_key_cert_pair);
  }
  void set_key_materials(grpc_core::UniquePtr<char> pem_root_certs,
                         PemKeyCertPairList pem_key_cert_pair_list);
  void set_version(int version) { version_ = version; }

 private:
  int version_ = 0;
  PemKeyCertPairList pem_key_cert_pair_list_;
  grpc_core::UniquePtr<char> pem_root_certs_;
};

/** TLS credential reload config. **/
struct grpc_tls_credential_reload_config
    : public grpc_core::RefCounted<grpc_tls_credential_reload_config> {
 public:
  grpc_tls_credential_reload_config(
      const void* config_user_data,
      int (*schedule)(void* config_user_data,
                      grpc_tls_credential_reload_arg* arg),
      void (*cancel)(void* config_user_data,
                     grpc_tls_credential_reload_arg* arg),
      void (*destruct)(void* config_user_data));
  ~grpc_tls_credential_reload_config();

  void* context() const { return context_; }
  void set_context(void* context) { context_ = context; }

  int Schedule(grpc_tls_credential_reload_arg* arg) const {
    if (schedule_ == nullptr) {
      gpr_log(GPR_ERROR, "schedule API is nullptr");
      if (arg != nullptr) {
        arg->status = GRPC_SSL_CERTIFICATE_CONFIG_RELOAD_FAIL;
        arg->error_details =
            gpr_strdup("schedule API in credential reload config is nullptr");
      }
      return 1;
    }
    if (arg != nullptr) {
      arg->config = const_cast<grpc_tls_credential_reload_config*>(this);
    }
    return schedule_(config_user_data_, arg);
  }
  void Cancel(grpc_tls_credential_reload_arg* arg) const {
    if (cancel_ == nullptr) {
      gpr_log(GPR_ERROR, "cancel API is nullptr.");
      if (arg != nullptr) {
        arg->status = GRPC_SSL_CERTIFICATE_CONFIG_RELOAD_FAIL;
        arg->error_details =
            gpr_strdup("cancel API in credential reload config is nullptr");
      }
      return;
    }
    if (arg != nullptr) {
      arg->config = const_cast<grpc_tls_credential_reload_config*>(this);
    }
    cancel_(config_user_data_, arg);
  }

 private:
  /** This is a pointer to the wrapped language implementation of
   * grpc_tls_credential_reload_config. It is necessary to implement the C
   * schedule and cancel functions, given the schedule or cancel function in a
   * wrapped language. **/
  void* context_ = nullptr;
  /** config-specific, read-only user data that works for all channels created
     with a credential using the config. */
  void* config_user_data_;
  /** callback function for invoking credential reload API. The implementation
     of this method has to be non-blocking, but can be performed synchronously
     or asynchronously.
     If processing occurs synchronously, it populates \a arg->key_materials, \a
     arg->status, and \a arg->error_details and returns zero.
     If processing occurs asynchronously, it returns a non-zero value.
     Application then invokes \a arg->cb when processing is completed. Note that
     \a arg->cb cannot be invoked before \a schedule returns.
  */
  int (*schedule_)(void* config_user_data, grpc_tls_credential_reload_arg* arg);
  /** callback function for cancelling a credential reload request scheduled via
     an asynchronous \a schedule. \a arg is used to pinpoint an exact reloading
     request to be cancelled, and the operation may not have any effect if the
     request has already been processed. */
  void (*cancel_)(void* config_user_data, grpc_tls_credential_reload_arg* arg);
  /** callback function for cleaning up any data associated with credential
     reload config. */
  void (*destruct_)(void* config_user_data);
};

/** TLS server authorization check config. **/
struct grpc_tls_server_authorization_check_config
    : public grpc_core::RefCounted<grpc_tls_server_authorization_check_config> {
 public:
  grpc_tls_server_authorization_check_config(
      const void* config_user_data,
      int (*schedule)(void* config_user_data,
                      grpc_tls_server_authorization_check_arg* arg),
      void (*cancel)(void* config_user_data,
                     grpc_tls_server_authorization_check_arg* arg),
      void (*destruct)(void* config_user_data));
  ~grpc_tls_server_authorization_check_config();

  void* context() const { return context_; }
  void set_context(void* context) { context_ = context; }

  int Schedule(grpc_tls_server_authorization_check_arg* arg) const {
    if (schedule_ == nullptr) {
      gpr_log(GPR_ERROR, "schedule API is nullptr");
      if (arg != nullptr) {
        arg->status = GRPC_STATUS_NOT_FOUND;
        arg->error_details = gpr_strdup(
            "schedule API in server authorization check config is nullptr");
      }
      return 1;
    }
    if (arg != nullptr && context_ != nullptr) {
      arg->config =
          const_cast<grpc_tls_server_authorization_check_config*>(this);
    }
    return schedule_(config_user_data_, arg);
  }
  void Cancel(grpc_tls_server_authorization_check_arg* arg) const {
    if (cancel_ == nullptr) {
      gpr_log(GPR_ERROR, "cancel API is nullptr.");
      if (arg != nullptr) {
        arg->status = GRPC_STATUS_NOT_FOUND;
        arg->error_details = gpr_strdup(
            "schedule API in server authorization check config is nullptr");
      }
      return;
    }
    if (arg != nullptr) {
      arg->config =
          const_cast<grpc_tls_server_authorization_check_config*>(this);
    }
    cancel_(config_user_data_, arg);
  }

 private:
  /** This is a pointer to the wrapped language implementation of
   * grpc_tls_server_authorization_check_config. It is necessary to implement
   * the C schedule and cancel functions, given the schedule or cancel function
   * in a wrapped language. **/
  void* context_ = nullptr;
  /** config-specific, read-only user data that works for all channels created
     with a Credential using the config. */
  void* config_user_data_;

  /** callback function for invoking server authorization check. The
     implementation of this method has to be non-blocking, but can be performed
     synchronously or asynchronously.
     If processing occurs synchronously, it populates \a arg->result, \a
     arg->status, and \a arg->error_details, and returns zero.
     If processing occurs asynchronously, it returns a non-zero value.
     Application then invokes \a arg->cb when processing is completed. Note that
     \a arg->cb cannot be invoked before \a schedule() returns.
  */
  int (*schedule_)(void* config_user_data,
                   grpc_tls_server_authorization_check_arg* arg);

  /** callback function for canceling a server authorization check request. */
  void (*cancel_)(void* config_user_data,
                  grpc_tls_server_authorization_check_arg* arg);

  /** callback function for cleaning up any data associated with server
     authorization check config. */
  void (*destruct_)(void* config_user_data);
};

/* TLS credentials options. */
struct grpc_tls_credentials_options
    : public grpc_core::RefCounted<grpc_tls_credentials_options> {
 public:
  ~grpc_tls_credentials_options() {
    if (key_materials_config_.get() != nullptr) {
      key_materials_config_.get()->Unref();
    }
    if (credential_reload_config_.get() != nullptr) {
      credential_reload_config_.get()->Unref();
    }
    if (server_authorization_check_config_.get() != nullptr) {
      server_authorization_check_config_.get()->Unref();
    }
  }

  /* Getters for member fields. */
  grpc_ssl_client_certificate_request_type cert_request_type() const {
    return cert_request_type_;
  }
  grpc_tls_server_verification_option server_verification_option() const {
    return server_verification_option_;
  }
  grpc_tls_key_materials_config* key_materials_config() const {
    return key_materials_config_.get();
  }
  grpc_tls_credential_reload_config* credential_reload_config() const {
    return credential_reload_config_.get();
  }
  grpc_tls_server_authorization_check_config*
  server_authorization_check_config() const {
    return server_authorization_check_config_.get();
  }

  /* Setters for member fields. */
  void set_cert_request_type(
      const grpc_ssl_client_certificate_request_type type) {
    cert_request_type_ = type;
  }
  void set_server_verification_option(
      const grpc_tls_server_verification_option server_verification_option) {
    server_verification_option_ = server_verification_option;
  }
  void set_key_materials_config(
      grpc_core::RefCountedPtr<grpc_tls_key_materials_config> config) {
    key_materials_config_ = std::move(config);
  }
  void set_credential_reload_config(
      grpc_core::RefCountedPtr<grpc_tls_credential_reload_config> config) {
    credential_reload_config_ = std::move(config);
  }
  void set_server_authorization_check_config(
      grpc_core::RefCountedPtr<grpc_tls_server_authorization_check_config>
          config) {
    server_authorization_check_config_ = std::move(config);
  }

 private:
  grpc_ssl_client_certificate_request_type cert_request_type_;
  grpc_tls_server_verification_option server_verification_option_;
  grpc_core::RefCountedPtr<grpc_tls_key_materials_config> key_materials_config_;
  grpc_core::RefCountedPtr<grpc_tls_credential_reload_config>
      credential_reload_config_;
  grpc_core::RefCountedPtr<grpc_tls_server_authorization_check_config>
      server_authorization_check_config_;
};

#endif /* GRPC_CORE_LIB_SECURITY_CREDENTIALS_TLS_GRPC_TLS_CREDENTIALS_OPTIONS_H \
        */
