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

#ifndef GRPC_CORE_LIB_SECURITY_SECURITY_CONNECTOR_SSL_UTILS_H
#define GRPC_CORE_LIB_SECURITY_SECURITY_CONNECTOR_SSL_UTILS_H

#include <grpc/support/port_platform.h>

#include <stdbool.h>

#include <grpc/grpc_security.h>
#include <grpc/slice_buffer.h>

#include "src/core/lib/gprpp/global_config.h"
#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/gprpp/string_view.h"
#include "src/core/lib/iomgr/error.h"
#include "src/core/lib/security/security_connector/security_connector.h"
#include "src/core/lib/security/security_connector/ssl_utils_config.h"
#include "src/core/tsi/ssl_transport_security.h"
#include "src/core/tsi/transport_security.h"
#include "src/core/tsi/transport_security_interface.h"

/* --- Util --- */

/* --- URL schemes. --- */
#define GRPC_SSL_URL_SCHEME "https"

/* Check ALPN information returned from SSL handshakes. */
grpc_error* grpc_ssl_check_alpn(const tsi_peer* peer);

/* Check peer name information returned from SSL handshakes. */
grpc_error* grpc_ssl_check_peer_name(grpc_core::StringView peer_name,
                                     const tsi_peer* peer);
/* Compare targer_name information extracted from SSL security connectors. */
int grpc_ssl_cmp_target_name(
    grpc_core::StringView target_name, grpc_core::StringView other_target_name,
    grpc_core::StringView overridden_target_name,
    grpc_core::StringView other_overridden_target_name);
/* Check the host that will be set for a call is acceptable.*/
bool grpc_ssl_check_call_host(grpc_core::StringView host,
                              grpc_core::StringView target_name,
                              grpc_core::StringView overridden_target_name,
                              grpc_auth_context* auth_context,
                              grpc_closure* on_call_host_checked,
                              grpc_error** error);
/* Return HTTP2-compliant cipher suites that gRPC accepts by default. */
const char* grpc_get_ssl_cipher_suites(void);

/* Map from grpc_ssl_client_certificate_request_type to
 * tsi_client_certificate_request_type. */
tsi_client_certificate_request_type
grpc_get_tsi_client_certificate_request_type(
    grpc_ssl_client_certificate_request_type grpc_request_type);

/* Map tsi_security_level string to grpc_security_level enum. */
grpc_security_level grpc_tsi_security_level_string_to_enum(
    const char* security_level);

/* Map grpc_security_level enum to a string. */
const char* grpc_security_level_to_string(grpc_security_level security_level);

/* Check security level of channel and call credential.*/
bool grpc_check_security_level(grpc_security_level channel_level,
                               grpc_security_level call_cred_level);

/* Return an array of strings containing alpn protocols. */
const char** grpc_fill_alpn_protocol_strings(size_t* num_alpn_protocols);

/* Initialize TSI SSL server/client handshaker factory. */
grpc_security_status grpc_ssl_tsi_client_handshaker_factory_init(
    tsi_ssl_pem_key_cert_pair* key_cert_pair, const char* pem_root_certs,
    bool skip_server_certificate_verification,
    tsi_ssl_session_cache* ssl_session_cache,
    tsi_ssl_client_handshaker_factory** handshaker_factory);

grpc_security_status grpc_ssl_tsi_server_handshaker_factory_init(
    tsi_ssl_pem_key_cert_pair* key_cert_pairs, size_t num_key_cert_pairs,
    const char* pem_root_certs,
    grpc_ssl_client_certificate_request_type client_certificate_request,
    tsi_ssl_server_handshaker_factory** handshaker_factory);

/* Exposed for testing only. */
grpc_core::RefCountedPtr<grpc_auth_context> grpc_ssl_peer_to_auth_context(
    const tsi_peer* peer, const char* transport_security_type);
tsi_peer grpc_shallow_peer_from_ssl_auth_context(
    const grpc_auth_context* auth_context);
void grpc_shallow_peer_destruct(tsi_peer* peer);
int grpc_ssl_host_matches_name(const tsi_peer* peer,
                               grpc_core::StringView peer_name);

/* --- Default SSL Root Store. --- */
namespace grpc_core {

// The class implements default SSL root store.
class DefaultSslRootStore {
 public:
  // Gets the default SSL root store. Returns nullptr if not found.
  static const tsi_ssl_root_certs_store* GetRootStore();

  // Gets the default PEM root certificate.
  static const char* GetPemRootCerts();

 protected:
  // Returns default PEM root certificates in nullptr terminated grpc_slice.
  // This function is protected instead of private, so that it can be tested.
  static grpc_slice ComputePemRootCerts();

 private:
  // Construct me not!
  DefaultSslRootStore();

  // Initialization of default SSL root store.
  static void InitRootStore();

  // One-time initialization of default SSL root store.
  static void InitRootStoreOnce();

  // SSL root store in tsi_ssl_root_certs_store object.
  static tsi_ssl_root_certs_store* default_root_store_;

  // Default PEM root certificates.
  static grpc_slice default_pem_root_certs_;
};

class PemKeyCertPair {
 public:
  // Construct from the C struct.  We steal its members and then immediately
  // free it.
  explicit PemKeyCertPair(grpc_ssl_pem_key_cert_pair* pair)
      : private_key_(const_cast<char*>(pair->private_key)),
        cert_chain_(const_cast<char*>(pair->cert_chain)) {
    gpr_free(pair);
  }

  // Movable.
  PemKeyCertPair(PemKeyCertPair&& other) {
    private_key_ = std::move(other.private_key_);
    cert_chain_ = std::move(other.cert_chain_);
  }
  PemKeyCertPair& operator=(PemKeyCertPair&& other) {
    private_key_ = std::move(other.private_key_);
    cert_chain_ = std::move(other.cert_chain_);
    return *this;
  }

  // Copyable.
  PemKeyCertPair(const PemKeyCertPair& other)
      : private_key_(gpr_strdup(other.private_key())),
        cert_chain_(gpr_strdup(other.cert_chain())) {}
  PemKeyCertPair& operator=(const PemKeyCertPair& other) {
    private_key_ = grpc_core::UniquePtr<char>(gpr_strdup(other.private_key()));
    cert_chain_ = grpc_core::UniquePtr<char>(gpr_strdup(other.cert_chain()));
    return *this;
  }

  char* private_key() const { return private_key_.get(); }
  char* cert_chain() const { return cert_chain_.get(); }

 private:
  grpc_core::UniquePtr<char> private_key_;
  grpc_core::UniquePtr<char> cert_chain_;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_SECURITY_SECURITY_CONNECTOR_SSL_UTILS_H \
        */
