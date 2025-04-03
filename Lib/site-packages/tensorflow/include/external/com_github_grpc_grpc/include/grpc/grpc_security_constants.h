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

#ifndef GRPC_GRPC_SECURITY_CONSTANTS_H
#define GRPC_GRPC_SECURITY_CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif

#define GRPC_TRANSPORT_SECURITY_TYPE_PROPERTY_NAME "transport_security_type"
#define GRPC_SSL_TRANSPORT_SECURITY_TYPE "ssl"

#define GRPC_X509_CN_PROPERTY_NAME "x509_common_name"
#define GRPC_X509_SAN_PROPERTY_NAME "x509_subject_alternative_name"
#define GRPC_X509_PEM_CERT_PROPERTY_NAME "x509_pem_cert"
#define GRPC_X509_PEM_CERT_CHAIN_PROPERTY_NAME "x509_pem_cert_chain"
#define GRPC_SSL_SESSION_REUSED_PROPERTY "ssl_session_reused"
#define GRPC_TRANSPORT_SECURITY_LEVEL_PROPERTY_NAME "security_level"

/** Environment variable that points to the default SSL roots file. This file
   must be a PEM encoded file with all the roots such as the one that can be
   downloaded from https://pki.google.com/roots.pem.  */
#define GRPC_DEFAULT_SSL_ROOTS_FILE_PATH_ENV_VAR \
  "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"

/** Environment variable that points to the google default application
   credentials json key or refresh token. Used in the
   grpc_google_default_credentials_create function. */
#define GRPC_GOOGLE_CREDENTIALS_ENV_VAR "GOOGLE_APPLICATION_CREDENTIALS"

/** Results for the SSL roots override callback. */
typedef enum {
  GRPC_SSL_ROOTS_OVERRIDE_OK,
  GRPC_SSL_ROOTS_OVERRIDE_FAIL_PERMANENTLY, /** Do not try fallback options. */
  GRPC_SSL_ROOTS_OVERRIDE_FAIL
} grpc_ssl_roots_override_result;

/** Callback results for dynamically loading a SSL certificate config. */
typedef enum {
  GRPC_SSL_CERTIFICATE_CONFIG_RELOAD_UNCHANGED,
  GRPC_SSL_CERTIFICATE_CONFIG_RELOAD_NEW,
  GRPC_SSL_CERTIFICATE_CONFIG_RELOAD_FAIL
} grpc_ssl_certificate_config_reload_status;

typedef enum {
  /** Server does not request client certificate.
     The certificate presented by the client is not checked by the server at
     all. (A client may present a self signed or signed certificate or not
     present a certificate at all and any of those option would be accepted) */
  GRPC_SSL_DONT_REQUEST_CLIENT_CERTIFICATE,
  /** Server requests client certificate but does not enforce that the client
     presents a certificate.

     If the client presents a certificate, the client authentication is left to
     the application (the necessary metadata will be available to the
     application via authentication context properties, see grpc_auth_context).

     The client's key certificate pair must be valid for the SSL connection to
     be established. */
  GRPC_SSL_REQUEST_CLIENT_CERTIFICATE_BUT_DONT_VERIFY,
  /** Server requests client certificate but does not enforce that the client
     presents a certificate.

     If the client presents a certificate, the client authentication is done by
     the gRPC framework. (For a successful connection the client needs to either
     present a certificate that can be verified against the root certificate
     configured by the server or not present a certificate at all)

     The client's key certificate pair must be valid for the SSL connection to
     be established. */
  GRPC_SSL_REQUEST_CLIENT_CERTIFICATE_AND_VERIFY,
  /** Server requests client certificate and enforces that the client presents a
     certificate.

     If the client presents a certificate, the client authentication is left to
     the application (the necessary metadata will be available to the
     application via authentication context properties, see grpc_auth_context).

     The client's key certificate pair must be valid for the SSL connection to
     be established. */
  GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_BUT_DONT_VERIFY,
  /** Server requests client certificate and enforces that the client presents a
     certificate.

     The certificate presented by the client is verified by the gRPC framework.
     (For a successful connection the client needs to present a certificate that
     can be verified against the root certificate configured by the server)

     The client's key certificate pair must be valid for the SSL connection to
     be established. */
  GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY
} grpc_ssl_client_certificate_request_type;

/* Security levels of grpc transport security. It represents an inherent
 * property of a backend connection and is determined by a channel credential
 * used to create the connection. */
typedef enum {
  GRPC_SECURITY_MIN,
  GRPC_SECURITY_NONE = GRPC_SECURITY_MIN,
  GRPC_INTEGRITY_ONLY,
  GRPC_PRIVACY_AND_INTEGRITY,
  GRPC_SECURITY_MAX = GRPC_PRIVACY_AND_INTEGRITY,
} grpc_security_level;

typedef enum {
  /** Default option: performs server certificate verification and hostname
     verification. */
  GRPC_TLS_SERVER_VERIFICATION,
  /** Performs server certificate verification, but skips hostname verification
     Client is responsible for verifying server's identity via
     server authorization check callback. */
  GRPC_TLS_SKIP_HOSTNAME_VERIFICATION,
  /** Skips both server certificate and hostname verification.
     Client is responsible for verifying server's identity and
     server's certificate via server authorization check callback. */
  GRPC_TLS_SKIP_ALL_SERVER_VERIFICATION
} grpc_tls_server_verification_option;

/**
 * Type of local connections for which local channel/server credentials will be
 * applied. It supports UDS and local TCP connections.
 */
typedef enum { UDS = 0, LOCAL_TCP } grpc_local_connect_type;

#ifdef __cplusplus
}
#endif

#endif /* GRPC_GRPC_SECURITY_CONSTANTS_H */
