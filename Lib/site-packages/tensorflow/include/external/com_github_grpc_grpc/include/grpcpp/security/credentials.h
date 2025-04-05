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

#ifndef GRPCPP_SECURITY_CREDENTIALS_H
#define GRPCPP_SECURITY_CREDENTIALS_H

#include <grpcpp/security/credentials_impl.h>

namespace grpc {

typedef ::grpc_impl::ChannelCredentials ChannelCredentials;
typedef ::grpc_impl::CallCredentials CallCredentials;
typedef ::grpc_impl::SslCredentialsOptions SslCredentialsOptions;
typedef ::grpc_impl::SecureCallCredentials SecureCallCredentials;
typedef ::grpc_impl::SecureChannelCredentials SecureChannelCredentials;
typedef ::grpc_impl::MetadataCredentialsPlugin MetadataCredentialsPlugin;

static inline std::shared_ptr<grpc_impl::ChannelCredentials>
GoogleDefaultCredentials() {
  return ::grpc_impl::GoogleDefaultCredentials();
}

static inline std::shared_ptr<ChannelCredentials> SslCredentials(
    const SslCredentialsOptions& options) {
  return ::grpc_impl::SslCredentials(options);
}

static inline std::shared_ptr<grpc_impl::CallCredentials>
GoogleComputeEngineCredentials() {
  return ::grpc_impl::GoogleComputeEngineCredentials();
}

/// Constant for maximum auth token lifetime.
constexpr long kMaxAuthTokenLifetimeSecs =
    ::grpc_impl::kMaxAuthTokenLifetimeSecs;

static inline std::shared_ptr<grpc_impl::CallCredentials>
ServiceAccountJWTAccessCredentials(
    const grpc::string& json_key,
    long token_lifetime_seconds = grpc::kMaxAuthTokenLifetimeSecs) {
  return ::grpc_impl::ServiceAccountJWTAccessCredentials(
      json_key, token_lifetime_seconds);
}

static inline std::shared_ptr<grpc_impl::CallCredentials>
GoogleRefreshTokenCredentials(const grpc::string& json_refresh_token) {
  return ::grpc_impl::GoogleRefreshTokenCredentials(json_refresh_token);
}

static inline std::shared_ptr<grpc_impl::CallCredentials>
AccessTokenCredentials(const grpc::string& access_token) {
  return ::grpc_impl::AccessTokenCredentials(access_token);
}

static inline std::shared_ptr<grpc_impl::CallCredentials> GoogleIAMCredentials(
    const grpc::string& authorization_token,
    const grpc::string& authority_selector) {
  return ::grpc_impl::GoogleIAMCredentials(authorization_token,
                                           authority_selector);
}

static inline std::shared_ptr<ChannelCredentials> CompositeChannelCredentials(
    const std::shared_ptr<ChannelCredentials>& channel_creds,
    const std::shared_ptr<CallCredentials>& call_creds) {
  return ::grpc_impl::CompositeChannelCredentials(channel_creds, call_creds);
}

static inline std::shared_ptr<grpc_impl::CallCredentials>
CompositeCallCredentials(const std::shared_ptr<CallCredentials>& creds1,
                         const std::shared_ptr<CallCredentials>& creds2) {
  return ::grpc_impl::CompositeCallCredentials(creds1, creds2);
}

static inline std::shared_ptr<grpc_impl::ChannelCredentials>
InsecureChannelCredentials() {
  return ::grpc_impl::InsecureChannelCredentials();
}

typedef ::grpc_impl::MetadataCredentialsPlugin MetadataCredentialsPlugin;

static inline std::shared_ptr<grpc_impl::CallCredentials>
MetadataCredentialsFromPlugin(
    std::unique_ptr<MetadataCredentialsPlugin> plugin) {
  return ::grpc_impl::MetadataCredentialsFromPlugin(std::move(plugin));
}

namespace experimental {

typedef ::grpc_impl::experimental::StsCredentialsOptions StsCredentialsOptions;

static inline grpc::Status StsCredentialsOptionsFromJson(
    const grpc::string& json_string, StsCredentialsOptions* options) {
  return ::grpc_impl::experimental::StsCredentialsOptionsFromJson(json_string,
                                                                  options);
}

static inline grpc::Status StsCredentialsOptionsFromEnv(
    StsCredentialsOptions* options) {
  return grpc_impl::experimental::StsCredentialsOptionsFromEnv(options);
}

static inline std::shared_ptr<grpc_impl::CallCredentials> StsCredentials(
    const StsCredentialsOptions& options) {
  return grpc_impl::experimental::StsCredentials(options);
}

typedef ::grpc_impl::experimental::AltsCredentialsOptions
    AltsCredentialsOptions;

static inline std::shared_ptr<grpc_impl::ChannelCredentials> AltsCredentials(
    const AltsCredentialsOptions& options) {
  return ::grpc_impl::experimental::AltsCredentials(options);
}

static inline std::shared_ptr<grpc_impl::ChannelCredentials> LocalCredentials(
    grpc_local_connect_type type) {
  return ::grpc_impl::experimental::LocalCredentials(type);
}

static inline std::shared_ptr<grpc_impl::ChannelCredentials> TlsCredentials(
    const ::grpc_impl::experimental::TlsCredentialsOptions& options) {
  return ::grpc_impl::experimental::TlsCredentials(options);
}

}  // namespace experimental
}  // namespace grpc

#endif  // GRPCPP_SECURITY_CREDENTIALS_H
