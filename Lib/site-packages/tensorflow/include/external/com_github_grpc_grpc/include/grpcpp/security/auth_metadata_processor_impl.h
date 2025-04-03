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

#ifndef GRPCPP_SECURITY_AUTH_METADATA_PROCESSOR_IMPL_H
#define GRPCPP_SECURITY_AUTH_METADATA_PROCESSOR_IMPL_H

#include <map>

#include <grpcpp/security/auth_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/string_ref.h>

namespace grpc_impl {

/// Interface allowing custom server-side authorization based on credentials
/// encoded in metadata.  Objects of this type can be passed to
/// \a ServerCredentials::SetAuthMetadataProcessor().
class AuthMetadataProcessor {
 public:
  typedef std::multimap<grpc::string_ref, grpc::string_ref> InputMetadata;
  typedef std::multimap<grpc::string, grpc::string> OutputMetadata;

  virtual ~AuthMetadataProcessor() {}

  /// If this method returns true, the \a Process function will be scheduled in
  /// a different thread from the one processing the call.
  virtual bool IsBlocking() const { return true; }

  /// context is read/write: it contains the properties of the channel peer and
  /// it is the job of the Process method to augment it with properties derived
  /// from the passed-in auth_metadata.
  /// consumed_auth_metadata needs to be filled with metadata that has been
  /// consumed by the processor and will be removed from the call.
  /// response_metadata is the metadata that will be sent as part of the
  /// response.
  /// If the return value is not Status::OK, the rpc call will be aborted with
  /// the error code and error message sent back to the client.
  virtual grpc::Status Process(const InputMetadata& auth_metadata,
                               grpc::AuthContext* context,
                               OutputMetadata* consumed_auth_metadata,
                               OutputMetadata* response_metadata) = 0;
};

}  // namespace grpc_impl

#endif  // GRPCPP_SECURITY_AUTH_METADATA_PROCESSOR_IMPL_H
