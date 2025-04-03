/*
 *
 * Copyright 2019 gRPC authors.
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

#ifndef GRPCPP_CREATE_CHANNEL_H
#define GRPCPP_CREATE_CHANNEL_H

#include <grpcpp/create_channel_impl.h>
#include <grpcpp/support/channel_arguments.h>

namespace grpc {

static inline std::shared_ptr<::grpc::Channel> CreateChannel(
    const grpc::string& target,
    const std::shared_ptr<ChannelCredentials>& creds) {
  return ::grpc_impl::CreateChannelImpl(target, creds);
}

static inline std::shared_ptr<::grpc::Channel> CreateCustomChannel(
    const grpc::string& target,
    const std::shared_ptr<ChannelCredentials>& creds,
    const ChannelArguments& args) {
  return ::grpc_impl::CreateCustomChannelImpl(target, creds, args);
}

namespace experimental {

static inline std::shared_ptr<::grpc::Channel>
CreateCustomChannelWithInterceptors(
    const grpc::string& target,
    const std::shared_ptr<ChannelCredentials>& creds,
    const ChannelArguments& args,
    std::vector<
        std::unique_ptr<experimental::ClientInterceptorFactoryInterface>>
        interceptor_creators) {
  return ::grpc_impl::experimental::CreateCustomChannelWithInterceptors(
      target, creds, args, std::move(interceptor_creators));
}

}  // namespace experimental
}  // namespace grpc

#endif  // GRPCPP_CREATE_CHANNEL_H
