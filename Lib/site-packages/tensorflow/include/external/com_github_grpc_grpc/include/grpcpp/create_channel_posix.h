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

#ifndef GRPCPP_CREATE_CHANNEL_POSIX_H
#define GRPCPP_CREATE_CHANNEL_POSIX_H

#include <grpcpp/create_channel_posix_impl.h>

namespace grpc {

#ifdef GPR_SUPPORT_CHANNELS_FROM_FD

static inline std::shared_ptr<Channel> CreateInsecureChannelFromFd(
    const grpc::string& target, int fd) {
  return ::grpc_impl::CreateInsecureChannelFromFd(target, fd);
}

static inline std::shared_ptr<Channel> CreateCustomInsecureChannelFromFd(
    const grpc::string& target, int fd, const ChannelArguments& args) {
  return ::grpc_impl::CreateCustomInsecureChannelFromFd(target, fd, args);
}

namespace experimental {

static inline std::shared_ptr<Channel>
CreateCustomInsecureChannelWithInterceptorsFromFd(
    const grpc::string& target, int fd, const ChannelArguments& args,
    std::unique_ptr<std::vector<
        std::unique_ptr<experimental::ClientInterceptorFactoryInterface>>>
        interceptor_creators) {
  return ::grpc_impl::experimental::
      CreateCustomInsecureChannelWithInterceptorsFromFd(
          target, fd, args, std::move(interceptor_creators));
}

}  // namespace experimental

#endif  // GPR_SUPPORT_CHANNELS_FROM_FD

}  // namespace grpc

#endif  // GRPCPP_CREATE_CHANNEL_POSIX_H
