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

#ifndef GRPC_INTERNAL_CPP_CLIENT_CREATE_CHANNEL_INTERNAL_H
#define GRPC_INTERNAL_CPP_CLIENT_CREATE_CHANNEL_INTERNAL_H

#include <memory>

#include <grpcpp/channel.h>
#include <grpcpp/impl/codegen/client_interceptor.h>
#include <grpcpp/support/config.h>

struct grpc_channel;

namespace grpc {

std::shared_ptr<Channel> CreateChannelInternal(
    const grpc::string& host, grpc_channel* c_channel,
    std::vector<std::unique_ptr<
        ::grpc::experimental::ClientInterceptorFactoryInterface>>
        interceptor_creators);

}  // namespace grpc

#endif  // GRPC_INTERNAL_CPP_CLIENT_CREATE_CHANNEL_INTERNAL_H
