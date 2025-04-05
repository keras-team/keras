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

#ifndef GRPCPP_IMPL_CODEGEN_METHOD_HANDLER_H
#define GRPCPP_IMPL_CODEGEN_METHOD_HANDLER_H

#include <grpcpp/impl/codegen/method_handler_impl.h>

namespace grpc {

namespace internal {

template <class ServiceType, class RequestType, class ResponseType>
using BidiStreamingHandler =
    ::grpc_impl::internal::BidiStreamingHandler<ServiceType, RequestType,
                                                ResponseType>;

template <class ServiceType, class RequestType, class ResponseType>
using RpcMethodHandler =
    ::grpc_impl::internal::RpcMethodHandler<ServiceType, RequestType,
                                            ResponseType>;

template <class ServiceType, class RequestType, class ResponseType>
using ClientStreamingHandler =
    ::grpc_impl::internal::ClientStreamingHandler<ServiceType, RequestType,
                                                  ResponseType>;

template <class ServiceType, class RequestType, class ResponseType>
using ServerStreamingHandler =
    ::grpc_impl::internal::ServerStreamingHandler<ServiceType, RequestType,
                                                  ResponseType>;

template <class Streamer, bool WriteNeeded>
using TemplatedBidiStreamingHandler =
    ::grpc_impl::internal::TemplatedBidiStreamingHandler<Streamer, WriteNeeded>;

template <class RequestType, class ResponseType>
using StreamedUnaryHandler =
    ::grpc_impl::internal::StreamedUnaryHandler<RequestType, ResponseType>;

template <class RequestType, class ResponseType>
using SplitServerStreamingHandler =
    ::grpc_impl::internal::SplitServerStreamingHandler<RequestType,
                                                       ResponseType>;

template <StatusCode code>
using ErrorMethodHandler = ::grpc_impl::internal::ErrorMethodHandler<code>;

using UnknownMethodHandler = ::grpc_impl::internal::UnknownMethodHandler;

using ResourceExhaustedHandler =
    ::grpc_impl::internal::ResourceExhaustedHandler;

}  // namespace internal

}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_METHOD_HANDLER_H
