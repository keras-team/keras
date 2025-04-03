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

#ifndef GRPCPP_IMPL_CODEGEN_CLIENT_CALLBACK_H
#define GRPCPP_IMPL_CODEGEN_CLIENT_CALLBACK_H

#include <grpcpp/impl/codegen/client_callback_impl.h>

namespace grpc {

#ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
template <class Response>
using ClientCallbackReader = ::grpc_impl::ClientCallbackReader<Response>;

template <class Request>
using ClientCallbackWriter = ::grpc_impl::ClientCallbackWriter<Request>;

template <class Request, class Response>
using ClientCallbackReaderWriter =
    ::grpc_impl::ClientCallbackReaderWriter<Request, Response>;

template <class Response>
using ClientReadReactor = ::grpc_impl::ClientReadReactor<Response>;

template <class Request>
using ClientWriteReactor = ::grpc_impl::ClientWriteReactor<Request>;

template <class Request, class Response>
using ClientBidiReactor = ::grpc_impl::ClientBidiReactor<Request, Response>;

typedef ::grpc_impl::ClientUnaryReactor ClientUnaryReactor;
#endif

// TODO(vjpai): Remove namespace experimental when de-experimentalized fully.
namespace experimental {

template <class Response>
using ClientCallbackReader = ::grpc_impl::ClientCallbackReader<Response>;

template <class Request>
using ClientCallbackWriter = ::grpc_impl::ClientCallbackWriter<Request>;

template <class Request, class Response>
using ClientCallbackReaderWriter =
    ::grpc_impl::ClientCallbackReaderWriter<Request, Response>;

template <class Response>
using ClientReadReactor = ::grpc_impl::ClientReadReactor<Response>;

template <class Request>
using ClientWriteReactor = ::grpc_impl::ClientWriteReactor<Request>;

template <class Request, class Response>
using ClientBidiReactor = ::grpc_impl::ClientBidiReactor<Request, Response>;

typedef ::grpc_impl::ClientUnaryReactor ClientUnaryReactor;

}  // namespace experimental
}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_CLIENT_CALLBACK_H
