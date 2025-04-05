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

#ifndef GRPCPP_IMPL_CODEGEN_ASYNC_GENERIC_SERVICE_H
#define GRPCPP_IMPL_CODEGEN_ASYNC_GENERIC_SERVICE_H

#include <grpc/impl/codegen/port_platform.h>

#include <grpcpp/impl/codegen/async_stream_impl.h>
#include <grpcpp/impl/codegen/byte_buffer.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_callback_impl.h>

struct grpc_server;

namespace grpc {

typedef ::grpc_impl::ServerAsyncReaderWriter<ByteBuffer, ByteBuffer>
    GenericServerAsyncReaderWriter;
typedef ::grpc_impl::ServerAsyncResponseWriter<ByteBuffer>
    GenericServerAsyncResponseWriter;
typedef ::grpc_impl::ServerAsyncReader<ByteBuffer, ByteBuffer>
    GenericServerAsyncReader;
typedef ::grpc_impl::ServerAsyncWriter<ByteBuffer> GenericServerAsyncWriter;

class GenericServerContext final : public ::grpc_impl::ServerContext {
 public:
  const grpc::string& method() const { return method_; }
  const grpc::string& host() const { return host_; }

 private:
  friend class grpc_impl::Server;
  friend class grpc::ServerInterface;

  void Clear() {
    method_.clear();
    host_.clear();
    ::grpc_impl::ServerContext::Clear();
  }

  grpc::string method_;
  grpc::string host_;
};

// A generic service at the server side accepts all RPC methods and hosts. It is
// typically used in proxies. The generic service can be registered to a server
// which also has other services.
// Sample usage:
//   ServerBuilder builder;
//   auto cq = builder.AddCompletionQueue();
//   AsyncGenericService generic_service;
//   builder.RegisterAsyncGenericService(&generic_service);
//   auto server = builder.BuildAndStart();
//
//   // request a new call
//   GenericServerContext context;
//   GenericServerAsyncReaderWriter stream;
//   generic_service.RequestCall(&context, &stream, cq.get(), cq.get(), tag);
//
// When tag is retrieved from cq->Next(), context.method() can be used to look
// at the method and the RPC can be handled accordingly.
class AsyncGenericService final {
 public:
  AsyncGenericService() : server_(nullptr) {}

  void RequestCall(GenericServerContext* ctx,
                   GenericServerAsyncReaderWriter* reader_writer,
                   ::grpc_impl::CompletionQueue* call_cq,
                   ::grpc_impl::ServerCompletionQueue* notification_cq,
                   void* tag);

 private:
  friend class grpc_impl::Server;
  grpc_impl::Server* server_;
};

#ifndef GRPC_CALLBACK_API_NONEXPERIMENTAL
namespace experimental {
#endif

/// \a ServerGenericBidiReactor is the reactor class for bidi streaming RPCs
/// invoked on a CallbackGenericService. It is just a ServerBidi reactor with
/// ByteBuffer arguments.
using ServerGenericBidiReactor =
    ::grpc_impl::ServerBidiReactor<ByteBuffer, ByteBuffer>;

class GenericCallbackServerContext final
    : public ::grpc_impl::CallbackServerContext {
 public:
  const grpc::string& method() const { return method_; }
  const grpc::string& host() const { return host_; }

 private:
  friend class ::grpc_impl::Server;
  friend class ::grpc::ServerInterface;

  void Clear() {
    method_.clear();
    host_.clear();
    ::grpc_impl::CallbackServerContext::Clear();
  }

  grpc::string method_;
  grpc::string host_;
};

/// \a CallbackGenericService is the base class for generic services implemented
/// using the callback API and registered through the ServerBuilder using
/// RegisterCallbackGenericService.
class CallbackGenericService {
 public:
  CallbackGenericService() {}
  virtual ~CallbackGenericService() {}

  /// The "method handler" for the generic API. This function should be
  /// overridden to provide a ServerGenericBidiReactor that implements the
  /// application-level interface for this RPC. Unimplemented by default.
  virtual ServerGenericBidiReactor* CreateReactor(
      GenericCallbackServerContext* /*ctx*/) {
    class Reactor : public ServerGenericBidiReactor {
     public:
      Reactor() { this->Finish(Status(StatusCode::UNIMPLEMENTED, "")); }
      void OnDone() override { delete this; }
    };
    return new Reactor;
  }

 private:
  friend class ::grpc_impl::Server;

  ::grpc_impl::internal::CallbackBidiHandler<ByteBuffer, ByteBuffer>*
  Handler() {
    return new ::grpc_impl::internal::CallbackBidiHandler<ByteBuffer,
                                                          ByteBuffer>(
        [this](::grpc_impl::CallbackServerContext* ctx) {
          return CreateReactor(static_cast<GenericCallbackServerContext*>(ctx));
        });
  }

  grpc_impl::Server* server_{nullptr};
};

#ifndef GRPC_CALLBACK_API_NONEXPERIMENTAL
}  // namespace experimental
#endif
}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_ASYNC_GENERIC_SERVICE_H
