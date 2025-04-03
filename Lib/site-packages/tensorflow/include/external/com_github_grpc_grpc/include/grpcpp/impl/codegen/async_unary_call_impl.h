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

#ifndef GRPCPP_IMPL_CODEGEN_ASYNC_UNARY_CALL_IMPL_H
#define GRPCPP_IMPL_CODEGEN_ASYNC_UNARY_CALL_IMPL_H

#include <grpcpp/impl/codegen/call.h>
#include <grpcpp/impl/codegen/channel_interface.h>
#include <grpcpp/impl/codegen/client_context_impl.h>
#include <grpcpp/impl/codegen/server_context_impl.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>

namespace grpc_impl {

/// An interface relevant for async client side unary RPCs (which send
/// one request message to a server and receive one response message).
template <class R>
class ClientAsyncResponseReaderInterface {
 public:
  virtual ~ClientAsyncResponseReaderInterface() {}

  /// Start the call that was set up by the constructor, but only if the
  /// constructor was invoked through the "Prepare" API which doesn't actually
  /// start the call
  virtual void StartCall() = 0;

  /// Request notification of the reading of initial metadata. Completion
  /// will be notified by \a tag on the associated completion queue.
  /// This call is optional, but if it is used, it cannot be used concurrently
  /// with or after the \a Finish method.
  ///
  /// \param[in] tag Tag identifying this request.
  virtual void ReadInitialMetadata(void* tag) = 0;

  /// Request to receive the server's response \a msg and final \a status for
  /// the call, and to notify \a tag on this call's completion queue when
  /// finished.
  ///
  /// This function will return when either:
  /// - when the server's response message and status have been received.
  /// - when the server has returned a non-OK status (no message expected in
  ///   this case).
  /// - when the call failed for some reason and the library generated a
  ///   non-OK status.
  ///
  /// \param[in] tag Tag identifying this request.
  /// \param[out] status To be updated with the operation status.
  /// \param[out] msg To be filled in with the server's response message.
  virtual void Finish(R* msg, ::grpc::Status* status, void* tag) = 0;
};

namespace internal {
template <class R>
class ClientAsyncResponseReaderFactory {
 public:
  /// Start a call and write the request out if \a start is set.
  /// \a tag will be notified on \a cq when the call has been started (i.e.
  /// intitial metadata sent) and \a request has been written out.
  /// If \a start is not set, the actual call must be initiated by StartCall
  /// Note that \a context will be used to fill in custom initial metadata
  /// used to send to the server when starting the call.
  template <class W>
  static ClientAsyncResponseReader<R>* Create(
      ::grpc::ChannelInterface* channel, ::grpc_impl::CompletionQueue* cq,
      const ::grpc::internal::RpcMethod& method,
      ::grpc_impl::ClientContext* context, const W& request, bool start) {
    ::grpc::internal::Call call = channel->CreateCall(method, context, cq);
    return new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
        call.call(), sizeof(ClientAsyncResponseReader<R>)))
        ClientAsyncResponseReader<R>(call, context, request, start);
  }
};
}  // namespace internal

/// Async API for client-side unary RPCs, where the message response
/// received from the server is of type \a R.
template <class R>
class ClientAsyncResponseReader final
    : public ClientAsyncResponseReaderInterface<R> {
 public:
  // always allocated against a call arena, no memory free required
  static void operator delete(void* /*ptr*/, std::size_t size) {
    GPR_CODEGEN_ASSERT(size == sizeof(ClientAsyncResponseReader));
  }

  // This operator should never be called as the memory should be freed as part
  // of the arena destruction. It only exists to provide a matching operator
  // delete to the operator new so that some compilers will not complain (see
  // https://github.com/grpc/grpc/issues/11301) Note at the time of adding this
  // there are no tests catching the compiler warning.
  static void operator delete(void*, void*) { GPR_CODEGEN_ASSERT(false); }

  void StartCall() override {
    GPR_CODEGEN_ASSERT(!started_);
    started_ = true;
    StartCallInternal();
  }

  /// See \a ClientAsyncResponseReaderInterface::ReadInitialMetadata for
  /// semantics.
  ///
  /// Side effect:
  ///   - the \a ClientContext associated with this call is updated with
  ///     possible initial and trailing metadata sent from the server.
  void ReadInitialMetadata(void* tag) override {
    GPR_CODEGEN_ASSERT(started_);
    GPR_CODEGEN_ASSERT(!context_->initial_metadata_received_);

    single_buf.set_output_tag(tag);
    single_buf.RecvInitialMetadata(context_);
    call_.PerformOps(&single_buf);
    initial_metadata_read_ = true;
  }

  /// See \a ClientAysncResponseReaderInterface::Finish for semantics.
  ///
  /// Side effect:
  ///   - the \a ClientContext associated with this call is updated with
  ///     possible initial and trailing metadata sent from the server.
  void Finish(R* msg, ::grpc::Status* status, void* tag) override {
    GPR_CODEGEN_ASSERT(started_);
    if (initial_metadata_read_) {
      finish_buf.set_output_tag(tag);
      finish_buf.RecvMessage(msg);
      finish_buf.AllowNoMessage();
      finish_buf.ClientRecvStatus(context_, status);
      call_.PerformOps(&finish_buf);
    } else {
      single_buf.set_output_tag(tag);
      single_buf.RecvInitialMetadata(context_);
      single_buf.RecvMessage(msg);
      single_buf.AllowNoMessage();
      single_buf.ClientRecvStatus(context_, status);
      call_.PerformOps(&single_buf);
    }
  }

 private:
  friend class internal::ClientAsyncResponseReaderFactory<R>;
  ::grpc_impl::ClientContext* const context_;
  ::grpc::internal::Call call_;
  bool started_;
  bool initial_metadata_read_ = false;

  template <class W>
  ClientAsyncResponseReader(::grpc::internal::Call call,
                            ::grpc_impl::ClientContext* context,
                            const W& request, bool start)
      : context_(context), call_(call), started_(start) {
    // Bind the metadata at time of StartCallInternal but set up the rest here
    // TODO(ctiller): don't assert
    GPR_CODEGEN_ASSERT(single_buf.SendMessage(request).ok());
    single_buf.ClientSendClose();
    if (start) StartCallInternal();
  }

  void StartCallInternal() {
    single_buf.SendInitialMetadata(&context_->send_initial_metadata_,
                                   context_->initial_metadata_flags());
  }

  // disable operator new
  static void* operator new(std::size_t size);
  static void* operator new(std::size_t /*size*/, void* p) { return p; }

  ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                              ::grpc::internal::CallOpSendMessage,
                              ::grpc::internal::CallOpClientSendClose,
                              ::grpc::internal::CallOpRecvInitialMetadata,
                              ::grpc::internal::CallOpRecvMessage<R>,
                              ::grpc::internal::CallOpClientRecvStatus>
      single_buf;
  ::grpc::internal::CallOpSet<::grpc::internal::CallOpRecvMessage<R>,
                              ::grpc::internal::CallOpClientRecvStatus>
      finish_buf;
};

/// Async server-side API for handling unary calls, where the single
/// response message sent to the client is of type \a W.
template <class W>
class ServerAsyncResponseWriter final
    : public ::grpc::internal::ServerAsyncStreamingInterface {
 public:
  explicit ServerAsyncResponseWriter(::grpc_impl::ServerContext* ctx)
      : call_(nullptr, nullptr, nullptr), ctx_(ctx) {}

  /// See \a ServerAsyncStreamingInterface::SendInitialMetadata for semantics.
  ///
  /// Side effect:
  ///   The initial metadata that will be sent to the client from this op will
  ///   be taken from the \a ServerContext associated with the call.
  ///
  /// \param[in] tag Tag identifying this request.
  void SendInitialMetadata(void* tag) override {
    GPR_CODEGEN_ASSERT(!ctx_->sent_initial_metadata_);

    meta_buf_.set_output_tag(tag);
    meta_buf_.SendInitialMetadata(&ctx_->initial_metadata_,
                                  ctx_->initial_metadata_flags());
    if (ctx_->compression_level_set()) {
      meta_buf_.set_compression_level(ctx_->compression_level());
    }
    ctx_->sent_initial_metadata_ = true;
    call_.PerformOps(&meta_buf_);
  }

  /// Indicate that the stream is to be finished and request notification
  /// when the server has sent the appropriate signals to the client to
  /// end the call. Should not be used concurrently with other operations.
  ///
  /// \param[in] tag Tag identifying this request.
  /// \param[in] status To be sent to the client as the result of the call.
  /// \param[in] msg Message to be sent to the client.
  ///
  /// Side effect:
  ///   - also sends initial metadata if not already sent (using the
  ///     \a ServerContext associated with this call).
  ///
  /// Note: if \a status has a non-OK code, then \a msg will not be sent,
  /// and the client will receive only the status with possible trailing
  /// metadata.
  void Finish(const W& msg, const ::grpc::Status& status, void* tag) {
    finish_buf_.set_output_tag(tag);
    finish_buf_.set_core_cq_tag(&finish_buf_);
    if (!ctx_->sent_initial_metadata_) {
      finish_buf_.SendInitialMetadata(&ctx_->initial_metadata_,
                                      ctx_->initial_metadata_flags());
      if (ctx_->compression_level_set()) {
        finish_buf_.set_compression_level(ctx_->compression_level());
      }
      ctx_->sent_initial_metadata_ = true;
    }
    // The response is dropped if the status is not OK.
    if (status.ok()) {
      finish_buf_.ServerSendStatus(&ctx_->trailing_metadata_,
                                   finish_buf_.SendMessage(msg));
    } else {
      finish_buf_.ServerSendStatus(&ctx_->trailing_metadata_, status);
    }
    call_.PerformOps(&finish_buf_);
  }

  /// Indicate that the stream is to be finished with a non-OK status,
  /// and request notification for when the server has finished sending the
  /// appropriate signals to the client to end the call.
  /// Should not be used concurrently with other operations.
  ///
  /// \param[in] tag Tag identifying this request.
  /// \param[in] status To be sent to the client as the result of the call.
  ///   - Note: \a status must have a non-OK code.
  ///
  /// Side effect:
  ///   - also sends initial metadata if not already sent (using the
  ///     \a ServerContext associated with this call).
  void FinishWithError(const ::grpc::Status& status, void* tag) {
    GPR_CODEGEN_ASSERT(!status.ok());
    finish_buf_.set_output_tag(tag);
    if (!ctx_->sent_initial_metadata_) {
      finish_buf_.SendInitialMetadata(&ctx_->initial_metadata_,
                                      ctx_->initial_metadata_flags());
      if (ctx_->compression_level_set()) {
        finish_buf_.set_compression_level(ctx_->compression_level());
      }
      ctx_->sent_initial_metadata_ = true;
    }
    finish_buf_.ServerSendStatus(&ctx_->trailing_metadata_, status);
    call_.PerformOps(&finish_buf_);
  }

 private:
  void BindCall(::grpc::internal::Call* call) override { call_ = *call; }

  ::grpc::internal::Call call_;
  ::grpc_impl::ServerContext* ctx_;
  ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata>
      meta_buf_;
  ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                              ::grpc::internal::CallOpSendMessage,
                              ::grpc::internal::CallOpServerSendStatus>
      finish_buf_;
};

}  // namespace grpc_impl

namespace std {
template <class R>
class default_delete<::grpc_impl::ClientAsyncResponseReader<R>> {
 public:
  void operator()(void* /*p*/) {}
};
template <class R>
class default_delete<::grpc_impl::ClientAsyncResponseReaderInterface<R>> {
 public:
  void operator()(void* /*p*/) {}
};
}  // namespace std

#endif  // GRPCPP_IMPL_CODEGEN_ASYNC_UNARY_CALL_IMPL_H
