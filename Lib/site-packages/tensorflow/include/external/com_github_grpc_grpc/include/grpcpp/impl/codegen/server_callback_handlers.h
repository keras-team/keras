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
 */

#ifndef GRPCPP_IMPL_CODEGEN_SERVER_CALLBACK_HANDLERS_H
#define GRPCPP_IMPL_CODEGEN_SERVER_CALLBACK_HANDLERS_H

#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/rpc_service_method.h>
#include <grpcpp/impl/codegen/server_callback_impl.h>
#include <grpcpp/impl/codegen/server_context_impl.h>
#include <grpcpp/impl/codegen/status.h>

namespace grpc_impl {
namespace internal {

template <class RequestType, class ResponseType>
class CallbackUnaryHandler : public ::grpc::internal::MethodHandler {
 public:
  explicit CallbackUnaryHandler(
      std::function<ServerUnaryReactor*(::grpc_impl::CallbackServerContext*,
                                        const RequestType*, ResponseType*)>
          get_reactor)
      : get_reactor_(std::move(get_reactor)) {}

  void SetMessageAllocator(
      ::grpc::experimental::MessageAllocator<RequestType, ResponseType>*
          allocator) {
    allocator_ = allocator;
  }

  void RunHandler(const HandlerParameter& param) final {
    // Arena allocate a controller structure (that includes request/response)
    ::grpc::g_core_codegen_interface->grpc_call_ref(param.call->call());
    auto* allocator_state = static_cast<
        ::grpc::experimental::MessageHolder<RequestType, ResponseType>*>(
        param.internal_data);

    auto* call = new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
        param.call->call(), sizeof(ServerCallbackUnaryImpl)))
        ServerCallbackUnaryImpl(
            static_cast<::grpc_impl::CallbackServerContext*>(
                param.server_context),
            param.call, allocator_state, std::move(param.call_requester));
    param.server_context->BeginCompletionOp(
        param.call, [call](bool) { call->MaybeDone(); }, call);

    ServerUnaryReactor* reactor = nullptr;
    if (param.status.ok()) {
      reactor = ::grpc::internal::CatchingReactorGetter<ServerUnaryReactor>(
          get_reactor_,
          static_cast<::grpc_impl::CallbackServerContext*>(
              param.server_context),
          call->request(), call->response());
    }

    if (reactor == nullptr) {
      // if deserialization or reactor creator failed, we need to fail the call
      reactor = new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
          param.call->call(), sizeof(UnimplementedUnaryReactor)))
          UnimplementedUnaryReactor(
              ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, ""));
    }

    /// Invoke SetupReactor as the last part of the handler
    call->SetupReactor(reactor);
  }

  void* Deserialize(grpc_call* call, grpc_byte_buffer* req,
                    ::grpc::Status* status, void** handler_data) final {
    ::grpc::ByteBuffer buf;
    buf.set_buffer(req);
    RequestType* request = nullptr;
    ::grpc::experimental::MessageHolder<RequestType, ResponseType>*
        allocator_state = nullptr;
    if (allocator_ != nullptr) {
      allocator_state = allocator_->AllocateMessages();
    } else {
      allocator_state =
          new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
              call, sizeof(DefaultMessageHolder<RequestType, ResponseType>)))
              DefaultMessageHolder<RequestType, ResponseType>();
    }
    *handler_data = allocator_state;
    request = allocator_state->request();
    *status =
        ::grpc::SerializationTraits<RequestType>::Deserialize(&buf, request);
    buf.Release();
    if (status->ok()) {
      return request;
    }
    // Clean up on deserialization failure.
    allocator_state->Release();
    return nullptr;
  }

 private:
  std::function<ServerUnaryReactor*(::grpc_impl::CallbackServerContext*,
                                    const RequestType*, ResponseType*)>
      get_reactor_;
  ::grpc::experimental::MessageAllocator<RequestType, ResponseType>*
      allocator_ = nullptr;

  class ServerCallbackUnaryImpl : public ServerCallbackUnary {
   public:
    void Finish(::grpc::Status s) override {
      finish_tag_.Set(
          call_.call(), [this](bool) { MaybeDone(); }, &finish_ops_,
          reactor_.load(std::memory_order_relaxed)->InternalInlineable());
      finish_ops_.set_core_cq_tag(&finish_tag_);

      if (!ctx_->sent_initial_metadata_) {
        finish_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                        ctx_->initial_metadata_flags());
        if (ctx_->compression_level_set()) {
          finish_ops_.set_compression_level(ctx_->compression_level());
        }
        ctx_->sent_initial_metadata_ = true;
      }
      // The response is dropped if the status is not OK.
      if (s.ok()) {
        finish_ops_.ServerSendStatus(&ctx_->trailing_metadata_,
                                     finish_ops_.SendMessagePtr(response()));
      } else {
        finish_ops_.ServerSendStatus(&ctx_->trailing_metadata_, s);
      }
      finish_ops_.set_core_cq_tag(&finish_tag_);
      call_.PerformOps(&finish_ops_);
    }

    void SendInitialMetadata() override {
      GPR_CODEGEN_ASSERT(!ctx_->sent_initial_metadata_);
      this->Ref();
      meta_tag_.Set(call_.call(),
                    [this](bool ok) {
                      reactor_.load(std::memory_order_relaxed)
                          ->OnSendInitialMetadataDone(ok);
                      MaybeDone();
                    },
                    &meta_ops_, false);
      meta_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                    ctx_->initial_metadata_flags());
      if (ctx_->compression_level_set()) {
        meta_ops_.set_compression_level(ctx_->compression_level());
      }
      ctx_->sent_initial_metadata_ = true;
      meta_ops_.set_core_cq_tag(&meta_tag_);
      call_.PerformOps(&meta_ops_);
    }

   private:
    friend class CallbackUnaryHandler<RequestType, ResponseType>;

    ServerCallbackUnaryImpl(
        ::grpc_impl::CallbackServerContext* ctx, ::grpc::internal::Call* call,
        ::grpc::experimental::MessageHolder<RequestType, ResponseType>*
            allocator_state,
        std::function<void()> call_requester)
        : ctx_(ctx),
          call_(*call),
          allocator_state_(allocator_state),
          call_requester_(std::move(call_requester)) {
      ctx_->set_message_allocator_state(allocator_state);
    }

    /// SetupReactor binds the reactor (which also releases any queued
    /// operations), maybe calls OnCancel if possible/needed, and maybe marks
    /// the completion of the RPC. This should be the last component of the
    /// handler.
    void SetupReactor(ServerUnaryReactor* reactor) {
      reactor_.store(reactor, std::memory_order_relaxed);
      this->BindReactor(reactor);
      this->MaybeCallOnCancel(reactor);
      this->MaybeDone();
    }

    const RequestType* request() { return allocator_state_->request(); }
    ResponseType* response() { return allocator_state_->response(); }

    void MaybeDone() override {
      if (GPR_UNLIKELY(this->Unref() == 1)) {
        reactor_.load(std::memory_order_relaxed)->OnDone();
        grpc_call* call = call_.call();
        auto call_requester = std::move(call_requester_);
        allocator_state_->Release();
        this->~ServerCallbackUnaryImpl();  // explicitly call destructor
        ::grpc::g_core_codegen_interface->grpc_call_unref(call);
        call_requester();
      }
    }

    ServerReactor* reactor() override {
      return reactor_.load(std::memory_order_relaxed);
    }

    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata>
        meta_ops_;
    ::grpc::internal::CallbackWithSuccessTag meta_tag_;
    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                                ::grpc::internal::CallOpSendMessage,
                                ::grpc::internal::CallOpServerSendStatus>
        finish_ops_;
    ::grpc::internal::CallbackWithSuccessTag finish_tag_;

    ::grpc_impl::CallbackServerContext* const ctx_;
    ::grpc::internal::Call call_;
    ::grpc::experimental::MessageHolder<RequestType, ResponseType>* const
        allocator_state_;
    std::function<void()> call_requester_;
    // reactor_ can always be loaded/stored with relaxed memory ordering because
    // its value is only set once, independently of other data in the object,
    // and the loads that use it will always actually come provably later even
    // though they are from different threads since they are triggered by
    // actions initiated only by the setting up of the reactor_ variable. In
    // a sense, it's a delayed "const": it gets its value from the SetupReactor
    // method (not the constructor, so it's not a true const), but it doesn't
    // change after that and it only gets used by actions caused, directly or
    // indirectly, by that setup. This comment also applies to the reactor_
    // variables of the other streaming objects in this file.
    std::atomic<ServerUnaryReactor*> reactor_;
    // callbacks_outstanding_ follows a refcount pattern
    std::atomic<intptr_t> callbacks_outstanding_{
        3};  // reserve for start, Finish, and CompletionOp
  };
};

template <class RequestType, class ResponseType>
class CallbackClientStreamingHandler : public ::grpc::internal::MethodHandler {
 public:
  explicit CallbackClientStreamingHandler(
      std::function<ServerReadReactor<RequestType>*(
          ::grpc_impl::CallbackServerContext*, ResponseType*)>
          get_reactor)
      : get_reactor_(std::move(get_reactor)) {}
  void RunHandler(const HandlerParameter& param) final {
    // Arena allocate a reader structure (that includes response)
    ::grpc::g_core_codegen_interface->grpc_call_ref(param.call->call());

    auto* reader = new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
        param.call->call(), sizeof(ServerCallbackReaderImpl)))
        ServerCallbackReaderImpl(
            static_cast<::grpc_impl::CallbackServerContext*>(
                param.server_context),
            param.call, std::move(param.call_requester));
    param.server_context->BeginCompletionOp(
        param.call, [reader](bool) { reader->MaybeDone(); }, reader);

    ServerReadReactor<RequestType>* reactor = nullptr;
    if (param.status.ok()) {
      reactor = ::grpc::internal::CatchingReactorGetter<
          ServerReadReactor<RequestType>>(
          get_reactor_,
          static_cast<::grpc_impl::CallbackServerContext*>(
              param.server_context),
          reader->response());
    }

    if (reactor == nullptr) {
      // if deserialization or reactor creator failed, we need to fail the call
      reactor = new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
          param.call->call(), sizeof(UnimplementedReadReactor<RequestType>)))
          UnimplementedReadReactor<RequestType>(
              ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, ""));
    }

    reader->SetupReactor(reactor);
  }

 private:
  std::function<ServerReadReactor<RequestType>*(
      ::grpc_impl::CallbackServerContext*, ResponseType*)>
      get_reactor_;

  class ServerCallbackReaderImpl : public ServerCallbackReader<RequestType> {
   public:
    void Finish(::grpc::Status s) override {
      finish_tag_.Set(call_.call(), [this](bool) { MaybeDone(); }, &finish_ops_,
                      false);
      if (!ctx_->sent_initial_metadata_) {
        finish_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                        ctx_->initial_metadata_flags());
        if (ctx_->compression_level_set()) {
          finish_ops_.set_compression_level(ctx_->compression_level());
        }
        ctx_->sent_initial_metadata_ = true;
      }
      // The response is dropped if the status is not OK.
      if (s.ok()) {
        finish_ops_.ServerSendStatus(&ctx_->trailing_metadata_,
                                     finish_ops_.SendMessagePtr(&resp_));
      } else {
        finish_ops_.ServerSendStatus(&ctx_->trailing_metadata_, s);
      }
      finish_ops_.set_core_cq_tag(&finish_tag_);
      call_.PerformOps(&finish_ops_);
    }

    void SendInitialMetadata() override {
      GPR_CODEGEN_ASSERT(!ctx_->sent_initial_metadata_);
      this->Ref();
      meta_tag_.Set(call_.call(),
                    [this](bool ok) {
                      reactor_.load(std::memory_order_relaxed)
                          ->OnSendInitialMetadataDone(ok);
                      MaybeDone();
                    },
                    &meta_ops_, false);
      meta_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                    ctx_->initial_metadata_flags());
      if (ctx_->compression_level_set()) {
        meta_ops_.set_compression_level(ctx_->compression_level());
      }
      ctx_->sent_initial_metadata_ = true;
      meta_ops_.set_core_cq_tag(&meta_tag_);
      call_.PerformOps(&meta_ops_);
    }

    void Read(RequestType* req) override {
      this->Ref();
      read_ops_.RecvMessage(req);
      call_.PerformOps(&read_ops_);
    }

   private:
    friend class CallbackClientStreamingHandler<RequestType, ResponseType>;

    ServerCallbackReaderImpl(::grpc_impl::CallbackServerContext* ctx,
                             ::grpc::internal::Call* call,
                             std::function<void()> call_requester)
        : ctx_(ctx), call_(*call), call_requester_(std::move(call_requester)) {}

    void SetupReactor(ServerReadReactor<RequestType>* reactor) {
      reactor_.store(reactor, std::memory_order_relaxed);
      read_tag_.Set(call_.call(),
                    [this](bool ok) {
                      reactor_.load(std::memory_order_relaxed)->OnReadDone(ok);
                      MaybeDone();
                    },
                    &read_ops_, false);
      read_ops_.set_core_cq_tag(&read_tag_);
      this->BindReactor(reactor);
      this->MaybeCallOnCancel(reactor);
      this->MaybeDone();
    }

    ~ServerCallbackReaderImpl() {}

    ResponseType* response() { return &resp_; }

    void MaybeDone() override {
      if (GPR_UNLIKELY(this->Unref() == 1)) {
        reactor_.load(std::memory_order_relaxed)->OnDone();
        grpc_call* call = call_.call();
        auto call_requester = std::move(call_requester_);
        this->~ServerCallbackReaderImpl();  // explicitly call destructor
        ::grpc::g_core_codegen_interface->grpc_call_unref(call);
        call_requester();
      }
    }

    ServerReactor* reactor() override {
      return reactor_.load(std::memory_order_relaxed);
    }

    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata>
        meta_ops_;
    ::grpc::internal::CallbackWithSuccessTag meta_tag_;
    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                                ::grpc::internal::CallOpSendMessage,
                                ::grpc::internal::CallOpServerSendStatus>
        finish_ops_;
    ::grpc::internal::CallbackWithSuccessTag finish_tag_;
    ::grpc::internal::CallOpSet<
        ::grpc::internal::CallOpRecvMessage<RequestType>>
        read_ops_;
    ::grpc::internal::CallbackWithSuccessTag read_tag_;

    ::grpc_impl::CallbackServerContext* const ctx_;
    ::grpc::internal::Call call_;
    ResponseType resp_;
    std::function<void()> call_requester_;
    // The memory ordering of reactor_ follows ServerCallbackUnaryImpl.
    std::atomic<ServerReadReactor<RequestType>*> reactor_;
    // callbacks_outstanding_ follows a refcount pattern
    std::atomic<intptr_t> callbacks_outstanding_{
        3};  // reserve for OnStarted, Finish, and CompletionOp
  };
};

template <class RequestType, class ResponseType>
class CallbackServerStreamingHandler : public ::grpc::internal::MethodHandler {
 public:
  explicit CallbackServerStreamingHandler(
      std::function<ServerWriteReactor<ResponseType>*(
          ::grpc_impl::CallbackServerContext*, const RequestType*)>
          get_reactor)
      : get_reactor_(std::move(get_reactor)) {}
  void RunHandler(const HandlerParameter& param) final {
    // Arena allocate a writer structure
    ::grpc::g_core_codegen_interface->grpc_call_ref(param.call->call());

    auto* writer = new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
        param.call->call(), sizeof(ServerCallbackWriterImpl)))
        ServerCallbackWriterImpl(
            static_cast<::grpc_impl::CallbackServerContext*>(
                param.server_context),
            param.call, static_cast<RequestType*>(param.request),
            std::move(param.call_requester));
    param.server_context->BeginCompletionOp(
        param.call, [writer](bool) { writer->MaybeDone(); }, writer);

    ServerWriteReactor<ResponseType>* reactor = nullptr;
    if (param.status.ok()) {
      reactor = ::grpc::internal::CatchingReactorGetter<
          ServerWriteReactor<ResponseType>>(
          get_reactor_,
          static_cast<::grpc_impl::CallbackServerContext*>(
              param.server_context),
          writer->request());
    }
    if (reactor == nullptr) {
      // if deserialization or reactor creator failed, we need to fail the call
      reactor = new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
          param.call->call(), sizeof(UnimplementedWriteReactor<ResponseType>)))
          UnimplementedWriteReactor<ResponseType>(
              ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, ""));
    }

    writer->SetupReactor(reactor);
  }

  void* Deserialize(grpc_call* call, grpc_byte_buffer* req,
                    ::grpc::Status* status, void** /*handler_data*/) final {
    ::grpc::ByteBuffer buf;
    buf.set_buffer(req);
    auto* request =
        new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
            call, sizeof(RequestType))) RequestType();
    *status =
        ::grpc::SerializationTraits<RequestType>::Deserialize(&buf, request);
    buf.Release();
    if (status->ok()) {
      return request;
    }
    request->~RequestType();
    return nullptr;
  }

 private:
  std::function<ServerWriteReactor<ResponseType>*(
      ::grpc_impl::CallbackServerContext*, const RequestType*)>
      get_reactor_;

  class ServerCallbackWriterImpl : public ServerCallbackWriter<ResponseType> {
   public:
    void Finish(::grpc::Status s) override {
      finish_tag_.Set(call_.call(), [this](bool) { MaybeDone(); }, &finish_ops_,
                      false);
      finish_ops_.set_core_cq_tag(&finish_tag_);

      if (!ctx_->sent_initial_metadata_) {
        finish_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                        ctx_->initial_metadata_flags());
        if (ctx_->compression_level_set()) {
          finish_ops_.set_compression_level(ctx_->compression_level());
        }
        ctx_->sent_initial_metadata_ = true;
      }
      finish_ops_.ServerSendStatus(&ctx_->trailing_metadata_, s);
      call_.PerformOps(&finish_ops_);
    }

    void SendInitialMetadata() override {
      GPR_CODEGEN_ASSERT(!ctx_->sent_initial_metadata_);
      this->Ref();
      meta_tag_.Set(call_.call(),
                    [this](bool ok) {
                      reactor_.load(std::memory_order_relaxed)
                          ->OnSendInitialMetadataDone(ok);
                      MaybeDone();
                    },
                    &meta_ops_, false);
      meta_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                    ctx_->initial_metadata_flags());
      if (ctx_->compression_level_set()) {
        meta_ops_.set_compression_level(ctx_->compression_level());
      }
      ctx_->sent_initial_metadata_ = true;
      meta_ops_.set_core_cq_tag(&meta_tag_);
      call_.PerformOps(&meta_ops_);
    }

    void Write(const ResponseType* resp,
               ::grpc::WriteOptions options) override {
      this->Ref();
      if (options.is_last_message()) {
        options.set_buffer_hint();
      }
      if (!ctx_->sent_initial_metadata_) {
        write_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                       ctx_->initial_metadata_flags());
        if (ctx_->compression_level_set()) {
          write_ops_.set_compression_level(ctx_->compression_level());
        }
        ctx_->sent_initial_metadata_ = true;
      }
      // TODO(vjpai): don't assert
      GPR_CODEGEN_ASSERT(write_ops_.SendMessagePtr(resp, options).ok());
      call_.PerformOps(&write_ops_);
    }

    void WriteAndFinish(const ResponseType* resp, ::grpc::WriteOptions options,
                        ::grpc::Status s) override {
      // This combines the write into the finish callback
      // Don't send any message if the status is bad
      if (s.ok()) {
        // TODO(vjpai): don't assert
        GPR_CODEGEN_ASSERT(finish_ops_.SendMessagePtr(resp, options).ok());
      }
      Finish(std::move(s));
    }

   private:
    friend class CallbackServerStreamingHandler<RequestType, ResponseType>;

    ServerCallbackWriterImpl(::grpc_impl::CallbackServerContext* ctx,
                             ::grpc::internal::Call* call,
                             const RequestType* req,
                             std::function<void()> call_requester)
        : ctx_(ctx),
          call_(*call),
          req_(req),
          call_requester_(std::move(call_requester)) {}

    void SetupReactor(ServerWriteReactor<ResponseType>* reactor) {
      reactor_.store(reactor, std::memory_order_relaxed);
      write_tag_.Set(
          call_.call(),
          [this](bool ok) {
            reactor_.load(std::memory_order_relaxed)->OnWriteDone(ok);
            MaybeDone();
          },
          &write_ops_, false);
      write_ops_.set_core_cq_tag(&write_tag_);
      this->BindReactor(reactor);
      this->MaybeCallOnCancel(reactor);
      this->MaybeDone();
    }
    ~ServerCallbackWriterImpl() { req_->~RequestType(); }

    const RequestType* request() { return req_; }

    void MaybeDone() override {
      if (GPR_UNLIKELY(this->Unref() == 1)) {
        reactor_.load(std::memory_order_relaxed)->OnDone();
        grpc_call* call = call_.call();
        auto call_requester = std::move(call_requester_);
        this->~ServerCallbackWriterImpl();  // explicitly call destructor
        ::grpc::g_core_codegen_interface->grpc_call_unref(call);
        call_requester();
      }
    }

    ServerReactor* reactor() override {
      return reactor_.load(std::memory_order_relaxed);
    }

    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata>
        meta_ops_;
    ::grpc::internal::CallbackWithSuccessTag meta_tag_;
    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                                ::grpc::internal::CallOpSendMessage,
                                ::grpc::internal::CallOpServerSendStatus>
        finish_ops_;
    ::grpc::internal::CallbackWithSuccessTag finish_tag_;
    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                                ::grpc::internal::CallOpSendMessage>
        write_ops_;
    ::grpc::internal::CallbackWithSuccessTag write_tag_;

    ::grpc_impl::CallbackServerContext* const ctx_;
    ::grpc::internal::Call call_;
    const RequestType* req_;
    std::function<void()> call_requester_;
    // The memory ordering of reactor_ follows ServerCallbackUnaryImpl.
    std::atomic<ServerWriteReactor<ResponseType>*> reactor_;
    // callbacks_outstanding_ follows a refcount pattern
    std::atomic<intptr_t> callbacks_outstanding_{
        3};  // reserve for OnStarted, Finish, and CompletionOp
  };
};

template <class RequestType, class ResponseType>
class CallbackBidiHandler : public ::grpc::internal::MethodHandler {
 public:
  explicit CallbackBidiHandler(
      std::function<ServerBidiReactor<RequestType, ResponseType>*(
          ::grpc_impl::CallbackServerContext*)>
          get_reactor)
      : get_reactor_(std::move(get_reactor)) {}
  void RunHandler(const HandlerParameter& param) final {
    ::grpc::g_core_codegen_interface->grpc_call_ref(param.call->call());

    auto* stream = new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
        param.call->call(), sizeof(ServerCallbackReaderWriterImpl)))
        ServerCallbackReaderWriterImpl(
            static_cast<::grpc_impl::CallbackServerContext*>(
                param.server_context),
            param.call, std::move(param.call_requester));
    param.server_context->BeginCompletionOp(
        param.call, [stream](bool) { stream->MaybeDone(); }, stream);

    ServerBidiReactor<RequestType, ResponseType>* reactor = nullptr;
    if (param.status.ok()) {
      reactor = ::grpc::internal::CatchingReactorGetter<
          ServerBidiReactor<RequestType, ResponseType>>(
          get_reactor_, static_cast<::grpc_impl::CallbackServerContext*>(
                            param.server_context));
    }

    if (reactor == nullptr) {
      // if deserialization or reactor creator failed, we need to fail the call
      reactor = new (::grpc::g_core_codegen_interface->grpc_call_arena_alloc(
          param.call->call(),
          sizeof(UnimplementedBidiReactor<RequestType, ResponseType>)))
          UnimplementedBidiReactor<RequestType, ResponseType>(
              ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, ""));
    }

    stream->SetupReactor(reactor);
  }

 private:
  std::function<ServerBidiReactor<RequestType, ResponseType>*(
      ::grpc_impl::CallbackServerContext*)>
      get_reactor_;

  class ServerCallbackReaderWriterImpl
      : public ServerCallbackReaderWriter<RequestType, ResponseType> {
   public:
    void Finish(::grpc::Status s) override {
      finish_tag_.Set(call_.call(), [this](bool) { MaybeDone(); }, &finish_ops_,
                      false);
      finish_ops_.set_core_cq_tag(&finish_tag_);

      if (!ctx_->sent_initial_metadata_) {
        finish_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                        ctx_->initial_metadata_flags());
        if (ctx_->compression_level_set()) {
          finish_ops_.set_compression_level(ctx_->compression_level());
        }
        ctx_->sent_initial_metadata_ = true;
      }
      finish_ops_.ServerSendStatus(&ctx_->trailing_metadata_, s);
      call_.PerformOps(&finish_ops_);
    }

    void SendInitialMetadata() override {
      GPR_CODEGEN_ASSERT(!ctx_->sent_initial_metadata_);
      this->Ref();
      meta_tag_.Set(call_.call(),
                    [this](bool ok) {
                      reactor_.load(std::memory_order_relaxed)
                          ->OnSendInitialMetadataDone(ok);
                      MaybeDone();
                    },
                    &meta_ops_, false);
      meta_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                    ctx_->initial_metadata_flags());
      if (ctx_->compression_level_set()) {
        meta_ops_.set_compression_level(ctx_->compression_level());
      }
      ctx_->sent_initial_metadata_ = true;
      meta_ops_.set_core_cq_tag(&meta_tag_);
      call_.PerformOps(&meta_ops_);
    }

    void Write(const ResponseType* resp,
               ::grpc::WriteOptions options) override {
      this->Ref();
      if (options.is_last_message()) {
        options.set_buffer_hint();
      }
      if (!ctx_->sent_initial_metadata_) {
        write_ops_.SendInitialMetadata(&ctx_->initial_metadata_,
                                       ctx_->initial_metadata_flags());
        if (ctx_->compression_level_set()) {
          write_ops_.set_compression_level(ctx_->compression_level());
        }
        ctx_->sent_initial_metadata_ = true;
      }
      // TODO(vjpai): don't assert
      GPR_CODEGEN_ASSERT(write_ops_.SendMessagePtr(resp, options).ok());
      call_.PerformOps(&write_ops_);
    }

    void WriteAndFinish(const ResponseType* resp, ::grpc::WriteOptions options,
                        ::grpc::Status s) override {
      // Don't send any message if the status is bad
      if (s.ok()) {
        // TODO(vjpai): don't assert
        GPR_CODEGEN_ASSERT(finish_ops_.SendMessagePtr(resp, options).ok());
      }
      Finish(std::move(s));
    }

    void Read(RequestType* req) override {
      this->Ref();
      read_ops_.RecvMessage(req);
      call_.PerformOps(&read_ops_);
    }

   private:
    friend class CallbackBidiHandler<RequestType, ResponseType>;

    ServerCallbackReaderWriterImpl(::grpc_impl::CallbackServerContext* ctx,
                                   ::grpc::internal::Call* call,
                                   std::function<void()> call_requester)
        : ctx_(ctx), call_(*call), call_requester_(std::move(call_requester)) {}

    void SetupReactor(ServerBidiReactor<RequestType, ResponseType>* reactor) {
      reactor_.store(reactor, std::memory_order_relaxed);
      write_tag_.Set(
          call_.call(),
          [this](bool ok) {
            reactor_.load(std::memory_order_relaxed)->OnWriteDone(ok);
            MaybeDone();
          },
          &write_ops_, false);
      write_ops_.set_core_cq_tag(&write_tag_);
      read_tag_.Set(call_.call(),
                    [this](bool ok) {
                      reactor_.load(std::memory_order_relaxed)->OnReadDone(ok);
                      MaybeDone();
                    },
                    &read_ops_, false);
      read_ops_.set_core_cq_tag(&read_tag_);
      this->BindReactor(reactor);
      this->MaybeCallOnCancel(reactor);
      this->MaybeDone();
    }

    void MaybeDone() override {
      if (GPR_UNLIKELY(this->Unref() == 1)) {
        reactor_.load(std::memory_order_relaxed)->OnDone();
        grpc_call* call = call_.call();
        auto call_requester = std::move(call_requester_);
        this->~ServerCallbackReaderWriterImpl();  // explicitly call destructor
        ::grpc::g_core_codegen_interface->grpc_call_unref(call);
        call_requester();
      }
    }

    ServerReactor* reactor() override {
      return reactor_.load(std::memory_order_relaxed);
    }

    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata>
        meta_ops_;
    ::grpc::internal::CallbackWithSuccessTag meta_tag_;
    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                                ::grpc::internal::CallOpSendMessage,
                                ::grpc::internal::CallOpServerSendStatus>
        finish_ops_;
    ::grpc::internal::CallbackWithSuccessTag finish_tag_;
    ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                                ::grpc::internal::CallOpSendMessage>
        write_ops_;
    ::grpc::internal::CallbackWithSuccessTag write_tag_;
    ::grpc::internal::CallOpSet<
        ::grpc::internal::CallOpRecvMessage<RequestType>>
        read_ops_;
    ::grpc::internal::CallbackWithSuccessTag read_tag_;

    ::grpc_impl::CallbackServerContext* const ctx_;
    ::grpc::internal::Call call_;
    std::function<void()> call_requester_;
    // The memory ordering of reactor_ follows ServerCallbackUnaryImpl.
    std::atomic<ServerBidiReactor<RequestType, ResponseType>*> reactor_;
    // callbacks_outstanding_ follows a refcount pattern
    std::atomic<intptr_t> callbacks_outstanding_{
        3};  // reserve for OnStarted, Finish, and CompletionOp
  };
};

}  // namespace internal
}  // namespace grpc_impl

#endif  // GRPCPP_IMPL_CODEGEN_SERVER_CALLBACK_HANDLERS_H
