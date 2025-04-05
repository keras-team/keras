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

#ifndef GRPCPP_IMPL_CODEGEN_SERVER_CONTEXT_IMPL_H
#define GRPCPP_IMPL_CODEGEN_SERVER_CONTEXT_IMPL_H

#include <atomic>
#include <map>
#include <memory>
#include <vector>

#include <grpc/impl/codegen/port_platform.h>

#include <grpc/impl/codegen/compression_types.h>
#include <grpcpp/impl/codegen/call.h>
#include <grpcpp/impl/codegen/call_op_set.h>
#include <grpcpp/impl/codegen/callback_common.h>
#include <grpcpp/impl/codegen/completion_queue_tag.h>
#include <grpcpp/impl/codegen/config.h>
#include <grpcpp/impl/codegen/create_auth_context.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/metadata_map.h>
#include <grpcpp/impl/codegen/security/auth_context.h>
#include <grpcpp/impl/codegen/server_callback_impl.h>
#include <grpcpp/impl/codegen/server_interceptor.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/string_ref.h>
#include <grpcpp/impl/codegen/time.h>

struct grpc_metadata;
struct grpc_call;
struct census_context;

namespace grpc_impl {
class ClientContext;
class CompletionQueue;
class Server;
template <class W, class R>
class ServerAsyncReader;
template <class W>
class ServerAsyncWriter;
template <class W>
class ServerAsyncResponseWriter;
template <class W, class R>
class ServerAsyncReaderWriter;
template <class R>
class ServerReader;
template <class W>
class ServerWriter;

namespace internal {
template <class ServiceType, class RequestType, class ResponseType>
class BidiStreamingHandler;
template <class RequestType, class ResponseType>
class CallbackUnaryHandler;
template <class RequestType, class ResponseType>
class CallbackClientStreamingHandler;
template <class RequestType, class ResponseType>
class CallbackServerStreamingHandler;
template <class RequestType, class ResponseType>
class CallbackBidiHandler;
template <class ServiceType, class RequestType, class ResponseType>
class ClientStreamingHandler;
template <class ServiceType, class RequestType, class ResponseType>
class RpcMethodHandler;
template <class Base>
class FinishOnlyReactor;
template <class W, class R>
class ServerReaderWriterBody;
template <class ServiceType, class RequestType, class ResponseType>
class ServerStreamingHandler;
class ServerReactor;
template <class Streamer, bool WriteNeeded>
class TemplatedBidiStreamingHandler;
template <::grpc::StatusCode code>
class ErrorMethodHandler;
}  // namespace internal

}  // namespace grpc_impl
namespace grpc {
class GenericServerContext;
class ServerInterface;

#ifndef GRPC_CALLBACK_API_NONEXPERIMENTAL
namespace experimental {
#endif
class GenericCallbackServerContext;
#ifndef GRPC_CALLBACK_API_NONEXPERIMENTAL
}  // namespace experimental
#endif
namespace internal {
class Call;
}  // namespace internal

namespace testing {
class InteropServerContextInspector;
class ServerContextTestSpouse;
class DefaultReactorTestPeer;
}  // namespace testing

}  // namespace grpc

namespace grpc_impl {

/// Base class of ServerContext. Experimental until callback API is final.
class ServerContextBase {
 public:
  virtual ~ServerContextBase();

  /// Return the deadline for the server call.
  std::chrono::system_clock::time_point deadline() const {
    return ::grpc::Timespec2Timepoint(deadline_);
  }

  /// Return a \a gpr_timespec representation of the server call's deadline.
  gpr_timespec raw_deadline() const { return deadline_; }

  /// Add the (\a key, \a value) pair to the initial metadata
  /// associated with a server call. These are made available at the client side
  /// by the \a grpc::ClientContext::GetServerInitialMetadata() method.
  ///
  /// \warning This method should only be called before sending initial metadata
  /// to the client (which can happen explicitly, or implicitly when sending a
  /// a response message or status to the client).
  ///
  /// \param key The metadata key. If \a value is binary data, it must
  /// end in "-bin".
  /// \param value The metadata value. If its value is binary, the key name
  /// must end in "-bin".
  ///
  /// Metadata must conform to the following format:
  /// Custom-Metadata -> Binary-Header / ASCII-Header
  /// Binary-Header -> {Header-Name "-bin" } {binary value}
  /// ASCII-Header -> Header-Name ASCII-Value
  /// Header-Name -> 1*( %x30-39 / %x61-7A / "_" / "-" / ".") ; 0-9 a-z _ - .
  /// ASCII-Value -> 1*( %x20-%x7E ) ; space and printable ASCII
  void AddInitialMetadata(const grpc::string& key, const grpc::string& value);

  /// Add the (\a key, \a value) pair to the initial metadata
  /// associated with a server call. These are made available at the client
  /// side by the \a grpc::ClientContext::GetServerTrailingMetadata() method.
  ///
  /// \warning This method should only be called before sending trailing
  /// metadata to the client (which happens when the call is finished and a
  /// status is sent to the client).
  ///
  /// \param key The metadata key. If \a value is binary data,
  /// it must end in "-bin".
  /// \param value The metadata value. If its value is binary, the key name
  /// must end in "-bin".
  ///
  /// Metadata must conform to the following format:
  /// Custom-Metadata -> Binary-Header / ASCII-Header
  /// Binary-Header -> {Header-Name "-bin" } {binary value}
  /// ASCII-Header -> Header-Name ASCII-Value
  /// Header-Name -> 1*( %x30-39 / %x61-7A / "_" / "-" / ".") ; 0-9 a-z _ - .
  /// ASCII-Value -> 1*( %x20-%x7E ) ; space and printable ASCII
  void AddTrailingMetadata(const grpc::string& key, const grpc::string& value);

  /// IsCancelled is always safe to call when using sync or callback API.
  /// When using async API, it is only safe to call IsCancelled after
  /// the AsyncNotifyWhenDone tag has been delivered. Thread-safe.
  bool IsCancelled() const;

  /// Cancel the Call from the server. This is a best-effort API and
  /// depending on when it is called, the RPC may still appear successful to
  /// the client.
  /// For example, if TryCancel() is called on a separate thread, it might race
  /// with the server handler which might return success to the client before
  /// TryCancel() was even started by the thread.
  ///
  /// It is the caller's responsibility to prevent such races and ensure that if
  /// TryCancel() is called, the serverhandler must return Status::CANCELLED.
  /// The only exception is that if the serverhandler is already returning an
  /// error status code, it is ok to not return Status::CANCELLED even if
  /// TryCancel() was called.
  ///
  /// Note that TryCancel() does not change any of the tags that are pending
  /// on the completion queue. All pending tags will still be delivered
  /// (though their ok result may reflect the effect of cancellation).
  void TryCancel() const;

  /// Return a collection of initial metadata key-value pairs sent from the
  /// client. Note that keys may happen more than
  /// once (ie, a \a std::multimap is returned).
  ///
  /// It is safe to use this method after initial metadata has been received,
  /// Calls always begin with the client sending initial metadata, so this is
  /// safe to access as soon as the call has begun on the server side.
  ///
  /// \return A multimap of initial metadata key-value pairs from the server.
  const std::multimap<grpc::string_ref, grpc::string_ref>& client_metadata()
      const {
    return *client_metadata_.map();
  }

  /// Return the compression algorithm to be used by the server call.
  grpc_compression_level compression_level() const {
    return compression_level_;
  }

  /// Set \a level to be the compression level used for the server call.
  ///
  /// \param level The compression level used for the server call.
  void set_compression_level(grpc_compression_level level) {
    compression_level_set_ = true;
    compression_level_ = level;
  }

  /// Return a bool indicating whether the compression level for this call
  /// has been set (either implicitly or through a previous call to
  /// \a set_compression_level.
  bool compression_level_set() const { return compression_level_set_; }

  /// Return the compression algorithm the server call will request be used.
  /// Note that the gRPC runtime may decide to ignore this request, for example,
  /// due to resource constraints, or if the server is aware the client doesn't
  /// support the requested algorithm.
  grpc_compression_algorithm compression_algorithm() const {
    return compression_algorithm_;
  }
  /// Set \a algorithm to be the compression algorithm used for the server call.
  ///
  /// \param algorithm The compression algorithm used for the server call.
  void set_compression_algorithm(grpc_compression_algorithm algorithm);

  /// Set the serialized load reporting costs in \a cost_data for the call.
  void SetLoadReportingCosts(const std::vector<grpc::string>& cost_data);

  /// Return the authentication context for this server call.
  ///
  /// \see grpc::AuthContext.
  std::shared_ptr<const ::grpc::AuthContext> auth_context() const {
    if (auth_context_.get() == nullptr) {
      auth_context_ = ::grpc::CreateAuthContext(call_);
    }
    return auth_context_;
  }

  /// Return the peer uri in a string.
  /// WARNING: this value is never authenticated or subject to any security
  /// related code. It must not be used for any authentication related
  /// functionality. Instead, use auth_context.
  grpc::string peer() const;

  /// Get the census context associated with this server call.
  const struct census_context* census_context() const;

  /// Should be used for framework-level extensions only.
  /// Applications never need to call this method.
  grpc_call* c_call() { return call_; }

 protected:
  /// Async only. Has to be called before the rpc starts.
  /// Returns the tag in completion queue when the rpc finishes.
  /// IsCancelled() can then be called to check whether the rpc was cancelled.
  /// TODO(vjpai): Fix this so that the tag is returned even if the call never
  /// starts (https://github.com/grpc/grpc/issues/10136).
  void AsyncNotifyWhenDone(void* tag) {
    has_notify_when_done_tag_ = true;
    async_notify_when_done_tag_ = tag;
  }

  /// NOTE: This is an API for advanced users who need custom allocators.
  /// Get and maybe mutate the allocator state associated with the current RPC.
  /// Currently only applicable for callback unary RPC methods.
  /// WARNING: This is experimental API and could be changed or removed.
  ::grpc::experimental::RpcAllocatorState* GetRpcAllocatorState() {
    return message_allocator_state_;
  }

  /// Get a library-owned default unary reactor for use in minimal reaction
  /// cases. This supports typical unary RPC usage of providing a response and
  /// status. It supports immediate Finish (finish from within the method
  /// handler) or delayed Finish (finish called after the method handler
  /// invocation). It does not support reacting to cancellation or completion,
  /// or early sending of initial metadata. Since this is a library-owned
  /// reactor, it should not be delete'd or freed in any way. This is more
  /// efficient than creating a user-owned reactor both because of avoiding an
  /// allocation and because its minimal reactions are optimized using a core
  /// surface flag that allows their reactions to run inline without any
  /// thread-hop.
  ///
  /// This method should not be called more than once or called after return
  /// from the method handler.
  ///
  /// WARNING: This is experimental API and could be changed or removed.
  ::grpc_impl::ServerUnaryReactor* DefaultReactor() {
    auto reactor = &default_reactor_;
    default_reactor_used_.store(true, std::memory_order_relaxed);
    return reactor;
  }

  /// Constructors for use by derived classes
  ServerContextBase();
  ServerContextBase(gpr_timespec deadline, grpc_metadata_array* arr);

 private:
  friend class ::grpc::testing::InteropServerContextInspector;
  friend class ::grpc::testing::ServerContextTestSpouse;
  friend class ::grpc::testing::DefaultReactorTestPeer;
  friend class ::grpc::ServerInterface;
  friend class ::grpc_impl::Server;
  template <class W, class R>
  friend class ::grpc_impl::ServerAsyncReader;
  template <class W>
  friend class ::grpc_impl::ServerAsyncWriter;
  template <class W>
  friend class ::grpc_impl::ServerAsyncResponseWriter;
  template <class W, class R>
  friend class ::grpc_impl::ServerAsyncReaderWriter;
  template <class R>
  friend class ::grpc_impl::ServerReader;
  template <class W>
  friend class ::grpc_impl::ServerWriter;
  template <class W, class R>
  friend class ::grpc_impl::internal::ServerReaderWriterBody;
  template <class ServiceType, class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::RpcMethodHandler;
  template <class ServiceType, class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::ClientStreamingHandler;
  template <class ServiceType, class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::ServerStreamingHandler;
  template <class Streamer, bool WriteNeeded>
  friend class ::grpc_impl::internal::TemplatedBidiStreamingHandler;
  template <class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::CallbackUnaryHandler;
  template <class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::CallbackClientStreamingHandler;
  template <class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::CallbackServerStreamingHandler;
  template <class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::CallbackBidiHandler;
  template <::grpc::StatusCode code>
  friend class ::grpc_impl::internal::ErrorMethodHandler;
  template <class Base>
  friend class ::grpc_impl::internal::FinishOnlyReactor;
  friend class ::grpc_impl::ClientContext;
  friend class ::grpc::GenericServerContext;
#ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
  friend class ::grpc::GenericCallbackServerContext;
#else
  friend class ::grpc::experimental::GenericCallbackServerContext;
#endif

  /// Prevent copying.
  ServerContextBase(const ServerContextBase&);
  ServerContextBase& operator=(const ServerContextBase&);

  class CompletionOp;

  void BeginCompletionOp(
      ::grpc::internal::Call* call, std::function<void(bool)> callback,
      ::grpc_impl::internal::ServerCallbackCall* callback_controller);
  /// Return the tag queued by BeginCompletionOp()
  ::grpc::internal::CompletionQueueTag* GetCompletionOpTag();

  void set_call(grpc_call* call) { call_ = call; }

  void BindDeadlineAndMetadata(gpr_timespec deadline, grpc_metadata_array* arr);

  void Clear();

  void Setup(gpr_timespec deadline);

  uint32_t initial_metadata_flags() const { return 0; }

  ::grpc::experimental::ServerRpcInfo* set_server_rpc_info(
      const char* method, ::grpc::internal::RpcMethod::RpcType type,
      const std::vector<std::unique_ptr<
          ::grpc::experimental::ServerInterceptorFactoryInterface>>& creators) {
    if (creators.size() != 0) {
      rpc_info_ = new ::grpc::experimental::ServerRpcInfo(this, method, type);
      rpc_info_->RegisterInterceptors(creators);
    }
    return rpc_info_;
  }

  void set_message_allocator_state(
      ::grpc::experimental::RpcAllocatorState* allocator_state) {
    message_allocator_state_ = allocator_state;
  }

  CompletionOp* completion_op_;
  bool has_notify_when_done_tag_;
  void* async_notify_when_done_tag_;
  ::grpc::internal::CallbackWithSuccessTag completion_tag_;

  gpr_timespec deadline_;
  grpc_call* call_;
  ::grpc_impl::CompletionQueue* cq_;
  bool sent_initial_metadata_;
  mutable std::shared_ptr<const ::grpc::AuthContext> auth_context_;
  mutable ::grpc::internal::MetadataMap client_metadata_;
  std::multimap<grpc::string, grpc::string> initial_metadata_;
  std::multimap<grpc::string, grpc::string> trailing_metadata_;

  bool compression_level_set_;
  grpc_compression_level compression_level_;
  grpc_compression_algorithm compression_algorithm_;

  ::grpc::internal::CallOpSet<::grpc::internal::CallOpSendInitialMetadata,
                              ::grpc::internal::CallOpSendMessage>
      pending_ops_;
  bool has_pending_ops_;

  ::grpc::experimental::ServerRpcInfo* rpc_info_;
  ::grpc::experimental::RpcAllocatorState* message_allocator_state_ = nullptr;

  class Reactor : public ServerUnaryReactor {
   public:
    void OnCancel() override {}
    void OnDone() override {}
    // Override InternalInlineable for this class since its reactions are
    // trivial and thus do not need to be run from the executor (triggering a
    // thread hop). This should only be used by internal reactors (thus the
    // name) and not by user application code.
    bool InternalInlineable() override { return true; }
  };

  void SetupTestDefaultReactor(std::function<void(::grpc::Status)> func) {
    test_unary_.reset(new TestServerCallbackUnary(this, std::move(func)));
  }
  bool test_status_set() const {
    return (test_unary_ != nullptr) && test_unary_->status_set();
  }
  ::grpc::Status test_status() const { return test_unary_->status(); }

  class TestServerCallbackUnary : public ::grpc_impl::ServerCallbackUnary {
   public:
    TestServerCallbackUnary(ServerContextBase* ctx,
                            std::function<void(::grpc::Status)> func)
        : reactor_(&ctx->default_reactor_), func_(std::move(func)) {
      this->BindReactor(reactor_);
    }
    void Finish(::grpc::Status s) override {
      status_ = s;
      func_(std::move(s));
      status_set_.store(true, std::memory_order_release);
    }
    void SendInitialMetadata() override {}

    bool status_set() const {
      return status_set_.load(std::memory_order_acquire);
    }
    ::grpc::Status status() const { return status_; }

   private:
    void MaybeDone() override {}
    ::grpc_impl::internal::ServerReactor* reactor() override {
      return reactor_;
    }

    ::grpc_impl::ServerUnaryReactor* const reactor_;
    std::atomic_bool status_set_{false};
    ::grpc::Status status_;
    const std::function<void(::grpc::Status s)> func_;
  };

  Reactor default_reactor_;
  std::atomic_bool default_reactor_used_{false};
  std::unique_ptr<TestServerCallbackUnary> test_unary_;
};

/// A ServerContext or CallbackServerContext allows the code implementing a
/// service handler to:
///
/// - Add custom initial and trailing metadata key-value pairs that will
///   propagated to the client side.
/// - Control call settings such as compression and authentication.
/// - Access metadata coming from the client.
/// - Get performance metrics (ie, census).
///
/// Context settings are only relevant to the call handler they are supplied to,
/// that is to say, they aren't sticky across multiple calls. Some of these
/// settings, such as the compression options, can be made persistent at server
/// construction time by specifying the appropriate \a ChannelArguments
/// to a \a grpc::ServerBuilder, via \a ServerBuilder::AddChannelArgument.
///
/// \warning ServerContext instances should \em not be reused across rpcs.
class ServerContext : public ServerContextBase {
 public:
  ServerContext() {}  // for async calls

  using ServerContextBase::AddInitialMetadata;
  using ServerContextBase::AddTrailingMetadata;
  using ServerContextBase::IsCancelled;
  using ServerContextBase::SetLoadReportingCosts;
  using ServerContextBase::TryCancel;
  using ServerContextBase::auth_context;
  using ServerContextBase::c_call;
  using ServerContextBase::census_context;
  using ServerContextBase::client_metadata;
  using ServerContextBase::compression_algorithm;
  using ServerContextBase::compression_level;
  using ServerContextBase::compression_level_set;
  using ServerContextBase::deadline;
  using ServerContextBase::peer;
  using ServerContextBase::raw_deadline;
  using ServerContextBase::set_compression_algorithm;
  using ServerContextBase::set_compression_level;

  // Sync/CQ-based Async ServerContext only
  using ServerContextBase::AsyncNotifyWhenDone;

 private:
  // Constructor for internal use by server only
  friend class ::grpc_impl::Server;
  ServerContext(gpr_timespec deadline, grpc_metadata_array* arr)
      : ServerContextBase(deadline, arr) {}

  // CallbackServerContext only
  using ServerContextBase::DefaultReactor;
  using ServerContextBase::GetRpcAllocatorState;

  /// Prevent copying.
  ServerContext(const ServerContext&) = delete;
  ServerContext& operator=(const ServerContext&) = delete;
};

class CallbackServerContext : public ServerContextBase {
 public:
  /// Public constructors are for direct use only by mocking tests. In practice,
  /// these objects will be owned by the library.
  CallbackServerContext() {}

  using ServerContextBase::AddInitialMetadata;
  using ServerContextBase::AddTrailingMetadata;
  using ServerContextBase::IsCancelled;
  using ServerContextBase::SetLoadReportingCosts;
  using ServerContextBase::TryCancel;
  using ServerContextBase::auth_context;
  using ServerContextBase::c_call;
  using ServerContextBase::census_context;
  using ServerContextBase::client_metadata;
  using ServerContextBase::compression_algorithm;
  using ServerContextBase::compression_level;
  using ServerContextBase::compression_level_set;
  using ServerContextBase::deadline;
  using ServerContextBase::peer;
  using ServerContextBase::raw_deadline;
  using ServerContextBase::set_compression_algorithm;
  using ServerContextBase::set_compression_level;

  // CallbackServerContext only
  using ServerContextBase::DefaultReactor;
  using ServerContextBase::GetRpcAllocatorState;

 private:
  // Sync/CQ-based Async ServerContext only
  using ServerContextBase::AsyncNotifyWhenDone;

  /// Prevent copying.
  CallbackServerContext(const CallbackServerContext&) = delete;
  CallbackServerContext& operator=(const CallbackServerContext&) = delete;
};

}  // namespace grpc_impl

static_assert(std::is_base_of<::grpc_impl::ServerContextBase,
                              ::grpc_impl::ServerContext>::value,
              "improper base class");
static_assert(std::is_base_of<::grpc_impl::ServerContextBase,
                              ::grpc_impl::CallbackServerContext>::value,
              "improper base class");
static_assert(sizeof(::grpc_impl::ServerContextBase) ==
                  sizeof(::grpc_impl::ServerContext),
              "wrong size");
static_assert(sizeof(::grpc_impl::ServerContextBase) ==
                  sizeof(::grpc_impl::CallbackServerContext),
              "wrong size");

#endif  // GRPCPP_IMPL_CODEGEN_SERVER_CONTEXT_IMPL_H
