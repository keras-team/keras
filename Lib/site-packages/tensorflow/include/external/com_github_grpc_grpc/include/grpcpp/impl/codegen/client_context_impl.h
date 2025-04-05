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

/// A ClientContext allows the person implementing a service client to:
///
/// - Add custom metadata key-value pairs that will propagated to the server
/// side.
/// - Control call settings such as compression and authentication.
/// - Initial and trailing metadata coming from the server.
/// - Get performance metrics (ie, census).
///
/// Context settings are only relevant to the call they are invoked with, that
/// is to say, they aren't sticky. Some of these settings, such as the
/// compression options, can be made persistent at channel construction time
/// (see \a grpc::CreateCustomChannel).
///
/// \warning ClientContext instances should \em not be reused across rpcs.

#ifndef GRPCPP_IMPL_CODEGEN_CLIENT_CONTEXT_IMPL_H
#define GRPCPP_IMPL_CODEGEN_CLIENT_CONTEXT_IMPL_H

#include <map>
#include <memory>
#include <string>

#include <grpc/impl/codegen/compression_types.h>
#include <grpc/impl/codegen/propagation_bits.h>
#include <grpcpp/impl/codegen/client_interceptor.h>
#include <grpcpp/impl/codegen/config.h>
#include <grpcpp/impl/codegen/core_codegen_interface.h>
#include <grpcpp/impl/codegen/create_auth_context.h>
#include <grpcpp/impl/codegen/metadata_map.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/security/auth_context.h>
#include <grpcpp/impl/codegen/slice.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/string_ref.h>
#include <grpcpp/impl/codegen/sync.h>
#include <grpcpp/impl/codegen/time.h>

struct census_context;
struct grpc_call;

namespace grpc {

class ChannelInterface;

namespace internal {
class RpcMethod;
template <class InputMessage, class OutputMessage>
class BlockingUnaryCallImpl;
class CallOpClientRecvStatus;
class CallOpRecvInitialMetadata;
class ServerContextImpl;
}  // namespace internal

namespace testing {
class InteropClientContextInspector;
}  // namespace testing
}  // namespace grpc
namespace grpc_impl {

namespace internal {
template <class InputMessage, class OutputMessage>
class CallbackUnaryCallImpl;
template <class Request, class Response>
class ClientCallbackReaderWriterImpl;
template <class Response>
class ClientCallbackReaderImpl;
template <class Request>
class ClientCallbackWriterImpl;
class ClientCallbackUnaryImpl;
class ClientContextAccessor;
}  // namespace internal

class CallCredentials;
class Channel;
class CompletionQueue;
class ServerContext;
template <class R>
class ClientReader;
template <class W>
class ClientWriter;
template <class W, class R>
class ClientReaderWriter;
template <class R>
class ClientAsyncReader;
template <class W>
class ClientAsyncWriter;
template <class W, class R>
class ClientAsyncReaderWriter;
template <class R>
class ClientAsyncResponseReader;

class ServerContextBase;
class CallbackServerContext;

/// Options for \a ClientContext::FromServerContext specifying which traits from
/// the \a ServerContext to propagate (copy) from it into a new \a
/// ClientContext.
///
/// \see ClientContext::FromServerContext
class PropagationOptions {
 public:
  PropagationOptions() : propagate_(GRPC_PROPAGATE_DEFAULTS) {}

  PropagationOptions& enable_deadline_propagation() {
    propagate_ |= GRPC_PROPAGATE_DEADLINE;
    return *this;
  }

  PropagationOptions& disable_deadline_propagation() {
    propagate_ &= ~GRPC_PROPAGATE_DEADLINE;
    return *this;
  }

  PropagationOptions& enable_census_stats_propagation() {
    propagate_ |= GRPC_PROPAGATE_CENSUS_STATS_CONTEXT;
    return *this;
  }

  PropagationOptions& disable_census_stats_propagation() {
    propagate_ &= ~GRPC_PROPAGATE_CENSUS_STATS_CONTEXT;
    return *this;
  }

  PropagationOptions& enable_census_tracing_propagation() {
    propagate_ |= GRPC_PROPAGATE_CENSUS_TRACING_CONTEXT;
    return *this;
  }

  PropagationOptions& disable_census_tracing_propagation() {
    propagate_ &= ~GRPC_PROPAGATE_CENSUS_TRACING_CONTEXT;
    return *this;
  }

  PropagationOptions& enable_cancellation_propagation() {
    propagate_ |= GRPC_PROPAGATE_CANCELLATION;
    return *this;
  }

  PropagationOptions& disable_cancellation_propagation() {
    propagate_ &= ~GRPC_PROPAGATE_CANCELLATION;
    return *this;
  }

  uint32_t c_bitmask() const { return propagate_; }

 private:
  uint32_t propagate_;
};

/// A ClientContext allows the person implementing a service client to:
///
/// - Add custom metadata key-value pairs that will propagated to the server
///   side.
/// - Control call settings such as compression and authentication.
/// - Initial and trailing metadata coming from the server.
/// - Get performance metrics (ie, census).
///
/// Context settings are only relevant to the call they are invoked with, that
/// is to say, they aren't sticky. Some of these settings, such as the
/// compression options, can be made persistent at channel construction time
/// (see \a grpc::CreateCustomChannel).
///
/// \warning ClientContext instances should \em not be reused across rpcs.
/// \warning The ClientContext instance used for creating an rpc must remain
///          alive and valid for the lifetime of the rpc.
class ClientContext {
 public:
  ClientContext();
  ~ClientContext();

  /// Create a new \a ClientContext as a child of an incoming server call,
  /// according to \a options (\see PropagationOptions).
  ///
  /// \param server_context The source server context to use as the basis for
  /// constructing the client context.
  /// \param options The options controlling what to copy from the \a
  /// server_context.
  ///
  /// \return A newly constructed \a ClientContext instance based on \a
  /// server_context, with traits propagated (copied) according to \a options.
  static std::unique_ptr<ClientContext> FromServerContext(
      const grpc_impl::ServerContext& server_context,
      PropagationOptions options = PropagationOptions());
  static std::unique_ptr<ClientContext> FromCallbackServerContext(
      const grpc_impl::CallbackServerContext& server_context,
      PropagationOptions options = PropagationOptions());

  /// Add the (\a meta_key, \a meta_value) pair to the metadata associated with
  /// a client call. These are made available at the server side by the \a
  /// grpc::ServerContext::client_metadata() method.
  ///
  /// \warning This method should only be called before invoking the rpc.
  ///
  /// \param meta_key The metadata key. If \a meta_value is binary data, it must
  /// end in "-bin".
  /// \param meta_value The metadata value. If its value is binary, the key name
  /// must end in "-bin".
  ///
  /// Metadata must conform to the following format:
  /// Custom-Metadata -> Binary-Header / ASCII-Header
  /// Binary-Header -> {Header-Name "-bin" } {binary value}
  /// ASCII-Header -> Header-Name ASCII-Value
  /// Header-Name -> 1*( %x30-39 / %x61-7A / "_" / "-" / ".") ; 0-9 a-z _ - .
  /// ASCII-Value -> 1*( %x20-%x7E ) ; space and printable ASCII
  void AddMetadata(const grpc::string& meta_key,
                   const grpc::string& meta_value);

  /// Return a collection of initial metadata key-value pairs. Note that keys
  /// may happen more than once (ie, a \a std::multimap is returned).
  ///
  /// \warning This method should only be called after initial metadata has been
  /// received. For streaming calls, see \a
  /// ClientReaderInterface::WaitForInitialMetadata().
  ///
  /// \return A multimap of initial metadata key-value pairs from the server.
  const std::multimap<grpc::string_ref, grpc::string_ref>&
  GetServerInitialMetadata() const {
    GPR_CODEGEN_ASSERT(initial_metadata_received_);
    return *recv_initial_metadata_.map();
  }

  /// Return a collection of trailing metadata key-value pairs. Note that keys
  /// may happen more than once (ie, a \a std::multimap is returned).
  ///
  /// \warning This method is only callable once the stream has finished.
  ///
  /// \return A multimap of metadata trailing key-value pairs from the server.
  const std::multimap<grpc::string_ref, grpc::string_ref>&
  GetServerTrailingMetadata() const {
    // TODO(yangg) check finished
    return *trailing_metadata_.map();
  }

  /// Set the deadline for the client call.
  ///
  /// \warning This method should only be called before invoking the rpc.
  ///
  /// \param deadline the deadline for the client call. Units are determined by
  /// the type used. The deadline is an absolute (not relative) time.
  template <typename T>
  void set_deadline(const T& deadline) {
    grpc::TimePoint<T> deadline_tp(deadline);
    deadline_ = deadline_tp.raw_time();
  }

  /// EXPERIMENTAL: Indicate that this request is idempotent.
  /// By default, RPCs are assumed to <i>not</i> be idempotent.
  ///
  /// If true, the gRPC library assumes that it's safe to initiate
  /// this RPC multiple times.
  void set_idempotent(bool idempotent) { idempotent_ = idempotent; }

  /// EXPERIMENTAL: Set this request to be cacheable.
  /// If set, grpc is free to use the HTTP GET verb for sending the request,
  /// with the possibility of receiving a cached response.
  void set_cacheable(bool cacheable) { cacheable_ = cacheable; }

  /// EXPERIMENTAL: Trigger wait-for-ready or not on this request.
  /// See https://github.com/grpc/grpc/blob/master/doc/wait-for-ready.md.
  /// If set, if an RPC is made when a channel's connectivity state is
  /// TRANSIENT_FAILURE or CONNECTING, the call will not "fail fast",
  /// and the channel will wait until the channel is READY before making the
  /// call.
  void set_wait_for_ready(bool wait_for_ready) {
    wait_for_ready_ = wait_for_ready;
    wait_for_ready_explicitly_set_ = true;
  }

  /// DEPRECATED: Use set_wait_for_ready() instead.
  void set_fail_fast(bool fail_fast) { set_wait_for_ready(!fail_fast); }

  /// Return the deadline for the client call.
  std::chrono::system_clock::time_point deadline() const {
    return grpc::Timespec2Timepoint(deadline_);
  }

  /// Return a \a gpr_timespec representation of the client call's deadline.
  gpr_timespec raw_deadline() const { return deadline_; }

  /// Set the per call authority header (see
  /// https://tools.ietf.org/html/rfc7540#section-8.1.2.3).
  void set_authority(const grpc::string& authority) { authority_ = authority; }

  /// Return the authentication context for the associated client call.
  /// It is only valid to call this during the lifetime of the client call.
  ///
  /// \see grpc::AuthContext.
  std::shared_ptr<const grpc::AuthContext> auth_context() const {
    if (auth_context_.get() == nullptr) {
      auth_context_ = grpc::CreateAuthContext(call_);
    }
    return auth_context_;
  }

  /// Set credentials for the client call.
  ///
  /// A credentials object encapsulates all the state needed by a client to
  /// authenticate with a server and make various assertions, e.g., about the
  /// client’s identity, role, or whether it is authorized to make a particular
  /// call.
  ///
  /// It is legal to call this only before initial metadata is sent.
  ///
  /// \see  https://grpc.io/docs/guides/auth.html
  void set_credentials(
      const std::shared_ptr<grpc_impl::CallCredentials>& creds);

  /// EXPERIMENTAL debugging API
  ///
  /// Returns the credentials for the client call. This should be used only in
  /// tests and for diagnostic purposes, and should not be used by application
  /// logic.
  std::shared_ptr<grpc_impl::CallCredentials> credentials() { return creds_; }

  /// Return the compression algorithm the client call will request be used.
  /// Note that the gRPC runtime may decide to ignore this request, for example,
  /// due to resource constraints.
  grpc_compression_algorithm compression_algorithm() const {
    return compression_algorithm_;
  }

  /// Set \a algorithm to be the compression algorithm used for the client call.
  ///
  /// \param algorithm The compression algorithm used for the client call.
  void set_compression_algorithm(grpc_compression_algorithm algorithm);

  /// Flag whether the initial metadata should be \a corked
  ///
  /// If \a corked is true, then the initial metadata will be coalesced with the
  /// write of first message in the stream. As a result, any tag set for the
  /// initial metadata operation (starting a client-streaming or bidi-streaming
  /// RPC) will not actually be sent to the completion queue or delivered
  /// via Next.
  ///
  /// \param corked The flag indicating whether the initial metadata is to be
  /// corked or not.
  void set_initial_metadata_corked(bool corked) {
    initial_metadata_corked_ = corked;
  }

  /// Return the peer uri in a string.
  /// It is only valid to call this during the lifetime of the client call.
  ///
  /// \warning This value is never authenticated or subject to any security
  /// related code. It must not be used for any authentication related
  /// functionality. Instead, use auth_context.
  ///
  /// \return The call's peer URI.
  grpc::string peer() const;

  /// Sets the census context.
  /// It is only valid to call this before the client call is created. A common
  /// place of setting census context is from within the DefaultConstructor
  /// method of GlobalCallbacks.
  void set_census_context(struct census_context* ccp) { census_context_ = ccp; }

  /// Returns the census context that has been set, or nullptr if not set.
  struct census_context* census_context() const {
    return census_context_;
  }

  /// Send a best-effort out-of-band cancel on the call associated with
  /// this client context.  The call could be in any stage; e.g., if it is
  /// already finished, it may still return success.
  ///
  /// There is no guarantee the call will be cancelled.
  ///
  /// Note that TryCancel() does not change any of the tags that are pending
  /// on the completion queue. All pending tags will still be delivered
  /// (though their ok result may reflect the effect of cancellation).
  void TryCancel();

  /// Global Callbacks
  ///
  /// Can be set exactly once per application to install hooks whenever
  /// a client context is constructed and destructed.
  class GlobalCallbacks {
   public:
    virtual ~GlobalCallbacks() {}
    virtual void DefaultConstructor(ClientContext* context) = 0;
    virtual void Destructor(ClientContext* context) = 0;
  };
  static void SetGlobalCallbacks(GlobalCallbacks* callbacks);

  /// Should be used for framework-level extensions only.
  /// Applications never need to call this method.
  grpc_call* c_call() { return call_; }

  /// EXPERIMENTAL debugging API
  ///
  /// if status is not ok() for an RPC, this will return a detailed string
  /// of the gRPC Core error that led to the failure. It should not be relied
  /// upon for anything other than gaining more debug data in failure cases.
  grpc::string debug_error_string() const { return debug_error_string_; }

 private:
  // Disallow copy and assign.
  ClientContext(const ClientContext&);
  ClientContext& operator=(const ClientContext&);

  friend class ::grpc::testing::InteropClientContextInspector;
  friend class ::grpc::internal::CallOpClientRecvStatus;
  friend class ::grpc::internal::CallOpRecvInitialMetadata;
  friend class ::grpc_impl::Channel;
  template <class R>
  friend class ::grpc_impl::ClientReader;
  template <class W>
  friend class ::grpc_impl::ClientWriter;
  template <class W, class R>
  friend class ::grpc_impl::ClientReaderWriter;
  template <class R>
  friend class ::grpc_impl::ClientAsyncReader;
  template <class W>
  friend class ::grpc_impl::ClientAsyncWriter;
  template <class W, class R>
  friend class ::grpc_impl::ClientAsyncReaderWriter;
  template <class R>
  friend class ::grpc_impl::ClientAsyncResponseReader;
  template <class InputMessage, class OutputMessage>
  friend class ::grpc::internal::BlockingUnaryCallImpl;
  template <class InputMessage, class OutputMessage>
  friend class ::grpc_impl::internal::CallbackUnaryCallImpl;
  template <class Request, class Response>
  friend class ::grpc_impl::internal::ClientCallbackReaderWriterImpl;
  template <class Response>
  friend class ::grpc_impl::internal::ClientCallbackReaderImpl;
  template <class Request>
  friend class ::grpc_impl::internal::ClientCallbackWriterImpl;
  friend class ::grpc_impl::internal::ClientCallbackUnaryImpl;
  friend class ::grpc_impl::internal::ClientContextAccessor;

  // Used by friend class CallOpClientRecvStatus
  void set_debug_error_string(const grpc::string& debug_error_string) {
    debug_error_string_ = debug_error_string;
  }

  grpc_call* call() const { return call_; }
  void set_call(grpc_call* call,
                const std::shared_ptr<::grpc_impl::Channel>& channel);

  grpc::experimental::ClientRpcInfo* set_client_rpc_info(
      const char* method, grpc::internal::RpcMethod::RpcType type,
      grpc::ChannelInterface* channel,
      const std::vector<std::unique_ptr<
          grpc::experimental::ClientInterceptorFactoryInterface>>& creators,
      size_t interceptor_pos) {
    rpc_info_ = grpc::experimental::ClientRpcInfo(this, type, method, channel);
    rpc_info_.RegisterInterceptors(creators, interceptor_pos);
    return &rpc_info_;
  }

  uint32_t initial_metadata_flags() const {
    return (idempotent_ ? GRPC_INITIAL_METADATA_IDEMPOTENT_REQUEST : 0) |
           (wait_for_ready_ ? GRPC_INITIAL_METADATA_WAIT_FOR_READY : 0) |
           (cacheable_ ? GRPC_INITIAL_METADATA_CACHEABLE_REQUEST : 0) |
           (wait_for_ready_explicitly_set_
                ? GRPC_INITIAL_METADATA_WAIT_FOR_READY_EXPLICITLY_SET
                : 0) |
           (initial_metadata_corked_ ? GRPC_INITIAL_METADATA_CORKED : 0);
  }

  grpc::string authority() { return authority_; }

  void SendCancelToInterceptors();

  static std::unique_ptr<ClientContext> FromInternalServerContext(
      const grpc_impl::ServerContextBase& server_context,
      PropagationOptions options);

  bool initial_metadata_received_;
  bool wait_for_ready_;
  bool wait_for_ready_explicitly_set_;
  bool idempotent_;
  bool cacheable_;
  std::shared_ptr<::grpc_impl::Channel> channel_;
  grpc::internal::Mutex mu_;
  grpc_call* call_;
  bool call_canceled_;
  gpr_timespec deadline_;
  grpc::string authority_;
  std::shared_ptr<grpc_impl::CallCredentials> creds_;
  mutable std::shared_ptr<const grpc::AuthContext> auth_context_;
  struct census_context* census_context_;
  std::multimap<grpc::string, grpc::string> send_initial_metadata_;
  mutable grpc::internal::MetadataMap recv_initial_metadata_;
  mutable grpc::internal::MetadataMap trailing_metadata_;

  grpc_call* propagate_from_call_;
  PropagationOptions propagation_options_;

  grpc_compression_algorithm compression_algorithm_;
  bool initial_metadata_corked_;

  grpc::string debug_error_string_;

  grpc::experimental::ClientRpcInfo rpc_info_;
};

}  // namespace grpc_impl

#endif  // GRPCPP_IMPL_CODEGEN_CLIENT_CONTEXT_IMPL_H
