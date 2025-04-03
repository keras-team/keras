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

#ifndef GRPCPP_IMPL_CODEGEN_CALL_OP_SET_H
#define GRPCPP_IMPL_CODEGEN_CALL_OP_SET_H

#include <cstring>
#include <map>
#include <memory>

#include <grpc/impl/codegen/compression_types.h>
#include <grpc/impl/codegen/grpc_types.h>
#include <grpcpp/impl/codegen/byte_buffer.h>
#include <grpcpp/impl/codegen/call.h>
#include <grpcpp/impl/codegen/call_hook.h>
#include <grpcpp/impl/codegen/call_op_set_interface.h>
#include <grpcpp/impl/codegen/client_context_impl.h>
#include <grpcpp/impl/codegen/completion_queue_impl.h>
#include <grpcpp/impl/codegen/completion_queue_tag.h>
#include <grpcpp/impl/codegen/config.h>
#include <grpcpp/impl/codegen/core_codegen_interface.h>
#include <grpcpp/impl/codegen/intercepted_channel.h>
#include <grpcpp/impl/codegen/interceptor_common.h>
#include <grpcpp/impl/codegen/serialization_traits.h>
#include <grpcpp/impl/codegen/slice.h>
#include <grpcpp/impl/codegen/string_ref.h>

namespace grpc {

extern CoreCodegenInterface* g_core_codegen_interface;

namespace internal {
class Call;
class CallHook;

// TODO(yangg) if the map is changed before we send, the pointers will be a
// mess. Make sure it does not happen.
inline grpc_metadata* FillMetadataArray(
    const std::multimap<grpc::string, grpc::string>& metadata,
    size_t* metadata_count, const grpc::string& optional_error_details) {
  *metadata_count = metadata.size() + (optional_error_details.empty() ? 0 : 1);
  if (*metadata_count == 0) {
    return nullptr;
  }
  grpc_metadata* metadata_array =
      (grpc_metadata*)(g_core_codegen_interface->gpr_malloc(
          (*metadata_count) * sizeof(grpc_metadata)));
  size_t i = 0;
  for (auto iter = metadata.cbegin(); iter != metadata.cend(); ++iter, ++i) {
    metadata_array[i].key = SliceReferencingString(iter->first);
    metadata_array[i].value = SliceReferencingString(iter->second);
  }
  if (!optional_error_details.empty()) {
    metadata_array[i].key =
        g_core_codegen_interface->grpc_slice_from_static_buffer(
            kBinaryErrorDetailsKey, sizeof(kBinaryErrorDetailsKey) - 1);
    metadata_array[i].value = SliceReferencingString(optional_error_details);
  }
  return metadata_array;
}
}  // namespace internal

/// Per-message write options.
class WriteOptions {
 public:
  WriteOptions() : flags_(0), last_message_(false) {}
  WriteOptions(const WriteOptions& other)
      : flags_(other.flags_), last_message_(other.last_message_) {}

  /// Default assignment operator
  WriteOptions& operator=(const WriteOptions& other) = default;

  /// Clear all flags.
  inline void Clear() { flags_ = 0; }

  /// Returns raw flags bitset.
  inline uint32_t flags() const { return flags_; }

  /// Sets flag for the disabling of compression for the next message write.
  ///
  /// \sa GRPC_WRITE_NO_COMPRESS
  inline WriteOptions& set_no_compression() {
    SetBit(GRPC_WRITE_NO_COMPRESS);
    return *this;
  }

  /// Clears flag for the disabling of compression for the next message write.
  ///
  /// \sa GRPC_WRITE_NO_COMPRESS
  inline WriteOptions& clear_no_compression() {
    ClearBit(GRPC_WRITE_NO_COMPRESS);
    return *this;
  }

  /// Get value for the flag indicating whether compression for the next
  /// message write is forcefully disabled.
  ///
  /// \sa GRPC_WRITE_NO_COMPRESS
  inline bool get_no_compression() const {
    return GetBit(GRPC_WRITE_NO_COMPRESS);
  }

  /// Sets flag indicating that the write may be buffered and need not go out on
  /// the wire immediately.
  ///
  /// \sa GRPC_WRITE_BUFFER_HINT
  inline WriteOptions& set_buffer_hint() {
    SetBit(GRPC_WRITE_BUFFER_HINT);
    return *this;
  }

  /// Clears flag indicating that the write may be buffered and need not go out
  /// on the wire immediately.
  ///
  /// \sa GRPC_WRITE_BUFFER_HINT
  inline WriteOptions& clear_buffer_hint() {
    ClearBit(GRPC_WRITE_BUFFER_HINT);
    return *this;
  }

  /// Get value for the flag indicating that the write may be buffered and need
  /// not go out on the wire immediately.
  ///
  /// \sa GRPC_WRITE_BUFFER_HINT
  inline bool get_buffer_hint() const { return GetBit(GRPC_WRITE_BUFFER_HINT); }

  /// corked bit: aliases set_buffer_hint currently, with the intent that
  /// set_buffer_hint will be removed in the future
  inline WriteOptions& set_corked() {
    SetBit(GRPC_WRITE_BUFFER_HINT);
    return *this;
  }

  inline WriteOptions& clear_corked() {
    ClearBit(GRPC_WRITE_BUFFER_HINT);
    return *this;
  }

  inline bool is_corked() const { return GetBit(GRPC_WRITE_BUFFER_HINT); }

  /// last-message bit: indicates this is the last message in a stream
  /// client-side:  makes Write the equivalent of performing Write, WritesDone
  /// in a single step
  /// server-side:  hold the Write until the service handler returns (sync api)
  /// or until Finish is called (async api)
  inline WriteOptions& set_last_message() {
    last_message_ = true;
    return *this;
  }

  /// Clears flag indicating that this is the last message in a stream,
  /// disabling coalescing.
  inline WriteOptions& clear_last_message() {
    last_message_ = false;
    return *this;
  }

  /// Guarantee that all bytes have been written to the socket before completing
  /// this write (usually writes are completed when they pass flow control).
  inline WriteOptions& set_write_through() {
    SetBit(GRPC_WRITE_THROUGH);
    return *this;
  }

  inline bool is_write_through() const { return GetBit(GRPC_WRITE_THROUGH); }

  /// Get value for the flag indicating that this is the last message, and
  /// should be coalesced with trailing metadata.
  ///
  /// \sa GRPC_WRITE_LAST_MESSAGE
  bool is_last_message() const { return last_message_; }

 private:
  void SetBit(const uint32_t mask) { flags_ |= mask; }

  void ClearBit(const uint32_t mask) { flags_ &= ~mask; }

  bool GetBit(const uint32_t mask) const { return (flags_ & mask) != 0; }

  uint32_t flags_;
  bool last_message_;
};

namespace internal {

/// Default argument for CallOpSet. I is unused by the class, but can be
/// used for generating multiple names for the same thing.
template <int I>
class CallNoOp {
 protected:
  void AddOp(grpc_op* /*ops*/, size_t* /*nops*/) {}
  void FinishOp(bool* /*status*/) {}
  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* /*interceptor_methods*/) {}
  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* /*interceptor_methods*/) {}
  void SetHijackingState(InterceptorBatchMethodsImpl* /*interceptor_methods*/) {
  }
};

class CallOpSendInitialMetadata {
 public:
  CallOpSendInitialMetadata() : send_(false) {
    maybe_compression_level_.is_set = false;
  }

  void SendInitialMetadata(std::multimap<grpc::string, grpc::string>* metadata,
                           uint32_t flags) {
    maybe_compression_level_.is_set = false;
    send_ = true;
    flags_ = flags;
    metadata_map_ = metadata;
  }

  void set_compression_level(grpc_compression_level level) {
    maybe_compression_level_.is_set = true;
    maybe_compression_level_.level = level;
  }

 protected:
  void AddOp(grpc_op* ops, size_t* nops) {
    if (!send_ || hijacked_) return;
    grpc_op* op = &ops[(*nops)++];
    op->op = GRPC_OP_SEND_INITIAL_METADATA;
    op->flags = flags_;
    op->reserved = NULL;
    initial_metadata_ =
        FillMetadataArray(*metadata_map_, &initial_metadata_count_, "");
    op->data.send_initial_metadata.count = initial_metadata_count_;
    op->data.send_initial_metadata.metadata = initial_metadata_;
    op->data.send_initial_metadata.maybe_compression_level.is_set =
        maybe_compression_level_.is_set;
    if (maybe_compression_level_.is_set) {
      op->data.send_initial_metadata.maybe_compression_level.level =
          maybe_compression_level_.level;
    }
  }
  void FinishOp(bool* /*status*/) {
    if (!send_ || hijacked_) return;
    g_core_codegen_interface->gpr_free(initial_metadata_);
    send_ = false;
  }

  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (!send_) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::PRE_SEND_INITIAL_METADATA);
    interceptor_methods->SetSendInitialMetadata(metadata_map_);
  }

  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* /*interceptor_methods*/) {}

  void SetHijackingState(InterceptorBatchMethodsImpl* /*interceptor_methods*/) {
    hijacked_ = true;
  }

  bool hijacked_ = false;
  bool send_;
  uint32_t flags_;
  size_t initial_metadata_count_;
  std::multimap<grpc::string, grpc::string>* metadata_map_;
  grpc_metadata* initial_metadata_;
  struct {
    bool is_set;
    grpc_compression_level level;
  } maybe_compression_level_;
};

class CallOpSendMessage {
 public:
  CallOpSendMessage() : send_buf_() {}

  /// Send \a message using \a options for the write. The \a options are cleared
  /// after use.
  template <class M>
  Status SendMessage(const M& message,
                     WriteOptions options) GRPC_MUST_USE_RESULT;

  template <class M>
  Status SendMessage(const M& message) GRPC_MUST_USE_RESULT;

  /// Send \a message using \a options for the write. The \a options are cleared
  /// after use. This form of SendMessage allows gRPC to reference \a message
  /// beyond the lifetime of SendMessage.
  template <class M>
  Status SendMessagePtr(const M* message,
                        WriteOptions options) GRPC_MUST_USE_RESULT;

  /// This form of SendMessage allows gRPC to reference \a message beyond the
  /// lifetime of SendMessage.
  template <class M>
  Status SendMessagePtr(const M* message) GRPC_MUST_USE_RESULT;

 protected:
  void AddOp(grpc_op* ops, size_t* nops) {
    if (msg_ == nullptr && !send_buf_.Valid()) return;
    if (hijacked_) {
      serializer_ = nullptr;
      return;
    }
    if (msg_ != nullptr) {
      GPR_CODEGEN_ASSERT(serializer_(msg_).ok());
    }
    serializer_ = nullptr;
    grpc_op* op = &ops[(*nops)++];
    op->op = GRPC_OP_SEND_MESSAGE;
    op->flags = write_options_.flags();
    op->reserved = NULL;
    op->data.send_message.send_message = send_buf_.c_buffer();
    // Flags are per-message: clear them after use.
    write_options_.Clear();
  }
  void FinishOp(bool* status) {
    if (msg_ == nullptr && !send_buf_.Valid()) return;
    if (hijacked_ && failed_send_) {
      // Hijacking interceptor failed this Op
      *status = false;
    } else if (!*status) {
      // This Op was passed down to core and the Op failed
      failed_send_ = true;
    }
  }

  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (msg_ == nullptr && !send_buf_.Valid()) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::PRE_SEND_MESSAGE);
    interceptor_methods->SetSendMessage(&send_buf_, &msg_, &failed_send_,
                                        serializer_);
  }

  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (msg_ != nullptr || send_buf_.Valid()) {
      interceptor_methods->AddInterceptionHookPoint(
          experimental::InterceptionHookPoints::POST_SEND_MESSAGE);
    }
    send_buf_.Clear();
    msg_ = nullptr;
    // The contents of the SendMessage value that was previously set
    // has had its references stolen by core's operations
    interceptor_methods->SetSendMessage(nullptr, nullptr, &failed_send_,
                                        nullptr);
  }

  void SetHijackingState(InterceptorBatchMethodsImpl* /*interceptor_methods*/) {
    hijacked_ = true;
  }

 private:
  const void* msg_ = nullptr;  // The original non-serialized message
  bool hijacked_ = false;
  bool failed_send_ = false;
  ByteBuffer send_buf_;
  WriteOptions write_options_;
  std::function<Status(const void*)> serializer_;
};

template <class M>
Status CallOpSendMessage::SendMessage(const M& message, WriteOptions options) {
  write_options_ = options;
  serializer_ = [this](const void* message) {
    bool own_buf;
    send_buf_.Clear();
    // TODO(vjpai): Remove the void below when possible
    // The void in the template parameter below should not be needed
    // (since it should be implicit) but is needed due to an observed
    // difference in behavior between clang and gcc for certain internal users
    Status result = SerializationTraits<M, void>::Serialize(
        *static_cast<const M*>(message), send_buf_.bbuf_ptr(), &own_buf);
    if (!own_buf) {
      send_buf_.Duplicate();
    }
    return result;
  };
  // Serialize immediately only if we do not have access to the message pointer
  if (msg_ == nullptr) {
    Status result = serializer_(&message);
    serializer_ = nullptr;
    return result;
  }
  return Status();
}

template <class M>
Status CallOpSendMessage::SendMessage(const M& message) {
  return SendMessage(message, WriteOptions());
}

template <class M>
Status CallOpSendMessage::SendMessagePtr(const M* message,
                                         WriteOptions options) {
  msg_ = message;
  return SendMessage(*message, options);
}

template <class M>
Status CallOpSendMessage::SendMessagePtr(const M* message) {
  msg_ = message;
  return SendMessage(*message, WriteOptions());
}

template <class R>
class CallOpRecvMessage {
 public:
  CallOpRecvMessage()
      : got_message(false),
        message_(nullptr),
        allow_not_getting_message_(false) {}

  void RecvMessage(R* message) { message_ = message; }

  // Do not change status if no message is received.
  void AllowNoMessage() { allow_not_getting_message_ = true; }

  bool got_message;

 protected:
  void AddOp(grpc_op* ops, size_t* nops) {
    if (message_ == nullptr || hijacked_) return;
    grpc_op* op = &ops[(*nops)++];
    op->op = GRPC_OP_RECV_MESSAGE;
    op->flags = 0;
    op->reserved = NULL;
    op->data.recv_message.recv_message = recv_buf_.c_buffer_ptr();
  }

  void FinishOp(bool* status) {
    if (message_ == nullptr || hijacked_) return;
    if (recv_buf_.Valid()) {
      if (*status) {
        got_message = *status =
            SerializationTraits<R>::Deserialize(recv_buf_.bbuf_ptr(), message_)
                .ok();
        recv_buf_.Release();
      } else {
        got_message = false;
        recv_buf_.Clear();
      }
    } else {
      got_message = false;
      if (!allow_not_getting_message_) {
        *status = false;
      }
    }
  }

  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (message_ == nullptr) return;
    interceptor_methods->SetRecvMessage(message_, &got_message);
  }

  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (message_ == nullptr) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::POST_RECV_MESSAGE);
    if (!got_message) interceptor_methods->SetRecvMessage(nullptr, nullptr);
  }
  void SetHijackingState(InterceptorBatchMethodsImpl* interceptor_methods) {
    hijacked_ = true;
    if (message_ == nullptr) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::PRE_RECV_MESSAGE);
    got_message = true;
  }

 private:
  R* message_;
  ByteBuffer recv_buf_;
  bool allow_not_getting_message_;
  bool hijacked_ = false;
};

class DeserializeFunc {
 public:
  virtual Status Deserialize(ByteBuffer* buf) = 0;
  virtual ~DeserializeFunc() {}
};

template <class R>
class DeserializeFuncType final : public DeserializeFunc {
 public:
  DeserializeFuncType(R* message) : message_(message) {}
  Status Deserialize(ByteBuffer* buf) override {
    return SerializationTraits<R>::Deserialize(buf->bbuf_ptr(), message_);
  }

  ~DeserializeFuncType() override {}

 private:
  R* message_;  // Not a managed pointer because management is external to this
};

class CallOpGenericRecvMessage {
 public:
  CallOpGenericRecvMessage()
      : got_message(false), allow_not_getting_message_(false) {}

  template <class R>
  void RecvMessage(R* message) {
    // Use an explicit base class pointer to avoid resolution error in the
    // following unique_ptr::reset for some old implementations.
    DeserializeFunc* func = new DeserializeFuncType<R>(message);
    deserialize_.reset(func);
    message_ = message;
  }

  // Do not change status if no message is received.
  void AllowNoMessage() { allow_not_getting_message_ = true; }

  bool got_message;

 protected:
  void AddOp(grpc_op* ops, size_t* nops) {
    if (!deserialize_ || hijacked_) return;
    grpc_op* op = &ops[(*nops)++];
    op->op = GRPC_OP_RECV_MESSAGE;
    op->flags = 0;
    op->reserved = NULL;
    op->data.recv_message.recv_message = recv_buf_.c_buffer_ptr();
  }

  void FinishOp(bool* status) {
    if (!deserialize_ || hijacked_) return;
    if (recv_buf_.Valid()) {
      if (*status) {
        got_message = true;
        *status = deserialize_->Deserialize(&recv_buf_).ok();
        recv_buf_.Release();
      } else {
        got_message = false;
        recv_buf_.Clear();
      }
    } else {
      got_message = false;
      if (!allow_not_getting_message_) {
        *status = false;
      }
    }
  }

  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (!deserialize_) return;
    interceptor_methods->SetRecvMessage(message_, &got_message);
  }

  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (!deserialize_) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::POST_RECV_MESSAGE);
    if (!got_message) interceptor_methods->SetRecvMessage(nullptr, nullptr);
    deserialize_.reset();
  }
  void SetHijackingState(InterceptorBatchMethodsImpl* interceptor_methods) {
    hijacked_ = true;
    if (!deserialize_) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::PRE_RECV_MESSAGE);
    got_message = true;
  }

 private:
  void* message_;
  bool hijacked_ = false;
  std::unique_ptr<DeserializeFunc> deserialize_;
  ByteBuffer recv_buf_;
  bool allow_not_getting_message_;
};

class CallOpClientSendClose {
 public:
  CallOpClientSendClose() : send_(false) {}

  void ClientSendClose() { send_ = true; }

 protected:
  void AddOp(grpc_op* ops, size_t* nops) {
    if (!send_ || hijacked_) return;
    grpc_op* op = &ops[(*nops)++];
    op->op = GRPC_OP_SEND_CLOSE_FROM_CLIENT;
    op->flags = 0;
    op->reserved = NULL;
  }
  void FinishOp(bool* /*status*/) { send_ = false; }

  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (!send_) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::PRE_SEND_CLOSE);
  }

  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* /*interceptor_methods*/) {}

  void SetHijackingState(InterceptorBatchMethodsImpl* /*interceptor_methods*/) {
    hijacked_ = true;
  }

 private:
  bool hijacked_ = false;
  bool send_;
};

class CallOpServerSendStatus {
 public:
  CallOpServerSendStatus() : send_status_available_(false) {}

  void ServerSendStatus(
      std::multimap<grpc::string, grpc::string>* trailing_metadata,
      const Status& status) {
    send_error_details_ = status.error_details();
    metadata_map_ = trailing_metadata;
    send_status_available_ = true;
    send_status_code_ = static_cast<grpc_status_code>(status.error_code());
    send_error_message_ = status.error_message();
  }

 protected:
  void AddOp(grpc_op* ops, size_t* nops) {
    if (!send_status_available_ || hijacked_) return;
    trailing_metadata_ = FillMetadataArray(
        *metadata_map_, &trailing_metadata_count_, send_error_details_);
    grpc_op* op = &ops[(*nops)++];
    op->op = GRPC_OP_SEND_STATUS_FROM_SERVER;
    op->data.send_status_from_server.trailing_metadata_count =
        trailing_metadata_count_;
    op->data.send_status_from_server.trailing_metadata = trailing_metadata_;
    op->data.send_status_from_server.status = send_status_code_;
    error_message_slice_ = SliceReferencingString(send_error_message_);
    op->data.send_status_from_server.status_details =
        send_error_message_.empty() ? nullptr : &error_message_slice_;
    op->flags = 0;
    op->reserved = NULL;
  }

  void FinishOp(bool* /*status*/) {
    if (!send_status_available_ || hijacked_) return;
    g_core_codegen_interface->gpr_free(trailing_metadata_);
    send_status_available_ = false;
  }

  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (!send_status_available_) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::PRE_SEND_STATUS);
    interceptor_methods->SetSendTrailingMetadata(metadata_map_);
    interceptor_methods->SetSendStatus(&send_status_code_, &send_error_details_,
                                       &send_error_message_);
  }

  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* /*interceptor_methods*/) {}

  void SetHijackingState(InterceptorBatchMethodsImpl* /*interceptor_methods*/) {
    hijacked_ = true;
  }

 private:
  bool hijacked_ = false;
  bool send_status_available_;
  grpc_status_code send_status_code_;
  grpc::string send_error_details_;
  grpc::string send_error_message_;
  size_t trailing_metadata_count_;
  std::multimap<grpc::string, grpc::string>* metadata_map_;
  grpc_metadata* trailing_metadata_;
  grpc_slice error_message_slice_;
};

class CallOpRecvInitialMetadata {
 public:
  CallOpRecvInitialMetadata() : metadata_map_(nullptr) {}

  void RecvInitialMetadata(::grpc_impl::ClientContext* context) {
    context->initial_metadata_received_ = true;
    metadata_map_ = &context->recv_initial_metadata_;
  }

 protected:
  void AddOp(grpc_op* ops, size_t* nops) {
    if (metadata_map_ == nullptr || hijacked_) return;
    grpc_op* op = &ops[(*nops)++];
    op->op = GRPC_OP_RECV_INITIAL_METADATA;
    op->data.recv_initial_metadata.recv_initial_metadata = metadata_map_->arr();
    op->flags = 0;
    op->reserved = NULL;
  }

  void FinishOp(bool* /*status*/) {
    if (metadata_map_ == nullptr || hijacked_) return;
  }

  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    interceptor_methods->SetRecvInitialMetadata(metadata_map_);
  }

  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (metadata_map_ == nullptr) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::POST_RECV_INITIAL_METADATA);
    metadata_map_ = nullptr;
  }

  void SetHijackingState(InterceptorBatchMethodsImpl* interceptor_methods) {
    hijacked_ = true;
    if (metadata_map_ == nullptr) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::PRE_RECV_INITIAL_METADATA);
  }

 private:
  bool hijacked_ = false;
  MetadataMap* metadata_map_;
};

class CallOpClientRecvStatus {
 public:
  CallOpClientRecvStatus()
      : recv_status_(nullptr), debug_error_string_(nullptr) {}

  void ClientRecvStatus(::grpc_impl::ClientContext* context, Status* status) {
    client_context_ = context;
    metadata_map_ = &client_context_->trailing_metadata_;
    recv_status_ = status;
    error_message_ = g_core_codegen_interface->grpc_empty_slice();
  }

 protected:
  void AddOp(grpc_op* ops, size_t* nops) {
    if (recv_status_ == nullptr || hijacked_) return;
    grpc_op* op = &ops[(*nops)++];
    op->op = GRPC_OP_RECV_STATUS_ON_CLIENT;
    op->data.recv_status_on_client.trailing_metadata = metadata_map_->arr();
    op->data.recv_status_on_client.status = &status_code_;
    op->data.recv_status_on_client.status_details = &error_message_;
    op->data.recv_status_on_client.error_string = &debug_error_string_;
    op->flags = 0;
    op->reserved = NULL;
  }

  void FinishOp(bool* /*status*/) {
    if (recv_status_ == nullptr || hijacked_) return;
    if (static_cast<StatusCode>(status_code_) == StatusCode::OK) {
      *recv_status_ = Status();
      GPR_CODEGEN_DEBUG_ASSERT(debug_error_string_ == nullptr);
    } else {
      *recv_status_ =
          Status(static_cast<StatusCode>(status_code_),
                 GRPC_SLICE_IS_EMPTY(error_message_)
                     ? grpc::string()
                     : grpc::string(GRPC_SLICE_START_PTR(error_message_),
                                    GRPC_SLICE_END_PTR(error_message_)),
                 metadata_map_->GetBinaryErrorDetails());
      if (debug_error_string_ != nullptr) {
        client_context_->set_debug_error_string(debug_error_string_);
        g_core_codegen_interface->gpr_free((void*)debug_error_string_);
      }
    }
    // TODO(soheil): Find callers that set debug string even for status OK,
    //               and fix them.
    g_core_codegen_interface->grpc_slice_unref(error_message_);
  }

  void SetInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    interceptor_methods->SetRecvStatus(recv_status_);
    interceptor_methods->SetRecvTrailingMetadata(metadata_map_);
  }

  void SetFinishInterceptionHookPoint(
      InterceptorBatchMethodsImpl* interceptor_methods) {
    if (recv_status_ == nullptr) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::POST_RECV_STATUS);
    recv_status_ = nullptr;
  }

  void SetHijackingState(InterceptorBatchMethodsImpl* interceptor_methods) {
    hijacked_ = true;
    if (recv_status_ == nullptr) return;
    interceptor_methods->AddInterceptionHookPoint(
        experimental::InterceptionHookPoints::PRE_RECV_STATUS);
  }

 private:
  bool hijacked_ = false;
  ::grpc_impl::ClientContext* client_context_;
  MetadataMap* metadata_map_;
  Status* recv_status_;
  const char* debug_error_string_;
  grpc_status_code status_code_;
  grpc_slice error_message_;
};

template <class Op1 = CallNoOp<1>, class Op2 = CallNoOp<2>,
          class Op3 = CallNoOp<3>, class Op4 = CallNoOp<4>,
          class Op5 = CallNoOp<5>, class Op6 = CallNoOp<6>>
class CallOpSet;

/// Primary implementation of CallOpSetInterface.
/// Since we cannot use variadic templates, we declare slots up to
/// the maximum count of ops we'll need in a set. We leverage the
/// empty base class optimization to slim this class (especially
/// when there are many unused slots used). To avoid duplicate base classes,
/// the template parmeter for CallNoOp is varied by argument position.
template <class Op1, class Op2, class Op3, class Op4, class Op5, class Op6>
class CallOpSet : public CallOpSetInterface,
                  public Op1,
                  public Op2,
                  public Op3,
                  public Op4,
                  public Op5,
                  public Op6 {
 public:
  CallOpSet() : core_cq_tag_(this), return_tag_(this) {}
  // The copy constructor and assignment operator reset the value of
  // core_cq_tag_, return_tag_, done_intercepting_ and interceptor_methods_
  // since those are only meaningful on a specific object, not across objects.
  CallOpSet(const CallOpSet& other)
      : core_cq_tag_(this),
        return_tag_(this),
        call_(other.call_),
        done_intercepting_(false),
        interceptor_methods_(InterceptorBatchMethodsImpl()) {}

  CallOpSet& operator=(const CallOpSet& other) {
    core_cq_tag_ = this;
    return_tag_ = this;
    call_ = other.call_;
    done_intercepting_ = false;
    interceptor_methods_ = InterceptorBatchMethodsImpl();
    return *this;
  }

  void FillOps(Call* call) override {
    done_intercepting_ = false;
    g_core_codegen_interface->grpc_call_ref(call->call());
    call_ =
        *call;  // It's fine to create a copy of call since it's just pointers

    if (RunInterceptors()) {
      ContinueFillOpsAfterInterception();
    } else {
      // After the interceptors are run, ContinueFillOpsAfterInterception will
      // be run
    }
  }

  bool FinalizeResult(void** tag, bool* status) override {
    if (done_intercepting_) {
      // Complete the avalanching since we are done with this batch of ops
      call_.cq()->CompleteAvalanching();
      // We have already finished intercepting and filling in the results. This
      // round trip from the core needed to be made because interceptors were
      // run
      *tag = return_tag_;
      *status = saved_status_;
      g_core_codegen_interface->grpc_call_unref(call_.call());
      return true;
    }

    this->Op1::FinishOp(status);
    this->Op2::FinishOp(status);
    this->Op3::FinishOp(status);
    this->Op4::FinishOp(status);
    this->Op5::FinishOp(status);
    this->Op6::FinishOp(status);
    saved_status_ = *status;
    if (RunInterceptorsPostRecv()) {
      *tag = return_tag_;
      g_core_codegen_interface->grpc_call_unref(call_.call());
      return true;
    }
    // Interceptors are going to be run, so we can't return the tag just yet.
    // After the interceptors are run, ContinueFinalizeResultAfterInterception
    return false;
  }

  void set_output_tag(void* return_tag) { return_tag_ = return_tag; }

  void* core_cq_tag() override { return core_cq_tag_; }

  /// set_core_cq_tag is used to provide a different core CQ tag than "this".
  /// This is used for callback-based tags, where the core tag is the core
  /// callback function. It does not change the use or behavior of any other
  /// function (such as FinalizeResult)
  void set_core_cq_tag(void* core_cq_tag) { core_cq_tag_ = core_cq_tag; }

  // This will be called while interceptors are run if the RPC is a hijacked
  // RPC. This should set hijacking state for each of the ops.
  void SetHijackingState() override {
    this->Op1::SetHijackingState(&interceptor_methods_);
    this->Op2::SetHijackingState(&interceptor_methods_);
    this->Op3::SetHijackingState(&interceptor_methods_);
    this->Op4::SetHijackingState(&interceptor_methods_);
    this->Op5::SetHijackingState(&interceptor_methods_);
    this->Op6::SetHijackingState(&interceptor_methods_);
  }

  // Should be called after interceptors are done running
  void ContinueFillOpsAfterInterception() override {
    static const size_t MAX_OPS = 6;
    grpc_op ops[MAX_OPS];
    size_t nops = 0;
    this->Op1::AddOp(ops, &nops);
    this->Op2::AddOp(ops, &nops);
    this->Op3::AddOp(ops, &nops);
    this->Op4::AddOp(ops, &nops);
    this->Op5::AddOp(ops, &nops);
    this->Op6::AddOp(ops, &nops);

    grpc_call_error err = g_core_codegen_interface->grpc_call_start_batch(
        call_.call(), ops, nops, core_cq_tag(), nullptr);

    if (err != GRPC_CALL_OK) {
      // A failure here indicates an API misuse; for example, doing a Write
      // while another Write is already pending on the same RPC or invoking
      // WritesDone multiple times
      // gpr_log(GPR_ERROR, "API misuse of type %s observed",
      //        g_core_codegen_interface->grpc_call_error_to_string(err));
      GPR_CODEGEN_ASSERT(false);
    }
  }

  // Should be called after interceptors are done running on the finalize result
  // path
  void ContinueFinalizeResultAfterInterception() override {
    done_intercepting_ = true;
    // The following call_start_batch is internally-generated so no need for an
    // explanatory log on failure.
    GPR_CODEGEN_ASSERT(g_core_codegen_interface->grpc_call_start_batch(
                           call_.call(), nullptr, 0, core_cq_tag(), nullptr) ==
                       GRPC_CALL_OK);
  }

 private:
  // Returns true if no interceptors need to be run
  bool RunInterceptors() {
    interceptor_methods_.ClearState();
    interceptor_methods_.SetCallOpSetInterface(this);
    interceptor_methods_.SetCall(&call_);
    this->Op1::SetInterceptionHookPoint(&interceptor_methods_);
    this->Op2::SetInterceptionHookPoint(&interceptor_methods_);
    this->Op3::SetInterceptionHookPoint(&interceptor_methods_);
    this->Op4::SetInterceptionHookPoint(&interceptor_methods_);
    this->Op5::SetInterceptionHookPoint(&interceptor_methods_);
    this->Op6::SetInterceptionHookPoint(&interceptor_methods_);
    if (interceptor_methods_.InterceptorsListEmpty()) {
      return true;
    }
    // This call will go through interceptors and would need to
    // schedule new batches, so delay completion queue shutdown
    call_.cq()->RegisterAvalanching();
    return interceptor_methods_.RunInterceptors();
  }
  // Returns true if no interceptors need to be run
  bool RunInterceptorsPostRecv() {
    // Call and OpSet had already been set on the set state.
    // SetReverse also clears previously set hook points
    interceptor_methods_.SetReverse();
    this->Op1::SetFinishInterceptionHookPoint(&interceptor_methods_);
    this->Op2::SetFinishInterceptionHookPoint(&interceptor_methods_);
    this->Op3::SetFinishInterceptionHookPoint(&interceptor_methods_);
    this->Op4::SetFinishInterceptionHookPoint(&interceptor_methods_);
    this->Op5::SetFinishInterceptionHookPoint(&interceptor_methods_);
    this->Op6::SetFinishInterceptionHookPoint(&interceptor_methods_);
    return interceptor_methods_.RunInterceptors();
  }

  void* core_cq_tag_;
  void* return_tag_;
  Call call_;
  bool done_intercepting_ = false;
  InterceptorBatchMethodsImpl interceptor_methods_;
  bool saved_status_;
};

}  // namespace internal
}  // namespace grpc

#endif  // GRPCPP_IMPL_CODEGEN_CALL_OP_SET_H
