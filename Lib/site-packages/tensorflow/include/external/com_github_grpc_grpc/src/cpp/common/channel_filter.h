/*
 *
 * Copyright 2016 gRPC authors.
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

#ifndef GRPCXX_CHANNEL_FILTER_H
#define GRPCXX_CHANNEL_FILTER_H

#include <grpc/grpc.h>
#include <grpc/support/alloc.h>
#include <grpcpp/impl/codegen/config.h>

#include <functional>
#include <vector>

#include "src/core/lib/channel/channel_stack.h"
#include "src/core/lib/surface/channel_init.h"
#include "src/core/lib/transport/metadata_batch.h"

/// An interface to define filters.
///
/// To define a filter, implement a subclass of each of \c CallData and
/// \c ChannelData. Then register the filter using something like this:
/// \code{.cpp}
///   RegisterChannelFilter<MyChannelDataSubclass, MyCallDataSubclass>(
///       "name-of-filter", GRPC_SERVER_CHANNEL, INT_MAX, nullptr);
/// \endcode

namespace grpc {

/// A C++ wrapper for the \c grpc_metadata_batch struct.
class MetadataBatch {
 public:
  /// Borrows a pointer to \a batch, but does NOT take ownership.
  /// The caller must ensure that \a batch continues to exist for as
  /// long as the MetadataBatch object does.
  explicit MetadataBatch(grpc_metadata_batch* batch) : batch_(batch) {}

  grpc_metadata_batch* batch() const { return batch_; }

  /// Adds metadata and returns the newly allocated storage.
  /// The caller takes ownership of the result, which must exist for the
  /// lifetime of the gRPC call.
  grpc_linked_mdelem* AddMetadata(const string& key, const string& value);

  class const_iterator : public std::iterator<std::bidirectional_iterator_tag,
                                              const grpc_mdelem> {
   public:
    const grpc_mdelem& operator*() const { return elem_->md; }
    const grpc_mdelem operator->() const { return elem_->md; }

    const_iterator& operator++() {
      elem_ = elem_->next;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp(*this);
      operator++();
      return tmp;
    }
    const_iterator& operator--() {
      elem_ = elem_->prev;
      return *this;
    }
    const_iterator operator--(int) {
      const_iterator tmp(*this);
      operator--();
      return tmp;
    }

    bool operator==(const const_iterator& other) const {
      return elem_ == other.elem_;
    }
    bool operator!=(const const_iterator& other) const {
      return elem_ != other.elem_;
    }

   private:
    friend class MetadataBatch;
    explicit const_iterator(grpc_linked_mdelem* elem) : elem_(elem) {}

    grpc_linked_mdelem* elem_;
  };

  const_iterator begin() const { return const_iterator(batch_->list.head); }
  const_iterator end() const { return const_iterator(nullptr); }

 private:
  grpc_metadata_batch* batch_;  // Not owned.
};

/// A C++ wrapper for the \c grpc_transport_op struct.
class TransportOp {
 public:
  /// Borrows a pointer to \a op, but does NOT take ownership.
  /// The caller must ensure that \a op continues to exist for as
  /// long as the TransportOp object does.
  explicit TransportOp(grpc_transport_op* op) : op_(op) {}

  grpc_transport_op* op() const { return op_; }

  // TODO(roth): Add a C++ wrapper for grpc_error?
  grpc_error* disconnect_with_error() const {
    return op_->disconnect_with_error;
  }
  bool send_goaway() const { return op_->goaway_error != GRPC_ERROR_NONE; }

  // TODO(roth): Add methods for additional fields as needed.

 private:
  grpc_transport_op* op_;  // Not owned.
};

/// A C++ wrapper for the \c grpc_transport_stream_op_batch struct.
class TransportStreamOpBatch {
 public:
  /// Borrows a pointer to \a op, but does NOT take ownership.
  /// The caller must ensure that \a op continues to exist for as
  /// long as the TransportStreamOpBatch object does.
  explicit TransportStreamOpBatch(grpc_transport_stream_op_batch* op)
      : op_(op),
        send_initial_metadata_(
            op->send_initial_metadata
                ? op->payload->send_initial_metadata.send_initial_metadata
                : nullptr),
        send_trailing_metadata_(
            op->send_trailing_metadata
                ? op->payload->send_trailing_metadata.send_trailing_metadata
                : nullptr),
        recv_initial_metadata_(
            op->recv_initial_metadata
                ? op->payload->recv_initial_metadata.recv_initial_metadata
                : nullptr),
        recv_trailing_metadata_(
            op->recv_trailing_metadata
                ? op->payload->recv_trailing_metadata.recv_trailing_metadata
                : nullptr) {}

  grpc_transport_stream_op_batch* op() const { return op_; }

  grpc_closure* on_complete() const { return op_->on_complete; }
  void set_on_complete(grpc_closure* closure) { op_->on_complete = closure; }

  MetadataBatch* send_initial_metadata() {
    return op_->send_initial_metadata ? &send_initial_metadata_ : nullptr;
  }
  MetadataBatch* send_trailing_metadata() {
    return op_->send_trailing_metadata ? &send_trailing_metadata_ : nullptr;
  }
  MetadataBatch* recv_initial_metadata() {
    return op_->recv_initial_metadata ? &recv_initial_metadata_ : nullptr;
  }
  MetadataBatch* recv_trailing_metadata() {
    return op_->recv_trailing_metadata ? &recv_trailing_metadata_ : nullptr;
  }

  uint32_t* send_initial_metadata_flags() const {
    return op_->send_initial_metadata ? &op_->payload->send_initial_metadata
                                             .send_initial_metadata_flags
                                      : nullptr;
  }

  grpc_closure* recv_initial_metadata_ready() const {
    return op_->recv_initial_metadata
               ? op_->payload->recv_initial_metadata.recv_initial_metadata_ready
               : nullptr;
  }
  void set_recv_initial_metadata_ready(grpc_closure* closure) {
    op_->payload->recv_initial_metadata.recv_initial_metadata_ready = closure;
  }

  grpc_core::OrphanablePtr<grpc_core::ByteStream>* send_message() const {
    return op_->send_message ? &op_->payload->send_message.send_message
                             : nullptr;
  }
  void set_send_message(
      grpc_core::OrphanablePtr<grpc_core::ByteStream> send_message) {
    op_->send_message = true;
    op_->payload->send_message.send_message = std::move(send_message);
  }

  grpc_core::OrphanablePtr<grpc_core::ByteStream>* recv_message() const {
    return op_->recv_message ? op_->payload->recv_message.recv_message
                             : nullptr;
  }
  void set_recv_message(
      grpc_core::OrphanablePtr<grpc_core::ByteStream>* recv_message) {
    op_->recv_message = true;
    op_->payload->recv_message.recv_message = recv_message;
  }

  census_context* get_census_context() const {
    return static_cast<census_context*>(
        op_->payload->context[GRPC_CONTEXT_TRACING].value);
  }

  const gpr_atm* get_peer_string() const {
    if (op_->send_initial_metadata &&
        op_->payload->send_initial_metadata.peer_string != nullptr) {
      return op_->payload->send_initial_metadata.peer_string;
    } else if (op_->recv_initial_metadata &&
               op_->payload->recv_initial_metadata.peer_string != nullptr) {
      return op_->payload->recv_initial_metadata.peer_string;
    } else {
      return nullptr;
    }
  }

 private:
  grpc_transport_stream_op_batch* op_;  // Not owned.
  MetadataBatch send_initial_metadata_;
  MetadataBatch send_trailing_metadata_;
  MetadataBatch recv_initial_metadata_;
  MetadataBatch recv_trailing_metadata_;
};

/// Represents channel data.
class ChannelData {
 public:
  ChannelData() {}
  virtual ~ChannelData() {}

  // TODO(roth): Come up with a more C++-like API for the channel element.

  /// Initializes the channel data.
  virtual grpc_error* Init(grpc_channel_element* /*elem*/,
                           grpc_channel_element_args* /*args*/) {
    return GRPC_ERROR_NONE;
  }

  // Called before destruction.
  virtual void Destroy(grpc_channel_element* /*elem*/) {}

  virtual void StartTransportOp(grpc_channel_element* elem, TransportOp* op);

  virtual void GetInfo(grpc_channel_element* elem,
                       const grpc_channel_info* channel_info);
};

/// Represents call data.
class CallData {
 public:
  CallData() {}
  virtual ~CallData() {}

  // TODO(roth): Come up with a more C++-like API for the call element.

  /// Initializes the call data.
  virtual grpc_error* Init(grpc_call_element* /*elem*/,
                           const grpc_call_element_args* /*args*/) {
    return GRPC_ERROR_NONE;
  }

  // Called before destruction.
  virtual void Destroy(grpc_call_element* /*elem*/,
                       const grpc_call_final_info* /*final_info*/,
                       grpc_closure* /*then_call_closure*/) {}

  /// Starts a new stream operation.
  virtual void StartTransportStreamOpBatch(grpc_call_element* elem,
                                           TransportStreamOpBatch* op);

  /// Sets a pollset or pollset set.
  virtual void SetPollsetOrPollsetSet(grpc_call_element* elem,
                                      grpc_polling_entity* pollent);
};

namespace internal {

// Defines static members for passing to C core.
// Members of this class correspond to the members of the C
// grpc_channel_filter struct.
template <typename ChannelDataType, typename CallDataType>
class ChannelFilter final {
 public:
  static const size_t channel_data_size = sizeof(ChannelDataType);

  static grpc_error* InitChannelElement(grpc_channel_element* elem,
                                        grpc_channel_element_args* args) {
    // Construct the object in the already-allocated memory.
    ChannelDataType* channel_data = new (elem->channel_data) ChannelDataType();
    return channel_data->Init(elem, args);
  }

  static void DestroyChannelElement(grpc_channel_element* elem) {
    ChannelDataType* channel_data =
        static_cast<ChannelDataType*>(elem->channel_data);
    channel_data->Destroy(elem);
    channel_data->~ChannelDataType();
  }

  static void StartTransportOp(grpc_channel_element* elem,
                               grpc_transport_op* op) {
    ChannelDataType* channel_data =
        static_cast<ChannelDataType*>(elem->channel_data);
    TransportOp op_wrapper(op);
    channel_data->StartTransportOp(elem, &op_wrapper);
  }

  static void GetChannelInfo(grpc_channel_element* elem,
                             const grpc_channel_info* channel_info) {
    ChannelDataType* channel_data =
        static_cast<ChannelDataType*>(elem->channel_data);
    channel_data->GetInfo(elem, channel_info);
  }

  static const size_t call_data_size = sizeof(CallDataType);

  static grpc_error* InitCallElement(grpc_call_element* elem,
                                     const grpc_call_element_args* args) {
    // Construct the object in the already-allocated memory.
    CallDataType* call_data = new (elem->call_data) CallDataType();
    return call_data->Init(elem, args);
  }

  static void DestroyCallElement(grpc_call_element* elem,
                                 const grpc_call_final_info* final_info,
                                 grpc_closure* then_call_closure) {
    CallDataType* call_data = static_cast<CallDataType*>(elem->call_data);
    call_data->Destroy(elem, final_info, then_call_closure);
    call_data->~CallDataType();
  }

  static void StartTransportStreamOpBatch(grpc_call_element* elem,
                                          grpc_transport_stream_op_batch* op) {
    CallDataType* call_data = static_cast<CallDataType*>(elem->call_data);
    TransportStreamOpBatch op_wrapper(op);
    call_data->StartTransportStreamOpBatch(elem, &op_wrapper);
  }

  static void SetPollsetOrPollsetSet(grpc_call_element* elem,
                                     grpc_polling_entity* pollent) {
    CallDataType* call_data = static_cast<CallDataType*>(elem->call_data);
    call_data->SetPollsetOrPollsetSet(elem, pollent);
  }
};

struct FilterRecord {
  grpc_channel_stack_type stack_type;
  int priority;
  std::function<bool(const grpc_channel_args&)> include_filter;
  grpc_channel_filter filter;
};
extern std::vector<FilterRecord>* channel_filters;

void ChannelFilterPluginInit();
void ChannelFilterPluginShutdown();

}  // namespace internal

/// Registers a new filter.
/// Must be called by only one thread at a time.
/// The \a include_filter argument specifies a function that will be called
/// to determine at run-time whether or not to add the filter. If the
/// value is nullptr, the filter will be added unconditionally.
/// If the channel stack type is GRPC_CLIENT_SUBCHANNEL, the caller should
/// ensure that subchannels with different filter lists will always have
/// different channel args. This requires setting a channel arg in case the
/// registration function relies on some condition other than channel args to
/// decide whether to add a filter or not.
template <typename ChannelDataType, typename CallDataType>
void RegisterChannelFilter(
    const char* name, grpc_channel_stack_type stack_type, int priority,
    std::function<bool(const grpc_channel_args&)> include_filter) {
  // If we haven't been called before, initialize channel_filters and
  // call grpc_register_plugin().
  if (internal::channel_filters == nullptr) {
    grpc_register_plugin(internal::ChannelFilterPluginInit,
                         internal::ChannelFilterPluginShutdown);
    internal::channel_filters = new std::vector<internal::FilterRecord>();
  }
  // Add an entry to channel_filters. The filter will be added when the
  // C-core initialization code calls ChannelFilterPluginInit().
  typedef internal::ChannelFilter<ChannelDataType, CallDataType> FilterType;
  internal::FilterRecord filter_record = {
      stack_type,
      priority,
      include_filter,
      {FilterType::StartTransportStreamOpBatch, FilterType::StartTransportOp,
       FilterType::call_data_size, FilterType::InitCallElement,
       FilterType::SetPollsetOrPollsetSet, FilterType::DestroyCallElement,
       FilterType::channel_data_size, FilterType::InitChannelElement,
       FilterType::DestroyChannelElement, FilterType::GetChannelInfo, name}};
  internal::channel_filters->push_back(filter_record);
}

}  // namespace grpc

#endif  // GRPCXX_CHANNEL_FILTER_H
