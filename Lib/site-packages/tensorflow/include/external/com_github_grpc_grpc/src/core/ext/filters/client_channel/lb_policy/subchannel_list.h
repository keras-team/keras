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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_SUBCHANNEL_LIST_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_SUBCHANNEL_LIST_H

#include <grpc/support/port_platform.h>

#include <string.h>

#include <grpc/support/alloc.h>

#include "src/core/ext/filters/client_channel/lb_policy_registry.h"
#include "src/core/ext/filters/client_channel/server_address.h"
// TODO(roth): Should not need the include of subchannel.h here, since
// that implementation should be hidden from the LB policy API.
#include "src/core/ext/filters/client_channel/subchannel.h"
#include "src/core/ext/filters/client_channel/subchannel_interface.h"
#include "src/core/lib/channel/channel_args.h"
#include "src/core/lib/debug/trace.h"
#include "src/core/lib/gprpp/inlined_vector.h"
#include "src/core/lib/gprpp/orphanable.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/iomgr/closure.h"
#include "src/core/lib/iomgr/sockaddr_utils.h"
#include "src/core/lib/transport/connectivity_state.h"

// Code for maintaining a list of subchannels within an LB policy.
//
// To use this, callers must create their own subclasses, like so:
/*

class MySubchannelList;  // Forward declaration.

class MySubchannelData
    : public SubchannelData<MySubchannelList, MySubchannelData> {
 public:
  void ProcessConnectivityChangeLocked(
      grpc_connectivity_state connectivity_state) override {
    // ...code to handle connectivity changes...
  }
};

class MySubchannelList
    : public SubchannelList<MySubchannelList, MySubchannelData> {
};

*/
// All methods will be called from within the client_channel combiner.

namespace grpc_core {

// Forward declaration.
template <typename SubchannelListType, typename SubchannelDataType>
class SubchannelList;

// Stores data for a particular subchannel in a subchannel list.
// Callers must create a subclass that implements the
// ProcessConnectivityChangeLocked() method.
template <typename SubchannelListType, typename SubchannelDataType>
class SubchannelData {
 public:
  // Returns a pointer to the subchannel list containing this object.
  SubchannelListType* subchannel_list() const {
    return static_cast<SubchannelListType*>(subchannel_list_);
  }

  // Returns the index into the subchannel list of this object.
  size_t Index() const {
    return static_cast<size_t>(static_cast<const SubchannelDataType*>(this) -
                               subchannel_list_->subchannel(0));
  }

  // Returns a pointer to the subchannel.
  SubchannelInterface* subchannel() const { return subchannel_.get(); }

  // Synchronously checks the subchannel's connectivity state.
  // Must not be called while there is a connectivity notification
  // pending (i.e., between calling StartConnectivityWatchLocked() and
  // calling CancelConnectivityWatchLocked()).
  grpc_connectivity_state CheckConnectivityStateLocked() {
    GPR_ASSERT(pending_watcher_ == nullptr);
    connectivity_state_ = subchannel_->CheckConnectivityState();
    return connectivity_state_;
  }

  // Resets the connection backoff.
  // TODO(roth): This method should go away when we move the backoff
  // code out of the subchannel and into the LB policies.
  void ResetBackoffLocked();

  // Starts watching the connectivity state of the subchannel.
  // ProcessConnectivityChangeLocked() will be called whenever the
  // connectivity state changes.
  void StartConnectivityWatchLocked();

  // Cancels watching the connectivity state of the subchannel.
  void CancelConnectivityWatchLocked(const char* reason);

  // Cancels any pending connectivity watch and unrefs the subchannel.
  void ShutdownLocked();

 protected:
  SubchannelData(
      SubchannelList<SubchannelListType, SubchannelDataType>* subchannel_list,
      const ServerAddress& address,
      RefCountedPtr<SubchannelInterface> subchannel);

  virtual ~SubchannelData();

  // After StartConnectivityWatchLocked() is called, this method will be
  // invoked whenever the subchannel's connectivity state changes.
  // To stop watching, use CancelConnectivityWatchLocked().
  virtual void ProcessConnectivityChangeLocked(
      grpc_connectivity_state connectivity_state) = 0;

 private:
  // Watcher for subchannel connectivity state.
  class Watcher
      : public SubchannelInterface::ConnectivityStateWatcherInterface {
   public:
    Watcher(
        SubchannelData<SubchannelListType, SubchannelDataType>* subchannel_data,
        RefCountedPtr<SubchannelListType> subchannel_list)
        : subchannel_data_(subchannel_data),
          subchannel_list_(std::move(subchannel_list)) {}

    ~Watcher() { subchannel_list_.reset(DEBUG_LOCATION, "Watcher dtor"); }

    void OnConnectivityStateChange(grpc_connectivity_state new_state) override;

    grpc_pollset_set* interested_parties() override {
      return subchannel_list_->policy()->interested_parties();
    }

   private:
    SubchannelData<SubchannelListType, SubchannelDataType>* subchannel_data_;
    RefCountedPtr<SubchannelListType> subchannel_list_;
  };

  // Unrefs the subchannel.
  void UnrefSubchannelLocked(const char* reason);

  // Backpointer to owning subchannel list.  Not owned.
  SubchannelList<SubchannelListType, SubchannelDataType>* subchannel_list_;
  // The subchannel.
  RefCountedPtr<SubchannelInterface> subchannel_;
  // Will be non-null when the subchannel's state is being watched.
  SubchannelInterface::ConnectivityStateWatcherInterface* pending_watcher_ =
      nullptr;
  // Data updated by the watcher.
  grpc_connectivity_state connectivity_state_;
};

// A list of subchannels.
template <typename SubchannelListType, typename SubchannelDataType>
class SubchannelList : public InternallyRefCounted<SubchannelListType> {
 public:
  typedef InlinedVector<SubchannelDataType, 10> SubchannelVector;

  // The number of subchannels in the list.
  size_t num_subchannels() const { return subchannels_.size(); }

  // The data for the subchannel at a particular index.
  SubchannelDataType* subchannel(size_t index) { return &subchannels_[index]; }

  // Returns true if the subchannel list is shutting down.
  bool shutting_down() const { return shutting_down_; }

  // Accessors.
  LoadBalancingPolicy* policy() const { return policy_; }
  TraceFlag* tracer() const { return tracer_; }

  // Resets connection backoff of all subchannels.
  // TODO(roth): We will probably need to rethink this as part of moving
  // the backoff code out of subchannels and into LB policies.
  void ResetBackoffLocked();

  void Orphan() override {
    ShutdownLocked();
    InternallyRefCounted<SubchannelListType>::Unref(DEBUG_LOCATION, "shutdown");
  }

 protected:
  SubchannelList(LoadBalancingPolicy* policy, TraceFlag* tracer,
                 const ServerAddressList& addresses,
                 LoadBalancingPolicy::ChannelControlHelper* helper,
                 const grpc_channel_args& args);

  virtual ~SubchannelList();

 private:
  // For accessing Ref() and Unref().
  friend class SubchannelData<SubchannelListType, SubchannelDataType>;

  void ShutdownLocked();

  // Backpointer to owning policy.
  LoadBalancingPolicy* policy_;

  TraceFlag* tracer_;

  // The list of subchannels.
  SubchannelVector subchannels_;

  // Is this list shutting down? This may be true due to the shutdown of the
  // policy itself or because a newer update has arrived while this one hadn't
  // finished processing.
  bool shutting_down_ = false;
};

//
// implementation -- no user-servicable parts below
//

//
// SubchannelData::Watcher
//

template <typename SubchannelListType, typename SubchannelDataType>
void SubchannelData<SubchannelListType, SubchannelDataType>::Watcher::
    OnConnectivityStateChange(grpc_connectivity_state new_state) {
  if (GRPC_TRACE_FLAG_ENABLED(*subchannel_list_->tracer())) {
    gpr_log(GPR_INFO,
            "[%s %p] subchannel list %p index %" PRIuPTR " of %" PRIuPTR
            " (subchannel %p): connectivity changed: state=%s, "
            "shutting_down=%d, pending_watcher=%p",
            subchannel_list_->tracer()->name(), subchannel_list_->policy(),
            subchannel_list_.get(), subchannel_data_->Index(),
            subchannel_list_->num_subchannels(),
            subchannel_data_->subchannel_.get(),
            ConnectivityStateName(new_state), subchannel_list_->shutting_down(),
            subchannel_data_->pending_watcher_);
  }
  if (!subchannel_list_->shutting_down() &&
      subchannel_data_->pending_watcher_ != nullptr) {
    subchannel_data_->connectivity_state_ = new_state;
    // Call the subclass's ProcessConnectivityChangeLocked() method.
    subchannel_data_->ProcessConnectivityChangeLocked(new_state);
  }
}

//
// SubchannelData
//

template <typename SubchannelListType, typename SubchannelDataType>
SubchannelData<SubchannelListType, SubchannelDataType>::SubchannelData(
    SubchannelList<SubchannelListType, SubchannelDataType>* subchannel_list,
    const ServerAddress& /*address*/,
    RefCountedPtr<SubchannelInterface> subchannel)
    : subchannel_list_(subchannel_list),
      subchannel_(std::move(subchannel)),
      // We assume that the current state is IDLE.  If not, we'll get a
      // callback telling us that.
      connectivity_state_(GRPC_CHANNEL_IDLE) {}

template <typename SubchannelListType, typename SubchannelDataType>
SubchannelData<SubchannelListType, SubchannelDataType>::~SubchannelData() {
  GPR_ASSERT(subchannel_ == nullptr);
}

template <typename SubchannelListType, typename SubchannelDataType>
void SubchannelData<SubchannelListType, SubchannelDataType>::
    UnrefSubchannelLocked(const char* reason) {
  if (subchannel_ != nullptr) {
    if (GRPC_TRACE_FLAG_ENABLED(*subchannel_list_->tracer())) {
      gpr_log(GPR_INFO,
              "[%s %p] subchannel list %p index %" PRIuPTR " of %" PRIuPTR
              " (subchannel %p): unreffing subchannel (%s)",
              subchannel_list_->tracer()->name(), subchannel_list_->policy(),
              subchannel_list_, Index(), subchannel_list_->num_subchannels(),
              subchannel_.get(), reason);
    }
    subchannel_.reset();
  }
}

template <typename SubchannelListType, typename SubchannelDataType>
void SubchannelData<SubchannelListType,
                    SubchannelDataType>::ResetBackoffLocked() {
  if (subchannel_ != nullptr) {
    subchannel_->ResetBackoff();
  }
}

template <typename SubchannelListType, typename SubchannelDataType>
void SubchannelData<SubchannelListType,
                    SubchannelDataType>::StartConnectivityWatchLocked() {
  if (GRPC_TRACE_FLAG_ENABLED(*subchannel_list_->tracer())) {
    gpr_log(GPR_INFO,
            "[%s %p] subchannel list %p index %" PRIuPTR " of %" PRIuPTR
            " (subchannel %p): starting watch (from %s)",
            subchannel_list_->tracer()->name(), subchannel_list_->policy(),
            subchannel_list_, Index(), subchannel_list_->num_subchannels(),
            subchannel_.get(), ConnectivityStateName(connectivity_state_));
  }
  GPR_ASSERT(pending_watcher_ == nullptr);
  pending_watcher_ =
      new Watcher(this, subchannel_list()->Ref(DEBUG_LOCATION, "Watcher"));
  subchannel_->WatchConnectivityState(
      connectivity_state_,
      std::unique_ptr<SubchannelInterface::ConnectivityStateWatcherInterface>(
          pending_watcher_));
}

template <typename SubchannelListType, typename SubchannelDataType>
void SubchannelData<SubchannelListType, SubchannelDataType>::
    CancelConnectivityWatchLocked(const char* reason) {
  if (GRPC_TRACE_FLAG_ENABLED(*subchannel_list_->tracer())) {
    gpr_log(GPR_INFO,
            "[%s %p] subchannel list %p index %" PRIuPTR " of %" PRIuPTR
            " (subchannel %p): canceling connectivity watch (%s)",
            subchannel_list_->tracer()->name(), subchannel_list_->policy(),
            subchannel_list_, Index(), subchannel_list_->num_subchannels(),
            subchannel_.get(), reason);
  }
  if (pending_watcher_ != nullptr) {
    subchannel_->CancelConnectivityStateWatch(pending_watcher_);
    pending_watcher_ = nullptr;
  }
}

template <typename SubchannelListType, typename SubchannelDataType>
void SubchannelData<SubchannelListType, SubchannelDataType>::ShutdownLocked() {
  if (pending_watcher_ != nullptr) CancelConnectivityWatchLocked("shutdown");
  UnrefSubchannelLocked("shutdown");
}

//
// SubchannelList
//

template <typename SubchannelListType, typename SubchannelDataType>
SubchannelList<SubchannelListType, SubchannelDataType>::SubchannelList(
    LoadBalancingPolicy* policy, TraceFlag* tracer,
    const ServerAddressList& addresses,
    LoadBalancingPolicy::ChannelControlHelper* helper,
    const grpc_channel_args& args)
    : InternallyRefCounted<SubchannelListType>(tracer),
      policy_(policy),
      tracer_(tracer) {
  if (GRPC_TRACE_FLAG_ENABLED(*tracer_)) {
    gpr_log(GPR_INFO,
            "[%s %p] Creating subchannel list %p for %" PRIuPTR " subchannels",
            tracer_->name(), policy, this, addresses.size());
  }
  subchannels_.reserve(addresses.size());
  // We need to remove the LB addresses in order to be able to compare the
  // subchannel keys of subchannels from a different batch of addresses.
  // We remove the service config, since it will be passed into the
  // subchannel via call context.
  static const char* keys_to_remove[] = {GRPC_ARG_SUBCHANNEL_ADDRESS,
                                         GRPC_ARG_SERVICE_CONFIG};
  // Create a subchannel for each address.
  for (size_t i = 0; i < addresses.size(); i++) {
    // TODO(roth): we should ideally hide this from the LB policy code. In
    // principle, if we're dealing with this special case in the client_channel
    // code for selecting grpclb, then we should also strip out these addresses
    // there if we're not using grpclb.
    if (addresses[i].IsBalancer()) {
      continue;
    }
    InlinedVector<grpc_arg, 3> args_to_add;
    const size_t subchannel_address_arg_index = args_to_add.size();
    args_to_add.emplace_back(
        Subchannel::CreateSubchannelAddressArg(&addresses[i].address()));
    if (addresses[i].args() != nullptr) {
      for (size_t j = 0; j < addresses[i].args()->num_args; ++j) {
        args_to_add.emplace_back(addresses[i].args()->args[j]);
      }
    }
    grpc_channel_args* new_args = grpc_channel_args_copy_and_add_and_remove(
        &args, keys_to_remove, GPR_ARRAY_SIZE(keys_to_remove),
        args_to_add.data(), args_to_add.size());
    gpr_free(args_to_add[subchannel_address_arg_index].value.string);
    RefCountedPtr<SubchannelInterface> subchannel =
        helper->CreateSubchannel(*new_args);
    grpc_channel_args_destroy(new_args);
    if (subchannel == nullptr) {
      // Subchannel could not be created.
      if (GRPC_TRACE_FLAG_ENABLED(*tracer_)) {
        char* address_uri = grpc_sockaddr_to_uri(&addresses[i].address());
        gpr_log(GPR_INFO,
                "[%s %p] could not create subchannel for address uri %s, "
                "ignoring",
                tracer_->name(), policy_, address_uri);
        gpr_free(address_uri);
      }
      continue;
    }
    if (GRPC_TRACE_FLAG_ENABLED(*tracer_)) {
      char* address_uri = grpc_sockaddr_to_uri(&addresses[i].address());
      gpr_log(GPR_INFO,
              "[%s %p] subchannel list %p index %" PRIuPTR
              ": Created subchannel %p for address uri %s",
              tracer_->name(), policy_, this, subchannels_.size(),
              subchannel.get(), address_uri);
      gpr_free(address_uri);
    }
    subchannels_.emplace_back(this, addresses[i], std::move(subchannel));
  }
}

template <typename SubchannelListType, typename SubchannelDataType>
SubchannelList<SubchannelListType, SubchannelDataType>::~SubchannelList() {
  if (GRPC_TRACE_FLAG_ENABLED(*tracer_)) {
    gpr_log(GPR_INFO, "[%s %p] Destroying subchannel_list %p", tracer_->name(),
            policy_, this);
  }
}

template <typename SubchannelListType, typename SubchannelDataType>
void SubchannelList<SubchannelListType, SubchannelDataType>::ShutdownLocked() {
  if (GRPC_TRACE_FLAG_ENABLED(*tracer_)) {
    gpr_log(GPR_INFO, "[%s %p] Shutting down subchannel_list %p",
            tracer_->name(), policy_, this);
  }
  GPR_ASSERT(!shutting_down_);
  shutting_down_ = true;
  for (size_t i = 0; i < subchannels_.size(); i++) {
    SubchannelDataType* sd = &subchannels_[i];
    sd->ShutdownLocked();
  }
}

template <typename SubchannelListType, typename SubchannelDataType>
void SubchannelList<SubchannelListType,
                    SubchannelDataType>::ResetBackoffLocked() {
  for (size_t i = 0; i < subchannels_.size(); i++) {
    SubchannelDataType* sd = &subchannels_[i];
    sd->ResetBackoffLocked();
  }
}

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_SUBCHANNEL_LIST_H */
