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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SUBCHANNEL_INTERFACE_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SUBCHANNEL_INTERFACE_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/gprpp/ref_counted_ptr.h"

namespace grpc_core {

// The interface for subchannels that is exposed to LB policy implementations.
class SubchannelInterface : public RefCounted<SubchannelInterface> {
 public:
  class ConnectivityStateWatcherInterface {
   public:
    virtual ~ConnectivityStateWatcherInterface() = default;

    // Will be invoked whenever the subchannel's connectivity state
    // changes.  There will be only one invocation of this method on a
    // given watcher instance at any given time.
    virtual void OnConnectivityStateChange(
        grpc_connectivity_state new_state) = 0;

    // TODO(roth): Remove this as soon as we move to EventManager-based
    // polling.
    virtual grpc_pollset_set* interested_parties() = 0;
  };

  template <typename TraceFlagT = TraceFlag>
  explicit SubchannelInterface(TraceFlagT* trace_flag = nullptr)
      : RefCounted<SubchannelInterface>(trace_flag) {}

  virtual ~SubchannelInterface() = default;

  // Returns the current connectivity state of the subchannel.
  virtual grpc_connectivity_state CheckConnectivityState() = 0;

  // Starts watching the subchannel's connectivity state.
  // The first callback to the watcher will be delivered when the
  // subchannel's connectivity state becomes a value other than
  // initial_state, which may happen immediately.
  // Subsequent callbacks will be delivered as the subchannel's state
  // changes.
  // The watcher will be destroyed either when the subchannel is
  // destroyed or when CancelConnectivityStateWatch() is called.
  // There can be only one watcher of a given subchannel.  It is not
  // valid to call this method a second time without first cancelling
  // the previous watcher using CancelConnectivityStateWatch().
  virtual void WatchConnectivityState(
      grpc_connectivity_state initial_state,
      std::unique_ptr<ConnectivityStateWatcherInterface> watcher) = 0;

  // Cancels a connectivity state watch.
  // If the watcher has already been destroyed, this is a no-op.
  virtual void CancelConnectivityStateWatch(
      ConnectivityStateWatcherInterface* watcher) = 0;

  // Attempt to connect to the backend.  Has no effect if already connected.
  // If the subchannel is currently in backoff delay due to a previously
  // failed attempt, the new connection attempt will not start until the
  // backoff delay has elapsed.
  virtual void AttemptToConnect() = 0;

  // Resets the subchannel's connection backoff state.  If AttemptToConnect()
  // has been called since the subchannel entered TRANSIENT_FAILURE state,
  // starts a new connection attempt immediately; otherwise, a new connection
  // attempt will be started as soon as AttemptToConnect() is called.
  virtual void ResetBackoff() = 0;

  // TODO(roth): Need a better non-grpc-specific abstraction here.
  virtual const grpc_channel_args* channel_args() = 0;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SUBCHANNEL_INTERFACE_H */
