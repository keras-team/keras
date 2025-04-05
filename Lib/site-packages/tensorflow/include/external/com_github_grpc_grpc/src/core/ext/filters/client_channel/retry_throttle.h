/*
 *
 * Copyright 2017 gRPC authors.
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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RETRY_THROTTLE_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RETRY_THROTTLE_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/gprpp/ref_counted.h"

namespace grpc_core {
namespace internal {

/// Tracks retry throttling data for an individual server name.
class ServerRetryThrottleData : public RefCounted<ServerRetryThrottleData> {
 public:
  ServerRetryThrottleData(intptr_t max_milli_tokens, intptr_t milli_token_ratio,
                          ServerRetryThrottleData* old_throttle_data);
  ~ServerRetryThrottleData();

  /// Records a failure.  Returns true if it's okay to send a retry.
  bool RecordFailure();

  /// Records a success.
  void RecordSuccess();

  intptr_t max_milli_tokens() const { return max_milli_tokens_; }
  intptr_t milli_token_ratio() const { return milli_token_ratio_; }

 private:
  void GetReplacementThrottleDataIfNeeded(
      ServerRetryThrottleData** throttle_data);

  const intptr_t max_milli_tokens_;
  const intptr_t milli_token_ratio_;
  gpr_atm milli_tokens_;
  // A pointer to the replacement for this ServerRetryThrottleData entry.
  // If non-nullptr, then this entry is stale and must not be used.
  // We hold a reference to the replacement.
  gpr_atm replacement_ = 0;
};

/// Global map of server name to retry throttle data.
class ServerRetryThrottleMap {
 public:
  /// Initializes global map of failure data for each server name.
  static void Init();
  /// Shuts down global map of failure data for each server name.
  static void Shutdown();

  /// Returns the failure data for \a server_name, creating a new entry if
  /// needed.
  static RefCountedPtr<ServerRetryThrottleData> GetDataForServer(
      const char* server_name, intptr_t max_milli_tokens,
      intptr_t milli_token_ratio);
};

}  // namespace internal
}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RETRY_THROTTLE_H */
