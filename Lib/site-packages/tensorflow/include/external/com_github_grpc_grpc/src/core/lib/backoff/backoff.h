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

#ifndef GRPC_CORE_LIB_BACKOFF_BACKOFF_H
#define GRPC_CORE_LIB_BACKOFF_BACKOFF_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/iomgr/exec_ctx.h"

namespace grpc_core {

/// Implementation of the backoff mechanism described in
/// doc/connection-backoff.md
class BackOff {
 public:
  class Options;

  /// Initialize backoff machinery - does not need to be destroyed
  explicit BackOff(const Options& options);

  /// Returns the time at which the next attempt should start.
  grpc_millis NextAttemptTime();

  /// Reset the backoff, so the next value returned by NextAttemptTime()
  /// will be the time of the second attempt (rather than the Nth).
  void Reset();

  void SetRandomSeed(unsigned int seed);

  class Options {
   public:
    Options& set_initial_backoff(grpc_millis initial_backoff) {
      initial_backoff_ = initial_backoff;
      return *this;
    }
    Options& set_multiplier(double multiplier) {
      multiplier_ = multiplier;
      return *this;
    }
    Options& set_jitter(double jitter) {
      jitter_ = jitter;
      return *this;
    }
    Options& set_max_backoff(grpc_millis max_backoff) {
      max_backoff_ = max_backoff;
      return *this;
    }
    /// how long to wait after the first failure before retrying
    grpc_millis initial_backoff() const { return initial_backoff_; }
    /// factor with which to multiply backoff after a failed retry
    double multiplier() const { return multiplier_; }
    /// amount to randomize backoffs
    double jitter() const { return jitter_; }
    /// maximum time between retries
    grpc_millis max_backoff() const { return max_backoff_; }

   private:
    grpc_millis initial_backoff_;
    double multiplier_;
    double jitter_;
    grpc_millis max_backoff_;
  };  // class Options

 private:
  const Options options_;
  uint32_t rng_state_;
  bool initial_;
  /// current delay before retries
  grpc_millis current_backoff_;
};

}  // namespace grpc_core
#endif /* GRPC_CORE_LIB_BACKOFF_BACKOFF_H */
