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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_FACTORY_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_FACTORY_H

#include <grpc/support/port_platform.h>

#include "src/core/ext/filters/client_channel/lb_policy.h"
#include "src/core/lib/gprpp/orphanable.h"

namespace grpc_core {

class LoadBalancingPolicyFactory {
 public:
  /// Returns a new LB policy instance.
  virtual OrphanablePtr<LoadBalancingPolicy> CreateLoadBalancingPolicy(
      LoadBalancingPolicy::Args) const = 0;

  /// Returns the LB policy name that this factory provides.
  /// Caller does NOT take ownership of result.
  virtual const char* name() const = 0;

  virtual RefCountedPtr<LoadBalancingPolicy::Config> ParseLoadBalancingConfig(
      const grpc_json* json, grpc_error** error) const = 0;

  virtual ~LoadBalancingPolicyFactory() {}
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_FACTORY_H */
