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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_REGISTRY_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_REGISTRY_H

#include <grpc/support/port_platform.h>

#include "src/core/ext/filters/client_channel/lb_policy_factory.h"
#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/gprpp/orphanable.h"

namespace grpc_core {

class LoadBalancingPolicyRegistry {
 public:
  /// Methods used to create and populate the LoadBalancingPolicyRegistry.
  /// NOT THREAD SAFE -- to be used only during global gRPC
  /// initialization and shutdown.
  class Builder {
   public:
    /// Global initialization and shutdown hooks.
    static void InitRegistry();
    static void ShutdownRegistry();

    /// Registers an LB policy factory.  The factory will be used to create an
    /// LB policy whose name matches that of the factory.
    static void RegisterLoadBalancingPolicyFactory(
        std::unique_ptr<LoadBalancingPolicyFactory> factory);
  };

  /// Creates an LB policy of the type specified by \a name.
  static OrphanablePtr<LoadBalancingPolicy> CreateLoadBalancingPolicy(
      const char* name, LoadBalancingPolicy::Args args);

  /// Returns true if the LB policy factory specified by \a name exists in this
  /// registry. If the load balancing policy requires a config to be specified
  /// then sets \a requires_config to true.
  static bool LoadBalancingPolicyExists(const char* name,
                                        bool* requires_config);

  /// Returns a parsed object of the load balancing policy to be used from a
  /// LoadBalancingConfig array \a json.
  static RefCountedPtr<LoadBalancingPolicy::Config> ParseLoadBalancingConfig(
      const grpc_json* json, grpc_error** error);
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_LB_POLICY_REGISTRY_H */
