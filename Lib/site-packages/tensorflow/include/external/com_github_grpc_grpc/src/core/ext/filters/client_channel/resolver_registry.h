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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_REGISTRY_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_REGISTRY_H

#include <grpc/support/port_platform.h>

#include "src/core/ext/filters/client_channel/resolver_factory.h"
#include "src/core/lib/gprpp/inlined_vector.h"
#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/gprpp/orphanable.h"
#include "src/core/lib/iomgr/pollset_set.h"

namespace grpc_core {

class ResolverRegistry {
 public:
  /// Methods used to create and populate the ResolverRegistry.
  /// NOT THREAD SAFE -- to be used only during global gRPC
  /// initialization and shutdown.
  class Builder {
   public:
    /// Global initialization and shutdown hooks.
    static void InitRegistry();
    static void ShutdownRegistry();

    /// Sets the default URI prefix to \a default_prefix.
    /// Calls InitRegistry() if it has not already been called.
    static void SetDefaultPrefix(const char* default_prefix);

    /// Registers a resolver factory.  The factory will be used to create a
    /// resolver for any URI whose scheme matches that of the factory.
    /// Calls InitRegistry() if it has not already been called.
    static void RegisterResolverFactory(
        std::unique_ptr<ResolverFactory> factory);
  };

  /// Checks whether the user input \a target is valid to create a resolver.
  static bool IsValidTarget(const char* target);

  /// Creates a resolver given \a target.
  /// First tries to parse \a target as a URI. If this succeeds, tries
  /// to locate a registered resolver factory based on the URI scheme.
  /// If parsing fails or there is no factory for the URI's scheme,
  /// prepends default_prefix to target and tries again.
  /// If a resolver factory is found, uses it to instantiate a resolver and
  /// returns it; otherwise, returns nullptr.
  /// \a args, \a pollset_set, and \a combiner are passed to the factory's
  /// \a CreateResolver() method.
  /// \a args are the channel args to be included in resolver results.
  /// \a pollset_set is used to drive I/O in the name resolution process.
  /// \a combiner is the combiner under which all resolver calls will be run.
  /// \a result_handler is used to return results from the resolver.
  static OrphanablePtr<Resolver> CreateResolver(
      const char* target, const grpc_channel_args* args,
      grpc_pollset_set* pollset_set, Combiner* combiner,
      std::unique_ptr<Resolver::ResultHandler> result_handler);

  /// Returns the default authority to pass from a client for \a target.
  static grpc_core::UniquePtr<char> GetDefaultAuthority(const char* target);

  /// Returns \a target with the default prefix prepended, if needed.
  static grpc_core::UniquePtr<char> AddDefaultPrefixIfNeeded(
      const char* target);

  /// Returns the resolver factory for \a scheme.
  /// Caller does NOT own the return value.
  static ResolverFactory* LookupResolverFactory(const char* scheme);
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_REGISTRY_H */
