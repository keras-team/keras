/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AUTO_PARTITIONER_REGISTRY_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AUTO_PARTITIONER_REGISTRY_H_

#include <functional>

#include "mlir/Pass/PassOptions.h"
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir {
namespace sdy {

// A callback that takes a `OpPassManager`, and appends to it custom automatic
// partitioning passes that operates in place on the module to add sharding
// annotations. The module may have been sharded earlier or may be sharded after
// the automatic partitioning passes have been invoked.
//
// AutomaticPartition passes may add new/modified sharding
// attributes, but should NOT modify the module itself (replace ops, etc).
using AutoPartitionerCallback = std::function<void(OpPassManager&)>;

// A callback that takes a `DialectRegistry`, and registers the dialects
// required for automatic partitioning.
using RegisterDependantDialectsCallback = std::function<void(DialectRegistry&)>;

// A registry for an auto-partitioner callback that Shardy propagation
// should use in case auto-partitioning is enabled.
//
// The registry is thread-safe, and a callback can only be set once.
class AutoPartitionerRegistry {
 public:
  // Registers the given `callback` and its required dependencies.
  //
  // Assumes no callback has been registered yet.
  static void setCallback(
      AutoPartitionerCallback callback,
      RegisterDependantDialectsCallback dialectsDependenciesCallback);

  // Adds passes to the given `pm` to invoke AutomaticPartitioner.
  //
  // Assumes a callback has been registered.
  static void addPasses(OpPassManager& pm);

  // Registers the dependencies of the auto-partitioner passes that needs
  // to exist before calling the auto-partitioner pipeline.
  //
  // Assumes a callback has been registered.
  static void getDependentDialects(DialectRegistry& registry);

  // Clears the registered callback.
  static void clear();

  // Returns true if a callback has been registered.
  static bool isRegistered();
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AUTO_PARTITIONER_REGISTRY_H_
