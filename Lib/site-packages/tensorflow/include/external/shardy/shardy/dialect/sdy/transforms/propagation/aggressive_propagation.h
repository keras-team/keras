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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AGGRESSIVE_PROPAGATION_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AGGRESSIVE_PROPAGATION_H_

#include <memory>

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"

namespace mlir {
namespace sdy {

// The strategy to use for propagating shardings along factors.
enum class PropagationStrategy {
  // Use the basic factor propagation strategy.
  Basic,

  // Use the aggressive factor propagation strategy.
  Aggressive,

  // Use the basic and aggressive strategies in order. We first propagate all
  // shardings without conflicts, and then resolve conflicts with the aggressive
  // strategy.
  BasicThenAggressive,
};

// The implementation class for the op-priority propagation pass.
class AggressivePropagationPassImpl : public BasicPropagationPassImpl {
 public:
  using BasicPropagationPassImpl::BasicPropagationPassImpl;

  AggressivePropagationPassImpl(const AggressivePropagationPassImpl& other)
      : BasicPropagationPassImpl(other) {}

 protected:
  // This method calls `BasicPropagationPassImpl::propagate` with specified
  // strategies. See `BasicPropagationPassImpl::propagate` for documentation.
  //
  // If `propagationStrategy` is `Basic`, we use the basic factor propagation
  // strategy, owned by `BasicPropagationPassImpl`.
  //
  // If `propagationStrategy` is `Aggressive`, we use the aggressive factor
  // propagation strategy in private member `aggressiveFactorPropagation`.
  //
  // If `propagationStrategy` is `BasicThenAggressive`, we first use the basic
  // strategy, and then the aggressive strategy to resolve conflicts.
  LogicalResult propagate(
      ModuleOp moduleOp,
      GetDirectionToPropagateFn getDirectionToPropagate) override;

  Option<PropagationStrategy> propagationStrategy = {
      *this, "propagation-strategy",
      llvm::cl::desc("which factor propagation strategy to use"),
      llvm::cl::init(PropagationStrategy::Aggressive),
      llvm::cl::values(
          clEnumValN(PropagationStrategy::Basic, "basic",
                     "basic factor propagation"),
          clEnumValN(PropagationStrategy::Aggressive, "aggressive",
                     "aggressive factor propagation"),
          clEnumValN(PropagationStrategy::BasicThenAggressive,
                     "basic-then-aggressive",
                     "basic factor propagation followed by aggressive factor "
                     "propagation"))};

 private:
  // This class owns the aggressive factor propagation strategy.
  AggressiveFactorPropagation aggressiveFactorPropagation;
};

// Runs op based sharding propagation (see `AggressivePropagationPass`).
std::unique_ptr<Pass> createAggressivePropagationPass(
    bool keepShardingRules, StringRef dumpDirectory = "",
    bool conservativePropagation = false,
    PropagationStrategy propagationStrategy = PropagationStrategy::Aggressive);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AGGRESSIVE_PROPAGATION_H_
