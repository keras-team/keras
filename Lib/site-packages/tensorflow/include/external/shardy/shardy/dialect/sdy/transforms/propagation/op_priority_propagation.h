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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_OP_PRIORITY_PROPAGATION_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_OP_PRIORITY_PROPAGATION_H_

#include <memory>

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"

namespace mlir {
namespace sdy {

// The implementation class for the op-priority propagation pass.
class OpPriorityPropagationPassImpl : public AggressivePropagationPassImpl {
 public:
  using AggressivePropagationPassImpl::AggressivePropagationPassImpl;

  OpPriorityPropagationPassImpl(const OpPriorityPropagationPassImpl& other)
      : AggressivePropagationPassImpl(other) {}

 protected:
  // See `AggressivePropagationPassImpl::propagate` for documentation.
  //
  // This override will take the intersection of what `getDirectionToPropagate`
  // returns, and the current direction for a given op and op-priority based on
  // an internally defined strategy.
  LogicalResult propagate(
      ModuleOp moduleOp,
      GetDirectionToPropagateFn getDirectionToPropagate) override;

  Option<bool> runOpPriorityPropagation = {
      *this, "run-op-priority-propagation",
      llvm::cl::desc("whether to run (or skip) op-priority propagation"),
      llvm::cl::init(true)};
};

// Runs op based sharding propagation (see `OpPriorityPropagationPass`).
std::unique_ptr<Pass> createOpPriorityPropagationPass(
    bool keepShardingRules, StringRef dumpDirectory = "",
    bool conservativePropagation = false);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_OP_PRIORITY_PROPAGATION_H_
