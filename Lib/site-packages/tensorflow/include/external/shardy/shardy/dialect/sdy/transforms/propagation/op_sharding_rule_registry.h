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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_OP_SHARDING_RULE_REGISTRY_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_OP_SHARDING_RULE_REGISTRY_H_

#include "mlir/IR/Operation.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// Creates a sharding rule based on an op.
//
// If `conservativePropagation` is true, the rule created will make sure that
// each operand/result dimension size will be mapped to one or more factors
// whose total size is equal to the dimension size, so that propagation won't
// shard this dimension along axes that don't divide its size.
//
// NOTE: an empty rule {([])->([])} will be created for scalar ops.
OpShardingRuleAttr createOpShardingRule(Operation* op,
                                        bool conservativePropagation = false);

// Gets the sharding rule if it exists already on the op. Else creates one,
// sets it on the op, and returns it.
//
// See `createOpShardingRule` for more info.
OpShardingRuleAttr getOrCreateShardingRule(
    Operation* op, bool conservativePropagation = false);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_OP_SHARDING_RULE_REGISTRY_H_
