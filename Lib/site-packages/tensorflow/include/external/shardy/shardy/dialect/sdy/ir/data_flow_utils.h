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

#ifndef SHARDY_DIALECT_SDY_IR_DATA_FLOW_UTILS_H_
#define SHARDY_DIALECT_SDY_IR_DATA_FLOW_UTILS_H_

#include <functional>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// TODO(b/330339693): This is a short term solution to define the data flow
// edges for stablehlo control flow ops. In the long term we should remove this
// dependency by introducing an op-interface that stablehlo control flow ops
// (and other dialects) will implement. This interface should define similar
// methods to those currently defined here.

// See `DataFlowEdgeOp` documentation for more information on data-flow edges.

// Returns true if the `op` defines data flow edges, e.g. , it's a
// `ShardableDataFlowOpInterface`.
bool isDataFlowOp(Operation* op);

// If `op` has data-flow edges, returns their op result edge owners (e.g., all
// results of a while/case op), otherwise returns an empty range.
ResultRange getDataFlowEdgeResultOwners(Operation* op);

// If `op` is a `ShardableDataFlowOpInterface` which can have block argument
// edge owners, returns the owners, otherwise returns an empty range.
ArrayRef<BlockArgument> getDataFlowEdgeBlockArgumentOwners(Operation* op);

// If `target` is a target of a data-flow edge, returns the corresponding
// `DataFlowEdgeOp`, otherwise returns `nullptr`.
DataFlowEdgeOp getDataFlowEdge(Value target);

// If `source` is a source of a data-flow edge, returns the corresponding
// `DataFlowEdgeOp`, otherwise returns `nullptr`.
DataFlowEdgeOp getDataFlowEdge(OpOperand& source);

// Returns all sources of the given `dataFlowEdge`.
SmallVector<Value> getDataFlowSources(DataFlowEdgeOp dataFlowEdge);

// Returns all non-edge-owner targets of the given `dataFlowEdge`.
SmallVector<Value> getNonEdgeOwnerTargets(DataFlowEdgeOp dataFlowEdge);

// Sets the block argument edge owner shardings if the `op` is a
// `ShardableDataFlowOpInterface`.
void setBlockArgumentEdgeOwnerShardings(Operation* op,
                                        ArrayRef<TensorShardingAttr> shardings);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_DATA_FLOW_UTILS_H_
