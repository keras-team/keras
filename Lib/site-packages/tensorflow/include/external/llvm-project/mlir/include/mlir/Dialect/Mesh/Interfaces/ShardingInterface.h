//===- ShardingInterface.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_
#define MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;
class IRMapping;
class SymbolTableCollection;

namespace mesh {

using ShardingArray = SmallVector<SmallVector<MeshAxis>>;
using ShardingArrayRef = ArrayRef<SmallVector<MeshAxis>>;

struct ShardingOption {
  // An array of int array. The sub-array at the i-th position signifies the
  // mesh axes the i-th loop will be sharded on.
  ShardingArray shardingArray = {};
  FlatSymbolRefAttr mesh = nullptr;
  // `empty` being true indicates that no sharding information can be inferred
  // at present. Note that it is different from the case where an operation is
  // not sharded.
  bool empty = false;
  ShardingOption() = default;
  ShardingOption(ShardingArray shardingArray, FlatSymbolRefAttr mesh)
      : shardingArray(std::move(shardingArray)), mesh(mesh) {}
  static ShardingOption makeEmpty() {
    auto res = ShardingOption();
    res.empty = true;
    return res;
  }
};

// This method retrieves the 'MeshSharding' from a given operation
// result and includes the 'annotate_for_users' information.
FailureOr<std::pair<bool, MeshSharding>> getMeshSharding(OpResult result);

// This method retrieves the 'MeshSharding' from a given operation
// operand and includes the 'annotate_for_users' information.
FailureOr<std::pair<bool, MeshSharding>> getMeshSharding(OpOperand &opOperand);

namespace detail {

FailureOr<ShardingOption>
defaultGetShardingOption(Operation *op, ArrayRef<MeshSharding> operandShardings,
                         ArrayRef<MeshSharding> resultShardings);

FailureOr<std::vector<MeshSharding>>
defaultGetShardingAnnotations(Operation *op,
                              const ShardingOption &shardingOption);

LogicalResult
defaultAddShardingAnnotations(Operation *op, OpBuilder &b,
                              const ShardingOption &shardingOption);

} // namespace detail

// Assumes full replication on all ranked tensor arguments and results.
void spmdizeFullyReplicatedOperation(Operation &op,
                                     ArrayRef<Value> spmdizedOperands,
                                     ArrayRef<MeshSharding> operandShardings,
                                     ArrayRef<MeshSharding> resultShardings,
                                     IRMapping &spmdizationMap,
                                     SymbolTableCollection &symbolTable,
                                     OpBuilder &builder);

} // namespace mesh
} // namespace mlir

/// Include the ODS generated interface header files.
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h.inc"

#endif // MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_
