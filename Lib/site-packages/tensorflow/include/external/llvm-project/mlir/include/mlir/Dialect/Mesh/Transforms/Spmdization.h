//===- Simplifications.h - Mesh Simplifications -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H
#define MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace mesh {

// Insert resharding spmdization of the value `sourceShardValue`
// from sharding `source` to sharding `target`.
// `sourceShardValue` is the already sharded value according to `source`.
//
// Example
//
// ```mlir
//   mesh.mesh @mesh_1d(shape = 2)
//   ...
//   %1 = mesh.shard %0 to <@mesh_1d, [[0]]> : tensor<2xi8>
//   %2 = mesh.shard %1 to <@mesh_1d, [[]]> annotate_for_users: tensor<2xi8>
// ```
//
// Will result in
//
// ```mlir
//   %1 = mesh.all_gather %0 on @mesh_1d mesh_axes = [0] gather_axis = 0 :
//     tensor<1xi8> -> tensor<2xi8>
// ```
TypedValue<ShapedType> reshard(OpBuilder &builder, MeshOp mesh, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue);
TypedValue<ShapedType> reshard(OpBuilder &builder, ShardOp source,
                               ShardOp target,
                               TypedValue<ShapedType> sourceShardValue,
                               SymbolTableCollection &symbolTableCollection);

void reshardingRegisterDependentDialects(DialectRegistry &registry);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H
