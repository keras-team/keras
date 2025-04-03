//===- SparseTensorInterfaces.h - sparse tensor operations interfaces------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORINTERFACES_H_
#define MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORINTERFACES_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class PatternRewriter;

namespace sparse_tensor {
class StageWithSortSparseOp;

namespace detail {
LogicalResult stageWithSortImpl(sparse_tensor::StageWithSortSparseOp op,
                                PatternRewriter &rewriter, Value &tmpBufs);
} // namespace detail
} // namespace sparse_tensor
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.h.inc"

#endif // MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORINTERFACES_H_
