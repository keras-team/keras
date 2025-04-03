//===- FoldInterfaces.h - Folding Interfaces --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_INTERFACES_FOLDINTERFACES_H_
#define MLIR_INTERFACES_FOLDINTERFACES_H_

#include "mlir/IR/DialectInterface.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Attribute;
class OpFoldResult;
class Region;

/// Define a fold interface to allow for dialects to control specific aspects
/// of the folding behavior for operations they define.
class DialectFoldInterface
    : public DialectInterface::Base<DialectFoldInterface> {
public:
  DialectFoldInterface(Dialect *dialect) : Base(dialect) {}

  /// Registered fallback fold for the dialect. Like the fold hook of each
  /// operation, it attempts to fold the operation with the specified constant
  /// operand values - the elements in "operands" will correspond directly to
  /// the operands of the operation, but may be null if non-constant.  If
  /// folding is successful, this fills in the `results` vector.  If not, this
  /// returns failure and `results` is unspecified.
  virtual LogicalResult fold(Operation *op, ArrayRef<Attribute> operands,
                             SmallVectorImpl<OpFoldResult> &results) const {
    return failure();
  }

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants. The folder will generally materialize constants
  /// into the top-level isolated region, this allows for materializing into a
  /// lower level ancestor region if it is more profitable/correct.
  virtual bool shouldMaterializeInto(Region *region) const { return false; }
};

} // namespace mlir

#endif // MLIR_INTERFACES_FOLDINTERFACES_H_
