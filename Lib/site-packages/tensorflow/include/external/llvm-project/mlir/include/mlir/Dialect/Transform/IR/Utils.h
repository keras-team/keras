//===- Utils.h - Utils related to the transform dialect ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_UTILS_H
#define MLIR_DIALECT_TRANSFORM_IR_UTILS_H

namespace mlir {
class InFlightDiagnostic;
class Operation;
template <typename>
class OwningOpRef;

namespace transform {
namespace detail {

/// Merge all symbols from `other` into `target`. Both ops need to implement the
/// `SymbolTable` trait. Operations are moved from `other`, i.e., `other` may be
/// modified by this function and might not verify after the function returns.
/// Upon merging, private symbols may be renamed in order to avoid collisions in
/// the result. Public symbols may not collide, with the exception of
/// instances of `SymbolOpInterface`, where collisions are allowed if at least
/// one of the two is external, in which case the other op preserved (or any one
/// of the two if both are external).
// TODO: Reconsider cloning individual ops rather than forcing users of the
//       function to clone (or move) `other` in order to improve efficiency.
//       This might primarily make sense if we can also prune the symbols that
//       are merged to a subset (such as those that are actually used).
InFlightDiagnostic mergeSymbolsInto(Operation *target,
                                    OwningOpRef<Operation *> other);

} // namespace detail
} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_IR_UTILS_H
