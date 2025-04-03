//===- FoldUtils.h - Operation Fold Utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares various operation folding utilities. These
// utilities are intended to be used by passes to unify and simply their logic.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_FOLDUTILS_H
#define MLIR_TRANSFORMS_FOLDUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FoldInterfaces.h"

namespace mlir {
class Operation;
class Value;

//===--------------------------------------------------------------------===//
// OperationFolder
//===--------------------------------------------------------------------===//

/// A utility class for folding operations, and unifying duplicated constants
/// generated along the way.
class OperationFolder {
public:
  OperationFolder(MLIRContext *ctx, OpBuilder::Listener *listener = nullptr)
      : erasedFoldedLocation(UnknownLoc::get(ctx)), interfaces(ctx),
        rewriter(ctx, listener) {}

  /// Tries to perform folding on the given `op`, including unifying
  /// deduplicated constants. If successful, replaces `op`'s uses with
  /// folded results, and returns success. If the op was completely folded it is
  /// erased. If it is just updated in place, `inPlaceUpdate` is set to true.
  LogicalResult tryToFold(Operation *op, bool *inPlaceUpdate = nullptr);

  /// Tries to fold a pre-existing constant operation. `constValue` represents
  /// the value of the constant, and can be optionally passed if the value is
  /// already known (e.g. if the constant was discovered by m_Constant). This is
  /// purely an optimization opportunity for callers that already know the value
  /// of the constant. Returns false if an existing constant for `op` already
  /// exists in the folder, in which case `op` is replaced and erased.
  /// Otherwise, returns true and `op` is inserted into the folder (and
  /// hoisted if necessary).
  bool insertKnownConstant(Operation *op, Attribute constValue = {});

  /// Notifies that the given constant `op` should be remove from this
  /// OperationFolder's internal bookkeeping.
  ///
  /// Note: this method must be called if a constant op is to be deleted
  /// externally to this OperationFolder. `op` must be a constant op.
  void notifyRemoval(Operation *op);

  /// Clear out any constants cached inside of the folder.
  void clear();

  /// Get or create a constant for use in the specified block. The constant may
  /// be created in a parent block. On success this returns the constant
  /// operation, nullptr otherwise.
  Value getOrCreateConstant(Block *block, Dialect *dialect, Attribute value,
                            Type type);

private:
  /// This map keeps track of uniqued constants by dialect, attribute, and type.
  /// A constant operation materializes an attribute with a type. Dialects may
  /// generate different constants with the same input attribute and type, so we
  /// also need to track per-dialect.
  using ConstantMap =
      DenseMap<std::tuple<Dialect *, Attribute, Type>, Operation *>;

  /// Returns true if the given operation is an already folded constant that is
  /// owned by this folder.
  bool isFolderOwnedConstant(Operation *op) const;

  /// Tries to perform folding on the given `op`. If successful, populates
  /// `results` with the results of the folding.
  LogicalResult tryToFold(Operation *op, SmallVectorImpl<Value> &results);

  /// Try to process a set of fold results. Populates `results` on success,
  /// otherwise leaves it unchanged.
  LogicalResult processFoldResults(Operation *op,
                                   SmallVectorImpl<Value> &results,
                                   ArrayRef<OpFoldResult> foldResults);

  /// Try to get or create a new constant entry. On success this returns the
  /// constant operation, nullptr otherwise.
  Operation *tryGetOrCreateConstant(ConstantMap &uniquedConstants,
                                    Dialect *dialect, Attribute value,
                                    Type type, Location loc);

  /// The location to overwrite with for folder-owned constants.
  UnknownLoc erasedFoldedLocation;

  /// A mapping between an insertion region and the constants that have been
  /// created within it.
  DenseMap<Region *, ConstantMap> foldScopes;

  /// This map tracks all of the dialects that an operation is referenced by;
  /// given that many dialects may generate the same constant.
  DenseMap<Operation *, SmallVector<Dialect *, 2>> referencedDialects;

  /// A collection of dialect folder interfaces.
  DialectInterfaceCollection<DialectFoldInterface> interfaces;

  /// A rewriter that performs all IR modifications.
  IRRewriter rewriter;
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_FOLDUTILS_H
