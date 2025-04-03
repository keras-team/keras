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

#ifndef SHARDY_DIALECT_SDY_IR_UTILS_H_
#define SHARDY_DIALECT_SDY_IR_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

template <typename... Ts>
void unreachableFormatv(const char* format, Ts&&... vals) {
  llvm_unreachable(
      llvm::formatv(format, std::forward<Ts>(vals)...).str().c_str());
}

template <class Dialect>
bool inDialect(Operation* op) {
  return op->getDialect()->getNamespace() == Dialect::getDialectNamespace();
}


// Emits a warning once for the given `flag`, with `op` attached as a note
// if `MLIRContext::shouldPrintOpOnDiagnostic` is true (assuming the op is
// verified).
void emitOpWarningOnce(llvm::once_flag& flag, Operation* op, StringRef msg);

// Converts `attr` to string.
std::string attributeToString(Attribute attr);

// Converts `op` to string with location information.
std::string operationToString(Operation* op);

// Converts `value` to string with location information.
std::string valueToString(Value value);

// If the given `type` is a `ShapedType` with a static shape, returns it,
// otherwise returns nullptr.
ShapedType dynCastStaticShapedType(Type type);

// Returns true if the given `type` is a `ShapedType` with a static shape.
bool isStaticShapedType(Type type);

// Returns the shape of the given `value` if its type is a `ShapeTensor`,
// otherwise returns an empty array.
//
// Assumes the `ShapeTensor` has a rank.
ArrayRef<int64_t> getTensorShape(Value value);

// Returns the rank of the given `value` if its type is a `ShapeTensor`,
// otherwise returns 0.
//
// Assumes the `ShapeTensor` has a rank.
int64_t getTensorRank(Value value);

// Returns true if the value is a tensor with rank 0.
int64_t isScalar(Value value);

// Looks up the mesh symbol with the given `meshName` in `symbolTable`, and
// returns its `MeshAttr` if it exists in the table, or nullptr otherwise.
MeshAttr getMeshAttr(const SymbolTable& symbolTable, StringRef meshName);

// Looks up the mesh symbol with the given `meshSymName` in `symbolTable`, and
// returns its `MeshAttr` if it exists in the table, or nullptr otherwise.
MeshAttr getMeshAttr(const SymbolTable& symbolTable, SymbolRefAttr meshSymName);

// Looks up the mesh symbol with the given `meshName` in the symbol table of
// the enclosing module of `op`, and returns its `MeshAttr` if it exists in the
// table, or nullptr otherwise.
MeshAttr getMeshAttr(Operation* op, StringRef meshName);

// Looks up the mesh symbol with the given `meshSymName` in the symbol table of
// the enclosing module of `op`, and returns its `MeshAttr` if it exists in the
// table, or nullptr otherwise.
MeshAttr getMeshAttr(Operation* op, SymbolRefAttr meshSymName);

// Returns the common mesh name used by all the `TensorShardingAttr` or
// std::nullopt if there is none.
std::optional<StringRef> getCommonMeshName(
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultsShardings);

// Creates the symbol equivalent of a factor index:
//   -  0 -> 'i'
//   -  1 -> 'j'
//   - 17 -> 'z'
//   - 18 -> 'z_1'
std::string factorSymbolString(int64_t factor);

// Returns the string representation of `attr` without dialect wrapping
//
// If `stripMnemonic` is true, also strips the mnemonic of the attribute.
template <class AttrTy>
std::string strippedAttrString(AttrTy attr, bool stripMnemonic = false) {
  std::string result;
  llvm::raw_string_ostream os(result);
  attr.printStripped(os);
  if (stripMnemonic) {
    result.erase(0, attr.getMnemonic().size());
  }
  return result;
}

// Returns the defining op of `value`, if it's an op result, or the parent op
// if it's a block argument.
Operation* getOwningOp(Value value);

// Returns the given `value`, if a sharding can be attached to it, or another
// value that holds the sharding for `value` (e.g. the operand corresponding to
// a block argument in a control flow op).
//
// Some op results and block arguments don't have shardings attached to them.
// Instead we recursively loop through the defining op of these ops' operands.
//
// Returns an empty value if the given `value` has no shardable value, e.g., a
// scalar block argument of a reduction function.
// TODO(tomnatan): consider moving this to a dedicated sdy/utils dir.
Value getShardableValue(Value value);

// Returns the sharding of the given `value`, whose location depends on the type
// of the value.
//
// For example, the sharding of a function block argument is a function argument
// attr.
//
// Some op results and block arguments don't have shardings attached to them.
// Instead we recursively loop through the defining op of these ops' operands.
//
// Returns an empty `TensorShardingAttr` if the given `value` has no sharding or
// if it has no shardable value (see `getShardableValue`)
// TODO(tomnatan): consider moving this to a dedicated sdy/utils dir.
TensorShardingAttr getSharding(Value value);

// Returns the sharding of the given `value`, or a fully open empty
// `TensorShardingAttr` if `value` doesn't have a sharding.
TensorShardingAttr getOrCreateSharding(Value value, StringRef meshName);

// Sets the sharding of the given `value`, whose location depends on the type of
// the value, to `sharding`.
//
// For example, the sharding of a function block argument is a function argument
// attr.
//
// Some op results and block arguments don't have shardings attached to them.
// Instead we recursively loop through the defining op of these ops' operands.
// TODO(tomnatan): consider moving this to a dedicated sdy/utils dir.
void setSharding(Value value, TensorShardingAttr sharding);

SmallVector<TensorShardingAttr> getShardings(ValueRange values);

// Cleanup the module by removing sharding rule attrs. Keep any user specified
// ones.
void removeShardingRules(Operation* rootOp);

// Gets the `op` body's terminator.
template <typename RegionOpTy>
Operation* getBodyTerminator(RegionOpTy op) {
  return op.getBody().front().getTerminator();
}

// Gets the operands of the `op` body terminator.
template <typename RegionOpTy>
MutableArrayRef<OpOperand> getBodyTerminatorOpOperands(RegionOpTy op) {
  return getBodyTerminator(op)->getOpOperands();
}

// Gets the values returned from the `op` body terminator.
template <typename RegionOpTy>
ValueRange getBodyTerminatorOperands(RegionOpTy op) {
  return getBodyTerminator(op)->getOperands();
}

// Gets the value types returned from the `op` body terminator.
template <typename RegionOpTy>
TypeRange getBodyTerminatorOpOperandTypes(RegionOpTy op) {
  return getBodyTerminator(op)->getOperandTypes();
}

// Inlines (i.e., move) operations from region `src` into `dst` and converts the
// terminator of each block in `dst` to `TerminatorOpTy`. The `rewriter`'s
// insertion point is modified.
template <typename TerminatorOpTy>
void inlineRegionAndConvertTerminatorOp(Region& src, Region& dst,
                                        PatternRewriter& rewriter) {
  rewriter.inlineRegionBefore(src, dst, dst.begin());

  for (Block& block : dst.getBlocks()) {
    Operation* returnOp = block.getTerminator();

    rewriter.setInsertionPointAfter(returnOp);
    rewriter.replaceOpWithNewOp<TerminatorOpTy>(returnOp,
                                                returnOp->getOperands());
  }
}

// Inlines (i.e., move) operations from region `src` into `dst` and converts the
// terminator of each block in `dst` to `TerminatorOpTy`.
template <typename TerminatorOpTy>
void inlineRegionAndConvertTerminatorOp(Region& src, Region& dst) {
  dst.takeBody(src);

  for (Block& block : dst.getBlocks()) {
    Operation* returnOp = block.getTerminator();
    OpBuilder::atBlockEnd(&block).create<TerminatorOpTy>(
        returnOp->getLoc(), returnOp->getOperands());
    returnOp->erase();
  }
}

// Clones region `src`, inserts `src` into the start of `dst`, and converts the
// terminator of each block in `dst` to `TerminatorOpTy`.
template <typename TerminatorOpTy>
void cloneRegionAndConvertTerminatorOp(Region& src, Region& dst,
                                       RewriterBase& rewriter) {
  Block::iterator savedInsertionPoint = rewriter.getInsertionPoint();
  Block* savedBlock = rewriter.getInsertionBlock();
  rewriter.cloneRegionBefore(src, dst, dst.begin());

  for (auto& block : dst.getBlocks()) {
    Operation* returnOp = block.getTerminator();
    rewriter.setInsertionPointAfter(returnOp);
    rewriter.replaceOpWithNewOp<TerminatorOpTy>(returnOp,
                                                returnOp->getOperands());
  }

  rewriter.setInsertionPoint(savedBlock, savedInsertionPoint);
}

// Clones region `src`, inserts `src` into the start of `dst`, and converts the
// terminator of each block in `dst` to `TerminatorOpTy`.
template <typename TerminatorOpTy>
void cloneRegionAndConvertTerminatorOp(Region& src, Region& dst) {
  IRRewriter rewriter(src.getContext());
  cloneRegionAndConvertTerminatorOp<TerminatorOpTy>(src, dst, rewriter);
}

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_UTILS_H_
