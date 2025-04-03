/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the access groups attribute or nullptr
::mlir::ArrayAttr mlir::LLVM::AccessGroupOpInterface::getAccessGroupsOrNull() {
      return getImpl()->getAccessGroupsOrNull(getImpl(), getOperation());
  }
/// Sets the access groups attribute
void mlir::LLVM::AccessGroupOpInterface::setAccessGroups(const ::mlir::ArrayAttr attr) {
      return getImpl()->setAccessGroups(getImpl(), getOperation(), attr);
  }
/// Returns the alias scopes attribute or nullptr
::mlir::ArrayAttr mlir::LLVM::AliasAnalysisOpInterface::getAliasScopesOrNull() {
      return getImpl()->getAliasScopesOrNull(getImpl(), getOperation());
  }
/// Sets the alias scopes attribute
void mlir::LLVM::AliasAnalysisOpInterface::setAliasScopes(const ::mlir::ArrayAttr attr) {
      return getImpl()->setAliasScopes(getImpl(), getOperation(), attr);
  }
/// Returns the noalias scopes attribute or nullptr
::mlir::ArrayAttr mlir::LLVM::AliasAnalysisOpInterface::getNoAliasScopesOrNull() {
      return getImpl()->getNoAliasScopesOrNull(getImpl(), getOperation());
  }
/// Sets the noalias scopes attribute
void mlir::LLVM::AliasAnalysisOpInterface::setNoAliasScopes(const ::mlir::ArrayAttr attr) {
      return getImpl()->setNoAliasScopes(getImpl(), getOperation(), attr);
  }
/// Returns the tbaa attribute or nullptr
::mlir::ArrayAttr mlir::LLVM::AliasAnalysisOpInterface::getTBAATagsOrNull() {
      return getImpl()->getTBAATagsOrNull(getImpl(), getOperation());
  }
/// Sets the tbaa attribute
void mlir::LLVM::AliasAnalysisOpInterface::setTBAATags(const ::mlir::ArrayAttr attr) {
      return getImpl()->setTBAATags(getImpl(), getOperation(), attr);
  }
/// Returns a list of all pointer operands accessed by the operation
::llvm::SmallVector<::mlir::Value> mlir::LLVM::AliasAnalysisOpInterface::getAccessedOperands() {
      return getImpl()->getAccessedOperands(getImpl(), getOperation());
  }
/// Returns the branch weights attribute or nullptr
::mlir::DenseI32ArrayAttr mlir::LLVM::BranchWeightOpInterface::getBranchWeightsOrNull() {
      return getImpl()->getBranchWeightsOrNull(getImpl(), getOperation());
  }
/// Sets the branch weights attribute
void mlir::LLVM::BranchWeightOpInterface::setBranchWeights(::mlir::DenseI32ArrayAttr attr) {
      return getImpl()->setBranchWeights(getImpl(), getOperation(), attr);
  }
/// Returns a FPExceptionBehavior attribute for the operation
::mlir::LLVM::FPExceptionBehaviorAttr mlir::LLVM::FPExceptionBehaviorOpInterface::getFPExceptionBehaviorAttr() {
      return getImpl()->getFPExceptionBehaviorAttr(getImpl(), getOperation());
  }
/// Returns the name of the FPExceptionBehaviorAttr
///                         attribute for the operation
::llvm::StringRef mlir::LLVM::FPExceptionBehaviorOpInterface::getFPExceptionBehaviorAttrName() {
      return getImpl()->getFPExceptionBehaviorAttrName();
  }
/// Returns a FastmathFlagsAttr attribute for the operation
::mlir::LLVM::FastmathFlagsAttr mlir::LLVM::FastmathFlagsInterface::getFastmathAttr() {
      return getImpl()->getFastmathAttr(getImpl(), getOperation());
  }
/// Returns the name of the FastmathFlagsAttr attribute
///                          for the operation
::llvm::StringRef mlir::LLVM::FastmathFlagsInterface::getFastmathAttrName() {
      return getImpl()->getFastmathAttrName();
  }
/// Get the integer overflow flags for the operation.
IntegerOverflowFlags mlir::LLVM::IntegerOverflowFlagsInterface::getOverflowFlags() {
      return getImpl()->getOverflowFlags(getImpl(), getOperation());
  }
/// Set the integer overflow flags for the operation.
void mlir::LLVM::IntegerOverflowFlagsInterface::setOverflowFlags(IntegerOverflowFlags flags) {
      return getImpl()->setOverflowFlags(getImpl(), getOperation(), flags);
  }
/// Returns whether the operation has the No Unsigned Wrap keyword.
bool mlir::LLVM::IntegerOverflowFlagsInterface::hasNoUnsignedWrap() {
      return getImpl()->hasNoUnsignedWrap(getImpl(), getOperation());
  }
/// Returns whether the operation has the No Signed Wrap keyword.
bool mlir::LLVM::IntegerOverflowFlagsInterface::hasNoSignedWrap() {
      return getImpl()->hasNoSignedWrap(getImpl(), getOperation());
  }
/// Get the attribute name of the overflow flags property.
StringRef mlir::LLVM::IntegerOverflowFlagsInterface::getOverflowFlagsAttrName() {
      return getImpl()->getOverflowFlagsAttrName();
  }
/// Returns a RoundingMode attribute for the operation
::mlir::LLVM::RoundingModeAttr mlir::LLVM::RoundingModeOpInterface::getRoundingModeAttr() {
      return getImpl()->getRoundingModeAttr(getImpl(), getOperation());
  }
/// Returns the name of the RoundingModeAttr attribute
///                          for the operation
::llvm::StringRef mlir::LLVM::RoundingModeOpInterface::getRoundingModeAttrName() {
      return getImpl()->getRoundingModeAttrName();
  }
