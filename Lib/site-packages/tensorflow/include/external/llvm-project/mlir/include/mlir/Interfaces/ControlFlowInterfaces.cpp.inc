/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the operands that correspond to the arguments of the successor
/// at the given index. It consists of a number of operands that are
/// internally produced by the operation, followed by a range of operands
/// that are forwarded. An example operation making use of produced
/// operands would be:
/// 
/// ```mlir
/// invoke %function(%0)
///     label ^success ^error(%1 : i32)
/// 
/// ^error(%e: !error, %arg0: i32):
///     ...
/// ```
/// 
/// The operand that would map to the `^error`s `%e` operand is produced
/// by the `invoke` operation, while `%1` is a forwarded operand that maps
/// to `%arg0` in the successor.
/// 
/// Produced operands always map to the first few block arguments of the
/// successor, followed by the forwarded operands. Mapping them in any
/// other order is not supported by the interface.
/// 
/// By having the forwarded operands last allows users of the interface
/// to append more forwarded operands to the branch operation without
/// interfering with other successor operands.
::mlir::SuccessorOperands mlir::BranchOpInterface::getSuccessorOperands(unsigned index) {
      return getImpl()->getSuccessorOperands(getImpl(), getOperation(), index);
  }
/// Returns the `BlockArgument` corresponding to operand `operandIndex` in
/// some successor, or std::nullopt if `operandIndex` isn't a successor operand
/// index.
::std::optional<::mlir::BlockArgument> mlir::BranchOpInterface::getSuccessorBlockArgument(unsigned operandIndex) {
      return getImpl()->getSuccessorBlockArgument(getImpl(), getOperation(), operandIndex);
  }
/// Returns the successor that would be chosen with the given constant
/// operands. Returns nullptr if a single successor could not be chosen.
::mlir::Block *mlir::BranchOpInterface::getSuccessorForOperands(::llvm::ArrayRef<::mlir::Attribute> operands) {
      return getImpl()->getSuccessorForOperands(getImpl(), getOperation(), operands);
  }
/// This method is called to compare types along control-flow edges. By
/// default, the types are checked as equal.
bool mlir::BranchOpInterface::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return getImpl()->areTypesCompatible(getImpl(), getOperation(), lhs, rhs);
  }
/// Returns the operands of this operation that are forwarded to the region
/// successor's block arguments or this operation's results when branching
/// to `point`. `point` is guaranteed to be among the successors that are
/// returned by `getEntrySuccessorRegions`/`getSuccessorRegions(parent())`.
/// 
/// Example: In the above example, this method returns the operand %b of the
/// `scf.for` op, regardless of the value of `point`. I.e., this op always
/// forwards the same operands, regardless of whether the loop has 0 or more
/// iterations.
::mlir::OperandRange mlir::RegionBranchOpInterface::getEntrySuccessorOperands(::mlir::RegionBranchPoint point) {
      return getImpl()->getEntrySuccessorOperands(getImpl(), getOperation(), point);
  }
/// Returns the potential region successors when first executing the op.
/// 
/// Unlike `getSuccessorRegions`, this method also passes along the
/// constant operands of this op. Based on these, the implementation may
/// filter out certain successors. By default, simply dispatches to
/// `getSuccessorRegions`. `operands` contains an entry for every
/// operand of this op, with a null attribute if the operand has no constant
/// value.
/// 
/// Note: The control flow does not necessarily have to enter any region of
/// this op.
/// 
/// Example: In the above example, this method may return two region
/// region successors: the single region of the `scf.for` op and the
/// `scf.for` operation (that implements this interface). If %lb, %ub, %step
/// are constants and it can be determined the loop does not have any
/// iterations, this method may choose to return only this operation.
/// Similarly, if it can be determined that the loop has at least one
/// iteration, this method may choose to return only the region of the loop.
void mlir::RegionBranchOpInterface::getEntrySuccessorRegions(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
      return getImpl()->getEntrySuccessorRegions(getImpl(), getOperation(), operands, regions);
  }
/// Returns the potential region successors when branching from `point`.
/// These are the regions that may be selected during the flow of control.
/// 
/// When `point = RegionBranchPoint::parent()`, this method returns the
/// region successors when entering the operation. Otherwise, this method
/// returns the successor regions when branching from the region indicated
/// by `point`.
/// 
/// Example: In the above example, this method returns the region of the
/// `scf.for` and this operation for either region branch point (`parent`
/// and the region of the `scf.for`). An implementation may choose to filter
/// out region successors when it is statically known (e.g., by examining
/// the operands of this op) that those successors are not branched to.
void mlir::RegionBranchOpInterface::getSuccessorRegions(::mlir::RegionBranchPoint point, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
      return getImpl()->getSuccessorRegions(getImpl(), getOperation(), point, regions);
  }
/// Populates `invocationBounds` with the minimum and maximum number of
/// times this operation will invoke the attached regions (assuming the
/// regions yield normally, i.e. do not abort or invoke an infinite loop).
/// The minimum number of invocations is at least 0. If the maximum number
/// of invocations cannot be statically determined, then it will be set to
/// `InvocationBounds::getUnknown()`.
/// 
/// This method also passes along the constant operands of this op.
/// `operands` contains an entry for every operand of this op, with a null
/// attribute if the operand has no constant value.
/// 
/// This method may be called speculatively on operations where the provided
/// operands are not necessarily the same as the operation's current
/// operands. This may occur in analyses that wish to determine "what would
/// be the region invocations if these were the operands?"
void mlir::RegionBranchOpInterface::getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
      return getImpl()->getRegionInvocationBounds(getImpl(), getOperation(), operands, invocationBounds);
  }
/// This method is called to compare types along control-flow edges. By
/// default, the types are checked as equal.
bool mlir::RegionBranchOpInterface::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return getImpl()->areTypesCompatible(getImpl(), getOperation(), lhs, rhs);
  }
/// Returns a mutable range of operands that are semantically "returned" by
/// passing them to the region successor indicated by `point`.
::mlir::MutableOperandRange mlir::RegionBranchTerminatorOpInterface::getMutableSuccessorOperands(::mlir::RegionBranchPoint point) {
      return getImpl()->getMutableSuccessorOperands(getImpl(), getOperation(), point);
  }
/// Returns the potential region successors that are branched to after this
/// terminator based on the given constant operands.
/// 
/// This method also passes along the constant operands of this op.
/// `operands` contains an entry for every operand of this op, with a null
/// attribute if the operand has no constant value.
/// 
/// The default implementation simply dispatches to the parent
/// `RegionBranchOpInterface`'s `getSuccessorRegions` implementation.
void mlir::RegionBranchTerminatorOpInterface::getSuccessorRegions(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
      return getImpl()->getSuccessorRegions(getImpl(), getOperation(), operands, regions);
  }
/// Returns the operand that would be chosen for a false condition.
::mlir::Value mlir::SelectLikeOpInterface::getFalseValue() {
      return getImpl()->getFalseValue(getImpl(), getOperation());
  }
/// Returns the operand that would be chosen for a true condition.
::mlir::Value mlir::SelectLikeOpInterface::getTrueValue() {
      return getImpl()->getTrueValue(getImpl(), getOperation());
  }
/// Returns the condition operand.
::mlir::Value mlir::SelectLikeOpInterface::getCondition() {
      return getImpl()->getCondition(getImpl(), getOperation());
  }
