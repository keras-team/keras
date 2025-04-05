/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return the source tensor operand.
::mlir::OpOperand &mlir::SubsetExtractionOpInterface::getSourceOperand() {
      return getImpl()->getSourceOperand(getImpl(), getOperation());
  }
/// Return "true" if this op and the given candidate subset op operate on
/// equivalent subsets. Return "false" if the two subsets are disjoint
/// or cannot be proven to be equivalent.
/// 
/// This interface method does not have to be implemented if
/// `getAccessedHyperrectangularSlice` is implemented.
bool mlir::SubsetExtractionOpInterface::operatesOnEquivalentSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return getImpl()->implSubsetOpInterface->operatesOnEquivalentSubset(getImpl()->implSubsetOpInterface, getOperation(), candidate, equivalenceFn);
  }
/// Return "true" if this op and the given candidate subset op operate on
/// disjoint subsets. Return "false" if the two subsets are equivalent,
/// overlapping or cannot be proven to be disjoint.
/// 
/// This interface method does not have to be implemented if
/// `getAccessedHyperrectangularSlice` is implemented.
bool mlir::SubsetExtractionOpInterface::operatesOnDisjointSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return getImpl()->implSubsetOpInterface->operatesOnDisjointSubset(getImpl()->implSubsetOpInterface, getOperation(), candidate, equivalenceFn);
  }
/// If this op operates on a hyperrectangular subset, return a
/// description of the subset in terms of offsets, sizes and strides.
/// Otherwise, return "failure".
/// 
/// This interface method is a convenience method for the most common case
/// of hyperrectangular subset ops. It is optional. If it is implemented,
/// `operatesOnEquivalentSubset` and `operatesOnDisjointSubset` do not
/// have to be implemented.
::mlir::FailureOr<::mlir::HyperrectangularSlice> mlir::SubsetExtractionOpInterface::getAccessedHyperrectangularSlice() {
      return getImpl()->implSubsetOpInterface->getAccessedHyperrectangularSlice(getImpl()->implSubsetOpInterface, getOperation());
  }
/// Return the source operand. The source operand can have any type.
::mlir::OpOperand &mlir::SubsetInsertionOpInterface::getSourceOperand() {
      return getImpl()->getSourceOperand(getImpl(), getOperation());
  }
/// Return the destination operand. The destination operand must be a
/// tensor.
/// 
/// This function does not have to be implemented for destination style
/// ops that have exactly one "init" operand.
::mlir::OpOperand &mlir::SubsetInsertionOpInterface::getDestinationOperand() {
      return getImpl()->getDestinationOperand(getImpl(), getOperation());
  }
/// Return the updated destination result.
/// 
/// This function does not have to be implemented for destination style
/// ops.
::mlir::OpResult mlir::SubsetInsertionOpInterface::getUpdatedDestination() {
      return getImpl()->getUpdatedDestination(getImpl(), getOperation());
  }
/// Return "true" if this operation inserts into a subset that is
/// equivalent to the subset defined by `candidate`.
/// 
/// Two subsets are "equivalent" and "same" if they can bufferize to the
/// same buffer views/aliases. If they are "equivalent", the tensor IR
/// may be expressed in terms of different SSA values (but they could
/// bufferize to MemRef SSA values that can CSE without breaking
/// correctness). `equivalenceFn` should return "true" if the two given
/// values are equivalent.
/// 
/// Example:
/// ```
/// // The subset of the SubsetInsertionOpInterface op %1 is equivalent to
/// // the subset defined by %2 (but not "same"):
/// %0 = arith.select %c, %t, %t : tensor<?xf32>
/// %1 = tensor.insert_slice %x into %0[0][5][1]
///     : tensor<5xf32> into tensor<?xf32>
/// %2 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
/// 
/// // The subset of the SubsetInsertionOpInterface op %1 is equivalent to
/// // and "same" as the subset defined by %2.
/// %1 = tensor.insert_slice %x into %t[0][5][1]
///     : tensor<5xf32> into tensor<?xf32>
/// %2 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
/// ```
/// 
/// Note: By default, this function calls
/// `SubsetOpInterface::operatesOnEquivalentSubset`.
bool mlir::SubsetInsertionOpInterface::isEquivalentSubset(::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return getImpl()->isEquivalentSubset(getImpl(), getOperation(), candidate, equivalenceFn);
  }
/// Return the subset of the destination tensor that this operation
/// inserts into.
/// 
/// Example:
/// ```
/// // SubsetOpInterface op:
/// %0 = tensor.insert_slice %t0 into %t1[%pos][5][1]
///     : tensor<5xf32> into tensor<?xf32>
/// // Subset (built by this function):
/// %1 = tensor.extract_slice %t1[%pos][5][1]
///     : tensor<?xf32> to tensor<5xf32>
/// ```
/// 
/// Note: Implementations do not necessarily have to build new IR. They
/// may return existing SSA values.
::mlir::Value mlir::SubsetInsertionOpInterface::buildSubsetExtraction(::mlir::OpBuilder & builder, Location loc) {
      return getImpl()->buildSubsetExtraction(getImpl(), getOperation(), builder, loc);
  }
/// Return all SSA values that are needed (i.e., must be in scope) at the
/// insertion of the builder when calling `buildSubsetExtraction`. Users
/// of `buildSubsetExtraction` can use this helper method to find a
/// suitable insertion point.
/// 
/// Example: The SSA values needed to build the subset in the example of
/// `buildSubsetExtraction` are %t1 and %pos.
::llvm::SmallVector<::mlir::Value> mlir::SubsetInsertionOpInterface::getValuesNeededToBuildSubsetExtraction() {
      return getImpl()->getValuesNeededToBuildSubsetExtraction(getImpl(), getOperation());
  }
/// Return "true" if this op and the given candidate subset op operate on
/// equivalent subsets. Return "false" if the two subsets are disjoint
/// or cannot be proven to be equivalent.
/// 
/// This interface method does not have to be implemented if
/// `getAccessedHyperrectangularSlice` is implemented.
bool mlir::SubsetInsertionOpInterface::operatesOnEquivalentSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return getImpl()->implSubsetOpInterface->operatesOnEquivalentSubset(getImpl()->implSubsetOpInterface, getOperation(), candidate, equivalenceFn);
  }
/// Return "true" if this op and the given candidate subset op operate on
/// disjoint subsets. Return "false" if the two subsets are equivalent,
/// overlapping or cannot be proven to be disjoint.
/// 
/// This interface method does not have to be implemented if
/// `getAccessedHyperrectangularSlice` is implemented.
bool mlir::SubsetInsertionOpInterface::operatesOnDisjointSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return getImpl()->implSubsetOpInterface->operatesOnDisjointSubset(getImpl()->implSubsetOpInterface, getOperation(), candidate, equivalenceFn);
  }
/// If this op operates on a hyperrectangular subset, return a
/// description of the subset in terms of offsets, sizes and strides.
/// Otherwise, return "failure".
/// 
/// This interface method is a convenience method for the most common case
/// of hyperrectangular subset ops. It is optional. If it is implemented,
/// `operatesOnEquivalentSubset` and `operatesOnDisjointSubset` do not
/// have to be implemented.
::mlir::FailureOr<::mlir::HyperrectangularSlice> mlir::SubsetInsertionOpInterface::getAccessedHyperrectangularSlice() {
      return getImpl()->implSubsetOpInterface->getAccessedHyperrectangularSlice(getImpl()->implSubsetOpInterface, getOperation());
  }
/// Return "true" if this op and the given candidate subset op operate on
/// equivalent subsets. Return "false" if the two subsets are disjoint
/// or cannot be proven to be equivalent.
/// 
/// This interface method does not have to be implemented if
/// `getAccessedHyperrectangularSlice` is implemented.
bool mlir::SubsetOpInterface::operatesOnEquivalentSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return getImpl()->operatesOnEquivalentSubset(getImpl(), getOperation(), candidate, equivalenceFn);
  }
/// Return "true" if this op and the given candidate subset op operate on
/// disjoint subsets. Return "false" if the two subsets are equivalent,
/// overlapping or cannot be proven to be disjoint.
/// 
/// This interface method does not have to be implemented if
/// `getAccessedHyperrectangularSlice` is implemented.
bool mlir::SubsetOpInterface::operatesOnDisjointSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return getImpl()->operatesOnDisjointSubset(getImpl(), getOperation(), candidate, equivalenceFn);
  }
/// If this op operates on a hyperrectangular subset, return a
/// description of the subset in terms of offsets, sizes and strides.
/// Otherwise, return "failure".
/// 
/// This interface method is a convenience method for the most common case
/// of hyperrectangular subset ops. It is optional. If it is implemented,
/// `operatesOnEquivalentSubset` and `operatesOnDisjointSubset` do not
/// have to be implemented.
::mlir::FailureOr<::mlir::HyperrectangularSlice> mlir::SubsetOpInterface::getAccessedHyperrectangularSlice() {
      return getImpl()->getAccessedHyperrectangularSlice(getImpl(), getOperation());
  }
