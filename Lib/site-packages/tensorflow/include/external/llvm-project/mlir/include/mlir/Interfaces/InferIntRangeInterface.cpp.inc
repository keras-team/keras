/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

///  Infer the bounds on the results of this op given the bounds on its arguments.
///  For each result value or block argument (that isn't a branch argument,
///  since the dataflow analysis handles those case), the method should call
///  `setValueRange` with that `Value` as an argument. When implemented,
///  `setValueRange` should be called on all result values for the operation.
///  When operations take non-integer inputs, the
/// `inferResultRangesFromOptional` method should be implemented instead.
/// 
///  When called on an op that also implements the RegionBranchOpInterface
///  or BranchOpInterface, this method should not attempt to infer the values
///  of the branch results, as this will be handled by the analyses that use
///  this interface.
/// 
///  This function will only be called when at least one result of the op is a
///  scalar integer value or the op has a region.
void mlir::InferIntRangeInterface::inferResultRanges(::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) {
      return getImpl()->inferResultRanges(getImpl(), getOperation(), argRanges, setResultRanges);
  }
/// Infer the bounds on the results of this op given the lattice representation
/// of the bounds for its arguments. For each result value or block argument
/// (that isn't a branch argument, since the dataflow analysis handles
/// those case), the method should call `setValueRange` with that `Value`
/// as an argument. When implemented, `setValueRange` should be called on
/// all result values for the operation.
/// 
/// This method allows for more precise implementations when operations
/// want to reason about inputs which may be undefined during the analysis.
void mlir::InferIntRangeInterface::inferResultRangesFromOptional(::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges) {
      return getImpl()->inferResultRangesFromOptional(getImpl(), getOperation(), argRanges, setResultRanges);
  }
