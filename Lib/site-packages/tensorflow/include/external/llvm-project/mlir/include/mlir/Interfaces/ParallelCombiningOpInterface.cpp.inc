/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return `idx`^th result of the parent operation.
::mlir::OpResult mlir::ParallelCombiningOpInterface::getParentResult(int64_t idx) {
      return getImpl()->getParentResult(getImpl(), getOperation(), idx);
  }
/// Return the contained ops that yield subvalues that this op combines to
/// yield to its parent.
::llvm::iterator_range<Block::iterator> mlir::ParallelCombiningOpInterface::getYieldingOps() {
      return getImpl()->getYieldingOps(getImpl(), getOperation());
  }
