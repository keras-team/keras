/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return true if the operation needs an extra sort to produce the final result.
bool mlir::sparse_tensor::StageWithSortSparseOp::needsExtraSort() {
      return getImpl()->needsExtraSort(getImpl(), getOperation());
  }
/// Stage the operation, return the final result value after staging.
::llvm::LogicalResult mlir::sparse_tensor::StageWithSortSparseOp::stageWithSort(::mlir::PatternRewriter & rewriter, Value & tmpBuf) {
      return getImpl()->stageWithSort(getImpl(), getOperation(), rewriter, tmpBuf);
  }
