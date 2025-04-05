/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return the `in_bounds` attribute name.
::mlir::StringRef mlir::VectorTransferOpInterface::getInBoundsAttrName() {
      return getImpl()->getInBoundsAttrName(getImpl(), getOperation());
  }
/// Return the `permutation_map` attribute name.
::mlir::StringRef mlir::VectorTransferOpInterface::getPermutationMapAttrName() {
      return getImpl()->getPermutationMapAttrName(getImpl(), getOperation());
  }
/// Return the optional in_bounds attribute that specifies for each vector
/// dimension whether it is in-bounds or not. (Broadcast dimensions are
/// always in-bounds).
::mlir::ArrayAttr mlir::VectorTransferOpInterface::getInBounds() {
      return getImpl()->getInBounds(getImpl(), getOperation());
  }
/// Return the memref or ranked tensor operand that this operation operates
/// on. In case of a "read" operation, that's the source from which the
/// operation reads. In case of a "write" operation, that's the destination
/// into which the operation writes.
/// TODO: Change name of operand, which is not accurate for xfer_write.
::mlir::Value mlir::VectorTransferOpInterface::getSource() {
      return getImpl()->getSource(getImpl(), getOperation());
  }
/// Return the vector that this operation operates on. In case of a "read",
/// that's the vector OpResult. In case of a "write", that's the vector
/// operand value that is written by the op.
::mlir::Value mlir::VectorTransferOpInterface::getVector() {
      return getImpl()->getVector(getImpl(), getOperation());
  }
/// Return the indices that specify the starting offsets into the source
/// operand. The starting offsets are guaranteed to be in-bounds.
::mlir::OperandRange mlir::VectorTransferOpInterface::getIndices() {
      return getImpl()->getIndices(getImpl(), getOperation());
  }
/// Return the permutation map that describes the mapping of vector
/// dimensions to source dimensions, as well as broadcast dimensions.
/// 
/// The permutation result has one result per vector transfer dimension.
/// Each result is either a dim expression, indicating the corresponding
/// dimension in the source operand, or a constant "0" expression,
/// indicating a broadcast dimension.
/// 
/// Note: Nested vector dimensions that are flattened by this op are not
/// accounted for in the permutation map. E.g.:
/// ```
/// // Vector type has rank 4, but permutation map has only 2 results. That
/// // is because there are only 2 transfer dimensions.
/// %0 = vector.transfer_read %arg1[%c3, %c3], %vf0
///     {permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
///     : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
/// ```
::mlir::AffineMap mlir::VectorTransferOpInterface::getPermutationMap() {
      return getImpl()->getPermutationMap(getImpl(), getOperation());
  }
/// Return the mask operand if the op has a mask. Otherwise, return an
/// empty value.
Value mlir::VectorTransferOpInterface::getMask() {
      return getImpl()->getMask(getImpl(), getOperation());
  }
/// Return the shape ratio of unrolling to the target vector shape
/// `targetShape`. Return `std::nullopt` if the op cannot be unrolled to the
/// target vector shape.
::std::optional<::llvm::SmallVector<int64_t, 4>> mlir::VectorUnrollOpInterface::getShapeForUnroll() {
      return getImpl()->getShapeForUnroll(getImpl(), getOperation());
  }
