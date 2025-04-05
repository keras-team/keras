/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Populate the constraint set with bounds for the given index-typed
/// value.
/// 
/// Note: If `value` is a block argument, it must belong to an entry block
/// of a region. Unstructured control flow graphs are not supported at the
/// moment.
void mlir::ValueBoundsOpInterface::populateBoundsForIndexValue(::mlir::Value value, ::mlir::ValueBoundsConstraintSet & cstr) {
      return getImpl()->populateBoundsForIndexValue(getImpl(), getOperation(), value, cstr);
  }
/// Populate the constraint set with bounds for the size of the specified
/// dimension of the given shaped value.
/// 
/// Note: If `value` is a block argument, it must belong to an entry block
/// of a region. Unstructured control flow graphs are not supported at the
/// moment.
void mlir::ValueBoundsOpInterface::populateBoundsForShapedValueDim(::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet & cstr) {
      return getImpl()->populateBoundsForShapedValueDim(getImpl(), getOperation(), value, dim, cstr);
  }
