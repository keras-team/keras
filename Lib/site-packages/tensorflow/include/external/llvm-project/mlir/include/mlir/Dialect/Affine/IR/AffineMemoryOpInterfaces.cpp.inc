/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the AffineMapAttr associated with 'memref'.
::mlir::NamedAttribute mlir::affine::AffineMapAccessInterface::getAffineMapAttrForMemRef(::mlir::Value memref) {
      return getImpl()->getAffineMapAttrForMemRef(getImpl(), getOperation(), memref);
  }
/// Returns the memref operand to read from.
::mlir::Value mlir::affine::AffineReadOpInterface::getMemRef() {
      return getImpl()->getMemRef(getImpl(), getOperation());
  }
/// Returns the type of the memref operand.
::mlir::MemRefType mlir::affine::AffineReadOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getImpl(), getOperation());
  }
/// Returns affine map operands.
::mlir::Operation::operand_range mlir::affine::AffineReadOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getImpl(), getOperation());
  }
/// Returns the affine map used to index the memref for this operation.
::mlir::AffineMap mlir::affine::AffineReadOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getImpl(), getOperation());
  }
/// Returns the value read by this operation.
::mlir::Value mlir::affine::AffineReadOpInterface::getValue() {
      return getImpl()->getValue(getImpl(), getOperation());
  }
/// Returns the memref operand to write to.
::mlir::Value mlir::affine::AffineWriteOpInterface::getMemRef() {
      return getImpl()->getMemRef(getImpl(), getOperation());
  }
/// Returns the type of the memref operand.
::mlir::MemRefType mlir::affine::AffineWriteOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getImpl(), getOperation());
  }
/// Returns affine map operands.
::mlir::Operation::operand_range mlir::affine::AffineWriteOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getImpl(), getOperation());
  }
/// Returns the affine map used to index the memref for this operation.
::mlir::AffineMap mlir::affine::AffineWriteOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getImpl(), getOperation());
  }
/// Returns the value to store.
::mlir::Value mlir::affine::AffineWriteOpInterface::getValueToStore() {
      return getImpl()->getValueToStore(getImpl(), getOperation());
  }
