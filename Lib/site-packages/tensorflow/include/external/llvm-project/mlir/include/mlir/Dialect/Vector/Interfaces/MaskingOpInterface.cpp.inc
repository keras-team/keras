/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the mask value of this masking operation.
mlir::Value mlir::vector::MaskingOpInterface::getMask() {
      return getImpl()->getMask(getImpl(), getOperation());
  }
/// Returns the operation masked by this masking operation.
Operation *mlir::vector::MaskingOpInterface::getMaskableOp() {
      return getImpl()->getMaskableOp(getImpl(), getOperation());
  }
/// Returns true if the masking operation has a passthru value.
bool mlir::vector::MaskingOpInterface::hasPassthru() {
      return getImpl()->hasPassthru(getImpl(), getOperation());
  }
/// Returns the passthru value of this masking operation.
mlir::Value mlir::vector::MaskingOpInterface::getPassthru() {
      return getImpl()->getPassthru(getImpl(), getOperation());
  }
