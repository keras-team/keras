/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns true if the operation is masked by a MaskingOpInterface.
bool mlir::vector::MaskableOpInterface::isMasked() {
      return getImpl()->isMasked(getImpl(), getOperation());
  }
/// Returns the MaskingOpInterface masking this operation.
mlir::vector::MaskingOpInterface mlir::vector::MaskableOpInterface::getMaskingOp() {
      return getImpl()->getMaskingOp(getImpl(), getOperation());
  }
/// Returns true if the operation can have a passthru argument when masked.
bool mlir::vector::MaskableOpInterface::supportsPassthru() {
      return getImpl()->supportsPassthru(getImpl(), getOperation());
  }
/// Returns the mask type expected by this operation. Mostly used for verification purposes. It requires the operation to be vectorized.
mlir::Type mlir::vector::MaskableOpInterface::getExpectedMaskType() {
      return getImpl()->getExpectedMaskType(getImpl(), getOperation());
  }
