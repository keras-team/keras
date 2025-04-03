/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the first operation in atomic capture region.
::mlir::Operation *mlir::accomp::AtomicCaptureOpInterface::getFirstOp() {
      return getImpl()->getFirstOp(getImpl(), getOperation());
  }
/// Returns the second operation in atomic capture region.
::mlir::Operation *mlir::accomp::AtomicCaptureOpInterface::getSecondOp() {
      return getImpl()->getSecondOp(getImpl(), getOperation());
  }
/// Common verifier of the required region for operation that implements
/// atomic capture interface.
::llvm::LogicalResult mlir::accomp::AtomicCaptureOpInterface::verifyRegionsCommon() {
      return getImpl()->verifyRegionsCommon(getImpl(), getOperation());
  }
/// Common verifier for operation that implements atomic read interface.
::llvm::LogicalResult mlir::accomp::AtomicReadOpInterface::verifyCommon() {
      return getImpl()->verifyCommon(getImpl(), getOperation());
  }
/// Obtains `x` which is the address from where the value is atomically
/// read.
::mlir::Value mlir::accomp::AtomicReadOpInterface::getX() {
      return getImpl()->getX(getImpl(), getOperation());
  }
/// Obtains `v` which is the address where the value is stored after
/// reading.
::mlir::Value mlir::accomp::AtomicReadOpInterface::getV() {
      return getImpl()->getV(getImpl(), getOperation());
  }
/// Obtains `x` which is the address to which the value is atomically
/// written to / read from.
::mlir::Value mlir::accomp::AtomicUpdateOpInterface::getX() {
      return getImpl()->getX(getImpl(), getOperation());
  }
/// Returns the first operation in atomic update region.
::mlir::Operation *mlir::accomp::AtomicUpdateOpInterface::getFirstOp() {
      return getImpl()->getFirstOp(getImpl(), getOperation());
  }
/// Returns true if the new value is same as old value and the operation is
/// a no-op, false otherwise.
bool mlir::accomp::AtomicUpdateOpInterface::isNoOp() {
      return getImpl()->isNoOp(getImpl(), getOperation());
  }
/// Returns the new value if the operation is equivalent to just a write
/// operation. Otherwise, returns nullptr.
::mlir::Value mlir::accomp::AtomicUpdateOpInterface::getWriteOpVal() {
      return getImpl()->getWriteOpVal(getImpl(), getOperation());
  }
/// Common verifier for operation that implements atomic update interface.
::llvm::LogicalResult mlir::accomp::AtomicUpdateOpInterface::verifyCommon() {
      return getImpl()->verifyCommon(getImpl(), getOperation());
  }
/// Common verifier of the required region for operation that implements
/// atomic update interface.
::llvm::LogicalResult mlir::accomp::AtomicUpdateOpInterface::verifyRegionsCommon() {
      return getImpl()->verifyRegionsCommon(getImpl(), getOperation());
  }
/// Common verifier for operation that implements atomic write interface.
::llvm::LogicalResult mlir::accomp::AtomicWriteOpInterface::verifyCommon() {
      return getImpl()->verifyCommon(getImpl(), getOperation());
  }
/// Obtains `x` which is the address to which the value is atomically
/// written to.
::mlir::Value mlir::accomp::AtomicWriteOpInterface::getX() {
      return getImpl()->getX(getImpl(), getOperation());
  }
/// Obtains `expr` which corresponds to the expression whose value is
/// written to `x`.
::mlir::Value mlir::accomp::AtomicWriteOpInterface::getExpr() {
      return getImpl()->getExpr(getImpl(), getOperation());
  }
