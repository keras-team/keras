/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the source value for this copy operation
::mlir::Value mlir::CopyOpInterface::getSource() {
      return getImpl()->getSource(getImpl(), getOperation());
  }
/// Returns the target value for this copy operation
::mlir::Value mlir::CopyOpInterface::getTarget() {
      return getImpl()->getTarget(getImpl(), getOperation());
  }
