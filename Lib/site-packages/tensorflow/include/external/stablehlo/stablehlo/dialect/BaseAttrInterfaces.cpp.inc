/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Get the attribute's bounds
::llvm::ArrayRef<int64_t> mlir::hlo::BoundedAttrInterface::getBounds() const {
      return getImpl()->getBounds(getImpl(), *this);
  }
