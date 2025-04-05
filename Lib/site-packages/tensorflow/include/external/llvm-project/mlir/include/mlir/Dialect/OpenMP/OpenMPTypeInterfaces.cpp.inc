/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the pointee type or null if the pointer has no pointee type
::mlir::Type mlir::omp::PointerLikeType::getElementType() const {
      return getImpl()->getElementType(getImpl(), *this);
  }
