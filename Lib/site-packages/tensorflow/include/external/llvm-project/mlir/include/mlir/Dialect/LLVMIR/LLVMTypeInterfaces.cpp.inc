/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the size of the type in bytes.
unsigned mlir::LLVM::PointerElementTypeInterface::getSizeInBytes(const ::mlir::DataLayout & dataLayout) const {
      return getImpl()->getSizeInBytes(getImpl(), *this, dataLayout);
  }
