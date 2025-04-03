/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns true if the given set of input and result types are compatible
/// to cast using this cast operation.
bool mlir::CastOpInterface::areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs) {
      return getImpl()->areCastCompatible(inputs, outputs);
  }
