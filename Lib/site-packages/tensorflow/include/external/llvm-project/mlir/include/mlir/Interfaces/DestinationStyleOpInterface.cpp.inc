/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return start and end indices of the init operands range.
::mlir::MutableOperandRange mlir::DestinationStyleOpInterface::getDpsInitsMutable() {
      return getImpl()->getDpsInitsMutable(getImpl(), getOperation());
  }
