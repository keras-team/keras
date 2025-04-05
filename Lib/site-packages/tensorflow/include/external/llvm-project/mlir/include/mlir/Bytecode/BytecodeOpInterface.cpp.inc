/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Read the properties for this operation from the bytecode and populate the state.
LogicalResult mlir::BytecodeOpInterface::readProperties(::mlir::DialectBytecodeReader & reader, ::mlir::OperationState & state) {
      return getImpl()->readProperties(reader, state);
  }
/// Write the properties for this operation to the bytecode.
void mlir::BytecodeOpInterface::writeProperties(::mlir::DialectBytecodeWriter & writer) {
      return getImpl()->writeProperties(getImpl(), getOperation(), writer);
  }
