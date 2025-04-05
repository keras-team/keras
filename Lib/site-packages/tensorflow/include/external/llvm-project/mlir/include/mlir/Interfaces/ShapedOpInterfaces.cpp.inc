/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return the shaped value operand. This is the value that the dimension
/// is taken from.
::mlir::Value mlir::ShapedDimOpInterface::getShapedValue() {
      return getImpl()->getShapedValue(getImpl(), getOperation());
  }
/// Return the dimension operand. This can be a constant or an SSA value.
::mlir::OpFoldResult mlir::ShapedDimOpInterface::getDimension() {
      return getImpl()->getDimension(getImpl(), getOperation());
  }
