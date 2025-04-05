/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns a FastMathFlagsAttr attribute for the operation
FastMathFlagsAttr mlir::arith::ArithFastMathInterface::getFastMathFlagsAttr() {
      return getImpl()->getFastMathFlagsAttr(getImpl(), getOperation());
  }
/// Returns the name of the FastMathFlagsAttr attribute
///                          for the operation
StringRef mlir::arith::ArithFastMathInterface::getFastMathAttrName() {
      return getImpl()->getFastMathAttrName();
  }
/// Returns an IntegerOverflowFlagsAttr attribute for the operation
IntegerOverflowFlagsAttr mlir::arith::ArithIntegerOverflowFlagsInterface::getOverflowAttr() {
      return getImpl()->getOverflowAttr(getImpl(), getOperation());
  }
/// Returns whether the operation has the No Unsigned Wrap keyword
bool mlir::arith::ArithIntegerOverflowFlagsInterface::hasNoUnsignedWrap() {
      return getImpl()->hasNoUnsignedWrap(getImpl(), getOperation());
  }
/// Returns whether the operation has the No Signed Wrap keyword
bool mlir::arith::ArithIntegerOverflowFlagsInterface::hasNoSignedWrap() {
      return getImpl()->hasNoSignedWrap(getImpl(), getOperation());
  }
/// Returns the name of the IntegerOverflowFlagsAttr attribute
///                          for the operation
StringRef mlir::arith::ArithIntegerOverflowFlagsInterface::getIntegerOverflowAttrName() {
      return getImpl()->getIntegerOverflowAttrName();
  }
/// Returns a RoundingModeAttr attribute for the operation
RoundingModeAttr mlir::arith::ArithRoundingModeInterface::getRoundingModeAttr() {
      return getImpl()->getRoundingModeAttr(getImpl(), getOperation());
  }
/// Returns the name of the RoundingModeAttr attribute for
///                          the operation
StringRef mlir::arith::ArithRoundingModeInterface::getRoundingModeAttrName() {
      return getImpl()->getRoundingModeAttrName();
  }
