/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return mapping as an integer from the attribute.
int64_t mlir::DeviceMappingAttrInterface::getMappingId() const {
      return getImpl()->getMappingId(getImpl(), *this);
  }
/// Return true if the attribute specifies a linear mapping
bool mlir::DeviceMappingAttrInterface::isLinearMapping() const {
      return getImpl()->isLinearMapping(getImpl(), *this);
  }
/// Return the [0..n) relative index of the attribute depending on its group.
/// This can be used to index into a contiguous array.
int64_t mlir::DeviceMappingAttrInterface::getRelativeIndex() const {
      return getImpl()->getRelativeIndex(getImpl(), *this);
  }
