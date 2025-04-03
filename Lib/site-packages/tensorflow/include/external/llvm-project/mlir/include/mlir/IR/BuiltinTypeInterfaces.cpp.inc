/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns a clone of this type with the given shape and element type.
/// 
/// If no shape is provided, the shape of this type is used. In that case, if
/// this type is unranked, so is the resulting type.
/// 
/// If a shape is provided, the resulting type is always ranked, even if this
/// type is unranked.
::mlir::ShapedType mlir::ShapedType::cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const {
      return getImpl()->cloneWith(getImpl(), *this, shape, elementType);
  }
/// Returns the element type of this shaped type.
::mlir::Type mlir::ShapedType::getElementType() const {
      return getImpl()->getElementType(getImpl(), *this);
  }
/// Returns if this type is ranked, i.e. it has a known number of dimensions.
bool mlir::ShapedType::hasRank() const {
      return getImpl()->hasRank(getImpl(), *this);
  }
/// Returns the shape of this type if it is ranked, otherwise asserts.
::llvm::ArrayRef<int64_t> mlir::ShapedType::getShape() const {
      return getImpl()->getShape(getImpl(), *this);
  }
