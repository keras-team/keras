/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Destructures the type into subelements into a map of index attributes to
/// types of subelements. Returns nothing if the type cannot be destructured.
::std::optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>> mlir::DestructurableTypeInterface::getSubelementIndexMap() const {
      return getImpl()->getSubelementIndexMap(getImpl(), *this);
  }
/// Indicates which type is held at the provided index, returning a null
/// Type if no type could be computed. While this can return information
/// even when the type cannot be completely destructured, it must be coherent
/// with the types returned by `getSubelementIndexMap` when they exist.
::mlir::Type mlir::DestructurableTypeInterface::getTypeAtIndex(::mlir::Attribute index) const {
      return getImpl()->getTypeAtIndex(getImpl(), *this, index);
  }
