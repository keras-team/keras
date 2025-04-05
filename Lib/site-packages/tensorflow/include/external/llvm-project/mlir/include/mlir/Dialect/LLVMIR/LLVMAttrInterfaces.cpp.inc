/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Get whether this attr describes a recursive self reference.
bool mlir::LLVM::DIRecursiveTypeAttrInterface::getIsRecSelf() const {
      return getImpl()->getIsRecSelf(getImpl(), *this);
  }
/// Get the recursive ID used for matching "rec-decl" with "rec-self".
/// If this attr instance is not recursive, return a null attribute.
DistinctAttr mlir::LLVM::DIRecursiveTypeAttrInterface::getRecId() const {
      return getImpl()->getRecId(getImpl(), *this);
  }
/// Get a copy of this type attr but with the recursive ID set to `recId`.
DIRecursiveTypeAttrInterface mlir::LLVM::DIRecursiveTypeAttrInterface::withRecId(DistinctAttr recId) const {
      return getImpl()->withRecId(getImpl(), *this, recId);
  }
/// Build a rec-self instance using the provided `recId`.
DIRecursiveTypeAttrInterface mlir::LLVM::DIRecursiveTypeAttrInterface::getRecSelf(DistinctAttr recId) const {
      return getImpl()->getRecSelf(recId);
  }
