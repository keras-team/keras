/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// For the given region of the operation return the block
/// inside the region, where an alloca-like operation should be inserted.
/// The default implementation returns the entry block of the region.
::mlir::Block *mlir::accomp::RecipeInterface::getAllocaBlock(::mlir::Region & region) {
      return getImpl()->getAllocaBlock(getImpl(), getOperation(), region);
  }
