/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Builds a deallocation operation using the provided builder and the
/// current allocation value (which refers to the current Op implementing
/// this interface). The allocation value is a result of the current
/// operation implementing this interface. If there is no compatible
/// deallocation operation, this method can return ::std::nullopt.
::std::optional<::mlir::Operation*> mlir::bufferization::AllocationOpInterface::buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return getImpl()->buildDealloc(builder, alloc);
  }
/// Builds a clone operation using the provided builder and the current
/// allocation value (which refers to the current Op implementing this
/// interface). The allocation value is a result of the current operation
/// implementing this interface. If there is no compatible clone operation,
/// this method can return ::std::nullopt.
::std::optional<::mlir::Value> mlir::bufferization::AllocationOpInterface::buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return getImpl()->buildClone(builder, alloc);
  }
/// Returns the kind of hoisting supported for the buffer allocated by this
/// operation.
::mlir::HoistingKind mlir::bufferization::AllocationOpInterface::getHoistingKind() {
      return getImpl()->getHoistingKind();
  }
/// Builds a stack allocation operation using the provided builder and the
/// current allocation value (which refers to the current Op implementing this
/// interface). The allocation value is a result of the current
/// operation implementing this interface. If there is no compatible
/// stack allocation operation, this method can return ::std::nullopt.
::std::optional<::mlir::Operation*> mlir::bufferization::AllocationOpInterface::buildPromotedAlloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return getImpl()->buildPromotedAlloc(builder, alloc);
  }
