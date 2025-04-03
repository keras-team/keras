/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Query the operands that represent async dependency tokens.
OperandRange mlir::gpu::AsyncOpInterface::getAsyncDependencies() {
      return getImpl()->getAsyncDependencies(getImpl(), getOperation());
  }
/// Adds a new token to the list of async dependencies if it is not already there.
void mlir::gpu::AsyncOpInterface::addAsyncDependency(Value token) {
      return getImpl()->addAsyncDependency(getImpl(), getOperation(), token);
  }
/// Query the result that represents the async token to depend on.
Value mlir::gpu::AsyncOpInterface::getAsyncToken() {
      return getImpl()->getAsyncToken(getImpl(), getOperation());
  }
