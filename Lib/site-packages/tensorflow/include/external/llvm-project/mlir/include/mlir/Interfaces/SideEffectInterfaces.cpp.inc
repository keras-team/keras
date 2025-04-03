/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns value indicating whether the specific operation in question can
/// be speculatively executed.  Please see the documentation on the
/// Speculatability enum to know how to interpret the return value.
::mlir::Speculation::Speculatability mlir::ConditionallySpeculatable::getSpeculatability() {
      return getImpl()->getSpeculatability(getImpl(), getOperation());
  }
/// Collects all of the operation's effects into `effects`.
void mlir::MemoryEffectOpInterface::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> & effects) {
      return getImpl()->getEffects(getImpl(), getOperation(), effects);
  }
