/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Optional function for setting a predicate, which 
/// always returns a `PtxPredicate` value of type i1. If no predicate is 
/// provided, the instruction is unguarded; otherwise, it's guarded by the 
/// predicate value. The `PtxPredicate` value must always be the last argument. 
/// The provided PTX code by `getPtx` should not include the predicate usage.
/// The interface automatically handles predicate usage in the generated
/// PTX code when necessary.
std::optional<::mlir::Value> mlir::NVVM::BasicPtxBuilderInterface::getPredicate() {
      return getImpl()->getPredicate(getImpl(), getOperation());
  }
/// Returns PTX assembly with operand number.
std::string mlir::NVVM::BasicPtxBuilderInterface::getPtx() {
      return getImpl()->getPtx(getImpl(), getOperation());
  }
/// This function indicates whether the operation is supported by LLVM 
/// intrinsics. It's particularly useful for operations that have 
/// specific cases with LLVM intrinsic support.
bool mlir::NVVM::BasicPtxBuilderInterface::hasIntrinsic() {
      return getImpl()->hasIntrinsic(getImpl(), getOperation());
  }
/// Return whether the operation has memory side effects.
bool mlir::NVVM::BasicPtxBuilderInterface::hasSideEffect() {
      return getImpl()->hasSideEffect(getImpl(), getOperation());
  }
/// Helper function to generate i32 constant value.
::mlir::Value mlir::NVVM::BasicPtxBuilderInterface::makeConstantI32(::mlir::RewriterBase & rewriter, int val) {
      return getImpl()->makeConstantI32(getImpl(), getOperation(), rewriter, val);
  }
/// This function supplies the necessary arguments for passing PTX code,
/// following this order:
///  1) Adds results 
///  2) Adds operands 
///  3) Adds attributes             
void mlir::NVVM::BasicPtxBuilderInterface::getAsmValues(::mlir::RewriterBase & rewriter, llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>>& asmValues) {
      return getImpl()->getAsmValues(getImpl(), getOperation(), rewriter, asmValues);
  }
