/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Serializes a GPU module to a string containing a representation of the
/// module.
/// 
/// If serialization fails then the method should return `std::nullopt`.
/// 
/// The `module` parameter must be a GPU Module Op. The `options` parameter
/// is meant to be used for passing additional options that are not in the
/// attribute.
std::optional<::mlir::SmallVector<char, 0>> mlir::gpu::TargetAttrInterface::serializeToObject(::mlir::Operation* module, const ::mlir::gpu::TargetOptions& options) const {
      return getImpl()->serializeToObject(getImpl(), *this, module, options);
  }
/// Creates a GPU object attribute from a binary string.
/// 
/// The `module` parameter must be a `GPUModuleOp` and can be used to
/// retrieve additional information like the list of kernels in the binary.
/// The `object` parameter is a binary string. The `options` parameter is
/// meant to be used for passing additional options that are not in the
/// attribute.
::mlir::Attribute mlir::gpu::TargetAttrInterface::createObject(::mlir::Operation * module, const ::llvm::SmallVector<char, 0> & object, const ::mlir::gpu::TargetOptions & options) const {
      return getImpl()->createObject(getImpl(), *this, module, object, options);
  }
/// Translates a `gpu.binary` Op into a sequence of LLVM IR target-specific
/// instructions, embedding the binary into a host LLVM module.
/// 
/// The LLVM translation mechanism invokes this function when translating a
/// `gpu.binary`.
/// 
/// The first argument has to be a GPU binary operation.
/// If the function fails at any point, it must return `failure`.
::llvm::LogicalResult mlir::gpu::OffloadingLLVMTranslationAttrInterface::embedBinary(::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation) const {
      return getImpl()->embedBinary(getImpl(), *this, binaryOp, hostBuilder, hostModuleTranslation);
  }
/// Translates a `gpu.launch_func` op into a sequence of LLVM IR
/// target-specific instructions, resulting in a kernel launch on host IR.
/// 
/// The LLVM translation mechanism invokes this function when translating a
/// `gpu.launch_func` operation; it searches the appropriate binary and uses
/// its offloading handler.
/// 
/// The first two arguments must be GPU launch and binary operations,
/// respectively. If the function fails at any point, it must return
/// `failure`.
::llvm::LogicalResult mlir::gpu::OffloadingLLVMTranslationAttrInterface::launchKernel(::mlir::Operation* launchFunc, ::mlir::Operation* binaryOp, ::llvm::IRBuilderBase& hostBuilder, ::mlir::LLVM::ModuleTranslation& hostModuleTranslation) const {
      return getImpl()->launchKernel(getImpl(), *this, launchFunc, binaryOp, hostBuilder, hostModuleTranslation);
  }
