/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVGPU.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::NVGPUDialect)
namespace mlir {
namespace nvgpu {

NVGPUDialect::NVGPUDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<NVGPUDialect>())
    
     {
  
  initialize();
}

NVGPUDialect::~NVGPUDialect() = default;

} // namespace nvgpu
} // namespace mlir
