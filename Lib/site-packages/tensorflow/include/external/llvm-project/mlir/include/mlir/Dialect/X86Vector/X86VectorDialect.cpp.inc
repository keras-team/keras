/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: X86Vector.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::x86vector::X86VectorDialect)
namespace mlir {
namespace x86vector {

X86VectorDialect::X86VectorDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<X86VectorDialect>())
    
     {
  
  initialize();
}

X86VectorDialect::~X86VectorDialect() = default;

} // namespace x86vector
} // namespace mlir
