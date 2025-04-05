
bool isValidVectorTypeElementType(::mlir::Type type) {
  return (((::llvm::isa<::mlir::IntegerType>(type))) || ((::llvm::isa<::mlir::IndexType>(type))) || ((::llvm::isa<::mlir::FloatType>(type))));
}
