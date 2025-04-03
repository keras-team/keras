/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: BufferizationEnums.td                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {
::llvm::StringRef stringifyLayoutMapOption(LayoutMapOption val) {
  switch (val) {
    case LayoutMapOption::InferLayoutMap: return "InferLayoutMap";
    case LayoutMapOption::IdentityLayoutMap: return "IdentityLayoutMap";
    case LayoutMapOption::FullyDynamicLayoutMap: return "FullyDynamicLayoutMap";
  }
  return "";
}

::std::optional<LayoutMapOption> symbolizeLayoutMapOption(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<LayoutMapOption>>(str)
      .Case("InferLayoutMap", LayoutMapOption::InferLayoutMap)
      .Case("IdentityLayoutMap", LayoutMapOption::IdentityLayoutMap)
      .Case("FullyDynamicLayoutMap", LayoutMapOption::FullyDynamicLayoutMap)
      .Default(::std::nullopt);
}
::std::optional<LayoutMapOption> symbolizeLayoutMapOption(uint32_t value) {
  switch (value) {
  case 0: return LayoutMapOption::InferLayoutMap;
  case 1: return LayoutMapOption::IdentityLayoutMap;
  case 2: return LayoutMapOption::FullyDynamicLayoutMap;
  default: return ::std::nullopt;
  }
}

bool LayoutMapOptionAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 0)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 1)) || ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() == 2)));
}
LayoutMapOptionAttr LayoutMapOptionAttr::get(::mlir::MLIRContext *context, LayoutMapOption val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<LayoutMapOptionAttr>(baseAttr);
}
LayoutMapOption LayoutMapOptionAttr::getValue() const {
  return static_cast<LayoutMapOption>(::mlir::IntegerAttr::getInt());
}
} // namespace bufferization
} // namespace mlir

