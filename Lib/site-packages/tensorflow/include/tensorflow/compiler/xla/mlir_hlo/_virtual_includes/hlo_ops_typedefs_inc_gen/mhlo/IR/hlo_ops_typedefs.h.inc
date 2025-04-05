/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace mhlo {
class AsyncBundleType;
class AsyncBundleType : public ::mlir::Type::TypeBase<AsyncBundleType, ::mlir::Type, detail::AsyncBundleTypeStorage> {
public:
  using Base::Base;
  // Return the number of held types.
  size_t size() const;

  // Iterate over the held elements.
  using iterator = ArrayRef<Type>::iterator;
  iterator begin() const { return getTypes().begin(); }
  iterator end() const { return getTypes().end(); }

  // Return the element type at index 'index'.
  Type getType(size_t index) const {
    assert(index < size() && "invalid index for tuple type");
    return getTypes()[index];
  }

  // Puts a flattened list of types in this bundle into `types`
  void getFlattenedTypes(SmallVectorImpl<Type> &types);
  static SmallVector<Type, 10> getFlattenedTypes(AsyncBundleType t) {
    SmallVector<Type, 10> fTypes;
    t.getFlattenedTypes(fTypes);
    return fTypes;
  }

  static constexpr ::llvm::StringLiteral name = "mhlo.async_bundle";
  static constexpr ::llvm::StringLiteral dialectName = "mhlo";
  static AsyncBundleType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Type> types);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"async_bundle"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<Type> getTypes() const;
};
} // namespace mhlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mhlo::AsyncBundleType)

#endif  // GET_TYPEDEF_CLASSES

