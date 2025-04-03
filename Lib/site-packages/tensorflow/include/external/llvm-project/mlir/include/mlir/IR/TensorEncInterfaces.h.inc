/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class VerifiableTensorEncoding;
namespace detail {
struct VerifiableTensorEncodingInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::LogicalResult (*verifyEncoding)(const Concept *impl, ::mlir::Attribute , ::mlir::ArrayRef<int64_t>, ::mlir::Type, ::llvm::function_ref<::mlir::InFlightDiagnostic()>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::VerifiableTensorEncoding;
    Model() : Concept{verifyEncoding} {}

    static inline ::llvm::LogicalResult verifyEncoding(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::VerifiableTensorEncoding;
    FallbackModel() : Concept{verifyEncoding} {}

    static inline ::llvm::LogicalResult verifyEncoding(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct VerifiableTensorEncodingTrait;

} // namespace detail
class VerifiableTensorEncoding : public ::mlir::AttributeInterface<VerifiableTensorEncoding, detail::VerifiableTensorEncodingInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<VerifiableTensorEncoding, detail::VerifiableTensorEncodingInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::VerifiableTensorEncodingTrait<ConcreteAttr> {};
  /// Verifies the encoding is valid for a tensor type with the
  /// given shape and element type. Generates a diagnostic using
  /// the supplied callback on failure.
  ::llvm::LogicalResult verifyEncoding(::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct VerifiableTensorEncodingTrait : public ::mlir::AttributeInterface<VerifiableTensorEncoding, detail::VerifiableTensorEncodingInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteAttr>
::llvm::LogicalResult detail::VerifiableTensorEncodingInterfaceTraits::Model<ConcreteAttr>::verifyEncoding(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).verifyEncoding(shape, elementType, emitError);
}
template<typename ConcreteAttr>
::llvm::LogicalResult detail::VerifiableTensorEncodingInterfaceTraits::FallbackModel<ConcreteAttr>::verifyEncoding(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  return static_cast<const ConcreteAttr *>(impl)->verifyEncoding(tablegen_opaque_val, shape, elementType, emitError);
}
} // namespace mlir
