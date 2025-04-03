/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vector {
class MaskingOpInterface;
namespace detail {
struct MaskingOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::Value (*getMask)(const Concept *impl, ::mlir::Operation *);
    Operation *(*getMaskableOp)(const Concept *impl, ::mlir::Operation *);
    bool (*hasPassthru)(const Concept *impl, ::mlir::Operation *);
    mlir::Value (*getPassthru)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::vector::MaskingOpInterface;
    Model() : Concept{getMask, getMaskableOp, hasPassthru, getPassthru} {}

    static inline mlir::Value getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation *getMaskableOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::Value getPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::vector::MaskingOpInterface;
    FallbackModel() : Concept{getMask, getMaskableOp, hasPassthru, getPassthru} {}

    static inline mlir::Value getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation *getMaskableOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::Value getPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct MaskingOpInterfaceTrait;

} // namespace detail
class MaskingOpInterface : public ::mlir::OpInterface<MaskingOpInterface, detail::MaskingOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MaskingOpInterface, detail::MaskingOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MaskingOpInterfaceTrait<ConcreteOp> {};
  /// Returns the mask value of this masking operation.
  mlir::Value getMask();
  /// Returns the operation masked by this masking operation.
  Operation *getMaskableOp();
  /// Returns true if the masking operation has a passthru value.
  bool hasPassthru();
  /// Returns the passthru value of this masking operation.
  mlir::Value getPassthru();
};
namespace detail {
  template <typename ConcreteOp>
  struct MaskingOpInterfaceTrait : public ::mlir::OpInterface<MaskingOpInterface, detail::MaskingOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
template<typename ConcreteOp>
mlir::Value detail::MaskingOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMask();
}
template<typename ConcreteOp>
Operation *detail::MaskingOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMaskableOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMaskableOp();
}
template<typename ConcreteOp>
bool detail::MaskingOpInterfaceInterfaceTraits::Model<ConcreteOp>::hasPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasPassthru();
}
template<typename ConcreteOp>
mlir::Value detail::MaskingOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPassthru();
}
template<typename ConcreteOp>
mlir::Value detail::MaskingOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMask(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMask(tablegen_opaque_val);
}
template<typename ConcreteOp>
Operation *detail::MaskingOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMaskableOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMaskableOp(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::MaskingOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasPassthru(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::Value detail::MaskingOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPassthru(tablegen_opaque_val);
}
} // namespace vector
} // namespace mlir
