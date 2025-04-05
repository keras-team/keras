/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vector {
class MaskableOpInterface;
namespace detail {
struct MaskableOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*isMasked)(const Concept *impl, ::mlir::Operation *);
    mlir::vector::MaskingOpInterface (*getMaskingOp)(const Concept *impl, ::mlir::Operation *);
    bool (*supportsPassthru)(const Concept *impl, ::mlir::Operation *);
    mlir::Type (*getExpectedMaskType)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::vector::MaskableOpInterface;
    Model() : Concept{isMasked, getMaskingOp, supportsPassthru, getExpectedMaskType} {}

    static inline bool isMasked(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::vector::MaskingOpInterface getMaskingOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool supportsPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::Type getExpectedMaskType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::vector::MaskableOpInterface;
    FallbackModel() : Concept{isMasked, getMaskingOp, supportsPassthru, getExpectedMaskType} {}

    static inline bool isMasked(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::vector::MaskingOpInterface getMaskingOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool supportsPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::Type getExpectedMaskType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool isMasked(::mlir::Operation *tablegen_opaque_val) const;
    mlir::vector::MaskingOpInterface getMaskingOp(::mlir::Operation *tablegen_opaque_val) const;
    bool supportsPassthru(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct MaskableOpInterfaceTrait;

} // namespace detail
class MaskableOpInterface : public ::mlir::OpInterface<MaskableOpInterface, detail::MaskableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MaskableOpInterface, detail::MaskableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MaskableOpInterfaceTrait<ConcreteOp> {};
  /// Returns true if the operation is masked by a MaskingOpInterface.
  bool isMasked();
  /// Returns the MaskingOpInterface masking this operation.
  mlir::vector::MaskingOpInterface getMaskingOp();
  /// Returns true if the operation can have a passthru argument when masked.
  bool supportsPassthru();
  /// Returns the mask type expected by this operation. Mostly used for verification purposes. It requires the operation to be vectorized.
  mlir::Type getExpectedMaskType();
};
namespace detail {
  template <typename ConcreteOp>
  struct MaskableOpInterfaceTrait : public ::mlir::OpInterface<MaskableOpInterface, detail::MaskableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns true if the operation is masked by a MaskingOpInterface.
    bool isMasked() {
      mlir::Operation *parentOp = (*static_cast<ConcreteOp *>(this))->getParentOp();
        return parentOp &&
               mlir::isa<mlir::vector::MaskingOpInterface>(parentOp);
    }
    /// Returns the MaskingOpInterface masking this operation.
    mlir::vector::MaskingOpInterface getMaskingOp() {
      return mlir::cast<mlir::vector::MaskingOpInterface>(
          (*static_cast<ConcreteOp *>(this))->getParentOp());
    }
    /// Returns true if the operation can have a passthru argument when masked.
    bool supportsPassthru() {
      return false;
    }
  };
}// namespace detail
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {
template<typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::Model<ConcreteOp>::isMasked(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isMasked();
}
template<typename ConcreteOp>
mlir::vector::MaskingOpInterface detail::MaskableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMaskingOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMaskingOp();
}
template<typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::Model<ConcreteOp>::supportsPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).supportsPassthru();
}
template<typename ConcreteOp>
mlir::Type detail::MaskableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getExpectedMaskType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getExpectedMaskType();
}
template<typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isMasked(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isMasked(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::vector::MaskingOpInterface detail::MaskableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMaskingOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMaskingOp(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::supportsPassthru(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->supportsPassthru(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::Type detail::MaskableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getExpectedMaskType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getExpectedMaskType(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isMasked(::mlir::Operation *tablegen_opaque_val) const {
mlir::Operation *parentOp = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getParentOp();
        return parentOp &&
               mlir::isa<mlir::vector::MaskingOpInterface>(parentOp);
}
template<typename ConcreteModel, typename ConcreteOp>
mlir::vector::MaskingOpInterface detail::MaskableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMaskingOp(::mlir::Operation *tablegen_opaque_val) const {
return mlir::cast<mlir::vector::MaskingOpInterface>(
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getParentOp());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::MaskableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::supportsPassthru(::mlir::Operation *tablegen_opaque_val) const {
return false;
}
} // namespace vector
} // namespace mlir
