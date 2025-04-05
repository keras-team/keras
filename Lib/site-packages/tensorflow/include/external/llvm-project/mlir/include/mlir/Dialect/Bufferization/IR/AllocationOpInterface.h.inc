/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {
class AllocationOpInterface;
namespace detail {
struct AllocationOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::std::optional<::mlir::Operation*> (*buildDealloc)(::mlir::OpBuilder&, ::mlir::Value);
    ::std::optional<::mlir::Value> (*buildClone)(::mlir::OpBuilder&, ::mlir::Value);
    ::mlir::HoistingKind (*getHoistingKind)();
    ::std::optional<::mlir::Operation*> (*buildPromotedAlloc)(::mlir::OpBuilder&, ::mlir::Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::bufferization::AllocationOpInterface;
    Model() : Concept{buildDealloc, buildClone, getHoistingKind, buildPromotedAlloc} {}

    static inline ::std::optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
    static inline ::std::optional<::mlir::Value> buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc);
    static inline ::mlir::HoistingKind getHoistingKind();
    static inline ::std::optional<::mlir::Operation*> buildPromotedAlloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::bufferization::AllocationOpInterface;
    FallbackModel() : Concept{buildDealloc, buildClone, getHoistingKind, buildPromotedAlloc} {}

    static inline ::std::optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
    static inline ::std::optional<::mlir::Value> buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc);
    static inline ::mlir::HoistingKind getHoistingKind();
    static inline ::std::optional<::mlir::Operation*> buildPromotedAlloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    static ::std::optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder&builder, ::mlir::Value alloc);
    static ::std::optional<::mlir::Value> buildClone(::mlir::OpBuilder&builder, ::mlir::Value alloc);
    static ::mlir::HoistingKind getHoistingKind();
    static ::std::optional<::mlir::Operation*> buildPromotedAlloc(::mlir::OpBuilder&builder, ::mlir::Value alloc);
  };
};
template <typename ConcreteOp>
struct AllocationOpInterfaceTrait;

} // namespace detail
class AllocationOpInterface : public ::mlir::OpInterface<AllocationOpInterface, detail::AllocationOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AllocationOpInterface, detail::AllocationOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AllocationOpInterfaceTrait<ConcreteOp> {};
  /// Builds a deallocation operation using the provided builder and the
  /// current allocation value (which refers to the current Op implementing
  /// this interface). The allocation value is a result of the current
  /// operation implementing this interface. If there is no compatible
  /// deallocation operation, this method can return ::std::nullopt.
  ::std::optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
  /// Builds a clone operation using the provided builder and the current
  /// allocation value (which refers to the current Op implementing this
  /// interface). The allocation value is a result of the current operation
  /// implementing this interface. If there is no compatible clone operation,
  /// this method can return ::std::nullopt.
  ::std::optional<::mlir::Value> buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc);
  /// Returns the kind of hoisting supported for the buffer allocated by this
  /// operation.
  ::mlir::HoistingKind getHoistingKind();
  /// Builds a stack allocation operation using the provided builder and the
  /// current allocation value (which refers to the current Op implementing this
  /// interface). The allocation value is a result of the current
  /// operation implementing this interface. If there is no compatible
  /// stack allocation operation, this method can return ::std::nullopt.
  ::std::optional<::mlir::Operation*> buildPromotedAlloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
};
namespace detail {
  template <typename ConcreteOp>
  struct AllocationOpInterfaceTrait : public ::mlir::OpInterface<AllocationOpInterface, detail::AllocationOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Builds a deallocation operation using the provided builder and the
    /// current allocation value (which refers to the current Op implementing
    /// this interface). The allocation value is a result of the current
    /// operation implementing this interface. If there is no compatible
    /// deallocation operation, this method can return ::std::nullopt.
    static ::std::optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return std::nullopt;
    }
    /// Builds a clone operation using the provided builder and the current
    /// allocation value (which refers to the current Op implementing this
    /// interface). The allocation value is a result of the current operation
    /// implementing this interface. If there is no compatible clone operation,
    /// this method can return ::std::nullopt.
    static ::std::optional<::mlir::Value> buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return std::nullopt;
    }
    /// Returns the kind of hoisting supported for the buffer allocated by this
    /// operation.
    static ::mlir::HoistingKind getHoistingKind() {
      return HoistingKind::None;
    }
    /// Builds a stack allocation operation using the provided builder and the
    /// current allocation value (which refers to the current Op implementing this
    /// interface). The allocation value is a result of the current
    /// operation implementing this interface. If there is no compatible
    /// stack allocation operation, this method can return ::std::nullopt.
    static ::std::optional<::mlir::Operation*> buildPromotedAlloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return std::nullopt;
    }
  };
}// namespace detail
} // namespace bufferization
} // namespace mlir
namespace mlir {
namespace bufferization {
template<typename ConcreteOp>
::std::optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildDealloc(builder, alloc);
}
template<typename ConcreteOp>
::std::optional<::mlir::Value> detail::AllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildClone(builder, alloc);
}
template<typename ConcreteOp>
::mlir::HoistingKind detail::AllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::getHoistingKind() {
  return ConcreteOp::getHoistingKind();
}
template<typename ConcreteOp>
::std::optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::buildPromotedAlloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildPromotedAlloc(builder, alloc);
}
template<typename ConcreteOp>
::std::optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildDealloc(builder, alloc);
}
template<typename ConcreteOp>
::std::optional<::mlir::Value> detail::AllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildClone(builder, alloc);
}
template<typename ConcreteOp>
::mlir::HoistingKind detail::AllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getHoistingKind() {
  return ConcreteOp::getHoistingKind();
}
template<typename ConcreteOp>
::std::optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::buildPromotedAlloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildPromotedAlloc(builder, alloc);
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::buildDealloc(::mlir::OpBuilder&builder, ::mlir::Value alloc) {
return std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::mlir::Value> detail::AllocationOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::buildClone(::mlir::OpBuilder&builder, ::mlir::Value alloc) {
return std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::HoistingKind detail::AllocationOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getHoistingKind() {
return HoistingKind::None;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::buildPromotedAlloc(::mlir::OpBuilder&builder, ::mlir::Value alloc) {
return std::nullopt;
}
} // namespace bufferization
} // namespace mlir
