/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class ValueBoundsOpInterface;
namespace detail {
struct ValueBoundsOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*populateBoundsForIndexValue)(const Concept *impl, ::mlir::Operation *, ::mlir::Value, ::mlir::ValueBoundsConstraintSet &);
    void (*populateBoundsForShapedValueDim)(const Concept *impl, ::mlir::Operation *, ::mlir::Value, int64_t, ::mlir::ValueBoundsConstraintSet &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ValueBoundsOpInterface;
    Model() : Concept{populateBoundsForIndexValue, populateBoundsForShapedValueDim} {}

    static inline void populateBoundsForIndexValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, ::mlir::ValueBoundsConstraintSet & cstr);
    static inline void populateBoundsForShapedValueDim(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet & cstr);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ValueBoundsOpInterface;
    FallbackModel() : Concept{populateBoundsForIndexValue, populateBoundsForShapedValueDim} {}

    static inline void populateBoundsForIndexValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, ::mlir::ValueBoundsConstraintSet & cstr);
    static inline void populateBoundsForShapedValueDim(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet & cstr);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    void populateBoundsForIndexValue(::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, ::mlir::ValueBoundsConstraintSet &cstr) const;
    void populateBoundsForShapedValueDim(::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet &cstr) const;
  };
};
template <typename ConcreteOp>
struct ValueBoundsOpInterfaceTrait;

} // namespace detail
class ValueBoundsOpInterface : public ::mlir::OpInterface<ValueBoundsOpInterface, detail::ValueBoundsOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ValueBoundsOpInterface, detail::ValueBoundsOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ValueBoundsOpInterfaceTrait<ConcreteOp> {};
  /// Populate the constraint set with bounds for the given index-typed
  /// value.
  /// 
  /// Note: If `value` is a block argument, it must belong to an entry block
  /// of a region. Unstructured control flow graphs are not supported at the
  /// moment.
  void populateBoundsForIndexValue(::mlir::Value value, ::mlir::ValueBoundsConstraintSet & cstr);
  /// Populate the constraint set with bounds for the size of the specified
  /// dimension of the given shaped value.
  /// 
  /// Note: If `value` is a block argument, it must belong to an entry block
  /// of a region. Unstructured control flow graphs are not supported at the
  /// moment.
  void populateBoundsForShapedValueDim(::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet & cstr);
};
namespace detail {
  template <typename ConcreteOp>
  struct ValueBoundsOpInterfaceTrait : public ::mlir::OpInterface<ValueBoundsOpInterface, detail::ValueBoundsOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Populate the constraint set with bounds for the given index-typed
    /// value.
    /// 
    /// Note: If `value` is a block argument, it must belong to an entry block
    /// of a region. Unstructured control flow graphs are not supported at the
    /// moment.
    void populateBoundsForIndexValue(::mlir::Value value, ::mlir::ValueBoundsConstraintSet & cstr) {
      llvm_unreachable("not implemented");
    }
    /// Populate the constraint set with bounds for the size of the specified
    /// dimension of the given shaped value.
    /// 
    /// Note: If `value` is a block argument, it must belong to an entry block
    /// of a region. Unstructured control flow graphs are not supported at the
    /// moment.
    void populateBoundsForShapedValueDim(::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet & cstr) {
      llvm_unreachable("not implemented");
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
void detail::ValueBoundsOpInterfaceInterfaceTraits::Model<ConcreteOp>::populateBoundsForIndexValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, ::mlir::ValueBoundsConstraintSet & cstr) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).populateBoundsForIndexValue(value, cstr);
}
template<typename ConcreteOp>
void detail::ValueBoundsOpInterfaceInterfaceTraits::Model<ConcreteOp>::populateBoundsForShapedValueDim(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet & cstr) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).populateBoundsForShapedValueDim(value, dim, cstr);
}
template<typename ConcreteOp>
void detail::ValueBoundsOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::populateBoundsForIndexValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, ::mlir::ValueBoundsConstraintSet & cstr) {
  return static_cast<const ConcreteOp *>(impl)->populateBoundsForIndexValue(tablegen_opaque_val, value, cstr);
}
template<typename ConcreteOp>
void detail::ValueBoundsOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::populateBoundsForShapedValueDim(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet & cstr) {
  return static_cast<const ConcreteOp *>(impl)->populateBoundsForShapedValueDim(tablegen_opaque_val, value, dim, cstr);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::ValueBoundsOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::populateBoundsForIndexValue(::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, ::mlir::ValueBoundsConstraintSet &cstr) const {
llvm_unreachable("not implemented");
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::ValueBoundsOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::populateBoundsForShapedValueDim(::mlir::Operation *tablegen_opaque_val, ::mlir::Value value, int64_t dim, ::mlir::ValueBoundsConstraintSet &cstr) const {
llvm_unreachable("not implemented");
}
} // namespace mlir
