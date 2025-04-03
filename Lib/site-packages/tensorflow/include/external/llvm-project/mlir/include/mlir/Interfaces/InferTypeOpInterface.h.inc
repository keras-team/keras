/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class InferTypeOpInterface;
namespace detail {
struct InferTypeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::LogicalResult (*inferReturnTypes)(::mlir::MLIRContext *, ::std::optional<::mlir::Location>, ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::OpaqueProperties, ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::Type>&);
    ::llvm::LogicalResult (*refineReturnTypes)(::mlir::MLIRContext *, ::std::optional<::mlir::Location>, ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::OpaqueProperties, ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::Type>&);
    bool (*isCompatibleReturnTypes)(::mlir::TypeRange, ::mlir::TypeRange);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::InferTypeOpInterface;
    Model() : Concept{inferReturnTypes, refineReturnTypes, isCompatibleReturnTypes} {}

    static inline ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
    static inline ::llvm::LogicalResult refineReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& returnTypes);
    static inline bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::InferTypeOpInterface;
    FallbackModel() : Concept{inferReturnTypes, refineReturnTypes, isCompatibleReturnTypes} {}

    static inline ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
    static inline ::llvm::LogicalResult refineReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& returnTypes);
    static inline bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    static ::llvm::LogicalResult refineReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&returnTypes);
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
  };
};
template <typename ConcreteOp>
struct InferTypeOpInterfaceTrait;

} // namespace detail
class InferTypeOpInterface : public ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::InferTypeOpInterfaceTrait<ConcreteOp> {};
  /// Infer the return types that an op would generate.
  /// 
  ///       The method takes an optional location which, if set, will be used to
  ///       report errors on. The operands and attributes correspond to those with
  ///       which an Operation would be created (e.g., as used in Operation::create)
  ///       and the regions of the op. Be aware that this method is supposed to be
  ///       called with valid arguments, e.g., operands are verified, or it may result
  ///       in an undefined behavior.
  ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
  /// Refine the return types that an op would generate.
  /// 
  ///       This method computes the return types as `inferReturnTypes` does but
  ///       additionally takes the existing result types as input. The existing
  ///       result types can be checked as part of inference to provide more
  ///       op-specific error messages as well as part of inference to merge
  ///       additional information, attributes, during inference. It is called during
  ///       verification for ops implementing this trait with default behavior
  ///       reporting mismatch with current and inferred types printed.
  /// 
  ///       The operands and attributes correspond to those with which an Operation
  ///       would be created (e.g., as used in Operation::create) and the regions of
  ///       the op. The method takes an optional location which, if set, will be used
  ///       to report errors on.
  /// 
  ///       The return types may be elided or specific elements be null for elements
  ///       that should just be returned but not verified.
  /// 
  ///       Because this method can be called from within different stages of IR
  ///       verification, implementations should not assume the arguments to
  ///       represent fully valid IR and are responsible for checking inputs for
  ///       validity to the degree necessary to perform the return type inference.
  ::llvm::LogicalResult refineReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& returnTypes);
  /// Returns whether two array of types are compatible result types for an op.
  bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
};
namespace detail {
  template <typename ConcreteOp>
  struct InferTypeOpInterfaceTrait : public ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Refine the return types that an op would generate.
    /// 
    ///       This method computes the return types as `inferReturnTypes` does but
    ///       additionally takes the existing result types as input. The existing
    ///       result types can be checked as part of inference to provide more
    ///       op-specific error messages as well as part of inference to merge
    ///       additional information, attributes, during inference. It is called during
    ///       verification for ops implementing this trait with default behavior
    ///       reporting mismatch with current and inferred types printed.
    /// 
    ///       The operands and attributes correspond to those with which an Operation
    ///       would be created (e.g., as used in Operation::create) and the regions of
    ///       the op. The method takes an optional location which, if set, will be used
    ///       to report errors on.
    /// 
    ///       The return types may be elided or specific elements be null for elements
    ///       that should just be returned but not verified.
    /// 
    ///       Because this method can be called from within different stages of IR
    ///       verification, implementations should not assume the arguments to
    ///       represent fully valid IR and are responsible for checking inputs for
    ///       validity to the degree necessary to perform the return type inference.
    static ::llvm::LogicalResult refineReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& returnTypes) {
      llvm::SmallVector<Type, 4> inferredReturnTypes;
          if (failed(ConcreteOp::inferReturnTypes(context, location, operands,
                                                  attributes, properties, regions,
                                                  inferredReturnTypes)))
            return failure();
          if (!ConcreteOp::isCompatibleReturnTypes(inferredReturnTypes,
                                                   returnTypes)) {
            return emitOptionalError(
                location, "'", ConcreteOp::getOperationName(),
                "' op inferred type(s) ", inferredReturnTypes,
                " are incompatible with return type(s) of operation ",
                returnTypes);
          }
          return success();
    }
    /// Returns whether two array of types are compatible result types for an op.
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
      /// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
    }
    static ::llvm::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyInferredResultTypes(op);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class InferShapedTypeOpInterface;
namespace detail {
struct InferShapedTypeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::LogicalResult (*inferReturnTypeComponents)(::mlir::MLIRContext*, ::std::optional<::mlir::Location>, ::mlir::ValueShapeRange, ::mlir::DictionaryAttr, ::mlir::OpaqueProperties, ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&);
    ::llvm::LogicalResult (*reifyReturnTypeShapes)(const Concept *impl, ::mlir::Operation *, ::mlir::OpBuilder&, ::mlir::ValueRange, ::llvm::SmallVectorImpl<::mlir::Value> &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::InferShapedTypeOpInterface;
    Model() : Concept{inferReturnTypeComponents, reifyReturnTypeShapes} {}

    static inline ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes);
    static inline ::llvm::LogicalResult reifyReturnTypeShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::InferShapedTypeOpInterface;
    FallbackModel() : Concept{inferReturnTypeComponents, reifyReturnTypeShapes} {}

    static inline ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes);
    static inline ::llvm::LogicalResult reifyReturnTypeShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
    ::llvm::LogicalResult reifyReturnTypeShapes(::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder&builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) const;
  };
};
template <typename ConcreteOp>
struct InferShapedTypeOpInterfaceTrait;

} // namespace detail
class InferShapedTypeOpInterface : public ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::InferShapedTypeOpInterfaceTrait<ConcreteOp> {};
  /// Infer the components of return type of shape containter.
  /// 
  ///       The method takes an optional location which, if set, will be used to
  ///       report errors on. The operands and attributes correspond to those with
  ///       which an Operation would be created (e.g., as used in Operation::create)
  ///       and the regions of the op.
  /// 
  ///       Unknown (e.g., unranked) shape and nullptrs for element type and attribute
  ///       may be returned by this function while returning success. E.g., partial
  ///       population of components is not error condition.
  /// 
  ///       Because this method can be called from within different stages of IR
  ///       verification, implementations should not assume the arguments to
  ///       represent fully valid IR and are responsible for checking inputs for
  ///       validity to the degree necessary to perform the return type inference.
  ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes);
  /// Reify the shape computation for the operation.
  /// 
  ///       Insert operations using the given OpBuilder that computes the
  ///       result shape. This interface is supposed to be workable during dialect
  ///       conversion (e.g. convert from tensor world to buffer world),
  ///       where `getOperand` may be invalid. For example, some ops (e.g.
  ///       dynamic_reshape(input, target_shape)) may depend on their operands
  ///       to calculate the result shape. When the `matchAndRewrite ` method
  ///       of a conversion pattern is called, the operands of the op to convert
  ///       may have been converted into other types, which makes it invalid to
  ///       call the `getOperand` method of such op directly inside the
  ///       conversion pattern.  To solve this problem, this interface follows
  ///       the design of the conversion pattern, that is, accepting passed in
  ///       operands to avoid calling `getOperand` directly inside the interface
  ///       implementation.
  ::llvm::LogicalResult reifyReturnTypeShapes(::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes);
};
namespace detail {
  template <typename ConcreteOp>
  struct InferShapedTypeOpInterfaceTrait : public ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Infer the components of return type of shape containter.
    /// 
    ///       The method takes an optional location which, if set, will be used to
    ///       report errors on. The operands and attributes correspond to those with
    ///       which an Operation would be created (e.g., as used in Operation::create)
    ///       and the regions of the op.
    /// 
    ///       Unknown (e.g., unranked) shape and nullptrs for element type and attribute
    ///       may be returned by this function while returning success. E.g., partial
    ///       population of components is not error condition.
    /// 
    ///       Because this method can be called from within different stages of IR
    ///       verification, implementations should not assume the arguments to
    ///       represent fully valid IR and are responsible for checking inputs for
    ///       validity to the degree necessary to perform the return type inference.
    static ::llvm::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
      return ::mlir::failure();
    }
    /// Reify the shape computation for the operation.
    /// 
    ///       Insert operations using the given OpBuilder that computes the
    ///       result shape. This interface is supposed to be workable during dialect
    ///       conversion (e.g. convert from tensor world to buffer world),
    ///       where `getOperand` may be invalid. For example, some ops (e.g.
    ///       dynamic_reshape(input, target_shape)) may depend on their operands
    ///       to calculate the result shape. When the `matchAndRewrite ` method
    ///       of a conversion pattern is called, the operands of the op to convert
    ///       may have been converted into other types, which makes it invalid to
    ///       call the `getOperand` method of such op directly inside the
    ///       conversion pattern.  To solve this problem, this interface follows
    ///       the design of the conversion pattern, that is, accepting passed in
    ///       operands to avoid calling `getOperand` directly inside the interface
    ///       implementation.
    ::llvm::LogicalResult reifyReturnTypeShapes(::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes) {
      return ::mlir::failure();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class ReifyRankedShapedTypeOpInterface;
namespace detail {
struct ReifyRankedShapedTypeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::LogicalResult (*reifyResultShapes)(const Concept *impl, ::mlir::Operation *, ::mlir::OpBuilder &, ::mlir::ReifiedRankedShapedTypeDims &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ReifyRankedShapedTypeOpInterface;
    Model() : Concept{reifyResultShapes} {}

    static inline ::llvm::LogicalResult reifyResultShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ReifyRankedShapedTypeOpInterface;
    FallbackModel() : Concept{reifyResultShapes} {}

    static inline ::llvm::LogicalResult reifyResultShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct ReifyRankedShapedTypeOpInterfaceTrait;

} // namespace detail
class ReifyRankedShapedTypeOpInterface : public ::mlir::OpInterface<ReifyRankedShapedTypeOpInterface, detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ReifyRankedShapedTypeOpInterface, detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ReifyRankedShapedTypeOpInterfaceTrait<ConcreteOp> {};
  /// Reify the shape of the result of an operation (typically in terms of the
  /// shape of its operands).
  /// 
  /// `reifiedReturnShapes` is populated with one vector per op result. Each
  /// of those vectors contains an OpFoldResult for each dimension of the
  /// shaped type. In case a dimension in the type is static, the
  /// corresponding entry is an IntegerAttr. Otherwise, it is a Value. The
  /// given builder may be used to insert ops that compute result shapes.
  /// 
  /// If the shape of a particular result cannot be computed it must be empty.
  ::llvm::LogicalResult reifyResultShapes(::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes);
};
namespace detail {
  template <typename ConcreteOp>
  struct ReifyRankedShapedTypeOpInterfaceTrait : public ::mlir::OpInterface<ReifyRankedShapedTypeOpInterface, detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::llvm::LogicalResult detail::InferTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  return ConcreteOp::inferReturnTypes(context, location, operands, attributes, properties, regions, inferredReturnTypes);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::InferTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::refineReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& returnTypes) {
  return ConcreteOp::refineReturnTypes(context, location, operands, attributes, properties, regions, returnTypes);
}
template<typename ConcreteOp>
bool detail::InferTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
  return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::InferTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  return ConcreteOp::inferReturnTypes(context, location, operands, attributes, properties, regions, inferredReturnTypes);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::InferTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::refineReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& returnTypes) {
  return ConcreteOp::refineReturnTypes(context, location, operands, attributes, properties, regions, returnTypes);
}
template<typename ConcreteOp>
bool detail::InferTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
  return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::InferTypeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::refineReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&returnTypes) {
llvm::SmallVector<Type, 4> inferredReturnTypes;
          if (failed(ConcreteOp::inferReturnTypes(context, location, operands,
                                                  attributes, properties, regions,
                                                  inferredReturnTypes)))
            return failure();
          if (!ConcreteOp::isCompatibleReturnTypes(inferredReturnTypes,
                                                   returnTypes)) {
            return emitOptionalError(
                location, "'", ConcreteOp::getOperationName(),
                "' op inferred type(s) ", inferredReturnTypes,
                " are incompatible with return type(s) of operation ",
                returnTypes);
          }
          return success();
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::InferTypeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
/// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::llvm::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferReturnTypeComponents(::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
  return ConcreteOp::inferReturnTypeComponents(context, location, operands, attributes, properties, regions, inferredReturnShapes);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::reifyReturnTypeShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).reifyReturnTypeShapes(builder, operands, reifiedReturnShapes);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferReturnTypeComponents(::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
  return ConcreteOp::inferReturnTypeComponents(context, location, operands, attributes, properties, regions, inferredReturnShapes);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::reifyReturnTypeShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes) {
  return static_cast<const ConcreteOp *>(impl)->reifyReturnTypeShapes(tablegen_opaque_val, builder, operands, reifiedReturnShapes);
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::inferReturnTypeComponents(::mlir::MLIRContext*context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes) {
return ::mlir::failure();
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::reifyReturnTypeShapes(::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder&builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) const {
return ::mlir::failure();
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::llvm::LogicalResult detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::reifyResultShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).reifyResultShapes(builder, reifiedReturnShapes);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::reifyResultShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes) {
  return static_cast<const ConcreteOp *>(impl)->reifyResultShapes(tablegen_opaque_val, builder, reifiedReturnShapes);
}
} // namespace mlir
