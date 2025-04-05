/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class InferIntRangeInterface;
namespace detail {
struct InferIntRangeInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*inferResultRanges)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::ConstantIntRanges>, ::mlir::SetIntRangeFn);
    void (*inferResultRangesFromOptional)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::IntegerValueRange>, ::mlir::SetIntLatticeFn);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::InferIntRangeInterface;
    Model() : Concept{inferResultRanges, inferResultRangesFromOptional} {}

    static inline void inferResultRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges);
    static inline void inferResultRangesFromOptional(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::InferIntRangeInterface;
    FallbackModel() : Concept{inferResultRanges, inferResultRangesFromOptional} {}

    static inline void inferResultRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges);
    static inline void inferResultRangesFromOptional(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    void inferResultRanges(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) const;
    void inferResultRangesFromOptional(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges) const;
  };
};
template <typename ConcreteOp>
struct InferIntRangeInterfaceTrait;

} // namespace detail
class InferIntRangeInterface : public ::mlir::OpInterface<InferIntRangeInterface, detail::InferIntRangeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferIntRangeInterface, detail::InferIntRangeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::InferIntRangeInterfaceTrait<ConcreteOp> {};
  ///  Infer the bounds on the results of this op given the bounds on its arguments.
  ///  For each result value or block argument (that isn't a branch argument,
  ///  since the dataflow analysis handles those case), the method should call
  ///  `setValueRange` with that `Value` as an argument. When implemented,
  ///  `setValueRange` should be called on all result values for the operation.
  ///  When operations take non-integer inputs, the
  /// `inferResultRangesFromOptional` method should be implemented instead.
  /// 
  ///  When called on an op that also implements the RegionBranchOpInterface
  ///  or BranchOpInterface, this method should not attempt to infer the values
  ///  of the branch results, as this will be handled by the analyses that use
  ///  this interface.
  /// 
  ///  This function will only be called when at least one result of the op is a
  ///  scalar integer value or the op has a region.
  void inferResultRanges(::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges);
  /// Infer the bounds on the results of this op given the lattice representation
  /// of the bounds for its arguments. For each result value or block argument
  /// (that isn't a branch argument, since the dataflow analysis handles
  /// those case), the method should call `setValueRange` with that `Value`
  /// as an argument. When implemented, `setValueRange` should be called on
  /// all result values for the operation.
  /// 
  /// This method allows for more precise implementations when operations
  /// want to reason about inputs which may be undefined during the analysis.
  void inferResultRangesFromOptional(::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges);
};
namespace detail {
  template <typename ConcreteOp>
  struct InferIntRangeInterfaceTrait : public ::mlir::OpInterface<InferIntRangeInterface, detail::InferIntRangeInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    ///  Infer the bounds on the results of this op given the bounds on its arguments.
    ///  For each result value or block argument (that isn't a branch argument,
    ///  since the dataflow analysis handles those case), the method should call
    ///  `setValueRange` with that `Value` as an argument. When implemented,
    ///  `setValueRange` should be called on all result values for the operation.
    ///  When operations take non-integer inputs, the
    /// `inferResultRangesFromOptional` method should be implemented instead.
    /// 
    ///  When called on an op that also implements the RegionBranchOpInterface
    ///  or BranchOpInterface, this method should not attempt to infer the values
    ///  of the branch results, as this will be handled by the analyses that use
    ///  this interface.
    /// 
    ///  This function will only be called when at least one result of the op is a
    ///  scalar integer value or the op has a region.
    void inferResultRanges(::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) {
      ::mlir::intrange::detail::defaultInferResultRangesFromOptional((*static_cast<ConcreteOp *>(this)),
                                                                     argRanges,
                                                                     setResultRanges);
    }
    /// Infer the bounds on the results of this op given the lattice representation
    /// of the bounds for its arguments. For each result value or block argument
    /// (that isn't a branch argument, since the dataflow analysis handles
    /// those case), the method should call `setValueRange` with that `Value`
    /// as an argument. When implemented, `setValueRange` should be called on
    /// all result values for the operation.
    /// 
    /// This method allows for more precise implementations when operations
    /// want to reason about inputs which may be undefined during the analysis.
    void inferResultRangesFromOptional(::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges) {
      ::mlir::intrange::detail::defaultInferResultRanges((*static_cast<ConcreteOp *>(this)),
                                                         argRanges,
                                                         setResultRanges);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
void detail::InferIntRangeInterfaceInterfaceTraits::Model<ConcreteOp>::inferResultRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferResultRanges(argRanges, setResultRanges);
}
template<typename ConcreteOp>
void detail::InferIntRangeInterfaceInterfaceTraits::Model<ConcreteOp>::inferResultRangesFromOptional(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferResultRangesFromOptional(argRanges, setResultRanges);
}
template<typename ConcreteOp>
void detail::InferIntRangeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferResultRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) {
  return static_cast<const ConcreteOp *>(impl)->inferResultRanges(tablegen_opaque_val, argRanges, setResultRanges);
}
template<typename ConcreteOp>
void detail::InferIntRangeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferResultRangesFromOptional(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges) {
  return static_cast<const ConcreteOp *>(impl)->inferResultRangesFromOptional(tablegen_opaque_val, argRanges, setResultRanges);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::InferIntRangeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::inferResultRanges(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) const {
::mlir::intrange::detail::defaultInferResultRangesFromOptional((llvm::cast<ConcreteOp>(tablegen_opaque_val)),
                                                                     argRanges,
                                                                     setResultRanges);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::InferIntRangeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::inferResultRangesFromOptional(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::IntegerValueRange> argRanges, ::mlir::SetIntLatticeFn setResultRanges) const {
::mlir::intrange::detail::defaultInferResultRanges((llvm::cast<ConcreteOp>(tablegen_opaque_val)),
                                                         argRanges,
                                                         setResultRanges);
}
} // namespace mlir
