/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
class MatchOpInterface;
namespace detail {
struct MatchOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    /// The base classes of this interface.
    const ::mlir::transform::TransformOpInterface::Concept *implTransformOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implTransformOpInterface = interfaceMap.lookup<::mlir::transform::TransformOpInterface>();
      assert(implTransformOpInterface && "`::mlir::transform::MatchOpInterface` expected its base interface `::mlir::transform::TransformOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::MatchOpInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::MatchOpInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct MatchOpInterfaceTrait;

} // namespace detail
class MatchOpInterface : public ::mlir::OpInterface<MatchOpInterface, detail::MatchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MatchOpInterface, detail::MatchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MatchOpInterfaceTrait<ConcreteOp> {};
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::transform::TransformOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::transform::TransformOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::transform::TransformOpInterface(*this, getImpl()->implTransformOpInterface);
  }

  /// Applies the transformation represented by the current operation. This
  /// accepts as arguments the object that must be populated with results of
  /// the current transformation and a transformation state object that can be
  /// used for queries, e.g., to obtain the list of operations on which the
  /// transformation represented by the current op is targeted. Returns a
  /// special status object indicating whether the transformation succeeded
  /// or failed, and, if it failed, whether the failure is recoverable or not.
  /// 
  /// IR must be created, modified and deleted with the provided rewriter.
  /// implementations are responsible for setting the insertion point of the
  /// rewriter to the desired location.
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformRewriter & rewriter, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
  /// Indicates whether the op instance allows its handle operands to be
  /// associated with the same payload operations.
  bool allowsRepeatedHandleOperands();

    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation. Silenceable failure must be suppressed or reported
    /// explicitly at some later time.
    DiagnosedSilenceableFailure
    emitSilenceableError(const ::llvm::Twine &message = {}) {
      return ::mlir::emitSilenceableFailure((*this), message);
    }

    /// Creates the definite failure object with a diagnostic located at the
    /// current operation. Definite failure will be reported when the object
    /// is destroyed or converted.
    DiagnosedDefiniteFailure
    emitDefiniteFailure(const ::llvm::Twine &message = {}) {
      return ::mlir::emitDefiniteFailure((*this), message);
    }

    /// Emits a generic definite failure for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    DiagnosedDefiniteFailure emitDefaultDefiniteFailure(
        ::mlir::Operation *target) {
      auto diag = ::mlir::emitDefiniteFailure((*this), "failed to apply");
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      DiagnosedSilenceableFailure diag = emitSilenceableFailure((*this)->getLoc());
      diag << (*this)->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return diag;
    }

    /// Returns all operands that are handles and being consumed by this op.
    ::llvm::SmallVector<OpOperand *> getConsumedHandleOpOperands() {
      return ::mlir::transform::detail::getConsumedHandleOpOperands((*this));
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct MatchOpInterfaceTrait : public ::mlir::OpInterface<MatchOpInterface, detail::MatchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
} // namespace transform
} // namespace mlir
