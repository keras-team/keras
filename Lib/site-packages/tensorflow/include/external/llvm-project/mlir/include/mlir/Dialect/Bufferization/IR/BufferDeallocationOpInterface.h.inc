/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {
class BufferDeallocationOpInterface;
namespace detail {
struct BufferDeallocationOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    FailureOr<Operation *> (*process)(const Concept *impl, ::mlir::Operation *, DeallocationState &, const DeallocationOptions &);
    std::pair<Value, Value> (*materializeUniqueOwnershipForMemref)(const Concept *impl, ::mlir::Operation *, DeallocationState &, const DeallocationOptions &, OpBuilder &, Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::bufferization::BufferDeallocationOpInterface;
    Model() : Concept{process, materializeUniqueOwnershipForMemref} {}

    static inline FailureOr<Operation *> process(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, DeallocationState & state, const DeallocationOptions & options);
    static inline std::pair<Value, Value> materializeUniqueOwnershipForMemref(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, DeallocationState & state, const DeallocationOptions & options, OpBuilder & builder, Value memref);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::bufferization::BufferDeallocationOpInterface;
    FallbackModel() : Concept{process, materializeUniqueOwnershipForMemref} {}

    static inline FailureOr<Operation *> process(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, DeallocationState & state, const DeallocationOptions & options);
    static inline std::pair<Value, Value> materializeUniqueOwnershipForMemref(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, DeallocationState & state, const DeallocationOptions & options, OpBuilder & builder, Value memref);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    std::pair<Value, Value> materializeUniqueOwnershipForMemref(::mlir::Operation *tablegen_opaque_val, DeallocationState &state, const DeallocationOptions &options, OpBuilder &builder, Value memref) const;
  };
};
template <typename ConcreteOp>
struct BufferDeallocationOpInterfaceTrait;

} // namespace detail
class BufferDeallocationOpInterface : public ::mlir::OpInterface<BufferDeallocationOpInterface, detail::BufferDeallocationOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BufferDeallocationOpInterface, detail::BufferDeallocationOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BufferDeallocationOpInterfaceTrait<ConcreteOp> {};
  /// This method takes the current deallocation state and transformation
  /// options and updates the deallocation state as necessary for the
  /// operation implementing this interface. It may also insert
  /// `bufferization.dealloc` operations and rebuild itself with different
  /// result types. For operations implementing this interface all other
  /// interface handlers (e.g., default handlers for interfaces like
  /// RegionBranchOpInterface, CallOpInterface, etc.) are skipped by the
  /// deallocation pass. On success, either the current operation or one of
  /// the newly inserted operations is returned from which on the driver
  /// should continue the processing. On failure, the deallocation pass
  /// will terminate. It is recommended to emit a useful error message in
  /// that case.
  FailureOr<Operation *> process(DeallocationState & state, const DeallocationOptions & options);
  /// This method allows the implementing operation to specify custom logic
  /// to materialize an ownership indicator value for the given MemRef typed
  /// value it defines (including block arguments of nested regions). Since
  /// the operation itself has more information about its semantics the
  /// materialized IR can be more efficient compared to the default
  /// implementation and avoid cloning MemRefs and/or doing alias checking
  /// at runtime.
  /// Note that the same logic could also be implemented in the 'process'
  /// method above, however, the IR is always materialized then. If
  /// it's desirable to only materialize the IR to compute an updated
  /// ownership indicator when needed, it should be implemented using this
  /// method (which is especially important if operations are created that
  /// cannot be easily canonicalized away anymore).
  std::pair<Value, Value> materializeUniqueOwnershipForMemref(DeallocationState & state, const DeallocationOptions & options, OpBuilder & builder, Value memref);
};
namespace detail {
  template <typename ConcreteOp>
  struct BufferDeallocationOpInterfaceTrait : public ::mlir::OpInterface<BufferDeallocationOpInterface, detail::BufferDeallocationOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// This method allows the implementing operation to specify custom logic
    /// to materialize an ownership indicator value for the given MemRef typed
    /// value it defines (including block arguments of nested regions). Since
    /// the operation itself has more information about its semantics the
    /// materialized IR can be more efficient compared to the default
    /// implementation and avoid cloning MemRefs and/or doing alias checking
    /// at runtime.
    /// Note that the same logic could also be implemented in the 'process'
    /// method above, however, the IR is always materialized then. If
    /// it's desirable to only materialize the IR to compute an updated
    /// ownership indicator when needed, it should be implemented using this
    /// method (which is especially important if operations are created that
    /// cannot be easily canonicalized away anymore).
    std::pair<Value, Value> materializeUniqueOwnershipForMemref(DeallocationState & state, const DeallocationOptions & options, OpBuilder & builder, Value memref) {
      return state.getMemrefWithUniqueOwnership(
            builder, memref, memref.getParentBlock());
    }
  };
}// namespace detail
} // namespace bufferization
} // namespace mlir
namespace mlir {
namespace bufferization {
template<typename ConcreteOp>
FailureOr<Operation *> detail::BufferDeallocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::process(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, DeallocationState & state, const DeallocationOptions & options) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).process(state, options);
}
template<typename ConcreteOp>
std::pair<Value, Value> detail::BufferDeallocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::materializeUniqueOwnershipForMemref(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, DeallocationState & state, const DeallocationOptions & options, OpBuilder & builder, Value memref) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).materializeUniqueOwnershipForMemref(state, options, builder, memref);
}
template<typename ConcreteOp>
FailureOr<Operation *> detail::BufferDeallocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::process(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, DeallocationState & state, const DeallocationOptions & options) {
  return static_cast<const ConcreteOp *>(impl)->process(tablegen_opaque_val, state, options);
}
template<typename ConcreteOp>
std::pair<Value, Value> detail::BufferDeallocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::materializeUniqueOwnershipForMemref(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, DeallocationState & state, const DeallocationOptions & options, OpBuilder & builder, Value memref) {
  return static_cast<const ConcreteOp *>(impl)->materializeUniqueOwnershipForMemref(tablegen_opaque_val, state, options, builder, memref);
}
template<typename ConcreteModel, typename ConcreteOp>
std::pair<Value, Value> detail::BufferDeallocationOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::materializeUniqueOwnershipForMemref(::mlir::Operation *tablegen_opaque_val, DeallocationState &state, const DeallocationOptions &options, OpBuilder &builder, Value memref) const {
return state.getMemrefWithUniqueOwnership(
            builder, memref, memref.getParentBlock());
}
} // namespace bufferization
} // namespace mlir
