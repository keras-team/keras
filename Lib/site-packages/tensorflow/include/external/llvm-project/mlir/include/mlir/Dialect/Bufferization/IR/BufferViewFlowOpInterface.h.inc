/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {
class BufferViewFlowOpInterface;
namespace detail {
struct BufferViewFlowOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*populateDependencies)(const Concept *impl, ::mlir::Operation *, ::mlir::bufferization::RegisterDependenciesFn);
    bool (*mayBeTerminalBuffer)(const Concept *impl, ::mlir::Operation *, Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::bufferization::BufferViewFlowOpInterface;
    Model() : Concept{populateDependencies, mayBeTerminalBuffer} {}

    static inline void populateDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::bufferization::RegisterDependenciesFn registerDependenciesFn);
    static inline bool mayBeTerminalBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value value);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::bufferization::BufferViewFlowOpInterface;
    FallbackModel() : Concept{populateDependencies, mayBeTerminalBuffer} {}

    static inline void populateDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::bufferization::RegisterDependenciesFn registerDependenciesFn);
    static inline bool mayBeTerminalBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value value);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool mayBeTerminalBuffer(::mlir::Operation *tablegen_opaque_val, Value value) const;
  };
};
template <typename ConcreteOp>
struct BufferViewFlowOpInterfaceTrait;

} // namespace detail
class BufferViewFlowOpInterface : public ::mlir::OpInterface<BufferViewFlowOpInterface, detail::BufferViewFlowOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BufferViewFlowOpInterface, detail::BufferViewFlowOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BufferViewFlowOpInterfaceTrait<ConcreteOp> {};
  /// Populate buffer dependencies between operands and op results/region
  /// entry block arguments.
  /// 
  /// Implementations should register dependencies between an operand ("X")
  /// and an op result/region entry block argument ("Y") if Y may depend
  /// on X. Y depends on X if Y and X are the same buffer or if Y is a
  /// subview of X.
  /// 
  /// Example:
  /// ```
  /// %r = arith.select %c, %m1, %m2 : memref<5xf32>
  /// ```
  /// In the above example, %0 may depend on %m1 or %m2 and a correct
  /// interface implementation should call:
  /// - "registerDependenciesFn(%m1, %r)".
  /// - "registerDependenciesFn(%m2, %r)"
  void populateDependencies(::mlir::bufferization::RegisterDependenciesFn registerDependenciesFn);
  /// Return "true" if the given value may be a terminal buffer. A buffer
  /// value is "terminal" if it cannot be traced back any further in the
  /// buffer view flow analysis.
  /// 
  /// Examples: A buffer could be terminal because:
  /// - it is a newly allocated buffer (e.g., "memref.alloc"),
  /// - or: because there is not enough compile-time information available
  ///   to make a definite decision (e.g., "memref.realloc" may reallocate
  ///   but we do not know for sure; another example are call ops where we
  ///   would have to analyze the body of the callee).
  /// 
  /// Implementations can assume that the given SSA value is an OpResult of
  /// this operation or a region entry block argument of this operation.
  bool mayBeTerminalBuffer(Value value);
};
namespace detail {
  template <typename ConcreteOp>
  struct BufferViewFlowOpInterfaceTrait : public ::mlir::OpInterface<BufferViewFlowOpInterface, detail::BufferViewFlowOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return "true" if the given value may be a terminal buffer. A buffer
    /// value is "terminal" if it cannot be traced back any further in the
    /// buffer view flow analysis.
    /// 
    /// Examples: A buffer could be terminal because:
    /// - it is a newly allocated buffer (e.g., "memref.alloc"),
    /// - or: because there is not enough compile-time information available
    ///   to make a definite decision (e.g., "memref.realloc" may reallocate
    ///   but we do not know for sure; another example are call ops where we
    ///   would have to analyze the body of the callee).
    /// 
    /// Implementations can assume that the given SSA value is an OpResult of
    /// this operation or a region entry block argument of this operation.
    bool mayBeTerminalBuffer(Value value) {
      return false;
    }
  };
}// namespace detail
} // namespace bufferization
} // namespace mlir
namespace mlir {
namespace bufferization {
template<typename ConcreteOp>
void detail::BufferViewFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::populateDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::bufferization::RegisterDependenciesFn registerDependenciesFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).populateDependencies(registerDependenciesFn);
}
template<typename ConcreteOp>
bool detail::BufferViewFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::mayBeTerminalBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value value) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).mayBeTerminalBuffer(value);
}
template<typename ConcreteOp>
void detail::BufferViewFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::populateDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::bufferization::RegisterDependenciesFn registerDependenciesFn) {
  return static_cast<const ConcreteOp *>(impl)->populateDependencies(tablegen_opaque_val, registerDependenciesFn);
}
template<typename ConcreteOp>
bool detail::BufferViewFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::mayBeTerminalBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value value) {
  return static_cast<const ConcreteOp *>(impl)->mayBeTerminalBuffer(tablegen_opaque_val, value);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::BufferViewFlowOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::mayBeTerminalBuffer(::mlir::Operation *tablegen_opaque_val, Value value) const {
return false;
}
} // namespace bufferization
} // namespace mlir
