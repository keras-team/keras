/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class BytecodeOpInterface;
namespace detail {
struct BytecodeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    LogicalResult (*readProperties)(::mlir::DialectBytecodeReader &, ::mlir::OperationState &);
    void (*writeProperties)(const Concept *impl, ::mlir::Operation *, ::mlir::DialectBytecodeWriter &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::BytecodeOpInterface;
    Model() : Concept{readProperties, writeProperties} {}

    static inline LogicalResult readProperties(::mlir::DialectBytecodeReader & reader, ::mlir::OperationState & state);
    static inline void writeProperties(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::DialectBytecodeWriter & writer);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::BytecodeOpInterface;
    FallbackModel() : Concept{readProperties, writeProperties} {}

    static inline LogicalResult readProperties(::mlir::DialectBytecodeReader & reader, ::mlir::OperationState & state);
    static inline void writeProperties(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::DialectBytecodeWriter & writer);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct BytecodeOpInterfaceTrait;

} // namespace detail
class BytecodeOpInterface : public ::mlir::OpInterface<BytecodeOpInterface, detail::BytecodeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BytecodeOpInterface, detail::BytecodeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BytecodeOpInterfaceTrait<ConcreteOp> {};
  /// Read the properties for this operation from the bytecode and populate the state.
  LogicalResult readProperties(::mlir::DialectBytecodeReader & reader, ::mlir::OperationState & state);
  /// Write the properties for this operation to the bytecode.
  void writeProperties(::mlir::DialectBytecodeWriter & writer);
};
namespace detail {
  template <typename ConcreteOp>
  struct BytecodeOpInterfaceTrait : public ::mlir::OpInterface<BytecodeOpInterface, detail::BytecodeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
LogicalResult detail::BytecodeOpInterfaceInterfaceTraits::Model<ConcreteOp>::readProperties(::mlir::DialectBytecodeReader & reader, ::mlir::OperationState & state) {
  return ConcreteOp::readProperties(reader, state);
}
template<typename ConcreteOp>
void detail::BytecodeOpInterfaceInterfaceTraits::Model<ConcreteOp>::writeProperties(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::DialectBytecodeWriter & writer) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).writeProperties(writer);
}
template<typename ConcreteOp>
LogicalResult detail::BytecodeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::readProperties(::mlir::DialectBytecodeReader & reader, ::mlir::OperationState & state) {
  return ConcreteOp::readProperties(reader, state);
}
template<typename ConcreteOp>
void detail::BytecodeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::writeProperties(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::DialectBytecodeWriter & writer) {
  return static_cast<const ConcreteOp *>(impl)->writeProperties(tablegen_opaque_val, writer);
}
} // namespace mlir
