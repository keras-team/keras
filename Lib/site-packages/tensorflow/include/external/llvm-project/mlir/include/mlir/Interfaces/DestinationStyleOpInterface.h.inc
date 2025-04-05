/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DestinationStyleOpInterface;
namespace detail {
struct DestinationStyleOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::MutableOperandRange (*getDpsInitsMutable)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DestinationStyleOpInterface;
    Model() : Concept{getDpsInitsMutable} {}

    static inline ::mlir::MutableOperandRange getDpsInitsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DestinationStyleOpInterface;
    FallbackModel() : Concept{getDpsInitsMutable} {}

    static inline ::mlir::MutableOperandRange getDpsInitsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DestinationStyleOpInterfaceTrait;

} // namespace detail
class DestinationStyleOpInterface : public ::mlir::OpInterface<DestinationStyleOpInterface, detail::DestinationStyleOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DestinationStyleOpInterface, detail::DestinationStyleOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DestinationStyleOpInterfaceTrait<ConcreteOp> {};
  /// Return start and end indices of the init operands range.
  ::mlir::MutableOperandRange getDpsInitsMutable();

    ::mlir::OperandRange getDpsInits() {
      return (*this).getDpsInitsMutable();
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInits() { return (*this).getDpsInits().size(); }

    /// Return the `i`-th DPS init.
    ::mlir::OpOperand *getDpsInitOperand(int64_t i) {
      return &(*this).getDpsInitsMutable()[i];
    }

    /// Set the `i`-th DPS init.
    void setDpsInitOperand(int64_t i, Value value) {
      assert(i >= 0 && i < (*this).getNumDpsInits() && "invalid index");
      (*this)->setOperand((*this).getDpsInits().getBeginOperandIndex() + i, value);
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInputs() {
      return (*this)->getNumOperands() - (*this).getNumDpsInits();
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::OpOperand *> getDpsInputOperands() {
      ::llvm::SmallVector<::mlir::OpOperand *> result;
      int64_t numOperands = (*this)->getNumOperands();
      ::mlir::OperandRange range = (*this).getDpsInits();
      if (range.empty()) {
        result.reserve(numOperands);
        for (int64_t i = 0; i < numOperands; ++i)
          result.push_back(&(*this)->getOpOperand(i));
        return result;
      }
      int64_t firstInitPos = range.getBeginOperandIndex();
      int64_t numInits = range.size();
      result.reserve(numOperands - numInits);
      for (int64_t i = 0; i < firstInitPos; ++i)
        result.push_back(&(*this)->getOpOperand(i));
      for (int64_t i = firstInitPos + numInits; i < numOperands; ++i)
        result.push_back(&(*this)->getOpOperand(i));
      return result;
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::Value> getDpsInputs() {
      return ::llvm::to_vector(::llvm::map_range(
          (*this).getDpsInputOperands(), [](OpOperand *o) { return o->get(); }));
    }

    /// Return the `i`-th DPS input operand.
    ::mlir::OpOperand *getDpsInputOperand(int64_t i) {
      ::mlir::OperandRange range = (*this).getDpsInits();
      if (range.empty())
        return &(*this)->getOpOperand(i);
      int64_t firstInitPos = range.getBeginOperandIndex();
      int64_t numInits = range.size();
      assert(i >= 0 && i < (*this)->getNumOperands() - numInits
             && "invalid index");
      return &(*this)->getOpOperand(
          i < firstInitPos ? i : i + firstInitPos + numInits);
    }

    /// Return "true" if `opOperand` is an "input".
    bool isDpsInput(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*this) && "invalid operand");
      return !(*this).isDpsInit(opOperand);
    }

    /// Return "true" if `opOperand` is an "init".
    bool isDpsInit(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*this) && "invalid operand");
      ::mlir::OperandRange range = (*this).getDpsInits();
      if (range.empty())
        return false;
      auto operandNumber = opOperand->getOperandNumber();
      return operandNumber >= range.getBeginOperandIndex()
          && operandNumber < range.getBeginOperandIndex() + range.size();
    }

    /// Return "true" if `opOperand` is a scalar value. A sclar is defined as
    /// neither a MemRef nor a tensor value.
    bool isScalar(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*this) && "invalid operand");
      return !::llvm::isa<BaseMemRefType, TensorType>(
          opOperand->get().getType());
    }

    /// Return the OpResult that is tied to the given OpOperand.
    ::mlir::OpResult getTiedOpResult(::mlir::OpOperand *opOperand) {
        assert(opOperand->getOwner() == (*this) && "invalid operand");
        ::mlir::OperandRange range = (*this).getDpsInits();
        assert(!range.empty() && "op has no inits");
        int64_t resultIndex =
            opOperand->getOperandNumber() - range.getBeginOperandIndex();
        assert(resultIndex >= 0 &&
               resultIndex < (*this)->getNumResults());
        return (*this)->getResult(resultIndex);
    }

    /// Return the OpOperand that is tied to the given OpResult.
    ::mlir::OpOperand *getTiedOpOperand(::mlir::OpResult opResult) {
      assert(opResult.getDefiningOp() == (*this) && "invalid opresult");
      return (*this).getDpsInitOperand(opResult.getResultNumber());
    }

    /// Return whether the op has pure buffer semantics. That is the case if the
    /// op has no tensor operands and at least one memref operand.
    bool hasPureBufferSemantics() {
      // No tensors.
      auto isTensor = [](Value v){
        return ::llvm::isa<::mlir::TensorType>(v.getType());
      };
      if (::llvm::any_of((*this)->getOperands(), isTensor))
        return false;
      // At least one memref.
      auto isMemref = [](Value v){
        return ::llvm::isa<::mlir::BaseMemRefType>(v.getType());
      };
      return llvm::any_of((*this)->getOperands(), isMemref);
    }

    /// Return whether the op has pure tensor semantics. That is the case if the
    /// op has no memref operands and at least one tensor operand.
    bool hasPureTensorSemantics() {
      // No memrefs.
      auto isMemref = [](Value v){
        return ::llvm::isa<::mlir::BaseMemRefType>(v.getType());
      };
      if (::llvm::any_of((*this)->getOperands(), isMemref))
        return false;
      // At least one tensor.
      auto isTensor = [](Value v){
        return ::llvm::isa<::mlir::TensorType>(v.getType());
      };
      return llvm::any_of((*this)->getOperands(), isTensor);    }
};
namespace detail {
  template <typename ConcreteOp>
  struct DestinationStyleOpInterfaceTrait : public ::mlir::OpInterface<DestinationStyleOpInterface, detail::DestinationStyleOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyDestinationStyleOpInterface(op);
    }

    ::mlir::OperandRange getDpsInits() {
      return (*static_cast<ConcreteOp *>(this)).getDpsInitsMutable();
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInits() { return (*static_cast<ConcreteOp *>(this)).getDpsInits().size(); }

    /// Return the `i`-th DPS init.
    ::mlir::OpOperand *getDpsInitOperand(int64_t i) {
      return &(*static_cast<ConcreteOp *>(this)).getDpsInitsMutable()[i];
    }

    /// Set the `i`-th DPS init.
    void setDpsInitOperand(int64_t i, Value value) {
      assert(i >= 0 && i < (*static_cast<ConcreteOp *>(this)).getNumDpsInits() && "invalid index");
      (*static_cast<ConcreteOp *>(this))->setOperand((*static_cast<ConcreteOp *>(this)).getDpsInits().getBeginOperandIndex() + i, value);
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInputs() {
      return (*static_cast<ConcreteOp *>(this))->getNumOperands() - (*static_cast<ConcreteOp *>(this)).getNumDpsInits();
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::OpOperand *> getDpsInputOperands() {
      ::llvm::SmallVector<::mlir::OpOperand *> result;
      int64_t numOperands = (*static_cast<ConcreteOp *>(this))->getNumOperands();
      ::mlir::OperandRange range = (*static_cast<ConcreteOp *>(this)).getDpsInits();
      if (range.empty()) {
        result.reserve(numOperands);
        for (int64_t i = 0; i < numOperands; ++i)
          result.push_back(&(*static_cast<ConcreteOp *>(this))->getOpOperand(i));
        return result;
      }
      int64_t firstInitPos = range.getBeginOperandIndex();
      int64_t numInits = range.size();
      result.reserve(numOperands - numInits);
      for (int64_t i = 0; i < firstInitPos; ++i)
        result.push_back(&(*static_cast<ConcreteOp *>(this))->getOpOperand(i));
      for (int64_t i = firstInitPos + numInits; i < numOperands; ++i)
        result.push_back(&(*static_cast<ConcreteOp *>(this))->getOpOperand(i));
      return result;
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::Value> getDpsInputs() {
      return ::llvm::to_vector(::llvm::map_range(
          (*static_cast<ConcreteOp *>(this)).getDpsInputOperands(), [](OpOperand *o) { return o->get(); }));
    }

    /// Return the `i`-th DPS input operand.
    ::mlir::OpOperand *getDpsInputOperand(int64_t i) {
      ::mlir::OperandRange range = (*static_cast<ConcreteOp *>(this)).getDpsInits();
      if (range.empty())
        return &(*static_cast<ConcreteOp *>(this))->getOpOperand(i);
      int64_t firstInitPos = range.getBeginOperandIndex();
      int64_t numInits = range.size();
      assert(i >= 0 && i < (*static_cast<ConcreteOp *>(this))->getNumOperands() - numInits
             && "invalid index");
      return &(*static_cast<ConcreteOp *>(this))->getOpOperand(
          i < firstInitPos ? i : i + firstInitPos + numInits);
    }

    /// Return "true" if `opOperand` is an "input".
    bool isDpsInput(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*static_cast<ConcreteOp *>(this)) && "invalid operand");
      return !(*static_cast<ConcreteOp *>(this)).isDpsInit(opOperand);
    }

    /// Return "true" if `opOperand` is an "init".
    bool isDpsInit(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*static_cast<ConcreteOp *>(this)) && "invalid operand");
      ::mlir::OperandRange range = (*static_cast<ConcreteOp *>(this)).getDpsInits();
      if (range.empty())
        return false;
      auto operandNumber = opOperand->getOperandNumber();
      return operandNumber >= range.getBeginOperandIndex()
          && operandNumber < range.getBeginOperandIndex() + range.size();
    }

    /// Return "true" if `opOperand` is a scalar value. A sclar is defined as
    /// neither a MemRef nor a tensor value.
    bool isScalar(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*static_cast<ConcreteOp *>(this)) && "invalid operand");
      return !::llvm::isa<BaseMemRefType, TensorType>(
          opOperand->get().getType());
    }

    /// Return the OpResult that is tied to the given OpOperand.
    ::mlir::OpResult getTiedOpResult(::mlir::OpOperand *opOperand) {
        assert(opOperand->getOwner() == (*static_cast<ConcreteOp *>(this)) && "invalid operand");
        ::mlir::OperandRange range = (*static_cast<ConcreteOp *>(this)).getDpsInits();
        assert(!range.empty() && "op has no inits");
        int64_t resultIndex =
            opOperand->getOperandNumber() - range.getBeginOperandIndex();
        assert(resultIndex >= 0 &&
               resultIndex < (*static_cast<ConcreteOp *>(this))->getNumResults());
        return (*static_cast<ConcreteOp *>(this))->getResult(resultIndex);
    }

    /// Return the OpOperand that is tied to the given OpResult.
    ::mlir::OpOperand *getTiedOpOperand(::mlir::OpResult opResult) {
      assert(opResult.getDefiningOp() == (*static_cast<ConcreteOp *>(this)) && "invalid opresult");
      return (*static_cast<ConcreteOp *>(this)).getDpsInitOperand(opResult.getResultNumber());
    }

    /// Return whether the op has pure buffer semantics. That is the case if the
    /// op has no tensor operands and at least one memref operand.
    bool hasPureBufferSemantics() {
      // No tensors.
      auto isTensor = [](Value v){
        return ::llvm::isa<::mlir::TensorType>(v.getType());
      };
      if (::llvm::any_of((*static_cast<ConcreteOp *>(this))->getOperands(), isTensor))
        return false;
      // At least one memref.
      auto isMemref = [](Value v){
        return ::llvm::isa<::mlir::BaseMemRefType>(v.getType());
      };
      return llvm::any_of((*static_cast<ConcreteOp *>(this))->getOperands(), isMemref);
    }

    /// Return whether the op has pure tensor semantics. That is the case if the
    /// op has no memref operands and at least one tensor operand.
    bool hasPureTensorSemantics() {
      // No memrefs.
      auto isMemref = [](Value v){
        return ::llvm::isa<::mlir::BaseMemRefType>(v.getType());
      };
      if (::llvm::any_of((*static_cast<ConcreteOp *>(this))->getOperands(), isMemref))
        return false;
      // At least one tensor.
      auto isTensor = [](Value v){
        return ::llvm::isa<::mlir::TensorType>(v.getType());
      };
      return llvm::any_of((*static_cast<ConcreteOp *>(this))->getOperands(), isTensor);    }
  
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDpsInitsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsMutable();
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDpsInitsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDpsInitsMutable(tablegen_opaque_val);
}
} // namespace mlir
