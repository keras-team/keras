/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace accomp {
class AtomicReadOpInterface;
namespace detail {
struct AtomicReadOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::LogicalResult (*verifyCommon)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getX)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getV)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::accomp::AtomicReadOpInterface;
    Model() : Concept{verifyCommon, getX, getV} {}

    static inline ::llvm::LogicalResult verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getV(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::accomp::AtomicReadOpInterface;
    FallbackModel() : Concept{verifyCommon, getX, getV} {}

    static inline ::llvm::LogicalResult verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getV(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::llvm::LogicalResult verifyCommon(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct AtomicReadOpInterfaceTrait;

} // namespace detail
class AtomicReadOpInterface : public ::mlir::OpInterface<AtomicReadOpInterface, detail::AtomicReadOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AtomicReadOpInterface, detail::AtomicReadOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AtomicReadOpInterfaceTrait<ConcreteOp> {};
  /// Common verifier for operation that implements atomic read interface.
  ::llvm::LogicalResult verifyCommon();
  /// Obtains `x` which is the address from where the value is atomically
  /// read.
  ::mlir::Value getX();
  /// Obtains `v` which is the address where the value is stored after
  /// reading.
  ::mlir::Value getV();
};
namespace detail {
  template <typename ConcreteOp>
  struct AtomicReadOpInterfaceTrait : public ::mlir::OpInterface<AtomicReadOpInterface, detail::AtomicReadOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Common verifier for operation that implements atomic read interface.
    ::llvm::LogicalResult verifyCommon() {
      if ((*static_cast<ConcreteOp *>(this)).getX() == (*static_cast<ConcreteOp *>(this)).getV()) {
          return (*static_cast<ConcreteOp *>(this)).emitError(
            "read and write must not be to the same location for atomic reads");
        }
        return mlir::success();
    }
  };
}// namespace detail
} // namespace accomp
} // namespace mlir
namespace mlir {
namespace accomp {
class AtomicWriteOpInterface;
namespace detail {
struct AtomicWriteOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::LogicalResult (*verifyCommon)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getX)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getExpr)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::accomp::AtomicWriteOpInterface;
    Model() : Concept{verifyCommon, getX, getExpr} {}

    static inline ::llvm::LogicalResult verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getExpr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::accomp::AtomicWriteOpInterface;
    FallbackModel() : Concept{verifyCommon, getX, getExpr} {}

    static inline ::llvm::LogicalResult verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getExpr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::llvm::LogicalResult verifyCommon(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct AtomicWriteOpInterfaceTrait;

} // namespace detail
class AtomicWriteOpInterface : public ::mlir::OpInterface<AtomicWriteOpInterface, detail::AtomicWriteOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AtomicWriteOpInterface, detail::AtomicWriteOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AtomicWriteOpInterfaceTrait<ConcreteOp> {};
  /// Common verifier for operation that implements atomic write interface.
  ::llvm::LogicalResult verifyCommon();
  /// Obtains `x` which is the address to which the value is atomically
  /// written to.
  ::mlir::Value getX();
  /// Obtains `expr` which corresponds to the expression whose value is
  /// written to `x`.
  ::mlir::Value getExpr();
};
namespace detail {
  template <typename ConcreteOp>
  struct AtomicWriteOpInterfaceTrait : public ::mlir::OpInterface<AtomicWriteOpInterface, detail::AtomicWriteOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Common verifier for operation that implements atomic write interface.
    ::llvm::LogicalResult verifyCommon() {
      mlir::Type elementType = (*static_cast<ConcreteOp *>(this)).getX().getType().getElementType();
        if (elementType && elementType != (*static_cast<ConcreteOp *>(this)).getExpr().getType())
          return (*static_cast<ConcreteOp *>(this)).emitError("address must dereference to value type");
        return mlir::success();
    }
  };
}// namespace detail
} // namespace accomp
} // namespace mlir
namespace mlir {
namespace accomp {
class AtomicUpdateOpInterface;
namespace detail {
struct AtomicUpdateOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getX)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Operation *(*getFirstOp)(const Concept *impl, ::mlir::Operation *);
    bool (*isNoOp)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getWriteOpVal)(const Concept *impl, ::mlir::Operation *);
    ::llvm::LogicalResult (*verifyCommon)(const Concept *impl, ::mlir::Operation *);
    ::llvm::LogicalResult (*verifyRegionsCommon)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::accomp::AtomicUpdateOpInterface;
    Model() : Concept{getX, getFirstOp, isNoOp, getWriteOpVal, verifyCommon, verifyRegionsCommon} {}

    static inline ::mlir::Value getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation *getFirstOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isNoOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getWriteOpVal(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyRegionsCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::accomp::AtomicUpdateOpInterface;
    FallbackModel() : Concept{getX, getFirstOp, isNoOp, getWriteOpVal, verifyCommon, verifyRegionsCommon} {}

    static inline ::mlir::Value getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation *getFirstOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isNoOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getWriteOpVal(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyRegionsCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Operation *getFirstOp(::mlir::Operation *tablegen_opaque_val) const;
    bool isNoOp(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Value getWriteOpVal(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::LogicalResult verifyCommon(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::LogicalResult verifyRegionsCommon(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct AtomicUpdateOpInterfaceTrait;

} // namespace detail
class AtomicUpdateOpInterface : public ::mlir::OpInterface<AtomicUpdateOpInterface, detail::AtomicUpdateOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AtomicUpdateOpInterface, detail::AtomicUpdateOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AtomicUpdateOpInterfaceTrait<ConcreteOp> {};
  /// Obtains `x` which is the address to which the value is atomically
  /// written to / read from.
  ::mlir::Value getX();
  /// Returns the first operation in atomic update region.
  ::mlir::Operation *getFirstOp();
  /// Returns true if the new value is same as old value and the operation is
  /// a no-op, false otherwise.
  bool isNoOp();
  /// Returns the new value if the operation is equivalent to just a write
  /// operation. Otherwise, returns nullptr.
  ::mlir::Value getWriteOpVal();
  /// Common verifier for operation that implements atomic update interface.
  ::llvm::LogicalResult verifyCommon();
  /// Common verifier of the required region for operation that implements
  /// atomic update interface.
  ::llvm::LogicalResult verifyRegionsCommon();
};
namespace detail {
  template <typename ConcreteOp>
  struct AtomicUpdateOpInterfaceTrait : public ::mlir::OpInterface<AtomicUpdateOpInterface, detail::AtomicUpdateOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the first operation in atomic update region.
    ::mlir::Operation *getFirstOp() {
      return &((*static_cast<ConcreteOp *>(this)).getRegion().front().getOperations().front());
    }
    /// Returns true if the new value is same as old value and the operation is
    /// a no-op, false otherwise.
    bool isNoOp() {
      // The atomic update is a no-op if the terminator is the first and only
        // operation in its region.
        mlir::Operation* terminator =
          llvm::dyn_cast<mlir::RegionBranchTerminatorOpInterface>((*static_cast<ConcreteOp *>(this)).getFirstOp());
        return terminator && terminator->getOperands().front() ==
          (*static_cast<ConcreteOp *>(this)).getRegion().front().getArgument(0);
    }
    /// Returns the new value if the operation is equivalent to just a write
    /// operation. Otherwise, returns nullptr.
    ::mlir::Value getWriteOpVal() {
      mlir::Operation* terminator =
          llvm::dyn_cast<mlir::RegionBranchTerminatorOpInterface>((*static_cast<ConcreteOp *>(this)).getFirstOp());
        if (terminator && terminator->getOperands().front() !=
          (*static_cast<ConcreteOp *>(this)).getRegion().front().getArgument(0)) {
          return terminator->getOperands().front();
        }
        return nullptr;
    }
    /// Common verifier for operation that implements atomic update interface.
    ::llvm::LogicalResult verifyCommon() {
      if ((*static_cast<ConcreteOp *>(this)).getRegion().getNumArguments() != 1)
          return (*static_cast<ConcreteOp *>(this)).emitError("the region must accept exactly one argument");

        Type elementType = (*static_cast<ConcreteOp *>(this)).getX().getType().getElementType();
        if (elementType && elementType != (*static_cast<ConcreteOp *>(this)).getRegion().getArgument(0).getType()) {
          return (*static_cast<ConcreteOp *>(this)).emitError("the type of the operand must be a pointer type whose "
                          "element type is the same as that of the region argument");
        }

        return mlir::success();
    }
    /// Common verifier of the required region for operation that implements
    /// atomic update interface.
    ::llvm::LogicalResult verifyRegionsCommon() {
      mlir::Operation *terminator = (*static_cast<ConcreteOp *>(this)).getRegion().front().getTerminator();

        if (terminator->getOperands().size() != 1)
          return (*static_cast<ConcreteOp *>(this)).emitError("only updated value must be returned");

        if (terminator->getOperands().front().getType() !=
            (*static_cast<ConcreteOp *>(this)).getRegion().getArgument(0).getType())
          return (*static_cast<ConcreteOp *>(this)).emitError("input and yielded value must have the same type");

        return mlir::success();
    }
  };
}// namespace detail
} // namespace accomp
} // namespace mlir
namespace mlir {
namespace accomp {
class AtomicCaptureOpInterface;
namespace detail {
struct AtomicCaptureOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Operation *(*getFirstOp)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Operation *(*getSecondOp)(const Concept *impl, ::mlir::Operation *);
    ::llvm::LogicalResult (*verifyRegionsCommon)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::accomp::AtomicCaptureOpInterface;
    Model() : Concept{getFirstOp, getSecondOp, verifyRegionsCommon} {}

    static inline ::mlir::Operation *getFirstOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation *getSecondOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyRegionsCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::accomp::AtomicCaptureOpInterface;
    FallbackModel() : Concept{getFirstOp, getSecondOp, verifyRegionsCommon} {}

    static inline ::mlir::Operation *getFirstOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation *getSecondOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::LogicalResult verifyRegionsCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Operation *getFirstOp(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Operation *getSecondOp(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::LogicalResult verifyRegionsCommon(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct AtomicCaptureOpInterfaceTrait;

} // namespace detail
class AtomicCaptureOpInterface : public ::mlir::OpInterface<AtomicCaptureOpInterface, detail::AtomicCaptureOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AtomicCaptureOpInterface, detail::AtomicCaptureOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AtomicCaptureOpInterfaceTrait<ConcreteOp> {};
  /// Returns the first operation in atomic capture region.
  ::mlir::Operation *getFirstOp();
  /// Returns the second operation in atomic capture region.
  ::mlir::Operation *getSecondOp();
  /// Common verifier of the required region for operation that implements
  /// atomic capture interface.
  ::llvm::LogicalResult verifyRegionsCommon();
};
namespace detail {
  template <typename ConcreteOp>
  struct AtomicCaptureOpInterfaceTrait : public ::mlir::OpInterface<AtomicCaptureOpInterface, detail::AtomicCaptureOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the first operation in atomic capture region.
    ::mlir::Operation *getFirstOp() {
      return &((*static_cast<ConcreteOp *>(this)).getRegion().front().getOperations().front());
    }
    /// Returns the second operation in atomic capture region.
    ::mlir::Operation *getSecondOp() {
      auto &ops = (*static_cast<ConcreteOp *>(this)).getRegion().front().getOperations();
        return ops.getNextNode(ops.front());
    }
    /// Common verifier of the required region for operation that implements
    /// atomic capture interface.
    ::llvm::LogicalResult verifyRegionsCommon() {
      Block::OpListType &ops = (*static_cast<ConcreteOp *>(this)).getRegion().front().getOperations();
        if (ops.size() != 3)
          return (*static_cast<ConcreteOp *>(this)).emitError()
                << "expected three operations in atomic.capture region (one "
                    "terminator, and two atomic ops)";
        auto &firstOp = ops.front();
        auto &secondOp = *ops.getNextNode(firstOp);
        auto firstReadStmt = dyn_cast<AtomicReadOpInterface>(firstOp);
        auto firstUpdateStmt = dyn_cast<AtomicUpdateOpInterface>(firstOp);
        auto secondReadStmt = dyn_cast<AtomicReadOpInterface>(secondOp);
        auto secondUpdateStmt = dyn_cast<AtomicUpdateOpInterface>(secondOp);
        auto secondWriteStmt = dyn_cast<AtomicWriteOpInterface>(secondOp);

        if (!((firstUpdateStmt && secondReadStmt) ||
              (firstReadStmt && secondUpdateStmt) ||
              (firstReadStmt && secondWriteStmt)))
          return ops.front().emitError()
                << "invalid sequence of operations in the capture region";
        if (firstUpdateStmt && secondReadStmt &&
            firstUpdateStmt.getX() != secondReadStmt.getX())
          return firstUpdateStmt.emitError()
                << "updated variable in atomic.update must be captured in "
                    "second operation";
        if (firstReadStmt && secondUpdateStmt &&
            firstReadStmt.getX() != secondUpdateStmt.getX())
          return firstReadStmt.emitError()
                << "captured variable in atomic.read must be updated in second "
                    "operation";
        if (firstReadStmt && secondWriteStmt &&
            firstReadStmt.getX() != secondWriteStmt.getX())
          return firstReadStmt.emitError()
                << "captured variable in atomic.read must be updated in "
                    "second operation";

        return mlir::success();
    }
  };
}// namespace detail
} // namespace accomp
} // namespace mlir
namespace mlir {
namespace accomp {
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyCommon();
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getX();
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getV(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getV();
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyCommon(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getX(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getV(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getV(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::AtomicReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyCommon(::mlir::Operation *tablegen_opaque_val) const {
if ((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getX() == (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getV()) {
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitError(
            "read and write must not be to the same location for atomic reads");
        }
        return mlir::success();
}
} // namespace accomp
} // namespace mlir
namespace mlir {
namespace accomp {
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyCommon();
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getX();
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getExpr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getExpr();
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyCommon(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getX(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getExpr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getExpr(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::AtomicWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyCommon(::mlir::Operation *tablegen_opaque_val) const {
mlir::Type elementType = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getX().getType().getElementType();
        if (elementType && elementType != (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getExpr().getType())
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitError("address must dereference to value type");
        return mlir::success();
}
} // namespace accomp
} // namespace mlir
namespace mlir {
namespace accomp {
template<typename ConcreteOp>
::mlir::Value detail::AtomicUpdateOpInterfaceInterfaceTraits::Model<ConcreteOp>::getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getX();
}
template<typename ConcreteOp>
::mlir::Operation *detail::AtomicUpdateOpInterfaceInterfaceTraits::Model<ConcreteOp>::getFirstOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFirstOp();
}
template<typename ConcreteOp>
bool detail::AtomicUpdateOpInterfaceInterfaceTraits::Model<ConcreteOp>::isNoOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isNoOp();
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicUpdateOpInterfaceInterfaceTraits::Model<ConcreteOp>::getWriteOpVal(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getWriteOpVal();
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicUpdateOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyCommon();
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicUpdateOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyRegionsCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyRegionsCommon();
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicUpdateOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getX(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getX(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Operation *detail::AtomicUpdateOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFirstOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFirstOp(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::AtomicUpdateOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isNoOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isNoOp(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::AtomicUpdateOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getWriteOpVal(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getWriteOpVal(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicUpdateOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyCommon(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicUpdateOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyRegionsCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyRegionsCommon(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Operation *detail::AtomicUpdateOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFirstOp(::mlir::Operation *tablegen_opaque_val) const {
return &((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front().getOperations().front());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::AtomicUpdateOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isNoOp(::mlir::Operation *tablegen_opaque_val) const {
// The atomic update is a no-op if the terminator is the first and only
        // operation in its region.
        mlir::Operation* terminator =
          llvm::dyn_cast<mlir::RegionBranchTerminatorOpInterface>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFirstOp());
        return terminator && terminator->getOperands().front() ==
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front().getArgument(0);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::AtomicUpdateOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getWriteOpVal(::mlir::Operation *tablegen_opaque_val) const {
mlir::Operation* terminator =
          llvm::dyn_cast<mlir::RegionBranchTerminatorOpInterface>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFirstOp());
        if (terminator && terminator->getOperands().front() !=
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front().getArgument(0)) {
          return terminator->getOperands().front();
        }
        return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::AtomicUpdateOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyCommon(::mlir::Operation *tablegen_opaque_val) const {
if ((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().getNumArguments() != 1)
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitError("the region must accept exactly one argument");

        Type elementType = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getX().getType().getElementType();
        if (elementType && elementType != (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().getArgument(0).getType()) {
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitError("the type of the operand must be a pointer type whose "
                          "element type is the same as that of the region argument");
        }

        return mlir::success();
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::AtomicUpdateOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyRegionsCommon(::mlir::Operation *tablegen_opaque_val) const {
mlir::Operation *terminator = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front().getTerminator();

        if (terminator->getOperands().size() != 1)
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitError("only updated value must be returned");

        if (terminator->getOperands().front().getType() !=
            (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().getArgument(0).getType())
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitError("input and yielded value must have the same type");

        return mlir::success();
}
} // namespace accomp
} // namespace mlir
namespace mlir {
namespace accomp {
template<typename ConcreteOp>
::mlir::Operation *detail::AtomicCaptureOpInterfaceInterfaceTraits::Model<ConcreteOp>::getFirstOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFirstOp();
}
template<typename ConcreteOp>
::mlir::Operation *detail::AtomicCaptureOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSecondOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSecondOp();
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicCaptureOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyRegionsCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyRegionsCommon();
}
template<typename ConcreteOp>
::mlir::Operation *detail::AtomicCaptureOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFirstOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFirstOp(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Operation *detail::AtomicCaptureOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSecondOp(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSecondOp(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::AtomicCaptureOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyRegionsCommon(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyRegionsCommon(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Operation *detail::AtomicCaptureOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFirstOp(::mlir::Operation *tablegen_opaque_val) const {
return &((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front().getOperations().front());
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Operation *detail::AtomicCaptureOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSecondOp(::mlir::Operation *tablegen_opaque_val) const {
auto &ops = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front().getOperations();
        return ops.getNextNode(ops.front());
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::AtomicCaptureOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyRegionsCommon(::mlir::Operation *tablegen_opaque_val) const {
Block::OpListType &ops = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front().getOperations();
        if (ops.size() != 3)
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitError()
                << "expected three operations in atomic.capture region (one "
                    "terminator, and two atomic ops)";
        auto &firstOp = ops.front();
        auto &secondOp = *ops.getNextNode(firstOp);
        auto firstReadStmt = dyn_cast<AtomicReadOpInterface>(firstOp);
        auto firstUpdateStmt = dyn_cast<AtomicUpdateOpInterface>(firstOp);
        auto secondReadStmt = dyn_cast<AtomicReadOpInterface>(secondOp);
        auto secondUpdateStmt = dyn_cast<AtomicUpdateOpInterface>(secondOp);
        auto secondWriteStmt = dyn_cast<AtomicWriteOpInterface>(secondOp);

        if (!((firstUpdateStmt && secondReadStmt) ||
              (firstReadStmt && secondUpdateStmt) ||
              (firstReadStmt && secondWriteStmt)))
          return ops.front().emitError()
                << "invalid sequence of operations in the capture region";
        if (firstUpdateStmt && secondReadStmt &&
            firstUpdateStmt.getX() != secondReadStmt.getX())
          return firstUpdateStmt.emitError()
                << "updated variable in atomic.update must be captured in "
                    "second operation";
        if (firstReadStmt && secondUpdateStmt &&
            firstReadStmt.getX() != secondUpdateStmt.getX())
          return firstReadStmt.emitError()
                << "captured variable in atomic.read must be updated in second "
                    "operation";
        if (firstReadStmt && secondWriteStmt &&
            firstReadStmt.getX() != secondWriteStmt.getX())
          return firstReadStmt.emitError()
                << "captured variable in atomic.read must be updated in "
                    "second operation";

        return mlir::success();
}
} // namespace accomp
} // namespace mlir
