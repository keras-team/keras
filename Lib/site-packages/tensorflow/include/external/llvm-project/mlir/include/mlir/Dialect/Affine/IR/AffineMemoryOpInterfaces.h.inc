/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace affine {
class AffineReadOpInterface;
namespace detail {
struct AffineReadOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getMemRef)(const Concept *impl, ::mlir::Operation *);
    ::mlir::MemRefType (*getMemRefType)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Operation::operand_range (*getMapOperands)(const Concept *impl, ::mlir::Operation *);
    ::mlir::AffineMap (*getAffineMap)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getValue)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::affine::AffineReadOpInterface;
    Model() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValue} {}

    static inline ::mlir::Value getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MemRefType getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation::operand_range getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::AffineMap getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::affine::AffineReadOpInterface;
    FallbackModel() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValue} {}

    static inline ::mlir::Value getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MemRefType getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation::operand_range getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::AffineMap getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Value getMemRef(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Value getValue(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct AffineReadOpInterfaceTrait;

} // namespace detail
class AffineReadOpInterface : public ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AffineReadOpInterfaceTrait<ConcreteOp> {};
  /// Returns the memref operand to read from.
  ::mlir::Value getMemRef();
  /// Returns the type of the memref operand.
  ::mlir::MemRefType getMemRefType();
  /// Returns affine map operands.
  ::mlir::Operation::operand_range getMapOperands();
  /// Returns the affine map used to index the memref for this operation.
  ::mlir::AffineMap getAffineMap();
  /// Returns the value read by this operation.
  ::mlir::Value getValue();
};
namespace detail {
  template <typename ConcreteOp>
  struct AffineReadOpInterfaceTrait : public ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the memref operand to read from.
    ::mlir::Value getMemRef() {
      return (*static_cast<ConcreteOp *>(this)).getOperand((*static_cast<ConcreteOp *>(this)).getMemRefOperandIndex());
    }
    /// Returns the type of the memref operand.
    ::mlir::MemRefType getMemRefType() {
      return ::llvm::cast<::mlir::MemRefType>((*static_cast<ConcreteOp *>(this)).getMemRef().getType());
    }
    /// Returns affine map operands.
    ::mlir::Operation::operand_range getMapOperands() {
      return llvm::drop_begin((*static_cast<ConcreteOp *>(this)).getOperands(), 1);
    }
    /// Returns the affine map used to index the memref for this operation.
    ::mlir::AffineMap getAffineMap() {
      return (*static_cast<ConcreteOp *>(this)).getAffineMapAttr().getValue();
    }
    /// Returns the value read by this operation.
    ::mlir::Value getValue() {
      return (*static_cast<ConcreteOp *>(this));
    }
  };
}// namespace detail
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineWriteOpInterface;
namespace detail {
struct AffineWriteOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getMemRef)(const Concept *impl, ::mlir::Operation *);
    ::mlir::MemRefType (*getMemRefType)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Operation::operand_range (*getMapOperands)(const Concept *impl, ::mlir::Operation *);
    ::mlir::AffineMap (*getAffineMap)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getValueToStore)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::affine::AffineWriteOpInterface;
    Model() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValueToStore} {}

    static inline ::mlir::Value getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MemRefType getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation::operand_range getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::AffineMap getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getValueToStore(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::affine::AffineWriteOpInterface;
    FallbackModel() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValueToStore} {}

    static inline ::mlir::Value getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MemRefType getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation::operand_range getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::AffineMap getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getValueToStore(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Value getMemRef(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Value getValueToStore(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct AffineWriteOpInterfaceTrait;

} // namespace detail
class AffineWriteOpInterface : public ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AffineWriteOpInterfaceTrait<ConcreteOp> {};
  /// Returns the memref operand to write to.
  ::mlir::Value getMemRef();
  /// Returns the type of the memref operand.
  ::mlir::MemRefType getMemRefType();
  /// Returns affine map operands.
  ::mlir::Operation::operand_range getMapOperands();
  /// Returns the affine map used to index the memref for this operation.
  ::mlir::AffineMap getAffineMap();
  /// Returns the value to store.
  ::mlir::Value getValueToStore();
};
namespace detail {
  template <typename ConcreteOp>
  struct AffineWriteOpInterfaceTrait : public ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the memref operand to write to.
    ::mlir::Value getMemRef() {
      return (*static_cast<ConcreteOp *>(this)).getOperand((*static_cast<ConcreteOp *>(this)).getMemRefOperandIndex());
    }
    /// Returns the type of the memref operand.
    ::mlir::MemRefType getMemRefType() {
      return ::llvm::cast<::mlir::MemRefType>((*static_cast<ConcreteOp *>(this)).getMemRef().getType());
    }
    /// Returns affine map operands.
    ::mlir::Operation::operand_range getMapOperands() {
      return llvm::drop_begin((*static_cast<ConcreteOp *>(this)).getOperands(), 2);
    }
    /// Returns the affine map used to index the memref for this operation.
    ::mlir::AffineMap getAffineMap() {
      return (*static_cast<ConcreteOp *>(this)).getAffineMapAttr().getValue();
    }
    /// Returns the value to store.
    ::mlir::Value getValueToStore() {
      return (*static_cast<ConcreteOp *>(this)).getOperand((*static_cast<ConcreteOp *>(this)).getStoredValOperandIndex());
    }
  };
}// namespace detail
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineMapAccessInterface;
namespace detail {
struct AffineMapAccessInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::NamedAttribute (*getAffineMapAttrForMemRef)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::affine::AffineMapAccessInterface;
    Model() : Concept{getAffineMapAttrForMemRef} {}

    static inline ::mlir::NamedAttribute getAffineMapAttrForMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value memref);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::affine::AffineMapAccessInterface;
    FallbackModel() : Concept{getAffineMapAttrForMemRef} {}

    static inline ::mlir::NamedAttribute getAffineMapAttrForMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value memref);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::NamedAttribute getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, ::mlir::Value memref) const;
  };
};
template <typename ConcreteOp>
struct AffineMapAccessInterfaceTrait;

} // namespace detail
class AffineMapAccessInterface : public ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AffineMapAccessInterfaceTrait<ConcreteOp> {};
  /// Returns the AffineMapAttr associated with 'memref'.
  ::mlir::NamedAttribute getAffineMapAttrForMemRef(::mlir::Value memref);
};
namespace detail {
  template <typename ConcreteOp>
  struct AffineMapAccessInterfaceTrait : public ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the AffineMapAttr associated with 'memref'.
    ::mlir::NamedAttribute getAffineMapAttrForMemRef(::mlir::Value memref) {
      assert(memref == (*static_cast<ConcreteOp *>(this)).getMemRef() &&
               "Expected memref argument to match memref operand");
        return {::mlir::StringAttr::get(
                    (*static_cast<ConcreteOp *>(this)).getContext(), (*static_cast<ConcreteOp *>(this)).getMapAttrStrName()),
                    (*static_cast<ConcreteOp *>(this)).getAffineMapAttr()};
    }
  };
}// namespace detail
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
template<typename ConcreteOp>
::mlir::Value detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef();
}
template<typename ConcreteOp>
::mlir::MemRefType detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
}
template<typename ConcreteOp>
::mlir::Operation::operand_range detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapOperands();
}
template<typename ConcreteOp>
::mlir::AffineMap detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMap();
}
template<typename ConcreteOp>
::mlir::Value detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValue();
}
template<typename ConcreteOp>
::mlir::Value detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemRef(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::MemRefType detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemRefType(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Operation::operand_range detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMapOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::AffineMap detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAffineMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getValue(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMemRef(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperand((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefOperandIndex());
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::MemRefType detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMemRefType(::mlir::Operation *tablegen_opaque_val) const {
return ::llvm::cast<::mlir::MemRefType>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef().getType());
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Operation::operand_range detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMapOperands(::mlir::Operation *tablegen_opaque_val) const {
return llvm::drop_begin((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperands(), 1);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::AffineMap detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAffineMap(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMapAttr().getValue();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getValue(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val));
}
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
template<typename ConcreteOp>
::mlir::Value detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef();
}
template<typename ConcreteOp>
::mlir::MemRefType detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
}
template<typename ConcreteOp>
::mlir::Operation::operand_range detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapOperands();
}
template<typename ConcreteOp>
::mlir::AffineMap detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMap();
}
template<typename ConcreteOp>
::mlir::Value detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getValueToStore(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValueToStore();
}
template<typename ConcreteOp>
::mlir::Value detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemRef(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::MemRefType detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemRefType(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Operation::operand_range detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMapOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::AffineMap detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAffineMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getValueToStore(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getValueToStore(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMemRef(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperand((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefOperandIndex());
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::MemRefType detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMemRefType(::mlir::Operation *tablegen_opaque_val) const {
return ::llvm::cast<::mlir::MemRefType>((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef().getType());
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Operation::operand_range detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMapOperands(::mlir::Operation *tablegen_opaque_val) const {
return llvm::drop_begin((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperands(), 2);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::AffineMap detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAffineMap(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMapAttr().getValue();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getValueToStore(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperand((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStoredValOperandIndex());
}
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
template<typename ConcreteOp>
::mlir::NamedAttribute detail::AffineMapAccessInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMapAttrForMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value memref) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMapAttrForMemRef(memref);
}
template<typename ConcreteOp>
::mlir::NamedAttribute detail::AffineMapAccessInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAffineMapAttrForMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value memref) {
  return static_cast<const ConcreteOp *>(impl)->getAffineMapAttrForMemRef(tablegen_opaque_val, memref);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::NamedAttribute detail::AffineMapAccessInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, ::mlir::Value memref) const {
assert(memref == (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef() &&
               "Expected memref argument to match memref operand");
        return {::mlir::StringAttr::get(
                    (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getContext(), (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapAttrStrName()),
                    (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMapAttr()};
}
} // namespace affine
} // namespace mlir
