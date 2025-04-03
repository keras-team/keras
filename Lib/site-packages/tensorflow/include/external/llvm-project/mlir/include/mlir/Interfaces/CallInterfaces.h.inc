/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class CallOpInterface;
namespace detail {
struct CallOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::CallInterfaceCallable (*getCallableForCallee)(const Concept *impl, ::mlir::Operation *);
    void (*setCalleeFromCallable)(const Concept *impl, ::mlir::Operation *, ::mlir::CallInterfaceCallable);
    ::mlir::Operation::operand_range (*getArgOperands)(const Concept *impl, ::mlir::Operation *);
    ::mlir::MutableOperandRange (*getArgOperandsMutable)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Operation *(*resolveCallableInTable)(const Concept *impl, ::mlir::Operation *, ::mlir::SymbolTableCollection *);
    ::mlir::Operation *(*resolveCallable)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::CallOpInterface;
    Model() : Concept{getCallableForCallee, setCalleeFromCallable, getArgOperands, getArgOperandsMutable, resolveCallableInTable, resolveCallable} {}

    static inline ::mlir::CallInterfaceCallable getCallableForCallee(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setCalleeFromCallable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::CallInterfaceCallable callee);
    static inline ::mlir::Operation::operand_range getArgOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getArgOperandsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation *resolveCallableInTable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection * symbolTable);
    static inline ::mlir::Operation *resolveCallable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::CallOpInterface;
    FallbackModel() : Concept{getCallableForCallee, setCalleeFromCallable, getArgOperands, getArgOperandsMutable, resolveCallableInTable, resolveCallable} {}

    static inline ::mlir::CallInterfaceCallable getCallableForCallee(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setCalleeFromCallable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::CallInterfaceCallable callee);
    static inline ::mlir::Operation::operand_range getArgOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getArgOperandsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Operation *resolveCallableInTable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection * symbolTable);
    static inline ::mlir::Operation *resolveCallable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Operation *resolveCallableInTable(::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection *symbolTable) const;
    ::mlir::Operation *resolveCallable(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct CallOpInterfaceTrait;

} // namespace detail
class CallOpInterface : public ::mlir::OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::CallOpInterfaceTrait<ConcreteOp> {};
  /// Returns the callee of this call-like operation. A `callee` is either a
  /// reference to a symbol, via SymbolRefAttr, or a reference to a defined
  /// SSA value. If the reference is an SSA value, the SSA value corresponds
  /// to a region of a lambda-like operation.
  ::mlir::CallInterfaceCallable getCallableForCallee();
  /// Sets the callee of this call-like operation. A `callee` is either a
  /// reference to a symbol, via SymbolRefAttr, or a reference to a defined
  /// SSA value. The type of the `callee` is expected to be the same as the
  /// return type of `getCallableForCallee`, e.g., `callee` should be
  /// SymbolRefAttr for `func.call`.
  void setCalleeFromCallable(::mlir::CallInterfaceCallable callee);
  /// Returns the operands within this call that are used as arguments to the
  /// callee.
  ::mlir::Operation::operand_range getArgOperands();
  /// Returns the operands within this call that are used as arguments to the
  /// callee as a mutable range.
  ::mlir::MutableOperandRange getArgOperandsMutable();
  /// Resolve the callable operation for given callee to a
  /// CallableOpInterface, or nullptr if a valid callable was not resolved.
  /// `symbolTable` parameter allow for using a cached symbol table for symbol
  /// lookups instead of performing an O(N) scan.
  ::mlir::Operation *resolveCallableInTable(::mlir::SymbolTableCollection * symbolTable);
  /// Resolve the callable operation for given callee to a
  /// CallableOpInterface, or nullptr if a valid callable was not resolved.
  ::mlir::Operation *resolveCallable();
};
namespace detail {
  template <typename ConcreteOp>
  struct CallOpInterfaceTrait : public ::mlir::OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Resolve the callable operation for given callee to a
    /// CallableOpInterface, or nullptr if a valid callable was not resolved.
    /// `symbolTable` parameter allow for using a cached symbol table for symbol
    /// lookups instead of performing an O(N) scan.
    ::mlir::Operation *resolveCallableInTable(::mlir::SymbolTableCollection * symbolTable) {
      return ::mlir::call_interface_impl::resolveCallable((*static_cast<ConcreteOp *>(this)), symbolTable);
    }
    /// Resolve the callable operation for given callee to a
    /// CallableOpInterface, or nullptr if a valid callable was not resolved.
    ::mlir::Operation *resolveCallable() {
      return ::mlir::call_interface_impl::resolveCallable((*static_cast<ConcreteOp *>(this)));
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class CallableOpInterface;
namespace detail {
struct CallableOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Region *(*getCallableRegion)(const Concept *impl, ::mlir::Operation *);
    ::llvm::ArrayRef<::mlir::Type> (*getArgumentTypes)(const Concept *impl, ::mlir::Operation *);
    ::llvm::ArrayRef<::mlir::Type> (*getResultTypes)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ArrayAttr (*getArgAttrsAttr)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ArrayAttr (*getResAttrsAttr)(const Concept *impl, ::mlir::Operation *);
    void (*setArgAttrsAttr)(const Concept *impl, ::mlir::Operation *, ::mlir::ArrayAttr);
    void (*setResAttrsAttr)(const Concept *impl, ::mlir::Operation *, ::mlir::ArrayAttr);
    ::mlir::Attribute (*removeArgAttrsAttr)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Attribute (*removeResAttrsAttr)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::CallableOpInterface;
    Model() : Concept{getCallableRegion, getArgumentTypes, getResultTypes, getArgAttrsAttr, getResAttrsAttr, setArgAttrsAttr, setResAttrsAttr, removeArgAttrsAttr, removeResAttrsAttr} {}

    static inline ::mlir::Region *getCallableRegion(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr getArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr getResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs);
    static inline void setResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs);
    static inline ::mlir::Attribute removeArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Attribute removeResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::CallableOpInterface;
    FallbackModel() : Concept{getCallableRegion, getArgumentTypes, getResultTypes, getArgAttrsAttr, getResAttrsAttr, setArgAttrsAttr, setResAttrsAttr, removeArgAttrsAttr, removeResAttrsAttr} {}

    static inline ::mlir::Region *getCallableRegion(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr getArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr getResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs);
    static inline void setResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs);
    static inline ::mlir::Attribute removeArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Attribute removeResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::ArrayAttr getArgAttrsAttr(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::ArrayAttr getResAttrsAttr(::mlir::Operation *tablegen_opaque_val) const;
    void setArgAttrsAttr(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) const;
    void setResAttrsAttr(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) const;
    ::mlir::Attribute removeArgAttrsAttr(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Attribute removeResAttrsAttr(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct CallableOpInterfaceTrait;

} // namespace detail
class CallableOpInterface : public ::mlir::OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::CallableOpInterfaceTrait<ConcreteOp> {};
  /// Returns the region on the current operation that is callable. This may
  /// return null in the case of an external callable object, e.g. an external
  /// function.
  ::mlir::Region *getCallableRegion();
  /// Returns the callable's argument types based exclusively on the type (to
  /// allow for this method may be called on function declarations).
  ::llvm::ArrayRef<::mlir::Type> getArgumentTypes();
  /// Returns the callable's result types based exclusively on the type (to
  /// allow for this method may be called on function declarations).
  ::llvm::ArrayRef<::mlir::Type> getResultTypes();
  /// Get the array of argument attribute dictionaries. The method should
  /// return an array attribute containing only dictionary attributes equal in
  /// number to the number of region arguments. Alternatively, the method can
  /// return null to indicate that the region has no argument attributes.
  ::mlir::ArrayAttr getArgAttrsAttr();
  /// Get the array of result attribute dictionaries. The method should return
  /// an array attribute containing only dictionary attributes equal in number
  /// to the number of region results. Alternatively, the method can return
  /// null to indicate that the region has no result attributes.
  ::mlir::ArrayAttr getResAttrsAttr();
  /// Set the array of argument attribute dictionaries.
  void setArgAttrsAttr(::mlir::ArrayAttr attrs);
  /// Set the array of result attribute dictionaries.
  void setResAttrsAttr(::mlir::ArrayAttr attrs);
  /// Remove the array of argument attribute dictionaries. This is the same as
  /// setting all argument attributes to an empty dictionary. The method should
  /// return the removed attribute.
  ::mlir::Attribute removeArgAttrsAttr();
  /// Remove the array of result attribute dictionaries. This is the same as
  /// setting all result attributes to an empty dictionary. The method should
  /// return the removed attribute.
  ::mlir::Attribute removeResAttrsAttr();
};
namespace detail {
  template <typename ConcreteOp>
  struct CallableOpInterfaceTrait : public ::mlir::OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Get the array of argument attribute dictionaries. The method should
    /// return an array attribute containing only dictionary attributes equal in
    /// number to the number of region arguments. Alternatively, the method can
    /// return null to indicate that the region has no argument attributes.
    ::mlir::ArrayAttr getArgAttrsAttr() {
      return nullptr;
    }
    /// Get the array of result attribute dictionaries. The method should return
    /// an array attribute containing only dictionary attributes equal in number
    /// to the number of region results. Alternatively, the method can return
    /// null to indicate that the region has no result attributes.
    ::mlir::ArrayAttr getResAttrsAttr() {
      return nullptr;
    }
    /// Set the array of argument attribute dictionaries.
    void setArgAttrsAttr(::mlir::ArrayAttr attrs) {
      return;
    }
    /// Set the array of result attribute dictionaries.
    void setResAttrsAttr(::mlir::ArrayAttr attrs) {
      return;
    }
    /// Remove the array of argument attribute dictionaries. This is the same as
    /// setting all argument attributes to an empty dictionary. The method should
    /// return the removed attribute.
    ::mlir::Attribute removeArgAttrsAttr() {
      return nullptr;
    }
    /// Remove the array of result attribute dictionaries. This is the same as
    /// setting all result attributes to an empty dictionary. The method should
    /// return the removed attribute.
    ::mlir::Attribute removeResAttrsAttr() {
      return nullptr;
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::CallInterfaceCallable detail::CallOpInterfaceInterfaceTraits::Model<ConcreteOp>::getCallableForCallee(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCallableForCallee();
}
template<typename ConcreteOp>
void detail::CallOpInterfaceInterfaceTraits::Model<ConcreteOp>::setCalleeFromCallable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::CallInterfaceCallable callee) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setCalleeFromCallable(callee);
}
template<typename ConcreteOp>
::mlir::Operation::operand_range detail::CallOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArgOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgOperands();
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::CallOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArgOperandsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgOperandsMutable();
}
template<typename ConcreteOp>
::mlir::Operation *detail::CallOpInterfaceInterfaceTraits::Model<ConcreteOp>::resolveCallableInTable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection * symbolTable) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).resolveCallableInTable(symbolTable);
}
template<typename ConcreteOp>
::mlir::Operation *detail::CallOpInterfaceInterfaceTraits::Model<ConcreteOp>::resolveCallable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).resolveCallable();
}
template<typename ConcreteOp>
::mlir::CallInterfaceCallable detail::CallOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getCallableForCallee(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCallableForCallee(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::CallOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setCalleeFromCallable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::CallInterfaceCallable callee) {
  return static_cast<const ConcreteOp *>(impl)->setCalleeFromCallable(tablegen_opaque_val, callee);
}
template<typename ConcreteOp>
::mlir::Operation::operand_range detail::CallOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getArgOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getArgOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::CallOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getArgOperandsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getArgOperandsMutable(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Operation *detail::CallOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::resolveCallableInTable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection * symbolTable) {
  return static_cast<const ConcreteOp *>(impl)->resolveCallableInTable(tablegen_opaque_val, symbolTable);
}
template<typename ConcreteOp>
::mlir::Operation *detail::CallOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::resolveCallable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->resolveCallable(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Operation *detail::CallOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::resolveCallableInTable(::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection *symbolTable) const {
return ::mlir::call_interface_impl::resolveCallable((llvm::cast<ConcreteOp>(tablegen_opaque_val)), symbolTable);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Operation *detail::CallOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::resolveCallable(::mlir::Operation *tablegen_opaque_val) const {
return ::mlir::call_interface_impl::resolveCallable((llvm::cast<ConcreteOp>(tablegen_opaque_val)));
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::Region *detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getCallableRegion(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCallableRegion();
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgumentTypes();
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultTypes();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgAttrsAttr();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResAttrsAttr();
}
template<typename ConcreteOp>
void detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::setArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setArgAttrsAttr(attrs);
}
template<typename ConcreteOp>
void detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::setResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setResAttrsAttr(attrs);
}
template<typename ConcreteOp>
::mlir::Attribute detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::removeArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeArgAttrsAttr();
}
template<typename ConcreteOp>
::mlir::Attribute detail::CallableOpInterfaceInterfaceTraits::Model<ConcreteOp>::removeResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).removeResAttrsAttr();
}
template<typename ConcreteOp>
::mlir::Region *detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getCallableRegion(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCallableRegion(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getArgumentTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResultTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getArgAttrsAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResAttrsAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) {
  return static_cast<const ConcreteOp *>(impl)->setArgAttrsAttr(tablegen_opaque_val, attrs);
}
template<typename ConcreteOp>
void detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) {
  return static_cast<const ConcreteOp *>(impl)->setResAttrsAttr(tablegen_opaque_val, attrs);
}
template<typename ConcreteOp>
::mlir::Attribute detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::removeArgAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->removeArgAttrsAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Attribute detail::CallableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::removeResAttrsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->removeResAttrsAttr(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::CallableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getArgAttrsAttr(::mlir::Operation *tablegen_opaque_val) const {
return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::CallableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getResAttrsAttr(::mlir::Operation *tablegen_opaque_val) const {
return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::CallableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setArgAttrsAttr(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) const {
return;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::CallableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setResAttrsAttr(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attrs) const {
return;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Attribute detail::CallableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::removeArgAttrsAttr(::mlir::Operation *tablegen_opaque_val) const {
return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Attribute detail::CallableOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::removeResAttrsAttr(::mlir::Operation *tablegen_opaque_val) const {
return nullptr;
}
} // namespace mlir
