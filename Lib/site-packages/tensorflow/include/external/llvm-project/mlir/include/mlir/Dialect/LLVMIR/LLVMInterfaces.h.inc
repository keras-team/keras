/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
class FastmathFlagsInterface;
namespace detail {
struct FastmathFlagsInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::LLVM::FastmathFlagsAttr (*getFastmathAttr)(const Concept *impl, ::mlir::Operation *);
    ::llvm::StringRef (*getFastmathAttrName)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::FastmathFlagsInterface;
    Model() : Concept{getFastmathAttr, getFastmathAttrName} {}

    static inline ::mlir::LLVM::FastmathFlagsAttr getFastmathAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::StringRef getFastmathAttrName();
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::FastmathFlagsInterface;
    FallbackModel() : Concept{getFastmathAttr, getFastmathAttrName} {}

    static inline ::mlir::LLVM::FastmathFlagsAttr getFastmathAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::StringRef getFastmathAttrName();
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::LLVM::FastmathFlagsAttr getFastmathAttr(::mlir::Operation *tablegen_opaque_val) const;
    static ::llvm::StringRef getFastmathAttrName();
  };
};
template <typename ConcreteOp>
struct FastmathFlagsInterfaceTrait;

} // namespace detail
class FastmathFlagsInterface : public ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FastmathFlagsInterfaceTrait<ConcreteOp> {};
  /// Returns a FastmathFlagsAttr attribute for the operation
  ::mlir::LLVM::FastmathFlagsAttr getFastmathAttr();
  /// Returns the name of the FastmathFlagsAttr attribute
  ///                          for the operation
  ::llvm::StringRef getFastmathAttrName();
};
namespace detail {
  template <typename ConcreteOp>
  struct FastmathFlagsInterfaceTrait : public ::mlir::OpInterface<FastmathFlagsInterface, detail::FastmathFlagsInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a FastmathFlagsAttr attribute for the operation
    ::mlir::LLVM::FastmathFlagsAttr getFastmathAttr() {
      auto op = cast<ConcreteOp>(this->getOperation());
        return op.getFastmathFlagsAttr();
    }
    /// Returns the name of the FastmathFlagsAttr attribute
    ///                          for the operation
    static ::llvm::StringRef getFastmathAttrName() {
      return "fastmathFlags";
    }
  };
}// namespace detail
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
class IntegerOverflowFlagsInterface;
namespace detail {
struct IntegerOverflowFlagsInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    IntegerOverflowFlags (*getOverflowFlags)(const Concept *impl, ::mlir::Operation *);
    void (*setOverflowFlags)(const Concept *impl, ::mlir::Operation *, IntegerOverflowFlags);
    bool (*hasNoUnsignedWrap)(const Concept *impl, ::mlir::Operation *);
    bool (*hasNoSignedWrap)(const Concept *impl, ::mlir::Operation *);
    StringRef (*getOverflowFlagsAttrName)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::IntegerOverflowFlagsInterface;
    Model() : Concept{getOverflowFlags, setOverflowFlags, hasNoUnsignedWrap, hasNoSignedWrap, getOverflowFlagsAttrName} {}

    static inline IntegerOverflowFlags getOverflowFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setOverflowFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, IntegerOverflowFlags flags);
    static inline bool hasNoUnsignedWrap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasNoSignedWrap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef getOverflowFlagsAttrName();
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::IntegerOverflowFlagsInterface;
    FallbackModel() : Concept{getOverflowFlags, setOverflowFlags, hasNoUnsignedWrap, hasNoSignedWrap, getOverflowFlagsAttrName} {}

    static inline IntegerOverflowFlags getOverflowFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setOverflowFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, IntegerOverflowFlags flags);
    static inline bool hasNoUnsignedWrap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasNoSignedWrap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef getOverflowFlagsAttrName();
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    IntegerOverflowFlags getOverflowFlags(::mlir::Operation *tablegen_opaque_val) const;
    void setOverflowFlags(::mlir::Operation *tablegen_opaque_val, IntegerOverflowFlags flags) const;
    bool hasNoUnsignedWrap(::mlir::Operation *tablegen_opaque_val) const;
    bool hasNoSignedWrap(::mlir::Operation *tablegen_opaque_val) const;
    static StringRef getOverflowFlagsAttrName();
  };
};
template <typename ConcreteOp>
struct IntegerOverflowFlagsInterfaceTrait;

} // namespace detail
class IntegerOverflowFlagsInterface : public ::mlir::OpInterface<IntegerOverflowFlagsInterface, detail::IntegerOverflowFlagsInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<IntegerOverflowFlagsInterface, detail::IntegerOverflowFlagsInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::IntegerOverflowFlagsInterfaceTrait<ConcreteOp> {};
  /// Get the integer overflow flags for the operation.
  IntegerOverflowFlags getOverflowFlags();
  /// Set the integer overflow flags for the operation.
  void setOverflowFlags(IntegerOverflowFlags flags);
  /// Returns whether the operation has the No Unsigned Wrap keyword.
  bool hasNoUnsignedWrap();
  /// Returns whether the operation has the No Signed Wrap keyword.
  bool hasNoSignedWrap();
  /// Get the attribute name of the overflow flags property.
  StringRef getOverflowFlagsAttrName();
};
namespace detail {
  template <typename ConcreteOp>
  struct IntegerOverflowFlagsInterfaceTrait : public ::mlir::OpInterface<IntegerOverflowFlagsInterface, detail::IntegerOverflowFlagsInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Get the integer overflow flags for the operation.
    IntegerOverflowFlags getOverflowFlags() {
      return (*static_cast<ConcreteOp *>(this)).getProperties().overflowFlags;
    }
    /// Set the integer overflow flags for the operation.
    void setOverflowFlags(IntegerOverflowFlags flags) {
      (*static_cast<ConcreteOp *>(this)).getProperties().overflowFlags = flags;
    }
    /// Returns whether the operation has the No Unsigned Wrap keyword.
    bool hasNoUnsignedWrap() {
      return bitEnumContainsAll((*static_cast<ConcreteOp *>(this)).getOverflowFlags(),
                                IntegerOverflowFlags::nuw);
    }
    /// Returns whether the operation has the No Signed Wrap keyword.
    bool hasNoSignedWrap() {
      return bitEnumContainsAll((*static_cast<ConcreteOp *>(this)).getOverflowFlags(),
                                IntegerOverflowFlags::nsw);
    }
    /// Get the attribute name of the overflow flags property.
    static StringRef getOverflowFlagsAttrName() {
      return "overflowFlags";
    }
  };
}// namespace detail
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
class BranchWeightOpInterface;
namespace detail {
struct BranchWeightOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::DenseI32ArrayAttr (*getBranchWeightsOrNull)(const Concept *impl, ::mlir::Operation *);
    void (*setBranchWeights)(const Concept *impl, ::mlir::Operation *, ::mlir::DenseI32ArrayAttr);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::BranchWeightOpInterface;
    Model() : Concept{getBranchWeightsOrNull, setBranchWeights} {}

    static inline ::mlir::DenseI32ArrayAttr getBranchWeightsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setBranchWeights(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::DenseI32ArrayAttr attr);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::BranchWeightOpInterface;
    FallbackModel() : Concept{getBranchWeightsOrNull, setBranchWeights} {}

    static inline ::mlir::DenseI32ArrayAttr getBranchWeightsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setBranchWeights(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::DenseI32ArrayAttr attr);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::DenseI32ArrayAttr getBranchWeightsOrNull(::mlir::Operation *tablegen_opaque_val) const;
    void setBranchWeights(::mlir::Operation *tablegen_opaque_val, ::mlir::DenseI32ArrayAttr attr) const;
  };
};
template <typename ConcreteOp>
struct BranchWeightOpInterfaceTrait;

} // namespace detail
class BranchWeightOpInterface : public ::mlir::OpInterface<BranchWeightOpInterface, detail::BranchWeightOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BranchWeightOpInterface, detail::BranchWeightOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BranchWeightOpInterfaceTrait<ConcreteOp> {};
  /// Returns the branch weights attribute or nullptr
  ::mlir::DenseI32ArrayAttr getBranchWeightsOrNull();
  /// Sets the branch weights attribute
  void setBranchWeights(::mlir::DenseI32ArrayAttr attr);
};
namespace detail {
  template <typename ConcreteOp>
  struct BranchWeightOpInterfaceTrait : public ::mlir::OpInterface<BranchWeightOpInterface, detail::BranchWeightOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the branch weights attribute or nullptr
    ::mlir::DenseI32ArrayAttr getBranchWeightsOrNull() {
      auto op = cast<ConcreteOp>(this->getOperation());
        return op.getBranchWeightsAttr();
    }
    /// Sets the branch weights attribute
    void setBranchWeights(::mlir::DenseI32ArrayAttr attr) {
      auto op = cast<ConcreteOp>(this->getOperation());
        op.setBranchWeightsAttr(attr);
    }
  };
}// namespace detail
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
class AccessGroupOpInterface;
namespace detail {
struct AccessGroupOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::ArrayAttr (*getAccessGroupsOrNull)(const Concept *impl, ::mlir::Operation *);
    void (*setAccessGroups)(const Concept *impl, ::mlir::Operation *, const ::mlir::ArrayAttr);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::AccessGroupOpInterface;
    Model() : Concept{getAccessGroupsOrNull, setAccessGroups} {}

    static inline ::mlir::ArrayAttr getAccessGroupsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAccessGroups(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::AccessGroupOpInterface;
    FallbackModel() : Concept{getAccessGroupsOrNull, setAccessGroups} {}

    static inline ::mlir::ArrayAttr getAccessGroupsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAccessGroups(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::ArrayAttr getAccessGroupsOrNull(::mlir::Operation *tablegen_opaque_val) const;
    void setAccessGroups(::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) const;
  };
};
template <typename ConcreteOp>
struct AccessGroupOpInterfaceTrait;

} // namespace detail
class AccessGroupOpInterface : public ::mlir::OpInterface<AccessGroupOpInterface, detail::AccessGroupOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AccessGroupOpInterface, detail::AccessGroupOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AccessGroupOpInterfaceTrait<ConcreteOp> {};
  /// Returns the access groups attribute or nullptr
  ::mlir::ArrayAttr getAccessGroupsOrNull();
  /// Sets the access groups attribute
  void setAccessGroups(const ::mlir::ArrayAttr attr);
};
namespace detail {
  template <typename ConcreteOp>
  struct AccessGroupOpInterfaceTrait : public ::mlir::OpInterface<AccessGroupOpInterface, detail::AccessGroupOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the access groups attribute or nullptr
    ::mlir::ArrayAttr getAccessGroupsOrNull() {
      auto op = cast<ConcreteOp>(this->getOperation());
        return op.getAccessGroupsAttr();
    }
    /// Sets the access groups attribute
    void setAccessGroups(const ::mlir::ArrayAttr attr) {
      auto op = cast<ConcreteOp>(this->getOperation());
        op.setAccessGroupsAttr(attr);
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyAccessGroupOpInterface(op);
    }
  };
}// namespace detail
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
class AliasAnalysisOpInterface;
namespace detail {
struct AliasAnalysisOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::ArrayAttr (*getAliasScopesOrNull)(const Concept *impl, ::mlir::Operation *);
    void (*setAliasScopes)(const Concept *impl, ::mlir::Operation *, const ::mlir::ArrayAttr);
    ::mlir::ArrayAttr (*getNoAliasScopesOrNull)(const Concept *impl, ::mlir::Operation *);
    void (*setNoAliasScopes)(const Concept *impl, ::mlir::Operation *, const ::mlir::ArrayAttr);
    ::mlir::ArrayAttr (*getTBAATagsOrNull)(const Concept *impl, ::mlir::Operation *);
    void (*setTBAATags)(const Concept *impl, ::mlir::Operation *, const ::mlir::ArrayAttr);
    ::llvm::SmallVector<::mlir::Value> (*getAccessedOperands)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::AliasAnalysisOpInterface;
    Model() : Concept{getAliasScopesOrNull, setAliasScopes, getNoAliasScopesOrNull, setNoAliasScopes, getTBAATagsOrNull, setTBAATags, getAccessedOperands} {}

    static inline ::mlir::ArrayAttr getAliasScopesOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAliasScopes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr);
    static inline ::mlir::ArrayAttr getNoAliasScopesOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setNoAliasScopes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr);
    static inline ::mlir::ArrayAttr getTBAATagsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setTBAATags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr);
    static inline ::llvm::SmallVector<::mlir::Value> getAccessedOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::AliasAnalysisOpInterface;
    FallbackModel() : Concept{getAliasScopesOrNull, setAliasScopes, getNoAliasScopesOrNull, setNoAliasScopes, getTBAATagsOrNull, setTBAATags, getAccessedOperands} {}

    static inline ::mlir::ArrayAttr getAliasScopesOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAliasScopes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr);
    static inline ::mlir::ArrayAttr getNoAliasScopesOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setNoAliasScopes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr);
    static inline ::mlir::ArrayAttr getTBAATagsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setTBAATags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr);
    static inline ::llvm::SmallVector<::mlir::Value> getAccessedOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::ArrayAttr getAliasScopesOrNull(::mlir::Operation *tablegen_opaque_val) const;
    void setAliasScopes(::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) const;
    ::mlir::ArrayAttr getNoAliasScopesOrNull(::mlir::Operation *tablegen_opaque_val) const;
    void setNoAliasScopes(::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) const;
    ::mlir::ArrayAttr getTBAATagsOrNull(::mlir::Operation *tablegen_opaque_val) const;
    void setTBAATags(::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) const;
  };
};
template <typename ConcreteOp>
struct AliasAnalysisOpInterfaceTrait;

} // namespace detail
class AliasAnalysisOpInterface : public ::mlir::OpInterface<AliasAnalysisOpInterface, detail::AliasAnalysisOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AliasAnalysisOpInterface, detail::AliasAnalysisOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AliasAnalysisOpInterfaceTrait<ConcreteOp> {};
  /// Returns the alias scopes attribute or nullptr
  ::mlir::ArrayAttr getAliasScopesOrNull();
  /// Sets the alias scopes attribute
  void setAliasScopes(const ::mlir::ArrayAttr attr);
  /// Returns the noalias scopes attribute or nullptr
  ::mlir::ArrayAttr getNoAliasScopesOrNull();
  /// Sets the noalias scopes attribute
  void setNoAliasScopes(const ::mlir::ArrayAttr attr);
  /// Returns the tbaa attribute or nullptr
  ::mlir::ArrayAttr getTBAATagsOrNull();
  /// Sets the tbaa attribute
  void setTBAATags(const ::mlir::ArrayAttr attr);
  /// Returns a list of all pointer operands accessed by the operation
  ::llvm::SmallVector<::mlir::Value> getAccessedOperands();
};
namespace detail {
  template <typename ConcreteOp>
  struct AliasAnalysisOpInterfaceTrait : public ::mlir::OpInterface<AliasAnalysisOpInterface, detail::AliasAnalysisOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the alias scopes attribute or nullptr
    ::mlir::ArrayAttr getAliasScopesOrNull() {
      auto op = cast<ConcreteOp>(this->getOperation());
        return op.getAliasScopesAttr();
    }
    /// Sets the alias scopes attribute
    void setAliasScopes(const ::mlir::ArrayAttr attr) {
      auto op = cast<ConcreteOp>(this->getOperation());
        op.setAliasScopesAttr(attr);
    }
    /// Returns the noalias scopes attribute or nullptr
    ::mlir::ArrayAttr getNoAliasScopesOrNull() {
      auto op = cast<ConcreteOp>(this->getOperation());
        return op.getNoaliasScopesAttr();
    }
    /// Sets the noalias scopes attribute
    void setNoAliasScopes(const ::mlir::ArrayAttr attr) {
      auto op = cast<ConcreteOp>(this->getOperation());
        op.setNoaliasScopesAttr(attr);
    }
    /// Returns the tbaa attribute or nullptr
    ::mlir::ArrayAttr getTBAATagsOrNull() {
      auto op = cast<ConcreteOp>(this->getOperation());
        return op.getTbaaAttr();
    }
    /// Sets the tbaa attribute
    void setTBAATags(const ::mlir::ArrayAttr attr) {
      auto op = cast<ConcreteOp>(this->getOperation());
        op.setTbaaAttr(attr);
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyAliasAnalysisOpInterface(op);
    }
  };
}// namespace detail
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
class FPExceptionBehaviorOpInterface;
namespace detail {
struct FPExceptionBehaviorOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::LLVM::FPExceptionBehaviorAttr (*getFPExceptionBehaviorAttr)(const Concept *impl, ::mlir::Operation *);
    ::llvm::StringRef (*getFPExceptionBehaviorAttrName)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::FPExceptionBehaviorOpInterface;
    Model() : Concept{getFPExceptionBehaviorAttr, getFPExceptionBehaviorAttrName} {}

    static inline ::mlir::LLVM::FPExceptionBehaviorAttr getFPExceptionBehaviorAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::StringRef getFPExceptionBehaviorAttrName();
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::FPExceptionBehaviorOpInterface;
    FallbackModel() : Concept{getFPExceptionBehaviorAttr, getFPExceptionBehaviorAttrName} {}

    static inline ::mlir::LLVM::FPExceptionBehaviorAttr getFPExceptionBehaviorAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::StringRef getFPExceptionBehaviorAttrName();
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::LLVM::FPExceptionBehaviorAttr getFPExceptionBehaviorAttr(::mlir::Operation *tablegen_opaque_val) const;
    static ::llvm::StringRef getFPExceptionBehaviorAttrName();
  };
};
template <typename ConcreteOp>
struct FPExceptionBehaviorOpInterfaceTrait;

} // namespace detail
class FPExceptionBehaviorOpInterface : public ::mlir::OpInterface<FPExceptionBehaviorOpInterface, detail::FPExceptionBehaviorOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FPExceptionBehaviorOpInterface, detail::FPExceptionBehaviorOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FPExceptionBehaviorOpInterfaceTrait<ConcreteOp> {};
  /// Returns a FPExceptionBehavior attribute for the operation
  ::mlir::LLVM::FPExceptionBehaviorAttr getFPExceptionBehaviorAttr();
  /// Returns the name of the FPExceptionBehaviorAttr
  ///                         attribute for the operation
  ::llvm::StringRef getFPExceptionBehaviorAttrName();
};
namespace detail {
  template <typename ConcreteOp>
  struct FPExceptionBehaviorOpInterfaceTrait : public ::mlir::OpInterface<FPExceptionBehaviorOpInterface, detail::FPExceptionBehaviorOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a FPExceptionBehavior attribute for the operation
    ::mlir::LLVM::FPExceptionBehaviorAttr getFPExceptionBehaviorAttr() {
      auto op = cast<ConcreteOp>(this->getOperation());
        return op.getFpExceptionBehaviorAttr();
    }
    /// Returns the name of the FPExceptionBehaviorAttr
    ///                         attribute for the operation
    static ::llvm::StringRef getFPExceptionBehaviorAttrName() {
      return "fpExceptionBehavior";
    }
  };
}// namespace detail
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
class RoundingModeOpInterface;
namespace detail {
struct RoundingModeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::LLVM::RoundingModeAttr (*getRoundingModeAttr)(const Concept *impl, ::mlir::Operation *);
    ::llvm::StringRef (*getRoundingModeAttrName)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::RoundingModeOpInterface;
    Model() : Concept{getRoundingModeAttr, getRoundingModeAttrName} {}

    static inline ::mlir::LLVM::RoundingModeAttr getRoundingModeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::StringRef getRoundingModeAttrName();
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::RoundingModeOpInterface;
    FallbackModel() : Concept{getRoundingModeAttr, getRoundingModeAttrName} {}

    static inline ::mlir::LLVM::RoundingModeAttr getRoundingModeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::StringRef getRoundingModeAttrName();
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::LLVM::RoundingModeAttr getRoundingModeAttr(::mlir::Operation *tablegen_opaque_val) const;
    static ::llvm::StringRef getRoundingModeAttrName();
  };
};
template <typename ConcreteOp>
struct RoundingModeOpInterfaceTrait;

} // namespace detail
class RoundingModeOpInterface : public ::mlir::OpInterface<RoundingModeOpInterface, detail::RoundingModeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RoundingModeOpInterface, detail::RoundingModeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RoundingModeOpInterfaceTrait<ConcreteOp> {};
  /// Returns a RoundingMode attribute for the operation
  ::mlir::LLVM::RoundingModeAttr getRoundingModeAttr();
  /// Returns the name of the RoundingModeAttr attribute
  ///                          for the operation
  ::llvm::StringRef getRoundingModeAttrName();
};
namespace detail {
  template <typename ConcreteOp>
  struct RoundingModeOpInterfaceTrait : public ::mlir::OpInterface<RoundingModeOpInterface, detail::RoundingModeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a RoundingMode attribute for the operation
    ::mlir::LLVM::RoundingModeAttr getRoundingModeAttr() {
      auto op = cast<ConcreteOp>(this->getOperation());
        return op.getRoundingmodeAttr();
    }
    /// Returns the name of the RoundingModeAttr attribute
    ///                          for the operation
    static ::llvm::StringRef getRoundingModeAttrName() {
      return "roundingmode";
    }
  };
}// namespace detail
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
template<typename ConcreteOp>
::mlir::LLVM::FastmathFlagsAttr detail::FastmathFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::getFastmathAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFastmathAttr();
}
template<typename ConcreteOp>
::llvm::StringRef detail::FastmathFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::getFastmathAttrName() {
  return ConcreteOp::getFastmathAttrName();
}
template<typename ConcreteOp>
::mlir::LLVM::FastmathFlagsAttr detail::FastmathFlagsInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFastmathAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFastmathAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::StringRef detail::FastmathFlagsInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFastmathAttrName() {
  return ConcreteOp::getFastmathAttrName();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LLVM::FastmathFlagsAttr detail::FastmathFlagsInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFastmathAttr(::mlir::Operation *tablegen_opaque_val) const {
auto op = cast<ConcreteOp>(this->getOperation());
        return op.getFastmathFlagsAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::StringRef detail::FastmathFlagsInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFastmathAttrName() {
return "fastmathFlags";
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
template<typename ConcreteOp>
IntegerOverflowFlags detail::IntegerOverflowFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::getOverflowFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOverflowFlags();
}
template<typename ConcreteOp>
void detail::IntegerOverflowFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::setOverflowFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, IntegerOverflowFlags flags) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setOverflowFlags(flags);
}
template<typename ConcreteOp>
bool detail::IntegerOverflowFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::hasNoUnsignedWrap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasNoUnsignedWrap();
}
template<typename ConcreteOp>
bool detail::IntegerOverflowFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::hasNoSignedWrap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasNoSignedWrap();
}
template<typename ConcreteOp>
StringRef detail::IntegerOverflowFlagsInterfaceInterfaceTraits::Model<ConcreteOp>::getOverflowFlagsAttrName() {
  return ConcreteOp::getOverflowFlagsAttrName();
}
template<typename ConcreteOp>
IntegerOverflowFlags detail::IntegerOverflowFlagsInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOverflowFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOverflowFlags(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::IntegerOverflowFlagsInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setOverflowFlags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, IntegerOverflowFlags flags) {
  return static_cast<const ConcreteOp *>(impl)->setOverflowFlags(tablegen_opaque_val, flags);
}
template<typename ConcreteOp>
bool detail::IntegerOverflowFlagsInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasNoUnsignedWrap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasNoUnsignedWrap(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::IntegerOverflowFlagsInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasNoSignedWrap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasNoSignedWrap(tablegen_opaque_val);
}
template<typename ConcreteOp>
StringRef detail::IntegerOverflowFlagsInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOverflowFlagsAttrName() {
  return ConcreteOp::getOverflowFlagsAttrName();
}
template<typename ConcreteModel, typename ConcreteOp>
IntegerOverflowFlags detail::IntegerOverflowFlagsInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOverflowFlags(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getProperties().overflowFlags;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::IntegerOverflowFlagsInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setOverflowFlags(::mlir::Operation *tablegen_opaque_val, IntegerOverflowFlags flags) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getProperties().overflowFlags = flags;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::IntegerOverflowFlagsInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasNoUnsignedWrap(::mlir::Operation *tablegen_opaque_val) const {
return bitEnumContainsAll((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOverflowFlags(),
                                IntegerOverflowFlags::nuw);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::IntegerOverflowFlagsInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasNoSignedWrap(::mlir::Operation *tablegen_opaque_val) const {
return bitEnumContainsAll((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOverflowFlags(),
                                IntegerOverflowFlags::nsw);
}
template<typename ConcreteModel, typename ConcreteOp>
StringRef detail::IntegerOverflowFlagsInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOverflowFlagsAttrName() {
return "overflowFlags";
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
template<typename ConcreteOp>
::mlir::DenseI32ArrayAttr detail::BranchWeightOpInterfaceInterfaceTraits::Model<ConcreteOp>::getBranchWeightsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBranchWeightsOrNull();
}
template<typename ConcreteOp>
void detail::BranchWeightOpInterfaceInterfaceTraits::Model<ConcreteOp>::setBranchWeights(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::DenseI32ArrayAttr attr) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setBranchWeights(attr);
}
template<typename ConcreteOp>
::mlir::DenseI32ArrayAttr detail::BranchWeightOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getBranchWeightsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBranchWeightsOrNull(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::BranchWeightOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setBranchWeights(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::DenseI32ArrayAttr attr) {
  return static_cast<const ConcreteOp *>(impl)->setBranchWeights(tablegen_opaque_val, attr);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::DenseI32ArrayAttr detail::BranchWeightOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getBranchWeightsOrNull(::mlir::Operation *tablegen_opaque_val) const {
auto op = cast<ConcreteOp>(this->getOperation());
        return op.getBranchWeightsAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::BranchWeightOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setBranchWeights(::mlir::Operation *tablegen_opaque_val, ::mlir::DenseI32ArrayAttr attr) const {
auto op = cast<ConcreteOp>(this->getOperation());
        op.setBranchWeightsAttr(attr);
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
template<typename ConcreteOp>
::mlir::ArrayAttr detail::AccessGroupOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccessGroupsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccessGroupsOrNull();
}
template<typename ConcreteOp>
void detail::AccessGroupOpInterfaceInterfaceTraits::Model<ConcreteOp>::setAccessGroups(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setAccessGroups(attr);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::AccessGroupOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccessGroupsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccessGroupsOrNull(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::AccessGroupOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setAccessGroups(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) {
  return static_cast<const ConcreteOp *>(impl)->setAccessGroups(tablegen_opaque_val, attr);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::AccessGroupOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAccessGroupsOrNull(::mlir::Operation *tablegen_opaque_val) const {
auto op = cast<ConcreteOp>(this->getOperation());
        return op.getAccessGroupsAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::AccessGroupOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setAccessGroups(::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) const {
auto op = cast<ConcreteOp>(this->getOperation());
        op.setAccessGroupsAttr(attr);
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
template<typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAliasScopesOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAliasScopesOrNull();
}
template<typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::Model<ConcreteOp>::setAliasScopes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setAliasScopes(attr);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNoAliasScopesOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNoAliasScopesOrNull();
}
template<typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::Model<ConcreteOp>::setNoAliasScopes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setNoAliasScopes(attr);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTBAATagsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTBAATagsOrNull();
}
template<typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::Model<ConcreteOp>::setTBAATags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setTBAATags(attr);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::AliasAnalysisOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccessedOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccessedOperands();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAliasScopesOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAliasScopesOrNull(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setAliasScopes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) {
  return static_cast<const ConcreteOp *>(impl)->setAliasScopes(tablegen_opaque_val, attr);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNoAliasScopesOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNoAliasScopesOrNull(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setNoAliasScopes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) {
  return static_cast<const ConcreteOp *>(impl)->setNoAliasScopes(tablegen_opaque_val, attr);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTBAATagsOrNull(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getTBAATagsOrNull(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setTBAATags(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) {
  return static_cast<const ConcreteOp *>(impl)->setTBAATags(tablegen_opaque_val, attr);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::AliasAnalysisOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccessedOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccessedOperands(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAliasScopesOrNull(::mlir::Operation *tablegen_opaque_val) const {
auto op = cast<ConcreteOp>(this->getOperation());
        return op.getAliasScopesAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setAliasScopes(::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) const {
auto op = cast<ConcreteOp>(this->getOperation());
        op.setAliasScopesAttr(attr);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNoAliasScopesOrNull(::mlir::Operation *tablegen_opaque_val) const {
auto op = cast<ConcreteOp>(this->getOperation());
        return op.getNoaliasScopesAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setNoAliasScopes(::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) const {
auto op = cast<ConcreteOp>(this->getOperation());
        op.setNoaliasScopesAttr(attr);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::AliasAnalysisOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTBAATagsOrNull(::mlir::Operation *tablegen_opaque_val) const {
auto op = cast<ConcreteOp>(this->getOperation());
        return op.getTbaaAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::AliasAnalysisOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setTBAATags(::mlir::Operation *tablegen_opaque_val, const ::mlir::ArrayAttr attr) const {
auto op = cast<ConcreteOp>(this->getOperation());
        op.setTbaaAttr(attr);
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
template<typename ConcreteOp>
::mlir::LLVM::FPExceptionBehaviorAttr detail::FPExceptionBehaviorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getFPExceptionBehaviorAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFPExceptionBehaviorAttr();
}
template<typename ConcreteOp>
::llvm::StringRef detail::FPExceptionBehaviorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getFPExceptionBehaviorAttrName() {
  return ConcreteOp::getFPExceptionBehaviorAttrName();
}
template<typename ConcreteOp>
::mlir::LLVM::FPExceptionBehaviorAttr detail::FPExceptionBehaviorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFPExceptionBehaviorAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFPExceptionBehaviorAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::StringRef detail::FPExceptionBehaviorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFPExceptionBehaviorAttrName() {
  return ConcreteOp::getFPExceptionBehaviorAttrName();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LLVM::FPExceptionBehaviorAttr detail::FPExceptionBehaviorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFPExceptionBehaviorAttr(::mlir::Operation *tablegen_opaque_val) const {
auto op = cast<ConcreteOp>(this->getOperation());
        return op.getFpExceptionBehaviorAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::StringRef detail::FPExceptionBehaviorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFPExceptionBehaviorAttrName() {
return "fpExceptionBehavior";
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
template<typename ConcreteOp>
::mlir::LLVM::RoundingModeAttr detail::RoundingModeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getRoundingModeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRoundingModeAttr();
}
template<typename ConcreteOp>
::llvm::StringRef detail::RoundingModeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getRoundingModeAttrName() {
  return ConcreteOp::getRoundingModeAttrName();
}
template<typename ConcreteOp>
::mlir::LLVM::RoundingModeAttr detail::RoundingModeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getRoundingModeAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRoundingModeAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::StringRef detail::RoundingModeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getRoundingModeAttrName() {
  return ConcreteOp::getRoundingModeAttrName();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LLVM::RoundingModeAttr detail::RoundingModeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRoundingModeAttr(::mlir::Operation *tablegen_opaque_val) const {
auto op = cast<ConcreteOp>(this->getOperation());
        return op.getRoundingmodeAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::StringRef detail::RoundingModeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRoundingModeAttrName() {
return "roundingmode";
}
} // namespace LLVM
} // namespace mlir
