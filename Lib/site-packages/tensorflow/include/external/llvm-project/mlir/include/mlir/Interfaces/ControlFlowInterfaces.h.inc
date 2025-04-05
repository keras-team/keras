/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class BranchOpInterface;
namespace detail {
struct BranchOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::SuccessorOperands (*getSuccessorOperands)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::std::optional<::mlir::BlockArgument> (*getSuccessorBlockArgument)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::mlir::Block *(*getSuccessorForOperands)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::Attribute>);
    bool (*areTypesCompatible)(const Concept *impl, ::mlir::Operation *, ::mlir::Type, ::mlir::Type);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::BranchOpInterface;
    Model() : Concept{getSuccessorOperands, getSuccessorBlockArgument, getSuccessorForOperands, areTypesCompatible} {}

    static inline ::mlir::SuccessorOperands getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline ::std::optional<::mlir::BlockArgument> getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex);
    static inline ::mlir::Block *getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::BranchOpInterface;
    FallbackModel() : Concept{getSuccessorOperands, getSuccessorBlockArgument, getSuccessorForOperands, areTypesCompatible} {}

    static inline ::mlir::SuccessorOperands getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index);
    static inline ::std::optional<::mlir::BlockArgument> getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex);
    static inline ::mlir::Block *getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Block *getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands) const;
    bool areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const;
  };
};
template <typename ConcreteOp>
struct BranchOpInterfaceTrait;

} // namespace detail
class BranchOpInterface : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BranchOpInterfaceTrait<ConcreteOp> {};
  /// Returns the operands that correspond to the arguments of the successor
  /// at the given index. It consists of a number of operands that are
  /// internally produced by the operation, followed by a range of operands
  /// that are forwarded. An example operation making use of produced
  /// operands would be:
  /// 
  /// ```mlir
  /// invoke %function(%0)
  ///     label ^success ^error(%1 : i32)
  /// 
  /// ^error(%e: !error, %arg0: i32):
  ///     ...
  /// ```
  /// 
  /// The operand that would map to the `^error`s `%e` operand is produced
  /// by the `invoke` operation, while `%1` is a forwarded operand that maps
  /// to `%arg0` in the successor.
  /// 
  /// Produced operands always map to the first few block arguments of the
  /// successor, followed by the forwarded operands. Mapping them in any
  /// other order is not supported by the interface.
  /// 
  /// By having the forwarded operands last allows users of the interface
  /// to append more forwarded operands to the branch operation without
  /// interfering with other successor operands.
  ::mlir::SuccessorOperands getSuccessorOperands(unsigned index);
  /// Returns the `BlockArgument` corresponding to operand `operandIndex` in
  /// some successor, or std::nullopt if `operandIndex` isn't a successor operand
  /// index.
  ::std::optional<::mlir::BlockArgument> getSuccessorBlockArgument(unsigned operandIndex);
  /// Returns the successor that would be chosen with the given constant
  /// operands. Returns nullptr if a single successor could not be chosen.
  ::mlir::Block *getSuccessorForOperands(::llvm::ArrayRef<::mlir::Attribute> operands);
  /// This method is called to compare types along control-flow edges. By
  /// default, the types are checked as equal.
  bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs);
};
namespace detail {
  template <typename ConcreteOp>
  struct BranchOpInterfaceTrait : public ::mlir::OpInterface<BranchOpInterface, detail::BranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the successor that would be chosen with the given constant
    /// operands. Returns nullptr if a single successor could not be chosen.
    ::mlir::Block *getSuccessorForOperands(::llvm::ArrayRef<::mlir::Attribute> operands) {
      return nullptr;
    }
    /// This method is called to compare types along control-flow edges. By
    /// default, the types are checked as equal.
    bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return lhs == rhs;
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      auto concreteOp = ::mlir::cast<ConcreteOp>(op);
    for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
      ::mlir::SuccessorOperands operands = concreteOp.getSuccessorOperands(i);
      if (::mlir::failed(::mlir::detail::verifyBranchSuccessorOperands(op, i, operands)))
        return ::mlir::failure();
    }
    return ::mlir::success();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class RegionBranchOpInterface;
namespace detail {
struct RegionBranchOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::OperandRange (*getEntrySuccessorOperands)(const Concept *impl, ::mlir::Operation *, ::mlir::RegionBranchPoint);
    void (*getEntrySuccessorRegions)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &);
    void (*getSuccessorRegions)(const Concept *impl, ::mlir::Operation *, ::mlir::RegionBranchPoint, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &);
    void (*getRegionInvocationBounds)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &);
    bool (*areTypesCompatible)(const Concept *impl, ::mlir::Operation *, ::mlir::Type, ::mlir::Type);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::RegionBranchOpInterface;
    Model() : Concept{getEntrySuccessorOperands, getEntrySuccessorRegions, getSuccessorRegions, getRegionInvocationBounds, areTypesCompatible} {}

    static inline ::mlir::OperandRange getEntrySuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point);
    static inline void getEntrySuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
    static inline void getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
    static inline void getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::RegionBranchOpInterface;
    FallbackModel() : Concept{getEntrySuccessorOperands, getEntrySuccessorRegions, getSuccessorRegions, getRegionInvocationBounds, areTypesCompatible} {}

    static inline ::mlir::OperandRange getEntrySuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point);
    static inline void getEntrySuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
    static inline void getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
    static inline void getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
    static inline bool areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::OperandRange getEntrySuccessorOperands(::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point) const;
    void getEntrySuccessorRegions(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) const;
    void getRegionInvocationBounds(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds) const;
    bool areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const;
  };
};
template <typename ConcreteOp>
struct RegionBranchOpInterfaceTrait;

} // namespace detail
class RegionBranchOpInterface : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RegionBranchOpInterfaceTrait<ConcreteOp> {};
  /// Returns the operands of this operation that are forwarded to the region
  /// successor's block arguments or this operation's results when branching
  /// to `point`. `point` is guaranteed to be among the successors that are
  /// returned by `getEntrySuccessorRegions`/`getSuccessorRegions(parent())`.
  /// 
  /// Example: In the above example, this method returns the operand %b of the
  /// `scf.for` op, regardless of the value of `point`. I.e., this op always
  /// forwards the same operands, regardless of whether the loop has 0 or more
  /// iterations.
  ::mlir::OperandRange getEntrySuccessorOperands(::mlir::RegionBranchPoint point);
  /// Returns the potential region successors when first executing the op.
  /// 
  /// Unlike `getSuccessorRegions`, this method also passes along the
  /// constant operands of this op. Based on these, the implementation may
  /// filter out certain successors. By default, simply dispatches to
  /// `getSuccessorRegions`. `operands` contains an entry for every
  /// operand of this op, with a null attribute if the operand has no constant
  /// value.
  /// 
  /// Note: The control flow does not necessarily have to enter any region of
  /// this op.
  /// 
  /// Example: In the above example, this method may return two region
  /// region successors: the single region of the `scf.for` op and the
  /// `scf.for` operation (that implements this interface). If %lb, %ub, %step
  /// are constants and it can be determined the loop does not have any
  /// iterations, this method may choose to return only this operation.
  /// Similarly, if it can be determined that the loop has at least one
  /// iteration, this method may choose to return only the region of the loop.
  void getEntrySuccessorRegions(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
  /// Returns the potential region successors when branching from `point`.
  /// These are the regions that may be selected during the flow of control.
  /// 
  /// When `point = RegionBranchPoint::parent()`, this method returns the
  /// region successors when entering the operation. Otherwise, this method
  /// returns the successor regions when branching from the region indicated
  /// by `point`.
  /// 
  /// Example: In the above example, this method returns the region of the
  /// `scf.for` and this operation for either region branch point (`parent`
  /// and the region of the `scf.for`). An implementation may choose to filter
  /// out region successors when it is statically known (e.g., by examining
  /// the operands of this op) that those successors are not branched to.
  void getSuccessorRegions(::mlir::RegionBranchPoint point, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
  /// Populates `invocationBounds` with the minimum and maximum number of
  /// times this operation will invoke the attached regions (assuming the
  /// regions yield normally, i.e. do not abort or invoke an infinite loop).
  /// The minimum number of invocations is at least 0. If the maximum number
  /// of invocations cannot be statically determined, then it will be set to
  /// `InvocationBounds::getUnknown()`.
  /// 
  /// This method also passes along the constant operands of this op.
  /// `operands` contains an entry for every operand of this op, with a null
  /// attribute if the operand has no constant value.
  /// 
  /// This method may be called speculatively on operations where the provided
  /// operands are not necessarily the same as the operation's current
  /// operands. This may occur in analyses that wish to determine "what would
  /// be the region invocations if these were the operands?"
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds);
  /// This method is called to compare types along control-flow edges. By
  /// default, the types are checked as equal.
  bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs);

    /// Return `true` if control flow originating from the given region may
    /// eventually branch back to the same region. (Maybe after passing through
    /// other regions.)
    bool isRepetitiveRegion(unsigned index);

    /// Return `true` if there is a loop in the region branching graph. Only
    /// reachable regions (starting from the entry regions) are considered.
    bool hasLoop();
};
namespace detail {
  template <typename ConcreteOp>
  struct RegionBranchOpInterfaceTrait : public ::mlir::OpInterface<RegionBranchOpInterface, detail::RegionBranchOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the operands of this operation that are forwarded to the region
    /// successor's block arguments or this operation's results when branching
    /// to `point`. `point` is guaranteed to be among the successors that are
    /// returned by `getEntrySuccessorRegions`/`getSuccessorRegions(parent())`.
    /// 
    /// Example: In the above example, this method returns the operand %b of the
    /// `scf.for` op, regardless of the value of `point`. I.e., this op always
    /// forwards the same operands, regardless of whether the loop has 0 or more
    /// iterations.
    ::mlir::OperandRange getEntrySuccessorOperands(::mlir::RegionBranchPoint point) {
      auto operandEnd = this->getOperation()->operand_end();
        return ::mlir::OperandRange(operandEnd, operandEnd);
    }
    /// Returns the potential region successors when first executing the op.
    /// 
    /// Unlike `getSuccessorRegions`, this method also passes along the
    /// constant operands of this op. Based on these, the implementation may
    /// filter out certain successors. By default, simply dispatches to
    /// `getSuccessorRegions`. `operands` contains an entry for every
    /// operand of this op, with a null attribute if the operand has no constant
    /// value.
    /// 
    /// Note: The control flow does not necessarily have to enter any region of
    /// this op.
    /// 
    /// Example: In the above example, this method may return two region
    /// region successors: the single region of the `scf.for` op and the
    /// `scf.for` operation (that implements this interface). If %lb, %ub, %step
    /// are constants and it can be determined the loop does not have any
    /// iterations, this method may choose to return only this operation.
    /// Similarly, if it can be determined that the loop has at least one
    /// iteration, this method may choose to return only the region of the loop.
    void getEntrySuccessorRegions(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
      (*static_cast<ConcreteOp *>(this)).getSuccessorRegions(mlir::RegionBranchPoint::parent(), regions);
    }
    /// Populates `invocationBounds` with the minimum and maximum number of
    /// times this operation will invoke the attached regions (assuming the
    /// regions yield normally, i.e. do not abort or invoke an infinite loop).
    /// The minimum number of invocations is at least 0. If the maximum number
    /// of invocations cannot be statically determined, then it will be set to
    /// `InvocationBounds::getUnknown()`.
    /// 
    /// This method also passes along the constant operands of this op.
    /// `operands` contains an entry for every operand of this op, with a null
    /// attribute if the operand has no constant value.
    /// 
    /// This method may be called speculatively on operations where the provided
    /// operands are not necessarily the same as the operation's current
    /// operands. This may occur in analyses that wish to determine "what would
    /// be the region invocations if these were the operands?"
    void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
      invocationBounds.append((*static_cast<ConcreteOp *>(this))->getNumRegions(),
                                ::mlir::InvocationBounds::getUnknown());
    }
    /// This method is called to compare types along control-flow edges. By
    /// default, the types are checked as equal.
    bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return lhs == rhs;
    }
    static ::llvm::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      static_assert(!ConcreteOp::template hasTrait<OpTrait::ZeroRegions>(),
                  "expected operation to have non-zero regions");
    return detail::verifyTypesAlongControlFlowEdges(op);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class RegionBranchTerminatorOpInterface;
namespace detail {
struct RegionBranchTerminatorOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::MutableOperandRange (*getMutableSuccessorOperands)(const Concept *impl, ::mlir::Operation *, ::mlir::RegionBranchPoint);
    void (*getSuccessorRegions)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::RegionBranchTerminatorOpInterface;
    Model() : Concept{getMutableSuccessorOperands, getSuccessorRegions} {}

    static inline ::mlir::MutableOperandRange getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point);
    static inline void getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::RegionBranchTerminatorOpInterface;
    FallbackModel() : Concept{getMutableSuccessorOperands, getSuccessorRegions} {}

    static inline ::mlir::MutableOperandRange getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point);
    static inline void getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    void getSuccessorRegions(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) const;
  };
};
template <typename ConcreteOp>
struct RegionBranchTerminatorOpInterfaceTrait;

} // namespace detail
class RegionBranchTerminatorOpInterface : public ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RegionBranchTerminatorOpInterfaceTrait<ConcreteOp> {};
  /// Returns a mutable range of operands that are semantically "returned" by
  /// passing them to the region successor indicated by `point`.
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::mlir::RegionBranchPoint point);
  /// Returns the potential region successors that are branched to after this
  /// terminator based on the given constant operands.
  /// 
  /// This method also passes along the constant operands of this op.
  /// `operands` contains an entry for every operand of this op, with a null
  /// attribute if the operand has no constant value.
  /// 
  /// The default implementation simply dispatches to the parent
  /// `RegionBranchOpInterface`'s `getSuccessorRegions` implementation.
  void getSuccessorRegions(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions);

    // Returns a range of operands that are semantically "returned" by passing
    // them to the region successor given by `index`.  If `index` is None, this
    // function returns the operands that are passed as a result to the parent
    // operation.
    ::mlir::OperandRange getSuccessorOperands(::mlir::RegionBranchPoint point) {
      return getMutableSuccessorOperands(point);
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct RegionBranchTerminatorOpInterfaceTrait : public ::mlir::OpInterface<RegionBranchTerminatorOpInterface, detail::RegionBranchTerminatorOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the potential region successors that are branched to after this
    /// terminator based on the given constant operands.
    /// 
    /// This method also passes along the constant operands of this op.
    /// `operands` contains an entry for every operand of this op, with a null
    /// attribute if the operand has no constant value.
    /// 
    /// The default implementation simply dispatches to the parent
    /// `RegionBranchOpInterface`'s `getSuccessorRegions` implementation.
    void getSuccessorRegions(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
      ::mlir::Operation *op = (*static_cast<ConcreteOp *>(this));
        ::llvm::cast<::mlir::RegionBranchOpInterface>(op->getParentOp())
          .getSuccessorRegions(op->getParentRegion(), regions);
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      static_assert(ConcreteOp::template hasTrait<OpTrait::IsTerminator>(),
                  "expected operation to be a terminator");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroResults>(),
                  "expected operation to have zero results");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroSuccessors>(),
                  "expected operation to have zero successors");
    return success();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class SelectLikeOpInterface;
namespace detail {
struct SelectLikeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getFalseValue)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getTrueValue)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getCondition)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SelectLikeOpInterface;
    Model() : Concept{getFalseValue, getTrueValue, getCondition} {}

    static inline ::mlir::Value getFalseValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getTrueValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getCondition(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SelectLikeOpInterface;
    FallbackModel() : Concept{getFalseValue, getTrueValue, getCondition} {}

    static inline ::mlir::Value getFalseValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getTrueValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getCondition(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct SelectLikeOpInterfaceTrait;

} // namespace detail
class SelectLikeOpInterface : public ::mlir::OpInterface<SelectLikeOpInterface, detail::SelectLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SelectLikeOpInterface, detail::SelectLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::SelectLikeOpInterfaceTrait<ConcreteOp> {};
  /// Returns the operand that would be chosen for a false condition.
  ::mlir::Value getFalseValue();
  /// Returns the operand that would be chosen for a true condition.
  ::mlir::Value getTrueValue();
  /// Returns the condition operand.
  ::mlir::Value getCondition();
};
namespace detail {
  template <typename ConcreteOp>
  struct SelectLikeOpInterfaceTrait : public ::mlir::OpInterface<SelectLikeOpInterface, detail::SelectLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::SuccessorOperands detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(index);
}
template<typename ConcreteOp>
::std::optional<::mlir::BlockArgument> detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) {
  ::mlir::Operation *opaqueOp = (llvm::cast<ConcreteOp>(tablegen_opaque_val));
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (::std::optional<::mlir::BlockArgument> arg = ::mlir::detail::getBranchSuccessorArgument(
                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return ::std::nullopt;
}
template<typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorForOperands(operands);
}
template<typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).areTypesCompatible(lhs, rhs);
}
template<typename ConcreteOp>
::mlir::SuccessorOperands detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorOperands(tablegen_opaque_val, index);
}
template<typename ConcreteOp>
::std::optional<::mlir::BlockArgument> detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned operandIndex) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorBlockArgument(tablegen_opaque_val, operandIndex);
}
template<typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorForOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorForOperands(tablegen_opaque_val, operands);
}
template<typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return static_cast<const ConcreteOp *>(impl)->areTypesCompatible(tablegen_opaque_val, lhs, rhs);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Block *detail::BranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorForOperands(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands) const {
return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::BranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const {
return lhs == rhs;
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getEntrySuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEntrySuccessorOperands(point);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getEntrySuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEntrySuccessorRegions(operands, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorRegions(point, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegionInvocationBounds(operands, invocationBounds);
}
template<typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::Model<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).areTypesCompatible(lhs, rhs);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getEntrySuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point) {
  return static_cast<const ConcreteOp *>(impl)->getEntrySuccessorOperands(tablegen_opaque_val, point);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getEntrySuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return static_cast<const ConcreteOp *>(impl)->getEntrySuccessorRegions(tablegen_opaque_val, operands, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorRegions(tablegen_opaque_val, point, regions);
}
template<typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getRegionInvocationBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
  return static_cast<const ConcreteOp *>(impl)->getRegionInvocationBounds(tablegen_opaque_val, operands, invocationBounds);
}
template<typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::areTypesCompatible(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) {
  return static_cast<const ConcreteOp *>(impl)->areTypesCompatible(tablegen_opaque_val, lhs, rhs);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OperandRange detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getEntrySuccessorOperands(::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point) const {
auto operandEnd = this->getOperation()->operand_end();
        return ::mlir::OperandRange(operandEnd, operandEnd);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getEntrySuccessorRegions(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) const {
(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorRegions(mlir::RegionBranchPoint::parent(), regions);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRegionInvocationBounds(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds) const {
invocationBounds.append((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumRegions(),
                                ::mlir::InvocationBounds::getUnknown());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::RegionBranchOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::areTypesCompatible(::mlir::Operation *tablegen_opaque_val, ::mlir::Type lhs, ::mlir::Type rhs) const {
return lhs == rhs;
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMutableSuccessorOperands(point);
}
template<typename ConcreteOp>
void detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSuccessorRegions(operands, regions);
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMutableSuccessorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RegionBranchPoint point) {
  return static_cast<const ConcreteOp *>(impl)->getMutableSuccessorOperands(tablegen_opaque_val, point);
}
template<typename ConcreteOp>
void detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSuccessorRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
  return static_cast<const ConcreteOp *>(impl)->getSuccessorRegions(tablegen_opaque_val, operands, regions);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::RegionBranchTerminatorOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSuccessorRegions(::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) const {
::mlir::Operation *op = (llvm::cast<ConcreteOp>(tablegen_opaque_val));
        ::llvm::cast<::mlir::RegionBranchOpInterface>(op->getParentOp())
          .getSuccessorRegions(op->getParentRegion(), regions);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::Value detail::SelectLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getFalseValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFalseValue();
}
template<typename ConcreteOp>
::mlir::Value detail::SelectLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTrueValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTrueValue();
}
template<typename ConcreteOp>
::mlir::Value detail::SelectLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getCondition(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCondition();
}
template<typename ConcreteOp>
::mlir::Value detail::SelectLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFalseValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFalseValue(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::SelectLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTrueValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getTrueValue(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::SelectLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getCondition(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCondition(tablegen_opaque_val);
}
} // namespace mlir
