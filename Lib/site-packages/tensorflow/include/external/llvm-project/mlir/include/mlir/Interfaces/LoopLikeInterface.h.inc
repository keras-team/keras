/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class LoopLikeOpInterface;
namespace detail {
struct LoopLikeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*isDefinedOutsideOfLoop)(const Concept *impl, ::mlir::Operation *, ::mlir::Value );
    ::llvm::SmallVector<::mlir::Region *> (*getLoopRegions)(const Concept *impl, ::mlir::Operation *);
    void (*moveOutOfLoop)(const Concept *impl, ::mlir::Operation *, ::mlir::Operation *);
    ::llvm::LogicalResult (*promoteIfSingleIteration)(const Concept *impl, ::mlir::Operation *, ::mlir::RewriterBase &);
    ::std::optional<::llvm::SmallVector<::mlir::Value>> (*getLoopInductionVars)(const Concept *impl, ::mlir::Operation *);
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> (*getLoopLowerBounds)(const Concept *impl, ::mlir::Operation *);
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> (*getLoopSteps)(const Concept *impl, ::mlir::Operation *);
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> (*getLoopUpperBounds)(const Concept *impl, ::mlir::Operation *);
    ::llvm::MutableArrayRef<::mlir::OpOperand> (*getInitsMutable)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Block::BlockArgListType (*getRegionIterArgs)(const Concept *impl, ::mlir::Operation *);
    std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> (*getYieldedValuesMutable)(const Concept *impl, ::mlir::Operation *);
    ::std::optional<::mlir::ResultRange> (*getLoopResults)(const Concept *impl, ::mlir::Operation *);
    ::mlir::FailureOr<::mlir::LoopLikeOpInterface> (*replaceWithAdditionalYields)(const Concept *impl, ::mlir::Operation *, ::mlir::RewriterBase &, ::mlir::ValueRange, bool, const ::mlir::NewYieldValuesFn &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LoopLikeOpInterface;
    Model() : Concept{isDefinedOutsideOfLoop, getLoopRegions, moveOutOfLoop, promoteIfSingleIteration, getLoopInductionVars, getLoopLowerBounds, getLoopSteps, getLoopUpperBounds, getInitsMutable, getRegionIterArgs, getYieldedValuesMutable, getLoopResults, replaceWithAdditionalYields} {}

    static inline bool isDefinedOutsideOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value  value);
    static inline ::llvm::SmallVector<::mlir::Region *> getLoopRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void moveOutOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Operation * op);
    static inline ::llvm::LogicalResult promoteIfSingleIteration(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase & rewriter);
    static inline ::std::optional<::llvm::SmallVector<::mlir::Value>> getLoopInductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopLowerBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopSteps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopUpperBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::MutableArrayRef<::mlir::OpOperand> getInitsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Block::BlockArgListType getRegionIterArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> getYieldedValuesMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::mlir::ResultRange> getLoopResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::FailureOr<::mlir::LoopLikeOpInterface> replaceWithAdditionalYields(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase & rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn & newYieldValuesFn);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LoopLikeOpInterface;
    FallbackModel() : Concept{isDefinedOutsideOfLoop, getLoopRegions, moveOutOfLoop, promoteIfSingleIteration, getLoopInductionVars, getLoopLowerBounds, getLoopSteps, getLoopUpperBounds, getInitsMutable, getRegionIterArgs, getYieldedValuesMutable, getLoopResults, replaceWithAdditionalYields} {}

    static inline bool isDefinedOutsideOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value  value);
    static inline ::llvm::SmallVector<::mlir::Region *> getLoopRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void moveOutOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Operation * op);
    static inline ::llvm::LogicalResult promoteIfSingleIteration(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase & rewriter);
    static inline ::std::optional<::llvm::SmallVector<::mlir::Value>> getLoopInductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopLowerBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopSteps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopUpperBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::MutableArrayRef<::mlir::OpOperand> getInitsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Block::BlockArgListType getRegionIterArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> getYieldedValuesMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::mlir::ResultRange> getLoopResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::FailureOr<::mlir::LoopLikeOpInterface> replaceWithAdditionalYields(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase & rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn & newYieldValuesFn);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool isDefinedOutsideOfLoop(::mlir::Operation *tablegen_opaque_val, ::mlir::Value value) const;
    void moveOutOfLoop(::mlir::Operation *tablegen_opaque_val, ::mlir::Operation *op) const;
    ::llvm::LogicalResult promoteIfSingleIteration(::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase &rewriter) const;
    ::std::optional<::llvm::SmallVector<::mlir::Value>> getLoopInductionVars(::mlir::Operation *tablegen_opaque_val) const;
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopLowerBounds(::mlir::Operation *tablegen_opaque_val) const;
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopSteps(::mlir::Operation *tablegen_opaque_val) const;
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopUpperBounds(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::MutableArrayRef<::mlir::OpOperand> getInitsMutable(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::Block::BlockArgListType getRegionIterArgs(::mlir::Operation *tablegen_opaque_val) const;
    std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> getYieldedValuesMutable(::mlir::Operation *tablegen_opaque_val) const;
    ::std::optional<::mlir::ResultRange> getLoopResults(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::FailureOr<::mlir::LoopLikeOpInterface> replaceWithAdditionalYields(::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase &rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn &newYieldValuesFn) const;
  };
};
template <typename ConcreteOp>
struct LoopLikeOpInterfaceTrait;

} // namespace detail
class LoopLikeOpInterface : public ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LoopLikeOpInterfaceTrait<ConcreteOp> {};
  /// Returns true if the given value is defined outside of the loop.
  /// A sensible implementation could be to check whether the value's defining
  /// operation lies outside of the loops body region. If the loop uses
  /// explicit capture of dependencies, an implementation could check whether
  /// the value corresponds to a captured dependency.
  bool isDefinedOutsideOfLoop(::mlir::Value  value);
  /// Returns the regions that make up the body of the loop and should be
  /// inspected for loop-invariant operations.
  ::llvm::SmallVector<::mlir::Region *> getLoopRegions();
  /// Moves the given loop-invariant operation out of the loop.
  void moveOutOfLoop(::mlir::Operation * op);
  /// Promotes the loop body to its containing block if the loop is known to
  /// have a single iteration. Returns "success" if the promotion was
  /// successful.
  ::llvm::LogicalResult promoteIfSingleIteration(::mlir::RewriterBase & rewriter);
  /// Return all induction variables, if they exist. If the op has no notion of
  /// induction variable, then return std::nullopt. If it does have
  /// a notion but an instance doesn't have induction variables, then
  /// return empty vector.
  ::std::optional<::llvm::SmallVector<::mlir::Value>> getLoopInductionVars();
  /// Return all lower bounds, if they exist. If the op has no notion of
  /// lower bounds, then return std::nullopt. If it does have
  /// a notion but an instance doesn't have lower bounds, then
  /// return empty vector.
  ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopLowerBounds();
  /// Return all steps, if they exist. If the op has no notion of
  /// steps, then return std::nullopt. If it does have
  /// a notion but an instance doesn't have steps, then
  /// return empty vector.
  ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopSteps();
  /// Return all upper bounds, if they exist. If the op has no notion of
  /// lower bounds, then return std::nullopt. If it does have
  /// a notion but an instance doesn't have lower bounds, then
  /// return empty vector.
  ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopUpperBounds();
  /// Return the mutable "init" operands that are used as initialization
  /// values for the region "iter_args" of this loop.
  ::llvm::MutableArrayRef<::mlir::OpOperand> getInitsMutable();
  /// Return the region "iter_args" (block arguments) that correspond to the
  /// "init" operands. If the op has multiple regions, return the
  /// corresponding block arguments of the entry region.
  ::mlir::Block::BlockArgListType getRegionIterArgs();
  /// Return the mutable operand range of values that are yielded to the next
  /// iteration by the loop terminator.
  /// 
  /// For loop operations that dont yield a value, this should return
  /// std::nullopt.
  std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> getYieldedValuesMutable();
  /// Return the range of results that are return from this loop and
  /// correspond to the "init" operands.
  /// 
  /// Note: This interface method is optional. If loop results are not
  /// exposed via this interface, "std::nullopt" should be returned.
  /// Otherwise, the number and types of results must match with the
  /// region iter_args, inits and yielded values that are exposed via this
  /// interface. If loop results are exposed but this loop op has no
  /// loop-carried variables, an empty result range (and not "std::nullopt")
  /// should be returned.
  ::std::optional<::mlir::ResultRange> getLoopResults();
  /// Append the specified additional "init" operands: replace this loop with
  /// a new loop that has the additional init operands. The loop body of
  /// this loop is moved over to the new loop.
  /// 
  /// `newInitOperands` specifies the additional "init" operands.
  /// `newYieldValuesFn` is a function that returns the yielded values (which
  /// can be computed based on the additional region iter_args). If
  /// `replaceInitOperandUsesInLoop` is set, all uses of the additional init
  /// operands inside of this loop are replaced with the corresponding, newly
  /// added region iter_args.
  /// 
  /// Note: Loops that do not support init/iter_args should return "failure".
  ::mlir::FailureOr<::mlir::LoopLikeOpInterface> replaceWithAdditionalYields(::mlir::RewriterBase & rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn & newYieldValuesFn);

    /// Returns if a block is inside a loop (within the current function). This
    /// can either be because the block is nested inside a LoopLikeInterface, or
    /// because the control flow graph is cyclic
    static bool blockIsInLoop(Block *block);

    /// If there is a single induction variable return it, otherwise return
    /// std::nullopt.
    ::std::optional<::mlir::Value> getSingleInductionVar() {
      auto inductionVars = (*this).getLoopInductionVars();
      if (inductionVars.has_value() && (*inductionVars).size() == 1)
          return (*inductionVars)[0];
        return std::nullopt;
    }
    /// Return the single lower bound value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleLowerBound() {
      auto lowerBounds = (*this).getLoopLowerBounds();
      if (lowerBounds.has_value() && (*lowerBounds).size() == 1)
          return (*lowerBounds)[0];
      return std::nullopt;
    }
    /// Return the single step value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleStep() {
      auto steps = (*this).getLoopSteps();
      if (steps.has_value() && (*steps).size() == 1)
          return (*steps)[0];
      return std::nullopt;
    }
    /// Return the single upper bound value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleUpperBound() {
      auto upperBounds = (*this).getLoopUpperBounds();
      if (upperBounds.has_value() && (*upperBounds).size() == 1)
          return (*upperBounds)[0];
      return std::nullopt;
    }

    /// Append the specified additional "init" operands: replace this loop with
    /// a new loop that has the additional init operands. The loop body of this
    /// loop is moved over to the new loop.
    ///
    /// The newly added region iter_args are yielded from the loop.
    ::mlir::FailureOr<::mlir::LoopLikeOpInterface>
        replaceWithAdditionalIterOperands(::mlir::RewriterBase &rewriter,
                                          ::mlir::ValueRange newInitOperands,
                                          bool replaceInitOperandUsesInLoop) {
      return (*this).replaceWithAdditionalYields(
          rewriter, newInitOperands, replaceInitOperandUsesInLoop,
          [](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBBArgs) {
            return SmallVector<Value>(newBBArgs);
          });
    }

    /// Return the values that are yielded to the next iteration. If
    /// the loop doesnt yield any values return `{}`.
    ::mlir::ValueRange getYieldedValues() {
      auto mutableValues = (*this).getYieldedValuesMutable();
      if (!mutableValues || mutableValues->empty())
        return {};
      Operation *yieldOp = mutableValues->begin()->getOwner();
      unsigned firstOperandIndex = mutableValues->begin()->getOperandNumber();
      return OperandRange(
          yieldOp->operand_begin() + firstOperandIndex,
          yieldOp->operand_begin() + firstOperandIndex + mutableValues->size());
    }

    /// Return the "init" operands that are used as initialization values for
    /// the region "iter_args" of this loop.
    ::mlir::OperandRange getInits() {
      auto initsMutable = (*this).getInitsMutable();
      if (initsMutable.empty())
        return ::mlir::OperandRange((*this)->operand_end(), (*this)->operand_end());
      unsigned firstOperandIndex = initsMutable.begin()->getOperandNumber();
      return OperandRange(
          (*this)->operand_begin() + firstOperandIndex,
          (*this)->operand_begin() + firstOperandIndex + initsMutable.size());
    }

    /// Return the region iter_arg that corresponds to the given init operand.
    /// Return an "empty" block argument if the given operand is not an init
    /// operand of this loop op.
    BlockArgument getTiedLoopRegionIterArg(OpOperand *opOperand) {
      auto initsMutable = (*this).getInitsMutable();
      auto it = llvm::find(initsMutable, *opOperand);
      if (it == initsMutable.end())
        return {};
      return (*this).getRegionIterArgs()[std::distance(initsMutable.begin(), it)];
    }

    /// Return the region iter_arg that corresponds to the given loop result.
    /// Return an "empty" block argument if the given OpResult is not a loop
    /// result or if this op does not expose any loop results.
    BlockArgument getTiedLoopRegionIterArg(OpResult opResult) {
      auto loopResults = (*this).getLoopResults();
      if (!loopResults)
        return {};
      auto it = llvm::find(*loopResults, opResult);
      if (it == loopResults->end())
        return {};
      return (*this).getRegionIterArgs()[std::distance(loopResults->begin(), it)];
    }

    /// Return the init operand that corresponds to the given region iter_arg.
    /// Return "nullptr" if the given block argument is not a region iter_arg
    /// of this loop op.
    OpOperand *getTiedLoopInit(BlockArgument bbArg) {
      auto iterArgs = (*this).getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      return &(*this).getInitsMutable()[std::distance(iterArgs.begin(), it)];
    }

    /// Return the init operand that corresponds to the given loop result.
    /// Return "nullptr" if the given OpResult is not a loop result or if this
    /// op does not expose any loop results.
    OpOperand *getTiedLoopInit(OpResult opResult) {
      auto loopResults = (*this).getLoopResults();
      if (!loopResults)
        return nullptr;
      auto it = llvm::find(*loopResults, opResult);
      if (it == loopResults->end())
        return nullptr;
      return &(*this).getInitsMutable()[std::distance(loopResults->begin(), it)];
    }

    /// Return the yielded value that corresponds to the given region iter_arg.
    /// Return "nullptr" if the given block argument is not a region iter_arg
    /// of this loop op or if there is no yield corresponding to this `bbArg`.
    OpOperand *getTiedLoopYieldedValue(BlockArgument bbArg) {
      auto iterArgs = (*this).getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      std::optional<llvm::MutableArrayRef<::mlir::OpOperand>> yieldValues =
        (*this).getYieldedValuesMutable();
      if (!yieldValues)
        return {};
      return &yieldValues.value()[std::distance(iterArgs.begin(), it)];
    }

    /// Return the loop result that corresponds to the given init operand.
    /// Return an "empty" OpResult if the given operand is not an init operand
    /// of this loop op or if this op does not expose any loop results.
    OpResult getTiedLoopResult(OpOperand *opOperand) {
      auto loopResults = (*this).getLoopResults();
      if (!loopResults)
        return {};
      auto initsMutable = (*this).getInitsMutable();
      auto it = llvm::find(initsMutable, *opOperand);
      if (it == initsMutable.end())
        return {};
      return (*loopResults)[std::distance(initsMutable.begin(), it)];
    }

    /// Return the loop result that corresponds to the given region iter_arg.
    /// Return an "empty" OpResult if the given block argument is not a region
    /// iter_arg of this loop op or if this op does not expose any loop results.
    OpResult getTiedLoopResult(BlockArgument bbArg) {
      auto loopResults = (*this).getLoopResults();
      if (!loopResults)
        return {};
      auto iterArgs = (*this).getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      return (*loopResults)[std::distance(iterArgs.begin(), it)];
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct LoopLikeOpInterfaceTrait : public ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns true if the given value is defined outside of the loop.
    /// A sensible implementation could be to check whether the value's defining
    /// operation lies outside of the loops body region. If the loop uses
    /// explicit capture of dependencies, an implementation could check whether
    /// the value corresponds to a captured dependency.
    bool isDefinedOutsideOfLoop(::mlir::Value  value) {
      return !(*static_cast<ConcreteOp *>(this))->isAncestor(value.getParentRegion()->getParentOp());
    }
    /// Moves the given loop-invariant operation out of the loop.
    void moveOutOfLoop(::mlir::Operation * op) {
      op->moveBefore((*static_cast<ConcreteOp *>(this)));
    }
    /// Promotes the loop body to its containing block if the loop is known to
    /// have a single iteration. Returns "success" if the promotion was
    /// successful.
    ::llvm::LogicalResult promoteIfSingleIteration(::mlir::RewriterBase & rewriter) {
      return ::mlir::failure();
    }
    /// Return all induction variables, if they exist. If the op has no notion of
    /// induction variable, then return std::nullopt. If it does have
    /// a notion but an instance doesn't have induction variables, then
    /// return empty vector.
    ::std::optional<::llvm::SmallVector<::mlir::Value>> getLoopInductionVars() {
      return ::std::nullopt;
    }
    /// Return all lower bounds, if they exist. If the op has no notion of
    /// lower bounds, then return std::nullopt. If it does have
    /// a notion but an instance doesn't have lower bounds, then
    /// return empty vector.
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopLowerBounds() {
      return ::std::nullopt;
    }
    /// Return all steps, if they exist. If the op has no notion of
    /// steps, then return std::nullopt. If it does have
    /// a notion but an instance doesn't have steps, then
    /// return empty vector.
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopSteps() {
      return ::std::nullopt;
    }
    /// Return all upper bounds, if they exist. If the op has no notion of
    /// lower bounds, then return std::nullopt. If it does have
    /// a notion but an instance doesn't have lower bounds, then
    /// return empty vector.
    ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopUpperBounds() {
      return ::std::nullopt;
    }
    /// Return the mutable "init" operands that are used as initialization
    /// values for the region "iter_args" of this loop.
    ::llvm::MutableArrayRef<::mlir::OpOperand> getInitsMutable() {
      return {};
    }
    /// Return the region "iter_args" (block arguments) that correspond to the
    /// "init" operands. If the op has multiple regions, return the
    /// corresponding block arguments of the entry region.
    ::mlir::Block::BlockArgListType getRegionIterArgs() {
      return ::mlir::Block::BlockArgListType();
    }
    /// Return the mutable operand range of values that are yielded to the next
    /// iteration by the loop terminator.
    /// 
    /// For loop operations that dont yield a value, this should return
    /// std::nullopt.
    std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> getYieldedValuesMutable() {
      return std::nullopt;
    }
    /// Return the range of results that are return from this loop and
    /// correspond to the "init" operands.
    /// 
    /// Note: This interface method is optional. If loop results are not
    /// exposed via this interface, "std::nullopt" should be returned.
    /// Otherwise, the number and types of results must match with the
    /// region iter_args, inits and yielded values that are exposed via this
    /// interface. If loop results are exposed but this loop op has no
    /// loop-carried variables, an empty result range (and not "std::nullopt")
    /// should be returned.
    ::std::optional<::mlir::ResultRange> getLoopResults() {
      return ::std::nullopt;
    }
    /// Append the specified additional "init" operands: replace this loop with
    /// a new loop that has the additional init operands. The loop body of
    /// this loop is moved over to the new loop.
    /// 
    /// `newInitOperands` specifies the additional "init" operands.
    /// `newYieldValuesFn` is a function that returns the yielded values (which
    /// can be computed based on the additional region iter_args). If
    /// `replaceInitOperandUsesInLoop` is set, all uses of the additional init
    /// operands inside of this loop are replaced with the corresponding, newly
    /// added region iter_args.
    /// 
    /// Note: Loops that do not support init/iter_args should return "failure".
    ::mlir::FailureOr<::mlir::LoopLikeOpInterface> replaceWithAdditionalYields(::mlir::RewriterBase & rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn & newYieldValuesFn) {
      return ::mlir::failure();
    }
    static ::llvm::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyLoopLikeOpInterface(op);
    }

    /// If there is a single induction variable return it, otherwise return
    /// std::nullopt.
    ::std::optional<::mlir::Value> getSingleInductionVar() {
      auto inductionVars = (*static_cast<ConcreteOp *>(this)).getLoopInductionVars();
      if (inductionVars.has_value() && (*inductionVars).size() == 1)
          return (*inductionVars)[0];
        return std::nullopt;
    }
    /// Return the single lower bound value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleLowerBound() {
      auto lowerBounds = (*static_cast<ConcreteOp *>(this)).getLoopLowerBounds();
      if (lowerBounds.has_value() && (*lowerBounds).size() == 1)
          return (*lowerBounds)[0];
      return std::nullopt;
    }
    /// Return the single step value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleStep() {
      auto steps = (*static_cast<ConcreteOp *>(this)).getLoopSteps();
      if (steps.has_value() && (*steps).size() == 1)
          return (*steps)[0];
      return std::nullopt;
    }
    /// Return the single upper bound value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleUpperBound() {
      auto upperBounds = (*static_cast<ConcreteOp *>(this)).getLoopUpperBounds();
      if (upperBounds.has_value() && (*upperBounds).size() == 1)
          return (*upperBounds)[0];
      return std::nullopt;
    }

    /// Append the specified additional "init" operands: replace this loop with
    /// a new loop that has the additional init operands. The loop body of this
    /// loop is moved over to the new loop.
    ///
    /// The newly added region iter_args are yielded from the loop.
    ::mlir::FailureOr<::mlir::LoopLikeOpInterface>
        replaceWithAdditionalIterOperands(::mlir::RewriterBase &rewriter,
                                          ::mlir::ValueRange newInitOperands,
                                          bool replaceInitOperandUsesInLoop) {
      return (*static_cast<ConcreteOp *>(this)).replaceWithAdditionalYields(
          rewriter, newInitOperands, replaceInitOperandUsesInLoop,
          [](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBBArgs) {
            return SmallVector<Value>(newBBArgs);
          });
    }

    /// Return the values that are yielded to the next iteration. If
    /// the loop doesnt yield any values return `{}`.
    ::mlir::ValueRange getYieldedValues() {
      auto mutableValues = (*static_cast<ConcreteOp *>(this)).getYieldedValuesMutable();
      if (!mutableValues || mutableValues->empty())
        return {};
      Operation *yieldOp = mutableValues->begin()->getOwner();
      unsigned firstOperandIndex = mutableValues->begin()->getOperandNumber();
      return OperandRange(
          yieldOp->operand_begin() + firstOperandIndex,
          yieldOp->operand_begin() + firstOperandIndex + mutableValues->size());
    }

    /// Return the "init" operands that are used as initialization values for
    /// the region "iter_args" of this loop.
    ::mlir::OperandRange getInits() {
      auto initsMutable = (*static_cast<ConcreteOp *>(this)).getInitsMutable();
      if (initsMutable.empty())
        return ::mlir::OperandRange((*static_cast<ConcreteOp *>(this))->operand_end(), (*static_cast<ConcreteOp *>(this))->operand_end());
      unsigned firstOperandIndex = initsMutable.begin()->getOperandNumber();
      return OperandRange(
          (*static_cast<ConcreteOp *>(this))->operand_begin() + firstOperandIndex,
          (*static_cast<ConcreteOp *>(this))->operand_begin() + firstOperandIndex + initsMutable.size());
    }

    /// Return the region iter_arg that corresponds to the given init operand.
    /// Return an "empty" block argument if the given operand is not an init
    /// operand of this loop op.
    BlockArgument getTiedLoopRegionIterArg(OpOperand *opOperand) {
      auto initsMutable = (*static_cast<ConcreteOp *>(this)).getInitsMutable();
      auto it = llvm::find(initsMutable, *opOperand);
      if (it == initsMutable.end())
        return {};
      return (*static_cast<ConcreteOp *>(this)).getRegionIterArgs()[std::distance(initsMutable.begin(), it)];
    }

    /// Return the region iter_arg that corresponds to the given loop result.
    /// Return an "empty" block argument if the given OpResult is not a loop
    /// result or if this op does not expose any loop results.
    BlockArgument getTiedLoopRegionIterArg(OpResult opResult) {
      auto loopResults = (*static_cast<ConcreteOp *>(this)).getLoopResults();
      if (!loopResults)
        return {};
      auto it = llvm::find(*loopResults, opResult);
      if (it == loopResults->end())
        return {};
      return (*static_cast<ConcreteOp *>(this)).getRegionIterArgs()[std::distance(loopResults->begin(), it)];
    }

    /// Return the init operand that corresponds to the given region iter_arg.
    /// Return "nullptr" if the given block argument is not a region iter_arg
    /// of this loop op.
    OpOperand *getTiedLoopInit(BlockArgument bbArg) {
      auto iterArgs = (*static_cast<ConcreteOp *>(this)).getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      return &(*static_cast<ConcreteOp *>(this)).getInitsMutable()[std::distance(iterArgs.begin(), it)];
    }

    /// Return the init operand that corresponds to the given loop result.
    /// Return "nullptr" if the given OpResult is not a loop result or if this
    /// op does not expose any loop results.
    OpOperand *getTiedLoopInit(OpResult opResult) {
      auto loopResults = (*static_cast<ConcreteOp *>(this)).getLoopResults();
      if (!loopResults)
        return nullptr;
      auto it = llvm::find(*loopResults, opResult);
      if (it == loopResults->end())
        return nullptr;
      return &(*static_cast<ConcreteOp *>(this)).getInitsMutable()[std::distance(loopResults->begin(), it)];
    }

    /// Return the yielded value that corresponds to the given region iter_arg.
    /// Return "nullptr" if the given block argument is not a region iter_arg
    /// of this loop op or if there is no yield corresponding to this `bbArg`.
    OpOperand *getTiedLoopYieldedValue(BlockArgument bbArg) {
      auto iterArgs = (*static_cast<ConcreteOp *>(this)).getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      std::optional<llvm::MutableArrayRef<::mlir::OpOperand>> yieldValues =
        (*static_cast<ConcreteOp *>(this)).getYieldedValuesMutable();
      if (!yieldValues)
        return {};
      return &yieldValues.value()[std::distance(iterArgs.begin(), it)];
    }

    /// Return the loop result that corresponds to the given init operand.
    /// Return an "empty" OpResult if the given operand is not an init operand
    /// of this loop op or if this op does not expose any loop results.
    OpResult getTiedLoopResult(OpOperand *opOperand) {
      auto loopResults = (*static_cast<ConcreteOp *>(this)).getLoopResults();
      if (!loopResults)
        return {};
      auto initsMutable = (*static_cast<ConcreteOp *>(this)).getInitsMutable();
      auto it = llvm::find(initsMutable, *opOperand);
      if (it == initsMutable.end())
        return {};
      return (*loopResults)[std::distance(initsMutable.begin(), it)];
    }

    /// Return the loop result that corresponds to the given region iter_arg.
    /// Return an "empty" OpResult if the given block argument is not a region
    /// iter_arg of this loop op or if this op does not expose any loop results.
    OpResult getTiedLoopResult(BlockArgument bbArg) {
      auto loopResults = (*static_cast<ConcreteOp *>(this)).getLoopResults();
      if (!loopResults)
        return {};
      auto iterArgs = (*static_cast<ConcreteOp *>(this)).getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      return (*loopResults)[std::distance(iterArgs.begin(), it)];
    }
  
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDefinedOutsideOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value  value) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDefinedOutsideOfLoop(value);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Region *> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopRegions();
}
template<typename ConcreteOp>
void detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::moveOutOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Operation * op) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).moveOutOfLoop(op);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::promoteIfSingleIteration(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase & rewriter) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).promoteIfSingleIteration(rewriter);
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::Value>> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopInductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopInductionVars();
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopLowerBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopLowerBounds();
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopSteps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopSteps();
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopUpperBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopUpperBounds();
}
template<typename ConcreteOp>
::llvm::MutableArrayRef<::mlir::OpOperand> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInitsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInitsMutable();
}
template<typename ConcreteOp>
::mlir::Block::BlockArgListType detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getRegionIterArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegionIterArgs();
}
template<typename ConcreteOp>
std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getYieldedValuesMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getYieldedValuesMutable();
}
template<typename ConcreteOp>
::std::optional<::mlir::ResultRange> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopResults();
}
template<typename ConcreteOp>
::mlir::FailureOr<::mlir::LoopLikeOpInterface> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::replaceWithAdditionalYields(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase & rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn & newYieldValuesFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).replaceWithAdditionalYields(rewriter, newInitOperands, replaceInitOperandUsesInLoop, newYieldValuesFn);
}
template<typename ConcreteOp>
bool detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDefinedOutsideOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value  value) {
  return static_cast<const ConcreteOp *>(impl)->isDefinedOutsideOfLoop(tablegen_opaque_val, value);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Region *> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopRegions(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::moveOutOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Operation * op) {
  return static_cast<const ConcreteOp *>(impl)->moveOutOfLoop(tablegen_opaque_val, op);
}
template<typename ConcreteOp>
::llvm::LogicalResult detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::promoteIfSingleIteration(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase & rewriter) {
  return static_cast<const ConcreteOp *>(impl)->promoteIfSingleIteration(tablegen_opaque_val, rewriter);
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::Value>> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopInductionVars(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopInductionVars(tablegen_opaque_val);
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopLowerBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopLowerBounds(tablegen_opaque_val);
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopSteps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopSteps(tablegen_opaque_val);
}
template<typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopUpperBounds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopUpperBounds(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::MutableArrayRef<::mlir::OpOperand> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInitsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInitsMutable(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Block::BlockArgListType detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getRegionIterArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRegionIterArgs(tablegen_opaque_val);
}
template<typename ConcreteOp>
std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getYieldedValuesMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getYieldedValuesMutable(tablegen_opaque_val);
}
template<typename ConcreteOp>
::std::optional<::mlir::ResultRange> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopResults(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopResults(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::FailureOr<::mlir::LoopLikeOpInterface> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::replaceWithAdditionalYields(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase & rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn & newYieldValuesFn) {
  return static_cast<const ConcreteOp *>(impl)->replaceWithAdditionalYields(tablegen_opaque_val, rewriter, newInitOperands, replaceInitOperandUsesInLoop, newYieldValuesFn);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDefinedOutsideOfLoop(::mlir::Operation *tablegen_opaque_val, ::mlir::Value value) const {
return !(llvm::cast<ConcreteOp>(tablegen_opaque_val))->isAncestor(value.getParentRegion()->getParentOp());
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::moveOutOfLoop(::mlir::Operation *tablegen_opaque_val, ::mlir::Operation *op) const {
op->moveBefore((llvm::cast<ConcreteOp>(tablegen_opaque_val)));
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::LogicalResult detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::promoteIfSingleIteration(::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase &rewriter) const {
return ::mlir::failure();
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::Value>> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopInductionVars(::mlir::Operation *tablegen_opaque_val) const {
return ::std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopLowerBounds(::mlir::Operation *tablegen_opaque_val) const {
return ::std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopSteps(::mlir::Operation *tablegen_opaque_val) const {
return ::std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopUpperBounds(::mlir::Operation *tablegen_opaque_val) const {
return ::std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::MutableArrayRef<::mlir::OpOperand> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInitsMutable(::mlir::Operation *tablegen_opaque_val) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Block::BlockArgListType detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRegionIterArgs(::mlir::Operation *tablegen_opaque_val) const {
return ::mlir::Block::BlockArgListType();
}
template<typename ConcreteModel, typename ConcreteOp>
std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getYieldedValuesMutable(::mlir::Operation *tablegen_opaque_val) const {
return std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::mlir::ResultRange> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopResults(::mlir::Operation *tablegen_opaque_val) const {
return ::std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::FailureOr<::mlir::LoopLikeOpInterface> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::replaceWithAdditionalYields(::mlir::Operation *tablegen_opaque_val, ::mlir::RewriterBase &rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn &newYieldValuesFn) const {
return ::mlir::failure();
}
} // namespace mlir
