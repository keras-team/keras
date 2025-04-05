/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns true if the given value is defined outside of the loop.
/// A sensible implementation could be to check whether the value's defining
/// operation lies outside of the loops body region. If the loop uses
/// explicit capture of dependencies, an implementation could check whether
/// the value corresponds to a captured dependency.
bool mlir::LoopLikeOpInterface::isDefinedOutsideOfLoop(::mlir::Value  value) {
      return getImpl()->isDefinedOutsideOfLoop(getImpl(), getOperation(), value);
  }
/// Returns the regions that make up the body of the loop and should be
/// inspected for loop-invariant operations.
::llvm::SmallVector<::mlir::Region *> mlir::LoopLikeOpInterface::getLoopRegions() {
      return getImpl()->getLoopRegions(getImpl(), getOperation());
  }
/// Moves the given loop-invariant operation out of the loop.
void mlir::LoopLikeOpInterface::moveOutOfLoop(::mlir::Operation * op) {
      return getImpl()->moveOutOfLoop(getImpl(), getOperation(), op);
  }
/// Promotes the loop body to its containing block if the loop is known to
/// have a single iteration. Returns "success" if the promotion was
/// successful.
::llvm::LogicalResult mlir::LoopLikeOpInterface::promoteIfSingleIteration(::mlir::RewriterBase & rewriter) {
      return getImpl()->promoteIfSingleIteration(getImpl(), getOperation(), rewriter);
  }
/// Return all induction variables, if they exist. If the op has no notion of
/// induction variable, then return std::nullopt. If it does have
/// a notion but an instance doesn't have induction variables, then
/// return empty vector.
::std::optional<::llvm::SmallVector<::mlir::Value>> mlir::LoopLikeOpInterface::getLoopInductionVars() {
      return getImpl()->getLoopInductionVars(getImpl(), getOperation());
  }
/// Return all lower bounds, if they exist. If the op has no notion of
/// lower bounds, then return std::nullopt. If it does have
/// a notion but an instance doesn't have lower bounds, then
/// return empty vector.
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> mlir::LoopLikeOpInterface::getLoopLowerBounds() {
      return getImpl()->getLoopLowerBounds(getImpl(), getOperation());
  }
/// Return all steps, if they exist. If the op has no notion of
/// steps, then return std::nullopt. If it does have
/// a notion but an instance doesn't have steps, then
/// return empty vector.
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> mlir::LoopLikeOpInterface::getLoopSteps() {
      return getImpl()->getLoopSteps(getImpl(), getOperation());
  }
/// Return all upper bounds, if they exist. If the op has no notion of
/// lower bounds, then return std::nullopt. If it does have
/// a notion but an instance doesn't have lower bounds, then
/// return empty vector.
::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> mlir::LoopLikeOpInterface::getLoopUpperBounds() {
      return getImpl()->getLoopUpperBounds(getImpl(), getOperation());
  }
/// Return the mutable "init" operands that are used as initialization
/// values for the region "iter_args" of this loop.
::llvm::MutableArrayRef<::mlir::OpOperand> mlir::LoopLikeOpInterface::getInitsMutable() {
      return getImpl()->getInitsMutable(getImpl(), getOperation());
  }
/// Return the region "iter_args" (block arguments) that correspond to the
/// "init" operands. If the op has multiple regions, return the
/// corresponding block arguments of the entry region.
::mlir::Block::BlockArgListType mlir::LoopLikeOpInterface::getRegionIterArgs() {
      return getImpl()->getRegionIterArgs(getImpl(), getOperation());
  }
/// Return the mutable operand range of values that are yielded to the next
/// iteration by the loop terminator.
/// 
/// For loop operations that dont yield a value, this should return
/// std::nullopt.
std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> mlir::LoopLikeOpInterface::getYieldedValuesMutable() {
      return getImpl()->getYieldedValuesMutable(getImpl(), getOperation());
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
::std::optional<::mlir::ResultRange> mlir::LoopLikeOpInterface::getLoopResults() {
      return getImpl()->getLoopResults(getImpl(), getOperation());
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
::mlir::FailureOr<::mlir::LoopLikeOpInterface> mlir::LoopLikeOpInterface::replaceWithAdditionalYields(::mlir::RewriterBase & rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn & newYieldValuesFn) {
      return getImpl()->replaceWithAdditionalYields(getImpl(), getOperation(), rewriter, newInitOperands, replaceInitOperandUsesInLoop, newYieldValuesFn);
  }
