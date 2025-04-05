/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Method to decompose the operation into simpler operations.
/// 
/// On success, this method returns one `Value` per result in the
/// original operation.
/// The order of the returned values must match the order of the
/// original values.
/// In other words, the returned vector can be used directly with
/// `RewriterBase::replaceOp(this, returnedValues)`.
FailureOr<SmallVector<Value>> mlir::linalg::AggregatedOpInterface::decomposeOperation(OpBuilder & b) {
      return getImpl()->decomposeOperation(getImpl(), getOperation(), b);
  }
/// Returns the left-hand side operand.
Value mlir::linalg::ContractionOpInterface::lhs() {
      return getImpl()->lhs(getImpl(), getOperation());
  }
/// Returns the right-hand side operand.
Value mlir::linalg::ContractionOpInterface::rhs() {
      return getImpl()->rhs(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// row-major matmul operation.
bool mlir::linalg::ContractionOpInterface::isRowMajorMatmul() {
      return getImpl()->isRowMajorMatmul(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// column-major matmul operation.
bool mlir::linalg::ContractionOpInterface::isColumnMajorMatmul() {
      return getImpl()->isColumnMajorMatmul(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// row-major batch matmul operation.
bool mlir::linalg::ContractionOpInterface::isRowMajorBatchMatmul() {
      return getImpl()->isRowMajorBatchMatmul(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// vector-matrix multiplication.
bool mlir::linalg::ContractionOpInterface::isVecmat() {
      return getImpl()->isVecmat(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// batched vector-matrix multiplication.
bool mlir::linalg::ContractionOpInterface::isBatchVecmat() {
      return getImpl()->isBatchVecmat(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// matrix-vector multiplication.
bool mlir::linalg::ContractionOpInterface::isMatvec() {
      return getImpl()->isMatvec(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// batched matrix-vector multiplication.
bool mlir::linalg::ContractionOpInterface::isBatchMatvec() {
      return getImpl()->isBatchMatvec(getImpl(), getOperation());
  }
/// Return the image operand.
Value mlir::linalg::ConvolutionOpInterface::image() {
      return getImpl()->image(getImpl(), getOperation());
  }
/// Return the filter operand.
Value mlir::linalg::ConvolutionOpInterface::filter() {
      return getImpl()->filter(getImpl(), getOperation());
  }
/// Return the fill value.
Value mlir::linalg::FillOpInterface::value() {
      return getImpl()->value(getImpl(), getOperation());
  }
/// Return the output operand.
Value mlir::linalg::FillOpInterface::output() {
      return getImpl()->output(getImpl(), getOperation());
  }
/// Return the result.
Value mlir::linalg::FillOpInterface::result() {
      return getImpl()->result(getImpl(), getOperation());
  }
/// Return the number of parallel loops.
unsigned mlir::linalg::LinalgOp::getNumParallelLoops() {
      return getImpl()->getNumParallelLoops(getImpl(), getOperation());
  }
/// Return the dims that are parallel loops.
void mlir::linalg::LinalgOp::getParallelDims(SmallVectorImpl<unsigned> & res) {
      return getImpl()->getParallelDims(getImpl(), getOperation(), res);
  }
/// Return the number of reduction loops.
unsigned mlir::linalg::LinalgOp::getNumReductionLoops() {
      return getImpl()->getNumReductionLoops(getImpl(), getOperation());
  }
/// Return the dims that are reduction loops.
void mlir::linalg::LinalgOp::getReductionDims(SmallVectorImpl<unsigned> & res) {
      return getImpl()->getReductionDims(getImpl(), getOperation(), res);
  }
/// Return the total number of loops within the current operation.
unsigned mlir::linalg::LinalgOp::getNumLoops() {
      return getImpl()->getNumLoops(getImpl(), getOperation());
  }
/// Returns true if the current operation has only one loop and it's a
/// reduction loop.
bool mlir::linalg::LinalgOp::hasSingleReductionLoop() {
      return getImpl()->hasSingleReductionLoop(getImpl(), getOperation());
  }
/// Return true if the payload uses the value loaded from `opOperand`. This
/// is useful to avoid loading from "write-only" memory that may be
/// uninitialized, as well as properly cloning "read-write" operands.
bool mlir::linalg::LinalgOp::payloadUsesValueFromOperand(OpOperand * opOperand) {
      return getImpl()->payloadUsesValueFromOperand(getImpl(), getOperation(), opOperand);
  }
/// Return true if `opOperand` is an init tensor. This is true when it is
/// an output tensor operand whose value is used in the payload region.
bool mlir::linalg::LinalgOp::isInitTensor(OpOperand * opOperand) {
      return getImpl()->isInitTensor(getImpl(), getOperation(), opOperand);
  }
/// Return the `opOperand` rank or zero for scalars or vectors not wrapped within a tensor or a memref.
int64_t mlir::linalg::LinalgOp::getRank(OpOperand* opOperand) {
      return getImpl()->getRank(getImpl(), getOperation(), opOperand);
  }
/// Return the input block arguments of the region.
Block::BlockArgListType mlir::linalg::LinalgOp::getRegionInputArgs() {
      return getImpl()->getRegionInputArgs(getImpl(), getOperation());
  }
/// Return the output block arguments of the region.
Block::BlockArgListType mlir::linalg::LinalgOp::getRegionOutputArgs() {
      return getImpl()->getRegionOutputArgs(getImpl(), getOperation());
  }
/// Return the `opOperand` shape or an empty vector for scalars or vectors
/// not wrapped within a tensor or a memref.
ArrayRef<int64_t> mlir::linalg::LinalgOp::getShape(OpOperand* opOperand) {
      return getImpl()->getShape(getImpl(), getOperation(), opOperand);
  }
/// Return the block argument for an `opOperand`.
BlockArgument mlir::linalg::LinalgOp::getMatchingBlockArgument(OpOperand * opOperand) {
      return getImpl()->getMatchingBlockArgument(getImpl(), getOperation(), opOperand);
  }
/// Return the operand for a `blockArgument`.
OpOperand *mlir::linalg::LinalgOp::getMatchingOpOperand(BlockArgument blockArgument) {
      return getImpl()->getMatchingOpOperand(getImpl(), getOperation(), blockArgument);
  }
/// Return the input or output indexing map for `opOperand`.
AffineMap mlir::linalg::LinalgOp::getMatchingIndexingMap(OpOperand* opOperand) {
      return getImpl()->getMatchingIndexingMap(getImpl(), getOperation(), opOperand);
  }
/// Return the indexing map for a `result`.
AffineMap mlir::linalg::LinalgOp::getIndexingMapMatchingResult(OpResult result) {
      return getImpl()->getIndexingMapMatchingResult(getImpl(), getOperation(), result);
  }
/// Return the value yielded by the region corresponding to an output
/// `opOperand`.
OpOperand *mlir::linalg::LinalgOp::getMatchingYieldValue(OpOperand* opOperand) {
      return getImpl()->getMatchingYieldValue(getImpl(), getOperation(), opOperand);
  }
/// Return the single block constituting the body of the operation by
/// calling the getBody method on the concrete operation.
Block*mlir::linalg::LinalgOp::getBlock() {
      return getImpl()->getBlock(getImpl(), getOperation());
  }
/// Return iterator types in the current operation.
/// 
/// Default implementation assumes that the operation has an attribute
/// `iterator_types`, but it's not always the case. Sometimes iterator types
/// can be infered from other parameters and in such cases default
/// getIteratorTypesArray should be overriden.
SmallVector<utils::IteratorType> mlir::linalg::LinalgOp::getIteratorTypesArray() {
      return getImpl()->getIteratorTypesArray(getImpl(), getOperation());
  }
/// Return true if the indexing map is depending on the current op instance.
/// This means that the indexing map is dynamically synthesized by using the
/// op instance's concrete attributes, instead of being static for all
/// instances of the same op kind.
bool mlir::linalg::LinalgOp::hasDynamicIndexingMaps() {
      return getImpl()->hasDynamicIndexingMaps(getImpl(), getOperation());
  }
/// Verify all attributes used by indexing maps are valid.
LogicalResult mlir::linalg::LinalgOp::verifyIndexingMapRequiredAttributes() {
      return getImpl()->verifyIndexingMapRequiredAttributes(getImpl(), getOperation());
  }
/// Return the indexing maps attribute within the current operation.
ArrayAttr mlir::linalg::LinalgOp::getIndexingMaps() {
      return getImpl()->getIndexingMaps(getImpl(), getOperation());
  }
/// Return the indexing maps within the current operation.
SmallVector<AffineMap> mlir::linalg::LinalgOp::getIndexingMapsArray() {
      return getImpl()->getIndexingMapsArray(getImpl(), getOperation());
  }
/// Return true if any of the operands has a dynamic shape.
bool mlir::linalg::LinalgOp::hasDynamicShape() {
      return getImpl()->hasDynamicShape(getImpl(), getOperation());
  }
/// Return the name registered for this op when lowering to an external
/// library call.
std::string mlir::linalg::LinalgOp::getLibraryCallName() {
      return getImpl()->getLibraryCallName(getImpl(), getOperation());
  }
/// Return whether the op accesses the iteration indices.
bool mlir::linalg::LinalgOp::hasIndexSemantics() {
      return getImpl()->hasIndexSemantics(getImpl(), getOperation());
  }
/// Return op operands that have a corresponding argument in the basic block.
/// By default, the block should have an argument for each operand, but there
/// are expection. For example, in `map` output operand isn't used in
/// the block.
::llvm::SmallVector<OpOperand *> mlir::linalg::LinalgOp::getOpOperandsMatchingBBargs() {
      return getImpl()->getOpOperandsMatchingBBargs(getImpl(), getOperation());
  }
/// Given a dimension of the iteration space of a Linalg operation, finds an
/// operand in the operation that is defined on such dimension. Returns
/// whether such operand was found or not. If found, also returns the
/// operand value and the dimension position within the operand.
LogicalResult mlir::linalg::LinalgOp::mapIterationSpaceDimToOperandDim(unsigned dimPos, ::mlir::Value & operand, unsigned & operandDimPos) {
      return getImpl()->mapIterationSpaceDimToOperandDim(getImpl(), getOperation(), dimPos, operand, operandDimPos);
  }
/// Given a dimension of the iteration space of a Linalg operation, finds
/// all the operands in the operation that are defined on such dimension.
/// Returns all the operand values found and their dimension positions in
/// `operandDimPairs`.
void mlir::linalg::LinalgOp::mapIterationSpaceDimToAllOperandDims(unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>& operandDimPairs) {
      return getImpl()->mapIterationSpaceDimToAllOperandDims(getImpl(), getOperation(), dimPos, operandDimPairs);
  }
/// Hook to provide a custom AffineMap used to compute all the operand
/// subshapes given loop bounds. This is used to answer the question: "given
/// an iteration space over the codomain, what are the subshapes of the
/// operands involved in the computation".
/// The default behavior is to just concatenate all the indexing maps.
/// A custom AffineMap allows providing a map that can be used to
/// compute subshapes even in cases where the concatenation of indexing maps
/// (i.e. the data traversal order) is not a simple permutation of the loop
/// traversal order. It is then possible to define ops with skewed data
/// traversal order for which we can still easily compute hyperrectangular
/// loop bounds and subviews.
AffineMap mlir::linalg::LinalgOp::getLoopsToShapesMap() {
      return getImpl()->getLoopsToShapesMap(getImpl(), getOperation());
  }
/// Hook to provide a custom AffineMap used to construct the
/// hyperrectangular loop iteration space given all the operand subshapes.
/// This is used to answer the question:
/// "Given a list of operand ranges, what is the subportion of the iteration
/// space involved in the computation".
/// This is the inverse problem of `getLoopsToShapesMap`.
/// Return the empty AffineMap when such an AffineMap cannot be constructed.
/// The default behavior is based on a very simple inference procedure that
/// only works with permutation affine maps.
/// A more advanced Tensor-Comprehension like inference is possible but has
/// proven to be ambiguous in unfavorable case.
/// A safer and more robust alternative is to allow each op to define
/// its own AffineMap.
AffineMap mlir::linalg::LinalgOp::getShapesToLoopsMap() {
      return getImpl()->getShapesToLoopsMap(getImpl(), getOperation());
  }
/// Checks if the given operands can be dropped, and the remaining
/// operands can still compute the bounds of the op.
bool mlir::linalg::LinalgOp::canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands) {
      return getImpl()->canOpOperandsBeDropped(getImpl(), getOperation(), droppedOperands);
  }
/// Like `getShape`, but only returns statically-known information, without
/// generating any new IR. For each shape dimension, returns >=0 if that
/// dimension is statically known, or ShapedType::kDynamic otherwise.
SmallVector<int64_t> mlir::linalg::LinalgOp::getStaticShape() {
      return getImpl()->getStaticShape(getImpl(), getOperation());
  }
/// Returns the statically-known loop ranges. Composes
/// `getShapesToLoopsMap()` with the result of `getStaticShape`.
/// Returns ShapedType::kDynamic for non-statically-known loop ranges.
/// This is expected to be called by a valid Linalg op
SmallVector<int64_t, 4> mlir::linalg::LinalgOp::getStaticLoopRanges() {
      return getImpl()->getStaticLoopRanges(getImpl(), getOperation());
  }
/// Returns the region builder for constructing the body for linalg.generic.
/// Returns a null function if this named op does not define a region
/// builder.
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> mlir::linalg::LinalgOp::getRegionBuilder() {
      return getImpl()->getRegionBuilder();
  }
/// Return true if all the indexing maps are projected permutations.
/// Otherwise return false.
bool mlir::linalg::LinalgOp::hasOnlyProjectedPermutations() {
      return getImpl()->hasOnlyProjectedPermutations(getImpl(), getOperation());
  }
/// Return start and end indices of the init operands range.
::mlir::MutableOperandRange mlir::linalg::LinalgOp::getDpsInitsMutable() {
      return getImpl()->implDestinationStyleOpInterface->getDpsInitsMutable(getImpl()->implDestinationStyleOpInterface, getOperation());
  }
