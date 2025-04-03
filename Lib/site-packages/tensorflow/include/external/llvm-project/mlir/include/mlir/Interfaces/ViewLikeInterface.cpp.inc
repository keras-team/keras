/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return the number of leading operands before the `offsets`, `sizes` and
/// and `strides` operands.
unsigned mlir::OffsetSizeAndStrideOpInterface::getOffsetSizeAndStrideStartOperandIndex() {
      return getImpl()->getOffsetSizeAndStrideStartOperandIndex();
  }
/// Return the expected rank of each of the`static_offsets`, `static_sizes`
/// and `static_strides` attributes.
std::array<unsigned, 3> mlir::OffsetSizeAndStrideOpInterface::getArrayAttrMaxRanks() {
      return getImpl()->getArrayAttrMaxRanks(getImpl(), getOperation());
  }
/// Return the dynamic offset operands.
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::getOffsets() {
      return getImpl()->getOffsets(getImpl(), getOperation());
  }
/// Return the dynamic size operands.
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::getSizes() {
      return getImpl()->getSizes(getImpl(), getOperation());
  }
/// Return the dynamic stride operands.
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::getStrides() {
      return getImpl()->getStrides(getImpl(), getOperation());
  }
/// Return the static offset attributes.
::llvm::ArrayRef<int64_t> mlir::OffsetSizeAndStrideOpInterface::getStaticOffsets() {
      return getImpl()->getStaticOffsets(getImpl(), getOperation());
  }
/// Return the static size attributes.
::llvm::ArrayRef<int64_t> mlir::OffsetSizeAndStrideOpInterface::getStaticSizes() {
      return getImpl()->getStaticSizes(getImpl(), getOperation());
  }
/// Return the dynamic stride attributes.
::llvm::ArrayRef<int64_t> mlir::OffsetSizeAndStrideOpInterface::getStaticStrides() {
      return getImpl()->getStaticStrides(getImpl(), getOperation());
  }
/// Return a vector of all the static or dynamic offsets of the op.
::llvm::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedOffsets() {
      return getImpl()->getMixedOffsets(getImpl(), getOperation());
  }
/// Return a vector of all the static or dynamic sizes of the op.
::llvm::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedSizes() {
      return getImpl()->getMixedSizes(getImpl(), getOperation());
  }
/// Return a vector of all the static or dynamic strides of the op.
::llvm::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedStrides() {
      return getImpl()->getMixedStrides(getImpl(), getOperation());
  }
/// Return true if the offset `idx` is dynamic.
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicOffset(unsigned idx) {
      return getImpl()->isDynamicOffset(getImpl(), getOperation(), idx);
  }
/// Return true if the size `idx` is dynamic.
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicSize(unsigned idx) {
      return getImpl()->isDynamicSize(getImpl(), getOperation(), idx);
  }
/// Return true if the stride `idx` is dynamic.
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicStride(unsigned idx) {
      return getImpl()->isDynamicStride(getImpl(), getOperation(), idx);
  }
/// Assert the offset `idx` is a static constant and return its value.
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticOffset(unsigned idx) {
      return getImpl()->getStaticOffset(getImpl(), getOperation(), idx);
  }
/// Assert the size `idx` is a static constant and return its value.
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticSize(unsigned idx) {
      return getImpl()->getStaticSize(getImpl(), getOperation(), idx);
  }
/// Assert the stride `idx` is a static constant and return its value.
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticStride(unsigned idx) {
      return getImpl()->getStaticStride(getImpl(), getOperation(), idx);
  }
/// Assert the offset `idx` is dynamic and return the position of the
/// corresponding operand.
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicOffset(unsigned idx) {
      return getImpl()->getIndexOfDynamicOffset(getImpl(), getOperation(), idx);
  }
/// Assert the size `idx` is dynamic and return the position of the
/// corresponding operand.
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicSize(unsigned idx) {
      return getImpl()->getIndexOfDynamicSize(getImpl(), getOperation(), idx);
  }
/// Assert the stride `idx` is dynamic and return the position of the
/// corresponding operand.
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicStride(unsigned idx) {
      return getImpl()->getIndexOfDynamicStride(getImpl(), getOperation(), idx);
  }
/// Assert the offset `idx` is dynamic and return its value.
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicOffset(unsigned idx) {
      return getImpl()->getDynamicOffset(getImpl(), getOperation(), idx);
  }
/// Assert the size `idx` is dynamic and return its value.
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicSize(unsigned idx) {
      return getImpl()->getDynamicSize(getImpl(), getOperation(), idx);
  }
/// Assert the stride `idx` is dynamic and return its value.
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicStride(unsigned idx) {
      return getImpl()->getDynamicStride(getImpl(), getOperation(), idx);
  }
/// Return true if all `other`'s offsets, sizes and strides are the same.
/// Takes a custom `cmp` comparison function on OpFoldResult to avoid taking
/// a dialect dependence.
bool mlir::OffsetSizeAndStrideOpInterface::isSameAs(::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp) {
      return getImpl()->isSameAs(getImpl(), getOperation(), other, cmp);
  }
/// Return true if all strides are guaranteed to be 1.
bool mlir::OffsetSizeAndStrideOpInterface::hasUnitStride() {
      return getImpl()->hasUnitStride(getImpl(), getOperation());
  }
/// Return true if all offsets are guaranteed to be 0.
bool mlir::OffsetSizeAndStrideOpInterface::hasZeroOffset() {
      return getImpl()->hasZeroOffset(getImpl(), getOperation());
  }
/// Returns the source buffer from which the view is created.
::mlir::Value mlir::ViewLikeOpInterface::getViewSource() {
      return getImpl()->getViewSource(getImpl(), getOperation());
  }
