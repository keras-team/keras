/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return `true` if the given Value may bufferize to a new buffer
/// allocation. If it is statically unknown that the given Value
/// bufferizes to a buffer allocation, `true` should be returned.
bool mlir::bufferization::BufferizableOpInterface::bufferizesToAllocation(::mlir::Value value) {
      return getImpl()->bufferizesToAllocation(getImpl(), getOperation(), value);
  }
/// Return `true` if the given OpOperand bufferizes to a memory read. This
/// method will never be called on OpOperands that do not have a tensor
/// type.
/// 
/// Note: It is always safe to consider an OpOperand as a memory read,
/// even if it does actually not read; however, this can introduce
/// unnecessary out-of-place bufferization decisions. One-Shot Analysis
/// considers OpOperands of unknown ops (that do not implement this
/// interface) as reading OpOperands.
bool mlir::bufferization::BufferizableOpInterface::bufferizesToMemoryRead(::mlir::OpOperand & opOperand, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->bufferizesToMemoryRead(getImpl(), getOperation(), opOperand, state);
  }
/// Return `true` if the given OpOperand bufferizes to a memory write.
/// 
/// This method will never be called on OpOperands that do not have a
/// tensor type.
/// 
/// This method will never be called on OpOperands that do not have an
/// aliasing Value. Intuitively, it does not make sense for an OpOperand
/// to bufferize to a memory write without returning an aliasing tensor,
/// because the write would have no visible effect outside of the op.
/// 
/// Note: It is always safe to consider an OpOperand as a memory write,
/// even if it does actually not write; however, this can introduce
/// unnecessary out-of-place bufferization decisions. One-Shot Analysis
/// considers OpOperands of unknown ops (that do not implement this
/// interface) as writing OpOperands.
bool mlir::bufferization::BufferizableOpInterface::bufferizesToMemoryWrite(::mlir::OpOperand & opOperand, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->bufferizesToMemoryWrite(getImpl(), getOperation(), opOperand, state);
  }
/// Return `true` if the operation bufferizes to IR that performs only
/// element-wise accesses on the specified tensor operands. (The operands
/// must have the same shape.) The `bufferize` method must be implemented
/// in such a way that it is free of loop-carried dependences. I.e., all
/// loads at a position appear before all stores at the same position.
/// 
/// Example: Consider a hypothetical op element-wise op, where the "ins"
/// bufferize to a memory read and the "outs" bufferize to a memory write.
/// ```
/// test.element_wise ins(%0), outs(%1) : tensor<3xf32>
/// ```
/// 
/// The following is a valid access pattern:
/// ```
/// load(%0[1])
/// store(%1[1])
/// load(%0[2])
/// store(%1[2])
/// load(%0[0])
/// store(%1[0])
/// ```
/// 
/// The following would be an invalid (not element-wise) access pattern:
/// ```
/// load(%0[1])
/// store(%0[1])
/// load(%0[1])
/// ...
/// ```
/// 
/// Element-wise ops can sometimes bufferize more efficiently: a RaW
/// conflict between two operands of the same op can be avoided if it is
/// guaranteed that an original element value is no longer needed after
/// writing a computed element value at the same location. E.g., such an
/// optimization is possible in the above example if %0 and %1 are
/// equivalent tensors. (It is not possible, if %0 and %1 are merely
/// aliasing. It is not necessary if %0 and %1 are not aliasing at all,
/// because there would be no conflict anyway.)
/// 
/// Note: Tensor operands that are not included in `opOperands` can be
/// ignored. A conservative implementation of this interface method may
/// always return "false".
bool mlir::bufferization::BufferizableOpInterface::bufferizesToElementwiseAccess(const ::mlir::bufferization::AnalysisState & state, ArrayRef<OpOperand *> opOperands) {
      return getImpl()->bufferizesToElementwiseAccess(getImpl(), getOperation(), state, opOperands);
  }
/// Return `true` if the given OpResult bufferizes to a memory write.
/// This is the same property as `bufferizesToMemoryWrite`, but from The
/// perspective of OpResults.
/// 
/// This method will never be called on OpResults that do not have a
/// tensor type.
/// 
/// This method has a default implementation. By default, it returns
/// `true` if any of the following three cases applies.
/// 
/// 1. There is no corresponding aliasing OpOperand.
/// 
///    Example: `tensor.generate ... : tensor<10xf32>`
///    The op fills a newly allocated buffer and bufferizes to a memory
///    write.
/// 
///    Counter-example: bufferization.alloc_tensor
///    The op just allocates and does not specify the data of the tensor,
///    so resultBufferizesToMemoryWrite is overridden to return false.
/// 
/// 2. At least one aliasing OpOperand bufferizes to a memory write.
/// 
///    Example: `tensor.insert %f into %t[...] : tensor<?xf32>`
///    The destination OpOperand bufferizes to a memory write, so the
///    result also bufferizes to a memory write.
/// 
/// 3. At least one aliasing OpOperand's value is defined inside the
///    defining op of the given OpResult and it is a memory write.
/// 
///    According to this rule, an aliasing OpOperand value that is defined
///    inside this op and is bufferizing to a memory write makes the given
///    OpResult bufferize to a memory write.
/// 
///    Example:
///    ```
///    %r = scf.if ... -> tensor<?xf32> {
///      %1 = tensor.insert %f into %t[...] : tensor<?xf32>
///      scf.yield %1 : tensor<?xf32>
///    } else { ... }
///    ```
///    The scf.if result bufferizes to a memory write because %1 (an
///    OpResult defined inside the scf.if op) bufferizes to a memory
///    write.
bool mlir::bufferization::BufferizableOpInterface::resultBufferizesToMemoryWrite(::mlir::OpResult opResult, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->resultBufferizesToMemoryWrite(getImpl(), getOperation(), opResult, state);
  }
/// Return `true` if the given OpOperand must bufferize in-place. Alias
/// sets and inplace attributes will be set up accordingly before making
/// any other bufferization decisions. This method will never be called on
/// OpOperands that do not have a tensor type.
/// 
/// Note: Unranked tensor OpOperands always bufferize in-place. This could
/// be extended in the future. Unranked tensors are used with external
/// functions only.
bool mlir::bufferization::BufferizableOpInterface::mustBufferizeInPlace(::mlir::OpOperand & opOperand, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->mustBufferizeInPlace(getImpl(), getOperation(), opOperand, state);
  }
/// Return the Values that may alias with a given OpOperand when
/// bufferized in-place. This method will never be called on OpOperands
/// that do not have a tensor type.
/// 
/// This method can return multiple Values, indicating that a given
/// OpOperand may at runtime alias with any (or multiple) of the returned
/// Values.
/// 
/// Each alias is specified with a degree of certainty:
/// 
/// * MAYBE (`isDefinite = false`): At runtime, buffer(opOperand) may
///   alias with the specified Value.
/// * DEFINITE (`isDefinite = true`, default): At runtime,
///   buffer(opOperand) is guaranteed to alias the buffer of the specified
///   Value. This is a stronger property than MAYBE and allows for more
///   precise analyses. DEFINITE properties should be used when possible.
/// 
/// Furthermore, each alias is specified with a buffer relation:
/// 
/// * `BufferRelation::Equivalent`: Both aliases are the exact same
///   buffer. I.e., same size, no offset, same strides.
/// * `BufferRelation::Unknown`: There is no further information apart
///   from the fact that both buffers alias.
/// 
/// False positives are allowed in the list of Values, but they can
/// adversely affect the accuracy of the anlysis. On the contrary,
/// omitting potential aliases is incorrect.
/// 
/// One possible (conservative) implementation of this interface method,
/// that is always safe, is to return all tensor Values with
/// BufferRelation::Unknown and MAYBE.
/// 
/// Examples:
/// 
/// ```
/// // aliasingValues(%t) = DEFINITE {Equivalent %r}
/// %r = tensor.insert_slice %f into %t : tensor<10xf32>
/// 
/// // aliasingValues(%t) = DEFINITE {Unknown %r}
/// // Note: "Buffer is subset of buffer" relationship are not yet
/// // supported, so "Unknown" is the best we can do for now.
/// %r = tensor.extract_slice %t[0]][5][1]
///     : tensor<10xf32> to tensor<5xf32>
/// 
/// // aliasingValues(%t1) = MAYBE {Equivalent %r}
/// // aliasingValues(%t2) = MAYBE {Equivalent %r}
/// %r = arith.select %c, %t1, %t2 : tensor<10xf32>
/// 
/// // A hypothetical op that bufferizes to rolling a dice and based on
/// // the result to either return buffer(%t) or a newly allocated copy
/// // thereof.
/// // aliasingValues(%t) = MAYBE {Equivalent %r}
/// %r = "dummy.alias_or_copy(%t) : (tensor<10xf32>) -> (tensor<10xf32>)"
/// ```
::mlir::bufferization::AliasingValueList mlir::bufferization::BufferizableOpInterface::getAliasingValues(::mlir::OpOperand & opOperand, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->getAliasingValues(getImpl(), getOperation(), opOperand, state);
  }
/// Return the OpOperands that alias with a given Value when bufferized
/// in-place. This method will never be called on Values that do not
/// have a tensor type.
/// 
/// By default, this method is the inverse of `getAliasingValues`. Ops
/// with a region that yield values may want to override this method to
/// return the OpOperands that are yielded by the terminator.
/// 
/// This method can return multiple OpOperands, indicating that a given
/// Value may at runtime alias with any (or multiple) of the returned
/// OpOperands.
/// 
/// This property is specified with a degree of certainty:
/// 
/// * MAYBE (`isDefinite = false`): At runtime, buffer(value) may alias
///   with the specified OpOperand.
/// * DEFINITE (`isDefinite = true`, default): At runtime,
///   buffer(value) is guaranteed to alias the buffer of the specified
///   OpOperand. This is a stronger property than MAYBE and allows for
///   more precise analyses. DEFINITE properties should be used when
///   possible.
/// 
/// For each alias, a BufferRelation can be specified:
/// 
/// * `BufferRelation::Equivalent`: Both aliases are the exact same
///   buffer. I.e., same size, no offset, same strides.
/// * `BufferRelation::Unknown`: There is no further information apart
///   from the fact that both buffers alias.
/// 
/// False positives are allowed in the list of OpOperands, but they can
/// adversely affect the accuracy of the anlysis. On the contrary,
/// omitting potential aliases is incorrect.
/// 
/// One possible (conservative) implementation of this interface method,
/// that is always safe, is to return all tensor OpOperands with
/// BufferRelation::Unknown and MAYBE.
/// 
/// Note: If the returned list of OpOperands is empty, this op definitely
/// bufferizes to a new allocation. In that case `bufferizesToAllocation`
/// must return `true`.
/// 
/// Examples:
/// 
/// ```
/// // aliasingOpOperands(%r) = DEFINITE {Equivalent %t}
/// %r = tensor.insert_slice %f into %t : tensor<10xf32>
/// 
/// // aliasingOpOperands(%r) = DEFINITE {Unknown %t}
/// %r = tensor.extract_slice %t[0]][5][1]
///     : tensor<10xf32> to tensor<5xf32>
/// 
/// // aliasingOpOperands(%r) = DEFINITE {Equivalent %t1, Equivalent %t2}
/// %r = arith.select %c, %t1, %t2 : tensor<10xf32>
/// 
/// // aliasingOpOperands(%r) = MAYBE {}
/// %r = tensor.empty() : tensor<10xf32>
/// ```
::mlir::bufferization::AliasingOpOperandList mlir::bufferization::BufferizableOpInterface::getAliasingOpOperands(::mlir::Value value, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->getAliasingOpOperands(getImpl(), getOperation(), value, state);
  }
/// Resolve all inplacability conflicts by inserting explicit
/// `bufferization.alloc_tensor` ops. Examples of inplacability conflicts
/// are read-after-write conflicts or writes into non-writable buffers.
/// 
/// This method should rewrite the IR in such a way that for each tensor
/// OpOperand t, buffer(t) can be directly used when during bufferization.
/// The bufferization does no longer have to care about inplacability
/// conflicts.
/// 
/// This method can query analysis information from the given analysis
/// state.
::llvm::LogicalResult mlir::bufferization::BufferizableOpInterface::resolveConflicts(::mlir::RewriterBase & rewriter, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->resolveConflicts(getImpl(), getOperation(), rewriter, state);
  }
/// Bufferize this op, i.e., rewrite it into a memref-based equivalent.
/// Buffers of tensor SSA values can be retrieved via `getBuffer`.
/// Uses of tensor results of the existing tensor op can be replaced with
/// `replaceOpWithBufferizedValues` or `replaceOpWithNewBufferizedOp`.
/// These two functions automatically handle the tensor-to-memref type
/// conversion.
/// 
/// The implementation of this method must be consistent with the
/// remaining methods, in particular `getAliasingOpOperands`. I.e., a
/// tensor result `r` may only be replaced with:
/// 
/// a) One of the buffers in getAliasingOpOperands(r).
/// b) Or: A newly allocated buffer (only if `bufferizesToAllocation`).
/// 
/// This method will never be called on ops that do not have at least one
/// tensor operand/result.
/// 
/// The return value of this method indicates whether there was an error
/// while bufferizing this op (such as failing to create a new buffer
/// allocation op). The bufferization driver immediately stops bufferizing
/// the input IR and returns `failure` in that case. If this op is
/// expected to survive bufferization, `success` should be returned
/// (together with `allow-unknown-ops` enabled).
/// 
/// Note: If this op supports unstructured control flow in its regions,
/// then this function should also bufferize all block signatures that
/// belong to this op. Branch ops (that branch to a block) are typically
/// bufferized together with the block signature (this is just a
/// suggestion to make sure IR is valid at every point in time and could
/// be done differently).
::llvm::LogicalResult mlir::bufferization::BufferizableOpInterface::bufferize(::mlir::RewriterBase & rewriter, const ::mlir::bufferization::BufferizationOptions & options) {
      return getImpl()->bufferize(getImpl(), getOperation(), rewriter, options);
  }
/// Return `true` if the given Value can be written to in-place. Value is
/// either an OpResult of this operation or a BlockArgument of a block of
/// this operation.
/// 
/// Most OpResult buffers can be written to, but some ops such as
/// ConstantOp may bufferize to non-writable (read-only) memory locations.
/// Therefore, by default, this method returns `true` for OpResults. This
/// method will never be called on OpResults that do not have a tensor
/// type.
/// 
/// Whether a BlockArgument can be written to or not depends on the
/// operation. This method conservatively returns `false`. This method
/// will never be called on BlockArguments that do not have a tensor type.
bool mlir::bufferization::BufferizableOpInterface::isWritable(::mlir::Value value, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->isWritable(getImpl(), getOperation(), value, state);
  }
/// Return `true` if the `uRead` and `uWrite` do not constitute a RaW
/// conflict. If they are conflicting or if it is unknown whether they are
/// conflicting, return `false`. This method will never be called with
/// OpOperands that do not have a tensor type. At least one of the two
/// given OpOperands belongs to this operation.
/// 
/// This method can be implemented to specify custom RaW analysis rules.
/// If this method returns `true` the given OpOperands are not considered
/// to be conflicting and do not force out-of-place bufferization. (There
/// may still be other conflicts that do.)
bool mlir::bufferization::BufferizableOpInterface::isNotConflicting(::mlir::OpOperand * uRead, ::mlir::OpOperand * uWrite, const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->isNotConflicting(getImpl(), getOperation(), uRead, uWrite, state);
  }
/// Return `failure` if this op does not pass the analysis. This method
/// is run during One-Shot Bufferize (after all post-analysis steps). If
/// the op does not pass the analysis, bufferization is aborted.
/// 
/// This method can be used to check expected invariants and limitations
/// of the current bufferization implementation.
::llvm::LogicalResult mlir::bufferization::BufferizableOpInterface::verifyAnalysis(const ::mlir::bufferization::AnalysisState & state) {
      return getImpl()->verifyAnalysis(getImpl(), getOperation(), state);
  }
/// Return the bufferized type of the given tensor value (without
/// bufferizing the IR). The value is either a BlockArgument of a block
/// that belongs to this op or an OpResult of the given op.
/// 
/// This method is useful when the bufferized type of value must be
/// predicted before modifying any IR.
/// 
/// Implementations may call `bufferization::getBufferType` to compute the
/// bufferized type of another SSA value. The same (unmodified)
/// `invocationStack` must be passed to that function. The stack contains
/// all SSA values for which a buffer type computation is currently in
/// progress. Implementations may inspect the stack to detect repetitive
/// computations for the same SSA value. (E.g., when bufferized types of a
/// loop.)
/// 
/// Note: This interface method should never be called directly from user
/// code. Always use `bufferization::getBufferType`.
::mlir::FailureOr<::mlir::BaseMemRefType> mlir::bufferization::BufferizableOpInterface::getBufferType(::mlir::Value value, const ::mlir::bufferization::BufferizationOptions & options, ::llvm::SmallVector<::mlir::Value> & invocationStack) {
      return getImpl()->getBufferType(getImpl(), getOperation(), value, options, invocationStack);
  }
/// Return `true` if the given region of this op is repetitive. By default
/// this information is queried from the `RegionBranchOpInterface`. Ops
/// that do not implement this inferface can override this method to
/// declare regions as repetitive.
/// 
/// The RaW conflict detection of One-Shot Analysis is more strict inside
/// repetitive regions: Op dominance cannot always be used to rule out
/// certain potential conflicts (e.g., a conflicting write happening after
/// a read), because there may not be a meaningful ordering of certain ops
/// that are executed multiple times. This is described in more detail in
/// documentation of One-Shot Analysis.
bool mlir::bufferization::BufferizableOpInterface::isRepetitiveRegion(unsigned index) {
      return getImpl()->isRepetitiveRegion(getImpl(), getOperation(), index);
  }
/// Return `true` if the given region of this op is parallel, i.e.,
/// multiple instances of the region may be executing at the same time.
/// If a region is parallel, it must also be marked as "repetitive".
/// 
/// The RaW conflict detection of One-Shot Analysis is more strict inside
/// parallel regions: Buffer may have to be privatized.
/// 
/// By default, regions are assumed to be sequential.
bool mlir::bufferization::BufferizableOpInterface::isParallelRegion(unsigned index) {
      return getImpl()->isParallelRegion(getImpl(), getOperation(), index);
  }
/// Return "true" if the this op has tensor semantics and should be
/// bufferized. By default, ops with tensor operands, tensor op results
/// and/or tensor block arguments have tensor semantics.
/// 
/// This interface methods can be implemented by ops that should be
/// bufferized but do not have tensor semantics according to the above
/// definition. E.g., this function can return "true" for symbols.
bool mlir::bufferization::BufferizableOpInterface::hasTensorSemantics() {
      return getImpl()->hasTensorSemantics(getImpl(), getOperation());
  }
/// Return `true` if the op and this interface implementation supports
/// unstructured control flow. I.e., regions with multiple blocks. This is
/// not supported in most ops, so the default answer is `false`.
bool mlir::bufferization::BufferizableOpInterface::supportsUnstructuredControlFlow() {
      return getImpl()->supportsUnstructuredControlFlow();
  }
