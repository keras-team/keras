/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Applies the transformation represented by the current operation. This
/// accepts as arguments the object that must be populated with results of
/// the current transformation and a transformation state object that can be
/// used for queries, e.g., to obtain the list of operations on which the
/// transformation represented by the current op is targeted. Returns a
/// special status object indicating whether the transformation succeeded
/// or failed, and, if it failed, whether the failure is recoverable or not.
/// 
/// IR must be created, modified and deleted with the provided rewriter.
/// implementations are responsible for setting the insertion point of the
/// rewriter to the desired location.
::mlir::DiagnosedSilenceableFailure mlir::transform::MatchOpInterface::apply(::mlir::transform::TransformRewriter & rewriter, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
      return getImpl()->implTransformOpInterface->apply(getImpl()->implTransformOpInterface, getOperation(), rewriter, transformResults, state);
  }
/// Indicates whether the op instance allows its handle operands to be
/// associated with the same payload operations.
bool mlir::transform::MatchOpInterface::allowsRepeatedHandleOperands() {
      return getImpl()->implTransformOpInterface->allowsRepeatedHandleOperands(getImpl()->implTransformOpInterface, getOperation());
  }
