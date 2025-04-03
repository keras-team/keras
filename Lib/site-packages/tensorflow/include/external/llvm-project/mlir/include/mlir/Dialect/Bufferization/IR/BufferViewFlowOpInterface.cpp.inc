/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Populate buffer dependencies between operands and op results/region
/// entry block arguments.
/// 
/// Implementations should register dependencies between an operand ("X")
/// and an op result/region entry block argument ("Y") if Y may depend
/// on X. Y depends on X if Y and X are the same buffer or if Y is a
/// subview of X.
/// 
/// Example:
/// ```
/// %r = arith.select %c, %m1, %m2 : memref<5xf32>
/// ```
/// In the above example, %0 may depend on %m1 or %m2 and a correct
/// interface implementation should call:
/// - "registerDependenciesFn(%m1, %r)".
/// - "registerDependenciesFn(%m2, %r)"
void mlir::bufferization::BufferViewFlowOpInterface::populateDependencies(::mlir::bufferization::RegisterDependenciesFn registerDependenciesFn) {
      return getImpl()->populateDependencies(getImpl(), getOperation(), registerDependenciesFn);
  }
/// Return "true" if the given value may be a terminal buffer. A buffer
/// value is "terminal" if it cannot be traced back any further in the
/// buffer view flow analysis.
/// 
/// Examples: A buffer could be terminal because:
/// - it is a newly allocated buffer (e.g., "memref.alloc"),
/// - or: because there is not enough compile-time information available
///   to make a definite decision (e.g., "memref.realloc" may reallocate
///   but we do not know for sure; another example are call ops where we
///   would have to analyze the body of the callee).
/// 
/// Implementations can assume that the given SSA value is an OpResult of
/// this operation or a region entry block argument of this operation.
bool mlir::bufferization::BufferViewFlowOpInterface::mayBeTerminalBuffer(Value value) {
      return getImpl()->mayBeTerminalBuffer(getImpl(), getOperation(), value);
  }
