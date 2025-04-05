#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class FunctionOpInterface;
class ModuleOp;
class RewritePatternSet;
class OpBuilder;
class SymbolTable;

namespace func {
class FuncOp;
} // namespace func

namespace bufferization {
struct OneShotBufferizationOptions;

/// Maps from symbol table to its corresponding dealloc helper function.
using DeallocHelperMap = llvm::DenseMap<Operation *, func::FuncOp>;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"

/// Creates an instance of the BufferDeallocation pass to free all allocated
/// buffers.
std::unique_ptr<Pass> createBufferDeallocationPass();

/// Creates an instance of the OwnershipBasedBufferDeallocation pass to free all
/// allocated buffers.
std::unique_ptr<Pass> createOwnershipBasedBufferDeallocationPass(
    DeallocationOptions options = DeallocationOptions());

/// Creates a pass that finds all temporary allocations
/// and attempts to move the deallocation after the last user/dependency 
/// of the allocation, thereby optimizing allocation liveness.
std::unique_ptr<Pass> createOptimizeAllocationLivenessPass();

/// Creates a pass that optimizes `bufferization.dealloc` operations. For
/// example, it reduces the number of alias checks needed at runtime using
/// static alias analysis.
std::unique_ptr<Pass> createBufferDeallocationSimplificationPass();

/// Creates an instance of the LowerDeallocations pass to lower
/// `bufferization.dealloc` operations to the `memref` dialect.
std::unique_ptr<Pass> createLowerDeallocationsPass();

/// Adds the conversion pattern of the `bufferization.dealloc` operation to the
/// given pattern set for use in other transformation passes.
void populateBufferizationDeallocLoweringPattern(
    RewritePatternSet &patterns, const DeallocHelperMap &deallocHelperFuncMap);

/// Construct the library function needed for the fully generic
/// `bufferization.dealloc` lowering implemented in the LowerDeallocations pass.
/// The function can then be called at bufferization dealloc sites to determine
/// aliasing and ownership.
///
/// The generated function takes two memrefs of indices and three memrefs of
/// booleans as arguments:
///   * The first argument A should contain the result of the
///     extract_aligned_pointer_as_index operation applied to the memrefs to be
///     deallocated
///   * The second argument B should contain the result of the
///     extract_aligned_pointer_as_index operation applied to the memrefs to be
///     retained
///   * The third argument C should contain the conditions as passed directly
///     to the deallocation operation.
///   * The fourth argument D is used to pass results to the caller. Those
///     represent the condition under which the memref at the corresponding
///     position in A should be deallocated.
///   * The fifth argument E is used to pass results to the caller. It
///     provides the ownership value corresponding the the memref at the same
///     position in B
///
/// This helper function is supposed to be called once for each
/// `bufferization.dealloc` operation to determine the deallocation need and new
/// ownership indicator for the retained values, but does not perform the
/// deallocation itself.
///
/// Generated code:
/// ```
/// func.func @dealloc_helper(
///     %dyn_dealloc_base_pointer_list: memref<?xindex>,
///     %dyn_retain_base_pointer_list: memref<?xindex>,
///     %dyn_cond_list: memref<?xi1>,
///     %dyn_dealloc_cond_out: memref<?xi1>,
///     %dyn_ownership_out: memref<?xi1>) {
///   %c0 = arith.constant 0 : index
///   %c1 = arith.constant 1 : index
///   %true = arith.constant true
///   %false = arith.constant false
///   %num_dealloc_memrefs = memref.dim %dyn_dealloc_base_pointer_list, %c0
///   %num_retain_memrefs = memref.dim %dyn_retain_base_pointer_list, %c0
///   // Zero initialize result buffer.
///   scf.for %i = %c0 to %num_retain_memrefs step %c1 {
///     memref.store %false, %dyn_ownership_out[%i] : memref<?xi1>
///   }
///   scf.for %i = %c0 to %num_dealloc_memrefs step %c1 {
///     %dealloc_bp = memref.load %dyn_dealloc_base_pointer_list[%i]
///     %cond = memref.load %dyn_cond_list[%i]
///     // Check for aliasing with retained memrefs.
///     %does_not_alias_retained = scf.for %j = %c0 to %num_retain_memrefs
///         step %c1 iter_args(%does_not_alias_aggregated = %true) -> (i1) {
///       %retain_bp = memref.load %dyn_retain_base_pointer_list[%j]
///       %does_alias = arith.cmpi eq, %retain_bp, %dealloc_bp : index
///       scf.if %does_alias {
///         %curr_ownership = memref.load %dyn_ownership_out[%j]
///         %updated_ownership = arith.ori %curr_ownership, %cond : i1
///         memref.store %updated_ownership, %dyn_ownership_out[%j]
///       }
///       %does_not_alias = arith.cmpi ne, %retain_bp, %dealloc_bp : index
///       %updated_aggregate = arith.andi %does_not_alias_aggregated,
///                                       %does_not_alias : i1
///       scf.yield %updated_aggregate : i1
///     }
///     // Check for aliasing with dealloc memrefs in the list before the
///     // current one, i.e.,
///     // `fix i, forall j < i: check_aliasing(%dyn_dealloc_base_pointer[j],
///     // %dyn_dealloc_base_pointer[i])`
///     %does_not_alias_any = scf.for %j = %c0 to %i step %c1
///        iter_args(%does_not_alias_agg = %does_not_alias_retained) -> (i1) {
///       %prev_dealloc_bp = memref.load %dyn_dealloc_base_pointer_list[%j]
///       %does_not_alias = arith.cmpi ne, %prev_dealloc_bp, %dealloc_bp
///       %updated_alias_agg = arith.andi %does_not_alias_agg, %does_not_alias
///       scf.yield %updated_alias_agg : i1
///     }
///     %dealloc_cond = arith.andi %does_not_alias_any, %cond : i1
///     memref.store %dealloc_cond, %dyn_dealloc_cond_out[%i] : memref<?xi1>
///   }
///   return
/// }
/// ```
func::FuncOp buildDeallocationLibraryFunction(OpBuilder &builder, Location loc,
                                              SymbolTable &symbolTable);

/// Run buffer deallocation.
LogicalResult deallocateBuffers(Operation *op);

/// Run the ownership-based buffer deallocation.
LogicalResult deallocateBuffersOwnershipBased(FunctionOpInterface op,
                                              DeallocationOptions options);

/// Creates a pass that moves allocations upwards to reduce the number of
/// required copies that are inserted during the BufferDeallocation pass.
std::unique_ptr<Pass> createBufferHoistingPass();

/// Creates a pass that moves allocations upwards out of loops. This avoids
/// reallocations inside of loops.
std::unique_ptr<Pass> createBufferLoopHoistingPass();

// Options struct for BufferResultsToOutParams pass.
// Note: defined only here, not in tablegen.
struct BufferResultsToOutParamsOpts {
  /// Memcpy function: Generate a memcpy between two memrefs.
  using MemCpyFn =
      std::function<LogicalResult(OpBuilder &, Location, Value, Value)>;

  // Filter function; returns true if the function should be converted.
  // Defaults to true, i.e. all functions are converted.
  llvm::function_ref<bool(func::FuncOp *)> filterFn = [](func::FuncOp *func) {
    return true;
  };

  /// Memcpy function; used to create a copy between two memrefs.
  /// If this is empty, memref.copy is used.
  std::optional<MemCpyFn> memCpyFn;

  /// If true, the pass adds a "bufferize.result" attribute to each output
  /// parameter.
  bool addResultAttribute = false;

  /// If true, the pass eliminates the memref.alloc and memcpy if the returned
  /// memref is allocated in the current function.
  bool hoistStaticAllocs = false;
};

/// Creates a pass that converts memref function results to out-params.
std::unique_ptr<Pass> createBufferResultsToOutParamsPass(
    const BufferResultsToOutParamsOpts &options = {});

/// Replace buffers that are returned from a function with an out parameter.
/// Also update all call sites.
LogicalResult
promoteBufferResultsToOutParams(ModuleOp module,
                                const BufferResultsToOutParamsOpts &options);

/// Creates a pass that drops memref function results that are equivalent to a
/// function argument.
std::unique_ptr<Pass> createDropEquivalentBufferResultsPass();

/// Create a pass that rewrites tensor.empty to bufferization.alloc_tensor.
std::unique_ptr<Pass> createEmptyTensorToAllocTensorPass();

/// Drop all memref function results that are equivalent to a function argument.
LogicalResult dropEquivalentBufferResults(ModuleOp module);

/// Creates a pass that finalizes a partial bufferization by removing remaining
/// bufferization.to_tensor and bufferization.to_memref operations.
std::unique_ptr<OperationPass<func::FuncOp>> createFinalizingBufferizePass();

/// Create a pass that bufferizes all ops that implement BufferizableOpInterface
/// with One-Shot Bufferize.
std::unique_ptr<Pass> createOneShotBufferizePass();

/// Create a pass that bufferizes all ops that implement BufferizableOpInterface
/// with One-Shot Bufferize and the specified bufferization options.
std::unique_ptr<Pass>
createOneShotBufferizePass(const OneShotBufferizationOptions &options);

/// Creates a pass that promotes heap-based allocations to stack-based ones.
/// Only buffers smaller than the provided size are promoted.
/// Dynamic shaped buffers are promoted up to the given rank.
std::unique_ptr<Pass>
createPromoteBuffersToStackPass(unsigned maxAllocSizeInBytes = 1024,
                                unsigned maxRankOfAllocatedMemRef = 1);

/// Creates a pass that promotes heap-based allocations to stack-based ones.
/// Only buffers smaller with `isSmallAlloc(alloc) == true` are promoted.
std::unique_ptr<Pass>
createPromoteBuffersToStackPass(std::function<bool(Value)> isSmallAlloc);

/// Create a pass that tries to eliminate tensor.empty ops that are anchored on
/// insert_slice ops.
std::unique_ptr<Pass> createEmptyTensorEliminationPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_PASSES_H
