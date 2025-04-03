//===- Transforms.h - Linalg transformations as patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H

#include <utility>

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace bufferization {
class AllocTensorOp;
class OneShotAnalysisState;
} // namespace bufferization

namespace linalg {

class LinalgOp;

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

/// Return vector::CombiningKind for the given op.
std::optional<vector::CombiningKind> getCombinerOpKind(Operation *combinerOp);

//===----------------------------------------------------------------------===//
// Bufferization-related transforms.
//===----------------------------------------------------------------------===//

struct BufferizeToAllocationOptions {
  enum class AllocOp { MemrefAlloc = 0, MemrefAlloca = 1 };
  AllocOp allocOp = AllocOp::MemrefAlloc;

  enum class MemcpyOp {
    MaterializeInDestination = 0,
    MemrefCopy = 1,
    LinalgCopy = 2
  };
  MemcpyOp memcpyOp = MemcpyOp::MaterializeInDestination;

  /// If set to "true", only the destination tensor operands are bufferized to
  /// a new allocation (and wrapped in "bufferization.to_tensor"), but not the
  /// targeted op itself.
  bool bufferizeDestinationOnly = false;

  /// If set to "true", a `memref.dealloc` operation will be emitted for each
  /// allocated buffer. Otherwise, the memory is leaked, which is useful if
  /// the buffer deallocation pipeline should be run after bufferization is
  /// done.
  bool emitDealloc = false;
};

/// Materialize a buffer allocation for the given tensor.pad op and lower the
/// op to linalg.fill/linalg.generic + bufferization.materialize_in_destination.
/// E.g.:
///
/// %0 = tensor.pad low[%l] high[%h] %t ...
///
/// is lowered to:
///
/// %alloc = memref.alloc
/// linalg.fill ... outs(%alloc)
/// %subview = memref.subview %alloc [%l] [...] [1]
/// bufferization.materialize_in_destination %t in %subview
/// %0 = bufferization.to_tensor %alloc restrict writable
///
/// In addition to rewriting the IR as shown above, this function returns the
/// newly allocated buffer. The `insertionPoint` parameter can be used to
/// specify a custom insertion point for the buffer allocation.
Value bufferizeToAllocation(RewriterBase &rewriter,
                            const BufferizeToAllocationOptions &options,
                            tensor::PadOp padOp, Attribute memorySpace = {},
                            Operation *insertionPoint = nullptr);

/// Materialize a buffer allocation for the given vector.mask op and bufferize
/// the op, including its region. E.g.:
///
/// %0 = vector.mask {
///   vector.transfer_write %v, %t : vector<16xf32>, tensor<?xf32>
/// } : vector<16xi1> -> tensor<?xf32>
///
/// is lowered to:
///
/// %alloc = memref.alloc
/// bufferization.materialize_in_destination %t in %subview
/// vector.mask {
///   vector.transfer_write %arg0, %alloc : vector<16xf32>, memref<?xf32>
/// } : vector<16xi1>
/// %0 = bufferization.to_tensor %alloc restrict writable
///
/// In addition to rewriting the IR as shown above, this function returns the
/// newly allocated buffer. The `insertionPoint` parameter can be used to
/// specify a custom insertion point for the buffer allocation.
Value bufferizeToAllocation(RewriterBase &rewriter,
                            const BufferizeToAllocationOptions &options,
                            vector::MaskOp maskOp, Attribute memorySpace = {},
                            Operation *insertionPoint = nullptr);

/// Materialize a buffer allocation for the given bufferization.alloc_tensor op
/// and lower the op to memref.alloc + memref.tensor_store.
///
/// In addition to rewriting the IR, this function returns the newly allocated
/// buffer. The `insertionPoint` parameter can be used to specify a custom
/// insertion point for the buffer allocation.
Value bufferizeToAllocation(RewriterBase &rewriter,
                            const BufferizeToAllocationOptions &options,
                            bufferization::AllocTensorOp allocTensorOp,
                            Attribute memorySpace = {},
                            Operation *insertionPoint = nullptr);

/// Bufferize the given op with tensor semantics and materialize the result in
/// a newly allocated buffer.
///
/// Only bufferizable ops that bufferize to a memory write or have an
/// aliasing OpOperand (and do not themselves bufferize to an allocation) are
/// supported. They are bufferized using their BufferizableOpInterface
/// implementation.
///
/// Selected ops that bufferize to an allocation (or need special handling) are
/// also supported:
/// - tensor.pad
/// - vector.mask
///
/// This function returns the newly allocated buffer. The `insertionPoint`
/// parameter can be used to specify a custom insertion point for the buffer
/// allocation.
Value bufferizeToAllocation(RewriterBase &rewriter,
                            const BufferizeToAllocationOptions &options,
                            Operation *op, Attribute memorySpace = {},
                            Operation *insertionPoint = nullptr);

/// Try to eliminate tensor::EmptyOps inside `op` that are anchored on a
/// LinalgOp. This transforms looks for LinalgOps that have an unused output
/// operand and an input operand that is rooted in a tensor::EmptyOp. The
/// tensor::EmptyOp uses are replaced with the output operand and the two
/// operands of the LinalgOp are swapped.
///
/// Example:
/// %0 = tensor.empty()
/// %1 = linalg.matmul ins(...) outs(%0)
/// %2 = linalg.generic ins(%1) outs(%dest) {
///   ^bb0(%in: f32, %out: f32):
///   // out not used
/// }
///
/// The IR is transformed as follows:
/// %0 = tensor.empty()
/// %1 = linalg.matmul ins(...) outs(%dest)
/// %2 = linalg.generic ins(%0) outs(%1) {
///   ^bb0(%in: f32, %out: f32):
///   // Use %out instead of %in
/// }
///
/// The "ins" operand has no uses inside the body of the LinalgOp and can be
/// folded away with existing cleanup patterns. Afterwards, the tensor::EmptyOp
/// can also fold away.
LogicalResult linalgOpAnchoredEmptyTensorEliminationStep(
    RewriterBase &rewriter, Operation *op,
    bufferization::OneShotAnalysisState &state);

//===----------------------------------------------------------------------===//
// Structs that configure the behavior of various transformations.
//===----------------------------------------------------------------------===//

using TileSizeComputationFunction =
    std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>;

struct LinalgTilingOptions {
  /// Computation function that returns the tile sizes for each operation.
  /// Delayed construction of constant tile sizes should occur to interoperate
  /// with folding.
  TileSizeComputationFunction tileSizeComputationFunction = nullptr;

  LinalgTilingOptions &
  setTileSizeComputationFunction(TileSizeComputationFunction fun) {
    tileSizeComputationFunction = std::move(fun);
    return *this;
  }
  /// Set the `tileSizeComputationFunction` to return the values `ts`. The
  /// values must not fold away when tiling. Otherwise, use a more robust
  /// `tileSizeComputationFunction`.
  LinalgTilingOptions &setTileSizes(const SmallVector<Value, 4> &ts) {
    tileSizeComputationFunction = [=](OpBuilder &, Operation *) { return ts; };
    return *this;
  }
  /// Convenience function to set the `tileSizeComputationFunction` to a
  /// function that computes tile sizes at the point they are needed. Allows
  /// proper interaction with folding.
  LinalgTilingOptions &setTileSizes(ArrayRef<int64_t> ts);

  /// Tile all dynamic dimensions by 1. I.e., scalarize those dimensions.
  /// Note: `scalarizeDynamicDims` and `setTileSizes` cannot be used together.
  LinalgTilingOptions &scalarizeDynamicDims();

  /// The interchange vector to reorder the tiled loops.
  SmallVector<unsigned, 4> interchangeVector = {};

  LinalgTilingOptions &setInterchange(ArrayRef<unsigned> interchange) {
    interchangeVector.assign(interchange.begin(), interchange.end());
    return *this;
  }

  /// The type of tile loops to generate.
  LinalgTilingLoopType loopType = LinalgTilingLoopType::Loops;

  LinalgTilingOptions &setLoopType(LinalgTilingLoopType lt) {
    loopType = lt;
    return *this;
  }

  /// When specified, specifies distribution of generated tile loops to
  /// processors.
  std::optional<LinalgLoopDistributionOptions> distribution;

  LinalgTilingOptions &
  setDistributionOptions(LinalgLoopDistributionOptions distributionOptions) {
    distribution = std::move(distributionOptions);
    return *this;
  }

  /// Specification markers of how to distribute the `linalg.tiled_loop`.
  SmallVector<StringRef, 2> distributionTypes = {};

  LinalgTilingOptions &setDistributionTypes(ArrayRef<StringRef> types) {
    distributionTypes.assign(types.begin(), types.end());
    return *this;
  }

  /// Peel the specified loops.
  SmallVector<int64_t> peeledLoops;

  LinalgTilingOptions &setPeeledLoops(ArrayRef<int64_t> loops) {
    peeledLoops.clear();
    peeledLoops.append(loops.begin(), loops.end());
    return *this;
  }
};

struct LinalgTilingAndFusionOptions {
  /// Tile sizes used to tile the root operation.
  SmallVector<int64_t> tileSizes;
  LinalgTilingAndFusionOptions &setTileSizes(ArrayRef<int64_t> ts) {
    tileSizes.assign(ts.begin(), ts.end());
    return *this;
  }
  /// Tile interchange used to permute the tile loops.
  SmallVector<int64_t> tileInterchange;
  /// When specified, specifies distribution of generated tile loops to
  /// processors.
  std::optional<LinalgLoopDistributionOptions> tileDistribution;
  LinalgTilingAndFusionOptions &
  setDistributionOptions(LinalgLoopDistributionOptions distributionOptions) {
    tileDistribution = std::move(distributionOptions);
    return *this;
  }
};

struct LinalgPaddingOptions {
  /// A padding value for every operand.
  SmallVector<Attribute> paddingValues;
  LinalgPaddingOptions &setPaddingValues(ArrayRef<Attribute> pv) {
    paddingValues.assign(pv.begin(), pv.end());
    return *this;
  }
  /// A list of iterator dimensions to pad.
  SmallVector<int64_t> paddingDimensions;
  LinalgPaddingOptions &setPaddingDimensions(ArrayRef<int64_t> pd) {
    paddingDimensions.assign(pd.begin(), pd.end());
    return *this;
  }
  /// A list of multiples to which each padding dimension should be padded to.
  std::optional<SmallVector<int64_t>> padToMultipleOf;
  LinalgPaddingOptions &setPadToMultipleOf(ArrayRef<int64_t> m) {
    padToMultipleOf.emplace(m.begin(), m.end());
    return *this;
  }
  /// A flag for every operand to mark the PadOp as nofold which enables
  /// packing for statically shaped operands.
  SmallVector<bool> packPaddings;
  LinalgPaddingOptions &setPackPaddings(ArrayRef<bool> pp) {
    packPaddings.assign(pp.begin(), pp.end());
    return *this;
  }
  /// A number of loops to hoist the PadOp out for every operand.
  SmallVector<int64_t> hoistPaddings;
  LinalgPaddingOptions &setHoistPaddings(ArrayRef<int64_t> hp) {
    hoistPaddings.assign(hp.begin(), hp.end());
    return *this;
  }
  /// A permutation vector for every operand used to transpose the packed
  /// PadOp results.
  SmallVector<SmallVector<int64_t>> transposePaddings;
  LinalgPaddingOptions &
  setTransposePaddings(ArrayRef<SmallVector<int64_t>> tp) {
    transposePaddings.assign(tp.begin(), tp.end());
    return *this;
  }
  enum class CopyBackOp : int8_t {
    None = 0,
    BufferizationMaterializeInDestination = 1,
    LinalgCopy = 2
  };
  /// The op to be used for copying the padded result to the original
  /// destination tensor.
  CopyBackOp copyBackOp = CopyBackOp::BufferizationMaterializeInDestination;
  LinalgPaddingOptions &setCopyBackOp(CopyBackOp op) {
    copyBackOp = op;
    return *this;
  }
};

/// Callback function type used to perform the allocation for the promoted
/// `subView`. In `boundingSubViewsize` a best attempt is made to find the
/// smallest constant value for the size of the buffer needed for each
/// dimension. If that is not possible, contains the dynamic size of the
/// subview. The call back should return the buffer to use.
using AllocBufferCallbackFn = std::function<std::optional<Value>(
    OpBuilder &b, memref::SubViewOp subView,
    ArrayRef<Value> boundingSubViewSize, DataLayout &layout)>;

/// Callback function type used to deallocate the buffers used to hold the
/// promoted subview.
using DeallocBufferCallbackFn =
    std::function<LogicalResult(OpBuilder &b, Value buffer)>;

/// Callback function type used to insert copy from original subview to
/// subview of the promoted region for the read operands/subview of promoted
/// region to original subview for the results. The copy has to happen from
/// `src` to `dst`.
using CopyCallbackFn =
    std::function<LogicalResult(OpBuilder &b, Value src, Value dst)>;

struct LinalgPromotionOptions {
  /// Indices of subViews to promote. If `std::nullopt`, try to promote all
  /// operands.
  std::optional<DenseSet<unsigned>> operandsToPromote;
  LinalgPromotionOptions &setOperandsToPromote(ArrayRef<int64_t> operands) {
    operandsToPromote = DenseSet<unsigned>();
    operandsToPromote->insert(operands.begin(), operands.end());
    return *this;
  }
  /// If ith element of `useFullTiles` is true the full view should be used
  /// for the promoted buffer of the ith operand in `operandsToPromote`.
  /// Otherwise the partial view will be used. The decision is defaulted to
  /// `useFullTileBuffersDefault` when `useFullTileBuffers` is std::nullopt and
  /// for operands missing from `useFullTileBuffers`.
  std::optional<llvm::SmallBitVector> useFullTileBuffers;
  LinalgPromotionOptions &setUseFullTileBuffers(ArrayRef<bool> useFullTiles) {
    unsigned size = useFullTiles.size();
    llvm::SmallBitVector tmp(size, false);
    for (unsigned i = 0; i < size; ++i)
      tmp[i] = useFullTiles[i];
    useFullTileBuffers = tmp;
    return *this;
  }
  /// If true all operands unspecified by `useFullTileBuffers` will use the
  /// full view, otherwise the partial view.
  bool useFullTileBuffersDefault = false;
  LinalgPromotionOptions &setUseFullTileBuffersByDefault(bool use) {
    useFullTileBuffersDefault = use;
    return *this;
  }
  /// Alignment of promoted buffer. If `std::nullopt` do not specify alignment.
  std::optional<unsigned> alignment;
  LinalgPromotionOptions &setAlignment(unsigned align) {
    alignment = align;
    return *this;
  }
  /// Memory space of promoted buffer. If `std::nullopt` do not specify memory
  /// space.
  std::optional<Attribute> memorySpace;
  LinalgPromotionOptions &setMemorySpace(Attribute memorySpc) {
    memorySpace = memorySpc;
    return *this;
  }
  /// Use alloca with the default allocation scheme.
  bool useAlloca = false;
  LinalgPromotionOptions &setUseAlloca(bool use) {
    useAlloca = use;
    return *this;
  }
  /// Callback function to do the allocation of the promoted buffer. If
  /// std::nullopt, then the default allocation scheme of allocating a
  /// memref<?xi8> buffer followed by a view operation is used.
  std::optional<AllocBufferCallbackFn> allocationFn;
  std::optional<DeallocBufferCallbackFn> deallocationFn;
  LinalgPromotionOptions &
  setAllocationDeallocationFns(AllocBufferCallbackFn const &allocFn,
                               DeallocBufferCallbackFn const &deallocFn) {
    allocationFn = allocFn;
    deallocationFn = deallocFn;
    return *this;
  }
  /// Callback function to do the copy of data to and from the promoted
  /// subview. If std::nullopt then a memref.copy is used.
  std::optional<CopyCallbackFn> copyInFn;
  std::optional<CopyCallbackFn> copyOutFn;
  LinalgPromotionOptions &setCopyInOutFns(CopyCallbackFn const &copyIn,
                                          CopyCallbackFn const &copyOut) {
    copyInFn = copyIn;
    copyOutFn = copyOut;
    return *this;
  }
};

/// Split Reduction options.
struct SplitReductionOptions {
  // Ratio used to split the reduction dimension.  If the ratio is <= 1,
  // nothing will be done.
  int64_t ratio = 0;
  // Index where the extra dimension is added to the intermediate tensor
  // shape.
  unsigned index = 0;
  // If the inner dimension after splitting is parallel or reduction.
  bool innerParallel = false;
};

/// Function signature to control reduction splitting. This returns
/// `SplitReductionOptions`.
// TODO: don't use unsigned unless doing bit manipulation.
using ControlSplitReductionFn =
    std::function<SplitReductionOptions(LinalgOp op)>;

//===----------------------------------------------------------------------===//
// Preconditions that ensure the corresponding transformation succeeds and can
// be applied as a rewrite pattern.
//===----------------------------------------------------------------------===//

/// Return true if two `linalg.generic` operations with producer/consumer
/// relationship through `fusedOperand` can be fused using elementwise op
/// fusion.
bool areElementwiseOpsFusable(OpOperand *fusedOperand);

/// Promote memref.subviews feeding linalg-on-buffers operations.
LogicalResult promoteSubviewsPrecondition(Operation *op,
                                          LinalgPromotionOptions options);

/// Return success if the operation can be vectorized.
LogicalResult vectorizeOpPrecondition(Operation *op,
                                      ArrayRef<int64_t> inputVectorSizes = {},
                                      ArrayRef<bool> inputScalableVecDims = {},
                                      bool vectorizeNDExtract = false,
                                      bool flatten1DDepthwiseConv = false);

//===----------------------------------------------------------------------===//
// Transformations exposed as functional-style API calls.
//===----------------------------------------------------------------------===//

using LinalgLoops = SmallVector<Operation *, 4>;

/// Transformation to drop unit-extent dimensions from `linalg.generic`
/// operations.
struct ControlDropUnitDims {
  enum class RankReductionStrategy { ReassociativeReshape, ExtractInsertSlice };

  RankReductionStrategy rankReductionStrategy =
      RankReductionStrategy::ReassociativeReshape;

  using ControlFnTy = std::function<SmallVector<unsigned>(Operation *)>;
  ControlFnTy controlFn = [](Operation *op) {
    if (auto genericOp = dyn_cast_or_null<GenericOp>(op)) {
      return llvm::to_vector(llvm::seq<unsigned>(0, genericOp.getNumLoops()));
    }
    if (auto padOp = dyn_cast_or_null<tensor::PadOp>(op)) {
      return llvm::to_vector(
          llvm::seq<unsigned>(0, padOp.getSourceType().getRank()));
    }
    return SmallVector<unsigned>{};
  };
};
struct DropUnitDimsResult {
  linalg::GenericOp resultOp;
  SmallVector<Value> replacements;
};
FailureOr<DropUnitDimsResult> dropUnitDims(RewriterBase &rewriter,
                                           GenericOp genericOp,
                                           const ControlDropUnitDims &options);

/// Fuse two `linalg.generic` operations that have a producer-consumer
/// relationship captured through `fusedOperand`. The method expects
/// that `areElementwiseOpsFusable` returns true for the given `fusedOperand`.
struct ElementwiseOpFusionResult {
  Operation *fusedOp;
  llvm::DenseMap<Value, Value> replacements;
};
FailureOr<ElementwiseOpFusionResult>
fuseElementwiseOps(RewriterBase &rewriter, OpOperand *fusedOperand);

/// Returns a set of indices of the producer's results which would
/// be preserved after the fusion.
/// * There is a chance that the implementation of the transformation does not
/// agree with the result of this method. This function gives a prediction based
/// on an optimized fusion.
llvm::SmallDenseSet<int> getPreservedProducerResults(GenericOp producer,
                                                     GenericOp consumer,
                                                     OpOperand *fusedOperand);

/// Try to peel and canonicalize loop `op` and return the new result.
/// Also applies affine_min/max bounds simplification on the fly where relevant.
// TODO: Add support for scf.parallel and affine.for loops.
SmallVector<Value> peelLoop(RewriterBase &rewriter, Operation *op);

/// Peel 'loops' and applies affine_min/max bounds simplification on the fly
/// where relevant.
void peelLoops(RewriterBase &rewriter, ArrayRef<scf::ForOp> loops);

/// Pad the iterator dimensions `paddingDimensions` of all `opToPad` operands
/// to a static bounding box. The original `opToPad` is cloned and operates on
/// the padded tensors.
///
/// * "options.padToMultipleOf" indicates that each padding dimension should be
///   padded to the specified multiple.
/// * Use "options.paddingValues" and "options.packPaddings" to set padding
///   value and nofold attribute of the created tensor::PadOps, respectively.
/// * The unpadded results (extracted slice of the cloned operation) are
///   returned via `replacements`.
/// * The tensor::PadOps are returned via `padOps`.
/// * "options.copyBackOp" specifies the op type for copying back the unpadded
///   result to the original destination tensor.
LogicalResult rewriteAsPaddedOp(RewriterBase &rewriter, LinalgOp opToPad,
                                const LinalgPaddingOptions &options,
                                LinalgOp &paddedOp,
                                SmallVector<Value> &replacements,
                                SmallVector<tensor::PadOp> &padOps);

namespace detail {

/// Helper struct to hold the results of building a packing loop nest.
struct PackingResult {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  SmallVector<Value> clonedLoopIvs, leadingPackedTensorIndexings;
  GenericOp maybeTransposeOp;
  tensor::PadOp hoistedPadOp;
};

/// Build the packing loop nest required to hoist `opToHoist` above
/// `outermostEnclosingForOp`.
/// The loop nest is built just before `outermostEnclosingForOp`.
FailureOr<PackingResult>
buildPackingLoopNest(RewriterBase &rewriter, tensor::PadOp opToHoist,
                     scf::ForOp outermostEnclosingForOp,
                     ArrayRef<int64_t> transposeVector);

} // namespace detail

/// Mechanically hoist padding operations on tensors by `numLoops` into a new,
/// generally larger tensor. This achieves packing of multiple padding ops into
/// a larger tensor. On success, `opToHoist` is replaced by the cloned version
/// in the packing loop so the caller can continue reasoning about the padding
/// operation. If `transposeVector` is non-empty, hoist padding introduces a
/// GenericOp to transpose the padded tensor before inserting it into the packed
/// tensor. A `transposeVector` can change the storage order of the padded
/// tensor but does not change the order of the pack or compute loops.
///
/// TODO: In the future, we should consider rewriting as a tensor.pack after
/// hoisting since this abstraction is now available.
///
/// Example in pseudo-mlir:
/// =======================
///
/// If hoistPaddingOnTensors is called with `nLoops` = 2 on the following IR.
/// ```
///    scf.for (%i, %j, %k)
///      %st0 = tensor.extract_slice f(%i, %k) : ... to tensor<?x?xf32>
///      %0 = tensor.pad %st0 low[0, 0] high[...] {
///      ^bb0( ... ):
///        linalg.yield %pad
///      } : tensor<?x?xf32> to tensor<4x8xf32>
///      compute(%0)
/// ```
///
/// IR resembling the following is produced:
///
/// ```
///    scf.for (%i) {
///      %packed_init = tensor.empty range(%j) : tensor<?x4x8xf32>
///      %packed = scf.for (%k) iter_args(%p : %packed_init) {
///        %st0 = tensor.extract_slice f(%i, %k) : ... to tensor<?x?xf32>
///        %0 = tensor.pad %st0 low[0, 0] high[...] {
///        ^bb0( ... ):
///          linalg.yield %pad
///        } : tensor<?x?xf32> to tensor<4x8xf32>
///        %1 = tensor.insert_slice %0 ...
///            : tensor<4x8xf32> to tensor<?x4x8xf32>
///        scf.yield %1: tensor<?x4x8xf32>
///      } -> tensor<?x4x8xf32>
///      scf.for (%j, %k) {
///        %st0 = tensor.extract_slice %packed [%k, 0, 0][1, 4, 8][1, 1, 1] :
///                 tensor<?x4x8xf32> to tensor<4x8xf32>
///        compute(%st0)
///      }
///    }
/// ```
FailureOr<Value>
hoistPaddingOnTensors(RewriterBase &rewriter, tensor::PadOp opToHoist,
                      int64_t numLoops, ArrayRef<int64_t> transposeVector,
                      tensor::PadOp &hoistedOp,
                      SmallVectorImpl<GenericOp> &transposeOps);
/// Calls into `hoistPaddingOnTensors` with a local IRRewriter.
FailureOr<Value>
hoistPaddingOnTensors(tensor::PadOp opToHoist, int64_t numLoops,
                      ArrayRef<int64_t> transposeVector,
                      tensor::PadOp &hoistedOp,
                      SmallVectorImpl<GenericOp> &transposeOps);

/// Apply padding and hoisting to `linalgOp` according to the configuration
/// specified in `options`.
FailureOr<LinalgOp> padAndHoistLinalgOp(RewriterBase &rewriter,
                                        LinalgOp linalgOp,
                                        const LinalgPaddingOptions &options);

/// Split the given `op` into two parts along the given iteration space
/// `dimension` at the specified `splitPoint`, and return the two parts.
/// If the second part is statically known to be empty, do not create it
/// and return nullptr instead. Error state is signalled by returning
/// a pair of nullptrs.
///
/// For example, the following op:
///
///   linalg.matmul ins(%0, %1 : tensor<128x32xf32>, tensor<32x64xf32>)
///                 outs(%2 : tensor<128x64xf32>)
///
/// split along the first dimension at position 42 will result in:
///
///   %3 = tensor.extract_slice %0[0, 0][42, 32][1, 1]
///   %4 = tensor.extract_slice %2[0, 0][42, 64][1, 1]
///   %5 = linalg.matmul ins(%3, %1 : tensor<42x32xf32>, tensor<32x64xf32>)
///                      outs(%5 : tensor<42x64xf32>)
///   %6 = tensor.insert_slice %5 into %2[0, 0][42, 64][1, 1]
///
///   %7 = tensor.extract_slice %0[42, 0][86, 32][1, 1]
///   %8 = tensor.extract_slice %6[42, 0][86, 64][1, 1]
///   %9 = linalg.matmul ins(%7, %1 : tensor<86x32xf32>, tensor<32x64xf32>)
///                      outs(%8 : tensor<86x64xf32>)
///   tensor.insert_slice %5 into %6[42, 0][86, 64][1, 1]
///
/// Note that there is no simplification other than constant propagation applied
/// to slice extraction and insertion.
std::pair<TilingInterface, TilingInterface> splitOp(RewriterBase &rewriter,
                                                    TilingInterface op,
                                                    unsigned dimension,
                                                    OpFoldResult splitPoint);

/// Perform standalone tiling of a single LinalgOp by `tileSizes`.
/// and permute the loop nest according to `interchangeVector`
/// The permutation is expressed as a list of integers that specify
/// the new ordering of the loop nest. The length of `interchangeVector`
/// must be equal to the length of `tileSizes`.
/// An empty vector is interpreted as the identity permutation and the
/// transformation returns early.
///
/// Return a struct containing the tiled loops in the specified order
/// and the cloned op if successful, std::nullopt otherwise.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed by
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
struct TiledLinalgOp {
  LinalgOp op;
  SmallVector<Operation *, 8> loops;
  SmallVector<Value, 4> tensorResults;
};
FailureOr<TiledLinalgOp> tileLinalgOp(RewriterBase &b, LinalgOp op,
                                      const LinalgTilingOptions &options);

/// Interchange the `iterator_types` and `iterator_maps` dimensions and adapts
/// the index accesses of `op`. This is an in-place transformation controlled
/// by `interchangeVector`. An empty vector is interpreted as the identity
/// permutation and the transformation returns early.
///
/// E.g. the permutation `(i,j,k) -> (j,k,i)` is expressed with
/// `interchangeVector = [1,2,0]`. All values in `interchangeVector` must be
/// integers, in the range 0..`op.rank` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation).
///
/// Return failure if the permutation is not valid.
FailureOr<GenericOp> interchangeGenericOp(RewriterBase &rewriter,
                                          GenericOp genericOp,
                                          ArrayRef<unsigned> interchangeVector);

/// Create a GenericOp from the given named operation `namedOp` and replace
/// namedOp.
/// Return failure if `namedOp` is a GenericOp or misses a region builder.
FailureOr<GenericOp> generalizeNamedOp(RewriterBase &rewriter,
                                       LinalgOp namedOp);

/// Create a namedOp from the given GenericOp and replace the GenericOp.
/// Currently we can specialize only trivial linalg copy operations.
FailureOr<LinalgOp> specializeGenericOp(RewriterBase &rewriter,
                                        GenericOp genericOp);

/// Create a new buffer using the `allocationFn` provided. The size of this
/// buffer is the smallest constant bounding size along each dimension that
/// can be computed for the size of the result of `subView`. Returns the
/// allocated buffer as `fullLocalView` and the view that matches the size of
/// the result of subview operation as `partialLocalView`.
struct PromotionInfo {
  Value fullLocalView;
  Value partialLocalView;
};
FailureOr<PromotionInfo>
promoteSubviewAsNewBuffer(OpBuilder &b, Location loc, memref::SubViewOp subView,
                          const AllocBufferCallbackFn &allocationFn,
                          DataLayout &layout);

/// Promote the `subViews` into a new buffer allocated at the insertion point
/// `b`. Promotion occurs in 3 steps:
///   1. Create a new buffer for a full tile (i.e. not clipped at the
///   boundary).
///   2. Take a full view on the buffer.
///   3. Take a partial slice of the full view in step 2. and copy into it.
///
/// Return the modified linalg op (the modification happens in place) as well
/// as all the copy ops created.
FailureOr<LinalgOp> promoteSubViews(OpBuilder &b, LinalgOp op,
                                    const LinalgPromotionOptions &options);

/// Allocate the subview in the GPU workgroup memory.
std::optional<Value> allocateWorkgroupMemory(OpBuilder &builder,
                                             memref::SubViewOp subview,
                                             ArrayRef<Value> sizeBounds,
                                             DataLayout &);

/// In case of GPU group memory there is no need to deallocate.
LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value /*buffer*/);

/// Create Memref copy operations and add gpu barrier guards before and after
/// the copy operation to ensure data integrity.
LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst);

/// Allocate the subview in the GPU private memory.
std::optional<Value> allocateGPUPrivateMemory(OpBuilder &builder,
                                              memref::SubViewOp subview,
                                              ArrayRef<Value> sizeBounds,
                                              DataLayout &);

/// Normal copy to between src and dst.
LogicalResult copyToGPUPrivateMemory(OpBuilder &b, Value src, Value dst);

/// In case of GPU private memory there is no need to deallocate since the
/// memory is freed when going outside of the scope.
LogicalResult deallocateGPUPrivateMemory(OpBuilder &, Value /*buffer*/);

/// Emit a suitable vector form for an operation. If provided,
/// `inputVectorSizes` are used to vectorize this operation. `inputVectorSizes`
/// must match the rank of the iteration space of the operation and the sizes
/// must be smaller or equal than their counterpart interation space sizes, if
/// static. `inputVectorShapes` also allows the vectorization of operations with
/// dynamic shapes.
LogicalResult vectorize(RewriterBase &rewriter, Operation *op,
                        ArrayRef<int64_t> inputVectorSizes = {},
                        ArrayRef<bool> inputScalableVecDims = {},
                        bool vectorizeNDExtract = false,
                        bool flatten1DDepthwiseConv = false);

/// Emit a suitable vector form for a Copy op with fully static shape.
LogicalResult vectorizeCopy(RewriterBase &builder, memref::CopyOp copyOp);

/// Emit a loop nest of `scf.for` with the proper body for `linalgOp`.
FailureOr<LinalgLoops> linalgOpToLoops(RewriterBase &rewriter,
                                       LinalgOp linalgOp);

/// Emit a loop nest of `scf.parallel` with the proper body for `linalgOp`.
FailureOr<LinalgLoops> linalgOpToParallelLoops(RewriterBase &rewriter,
                                               LinalgOp linalgOp);

/// Emit a loop nest of `affine.for` with the proper body for `linalgOp`.
FailureOr<LinalgLoops> linalgOpToAffineLoops(RewriterBase &rewriter,
                                             LinalgOp linalgOp);

/// Creates a number of ranges equal to the number of non-zero in `tileSizes`.
/// One for each loop of the LinalgOp that is tiled. The `tileSizes` argument
/// has one entry per surrounding loop. It uses zero as the convention that a
/// particular loop is not tiled. This convention simplifies implementations
/// by avoiding affine map manipulations. The returned ranges correspond to
/// the loop ranges, in the proper order, that are tiled and for which new
/// loops will be created. Also the function returns a map from loop indices
/// of the LinalgOp to the corresponding non-empty range indices of newly
/// created loops.
using LoopIndexToRangeIndexMap = DenseMap<int, int>;
std::tuple<SmallVector<Range, 4>, LoopIndexToRangeIndexMap>
makeTiledLoopRanges(RewriterBase &b, Location loc, AffineMap map,
                    ArrayRef<OpFoldResult> allShapeSizes,
                    ArrayRef<OpFoldResult> allTileSizes);

namespace detail {
template <typename T>
struct MultiSizeSpecificationBase {
  /// Tile sizes.
  T lowTileSize, highTileSize;
  /// Number of tiles associated with each size.
  T lowTripCount, highTripCount;
};

template <typename T>
struct ContinuousTileSizeSpecificationBase {
  /// Tile sizes.
  SmallVector<T> tileSizes;
  /// Number of tiles associated with each size.
  SmallVector<T> tripCounts;
};

} // namespace detail

/// A description of a multi-size tiling comprising tile sizes and numbers of
/// tiles, expressed as Values which may or may not be constant. Multi-size
/// currently means two-size.
struct MultiSizeSpecification
    : public detail::MultiSizeSpecificationBase<Value> {};
struct StaticMultiSizeSpecification
    : public detail::MultiSizeSpecificationBase<int64_t> {};

struct ContinuousTileSizeSpecification
    : public detail::ContinuousTileSizeSpecificationBase<Value> {};
struct StaticContinuousTileSizeSpecification
    : public detail::ContinuousTileSizeSpecificationBase<int64_t> {};

/// Emits the IR computing the multi-sized tiling specification with two tile
/// sizes not exceeding `targetSize`, each divisible by `sizeDivisor`, such
/// that there exist numbers of tiles with these sizes that fully cover the
/// given iteration space `dimension` of the structured `op`.
///
/// The computation is as follows:
///
///   b = originalTripCount floordiv sizeDivisor
///   t = (targetSize + sizeDivisor - 1) floordiv sizeDivisor
///   d = (b + t - 1) floordiv t
///   s = (b floordiv d) * sizeDivisor
///   v = b % d
///   u = d - v
///
/// where the tile sizes are `s` and `s` + `sizeDivisor`, and the numbers of
/// the corresponding tiles are `u` and `v`, respectively.  Alternatively,
///
///   s * u + (s + sizeDivisor) * v == original size,
///   where s mod sizeDivisor = 0.
///
/// Expects all values to be positive. In some cases with the target tile size
/// sufficiently close to the dimension shape and non-unit divisor, it is
/// impossible to compute such sizes. If `emitAssertion` is set, also emit the
/// assertion that size computation succeeded.
///
/// Returns the specification consisting of both tile values and the number of
/// tiles of each size.
FailureOr<MultiSizeSpecification>
computeMultiTileSizes(OpBuilder &builder, LinalgOp op, unsigned dimension,
                      OpFoldResult targetSize, OpFoldResult divisor,
                      bool emitAssertions = true);
FailureOr<StaticMultiSizeSpecification>
computeStaticMultiTileSizes(LinalgOp op, unsigned dimension, int64_t targetSize,
                            int64_t divisor);

FailureOr<StaticContinuousTileSizeSpecification>
computeStaticContinuousTileSizes(LinalgOp op, unsigned dimension,
                                 unsigned targetSize);
FailureOr<ContinuousTileSizeSpecification>
computeContinuousTileSizes(OpBuilder &builder, TilingInterface op,
                           unsigned dimension, OpFoldResult targetSize,
                           bool emitAssertions);

/// Transformation information returned after reduction tiling.
struct ForallReductionTilingResult {
  /// The partial reduction tiled op generated.
  SmallVector<Operation *> parallelTiledOps;
  /// The final reduction operation merging all the partial reductions.
  SmallVector<Operation *> mergeOps;
  /// Initial values used for partial reductions.
  SmallVector<Value> initialValues;
  /// The `scf.forall` operation that iterate over the tiles.
  scf::ForallOp loops;
};

/// Method to tile a reduction to parallel iterations computing partial
/// reductions. After the loop all the partial reduction are merged into a final
/// reduction. For example for the following sequence
///
/// ```mlir
/// %0 = linalg.generic %in ["parallel", "reduction"]
///   : tensor<7x9xf32> -> tensor<7xf32>
/// ```
///
/// into:
///
/// ```mlir
/// %0 = linalg.fill ... : tensor<7x4xf32>
/// %1 = scf.forall (%iv) in (%c4) shared_outs(%arg0 = %0)
///   -> (tensor<7x4xf32>) {
///   %2 = tensor.extract_slice %arg3 : tensor<7x4xf32> to tensor<7xf32>
///   %3 = tensor.extract_slice %in : tensor<7x9xf32> -> tensor<7x?xf32>
///   %4 = linalg.generic %2, %3 ["parallel", "reduction"]
///     : tensor<7x?xf32> -> tensor<7xf32>
///   %5 = tensor.insert_slice %3, %arg0[0, %iv] : tensor<7x4xf32>
/// }
/// %6 = linalg.generic %1 ["parallel", "reduction"]
///   : tensor<7x4xf32> -> tensor<7xf32>
/// ```
FailureOr<ForallReductionTilingResult>
tileReductionUsingForall(RewriterBase &b, PartialReductionOpInterface op,
                         ArrayRef<OpFoldResult> numThreads,
                         ArrayRef<OpFoldResult> tileSizes = {},
                         std::optional<ArrayAttr> mapping = std::nullopt);

/// All indices returned by IndexOp should be invariant with respect to
/// tiling. Therefore, if an operation is tiled, we have to transform the
/// indices accordingly, i.e. offset them by the values of the corresponding
/// induction variables that are captured implicitly in the body of the op.
///
/// Example. `linalg.generic` before tiling:
///
/// #id_2d = (i, j) -> (i, j)
/// #pointwise_2d_trait = {
///   indexing_maps = [#id_2d, #id_2d],
///   iterator_types = ["parallel", "parallel"]
/// }
/// linalg.generic #pointwise_2d_trait %operand, %result {
///   ^bb0(%operand_in: f32, %result_in: f32):
///     %i = linalg.index 0 : index
///     %j = linalg.index 1 : index
///     <some operations that use %i, %j>
/// }: memref<50x100xf32>, memref<50x100xf32>
///
/// After tiling pass with tiles sizes 10 and 25:
///
/// #strided = (i, j)[s0, s1, s2] -> (i * s1 + s0 + j * s2)
///
/// %c1 = arith.constant 1 : index
/// %c0 = arith.constant 0 : index
/// %c25 = arith.constant 25 : index
/// %c10 = arith.constant 10 : index
/// operand_dim_0 = dim %operand, 0 : memref<50x100xf32>
/// operand_dim_1 = dim %operand, 1 : memref<50x100xf32>
/// scf.for %k = %c0 to operand_dim_0 step %c10 {
///   scf.for %l = %c0 to operand_dim_1 step %c25 {
///     %4 = memref.subview %operand[%k, %l][%c10, %c25][%c1, %c1]
///       : memref<50x100xf32> to memref<?x?xf32, #strided>
///     %5 = memref.subview %result[%k, %l][%c10, %c25][%c1, %c1]
///       : memref<50x100xf32> to memref<?x?xf32, #strided>
///     linalg.generic pointwise_2d_trait %4, %5 {
///     ^bb0(%operand_in: f32, %result_in: f32):
///       %i = linalg.index 0 : index
///       %j = linalg.index 1 : index
///       // Indices `k` and `l` are implicitly captured in the body.
///       %transformed_i = arith.addi %i, %k : index // index `i` is offset by
///       %k %transformed_j = arith.addi %j, %l : index // index `j` is offset
///       by %l
///       // Every use of %i, %j is replaced with %transformed_i,
///       %transformed_j <some operations that use %transformed_i,
///       %transformed_j>
///     }: memref<?x?xf32, #strided>, memref<?x?xf32, #strided>
///   }
/// }
///
/// TODO: Investigate whether mixing implicit and explicit indices
/// does not lead to losing information.
void transformIndexOps(RewriterBase &b, LinalgOp op,
                       SmallVectorImpl<Value> &ivs,
                       const LoopIndexToRangeIndexMap &loopIndexToRangeIndex);

/// Apply transformation to split the single linalg op reduction into a
/// parallel and reduction dimension. Then create a new linalg.generic op
/// doing the rest of the reduction. Return the new linalg op with an extra
/// parallel dimension or failure if the transformation didn't happen.
///
/// Example:
/// ```
///  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
///                                        affine_map<(d0) -> ()>],
///       iterator_types = ["reduction"]}
///  ins(%in : tensor<32xf32>)
///  outs(%out : tensor<f32>) {
///  ^bb0(%arg1: f32, %arg2: f32):
///    %y = arith.addf %arg1, %arg2 : f32
///    linalg.yield %y : f32
///  } -> tensor<f32>
/// ```
/// To:
/// ```
///  %cst = arith.constant 0.000000e+00 : f32
///  %0 = tensor.expand_shape %in [[0, 1]] : tensor<32xf32> into
///  tensor<4x8xf32> %1 = tensor.empty [4] : tensor<4xf32> %2 = linalg.fill
///  ins(%cst : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32> %3 =
///  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
///                                        affine_map<(d0, d1) -> (d0)>],
///    iterator_types = ["parallel", "reduction"]}
///    ins(%0 : tensor<4x8xf32>) outs(%2 : tensor<4xf32>) {
///    ^bb0(%arg3: f32, %arg5: f32):
///    %5 = arith.addf %arg3, %arg4 : f32
///    linalg.yield %5 : f32
///  } -> tensor<4xf32>
/// %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
///                                       affine_map<(d0) -> ()>],
///   iterator_types = ["reduction"]}
///   ins(%3 : tensor<4xf32>) outs(%out : tensor<f32>) {
///   ^bb0(%arg3: f32, %arg4: f32):
///   %5 = arith.addf %arg3, %arg4 : f32
///   linalg.yield %5 : f32
/// } -> tensor<f32>
/// ```
struct SplitReductionResult {
  Operation *initOrAlloc;
  FillOp fillOp;
  LinalgOp splitLinalgOp;
  LinalgOp resultCombiningLinalgOp;
};
FailureOr<SplitReductionResult>
splitReduction(RewriterBase &b, LinalgOp op,
               const ControlSplitReductionFn &controlSplitReductionFn,
               bool useAlloc = false);

/// Scaling-based implementation of the split reduction transformation.
/// Instead of introducing an ExpandShapeOp, this rewrites a reduction
/// dimension `k` into `k * scale + kk`.
///
/// Example:
/// ```
///  %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
///    outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
/// ```
///
/// Is transformed to:
///
/// ```
///  #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 4 + d3)>
///  #map1 = affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3, d1)>
///  #map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
///  #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
///  #map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
///  #map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
///  %0 = tensor.empty [16, 32, 64] : tensor<16x32x64xf32>
///  %cst = arith.constant 0.000000e+00 : f32
///  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32x64xf32>) ->
///     tensor<16x32x64xf32>
///  %2 = tensor.empty [64, 4] : tensor<64x4xi1>
///
///  %3 = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3],
///    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
///    ins(%A, %B, %2 : tensor<16x256xf32>, tensor<256x32xf32>,
///    tensor<64x4xi1>)
///   outs(%1 : tensor<16x32x64xf32>) {
///      ^bb0(%arg3: f32, %arg4: f32, %arg5: i1, %arg6: f32):
///        %5 = arith.mulf %arg3, %arg4 : f32
///        %6 = arith.addf %arg6, %5 : f32
///        linalg.yield %6 : f32
///  } -> tensor<16x32x64xf32>
///
///  %4 = linalg.generic {indexing_maps = [#map4, #map5],
///    iterator_types = ["parallel", "parallel", "reduction"]}
//     ins(%3 : tensor<16x32x64xf32>)
///    outs(%C : tensor<16x32xf32>) {
///      ^bb0(%arg3: f32, %arg4: f32):
///        %5 = arith.addf %arg3, %arg4 : f32
///        linalg.yield %5 : f32
///  } -> tensor<16x32xf32>
///
///  return %4 : tensor<16x32xf32>
/// ```
FailureOr<SplitReductionResult>
splitReductionByScaling(RewriterBase &b, LinalgOp op,
                        const ControlSplitReductionFn &controlSplitReductionFn,
                        bool useAlloc = false);

/// Return `true`  if a given sequence of dimensions are contiguous in the
/// range of the specified indexing map.
bool isDimSequencePreserved(AffineMap map, ReassociationIndicesRef dimSequence);
/// Return `true` if all sequences of dimensions specified in `dimSequences` are
/// contiguous in all the ranges of the `maps`.
bool areDimSequencesPreserved(ArrayRef<AffineMap> maps,
                              ArrayRef<ReassociationIndices> dimSequences);

struct CollapseResult {
  SmallVector<Value> results;
  LinalgOp collapsedOp;
};

/// Collapses dimensions of linalg.generic/linalg.copy operation. A precondition
/// to calling this method is that for each list in `foldedIterationDim`, the
/// sequence of dimensions is contiguous in domains of all `indexing_maps` of
/// the `linalgOp`. This can be checked using `areDimSequencePreserved` method.
/// When valid, the method also collapses the operands of the op. Returns
/// replacement values of the results of the original `linalgOp` by inserting
/// reshapes to get back values of compatible types.
FailureOr<CollapseResult>
collapseOpIterationDims(LinalgOp op,
                        ArrayRef<ReassociationIndices> foldedIterationDims,
                        RewriterBase &rewriter);

struct LowerPackResult {
  tensor::PadOp padOp;
  tensor::ExpandShapeOp expandShapeOp;
  linalg::TransposeOp transposeOp;
};

/// Rewrite pack as pad + reshape + transpose.
FailureOr<LowerPackResult> lowerPack(RewriterBase &rewriter,
                                     tensor::PackOp packOp);

struct LowerUnPackOpResult {
  tensor::EmptyOp emptyOp;
  linalg::TransposeOp transposeOp;
  tensor::CollapseShapeOp collapseShapeOp;
  tensor::ExtractSliceOp extractSliceOp;
};

/// Rewrite pack as empty + transpose + reshape + extract_slice.
FailureOr<LowerUnPackOpResult> lowerUnPack(RewriterBase &rewriter,
                                           tensor::UnPackOp unPackOp);

/// Struct to hold the result of a `pack` call.
struct PackResult {
  SmallVector<tensor::PackOp> packOps;
  linalg::LinalgOp packedLinalgOp;
  SmallVector<tensor::UnPackOp> unPackOps;
};
/// Implement packing of a single LinalgOp by `packedSizes`.
/// There must be one packedSizes entry per `linalgOp` iterator.
/// Return the packed Linalg op on success, failure otherwise.
FailureOr<PackResult> pack(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                           ArrayRef<OpFoldResult> packedSizes);

/// Struct to hold the result of a `packTranspose` call.
struct PackTransposeResult {
  tensor::PackOp transposedPackOp;
  linalg::LinalgOp transposedLinalgOp;
  tensor::UnPackOp transposedUnPackOp;
};
/// Transpose a single PackOp -> LinalgOp -> UnPackOp chain and return the
/// transposed PackOp -> LinalgOp -> UnPackOp chain after replacements.
/// Return failure if either:
///   1. the `packOp` does not have the `linalgOp` as its unique use.
///   2. the `maybeUnPackOp`, if specified must be a consumer of the result tied
///      to the unique `packOp` use.
///   3. `outerPerm` (resp. `innerPerm`) must be valid permutations of
///      `packOp.getOuterDimsPerm` (resp. `packOp.getInnerDimsPerm`) or empty.
FailureOr<PackTransposeResult>
packTranspose(RewriterBase &rewriter, tensor::PackOp packOp,
              linalg::LinalgOp linalgOp, tensor::UnPackOp maybeUnPackOp,
              ArrayRef<int64_t> outerPerm, ArrayRef<int64_t> innerPerm);

/// Pack a LinalgOp by greedily inferring matmul dimensions (m, n, k) where m
/// and n are proper parallel dimensions and k is a proper reduction
/// dimension. Packing occurs by rewriting the op as a linalg.generic and
/// calling linalg::pack by `mnkPackedSizes`. The order of the packed
/// dimensions is customizable: the `mnkOrder` is a permutation of {0, 1, 2}
/// to reorder {m, n, k} into one of the 8 possible forms. The outer
/// dimensions of the operands are not permuted at this time, this is left for
/// future work.
FailureOr<PackResult>
packMatmulGreedily(RewriterBase &rewriter, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> mnkPackedSizes,
                   ArrayRef<int64_t> mnkPaddedSizesNextMultipleOf,
                   ArrayRef<int64_t> mnkOrder);

struct BlockPackMatmulOptions {
  /// Minor block factors (mb, nb, kb) for packing relayout where mb, mn are
  /// the parallel dimensions and kb is the reduction dimension.
  SmallVector<int64_t, 3> blockFactors;

  /// If true, allows packing of dimensions that only partially fit into the
  /// block factors.
  bool allowPadding = true;

  /// Next multiples of the packing sizes.
  SmallVector<int64_t, 3> mnkPaddedSizesNextMultipleOf;

  /// Permutation of matmul (M, N, K) dimensions order.
  SmallVector<int64_t, 3> mnkOrder = {0, 1, 2};

  /// Transpose LHS outer block layout [MB][KB] -> [KB][MB].
  bool lhsTransposeOuterBlocks = false;

  /// Transpose LHS inner block layout [mb][kb] -> [kb][mb].
  bool lhsTransposeInnerBlocks = false;

  /// Transpose RHS outer block layout [KB][NB] -> [NB][KB].
  bool rhsTransposeOuterBlocks = true;

  /// Transpose RHS inner block layout [kb][nb] -> [nb][kb].
  bool rhsTransposeInnerBlocks = true;
};

/// Function type which is used to control matmul packing.
/// It is expected to return valid packing configuration for each operation.
/// Lack of packing options indicates that no valid configuration could be
/// assigned and the operation will not be packed.
using ControlBlockPackMatmulFn =
    std::function<std::optional<BlockPackMatmulOptions>(linalg::LinalgOp)>;

/// Pack a matmul operation into blocked 4D layout.
///
/// Relayout a matmul operation into blocked layout with two levels of
/// subdivision:
///   - major 2D blocks - outer dimensions, consist of minor blocks
///   - minor 2D blocks - inner dimensions, consist of scalar elements
///
/// A 2D matmul MxNxK gets reshaped into blocked 4D representation
/// as: [MB][NB][mb][nb] += [MB][KB][mb][kb] * [NB][KB][nb][kb]
/// where the (MB, NB, KB) dimensions represent the major blocks,
/// and the (mb, nb, kb) are the minor blocks of their respective
/// original 2D dimensions (M, N, K).
///
/// Depending on the initial operands' data layout and the specified
/// packing options, the major blocks dimensions might get transposed
/// e.g., [MB][KB] -> [KB][MB]. The minor blocks can also be transposed
/// e.g., [mb][kb] -> [kb][mb].
/// Any present batch dimensions remain unchanged.
/// The final result is unpacked back to the original shape.
///
/// Return failure if no valid packing options are provided.
FailureOr<PackResult>
blockPackMatmul(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                const ControlBlockPackMatmulFn &controlPackMatmul);

/// Rewrite tensor.from_elements to linalg.generic.
FailureOr<Operation *>
rewriteInDestinationPassingStyle(RewriterBase &rewriter,
                                 tensor::FromElementsOp fromElementsOp);

/// Rewrite tensor.generate to linalg.generic.
FailureOr<Operation *>
rewriteInDestinationPassingStyle(RewriterBase &rewriter,
                                 tensor::GenerateOp generateOp);

/// Rewrite tensor.pad to linalg.generic + tensor.insert_slice.
FailureOr<Operation *> rewriteInDestinationPassingStyle(RewriterBase &rewriter,
                                                        tensor::PadOp padOp);

/// Convert linalg.conv_2d_nhwc_hwcf into linalg.generic (for img2col packing)
/// and linalg.matmul.
///
/// A convolution operation can be written as a matrix-matrix multiplication by
/// unfolding the cross-correlation between input and filter and explicitly copy
/// overlapped sliding window inputs.
///
/// Consider 2D input X with single channel input and output and 2x2 filter W:
/// [x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
/// [x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
/// [.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
/// [.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
/// [.        ,  .       ,   .,      .     ]
/// [x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
///
/// The packed input data (img2col) is a matrix with |rows| = output spatial
/// size, |columns| = filter spatial size. To compute the output Y(i, j) we need
/// to calculate the dot product between filter window at input X(x, y)) and the
/// filter which will look like the following where r.h.s is the img2col matrix
/// and l.h.s is the flattened filter:
///
/// [x(0,0), x(0,1), x(1,0), x(1,1)]
/// [x(0,1), x(1,1), x(0,2), x(1,2)] (matmul) [w(0,0), w(0,1), w(1,0), w(1,1)]
/// [x(0,1), x(1,1), x(0,2), x(1,2)]
/// [   .  ,    .  ,    .  ,    .  ]
///
/// In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
/// and output (N, Ho, Wo, D) the convolution is the following matrix-matrix
/// multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
/// the N input. For the case where N > 1 its a batched matrix-matrix
/// multiplication.
///
/// On success, return both the operation that produces the img2col tensor and
/// the final operation of the sequence that replaces the original convolution.
FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNhwcHwcfOp convOp);

/// Same as the above but for Fhwc channel orderings in the filter. In this case
/// the matrix multiplication is actually a row-wise dot-product rather than a
/// row-column dot-product. This is to avoid transposing the filter matrix which
/// would be required for a regular matrix multiplication to produce the correct
/// output dimensions.
FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNhwcFhwcOp convOp);

/// Similar to rewriteInIm2Col with linalg::Conv2DNhwcHwcfOp except there is no
/// reduction among the input channels so each convolution can be a
/// matrix-vector product and by transposing both input filter so channels are
/// outer most the computation is a batched matrix-vector product.
FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter,
                linalg::DepthwiseConv2DNhwcHwcOp convOp);

/// Similar to rewriteInIm2Col with linalg::Conv2DNhwcHwcfOp except because the
/// channels are to the left of the image shape dimensions, the position of the
/// contraction dimension in the resulting matmul is reversed. This swaps the
/// LHS and RHS of the matmul when compared with nhwc (i.e. (D, C x Kh x Kw) *
/// (C x Kh x Kw, Ho x Wo))
FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNchwFchwOp convOp);

/// Convert linalg.conv_2d_nhwc_fhwc(_q) to linalg.conv_2d_nhwc_hwcf(_q) by
/// materializing transpose.
FailureOr<Operation *> transposeConv2D(RewriterBase &rewriter,
                                       linalg::Conv2DNhwcFhwcOp op);
FailureOr<Operation *> transposeConv2D(RewriterBase &rewriter,
                                       linalg::Conv2DNhwcFhwcQOp op);

/// Convert Linalg matmul ops to transposed variants.
FailureOr<Operation *> transposeMatmul(RewriterBase &rewriter,
                                       linalg::MatmulOp op,
                                       bool transposeLHS = true);
FailureOr<Operation *> transposeBatchMatmul(RewriterBase &rewriter,
                                            linalg::BatchMatmulOp op,
                                            bool transposeLHS = true);

/// Convert linalg.conv_2d_nhwc_fhwc to Winograd Conv2D algorithm
/// F(m x m, r x r). m is the dimension size of output and r is the dimension
/// size of filter.
FailureOr<Operation *> winogradConv2D(RewriterBase &rewriter,
                                      linalg::Conv2DNhwcFhwcOp op, int64_t m,
                                      int64_t r);

/// Rewrite linalg.winograd_filter_transform. The data layout of the filter is
/// FHWC. The transformation matrix is 2-dimension. We need to extract H x W
/// from FHWC first. We generate 2 levels of loops to iterate on F and C. After
/// the rewriting, we get
///
/// scf.for %f = lo_f to hi_f step 1
///   scf.for %c = lo_c to hi_c step 1
///     %extracted = extract filter<h x w> from filter<f x h x w x c>
///     %ret = linalg.matmul G, %extracted
///     %ret = linalg.matmul %ret, GT
///     %inserted = insert %ret into filter<h x w x c x f>
FailureOr<Operation *>
decomposeWinogradFilterTransformOp(RewriterBase &rewriter,
                                   linalg::WinogradFilterTransformOp op);

/// Rewrite linalg.winograd_input_transform. The data layout of the input is
/// NHWC. The transformation matrix is 2-dimension. We need to extract H x W
/// from NHWC first. We generate 4 levels of loops to iterate on N, C, tileH,
/// and tileW. After the rewriting, we get
///
/// scf.for %h = 0 to tileH step 1
///   scf.for %w = 0 to tileW step 1
///     scf.for %n = 0 to N step 1
///       scf.for %c = 0 to C step 1
///         %extracted = extract %extracted<alphaH x alphaW> from
///                              %input<N x H x W x C>
///                              at [%n, (%h x m), (%w x m), %c]
///         %ret = linalg.matmul BT, %extracted
///         %ret = linalg.matmul %ret, B
///         %inserted = insert %ret<alphaH x alphaW> into
///                            %output<alphaH x alphaW x tileH x tileW x N x C>
///                            at [0, 0, %h, %w, %n, %c]
FailureOr<Operation *>
decomposeWinogradInputTransformOp(RewriterBase &rewriter,
                                  linalg::WinogradInputTransformOp op);

/// Rewrite linalg.winograd_output_transform. The data layout of the output is
/// HWNF. The transformation matrix is 2-dimension. We need to extract H x W
/// from HWNF first. We generate 4 levels of loops to iterate on N, F, tileH,
/// and tileW. After the transformation, we get
///
/// scf.for %h = 0 to tileH step 1
///   scf.for %w = 0 to tileW step 1
///     scf.for %n = 0 to N step 1
///       scf.for %f = 0 to F step 1
///         %extracted = extract %extracted<alphaH x alphaW> from
///                              %input<alphaH x alphaW x tileH x tileW x N x F>
///                              at [0, 0, %h, %w, %n, %f]
///         %ret = linalg.matmul AT, %extracted
///         %ret = linalg.matmul %ret, A
///         %inserted = insert %ret<alphaH x alphaW> into
///                            output<N x H x W x F>
///                            at [%n, (%h x m), (%w x m), %f]
FailureOr<Operation *>
decomposeWinogradOutputTransformOp(RewriterBase &rewriter,
                                   linalg::WinogradOutputTransformOp op);

//===----------------------------------------------------------------------===//
// Rewrite patterns wrapping transformations.
// TODO: every single such pattern should be a close to noop wrapper around a
// functional-stye API call.
//===----------------------------------------------------------------------===//

/// Rewrites 2-D convolution ops with size-1 window dimensions into 1-D
/// convolution ops.
template <typename Conv2DOp, typename Conv1DOp>
struct DownscaleSizeOneWindowed2DConvolution final
    : public OpRewritePattern<Conv2DOp> {
  using OpRewritePattern<Conv2DOp>::OpRewritePattern;

  FailureOr<Conv1DOp> returningMatchAndRewrite(Conv2DOp convOp,
                                               PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(Conv2DOp convOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(convOp, rewriter);
  }
};

extern template struct DownscaleSizeOneWindowed2DConvolution<Conv2DNhwcHwcfOp,
                                                             Conv1DNwcWcfOp>;
extern template struct DownscaleSizeOneWindowed2DConvolution<Conv2DNchwFchwOp,
                                                             Conv1DNcwFcwOp>;

/// Rewrites 2-D depthwise convolution ops with size-1 (w, kw) or (h, kh)
/// dimensions into 1-D depthwise convolution ops.
struct DownscaleDepthwiseConv2DNhwcHwcOp final
    : public OpRewritePattern<DepthwiseConv2DNhwcHwcOp> {
  DownscaleDepthwiseConv2DNhwcHwcOp(MLIRContext *context,
                                    PatternBenefit benefit = 1)
      : OpRewritePattern<DepthwiseConv2DNhwcHwcOp>(context, benefit) {}

  FailureOr<DepthwiseConv1DNwcWcOp>
  returningMatchAndRewrite(DepthwiseConv2DNhwcHwcOp convOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(DepthwiseConv2DNhwcHwcOp convOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(convOp, rewriter);
  }
};

struct DownscaleConv2DOp final : public OpRewritePattern<Conv2DOp> {
  DownscaleConv2DOp(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<Conv2DOp>(context, benefit) {}

  FailureOr<Conv1DOp> returningMatchAndRewrite(Conv2DOp convOp,
                                               PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(Conv2DOp convOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(convOp, rewriter);
  }
};

///
/// Linalg generalization pattern.
///
/// Apply the `generalization` transformation as a pattern.
/// See `generalization` for more details.
//
// TODO: Automatic default pattern class that just unwraps a function
// returning FailureOr<GenericOp>.
struct LinalgGeneralizationPattern
    : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  /// `matchAndRewrite` implementation that returns the significant
  /// transformed pieces of IR.
  FailureOr<GenericOp>
  returningMatchAndRewrite(LinalgOp op, PatternRewriter &rewriter) const {
    return generalizeNamedOp(rewriter, op);
  }

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }
};

struct LinalgSpecializationPattern : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  FailureOr<GenericOp>
  returningMatchAndRewrite(GenericOp op, PatternRewriter &rewriter) const {
    return specializeGenericOp(rewriter, op);
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }
};

/// Vectorization pattern for memref::CopyOp.
struct CopyVectorizationPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override;
};

using OptimizeCopyFn =
    std::function<LogicalResult(RewriterBase &, tensor::PadOp, Value)>;

/// Rewrite a tensor::PadOp into a sequence of EmptyOp, FillOp and
/// InsertSliceOp. For now, only constant padding values are supported.
/// `OptimizeCopyFn` can be used to customize copying step optimization.
struct GeneralizePadOpPattern : public OpRewritePattern<tensor::PadOp> {
  GeneralizePadOpPattern(MLIRContext *context,
                         OptimizeCopyFn optimizeCopyFn = nullptr,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::PadOp>(context, benefit),
        optimizeCopyFn(std::move(optimizeCopyFn)) {}
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override;

protected:
  OptimizeCopyFn optimizeCopyFn;
  Value createFillOrGenerateOp(RewriterBase &rewriter, tensor::PadOp padOp,
                               Value dest,
                               const SmallVector<Value> &dynSizes) const;
};

/// Rewrites a tensor::PackOp into a sequence of tensor.pad + linalg.transpose +
/// tensor.insert_slice ops, where the tensor::PackOp has outer dims being all
/// 1s.
struct GeneralizeOuterUnitDimsPackOpPattern
    : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override;
};

/// Rewrites a tensor::UnPackOp into a sequence of rank-reduced extract_slice op
/// + transpose op + insert_slice op, where the tensor::UnPackOp has outer dims
/// being all 1s.
struct GeneralizeOuterUnitDimsUnPackOpPattern
    : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override;
};

/// Match and rewrite for the pattern:
/// ```
///    %alloc = ...
///    [optional] %view = memref.view %alloc ...
///    %subView = subview %allocOrView ...
///    [optional] linalg.fill(%allocOrView, %cst) ...
///    ...
///    memref.copy(%in, %subView) ...
///    vector.transfer_read %allocOrView[...], %cst ...
/// ```
/// into
/// ```
///    [unchanged] %alloc = ...
///    [unchanged] [optional] %view = memref.view %alloc ...
///    [unchanged] [unchanged] %subView = subview %allocOrView ...
///    ...
///    vector.transfer_read %in[...], %cst ...
/// ```
/// Where there is no interleaved use between memref.copy and transfer_read as
/// well as no interleaved use between linalg.fill and memref.copy (if
/// linalg.fill is specified).
/// This is a custom rewrite to forward partial reads (with optional fills) to
/// vector.transfer_read.
struct LinalgCopyVTRForwardingPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override;
};

/// Match and rewrite for the pattern:
/// ```
///    %alloc = ...
///    [optional] %view = memref.view %alloc ...
///    %subView = subview %allocOrView...
///    ...
///    vector.transfer_write %..., %allocOrView[...]
///    memref.copy(%subView, %out)
/// ```
/// into
/// ```
///    [unchanged] %alloc = ...
///    [unchanged] [optional] %view = memref.view %alloc ...
///    [unchanged] %subView = subview %allocOrView...
///    ...
///    vector.transfer_write %..., %out[...]
/// ```
/// Where there is no interleaved use between transfer_write and memref.copy.
/// This is a custom rewrite to forward partial writes to
/// vector.transfer_write.
struct LinalgCopyVTWForwardingPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override;
};

/// Rewrite extract_slice(tensor.pad(x)) into tensor.pad(extract_slice(x)).
struct ExtractSliceOfPadTensorSwapPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  /// A function to control pattern application and rewrite logic.
  ///
  /// The function will be given the slice op and should return:
  /// -  std::nullopt: to fail the match and not apply the pattern;
  /// -  true: to apply the pattern with zero slice guard;
  /// - false: to apply the pattern without zero slice guard.
  ///
  /// See the documentation for tensor::bubbleUpPadSlice regarding zero slice
  /// guard.
  using ControlFn = std::function<std::optional<bool>(tensor::ExtractSliceOp)>;

  ExtractSliceOfPadTensorSwapPattern(MLIRContext *context,
                                     ControlFn controlFn = nullptr,
                                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override;

private:
  ControlFn controlFn;
};

//===----------------------------------------------------------------------===//
// Populate functions.
//===----------------------------------------------------------------------===//

/// Canonicalization patterns relevant to apply after tiling patterns. These
/// are applied automatically by the tiling pass but need to be applied
/// manually when tiling is called programmatically.
RewritePatternSet getLinalgTilingCanonicalizationPatterns(MLIRContext *ctx);
void populateLinalgTilingCanonicalizationPatterns(RewritePatternSet &patterns);

/// Linalg generalization patterns

/// Populates `patterns` with patterns to convert spec-generated named ops to
/// linalg.generic ops.
void populateLinalgNamedOpsGeneralizationPatterns(RewritePatternSet &patterns);

/// Populates `patterns` with patterns to convert linalg.generic ops to named
/// ops where possible. A linalg.generic can represent wide range and complex
/// computations for which equivalent linalg named op may not exist e.g.
/// linalg.generic that takes a tensor and computes a polynomial such as:
///     p(x) = an*x^n + ... + a1x + a0
/// There is no equivalent named op to convert to. Many such cases exist.
void populateLinalgGenericOpsSpecializationPatterns(
    RewritePatternSet &patterns);

/// Linalg decompose convolutions patterns

/// Populates patterns to decompose high-D convolution ops into low-D ones.
/// This is a step in progressive lowering for convolution ops, afterwards we
/// can vectorize the low-D convolution ops.
void populateDecomposeConvolutionPatterns(RewritePatternSet &patterns,
                                          PatternBenefit benefit = 1);

/// Populates patterns to transform linalg.conv_2d_xxx operations into
/// linalg.generic (for img2col packing) and linalg.matmul.
/// \see rewriteInIm2Col for more details.
void populateConvertConv2DToImg2ColPatterns(RewritePatternSet &patterns);

/// Populates `patterns` with patterns that vectorize tensor.pad.
/// These patterns are meant to apply in a complementary fashion. Benefits
/// are used to encode a certain ordering of pattern application. To avoid
/// scattering magic constants throughout the code base, the patterns must be
/// added with this function. `baseBenefit` can be used to offset the benefit
/// of all tensor::PadOp vectorization patterns by a certain value.
void populatePadOpVectorizationPatterns(RewritePatternSet &patterns,
                                        PatternBenefit baseBenefit = 1);

/// Populate patterns for splitting a `LinalgOp` with multiple statements within
/// its payload into multiple `GenericOp` that have a single statement.
/// The option `removeDeadArgsAndResults` adds patterns to remove dead arguments
/// and results from the generated decomposed ops. This is default `true` since
/// the core decomposition patterns relies on these clean up patterns. It is set
/// to false only for testing purposes.
void populateDecomposeLinalgOpsPattern(RewritePatternSet &patterns,
                                       bool removeDeadArgsAndResults = true);

/// Populate patterns that convert non-destination-style ops to destination
/// style ops.
void populateConvertToDestinationStylePatterns(RewritePatternSet &patterns);

/// Populate patterns for vectorizing low-D convolution ops. This is a step in
/// progressive lowering for convolution ops, it assume high-D convolution ops
/// were decomposed previously.
void populateConvolutionVectorizationPatterns(RewritePatternSet &patterns,
                                              PatternBenefit benefit = 1);

/// Populate patterns that convert `ElementwiseMappable` ops to linalg
/// parallel loops.
void populateElementwiseToLinalgConversionPatterns(RewritePatternSet &patterns);

/// Populate patterns that are only useful in the context of sparse tensors.
void populateSparseTensorRewriting(RewritePatternSet &patterns);

/// Function type which is used to control when to stop fusion. It is expected
/// that OpOperand is not modified in the callback. The OpOperand is not marked
/// as const to allow callers to use non-const methods.
using ControlFusionFn = std::function<bool(OpOperand *fusedOperand)>;

/// Patterns for fusing linalg operation on tensors.

/// Pattern to fuse `linalg.generic` -> `linalg.generic` operations
/// when both operations are fusable elementwise operations.
void populateElementwiseOpsFusionPatterns(
    RewritePatternSet &patterns,
    const ControlFusionFn &controlElementwiseOpFusion);

/// Function type which is used to control propagation of tensor.pack/unpack
/// ops.
using ControlPropagationFn = std::function<bool(OpOperand *opOperand)>;

/// Patterns to bubble up or down data layout ops across other operations.
void populateDataLayoutPropagationPatterns(
    RewritePatternSet &patterns,
    const ControlPropagationFn &controlPackUnPackPropagation);

/// Pattern to remove dead operands and results of `linalg.generic` operations.
/// This is effectively DCE for a linalg op.
void populateEraseUnusedOperandsAndResultsPatterns(RewritePatternSet &patterns);

/// Patterns to promote inputs to outputs and remove unused inputs of
/// `linalg.generic` ops.
void populateEraseUnnecessaryInputsPatterns(RewritePatternSet &patterns);

/// Function type to control generic op dimension collapsing. It is expected
/// to return an array of `ReassociationIndices` representing dimensions that
/// should be merged.
using GetCollapsableDimensionsFn =
    std::function<SmallVector<ReassociationIndices>(linalg::LinalgOp)>;

/// Pattern to collapse dimensions in a linalg.generic op. This will collapse
/// tensor operands when needed and expand back the result tensors.
void populateCollapseDimensions(
    RewritePatternSet &patterns,
    const GetCollapsableDimensionsFn &controlCollapseDimensions);

/// Patterns to fold an expanding (collapsing) tensor_reshape operation with its
/// producer (consumer) generic operation by expanding the dimensionality of the
/// loop in the generic op.
void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns, const ControlFusionFn &controlFoldingReshapes);

/// Patterns to fold an expanding tensor.expand_shape operation with its
/// producer generic operation by collapsing the dimensions of the generic op.
void populateFoldReshapeOpsByCollapsingPatterns(
    RewritePatternSet &patterns, const ControlFusionFn &controlFoldingReshapes);

/// Patterns to constant fold Linalg operations.
void populateConstantFoldLinalgOperations(RewritePatternSet &patterns,
                                          const ControlFusionFn &controlFn);

/// Pattern to fuse a `tensor.pad` operation with the producer of its source,
/// if the producer is a `linalg` operation with all parallel iterator types.
void populateFuseTensorPadWithProducerLinalgOpPatterns(
    RewritePatternSet &patterns);

/// Patterns to convert from one named op to another. These can be seen as
/// canonicalizations of named ops into another named op.
void populateLinalgNamedOpConversionPatterns(RewritePatternSet &patterns);

/// Patterns to fold unit-extent dimensions in operands/results of linalg ops on
/// tensors via reassociative reshape ops.
void populateFoldUnitExtentDimsPatterns(RewritePatternSet &patterns,
                                        ControlDropUnitDims &options);

/// A pattern that converts init operands to input operands.
void populateMoveInitOperandsToInputPattern(RewritePatternSet &patterns);

/// Patterns that are used to inline constant operands into linalg generic ops.
void populateInlineConstantOperandsPatterns(RewritePatternSet &patterns);

/// Patterns that are used to bubble up extract slice op above linalg op.
void populateBubbleUpExtractSliceOpPatterns(RewritePatternSet &patterns);

/// Adds patterns that waps tensor.extract_slice(linalg.fill(%cst, %init)) into
/// linalg.fill(%cst, tensor.extract_slice(%init)).
void populateSwapExtractSliceWithFillPatterns(RewritePatternSet &patterns);

/// Patterns to apply `splitReduction` below.
void populateSplitReductionPattern(
    RewritePatternSet &patterns,
    const ControlSplitReductionFn &controlSplitReductionFn,
    bool useAlloc = false);

/// Patterns to convert Linalg matmul ops to transposed variants.
void populateTransposeMatmulPatterns(RewritePatternSet &patterns,
                                     bool transposeLHS = true);

/// Patterns to block pack Linalg matmul ops.
void populateBlockPackMatmulPatterns(RewritePatternSet &patterns,
                                     const ControlBlockPackMatmulFn &controlFn);

/// Patterns to apply Winograd Conv2D algorithm F(m x m, r x r).
void populateWinogradConv2DPatterns(RewritePatternSet &patterns, int64_t m,
                                    int64_t r);

/// Patterns to decompose Winograd operators.
void populateDecomposeWinogradOpsPatterns(RewritePatternSet &patterns);

/// Adds patterns that reduce the rank of named contraction ops that have
/// unit dimensions in the operand(s) by converting to a sequence of
/// `collapse_shape`,
/// `<corresponding linalg named op>`, `expand_shape` (if on tensors).  For
/// example a `linalg.batch_matmul` with unit batch size will convert to
/// `linalg.matmul` and a `linalg.matvec` with with unit spatial dim in lhs will
/// convert to a `linalg.dot`.
void populateContractionOpRankReducingPatterns(RewritePatternSet &patterns);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
