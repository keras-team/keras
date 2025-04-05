//===- Utils.h - General analysis utilities ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// memref's and non-loop IR structures. These are not passes by themselves but
// are used either by passes, optimization sequences, or in turn by other
// transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_ANALYSIS_UTILS_H
#define MLIR_DIALECT_AFFINE_ANALYSIS_UTILS_H

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include <memory>
#include <optional>

namespace mlir {
class Block;
class Location;
class Operation;
class Value;

namespace affine {
class AffineForOp;
class AffineValueMap;
struct MemRefAccess;

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not a region holding op other than ForOp and IfOp
// was encountered in the loop nest.
struct LoopNestStateCollector {
  SmallVector<AffineForOp, 4> forOps;
  SmallVector<Operation *, 4> loadOpInsts;
  SmallVector<Operation *, 4> storeOpInsts;
  bool hasNonAffineRegionOp = false;

  // Collects load and store operations, and whether or not a region holding op
  // other than ForOp and IfOp was encountered in the loop nest.
  void collect(Operation *opToWalk);
};

// MemRefDependenceGraph is a graph data structure where graph nodes are
// top-level operations in a `Block` which contain load/store ops, and edges
// are memref dependences between the nodes.
// TODO: Add a more flexible dependence graph representation.
struct MemRefDependenceGraph {
public:
  // Node represents a node in the graph. A Node is either an entire loop nest
  // rooted at the top level which contains loads/stores, or a top level
  // load/store.
  struct Node {
    // The unique identifier of this node in the graph.
    unsigned id;
    // The top-level statement which is (or contains) a load/store.
    Operation *op;
    // List of load operations.
    SmallVector<Operation *, 4> loads;
    // List of store op insts.
    SmallVector<Operation *, 4> stores;

    Node(unsigned id, Operation *op) : id(id), op(op) {}

    // Returns the load op count for 'memref'.
    unsigned getLoadOpCount(Value memref) const;

    // Returns the store op count for 'memref'.
    unsigned getStoreOpCount(Value memref) const;

    // Returns all store ops in 'storeOps' which access 'memref'.
    void getStoreOpsForMemref(Value memref,
                              SmallVectorImpl<Operation *> *storeOps) const;

    // Returns all load ops in 'loadOps' which access 'memref'.
    void getLoadOpsForMemref(Value memref,
                             SmallVectorImpl<Operation *> *loadOps) const;

    // Returns all memrefs in 'loadAndStoreMemrefSet' for which this node
    // has at least one load and store operation.
    void getLoadAndStoreMemrefSet(DenseSet<Value> *loadAndStoreMemrefSet) const;
  };

  // Edge represents a data dependence between nodes in the graph.
  struct Edge {
    // The id of the node at the other end of the edge.
    // If this edge is stored in Edge = Node.inEdges[i], then
    // 'Node.inEdges[i].id' is the identifier of the source node of the edge.
    // If this edge is stored in Edge = Node.outEdges[i], then
    // 'Node.outEdges[i].id' is the identifier of the dest node of the edge.
    unsigned id;
    // The SSA value on which this edge represents a dependence.
    // If the value is a memref, then the dependence is between graph nodes
    // which contain accesses to the same memref 'value'. If the value is a
    // non-memref value, then the dependence is between a graph node which
    // defines an SSA value and another graph node which uses the SSA value
    // (e.g. a constant or load operation defining a value which is used inside
    // a loop nest).
    Value value;
  };

  // Map from node id to Node.
  DenseMap<unsigned, Node> nodes;
  // Map from node id to list of input edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> inEdges;
  // Map from node id to list of output edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> outEdges;
  // Map from memref to a count on the dependence edges associated with that
  // memref.
  DenseMap<Value, unsigned> memrefEdgeCount;
  // The next unique identifier to use for newly created graph nodes.
  unsigned nextNodeId = 0;

  MemRefDependenceGraph(Block &block) : block(block) {}

  // Initializes the dependence graph based on operations in `block'.
  // Returns true on success, false otherwise.
  bool init();

  // Returns the graph node for 'id'.
  Node *getNode(unsigned id);

  // Returns the graph node for 'forOp'.
  Node *getForOpNode(AffineForOp forOp);

  // Adds a node with 'op' to the graph and returns its unique identifier.
  unsigned addNode(Operation *op);

  // Remove node 'id' (and its associated edges) from graph.
  void removeNode(unsigned id);

  // Returns true if node 'id' writes to any memref which escapes (or is an
  // argument to) the block. Returns false otherwise.
  bool writesToLiveInOrEscapingMemrefs(unsigned id);

  // Returns true iff there is an edge from node 'srcId' to node 'dstId' which
  // is for 'value' if non-null, or for any value otherwise. Returns false
  // otherwise.
  bool hasEdge(unsigned srcId, unsigned dstId, Value value = nullptr);

  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(unsigned srcId, unsigned dstId, Value value);

  // Removes an edge from node 'srcId' to node 'dstId' for 'value'.
  void removeEdge(unsigned srcId, unsigned dstId, Value value);

  // Returns true if there is a path in the dependence graph from node 'srcId'
  // to node 'dstId'. Returns false otherwise. `srcId`, `dstId`, and the
  // operations that the edges connected are expected to be from the same block.
  bool hasDependencePath(unsigned srcId, unsigned dstId);

  // Returns the input edge count for node 'id' and 'memref' from src nodes
  // which access 'memref' with a store operation.
  unsigned getIncomingMemRefAccesses(unsigned id, Value memref);

  // Returns the output edge count for node 'id' and 'memref' (if non-null),
  // otherwise returns the total output edge count from node 'id'.
  unsigned getOutEdgeCount(unsigned id, Value memref = nullptr);

  /// Return all nodes which define SSA values used in node 'id'.
  void gatherDefiningNodes(unsigned id, DenseSet<unsigned> &definingNodes);

  // Computes and returns an insertion point operation, before which the
  // the fused <srcId, dstId> loop nest can be inserted while preserving
  // dependences. Returns nullptr if no such insertion point is found.
  Operation *getFusedLoopNestInsertionPoint(unsigned srcId, unsigned dstId);

  // Updates edge mappings from node 'srcId' to node 'dstId' after fusing them,
  // taking into account that:
  //   *) if 'removeSrcId' is true, 'srcId' will be removed after fusion,
  //   *) memrefs in 'privateMemRefs' has been replaced in node at 'dstId' by a
  //      private memref.
  void updateEdges(unsigned srcId, unsigned dstId,
                   const DenseSet<Value> &privateMemRefs, bool removeSrcId);

  // Update edge mappings for nodes 'sibId' and 'dstId' to reflect fusion
  // of sibling node 'sibId' into node 'dstId'.
  void updateEdges(unsigned sibId, unsigned dstId);

  // Adds ops in 'loads' and 'stores' to node at 'id'.
  void addToNode(unsigned id, const SmallVectorImpl<Operation *> &loads,
                 const SmallVectorImpl<Operation *> &stores);

  void clearNodeLoadAndStores(unsigned id);

  // Calls 'callback' for each input edge incident to node 'id' which carries a
  // memref dependence.
  void forEachMemRefInputEdge(unsigned id,
                              const std::function<void(Edge)> &callback);

  // Calls 'callback' for each output edge from node 'id' which carries a
  // memref dependence.
  void forEachMemRefOutputEdge(unsigned id,
                               const std::function<void(Edge)> &callback);

  // Calls 'callback' for each edge in 'edges' which carries a memref
  // dependence.
  void forEachMemRefEdge(ArrayRef<Edge> edges,
                         const std::function<void(Edge)> &callback);

  void print(raw_ostream &os) const;

  void dump() const { print(llvm::errs()); }

  /// The block for which this graph is created to perform fusion.
  Block &block;
};

/// Populates 'loops' with IVs of the affine.for ops surrounding 'op' ordered
/// from the outermost 'affine.for' operation to the innermost one while not
/// traversing outside of the surrounding affine scope.
void getAffineForIVs(Operation &op, SmallVectorImpl<AffineForOp> *loops);

/// Populates 'ivs' with IVs of the surrounding affine.for and affine.parallel
/// ops ordered from the outermost one to the innermost while not traversing
/// outside of the surrounding affine scope.
void getAffineIVs(Operation &op, SmallVectorImpl<Value> &ivs);

/// Populates 'ops' with affine operations enclosing `op` ordered from outermost
/// to innermost while stopping at the boundary of the affine scope. affine.for,
/// affine.if, or affine.parallel ops comprise such surrounding affine ops.
/// `ops` is guaranteed by design to have a successive chain of affine parent
/// ops.
void getEnclosingAffineOps(Operation &op, SmallVectorImpl<Operation *> *ops);

/// Returns the nesting depth of this operation, i.e., the number of loops
/// surrounding this operation.
unsigned getNestingDepth(Operation *op);

/// Returns whether a loop is a parallel loop and contains a reduction loop.
bool isLoopParallelAndContainsReduction(AffineForOp forOp);

/// Returns in 'sequentialLoops' all sequential loops in loop nest rooted
/// at 'forOp'.
void getSequentialLoops(AffineForOp forOp,
                        llvm::SmallDenseSet<Value, 8> *sequentialLoops);

/// Enumerates different result statuses of slice computation by
/// `computeSliceUnion`
// TODO: Identify and add different kinds of failures during slice computation.
struct SliceComputationResult {
  enum ResultEnum {
    Success,
    IncorrectSliceFailure, // Slice is computed, but it is incorrect.
    GenericFailure,        // Unable to compute src loop computation slice.
  } value;
  SliceComputationResult(ResultEnum v) : value(v) {}
};

/// ComputationSliceState aggregates loop IVs, loop bound AffineMaps and their
/// associated operands for a set of loops within a loop nest (typically the
/// set of loops surrounding a store operation). Loop bound AffineMaps which
/// are non-null represent slices of that loop's iteration space.
struct ComputationSliceState {
  // List of sliced loop IVs (ordered from outermost to innermost).
  // EX: 'ivs[i]' has lower bound 'lbs[i]' and upper bound 'ubs[i]'.
  SmallVector<Value, 4> ivs;
  // List of lower bound AffineMaps.
  SmallVector<AffineMap, 4> lbs;
  // List of upper bound AffineMaps.
  SmallVector<AffineMap, 4> ubs;
  // List of lower bound operands (lbOperands[i] are used by 'lbs[i]').
  std::vector<SmallVector<Value, 4>> lbOperands;
  // List of upper bound operands (ubOperands[i] are used by 'ubs[i]').
  std::vector<SmallVector<Value, 4>> ubOperands;
  // Slice loop nest insertion point in target loop nest.
  Block::iterator insertPoint;
  // Adds to 'cst' with constraints which represent the slice bounds on 'ivs'
  // in 'this'. Specifically, the values in 'ivs' are added to 'cst' as dim
  // variables and the values in 'lb/ubOperands' are added as symbols.
  // Constraints are added for all loop IV bounds (dim or symbol), and
  // constraints are added for slice bounds in 'lbs'/'ubs'.
  // Returns failure if we cannot add loop bounds because of unsupported cases.
  LogicalResult getAsConstraints(FlatAffineValueConstraints *cst) const;

  /// Adds to 'cst' constraints which represent the original loop bounds on
  /// 'ivs' in 'this'. This corresponds to the original domain of the loop nest
  /// from which the slice is being computed. Returns failure if we cannot add
  /// loop bounds because of unsupported cases.
  LogicalResult getSourceAsConstraints(FlatAffineValueConstraints &cst) const;

  // Clears all bounds and operands in slice state.
  void clearBounds();

  /// Returns true if the computation slice is empty.
  bool isEmpty() const { return ivs.empty(); }

  /// Returns true if the computation slice encloses all the iterations of the
  /// sliced loop nest. Returns false if it does not. Returns std::nullopt if it
  /// cannot determine if the slice is maximal or not.
  // TODO: Cache 'isMaximal' so that we don't recompute it when the slice
  // information hasn't changed.
  std::optional<bool> isMaximal() const;

  /// Checks the validity of the slice computed. This is done using the
  /// following steps:
  /// 1. Get the new domain of the slice that would be created if fusion
  /// succeeds. This domain gets constructed with source loop IVS and
  /// destination loop IVS as dimensions.
  /// 2. Project out the dimensions of the destination loop from the domain
  /// above calculated in step(1) to express it purely in terms of the source
  /// loop IVs.
  /// 3. Calculate a set difference between the iterations of the new domain and
  /// the original domain of the source loop.
  /// If this difference is empty, the slice is declared to be valid. Otherwise,
  /// return false as it implies that the effective fusion results in at least
  /// one iteration of the slice that was not originally in the source's domain.
  /// If the validity cannot be determined, returns std::nullopt.
  std::optional<bool> isSliceValid() const;

  void dump() const;

private:
  /// Fast check to determine if the computation slice is maximal. Returns true
  /// if each slice dimension maps to an existing dst dimension and both the src
  /// and the dst loops for those dimensions have the same bounds. Returns false
  /// if both the src and the dst loops don't have the same bounds. Returns
  /// std::nullopt if none of the above can be proven.
  std::optional<bool> isSliceMaximalFastCheck() const;
};

/// Computes the computation slice loop bounds for one loop nest as affine maps
/// of the other loop nest's IVs and symbols, using 'dependenceConstraints'
/// computed between 'depSourceAccess' and 'depSinkAccess'.
/// If 'isBackwardSlice' is true, a backwards slice is computed in which the
/// slice bounds of loop nest surrounding 'depSourceAccess' are computed in
/// terms of loop IVs and symbols of the loop nest surrounding 'depSinkAccess'
/// at 'loopDepth'.
/// If 'isBackwardSlice' is false, a forward slice is computed in which the
/// slice bounds of loop nest surrounding 'depSinkAccess' are computed in terms
/// of loop IVs and symbols of the loop nest surrounding 'depSourceAccess' at
/// 'loopDepth'.
/// The slice loop bounds and associated operands are returned in 'sliceState'.
//
//  Backward slice example:
//
//    affine.for %i0 = 0 to 10 {
//      affine.store %cst, %0[%i0] : memref<100xf32>  // 'depSourceAccess'
//    }
//    affine.for %i1 = 0 to 10 {
//      %v = affine.load %0[%i1] : memref<100xf32>    // 'depSinkAccess'
//    }
//
//    // Backward computation slice of loop nest '%i0'.
//    affine.for %i0 = (d0) -> (d0)(%i1) to (d0) -> (d0 + 1)(%i1) {
//      affine.store %cst, %0[%i0] : memref<100xf32>  // 'depSourceAccess'
//    }
//
//  Forward slice example:
//
//    affine.for %i0 = 0 to 10 {
//      affine.store %cst, %0[%i0] : memref<100xf32>  // 'depSourceAccess'
//    }
//    affine.for %i1 = 0 to 10 {
//      %v = affine.load %0[%i1] : memref<100xf32>    // 'depSinkAccess'
//    }
//
//    // Forward computation slice of loop nest '%i1'.
//    affine.for %i1 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 1)(%i0) {
//      %v = affine.load %0[%i1] : memref<100xf32>    // 'depSinkAccess'
//    }
//
void getComputationSliceState(Operation *depSourceOp, Operation *depSinkOp,
                              FlatAffineValueConstraints *dependenceConstraints,
                              unsigned loopDepth, bool isBackwardSlice,
                              ComputationSliceState *sliceState);

/// Return the number of iterations for the `slicetripCountMap` provided.
uint64_t getSliceIterationCount(
    const llvm::SmallDenseMap<Operation *, uint64_t, 8> &sliceTripCountMap);

/// Builds a map 'tripCountMap' from AffineForOp to constant trip count for
/// loop nest surrounding represented by slice loop bounds in 'slice'. Returns
/// true on success, false otherwise (if a non-constant trip count was
/// encountered).
bool buildSliceTripCountMap(
    const ComputationSliceState &slice,
    llvm::SmallDenseMap<Operation *, uint64_t, 8> *tripCountMap);

/// Computes in 'sliceUnion' the union of all slice bounds computed at
/// 'loopDepth' between all dependent pairs of ops in 'opsA' and 'opsB', and
/// then verifies if it is valid. The parameter 'numCommonLoops' is the number
/// of loops common to the operations in 'opsA' and 'opsB'. If 'isBackwardSlice'
/// is true, computes slice bounds for loop nest surrounding ops in 'opsA', as a
/// function of IVs and symbols of loop nest surrounding ops in 'opsB' at
/// 'loopDepth'. If 'isBackwardSlice' is false, computes slice bounds for loop
/// nest surrounding ops in 'opsB', as a function of IVs and symbols of loop
/// nest surrounding ops in 'opsA' at 'loopDepth'. Returns
/// 'SliceComputationResult::Success' if union was computed correctly, an
/// appropriate 'failure' otherwise.
SliceComputationResult
computeSliceUnion(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                  unsigned loopDepth, unsigned numCommonLoops,
                  bool isBackwardSlice, ComputationSliceState *sliceUnion);

/// Creates a clone of the computation contained in the loop nest surrounding
/// 'srcOpInst', slices the iteration space of src loop based on slice bounds
/// in 'sliceState', and inserts the computation slice at the beginning of the
/// operation block of the loop at 'dstLoopDepth' in the loop nest surrounding
/// 'dstOpInst'. Returns the top-level loop of the computation slice on
/// success, returns nullptr otherwise.
// Loop depth is a crucial optimization choice that determines where to
// materialize the results of the backward slice - presenting a trade-off b/w
// storage and redundant computation in several cases.
// TODO: Support computation slices with common surrounding loops.
AffineForOp insertBackwardComputationSlice(Operation *srcOpInst,
                                           Operation *dstOpInst,
                                           unsigned dstLoopDepth,
                                           ComputationSliceState *sliceState);

/// A region of a memref's data space; this is typically constructed by
/// analyzing load/store op's on this memref and the index space of loops
/// surrounding such op's.
// For example, the memref region for a load operation at loop depth = 1:
//
//    affine.for %i = 0 to 32 {
//      affine.for %ii = %i to (d0) -> (d0 + 8) (%i) {
//        affine.load %A[%ii]
//      }
//    }
//
// Region:  {memref = %A, write = false, {%i <= m0 <= %i + 7} }
// The last field is a 2-d FlatAffineValueConstraints symbolic in %i.
//
struct MemRefRegion {
  explicit MemRefRegion(Location loc) : loc(loc) {}

  /// Computes the memory region accessed by this memref with the region
  /// represented as constraints symbolic/parametric in 'loopDepth' loops
  /// surrounding opInst. The computed region's 'cst' field has exactly as many
  /// dimensional variables as the rank of the memref, and *potentially*
  /// additional symbolic variables which could include any of the loop IVs
  /// surrounding opInst up until 'loopDepth' and another additional Function
  /// symbols involved with the access (for eg., those appear in affine.apply's,
  /// loop bounds, etc.). If 'sliceState' is non-null, operands from
  /// 'sliceState' are added as symbols, and the following constraints are added
  /// to the system:
  /// *) Inequality constraints which represent loop bounds for 'sliceState'
  ///    operands which are loop IVS (these represent the destination loop IVs
  ///    of the slice, and are added as symbols to MemRefRegion's constraint
  ///    system).
  /// *) Inequality constraints for the slice bounds in 'sliceState', which
  ///    represent the bounds on the loop IVs in this constraint system w.r.t
  ///    to slice operands (which correspond to symbols).
  /// If 'addMemRefDimBounds' is true, constant upper/lower bounds
  /// [0, memref.getDimSize(i)) are added for each MemRef dimension 'i'.
  ///
  ///  For example, the memref region for this operation at loopDepth = 1 will
  ///  be:
  ///
  ///    affine.for %i = 0 to 32 {
  ///      affine.for %ii = %i to (d0) -> (d0 + 8) (%i) {
  ///        load %A[%ii]
  ///      }
  ///    }
  ///
  ///   {memref = %A, write = false, {%i <= m0 <= %i + 7} }
  /// The last field is a 2-d FlatAffineValueConstraints symbolic in %i.
  ///
  LogicalResult compute(Operation *op, unsigned loopDepth,
                        const ComputationSliceState *sliceState = nullptr,
                        bool addMemRefDimBounds = true);

  FlatAffineValueConstraints *getConstraints() { return &cst; }
  const FlatAffineValueConstraints *getConstraints() const { return &cst; }
  bool isWrite() const { return write; }
  void setWrite(bool flag) { write = flag; }

  /// Returns a constant upper bound on the number of elements in this region if
  /// bounded by a known constant (always possible for static shapes),
  /// std::nullopt otherwise. Note that the symbols of the region are treated
  /// specially, i.e., the returned bounding constant holds for *any given*
  /// value of the symbol variables. The 'shape' vector is set to the
  /// corresponding dimension-wise bounds major to minor. The number of elements
  /// and all the dimension-wise bounds are guaranteed to be non-negative. We
  /// use int64_t instead of uint64_t since index types can be at most
  /// int64_t. `lbs` are set to the lower bounds for each of the rank
  /// dimensions, and lbDivisors contains the corresponding denominators for
  /// floorDivs.
  std::optional<int64_t> getConstantBoundingSizeAndShape(
      SmallVectorImpl<int64_t> *shape = nullptr,
      std::vector<SmallVector<int64_t, 4>> *lbs = nullptr,
      SmallVectorImpl<int64_t> *lbDivisors = nullptr) const;

  /// Gets the lower and upper bound map for the dimensional variable at
  /// `pos`.
  void getLowerAndUpperBound(unsigned pos, AffineMap &lbMap,
                             AffineMap &ubMap) const;

  /// A wrapper around FlatAffineValueConstraints::getConstantBoundOnDimSize().
  /// 'pos' corresponds to the position of the memref shape's dimension (major
  /// to minor) which matches 1:1 with the dimensional variable positions in
  /// 'cst'.
  std::optional<int64_t>
  getConstantBoundOnDimSize(unsigned pos,
                            SmallVectorImpl<int64_t> *lb = nullptr,
                            int64_t *lbFloorDivisor = nullptr) const {
    assert(pos < getRank() && "invalid position");
    return cst.getConstantBoundOnDimSize64(pos, lb);
  }

  /// Returns the size of this MemRefRegion in bytes.
  std::optional<int64_t> getRegionSize();

  // Wrapper around FlatAffineValueConstraints::unionBoundingBox.
  LogicalResult unionBoundingBox(const MemRefRegion &other);

  /// Returns the rank of the memref that this region corresponds to.
  unsigned getRank() const;

  /// Memref that this region corresponds to.
  Value memref;

  /// Read or write.
  bool write = false;

  /// If there is more than one load/store op associated with the region, the
  /// location information would correspond to one of those op's.
  Location loc;

  /// Region (data space) of the memref accessed. This set will thus have at
  /// least as many dimensional variables as the shape dimensionality of the
  /// memref, and these are the leading dimensions of the set appearing in that
  /// order (major to minor / outermost to innermost). There may be additional
  /// variables since getMemRefRegion() is called with a specific loop depth,
  /// and thus the region is symbolic in the outer surrounding loops at that
  /// depth.
  FlatAffineValueConstraints cst;
};

/// Returns the size of a memref with element type int or float in bytes if it's
/// statically shaped, std::nullopt otherwise.
std::optional<uint64_t> getIntOrFloatMemRefSizeInBytes(MemRefType memRefType);

/// Checks a load or store op for an out of bound access; returns failure if the
/// access is out of bounds along any of the dimensions, success otherwise.
/// Emits a diagnostic error (with location information) if emitError is true.
template <typename LoadOrStoreOpPointer>
LogicalResult boundCheckLoadOrStoreOp(LoadOrStoreOpPointer loadOrStoreOp,
                                      bool emitError = true);

/// Returns the number of surrounding loops common to both A and B.
unsigned getNumCommonSurroundingLoops(Operation &a, Operation &b);

/// Gets the memory footprint of all data touched in the specified memory space
/// in bytes; if the memory space is unspecified, considers all memory spaces.
std::optional<int64_t> getMemoryFootprintBytes(AffineForOp forOp,
                                               int memorySpace = -1);

/// Returns the memref's element type's size in bytes where the elemental type
/// is an int or float or a vector of such types.
std::optional<int64_t> getMemRefIntOrFloatEltSizeInBytes(MemRefType memRefType);

/// Simplify the integer set by simplifying the underlying affine expressions by
/// flattening and some simple inference. Also, drop any duplicate constraints.
/// Returns the simplified integer set. This method runs in time linear in the
/// number of constraints.
IntegerSet simplifyIntegerSet(IntegerSet set);

/// Returns the innermost common loop depth for the set of operations in 'ops'.
unsigned getInnermostCommonLoopDepth(
    ArrayRef<Operation *> ops,
    SmallVectorImpl<AffineForOp> *surroundingLoops = nullptr);

/// Try to simplify the given affine.min or affine.max op to an affine map with
/// a single result and operands, taking into account the specified constraint
/// set. Return failure if no simplified version could be found.
FailureOr<AffineValueMap>
simplifyConstrainedMinMaxOp(Operation *op,
                            FlatAffineValueConstraints constraints);

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_ANALYSIS_UTILS_H
