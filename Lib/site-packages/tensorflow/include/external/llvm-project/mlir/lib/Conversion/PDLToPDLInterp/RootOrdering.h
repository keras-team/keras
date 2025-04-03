//===- RootOrdering.h - Optimal root ordering  ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definition for a cost graph over candidate roots and
// an implementation of an algorithm to determine the optimal ordering over
// these roots. Each edge in this graph indicates that the target root can be
// connected (via a chain of positions) to the source root, and their cost
// indicates the estimated cost of such traversal. The optimal root ordering
// is then formulated as that of finding a spanning arborescence (i.e., a
// directed spanning tree) of minimal weight.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_CONVERSION_PDLTOPDLINTERP_ROOTORDERING_H_
#define MLIR_LIB_CONVERSION_PDLTOPDLINTERP_ROOTORDERING_H_

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <vector>

namespace mlir {
namespace pdl_to_pdl_interp {

/// The information associated with an edge in the cost graph. Each node in
/// the cost graph corresponds to a candidate root detected in the pdl.pattern,
/// and each edge in the cost graph corresponds to connecting the two candidate
/// roots via a chain of operations. The cost of an edge is the smallest number
/// of upward traversals required to go from the source to the target root, and
/// the connector is a `Value` in the intersection of the two subtrees rooted at
/// the source and target root that results in that smallest number of upward
/// traversals. Consider the following pattern with 3 roots op3, op4, and op5:
///
///                 argA ---> op1 ---> op2 ---> op3 ---> res3
///                            ^        ^
///                            |        |
///                           argB     argC
///                            |        |
///                            v        v
///                 res4 <--- op4      op5 ---> res5
///                            ^        ^
///                            |        |
///                           op6      op7
///
/// The cost of the edge op3 -> op4 is 1 (the upward traversal argB -> op4),
/// with argB being the connector `Value` and similarly for op3 -> op5 (cost 1,
/// connector argC). The cost of the edge op4 -> op3 is 3 (upward traversals
/// argB -> op1 -> op2 -> op3, connector argB), while the cost of edge op5 ->
/// op3 is 2 (uwpard traversals argC -> op2 -> op3). There are no edges between
/// op4 and op5 in the cost graph, because the subtrees rooted at these two
/// roots do not intersect. It is easy to see that the optimal root for this
/// pattern is op3, resulting in the spanning arborescence op3 -> {op4, op5}.
struct RootOrderingEntry {
  /// The depth of the connector `Value` w.r.t. the target root.
  ///
  /// This is a pair where the first value is the additive cost (the depth of
  /// the connector), and the second value is a priority for breaking ties
  /// (with 0 being the highest). Typically, the priority is a unique edge ID.
  std::pair<unsigned, unsigned> cost;

  /// The connector value in the intersection of the two subtrees rooted at
  /// the source and target root that results in that smallest depth w.r.t.
  /// the target root.
  Value connector;
};

/// A directed graph representing the cost of ordering the roots in the
/// predicate tree. It is represented as an adjacency map, where the outer map
/// is indexed by the target node, and the inner map is indexed by the source
/// node. Each edge is associated with a cost and the underlying connector
/// value.
using RootOrderingGraph = DenseMap<Value, DenseMap<Value, RootOrderingEntry>>;

/// The optimal branching algorithm solver. This solver accepts a graph and the
/// root in its constructor, and is invoked via the solve() member function.
/// This is a direct implementation of the Edmonds' algorithm, see
/// https://en.wikipedia.org/wiki/Edmonds%27_algorithm. The worst-case
/// computational complexity of this algorithm is O(N^3), for a single root.
/// The PDL-to-PDLInterp lowering calls this N times (once for each candidate
/// root), so the overall complexity root ordering is O(N^4). If needed, this
/// could be reduced to O(N^3) with a more efficient algorithm. However, note
/// that the underlying implementation is very efficient, and N in our
/// instances tends to be very small (<10).
class OptimalBranching {
public:
  /// A list of edges (child, parent).
  using EdgeList = std::vector<std::pair<Value, Value>>;

  /// Constructs the solver for the given graph and root value.
  OptimalBranching(RootOrderingGraph graph, Value root);

  /// Runs the Edmonds' algorithm for the current `graph`, returning the total
  /// cost of the minimum-weight spanning arborescence (sum of the edge costs).
  /// This function first determines the optimal local choice of the parents
  /// and stores this choice in the `parents` mapping. If this choice results
  /// in an acyclic graph, the function returns immediately. Otherwise, it
  /// takes an arbitrary cycle, contracts it, and recurses on the new graph
  /// (which is guaranteed to have fewer nodes than we began with). After we
  /// return from recursion, we redirect the edges to/from the contracted node,
  /// so the `parents` map contains a valid solution for the current graph.
  unsigned solve();

  /// Returns the computed parent map. This is the unique predecessor for each
  /// node (root) in the optimal branching.
  const DenseMap<Value, Value> &getRootOrderingParents() const {
    return parents;
  }

  /// Returns the computed edges as visited in the preorder traversal.
  /// The specified array determines the order for breaking any ties.
  EdgeList preOrderTraversal(ArrayRef<Value> nodes) const;

private:
  /// The graph whose optimal branching we wish to determine.
  RootOrderingGraph graph;

  /// The root of the optimal branching.
  Value root;

  /// The computed parent mapping. This is the unique predecessor for each node
  /// in the optimal branching. The keys of this map correspond to the keys of
  /// the outer map of the input graph, and each value is one of the keys of
  /// the inner map for this node. Also used as an intermediate (possibly
  /// cyclical) result in the optimal branching algorithm.
  DenseMap<Value, Value> parents;
};

} // namespace pdl_to_pdl_interp
} // namespace mlir

#endif // MLIR_CONVERSION_PDLTOPDLINTERP_ROOTORDERING_H_
