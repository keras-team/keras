//===- AffineMap.h - MLIR Affine Map Class ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Affine maps are mathematical functions which map a list of dimension
// identifiers and symbols, to multidimensional affine expressions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_AFFINEMAP_H
#define MLIR_IR_AFFINEMAP_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <optional>

namespace llvm {
class SmallBitVector;
} // namespace llvm

namespace mlir {

namespace detail {
struct AffineMapStorage;
} // namespace detail

class Attribute;
class Builder;
class OpFoldResult;
class MLIRContext;

/// A multi-dimensional affine map
/// Affine map's are immutable like Type's, and they are uniqued.
/// Eg: (d0, d1) -> (d0/128, d0 mod 128, d1)
/// The names used (d0, d1) don't matter - it's the mathematical function that
/// is unique to this affine map.
class AffineMap {
public:
  using ImplType = detail::AffineMapStorage;

  constexpr AffineMap() = default;
  explicit AffineMap(ImplType *map) : map(map) {}

  /// Returns a zero result affine map with no dimensions or symbols: () -> ().
  static AffineMap get(MLIRContext *context);

  /// Returns a zero result affine map with `dimCount` dimensions and
  /// `symbolCount` symbols, e.g.: `(...) -> ()`.
  static AffineMap get(unsigned dimCount, unsigned symbolCount,
                       MLIRContext *context);

  /// Returns an affine map with `dimCount` dimensions and `symbolCount` mapping
  /// to a single output dimension
  static AffineMap get(unsigned dimCount, unsigned symbolCount,
                       AffineExpr result);

  /// Returns an affine map with `dimCount` dimensions and `symbolCount` mapping
  /// to the given results.
  static AffineMap get(unsigned dimCount, unsigned symbolCount,
                       ArrayRef<AffineExpr> results, MLIRContext *context);

  /// Returns a single constant result affine map.
  static AffineMap getConstantMap(int64_t val, MLIRContext *context);

  /// Returns an AffineMap with 'numDims' identity result dim exprs.
  static AffineMap getMultiDimIdentityMap(unsigned numDims,
                                          MLIRContext *context);

  /// Returns an identity affine map (d0, ..., dn) -> (dp, ..., dn) on the most
  /// minor dimensions.
  static AffineMap getMinorIdentityMap(unsigned dims, unsigned results,
                                       MLIRContext *context);

  /// Returns an identity affine map with `numDims` input dimensions and
  /// filtered results using `keepDimFilter`. If `keepDimFilter` returns true
  /// for a dimension, the dimension is kept in the affine map results.
  /// Otherwise, the dimension is dropped from the results.
  ///
  /// Examples:
  ///   * getFilteredIdentityMap(4, [false, true, false, true])
  ///       -> affine_map<(d0, d1, d2, d3) -> (d1, d3)>
  ///   * getFilteredIdentityMap(3, [false, false, true])
  ///       -> affine_map<(d0, d1, d2) -> (d2)>
  static AffineMap
  getFilteredIdentityMap(MLIRContext *ctx, unsigned numDims,
                         llvm::function_ref<bool(AffineDimExpr)> keepDimFilter);

  /// Returns an AffineMap representing a permutation.
  /// The permutation is expressed as a non-empty vector of integers.
  /// E.g. the permutation `(i,j,k) -> (j,k,i)` will be expressed with
  /// `permutation = [1,2,0]`. All values in `permutation` must be
  /// integers, in the range 0..`permutation.size()-1` without duplications
  /// (i.e. `[1,1,2]` is an invalid permutation).
  static AffineMap getPermutationMap(ArrayRef<unsigned> permutation,
                                     MLIRContext *context);
  static AffineMap getPermutationMap(ArrayRef<int64_t> permutation,
                                     MLIRContext *context);

  /// Returns an affine map with `numDims` input dimensions and results
  /// specified by `targets`.
  ///
  /// Examples:
  /// * getMultiDimMapWithTargets(3, [0, 2, 1])
  ///       -> affine_map<(d0, d1, d2) -> (d0, d2, d1)>
  /// * getMultiDimMapWithTargets(3, [2, 1])
  ///       -> affine_map<(d0, d1, d2) -> (d2, d1)>
  static AffineMap getMultiDimMapWithTargets(unsigned numDims,
                                             ArrayRef<unsigned> targets,
                                             MLIRContext *context);

  /// Returns a vector of AffineMaps; each with as many results as
  /// `exprs.size()`, as many dims as the largest dim in `exprs` and as many
  /// symbols as the largest symbol in `exprs`.
  static SmallVector<AffineMap, 4>
  inferFromExprList(ArrayRef<ArrayRef<AffineExpr>> exprsList,
                    MLIRContext *context);
  static SmallVector<AffineMap, 4>
  inferFromExprList(ArrayRef<SmallVector<AffineExpr, 4>> exprsList,
                    MLIRContext *context);

  MLIRContext *getContext() const;

  explicit operator bool() const { return map != nullptr; }
  bool operator==(AffineMap other) const { return other.map == map; }
  bool operator!=(AffineMap other) const { return !(other.map == map); }

  /// Returns true if this affine map is an identity affine map.
  /// An identity affine map corresponds to an identity affine function on the
  /// dimensional identifiers.
  bool isIdentity() const;

  /// Returns true if this affine map is an identity affine map on the symbol
  /// identifiers.
  bool isSymbolIdentity() const;

  /// Returns true if this affine map is a minor identity, i.e. an identity
  /// affine map (d0, ..., dn) -> (dp, ..., dn) on the most minor dimensions.
  bool isMinorIdentity() const;

  /// Returns the list of broadcast dimensions (i.e. dims indicated by value 0
  /// in the result).
  /// Ex:
  ///  * (d0, d1, d2) -> (0, d1) gives [0]
  ///  * (d0, d1, d2) -> (d2, d1) gives []
  ///  * (d0, d1, d2, d4) -> (d0, 0, d1, 0) gives [1, 3]
  SmallVector<unsigned> getBroadcastDims() const;

  /// Returns true if this affine map is a minor identity up to broadcasted
  /// dimensions which are indicated by value 0 in the result. If
  /// `broadcastedDims` is not null, it will be populated with the indices of
  /// the broadcasted dimensions in the result array.
  /// Example: affine_map<(d0, d1, d2, d3, d4) -> (0, d2, 0, d4)>
  ///          (`broadcastedDims` will contain [0, 2])
  bool isMinorIdentityWithBroadcasting(
      SmallVectorImpl<unsigned> *broadcastedDims = nullptr) const;

  /// Return true if this affine map can be converted to a minor identity with
  /// broadcast by doing a permute. Return a permutation (there may be
  /// several) to apply to get to a minor identity with broadcasts.
  /// Ex:
  ///  * (d0, d1, d2) -> (0, d1) maps to minor identity (d1, 0 = d2) with
  ///  perm = [1, 0] and broadcast d2
  ///  * (d0, d1, d2) -> (d0, 0) cannot be mapped to a minor identity by
  ///  permutation + broadcast
  ///  * (d0, d1, d2, d3) -> (0, d1, d3) maps to minor identity (d1, 0 = d2, d3)
  ///  with perm = [1, 0, 2] and broadcast d2
  ///  * (d0, d1) -> (d1, 0, 0, d0) maps to minor identity (d0, d1) with extra
  ///  leading broadcat dimensions. The map returned would be (0, 0, d0, d1)
  ///  with perm = [3, 0, 1, 2]
  bool isPermutationOfMinorIdentityWithBroadcasting(
      SmallVectorImpl<unsigned> &permutedDims) const;

  /// Returns true if this affine map is an empty map, i.e., () -> ().
  bool isEmpty() const;

  /// Returns true if this affine map is a single result constant function.
  bool isSingleConstant() const;

  /// Returns true if this affine map has only constant results.
  bool isConstant() const;

  /// Returns the constant result of this map. This methods asserts that the map
  /// has a single constant result.
  int64_t getSingleConstantResult() const;

  /// Returns the constant results of this map. This method asserts that the map
  /// has all constant results.
  SmallVector<int64_t> getConstantResults() const;

  // Prints affine map to 'os'.
  void print(raw_ostream &os) const;
  void dump() const;

  unsigned getNumDims() const;
  unsigned getNumSymbols() const;
  unsigned getNumResults() const;
  unsigned getNumInputs() const;

  ArrayRef<AffineExpr> getResults() const;
  AffineExpr getResult(unsigned idx) const;

  /// Extracts the position of the dimensional expression at the given result,
  /// when the caller knows it is safe to do so.
  unsigned getDimPosition(unsigned idx) const;

  /// Extracts the first result position where `input` dimension resides.
  /// Returns `std::nullopt` if `input` is not a dimension expression or cannot
  /// be found in results.
  std::optional<unsigned> getResultPosition(AffineExpr input) const;

  /// Return true if any affine expression involves AffineDimExpr `position`.
  bool isFunctionOfDim(unsigned position) const {
    return llvm::any_of(getResults(), [&](AffineExpr e) {
      return e.isFunctionOfDim(position);
    });
  }

  /// Return true if any affine expression involves AffineSymbolExpr `position`.
  bool isFunctionOfSymbol(unsigned position) const {
    return llvm::any_of(getResults(), [&](AffineExpr e) {
      return e.isFunctionOfSymbol(position);
    });
  }

  /// Walk all of the AffineExpr's in this mapping. Each node in an expression
  /// tree is visited in postorder.
  void walkExprs(llvm::function_ref<void(AffineExpr)> callback) const;

  /// This method substitutes any uses of dimensions and symbols (e.g.
  /// dim#0 with dimReplacements[0]) in subexpressions and returns the modified
  /// expression mapping.  Because this can be used to eliminate dims and
  /// symbols, the client needs to specify the number of dims and symbols in
  /// the result.  The returned map always has the same number of results.
  AffineMap replaceDimsAndSymbols(ArrayRef<AffineExpr> dimReplacements,
                                  ArrayRef<AffineExpr> symReplacements,
                                  unsigned numResultDims,
                                  unsigned numResultSyms) const;

  /// Sparse replace method. Apply AffineExpr::replace(`expr`, `replacement`) to
  /// each of the results and return a new AffineMap with the new results and
  /// with the specified number of dims and symbols.
  AffineMap replace(AffineExpr expr, AffineExpr replacement,
                    unsigned numResultDims, unsigned numResultSyms) const;

  /// Sparse replace method. Apply AffineExpr::replace(`map`) to each of the
  /// results and return a new AffineMap with the new results and with inferred
  /// number of dims and symbols.
  AffineMap replace(const DenseMap<AffineExpr, AffineExpr> &map) const;

  /// Sparse replace method. Apply AffineExpr::replace(`map`) to each of the
  /// results and return a new AffineMap with the new results and with the
  /// specified number of dims and symbols.
  AffineMap replace(const DenseMap<AffineExpr, AffineExpr> &map,
                    unsigned numResultDims, unsigned numResultSyms) const;

  /// Replace dims[offset ... numDims)
  /// by dims[offset + shift ... shift + numDims).
  AffineMap shiftDims(unsigned shift, unsigned offset = 0) const {
    assert(offset <= getNumDims());
    return AffineMap::get(getNumDims() + shift, getNumSymbols(),
                          llvm::map_to_vector<4>(
                              getResults(),
                              [&](AffineExpr e) {
                                return e.shiftDims(getNumDims(), shift, offset);
                              }),
                          getContext());
  }

  /// Replace symbols[offset ... numSymbols)
  /// by symbols[offset + shift ... shift + numSymbols).
  AffineMap shiftSymbols(unsigned shift, unsigned offset = 0) const {
    return AffineMap::get(getNumDims(), getNumSymbols() + shift,
                          llvm::map_to_vector<4>(getResults(),
                                                 [&](AffineExpr e) {
                                                   return e.shiftSymbols(
                                                       getNumSymbols(), shift,
                                                       offset);
                                                 }),
                          getContext());
  }

  /// Returns a new AffineMap with the same number of dims and symbols and one
  /// less result at `pos`, dropped.
  AffineMap dropResult(int64_t pos) const {
    return dropResults(ArrayRef({pos}));
  }

  // Returns a new AffineMap with the same number of dims and symbols, but all
  // results in `positions` dropped.
  AffineMap dropResults(ArrayRef<int64_t> positions) const {
    SmallVector<int64_t> reverse_sorted_positions = llvm::to_vector(positions);
    llvm::sort(reverse_sorted_positions, std::greater<int64_t>());

    auto exprs = llvm::to_vector<4>(getResults());
    for (int64_t pos : reverse_sorted_positions)
      exprs.erase(exprs.begin() + pos);
    return AffineMap::get(getNumDims(), getNumSymbols(), exprs, getContext());
  }

  // Returns a new AffineMap with the same number of dims and symbols, but all
  // results in `positions` dropped.
  AffineMap dropResults(const llvm::SmallBitVector &positions) const;

  /// Returns a new AffineMap with the same number of dims and symbols and an
  /// extra result inserted at `pos`.
  AffineMap insertResult(AffineExpr expr, unsigned pos) const {
    auto exprs = llvm::to_vector<4>(getResults());
    exprs.insert(exprs.begin() + pos, expr);
    return AffineMap::get(getNumDims(), getNumSymbols(), exprs, getContext());
  }

  /// Folds the results of the application of an affine map on the provided
  /// operands to a constant if possible.
  LogicalResult constantFold(ArrayRef<Attribute> operandConstants,
                             SmallVectorImpl<Attribute> &results,
                             bool *hasPoison = nullptr) const;

  /// Propagates the constant operands into this affine map. Operands are
  /// allowed to be null, at which point they are treated as non-constant. This
  /// does not change the number of symbols and dimensions. Returns a new map,
  /// which may be equal to the old map if no folding happened. If `results` is
  /// provided and if all expressions in the map were folded to constants,
  /// `results` will contain the values of these constants.
  AffineMap partialConstantFold(ArrayRef<Attribute> operandConstants,
                                SmallVectorImpl<int64_t> *results = nullptr,
                                bool *hasPoison = nullptr) const;

  /// Returns the AffineMap resulting from composing `this` with `map`.
  /// The resulting AffineMap has as many AffineDimExpr as `map` and as many
  /// AffineSymbolExpr as the concatenation of `this` and `map` (in which case
  /// the symbols of `this` map come first).
  ///
  /// Prerequisites:
  /// The maps are composable, i.e. that the number of AffineDimExpr of `this`
  /// matches the number of results of `map`.
  ///
  /// Example:
  ///   map1: `(d0, d1)[s0, s1] -> (d0 + 1 + s1, d1 - 1 - s0)`
  ///   map2: `(d0)[s0] -> (d0 + s0, d0 - s0)`
  ///   map1.compose(map2):
  ///     `(d0)[s0, s1, s2] -> (d0 + s1 + s2 + 1, d0 - s0 - s2 - 1)`
  AffineMap compose(AffineMap map) const;

  /// Applies composition by the dims of `this` to the integer `values` and
  /// returns the resulting values. `this` must be symbol-less.
  SmallVector<int64_t, 4> compose(ArrayRef<int64_t> values) const;

  /// Returns true if the AffineMap represents a subset (i.e. a projection) of a
  /// symbol-less permutation map. `allowZeroInResults` allows projected
  /// permutation maps with constant zero result expressions.
  /// TODO: Remove `allowZeroInResults` when constant zero result expressions
  /// are broadly supported.
  bool isProjectedPermutation(bool allowZeroInResults = false) const;

  /// Returns true if the AffineMap represents a symbol-less permutation map.
  bool isPermutation() const;

  /// Returns the map consisting of the `resultPos` subset.
  AffineMap getSubMap(ArrayRef<unsigned> resultPos) const;

  /// Returns the map consisting of `length` expressions starting from `start`.
  AffineMap getSliceMap(unsigned start, unsigned length) const;

  /// Returns the map consisting of the most major `numResults` results.
  /// Returns the null AffineMap if `numResults` == 0.
  /// Returns `*this` if `numResults` >= `this->getNumResults()`.
  AffineMap getMajorSubMap(unsigned numResults) const;

  /// Returns the map consisting of the most minor `numResults` results.
  /// Returns the null AffineMap if `numResults` == 0.
  /// Returns `*this` if `numResults` >= `this->getNumResults()`.
  AffineMap getMinorSubMap(unsigned numResults) const;

  /// Get the largest known divisor of all map expressions.
  /// For eg: for (d0, d1) -> (8*d0 + 4, 4*d1 + 2), the result is 2.
  /// In the case of maps with no expressions or all zero constant expressions,
  /// the largest known divisor is trivially the max uint64_t value.
  uint64_t getLargestKnownDivisorOfMapExprs();

  friend ::llvm::hash_code hash_value(AffineMap arg);

  /// Methods supporting C API.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(map);
  }
  static AffineMap getFromOpaquePointer(const void *pointer) {
    return AffineMap(reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

private:
  ImplType *map{nullptr};

  static AffineMap getImpl(unsigned dimCount, unsigned symbolCount,
                           ArrayRef<AffineExpr> results, MLIRContext *context);
};

// Make AffineExpr hashable.
inline ::llvm::hash_code hash_value(AffineMap arg) {
  return ::llvm::hash_value(arg.map);
}

/// A mutable affine map. Its affine expressions are however unique.
struct MutableAffineMap {
public:
  MutableAffineMap() = default;
  MutableAffineMap(AffineMap map);

  ArrayRef<AffineExpr> getResults() const { return results; }
  AffineExpr getResult(unsigned idx) const { return results[idx]; }
  void setResult(unsigned idx, AffineExpr result) { results[idx] = result; }
  unsigned getNumResults() const { return results.size(); }
  unsigned getNumDims() const { return numDims; }
  void setNumDims(unsigned d) { numDims = d; }
  unsigned getNumSymbols() const { return numSymbols; }
  void setNumSymbols(unsigned d) { numSymbols = d; }
  MLIRContext *getContext() const { return context; }

  /// Returns true if the idx'th result expression is a multiple of factor.
  bool isMultipleOf(unsigned idx, int64_t factor) const;

  /// Resets this MutableAffineMap with 'map'.
  void reset(AffineMap map);

  /// Simplify the (result) expressions in this map using analysis (used by
  //-simplify-affine-expr pass).
  void simplify();
  /// Get the AffineMap corresponding to this MutableAffineMap. Note that an
  /// AffineMap will be uniqued and stored in context, while a mutable one
  /// isn't.
  AffineMap getAffineMap() const;

private:
  // Same meaning as AffineMap's fields.
  SmallVector<AffineExpr, 8> results;
  unsigned numDims = 0;
  unsigned numSymbols = 0;
  /// A pointer to the IR's context to store all newly created
  /// AffineExprStorage's.
  MLIRContext *context = nullptr;
};

/// Simplifies an affine map by simplifying its underlying AffineExpr results.
AffineMap simplifyAffineMap(AffineMap map);

/// Drop the dims that are listed in `unusedDims`.
AffineMap compressDims(AffineMap map, const llvm::SmallBitVector &unusedDims);

/// Drop the dims that are not used.
AffineMap compressUnusedDims(AffineMap map);

/// Drop the dims that are not used by any of the individual maps in `maps`.
/// Asserts that all maps in `maps` are normalized to the same number of
/// dims and symbols.
SmallVector<AffineMap> compressUnusedDims(ArrayRef<AffineMap> maps);

/// Drop the symbols that are listed in `unusedSymbols`.
AffineMap compressSymbols(AffineMap map,
                          const llvm::SmallBitVector &unusedSymbols);

/// Drop the symbols that are not used.
AffineMap compressUnusedSymbols(AffineMap map);

/// Drop the symbols that are not used by any of the individual maps in `maps`.
/// Asserts that all maps in `maps` are normalized to the same number of
/// dims and symbols.
SmallVector<AffineMap> compressUnusedSymbols(ArrayRef<AffineMap> maps);

/// Fold all attributes among the given operands into the affine map. Return the
/// folded affine map. Return all remaining values via `remainingValues`.
AffineMap foldAttributesIntoMap(Builder &b, AffineMap map,
                                ArrayRef<OpFoldResult> operands,
                                SmallVector<Value> &remainingValues);

/// Returns a map with the same dimension and symbol count as `map`, but whose
/// results are the unique affine expressions of `map`.
AffineMap removeDuplicateExprs(AffineMap map);

/// Returns a map of codomain to domain dimensions such that the first codomain
/// dimension for a particular domain dimension is selected.
/// Returns an empty map if the input map is empty.
/// Returns null map (not empty map) if `map` is not invertible (i.e. `map` does
/// not contain a subset that is a permutation of full domain rank).
///
/// Prerequisites:
///   1. `map` has no symbols.
///
/// Example 1:
///
/// ```mlir
///    (d0, d1, d2) -> (d1, d1, d0, d2, d1, d2, d1, d0)
///                      0       2   3
/// ```
///
/// returns:
///
/// ```mlir
///    (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
/// ```
///
/// Example 2:
///
/// ```mlir
///    (d0, d1, d2) -> (d1, d0 + d1, d0, d2, d1, d2, d1, d0)
///                      0            2   3
/// ```
///
/// returns:
///
/// ```mlir
///    (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
/// ```
AffineMap inversePermutation(AffineMap map);

/// Return the reverse map of a projected permutation where the projected
/// dimensions are transformed into 0s.
///
/// Prerequisites: `map` must be a projected permutation.
///
/// Example 1:
///
/// ```mlir
///    affine_map<(d0, d1, d2, d3) -> (d2, d0)>
/// ```
///
/// returns:
///
/// ```mlir
///    affine_map<(d0, d1) -> (d1, 0, d0, 0)>
/// ```
///
/// Example 2:
///
/// ```mlir
///    affine_map<(d0, d1, d2, d3) -> (d0, d3)>
/// ```
///
/// returns:
///
/// ```mlir
///    affine_map<(d0, d1) -> (d0, 0, 0, d1)>
/// ```
///
/// Example 3:
///
/// ```mlir
///    affine_map<(d0, d1, d2, d3) -> (d2)>
/// ```
///
/// returns:
///
/// ```mlir
///    affine_map<(d0) -> (0, 0, d0, 0)>
/// ```
/// Example 4:
///
/// ```mlir
///    affine_map<(d0, d1, d2) -> (d0, 0)>
/// ```
///
/// returns:
///
/// ```mlir
///    affine_map<(d0, d1) -> (d0, 0, 0)>
/// ```
AffineMap inverseAndBroadcastProjectedPermutation(AffineMap map);

/// Concatenates a list of `maps` into a single AffineMap, stepping over
/// potentially empty maps. Assumes each of the underlying map has 0 symbols.
/// The resulting map has a number of dims equal to the max of `maps`' dims and
/// the concatenated results as its results.
/// Returns an empty map if all input `maps` are empty.
///
/// Example:
/// When applied to the following list of 3 affine maps,
///
/// ```mlir
///    {
///      (i, j, k) -> (i, k),
///      (i, j, k) -> (k, j),
///      (i, j, k) -> (i, j)
///    }
/// ```
///
/// Returns the map:
///
/// ```mlir
///     (i, j, k) -> (i, k, k, j, i, j)
/// ```
AffineMap concatAffineMaps(ArrayRef<AffineMap> maps);

/// Returns the map that results from projecting out the dimensions specified in
/// `projectedDimensions`. The projected dimensions are set to 0.
///
/// Example:
/// 1) map                  : affine_map<(d0, d1, d2) -> (d0, d1)>
///    projected_dimensions : {2}
///    result               : affine_map<(d0, d1) -> (d0, d1)>
///
/// 2) map                  : affine_map<(d0, d1) -> (d0 + d1)>
///    projected_dimensions : {1}
///    result               : affine_map<(d0) -> (d0)>
///
/// 3) map                  : affine_map<(d0, d1, d2) -> (d0, d1)>
///    projected_dimensions : {1}
///    result               : affine_map<(d0, d1) -> (d0, 0)>
///
/// This function also compresses the dims when the boolean flag is true.
AffineMap projectDims(AffineMap map,
                      const llvm::SmallBitVector &projectedDimensions,
                      bool compressDimsFlag = false);
/// Symbol counterpart of `projectDims`.
/// This function also compresses the symbols when the boolean flag is true.
AffineMap projectSymbols(AffineMap map,
                         const llvm::SmallBitVector &projectedSymbols,
                         bool compressSymbolsFlag = false);
/// Calls `projectDims(map, projectedDimensions, compressDimsFlag)`.
/// If `compressSymbolsFlag` is true, additionally call `compressUnusedSymbols`.
AffineMap getProjectedMap(AffineMap map,
                          const llvm::SmallBitVector &projectedDimensions,
                          bool compressDimsFlag = true,
                          bool compressSymbolsFlag = true);

// Return a bitvector where each bit set indicates a dimension that is not used
// by any of the maps in the input array `maps`.
llvm::SmallBitVector getUnusedDimsBitVector(ArrayRef<AffineMap> maps);

// Return a bitvector where each bit set indicates a symbol that is not used
// by any of the maps in the input array `maps`.
llvm::SmallBitVector getUnusedSymbolsBitVector(ArrayRef<AffineMap> maps);

/// Expand `map` to operate on `rank` dims while projecting out the dims in
/// `projectedDimensions`. This amounts to composing `map` with
/// `id(rank).dropResults(projectedDimensions)`.
AffineMap expandDimsToRank(AffineMap map, int64_t rank,
                           const llvm::SmallBitVector &projectedDimensions);

inline raw_ostream &operator<<(raw_ostream &os, AffineMap map) {
  map.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// Templated helper functions.
//===----------------------------------------------------------------------===//

/// Apply a permutation from `map` to `source` and return the result.
template <typename T>
SmallVector<T> applyPermutationMap(AffineMap map, llvm::ArrayRef<T> source) {
  assert(map.isProjectedPermutation());
  assert(map.getNumInputs() == source.size());
  SmallVector<T> result;
  result.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      result.push_back(source[dimExpr.getPosition()]);
    } else if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
      assert(constExpr.getValue() == 0 &&
             "Unexpected constant in projected permutation map");
      result.push_back(0);
    } else {
      llvm_unreachable("Unexpected result in projected permutation map");
    }
  }
  return result;
}

/// Calculates maximum dimension and symbol positions from the expressions
/// in `exprsLists` and stores them in `maxDim` and `maxSym` respectively.
template <typename AffineExprContainer>
static void getMaxDimAndSymbol(ArrayRef<AffineExprContainer> exprsList,
                               int64_t &maxDim, int64_t &maxSym) {
  for (const auto &exprs : exprsList) {
    for (auto expr : exprs) {
      expr.walk([&maxDim, &maxSym](AffineExpr e) {
        if (auto d = dyn_cast<AffineDimExpr>(e))
          maxDim = std::max(maxDim, static_cast<int64_t>(d.getPosition()));
        if (auto s = dyn_cast<AffineSymbolExpr>(e))
          maxSym = std::max(maxSym, static_cast<int64_t>(s.getPosition()));
      });
    }
  }
}

} // namespace mlir

namespace llvm {

// AffineExpr hash just like pointers
template <>
struct DenseMapInfo<mlir::AffineMap> {
  static mlir::AffineMap getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::AffineMap(static_cast<mlir::AffineMap::ImplType *>(pointer));
  }
  static mlir::AffineMap getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::AffineMap(static_cast<mlir::AffineMap::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::AffineMap val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::AffineMap LHS, mlir::AffineMap RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // MLIR_IR_AFFINEMAP_H
