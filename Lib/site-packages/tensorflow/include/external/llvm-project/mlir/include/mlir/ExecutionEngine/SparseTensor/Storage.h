//===- Storage.h - TACO-flavored sparse tensor representation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the following classes:
//
// * `SparseTensorStorageBase`
// * `SparseTensorStorage<P, C, V>`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/SparseTensor/ArithmeticUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/MapRef.h"

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
//
//  SparseTensorStorage Classes
//
//===----------------------------------------------------------------------===//

/// Abstract base class for `SparseTensorStorage<P,C,V>`. This class
/// takes responsibility for all the `<P,C,V>`-independent aspects
/// of the tensor (e.g., sizes, sparsity, mapping). In addition,
/// we use function overloading to implement "partial" method
/// specialization, which the C-API relies on to catch type errors
/// arising from our use of opaque pointers.
///
/// Because this class forms a bridge between the denotational semantics
/// of "tensors" and the operational semantics of how we store and
/// compute with them, it also distinguishes between two different
/// coordinate spaces (and their associated rank, sizes, etc).
/// Denotationally, we have the *dimensions* of the tensor represented
/// by this object.  Operationally, we have the *levels* of the storage
/// representation itself.
///
/// The *size* of an axis is the cardinality of possible coordinate
/// values along that axis (regardless of which coordinates have stored
/// element values). As such, each size must be non-zero since if any
/// axis has size-zero then the whole tensor would have trivial storage
/// (since there are no possible coordinates). Thus we use the plural
/// term *sizes* for a collection of non-zero cardinalities, and use
/// this term whenever referring to run-time cardinalities. Whereas we
/// use the term *shape* for a collection of compile-time cardinalities,
/// where zero is used to indicate cardinalities which are dynamic (i.e.,
/// unknown/unspecified at compile-time). At run-time, these dynamic
/// cardinalities will be inferred from or checked against sizes otherwise
/// specified. Thus, dynamic cardinalities always have an "immutable but
/// unknown" value; so the term "dynamic" should not be taken to indicate
/// run-time mutability.
class SparseTensorStorageBase {
protected:
  SparseTensorStorageBase(const SparseTensorStorageBase &) = default;
  SparseTensorStorageBase &operator=(const SparseTensorStorageBase &) = delete;

public:
  /// Constructs a new sparse-tensor storage object with the given encoding.
  SparseTensorStorageBase(uint64_t dimRank, const uint64_t *dimSizes,
                          uint64_t lvlRank, const uint64_t *lvlSizes,
                          const LevelType *lvlTypes, const uint64_t *dim2lvl,
                          const uint64_t *lvl2dim);
  virtual ~SparseTensorStorageBase() = default;

  /// Gets the number of tensor-dimensions.
  uint64_t getDimRank() const { return dimSizes.size(); }

  /// Gets the number of storage-levels.
  uint64_t getLvlRank() const { return lvlSizes.size(); }

  /// Gets the tensor-dimension sizes array.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Safely looks up the size of the given tensor-dimension.
  uint64_t getDimSize(uint64_t d) const {
    assert(d < getDimRank());
    return dimSizes[d];
  }

  /// Gets the storage-level sizes array.
  const std::vector<uint64_t> &getLvlSizes() const { return lvlSizes; }

  /// Safely looks up the size of the given storage-level.
  uint64_t getLvlSize(uint64_t l) const {
    assert(l < getLvlRank());
    return lvlSizes[l];
  }

  /// Gets the level-types array.
  const std::vector<LevelType> &getLvlTypes() const { return lvlTypes; }

  /// Safely looks up the type of the given level.
  LevelType getLvlType(uint64_t l) const {
    assert(l < getLvlRank());
    return lvlTypes[l];
  }

  /// Safely checks if the level uses dense storage.
  bool isDenseLvl(uint64_t l) const { return isDenseLT(getLvlType(l)); }

  /// Safely checks if the level uses compressed storage.
  bool isCompressedLvl(uint64_t l) const {
    return isCompressedLT(getLvlType(l));
  }

  /// Safely checks if the level uses loose compressed storage.
  bool isLooseCompressedLvl(uint64_t l) const {
    return isLooseCompressedLT(getLvlType(l));
  }

  /// Safely checks if the level uses singleton storage.
  bool isSingletonLvl(uint64_t l) const { return isSingletonLT(getLvlType(l)); }

  /// Safely checks if the level uses n out of m storage.
  bool isNOutOfMLvl(uint64_t l) const { return isNOutOfMLT(getLvlType(l)); }

  /// Safely checks if the level is ordered.
  bool isOrderedLvl(uint64_t l) const { return isOrderedLT(getLvlType(l)); }

  /// Safely checks if the level is unique.
  bool isUniqueLvl(uint64_t l) const { return isUniqueLT(getLvlType(l)); }

  /// Gets positions-overhead storage for the given level.
#define DECL_GETPOSITIONS(PNAME, P)                                            \
  virtual void getPositions(std::vector<P> **, uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DECL_GETPOSITIONS)
#undef DECL_GETPOSITIONS

  /// Gets coordinates-overhead storage for the given level.
#define DECL_GETCOORDINATES(INAME, C)                                          \
  virtual void getCoordinates(std::vector<C> **, uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DECL_GETCOORDINATES)
#undef DECL_GETCOORDINATES

  /// Gets coordinates-overhead storage buffer for the given level.
#define DECL_GETCOORDINATESBUFFER(INAME, C)                                    \
  virtual void getCoordinatesBuffer(std::vector<C> **, uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_FIXED_O(DECL_GETCOORDINATESBUFFER)
#undef DECL_GETCOORDINATESBUFFER

  /// Gets primary storage.
#define DECL_GETVALUES(VNAME, V) virtual void getValues(std::vector<V> **);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_GETVALUES)
#undef DECL_GETVALUES

  /// Element-wise insertion in lexicographic coordinate order. The first
  /// argument is the level-coordinates for the value being inserted.
#define DECL_LEXINSERT(VNAME, V) virtual void lexInsert(const uint64_t *, V);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_LEXINSERT)
#undef DECL_LEXINSERT

  /// Expanded insertion.  Note that this method resets the
  /// values/filled-switch array back to all-zero/false while only
  /// iterating over the nonzero elements.
#define DECL_EXPINSERT(VNAME, V)                                               \
  virtual void expInsert(uint64_t *, V *, bool *, uint64_t *, uint64_t,        \
                         uint64_t);
  MLIR_SPARSETENSOR_FOREVERY_V(DECL_EXPINSERT)
#undef DECL_EXPINSERT

  /// Finalizes lexicographic insertions.
  virtual void endLexInsert() = 0;

private:
  const std::vector<uint64_t> dimSizes;
  const std::vector<uint64_t> lvlSizes;
  const std::vector<LevelType> lvlTypes;
  const std::vector<uint64_t> dim2lvlVec;
  const std::vector<uint64_t> lvl2dimVec;

protected:
  const MapRef map; // non-owning pointers into dim2lvl/lvl2dim vectors
  const bool allDense;
};

/// A memory-resident sparse tensor using a storage scheme based on
/// per-level sparse/dense annotations. This data structure provides
/// a bufferized form of a sparse tensor type. In contrast to generating
/// setup methods for each differently annotated sparse tensor, this
/// method provides a convenient "one-size-fits-all" solution that simply
/// takes an input tensor and annotations to implement all required setup
/// in a general manner.
template <typename P, typename C, typename V>
class SparseTensorStorage final : public SparseTensorStorageBase {
  /// Private constructor to share code between the other constructors.
  /// Beware that the object is not necessarily guaranteed to be in a
  /// valid state after this constructor alone; e.g., `isCompressedLvl(l)`
  /// doesn't entail `!(positions[l].empty())`.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const LevelType *lvlTypes, const uint64_t *dim2lvl,
                      const uint64_t *lvl2dim)
      : SparseTensorStorageBase(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                                dim2lvl, lvl2dim),
        positions(lvlRank), coordinates(lvlRank), lvlCursor(lvlRank) {}

public:
  /// Constructs a sparse tensor with the given encoding, and allocates
  /// overhead storage according to some simple heuristics. When lvlCOO
  /// is set, the sparse tensor initializes with the contents from that
  /// data structure. Otherwise, an empty sparse tensor results.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const LevelType *lvlTypes, const uint64_t *dim2lvl,
                      const uint64_t *lvl2dim, SparseTensorCOO<V> *lvlCOO);

  /// Constructs a sparse tensor with the given encoding, and initializes
  /// the contents from the level buffers. The constructor assumes that the
  /// data provided by `lvlBufs` can be directly used to interpret the result
  /// sparse tensor and performs no integrity test on the input data.
  SparseTensorStorage(uint64_t dimRank, const uint64_t *dimSizes,
                      uint64_t lvlRank, const uint64_t *lvlSizes,
                      const LevelType *lvlTypes, const uint64_t *dim2lvl,
                      const uint64_t *lvl2dim, const intptr_t *lvlBufs);

  /// Allocates a new empty sparse tensor.
  static SparseTensorStorage<P, C, V> *
  newEmpty(uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
           const uint64_t *lvlSizes, const LevelType *lvlTypes,
           const uint64_t *dim2lvl, const uint64_t *lvl2dim);

  /// Allocates a new sparse tensor and initializes it from the given COO.
  static SparseTensorStorage<P, C, V> *
  newFromCOO(uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
             const uint64_t *lvlSizes, const LevelType *lvlTypes,
             const uint64_t *dim2lvl, const uint64_t *lvl2dim,
             SparseTensorCOO<V> *lvlCOO);

  /// Allocates a new sparse tensor and initialize it from the given buffers.
  static SparseTensorStorage<P, C, V> *
  newFromBuffers(uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
                 const uint64_t *lvlSizes, const LevelType *lvlTypes,
                 const uint64_t *dim2lvl, const uint64_t *lvl2dim,
                 uint64_t srcRank, const intptr_t *buffers);

  ~SparseTensorStorage() final = default;

  /// Partially specialize these getter methods based on template types.
  void getPositions(std::vector<P> **out, uint64_t lvl) final {
    assert(out && "Received nullptr for out parameter");
    assert(lvl < getLvlRank());
    *out = &positions[lvl];
  }
  void getCoordinates(std::vector<C> **out, uint64_t lvl) final {
    assert(out && "Received nullptr for out parameter");
    assert(lvl < getLvlRank());
    *out = &coordinates[lvl];
  }
  void getCoordinatesBuffer(std::vector<C> **out, uint64_t lvl) final {
    assert(out && "Received nullptr for out parameter");
    assert(lvl < getLvlRank());
    // Note that the sparse tensor support library always stores COO in SoA
    // format, even when AoS is requested. This is never an issue, since all
    // actual code/library generation requests "views" into the coordinate
    // storage for the individual levels, which is trivially provided for
    // both AoS and SoA (as well as all the other storage formats). The only
    // exception is when the buffer version of coordinate storage is requested
    // (currently only for printing). In that case, we do the following
    // potentially expensive transformation to provide that view. If this
    // operation becomes more common beyond debugging, we should consider
    // implementing proper AoS in the support library as well.
    uint64_t lvlRank = getLvlRank();
    uint64_t nnz = values.size();
    crdBuffer.clear();
    crdBuffer.reserve(nnz * (lvlRank - lvl));
    for (uint64_t i = 0; i < nnz; i++) {
      for (uint64_t l = lvl; l < lvlRank; l++) {
        assert(i < coordinates[l].size());
        crdBuffer.push_back(coordinates[l][i]);
      }
    }
    *out = &crdBuffer;
  }
  void getValues(std::vector<V> **out) final {
    assert(out && "Received nullptr for out parameter");
    *out = &values;
  }

  /// Partially specialize lexicographical insertions based on template types.
  void lexInsert(const uint64_t *lvlCoords, V val) final {
    assert(lvlCoords);
    if (allDense) {
      uint64_t lvlRank = getLvlRank();
      uint64_t valIdx = 0;
      // Linearize the address.
      for (uint64_t l = 0; l < lvlRank; l++)
        valIdx = valIdx * getLvlSize(l) + lvlCoords[l];
      values[valIdx] = val;
      return;
    }
    // First, wrap up pending insertion path.
    uint64_t diffLvl = 0;
    uint64_t full = 0;
    if (!values.empty()) {
      diffLvl = lexDiff(lvlCoords);
      endPath(diffLvl + 1);
      full = lvlCursor[diffLvl] + 1;
    }
    // Then continue with insertion path.
    insPath(lvlCoords, diffLvl, full, val);
  }

  /// Partially specialize expanded insertions based on template types.
  void expInsert(uint64_t *lvlCoords, V *values, bool *filled, uint64_t *added,
                 uint64_t count, uint64_t expsz) final {
    assert((lvlCoords && values && filled && added) && "Received nullptr");
    if (count == 0)
      return;
    // Sort.
    std::sort(added, added + count);
    // Restore insertion path for first insert.
    const uint64_t lastLvl = getLvlRank() - 1;
    uint64_t c = added[0];
    assert(c <= expsz);
    assert(filled[c] && "added coordinate is not filled");
    lvlCoords[lastLvl] = c;
    lexInsert(lvlCoords, values[c]);
    values[c] = 0;
    filled[c] = false;
    // Subsequent insertions are quick.
    for (uint64_t i = 1; i < count; i++) {
      assert(c < added[i] && "non-lexicographic insertion");
      c = added[i];
      assert(c <= expsz);
      assert(filled[c] && "added coordinate is not filled");
      lvlCoords[lastLvl] = c;
      insPath(lvlCoords, lastLvl, added[i - 1] + 1, values[c]);
      values[c] = 0;
      filled[c] = false;
    }
  }

  /// Finalizes lexicographic insertions.
  void endLexInsert() final {
    if (!allDense) {
      if (values.empty())
        finalizeSegment(0);
      else
        endPath(0);
    }
  }

  /// Sort the unordered tensor in place, the method assumes that it is
  /// an unordered COO tensor.
  void sortInPlace() {
    uint64_t nnz = values.size();
#ifndef NDEBUG
    for (uint64_t l = 0; l < getLvlRank(); l++)
      assert(nnz == coordinates[l].size());
#endif

    // In-place permutation.
    auto applyPerm = [this](std::vector<uint64_t> &perm) {
      uint64_t length = perm.size();
      uint64_t lvlRank = getLvlRank();
      // Cache for the current level coordinates.
      std::vector<P> lvlCrds(lvlRank);
      for (uint64_t i = 0; i < length; i++) {
        uint64_t current = i;
        if (i != perm[current]) {
          for (uint64_t l = 0; l < lvlRank; l++)
            lvlCrds[l] = coordinates[l][i];
          V val = values[i];
          // Deals with a permutation cycle.
          while (i != perm[current]) {
            uint64_t next = perm[current];
            // Swaps the level coordinates and value.
            for (uint64_t l = 0; l < lvlRank; l++)
              coordinates[l][current] = coordinates[l][next];
            values[current] = values[next];
            perm[current] = current;
            current = next;
          }
          for (uint64_t l = 0; l < lvlRank; l++)
            coordinates[l][current] = lvlCrds[l];
          values[current] = val;
          perm[current] = current;
        }
      }
    };

    std::vector<uint64_t> sortedIdx(nnz, 0);
    for (uint64_t i = 0; i < nnz; i++)
      sortedIdx[i] = i;

    std::sort(sortedIdx.begin(), sortedIdx.end(),
              [this](uint64_t lhs, uint64_t rhs) {
                for (uint64_t l = 0; l < getLvlRank(); l++) {
                  if (coordinates[l][lhs] == coordinates[l][rhs])
                    continue;
                  return coordinates[l][lhs] < coordinates[l][rhs];
                }
                assert(lhs == rhs && "duplicate coordinates");
                return false;
              });

    applyPerm(sortedIdx);
  }

private:
  /// Appends coordinate `crd` to level `lvl`, in the semantically
  /// general sense.  For non-dense levels, that means appending to the
  /// `coordinates[lvl]` array, checking that `crd` is representable in
  /// the `C` type; however, we do not verify other semantic requirements
  /// (e.g., that `crd` is in bounds for `lvlSizes[lvl]`, and not previously
  /// occurring in the same segment).  For dense levels, this method instead
  /// appends the appropriate number of zeros to the `values` array, where
  /// `full` is the number of "entries" already written to `values` for this
  /// segment (aka one after the highest coordinate previously appended).
  void appendCrd(uint64_t lvl, uint64_t full, uint64_t crd) {
    if (!isDenseLvl(lvl)) {
      assert(isCompressedLvl(lvl) || isLooseCompressedLvl(lvl) ||
             isSingletonLvl(lvl) || isNOutOfMLvl(lvl));
      coordinates[lvl].push_back(detail::checkOverflowCast<C>(crd));
    } else { // Dense level.
      assert(crd >= full && "Coordinate was already filled");
      if (crd == full)
        return; // Short-circuit, since it'll be a nop.
      if (lvl + 1 == getLvlRank())
        values.insert(values.end(), crd - full, 0);
      else
        finalizeSegment(lvl + 1, 0, crd - full);
    }
  }

  /// Computes the assembled-size associated with the `l`-th level,
  /// given the assembled-size associated with the `(l-1)`-th level.
  uint64_t assembledSize(uint64_t parentSz, uint64_t l) const {
    if (isCompressedLvl(l))
      return positions[l][parentSz];
    if (isLooseCompressedLvl(l))
      return positions[l][2 * parentSz - 1];
    if (isSingletonLvl(l) || isNOutOfMLvl(l))
      return parentSz; // new size same as the parent
    assert(isDenseLvl(l));
    return parentSz * getLvlSize(l);
  }

  /// Initializes sparse tensor storage scheme from a memory-resident sparse
  /// tensor in coordinate scheme. This method prepares the positions and
  /// coordinates arrays under the given per-level dense/sparse annotations.
  void fromCOO(const std::vector<Element<V>> &lvlElements, uint64_t lo,
               uint64_t hi, uint64_t l) {
    const uint64_t lvlRank = getLvlRank();
    assert(l <= lvlRank && hi <= lvlElements.size());
    // Once levels are exhausted, insert the numerical values.
    if (l == lvlRank) {
      assert(lo < hi);
      values.push_back(lvlElements[lo].value);
      return;
    }
    // Visit all elements in this interval.
    uint64_t full = 0;
    while (lo < hi) { // If `hi` is unchanged, then `lo < lvlElements.size()`.
      // Find segment in interval with same coordinate at this level.
      const uint64_t c = lvlElements[lo].coords[l];
      uint64_t seg = lo + 1;
      if (isUniqueLvl(l))
        while (seg < hi && lvlElements[seg].coords[l] == c)
          seg++;
      // Handle segment in interval for sparse or dense level.
      appendCrd(l, full, c);
      full = c + 1;
      fromCOO(lvlElements, lo, seg, l + 1);
      // And move on to next segment in interval.
      lo = seg;
    }
    // Finalize the sparse position structure at this level.
    finalizeSegment(l, full);
  }

  /// Finalizes the sparse position structure at this level.
  void finalizeSegment(uint64_t l, uint64_t full = 0, uint64_t count = 1) {
    if (count == 0)
      return; // Short-circuit, since it'll be a nop.
    if (isCompressedLvl(l)) {
      uint64_t pos = coordinates[l].size();
      positions[l].insert(positions[l].end(), count,
                          detail::checkOverflowCast<P>(pos));
    } else if (isLooseCompressedLvl(l)) {
      // Finish this level, and push pairs for the empty ones, and one
      // more for next level. Note that this always leaves one extra
      // unused element at the end.
      uint64_t pos = coordinates[l].size();
      positions[l].insert(positions[l].end(), 2 * count,
                          detail::checkOverflowCast<P>(pos));
    } else if (isSingletonLvl(l) || isNOutOfMLvl(l)) {
      return; // Nothing to finalize.
    } else {  // Dense dimension.
      assert(isDenseLvl(l));
      const uint64_t sz = getLvlSizes()[l];
      assert(sz >= full && "Segment is overfull");
      count = detail::checkedMul(count, sz - full);
      // For dense storage we must enumerate all the remaining coordinates
      // in this level (i.e., coordinates after the last non-zero
      // element), and either fill in their zero values or else recurse
      // to finalize some deeper level.
      if (l + 1 == getLvlRank())
        values.insert(values.end(), count, 0);
      else
        finalizeSegment(l + 1, 0, count);
    }
  }

  /// Wraps up a single insertion path, inner to outer.
  void endPath(uint64_t diffLvl) {
    const uint64_t lvlRank = getLvlRank();
    const uint64_t lastLvl = lvlRank - 1;
    assert(diffLvl <= lvlRank);
    const uint64_t stop = lvlRank - diffLvl;
    for (uint64_t i = 0; i < stop; i++) {
      const uint64_t l = lastLvl - i;
      finalizeSegment(l, lvlCursor[l] + 1);
    }
  }

  /// Continues a single insertion path, outer to inner. The first
  /// argument is the level-coordinates for the value being inserted.
  void insPath(const uint64_t *lvlCoords, uint64_t diffLvl, uint64_t full,
               V val) {
    const uint64_t lvlRank = getLvlRank();
    assert(diffLvl <= lvlRank);
    for (uint64_t l = diffLvl; l < lvlRank; l++) {
      const uint64_t c = lvlCoords[l];
      appendCrd(l, full, c);
      full = 0;
      lvlCursor[l] = c;
    }
    values.push_back(val);
  }

  /// Finds the lexicographically first level where the level-coordinates
  /// in the argument differ from those in the current cursor.
  uint64_t lexDiff(const uint64_t *lvlCoords) const {
    const uint64_t lvlRank = getLvlRank();
    for (uint64_t l = 0; l < lvlRank; l++) {
      const auto crd = lvlCoords[l];
      const auto cur = lvlCursor[l];
      if (crd > cur || (crd == cur && !isUniqueLvl(l)) ||
          (crd < cur && !isOrderedLvl(l))) {
        return l;
      }
      if (crd < cur) {
        assert(false && "non-lexicographic insertion");
        return -1u;
      }
    }
    assert(false && "duplicate insertion");
    return -1u;
  }

  // Sparse tensor storage components.
  std::vector<std::vector<P>> positions;
  std::vector<std::vector<C>> coordinates;
  std::vector<V> values;

  // Auxiliary data structures.
  std::vector<uint64_t> lvlCursor;
  std::vector<C> crdBuffer; // just for AoS view
};

//===----------------------------------------------------------------------===//
//
//  SparseTensorStorage Factories
//
//===----------------------------------------------------------------------===//

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::newEmpty(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const LevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim) {
  SparseTensorCOO<V> *noLvlCOO = nullptr;
  return new SparseTensorStorage<P, C, V>(dimRank, dimSizes, lvlRank, lvlSizes,
                                          lvlTypes, dim2lvl, lvl2dim, noLvlCOO);
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::newFromCOO(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const LevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim,
    SparseTensorCOO<V> *lvlCOO) {
  assert(lvlCOO);
  return new SparseTensorStorage<P, C, V>(dimRank, dimSizes, lvlRank, lvlSizes,
                                          lvlTypes, dim2lvl, lvl2dim, lvlCOO);
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V> *SparseTensorStorage<P, C, V>::newFromBuffers(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const LevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim, uint64_t srcRank,
    const intptr_t *buffers) {
  return new SparseTensorStorage<P, C, V>(dimRank, dimSizes, lvlRank, lvlSizes,
                                          lvlTypes, dim2lvl, lvl2dim, buffers);
}

//===----------------------------------------------------------------------===//
//
//  SparseTensorStorage Constructors
//
//===----------------------------------------------------------------------===//

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V>::SparseTensorStorage(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const LevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim,
    SparseTensorCOO<V> *lvlCOO)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                          dim2lvl, lvl2dim) {
  // Provide hints on capacity of positions and coordinates.
  // TODO: needs much fine-tuning based on actual sparsity; currently
  // we reserve position/coordinate space based on all previous dense
  // levels, which works well up to first sparse level; but we should
  // really use nnz and dense/sparse distribution.
  uint64_t sz = 1;
  for (uint64_t l = 0; l < lvlRank; l++) {
    if (isCompressedLvl(l)) {
      positions[l].reserve(sz + 1);
      positions[l].push_back(0);
      coordinates[l].reserve(sz);
      sz = 1;
    } else if (isLooseCompressedLvl(l)) {
      positions[l].reserve(2 * sz + 1); // last one unused
      positions[l].push_back(0);
      coordinates[l].reserve(sz);
      sz = 1;
    } else if (isSingletonLvl(l)) {
      coordinates[l].reserve(sz);
      sz = 1;
    } else if (isNOutOfMLvl(l)) {
      assert(l == lvlRank - 1 && "unexpected n:m usage");
      sz = detail::checkedMul(sz, lvlSizes[l]) / 2;
      coordinates[l].reserve(sz);
      values.reserve(sz);
    } else { // Dense level.
      assert(isDenseLvl(l));
      sz = detail::checkedMul(sz, lvlSizes[l]);
    }
  }
  if (lvlCOO) {
    /* New from COO: ensure it is sorted. */
    assert(lvlCOO->getRank() == lvlRank);
    lvlCOO->sort();
    // Now actually insert the `elements`.
    const auto &elements = lvlCOO->getElements();
    const uint64_t nse = elements.size();
    assert(values.size() == 0);
    values.reserve(nse);
    fromCOO(elements, 0, nse, 0);
  } else if (allDense) {
    /* New empty (all dense) */
    values.resize(sz, 0);
  }
}

template <typename P, typename C, typename V>
SparseTensorStorage<P, C, V>::SparseTensorStorage(
    uint64_t dimRank, const uint64_t *dimSizes, uint64_t lvlRank,
    const uint64_t *lvlSizes, const LevelType *lvlTypes,
    const uint64_t *dim2lvl, const uint64_t *lvl2dim, const intptr_t *lvlBufs)
    : SparseTensorStorage(dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes,
                          dim2lvl, lvl2dim) {
  // Note that none of the buffers can be reused because ownership
  // of the memory passed from clients is not necessarily transferred.
  // Therefore, all data is copied over into a new SparseTensorStorage.
  uint64_t trailCOOLen = 0, parentSz = 1, bufIdx = 0;
  for (uint64_t l = 0; l < lvlRank; l++) {
    if (!isUniqueLvl(l) && (isCompressedLvl(l) || isLooseCompressedLvl(l))) {
      // A `(loose)compressed_nu` level marks the start of trailing COO
      // start level. Since the coordinate buffer used for trailing COO
      // is passed in as AoS scheme and SparseTensorStorage uses a SoA
      // scheme, we cannot simply copy the value from the provided buffers.
      trailCOOLen = lvlRank - l;
      break;
    }
    if (isCompressedLvl(l) || isLooseCompressedLvl(l)) {
      P *posPtr = reinterpret_cast<P *>(lvlBufs[bufIdx++]);
      C *crdPtr = reinterpret_cast<C *>(lvlBufs[bufIdx++]);
      if (isLooseCompressedLvl(l)) {
        positions[l].assign(posPtr, posPtr + 2 * parentSz);
        coordinates[l].assign(crdPtr, crdPtr + positions[l][2 * parentSz - 1]);
      } else {
        positions[l].assign(posPtr, posPtr + parentSz + 1);
        coordinates[l].assign(crdPtr, crdPtr + positions[l][parentSz]);
      }
    } else if (isSingletonLvl(l)) {
      assert(0 && "general singleton not supported yet");
    } else if (isNOutOfMLvl(l)) {
      assert(0 && "n ouf of m not supported yet");
    } else {
      assert(isDenseLvl(l));
    }
    parentSz = assembledSize(parentSz, l);
  }

  // Handle Aos vs. SoA mismatch for COO.
  if (trailCOOLen != 0) {
    uint64_t cooStartLvl = lvlRank - trailCOOLen;
    assert(!isUniqueLvl(cooStartLvl) &&
           (isCompressedLvl(cooStartLvl) || isLooseCompressedLvl(cooStartLvl)));
    P *posPtr = reinterpret_cast<P *>(lvlBufs[bufIdx++]);
    C *aosCrdPtr = reinterpret_cast<C *>(lvlBufs[bufIdx++]);
    P crdLen;
    if (isLooseCompressedLvl(cooStartLvl)) {
      positions[cooStartLvl].assign(posPtr, posPtr + 2 * parentSz);
      crdLen = positions[cooStartLvl][2 * parentSz - 1];
    } else {
      positions[cooStartLvl].assign(posPtr, posPtr + parentSz + 1);
      crdLen = positions[cooStartLvl][parentSz];
    }
    for (uint64_t l = cooStartLvl; l < lvlRank; l++) {
      coordinates[l].resize(crdLen);
      for (uint64_t n = 0; n < crdLen; n++) {
        coordinates[l][n] = *(aosCrdPtr + (l - cooStartLvl) + n * trailCOOLen);
      }
    }
    parentSz = assembledSize(parentSz, cooStartLvl);
  }

  // Copy the values buffer.
  V *valPtr = reinterpret_cast<V *>(lvlBufs[bufIdx]);
  values.assign(valPtr, valPtr + parentSz);
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_STORAGE_H
