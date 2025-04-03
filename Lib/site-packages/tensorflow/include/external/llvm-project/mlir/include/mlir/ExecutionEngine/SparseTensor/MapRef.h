//===- MapRef.h - A dim2lvl/lvl2dim map encoding ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A dim2lvl/lvl2dim map encoding class, with utility methods.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_MAPREF_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_MAPREF_H

#include <cinttypes>

#include <cassert>
#include <vector>

namespace mlir {
namespace sparse_tensor {

/// A class for capturing the sparse tensor type map with a compact encoding.
///
/// Currently, the following situations are supported:
///   (1) map is a permutation
///   (2) map has affine ops (restricted set)
///
/// The pushforward/backward operations are fast for (1) but incur some obvious
/// overhead for situation (2).
///
class MapRef final {
public:
  MapRef(uint64_t d, uint64_t l, const uint64_t *d2l, const uint64_t *l2d);

  //
  // Push forward maps from dimensions to levels.
  //

  // Map from dimRank in to lvlRank out.
  template <typename T>
  inline void pushforward(const T *in, T *out) const {
    if (isPermutation) {
      for (uint64_t l = 0; l < lvlRank; l++) {
        out[l] = in[dim2lvl[l]];
      }
    } else {
      uint64_t i, c;
      for (uint64_t l = 0; l < lvlRank; l++)
        if (isFloor(l, i, c)) {
          out[l] = in[i] / c;
        } else if (isMod(l, i, c)) {
          out[l] = in[i] % c;
        } else {
          out[l] = in[dim2lvl[l]];
        }
    }
  }

  //
  // Push backward maps from levels to dimensions.
  //

  // Map from lvlRank in to dimRank out.
  template <typename T>
  inline void pushbackward(const T *in, T *out) const {
    if (isPermutation) {
      for (uint64_t d = 0; d < dimRank; d++)
        out[d] = in[lvl2dim[d]];
    } else {
      uint64_t i, c, ii;
      for (uint64_t d = 0; d < dimRank; d++)
        if (isMul(d, i, c, ii)) {
          out[d] = in[i] + c * in[ii];
        } else {
          out[d] = in[lvl2dim[d]];
        }
    }
  }

  uint64_t getDimRank() const { return dimRank; }
  uint64_t getLvlRank() const { return lvlRank; }

private:
  bool isPermutationMap() const;

  bool isFloor(uint64_t l, uint64_t &i, uint64_t &c) const;
  bool isMod(uint64_t l, uint64_t &i, uint64_t &c) const;
  bool isMul(uint64_t d, uint64_t &i, uint64_t &c, uint64_t &ii) const;

  const uint64_t dimRank;
  const uint64_t lvlRank;
  const uint64_t *const dim2lvl; // non-owning pointer
  const uint64_t *const lvl2dim; // non-owning pointer
  const bool isPermutation;
};

} // namespace sparse_tensor
} // namespace mlir

#endif //  MLIR_EXECUTIONENGINE_SPARSETENSOR_MAPREF_H
