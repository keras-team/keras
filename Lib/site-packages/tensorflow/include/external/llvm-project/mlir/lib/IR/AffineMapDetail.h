//===- AffineMapDetail.h - MLIR Affine Map details Class --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of AffineMap.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINEMAPDETAIL_H_
#define AFFINEMAPDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace detail {

struct AffineMapStorage final
    : public StorageUniquer::BaseStorage,
      public llvm::TrailingObjects<AffineMapStorage, AffineExpr> {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, unsigned, ArrayRef<AffineExpr>>;

  unsigned numDims;
  unsigned numSymbols;
  unsigned numResults;

  MLIRContext *context;

  /// The affine expressions for this (multi-dimensional) map.
  ArrayRef<AffineExpr> results() const {
    return {getTrailingObjects<AffineExpr>(), numResults};
  }

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == numDims && std::get<1>(key) == numSymbols &&
           std::get<2>(key) == results();
  }

  // Constructs an AffineMapStorage from a key. The context must be set by the
  // caller.
  static AffineMapStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto results = std::get<2>(key);
    auto byteSize =
        AffineMapStorage::totalSizeToAlloc<AffineExpr>(results.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(AffineMapStorage));
    auto *res = new (rawMem) AffineMapStorage();
    res->numDims = std::get<0>(key);
    res->numSymbols = std::get<1>(key);
    res->numResults = results.size();
    std::uninitialized_copy(results.begin(), results.end(),
                            res->getTrailingObjects<AffineExpr>());
    return res;
  }
};

} // namespace detail
} // namespace mlir

#endif // AFFINEMAPDETAIL_H_
