//===- IntegerSetDetail.h - MLIR IntegerSet storage details -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of IntegerSet.
//
//===----------------------------------------------------------------------===//

#ifndef INTEGERSETDETAIL_H_
#define INTEGERSETDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace detail {

struct IntegerSetStorage : public StorageUniquer::BaseStorage {
  /// The hash key used for uniquing.
  using KeyTy =
      std::tuple<unsigned, unsigned, ArrayRef<AffineExpr>, ArrayRef<bool>>;

  unsigned dimCount;
  unsigned symbolCount;

  /// Array of affine constraints: a constraint is either an equality
  /// (affine_expr == 0) or an inequality (affine_expr >= 0).
  ArrayRef<AffineExpr> constraints;

  // Bits to check whether a constraint is an equality or an inequality.
  ArrayRef<bool> eqFlags;

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == dimCount && std::get<1>(key) == symbolCount &&
           std::get<2>(key) == constraints && std::get<3>(key) == eqFlags;
  }

  static IntegerSetStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *res =
        new (allocator.allocate<IntegerSetStorage>()) IntegerSetStorage();
    res->dimCount = std::get<0>(key);
    res->symbolCount = std::get<1>(key);
    res->constraints = allocator.copyInto(std::get<2>(key));
    res->eqFlags = allocator.copyInto(std::get<3>(key));
    return res;
  }
};

} // namespace detail
} // namespace mlir
#endif // INTEGERSETDETAIL_H_
