//===- AffineExprDetail.h - MLIR Affine Expr storage details ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This holds implementation details of AffineExpr. Ideally it would not be
// exposed and would be kept local to AffineExpr.cpp however, MLIRContext.cpp
// needs to know the sizes for placement-new style Allocation.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_IR_AFFINEEXPRDETAIL_H_
#define MLIR_IR_AFFINEEXPRDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {

class MLIRContext;

namespace detail {

/// Base storage class appearing in an affine expression.
struct AffineExprStorage : public StorageUniquer::BaseStorage {
  MLIRContext *context;
  AffineExprKind kind;
};

/// A binary operation appearing in an affine expression.
struct AffineBinaryOpExprStorage : public AffineExprStorage {
  using KeyTy = std::tuple<unsigned, AffineExpr, AffineExpr>;

  bool operator==(const KeyTy &key) const {
    return static_cast<AffineExprKind>(std::get<0>(key)) == kind &&
           std::get<1>(key) == lhs && std::get<2>(key) == rhs;
  }

  static AffineBinaryOpExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<AffineBinaryOpExprStorage>();
    result->kind = static_cast<AffineExprKind>(std::get<0>(key));
    result->lhs = std::get<1>(key);
    result->rhs = std::get<2>(key);
    result->context = result->lhs.getContext();
    return result;
  }

  AffineExpr lhs;
  AffineExpr rhs;
};

/// A dimensional or symbolic identifier appearing in an affine expression.
struct AffineDimExprStorage : public AffineExprStorage {
  using KeyTy = std::pair<unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<AffineExprKind>(key.first) &&
           position == key.second;
  }

  static AffineDimExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<AffineDimExprStorage>();
    result->kind = static_cast<AffineExprKind>(key.first);
    result->position = key.second;
    return result;
  }

  /// Position of this identifier in the argument list.
  unsigned position;
};

/// An integer constant appearing in affine expression.
struct AffineConstantExprStorage : public AffineExprStorage {
  using KeyTy = int64_t;

  bool operator==(const KeyTy &key) const { return constant == key; }

  static AffineConstantExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<AffineConstantExprStorage>();
    result->kind = AffineExprKind::Constant;
    result->constant = key;
    return result;
  }

  // The constant.
  int64_t constant;
};

} // namespace detail
} // namespace mlir
#endif // MLIR_IR_AFFINEEXPRDETAIL_H_
