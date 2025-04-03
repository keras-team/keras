//===- LinearTransform.h - MLIR LinearTransform Class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for linear transforms and applying them to an IntegerRelation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_LINEARTRANSFORM_H
#define MLIR_ANALYSIS_PRESBURGER_LINEARTRANSFORM_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace presburger {

class LinearTransform {
public:
  explicit LinearTransform(IntMatrix &&oMatrix);
  explicit LinearTransform(const IntMatrix &oMatrix);

  // Returns a linear transform T such that MT is M in column echelon form.
  // Also returns the number of non-zero columns in MT.
  //
  // Specifically, T is such that in every column the first non-zero row is
  // strictly below that of the previous column, and all columns which have only
  // zeros are at the end.
  static std::pair<unsigned, LinearTransform>
  makeTransformToColumnEchelon(const IntMatrix &m);

  // Returns an IntegerRelation having a constraint vector vT for every
  // constraint vector v in rel, where T is this transform.
  IntegerRelation applyTo(const IntegerRelation &rel) const;

  // The given vector is interpreted as a row vector v. Post-multiply v with
  // this transform, say T, and return vT.
  SmallVector<DynamicAPInt, 8>
  preMultiplyWithRow(ArrayRef<DynamicAPInt> rowVec) const {
    return matrix.preMultiplyWithRow(rowVec);
  }

  // The given vector is interpreted as a column vector v. Pre-multiply v with
  // this transform, say T, and return Tv.
  SmallVector<DynamicAPInt, 8>
  postMultiplyWithColumn(ArrayRef<DynamicAPInt> colVec) const {
    return matrix.postMultiplyWithColumn(colVec);
  }

private:
  IntMatrix matrix;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_LINEARTRANSFORM_H
