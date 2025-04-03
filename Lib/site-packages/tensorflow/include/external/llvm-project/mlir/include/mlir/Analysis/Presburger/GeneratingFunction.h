//===- GeneratingFunction.h - Generating Functions over Q^d -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the GeneratingFunction class for Barvinok's algorithm,
// which represents a function over Q^n, parameterized by d parameters.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_GENERATINGFUNCTION_H
#define MLIR_ANALYSIS_PRESBURGER_GENERATINGFUNCTION_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"

namespace mlir {
namespace presburger {
namespace detail {

// A parametric point is a vector, each of whose elements
// is an affine function of n parameters. Each column
// in the matrix represents the affine function and
// has n+1 elements.
using ParamPoint = FracMatrix;

// A point is simply a vector.
using Point = SmallVector<Fraction>;

// A class to describe the type of generating function
// used to enumerate the integer points in a polytope.
// Consists of a set of terms, where the ith term has
// * a sign, ±1, stored in `signs[i]`
// * a numerator, of the form x^{n},
//      where n, stored in `numerators[i]`,
//      is a parametric point.
// * a denominator, of the form (1 - x^{d1})...(1 - x^{dn}),
//      where each dj, stored in `denominators[i][j]`,
//      is a vector.
//
// Represents functions f_p : Q^n -> Q of the form
//
// f_p(x) = \sum_i s_i * (x^n_i(p)) / (\prod_j (1 - x^d_{ij})
//
// where s_i is ±1,
// n_i \in Q^d -> Q^n is an n-vector of affine functions on d parameters, and
// g_{ij} \in Q^n are vectors.
class GeneratingFunction {
public:
  GeneratingFunction(unsigned numParam, SmallVector<int> signs,
                     std::vector<ParamPoint> nums,
                     std::vector<std::vector<Point>> dens)
      : numParam(numParam), signs(signs), numerators(nums), denominators(dens) {
#ifndef NDEBUG
    for (const ParamPoint &term : numerators)
      assert(term.getNumRows() == numParam + 1 &&
             "dimensionality of numerator exponents does not match number of "
             "parameters!");
#endif // NDEBUG
  }

  unsigned getNumParams() const { return numParam; }

  SmallVector<int> getSigns() const { return signs; }

  std::vector<ParamPoint> getNumerators() const { return numerators; }

  std::vector<std::vector<Point>> getDenominators() const {
    return denominators;
  }

  GeneratingFunction operator+(const GeneratingFunction &gf) const {
    assert(numParam == gf.getNumParams() &&
           "two generating functions with different numbers of parameters "
           "cannot be added!");
    SmallVector<int> sumSigns = signs;
    sumSigns.append(gf.signs);

    std::vector<ParamPoint> sumNumerators = numerators;
    sumNumerators.insert(sumNumerators.end(), gf.numerators.begin(),
                         gf.numerators.end());

    std::vector<std::vector<Point>> sumDenominators = denominators;
    sumDenominators.insert(sumDenominators.end(), gf.denominators.begin(),
                           gf.denominators.end());
    return GeneratingFunction(numParam, sumSigns, sumNumerators,
                              sumDenominators);
  }

  llvm::raw_ostream &print(llvm::raw_ostream &os) const {
    for (unsigned i = 0, e = signs.size(); i < e; i++) {
      if (i == 0) {
        if (signs[i] == -1)
          os << "- ";
      } else {
        if (signs[i] == 1)
          os << " + ";
        else
          os << " - ";
      }

      os << "x^[";
      unsigned r = numerators[i].getNumRows();
      for (unsigned j = 0; j < r - 1; j++) {
        os << "[";
        for (unsigned k = 0, c = numerators[i].getNumColumns(); k < c - 1; k++)
          os << numerators[i].at(j, k) << ",";
        os << numerators[i].getRow(j).back() << "],";
      }
      os << "[";
      for (unsigned k = 0, c = numerators[i].getNumColumns(); k < c - 1; k++)
        os << numerators[i].at(r - 1, k) << ",";
      os << numerators[i].getRow(r - 1).back() << "]]/";

      for (const Point &den : denominators[i]) {
        os << "(x^[";
        for (unsigned j = 0, e = den.size(); j < e - 1; j++)
          os << den[j] << ",";
        os << den.back() << "])";
      }
    }
    return os;
  }

private:
  unsigned numParam;
  SmallVector<int> signs;
  std::vector<ParamPoint> numerators;
  std::vector<std::vector<Point>> denominators;
};

} // namespace detail
} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_GENERATINGFUNCTION_H
