//===- QuasiPolynomial.h - QuasiPolynomial Class ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the QuasiPolynomial class for Barvinok's algorithm,
// which represents a single-valued function on a set of parameters.
// It is an expression of the form
// f(x) = \sum_i c_i * \prod_j ⌊g_{ij}(x)⌋
// where c_i \in Q and
// g_{ij} : Q^d -> Q are affine functionals over d parameters.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_QUASIPOLYNOMIAL_H
#define MLIR_ANALYSIS_PRESBURGER_QUASIPOLYNOMIAL_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"

namespace mlir {
namespace presburger {

// A class to describe quasi-polynomials.
// A quasipolynomial consists of a set of terms.
// The ith term is a constant `coefficients[i]`, multiplied
// by the product of a set of affine functions on n parameters.
// Represents functions f : Q^n -> Q of the form
//
// f(x) = \sum_i c_i * \prod_j ⌊g_{ij}(x)⌋
//
// where c_i \in Q and
// g_{ij} : Q^n -> Q are affine functionals.
class QuasiPolynomial : public PresburgerSpace {
public:
  QuasiPolynomial(unsigned numVars, ArrayRef<Fraction> coeffs = {},
                  ArrayRef<std::vector<SmallVector<Fraction>>> aff = {});

  QuasiPolynomial(unsigned numVars, const Fraction &constant);

  // Find the number of inputs (numDomain) to the polynomial.
  // numSymbols is set to zero.
  unsigned getNumInputs() const {
    return getNumDomainVars() + getNumSymbolVars();
  }

  const SmallVector<Fraction> &getCoefficients() const { return coefficients; }

  const std::vector<std::vector<SmallVector<Fraction>>> &getAffine() const {
    return affine;
  }

  // Arithmetic operations.
  QuasiPolynomial operator+(const QuasiPolynomial &x) const;
  QuasiPolynomial operator-(const QuasiPolynomial &x) const;
  QuasiPolynomial operator*(const QuasiPolynomial &x) const;
  QuasiPolynomial operator/(const Fraction &x) const;

  // Removes terms which evaluate to zero from the expression
  // and folds affine functions which are constant into the
  // constant coefficients.
  QuasiPolynomial simplify();

  // Group together like terms in the expression.
  QuasiPolynomial collectTerms();

  Fraction getConstantTerm();

private:
  SmallVector<Fraction> coefficients;
  std::vector<std::vector<SmallVector<Fraction>>> affine;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_QUASIPOLYNOMIAL_H
