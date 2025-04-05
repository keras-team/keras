//===- Barvinok.h - Barvinok's Algorithm ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of Barvinok's algorithm and related utility functions.
// Currently a work in progress.
// These include functions to manipulate cones (define a cone object, get its
// dual, and find its index).
//
// The implementation is based on:
// 1. Barvinok, Alexander, and James E. Pommersheim. "An algorithmic theory of
//    lattice points in polyhedra." New perspectives in algebraic combinatorics
//    38 (1999): 91-147.
// 2. Verdoolaege, Sven, et al. "Counting integer points in parametric
//    polytopes using Barvinok's rational functions." Algorithmica 48 (2007):
//    37-66.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_BARVINOK_H
#define MLIR_ANALYSIS_PRESBURGER_BARVINOK_H

#include "mlir/Analysis/Presburger/GeneratingFunction.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/QuasiPolynomial.h"
#include <optional>

namespace mlir {
namespace presburger {
namespace detail {

/// A polyhedron in H-representation is a set of inequalities
/// in d variables with integer coefficients.
using PolyhedronH = IntegerRelation;

/// A polyhedron in V-representation is a set of rays and points, i.e.,
/// vectors, stored as rows of a matrix.
using PolyhedronV = IntMatrix;

/// A cone in either representation is a special case of
/// a polyhedron in that representation.
using ConeH = PolyhedronH;
using ConeV = PolyhedronV;

inline PolyhedronH defineHRep(int numVars, int numSymbols = 0) {
  // We don't distinguish between domain and range variables, so
  // we set the number of domain variables as 0 and the number of
  // range variables as the number of actual variables.
  //
  // numSymbols is the number of parameters.
  //
  // There are no local (existentially quantified) variables.
  //
  // The number of symbols is the number of parameters. By default, we consider
  // nonparametric polyhedra.
  //
  // Once the cone is defined, we use `addInequality()` to set inequalities.
  return PolyhedronH(PresburgerSpace::getSetSpace(/*numDims=*/numVars,
                                                  /*numSymbols=*/numSymbols,
                                                  /*numLocals=*/0));
}

/// Get the index of a cone, i.e., the volume of the parallelepiped
/// spanned by its generators, which is equal to the number of integer
/// points in its fundamental parallelepiped.
/// If the index is 1, the cone is unimodular.
/// Barvinok, A., and J. E. Pommersheim. "An algorithmic theory of lattice
/// points in polyhedra." p. 107 If it has more rays than the dimension, return
/// 0.
DynamicAPInt getIndex(const ConeV &cone);

/// Given a cone in H-representation, return its dual. The dual cone is in
/// V-representation.
/// This assumes that the input is pointed at the origin; it assert-fails
/// otherwise.
ConeV getDual(ConeH cone);

/// Given a cone in V-representation, return its dual. The dual cone is in
/// H-representation.
/// The returned cone is pointed at the origin.
ConeH getDual(ConeV cone);

/// Compute the generating function for a unimodular cone.
/// The input cone must be unimodular; it assert-fails otherwise.
GeneratingFunction computeUnimodularConeGeneratingFunction(ParamPoint vertex,
                                                           int sign,
                                                           const ConeH &cone);

/// Find the solution of a set of equations that express affine constraints
/// between a set of variables and a set of parameters. The solution expresses
/// each variable as an affine function of the parameters.
///
/// If there is no solution, return null.
std::optional<ParamPoint> solveParametricEquations(FracMatrix equations);

/// Given a list of possibly intersecting regions (PresburgerSet) and the
/// generating functions active in each region, produce a pairwise disjoint
/// list of regions (chambers) and identify the generating function of the
/// polytope in each chamber.
///
/// "Disjoint" here means that the intersection of two chambers is no full-
/// dimensional.
///
/// The returned list partitions the universe into parts depending on which
/// subset of GFs is active there, and gives the sum of active GFs for each
/// part.
std::vector<std::pair<PresburgerSet, GeneratingFunction>>
computeChamberDecomposition(
    unsigned numSymbols, ArrayRef<std::pair<PresburgerSet, GeneratingFunction>>
                             regionsAndGeneratingFunctions);

/// Compute the generating function corresponding to a polytope.
///
/// All tangent cones of the polytope must be unimodular.
std::vector<std::pair<PresburgerSet, GeneratingFunction>>
computePolytopeGeneratingFunction(const PolyhedronH &poly);

/// Find a vector that is not orthogonal to any of the given vectors,
/// i.e., has nonzero dot product with those of the given vectors
/// that are not null.
/// If any of the vectors is null, it is ignored.
Point getNonOrthogonalVector(ArrayRef<Point> vectors);

/// Find the coefficient of a given power of s in a rational function
/// given by P(s)/Q(s), where
/// P is a polynomial, in which the coefficients are QuasiPolynomials
/// over d parameters (distinct from s), and
/// and Q is a polynomial with Fraction coefficients.
QuasiPolynomial getCoefficientInRationalFunction(unsigned power,
                                                 ArrayRef<QuasiPolynomial> num,
                                                 ArrayRef<Fraction> den);

/// Find the number of terms in a generating function, as
/// a quasipolynomial in the parameter space of the input function.
/// The generating function must be such that for all values of the
/// parameters, the number of terms is finite.
QuasiPolynomial computeNumTerms(const GeneratingFunction &gf);

} // namespace detail
} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_BARVINOK_H
