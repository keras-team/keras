//===- Simplex.h - MLIR Simplex Class ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality to perform analysis on an IntegerRelation. In particular,
// support for performing emptiness checks, redundancy checks and obtaining the
// lexicographically minimum rational element in a set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
#define MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/SmallBitVector.h"
#include <optional>

namespace mlir {
namespace presburger {

class GBRSimplex;

/// The Simplex class implements a version of the Simplex and Generalized Basis
/// Reduction algorithms, which can perform analysis of integer sets with affine
/// inequalities and equalities. A Simplex can be constructed
/// by specifying the dimensionality of the set. It supports adding affine
/// inequalities and equalities, and can perform emptiness checks, i.e., it can
/// find a solution to the set of constraints if one exists, or say that the
/// set is empty if no solution exists. Furthermore, it can find a subset of
/// these constraints that are redundant, i.e. a subset of constraints that
/// doesn't constrain the affine set further after adding the non-redundant
/// constraints. The LexSimplex class provides support for computing the
/// lexicographic minimum of an IntegerRelation. The SymbolicLexOpt class
/// provides support for computing symbolic lexicographic minimums. All of these
/// classes can be constructed from an IntegerRelation, and all inherit common
/// functionality from SimplexBase.
///
/// The implementations of the Simplex and SimplexBase classes, other than the
/// functionality for obtaining an integer sample, are based on the paper
/// "Simplify: A Theorem Prover for Program Checking"
/// by D. Detlefs, G. Nelson, J. B. Saxe.
///
/// We define variables, constraints, and unknowns. Consider the example of a
/// two-dimensional set defined by 1 + 2x + 3y >= 0 and 2x - 3y >= 0. Here,
/// x, y, are variables while 1 + 2x + 3y >= 0, 2x - 3y >= 0 are constraints.
/// Unknowns are either variables or constraints, i.e., x, y, 1 + 2x + 3y >= 0,
/// 2x - 3y >= 0 are all unknowns.
///
/// The implementation involves a matrix called a tableau, which can be thought
/// of as a 2D matrix of rational numbers having number of rows equal to the
/// number of constraints and number of columns equal to one plus the number of
/// variables. In our implementation, instead of storing rational numbers, we
/// store a common denominator for each row, so it is in fact a matrix of
/// integers with number of rows equal to number of constraints and number of
/// columns equal to _two_ plus the number of variables. For example, instead of
/// storing a row of three rationals [1/2, 2/3, 3], we would store [6, 3, 4, 18]
/// since 3/6 = 1/2, 4/6 = 2/3, and 18/6 = 3.
///
/// Every row and column except the first and second columns is associated with
/// an unknown and every unknown is associated with a row or column. An unknown
/// associated with a row or column is said to be in row or column orientation
/// respectively. As described above, the first column is the common
/// denominator. The second column represents the constant term, explained in
/// more detail below. These two are _fixed columns_; they always retain their
/// position as the first and second columns. Additionally, LexSimplexBase
/// stores a so-call big M parameter (explained below) in the third column, so
/// LexSimplexBase has three fixed columns. Finally, SymbolicLexSimplex has
/// `nSymbol` variables designated as symbols. These occupy the next `nSymbol`
/// columns, viz. the columns [3, 3 + nSymbol). For more information on symbols,
/// see LexSimplexBase and SymbolicLexSimplex.
///
/// LexSimplexBase does not directly support variables which can be negative, so
/// we introduce the so-called big M parameter, an artificial variable that is
/// considered to have an arbitrarily large value. We then transform the
/// variables, say x, y, z, ... to M, M + x, M + y, M + z. Since M has been
/// added to these variables, they are now known to have non-negative values.
/// For more details, see the documentation for LexSimplexBase. The big M
/// parameter is not considered a real unknown and is not stored in the `var`
/// data structure; rather the tableau just has an extra fixed column for it
/// just like the constant term.
///
/// The vectors var and con store information about the variables and
/// constraints respectively, namely, whether they are in row or column
/// position, which row or column they are associated with, and whether they
/// correspond to a variable or a constraint.
///
/// An unknown is addressed by its index. If the index i is non-negative, then
/// the variable var[i] is being addressed. If the index i is negative, then
/// the constraint con[~i] is being addressed. Effectively this maps
/// 0 -> var[0], 1 -> var[1], -1 -> con[0], -2 -> con[1], etc. rowUnknown[r] and
/// colUnknown[c] are the indexes of the unknowns associated with row r and
/// column c, respectively.
///
/// The unknowns in column position are together called the basis. Initially the
/// basis is the set of variables -- in our example above, the initial basis is
/// x, y.
///
/// The unknowns in row position are represented in terms of the basis unknowns.
/// If the basis unknowns are u_1, u_2, ... u_m, and a row in the tableau is
/// d, c, a_1, a_2, ... a_m, this represents the unknown for that row as
/// (c + a_1*u_1 + a_2*u_2 + ... + a_m*u_m)/d. In our running example, if the
/// basis is the initial basis of x, y, then the constraint 1 + 2x + 3y >= 0
/// would be represented by the row [1, 1, 2, 3].
///
/// The association of unknowns to rows and columns can be changed by a process
/// called pivoting, where a row unknown and a column unknown exchange places
/// and the remaining row variables' representation is changed accordingly
/// by eliminating the old column unknown in favour of the new column unknown.
/// If we had pivoted the column for x with the row for 2x - 3y >= 0,
/// the new row for x would be [2, 1, 3] since x = (1*(2x - 3y) + 3*y)/2.
/// See the documentation for the pivot member function for details.
///
/// The association of unknowns to rows and columns is called the _tableau
/// configuration_. The _sample value_ of an unknown in a particular tableau
/// configuration is its value if all the column unknowns were set to zero.
/// Concretely, for unknowns in column position the sample value is zero; when
/// the big M parameter is not used, for unknowns in row position the sample
/// value is the constant term divided by the common denominator. When the big M
/// parameter is used, if d is the denominator, p is the big M coefficient, and
/// c is the constant term, then the sample value is (p*M + c)/d. Since M is
/// considered to be positive infinity, this is positive (negative) infinity
/// when p is positive or negative, and c/d when p is zero.
///
/// The tableau configuration is called _consistent_ if the sample value of all
/// restricted unknowns is non-negative. Initially there are no constraints, and
/// the tableau is consistent. When a new constraint is added, its sample value
/// in the current tableau configuration may be negative. In that case, we try
/// to find a series of pivots to bring us to a consistent tableau
/// configuration, i.e. we try to make the new constraint's sample value
/// non-negative without making that of any other constraints negative. (See
/// findPivot and findPivotRow for details.) If this is not possible, then the
/// set of constraints is mutually contradictory and the tableau is marked
/// _empty_, which means the set of constraints has no solution.
///
/// This SimplexBase class also supports taking snapshots of the current state
/// and rolling back to prior snapshots. This works by maintaining an undo log
/// of operations. Snapshots are just pointers to a particular location in the
/// log, and rolling back to a snapshot is done by reverting each log entry's
/// operation from the end until we reach the snapshot's location. SimplexBase
/// also supports taking a snapshot including the exact set of basis unknowns;
/// if this functionality is used, then on rolling back the exact basis will
/// also be restored. This is used by LexSimplexBase because the lex algorithm,
/// unlike `Simplex`, is sensitive to the exact basis used at a point.
class SimplexBase {
public:
  SimplexBase() = delete;
  virtual ~SimplexBase() = default;

  /// Returns true if the tableau is empty (has conflicting constraints),
  /// false otherwise.
  bool isEmpty() const;

  /// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  virtual void addInequality(ArrayRef<DynamicAPInt> coeffs) = 0;

  /// Returns the number of variables in the tableau.
  unsigned getNumVariables() const;

  /// Returns the number of constraints in the tableau.
  unsigned getNumConstraints() const;

  /// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding equality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
  void addEquality(ArrayRef<DynamicAPInt> coeffs);

  /// Add new variables to the end of the list of variables.
  void appendVariable(unsigned count = 1);

  /// Append a new variable to the simplex and constrain it such that its only
  /// integer value is the floor div of `coeffs` and `denom`.
  ///
  /// `denom` must be positive.
  void addDivisionVariable(ArrayRef<DynamicAPInt> coeffs,
                           const DynamicAPInt &denom);

  /// Mark the tableau as being empty.
  void markEmpty();

  /// Get a snapshot of the current state. This is used for rolling back.
  /// The same basis will not necessarily be restored on rolling back.
  /// The snapshot only captures the set of variables and constraints present
  /// in the Simplex.
  unsigned getSnapshot() const;

  /// Get a snapshot of the current state including the basis. When rolling
  /// back, the exact basis will be restored.
  unsigned getSnapshotBasis();

  /// Rollback to a snapshot. This invalidates all later snapshots.
  void rollback(unsigned snapshot);

  /// Add all the constraints from the given IntegerRelation.
  void intersectIntegerRelation(const IntegerRelation &rel);

  /// Print the tableau's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  /// Construct a SimplexBase with the specified number of variables and fixed
  /// columns. The first overload should be used when there are nosymbols.
  /// With the second overload, the specified range of vars will be marked
  /// as symbols. With the third overload, `isSymbol` is a bitmask denoting
  /// which vars are symbols. The size of `isSymbol` must be `nVar`.
  ///
  /// For example, Simplex uses two fixed columns: the denominator and the
  /// constant term, whereas LexSimplex has an extra fixed column for the
  /// so-called big M parameter. For more information see the documentation for
  /// LexSimplex.
  SimplexBase(unsigned nVar, bool mustUseBigM);
  SimplexBase(unsigned nVar, bool mustUseBigM,
              const llvm::SmallBitVector &isSymbol);

  enum class Orientation { Row, Column };

  /// An Unknown is either a variable or a constraint. It is always associated
  /// with either a row or column. Whether it's a row or a column is specified
  /// by the orientation and pos identifies the specific row or column it is
  /// associated with. If the unknown is restricted, then it has a
  /// non-negativity constraint associated with it, i.e., its sample value must
  /// always be non-negative and if it cannot be made non-negative without
  /// violating other constraints, the tableau is empty.
  struct Unknown {
    Unknown(Orientation oOrientation, bool oRestricted, unsigned oPos,
            bool oIsSymbol = false)
        : pos(oPos), orientation(oOrientation), restricted(oRestricted),
          isSymbol(oIsSymbol) {}
    unsigned pos;
    Orientation orientation;
    bool restricted : 1;
    bool isSymbol : 1;

    void print(raw_ostream &os) const {
      os << (orientation == Orientation::Row ? "r" : "c");
      os << pos;
      if (restricted)
        os << " [>=0]";
    }
  };

  struct Pivot {
    unsigned row, column;
  };

  /// Return any row that this column can be pivoted with, ignoring tableau
  /// consistency.
  ///
  /// Returns an empty optional if no pivot is possible, which happens only when
  /// the column unknown is a variable and no constraint has a non-zero
  /// coefficient for it.
  std::optional<unsigned> findAnyPivotRow(unsigned col);

  /// Swap the row with the column in the tableau's data structures but not the
  /// tableau itself. This is used by pivot.
  void swapRowWithCol(unsigned row, unsigned col);

  /// Pivot the row with the column.
  void pivot(unsigned row, unsigned col);
  void pivot(Pivot pair);

  /// Returns the unknown associated with index.
  const Unknown &unknownFromIndex(int index) const;
  /// Returns the unknown associated with col.
  const Unknown &unknownFromColumn(unsigned col) const;
  /// Returns the unknown associated with row.
  const Unknown &unknownFromRow(unsigned row) const;
  /// Returns the unknown associated with index.
  Unknown &unknownFromIndex(int index);
  /// Returns the unknown associated with col.
  Unknown &unknownFromColumn(unsigned col);
  /// Returns the unknown associated with row.
  Unknown &unknownFromRow(unsigned row);

  /// Add a new row to the tableau and the associated data structures. The row
  /// is initialized to zero. Returns the index of the added row.
  unsigned addZeroRow(bool makeRestricted = false);

  /// Add a new row to the tableau and the associated data structures.
  /// The new row is considered to be a constraint; the new Unknown lives in
  /// con.
  ///
  /// Returns the index of the new Unknown in con.
  unsigned addRow(ArrayRef<DynamicAPInt> coeffs, bool makeRestricted = false);

  /// Swap the two rows/columns in the tableau and associated data structures.
  void swapRows(unsigned i, unsigned j);
  void swapColumns(unsigned i, unsigned j);

  /// Enum to denote operations that need to be undone during rollback.
  enum class UndoLogEntry {
    RemoveLastConstraint,
    RemoveLastVariable,
    UnmarkEmpty,
    UnmarkLastRedundant,
    RestoreBasis
  };

  /// Undo the addition of the last constraint. This will only be called from
  /// undo, when rolling back.
  virtual void undoLastConstraint() = 0;

  /// Remove the last constraint, which must be in row orientation.
  void removeLastConstraintRowOrientation();

  /// Undo the operation represented by the log entry.
  void undo(UndoLogEntry entry);

  /// Return the number of fixed columns, as described in the constructor above,
  /// this is the number of columns beyond those for the variables in var.
  unsigned getNumFixedCols() const { return usingBigM ? 3u : 2u; }
  unsigned getNumRows() const { return tableau.getNumRows(); }
  unsigned getNumColumns() const { return tableau.getNumColumns(); }

  /// Stores whether or not a big M column is present in the tableau.
  bool usingBigM;

  /// The number of redundant rows in the tableau. These are the first
  /// nRedundant rows.
  unsigned nRedundant;

  /// The number of parameters. This must be consistent with the number of
  /// Unknowns in `var` below that have `isSymbol` set to true.
  unsigned nSymbol;

  /// The matrix representing the tableau.
  IntMatrix tableau;

  /// This is true if the tableau has been detected to be empty, false
  /// otherwise.
  bool empty;

  /// Holds a log of operations, used for rolling back to a previous state.
  SmallVector<UndoLogEntry, 8> undoLog;

  /// Holds a vector of bases. The ith saved basis is the basis that should be
  /// restored when processing the ith occurrance of UndoLogEntry::RestoreBasis
  /// in undoLog. This is used by getSnapshotBasis.
  SmallVector<SmallVector<int, 8>, 8> savedBases;

  /// These hold the indexes of the unknown at a given row or column position.
  /// We keep these as signed integers since that makes it convenient to check
  /// if an index corresponds to a variable or a constraint by checking the
  /// sign.
  ///
  /// colUnknown is padded with two null indexes at the front since the first
  /// two columns don't correspond to any unknowns.
  SmallVector<int, 8> rowUnknown, colUnknown;

  /// These hold information about each unknown.
  SmallVector<Unknown, 8> con, var;
};

/// Simplex class using the lexicographic pivot rule. Used for lexicographic
/// optimization. The implementation of this class is based on the paper
/// "Parametric Integer Programming" by Paul Feautrier.
///
/// This does not directly support negative-valued variables, so it uses the big
/// M parameter trick to make all the variables non-negative. Basically we
/// introduce an artifical variable M that is considered to have a value of
/// +infinity and instead of the variables x, y, z, we internally use variables
/// M + x, M + y, M + z, which are now guaranteed to be non-negative. See the
/// documentation for SimplexBase for more details. M is also considered to be
/// an integer that is divisible by everything.
///
/// The whole algorithm is performed with M treated as a symbol;
/// it is just considered to be infinite throughout and it never appears in the
/// final outputs. We will deal with sample values throughout that may in
/// general be some affine expression involving M, like pM + q or aM + b. We can
/// compare these with each other. They have a total order:
///
/// aM + b < pM + q iff  a < p or (a == p and b < q).
/// In particular, aM + b < 0 iff a < 0 or (a == 0 and b < 0).
///
/// When performing symbolic optimization, sample values will be affine
/// expressions in M and the symbols. For example, we could have sample values
/// aM + bS + c and pM + qS + r, where S is a symbol. Now we have
/// aM + bS + c < pM + qS + r iff (a < p) or (a == p and bS + c < qS + r).
/// bS + c < qS + r can be always true, always false, or neither,
/// depending on the set of values S can take. The symbols are always stored
/// in columns [3, 3 + nSymbols). For more details, see the
/// documentation for SymbolicLexSimplex.
///
/// Initially all the constraints to be added are added as rows, with no attempt
/// to keep the tableau consistent. Pivots are only performed when some query
/// is made, such as a call to getRationalLexMin. Care is taken to always
/// maintain a lexicopositive basis transform, explained below.
///
/// Let the variables be x = (x_1, ... x_n).
/// Let the symbols be   s = (s_1, ... s_m). Let the basis unknowns at a
/// particular point be  y = (y_1, ... y_n). We know that x = A*y + T*s + b for
/// some n x n matrix A, n x m matrix s, and n x 1 column vector b. We want
/// every column in A to be lexicopositive, i.e., have at least one non-zero
/// element, with the first such element being positive. This property is
/// preserved throughout the operation of LexSimplexBase. Note that on
/// construction, the basis transform A is the identity matrix and so every
/// column is lexicopositive. Note that for LexSimplexBase, for the tableau to
/// be consistent we must have non-negative sample values not only for the
/// constraints but also for the variables. So if the tableau is consistent then
/// x >= 0 and y >= 0, by which we mean every element in these vectors is
/// non-negative. (note that this is a different concept from lexicopositivity!)
class LexSimplexBase : public SimplexBase {
public:
  ~LexSimplexBase() override = default;

  /// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  ///
  /// This just adds the inequality to the tableau and does not try to create a
  /// consistent tableau configuration.
  void addInequality(ArrayRef<DynamicAPInt> coeffs) final;

  /// Get a snapshot of the current state. This is used for rolling back.
  unsigned getSnapshot() { return SimplexBase::getSnapshotBasis(); }

protected:
  LexSimplexBase(unsigned nVar) : SimplexBase(nVar, /*mustUseBigM=*/true) {}
  LexSimplexBase(unsigned nVar, const llvm::SmallBitVector &isSymbol)
      : SimplexBase(nVar, /*mustUseBigM=*/true, isSymbol) {}
  explicit LexSimplexBase(const IntegerRelation &constraints)
      : LexSimplexBase(constraints.getNumVars()) {
    intersectIntegerRelation(constraints);
  }
  explicit LexSimplexBase(const IntegerRelation &constraints,
                          const llvm::SmallBitVector &isSymbol)
      : LexSimplexBase(constraints.getNumVars(), isSymbol) {
    intersectIntegerRelation(constraints);
  }

  /// Add new symbolic variables to the end of the list of variables.
  void appendSymbol();

  /// Try to move the specified row to column orientation while preserving the
  /// lexicopositivity of the basis transform. The row must have a non-positive
  /// sample value. If this is not possible, return failure. This occurs when
  /// the constraints have no solution or the sample value is zero.
  LogicalResult moveRowUnknownToColumn(unsigned row);

  /// Given a row that has a non-integer sample value, add an inequality to cut
  /// away this fractional sample value from the polytope without removing any
  /// integer points. The integer lexmin, if one existed, remains the same on
  /// return.
  ///
  /// This assumes that the symbolic part of the sample is integral,
  /// i.e., if the symbolic sample is (c + aM + b_1*s_1 + ... b_n*s_n)/d,
  /// where s_1, ... s_n are symbols, this assumes that
  /// (b_1*s_1 + ... + b_n*s_n)/s is integral.
  ///
  /// Return failure if the tableau became empty, and success if it didn't.
  /// Failure status indicates that the polytope was integer empty.
  LogicalResult addCut(unsigned row);

  /// Undo the addition of the last constraint. This is only called while
  /// rolling back.
  void undoLastConstraint() final;

  /// Given two potential pivot columns for a row, return the one that results
  /// in the lexicographically smallest sample vector. The row's sample value
  /// must be negative. If symbols are involved, the sample value must be
  /// negative for all possible assignments to the symbols.
  unsigned getLexMinPivotColumn(unsigned row, unsigned colA,
                                unsigned colB) const;
};

/// A class for lexicographic optimization without any symbols. This also
/// provides support for integer-exact redundancy and separateness checks.
class LexSimplex : public LexSimplexBase {
public:
  explicit LexSimplex(unsigned nVar) : LexSimplexBase(nVar) {}
  // Note that LexSimplex does NOT support symbolic lexmin;
  // use SymbolicLexSimplex if that is required. LexSimplex ignores the VarKinds
  // of the passed IntegerRelation. Symbols will be treated as ordinary vars.
  explicit LexSimplex(const IntegerRelation &constraints)
      : LexSimplexBase(constraints) {}

  /// Return the lexicographically minimum rational solution to the constraints.
  MaybeOptimum<SmallVector<Fraction, 8>> findRationalLexMin();

  /// Return the lexicographically minimum integer solution to the constraints.
  ///
  /// Note: this should be used only when the lexmin is really needed. To obtain
  /// any integer sample, use Simplex::findIntegerSample as that is more robust.
  MaybeOptimum<SmallVector<DynamicAPInt, 8>> findIntegerLexMin();

  /// Return whether the specified inequality is redundant/separate for the
  /// polytope. Redundant means every point satisfies the given inequality, and
  /// separate means no point satisfies it.
  ///
  /// These checks are integer-exact.
  bool isSeparateInequality(ArrayRef<DynamicAPInt> coeffs);
  bool isRedundantInequality(ArrayRef<DynamicAPInt> coeffs);

private:
  /// Returns the current sample point, which may contain non-integer (rational)
  /// coordinates. Returns an empty optimum when the tableau is empty.
  ///
  /// Returns an unbounded optimum when the big M parameter is used and a
  /// variable has a non-zero big M coefficient, meaning its value is infinite
  /// or unbounded.
  MaybeOptimum<SmallVector<Fraction, 8>> getRationalSample() const;

  /// Make the tableau configuration consistent.
  LogicalResult restoreRationalConsistency();

  /// Return whether the specified row is violated;
  bool rowIsViolated(unsigned row) const;

  /// Get a constraint row that is violated, if one exists.
  /// Otherwise, return an empty optional.
  std::optional<unsigned> maybeGetViolatedRow() const;

  /// Get a row corresponding to a var that has a non-integral sample value, if
  /// one exists. Otherwise, return an empty optional.
  std::optional<unsigned> maybeGetNonIntegralVarRow() const;
};

/// Represents the result of a symbolic lexicographic optimization computation.
struct SymbolicLexOpt {
  SymbolicLexOpt(const PresburgerSpace &space)
      : lexopt(space),
        unboundedDomain(PresburgerSet::getEmpty(space.getDomainSpace())) {}

  /// This maps assignments of symbols to the corresponding lexopt.
  /// Takes no value when no integer sample exists for the assignment or if the
  /// lexopt is unbounded.
  PWMAFunction lexopt;
  /// Contains all assignments to the symbols that made the lexopt unbounded.
  /// Note that the symbols of the input set to the symbolic lexopt are dims
  /// of this PrebsurgerSet.
  PresburgerSet unboundedDomain;
};

/// A class to perform symbolic lexicographic optimization,
/// i.e., to find, for every assignment to the symbols the specified
/// `symbolDomain`, the lexicographically minimum value integer value attained
/// by the non-symbol variables.
///
/// The input is a set parametrized by some symbols, i.e., the constant terms
/// of the constraints in the set are affine expressions in the symbols, and
/// every assignment to the symbols defines a non-symbolic set.
///
/// Accordingly, the sample values of the rows in our tableau will be affine
/// expressions in the symbols, and every assignment to the symbols will define
/// a non-symbolic LexSimplex. We then run the algorithm of
/// LexSimplex::findIntegerLexMin simultaneously for every value of the symbols
/// in the domain.
///
/// Often, the pivot to be performed is the same for all values of the symbols,
/// in which case we just do it. For example, if the symbolic sample of a row is
/// negative for all values in the symbol domain, the row needs to be pivoted
/// irrespective of the precise value of the symbols. To answer queries like
/// "Is this symbolic sample always negative in the symbol domain?", we maintain
/// a `LexSimplex domainSimplex` correponding to the symbol domain.
///
/// In other cases, it may be that the symbolic sample is violated at some
/// values in the symbol domain and not violated at others. In this case,
/// the pivot to be performed does depend on the value of the symbols. We
/// handle this by splitting the symbol domain. We run the algorithm for the
/// case where the row isn't violated, and then come back and run the case
/// where it is.
class SymbolicLexSimplex : public LexSimplexBase {
public:
  /// `constraints` is the set for which the symbolic lexopt will be computed.
  /// `symbolDomain` is the set of values of the symbols for which the lexopt
  /// will be computed. `symbolDomain` should have a dim var for every symbol in
  /// `constraints`, and no other vars. `isSymbol` specifies which vars of
  /// `constraints` should be considered as symbols.
  ///
  /// The resulting SymbolicLexOpt's space will be compatible with that of
  /// symbolDomain.
  SymbolicLexSimplex(const IntegerRelation &constraints,
                     const IntegerPolyhedron &symbolDomain,
                     const llvm::SmallBitVector &isSymbol)
      : LexSimplexBase(constraints, isSymbol), domainPoly(symbolDomain),
        domainSimplex(symbolDomain) {
    // TODO consider supporting this case. It amounts
    // to just returning the input constraints.
    assert(domainPoly.getNumVars() > 0 &&
           "there must be some non-symbols to optimize!");
  }

  /// An overload to select some subrange of ids as symbols for lexopt.
  /// The symbol ids are the range of ids with absolute index
  /// [symbolOffset, symbolOffset + symbolDomain.getNumVars())
  SymbolicLexSimplex(const IntegerRelation &constraints, unsigned symbolOffset,
                     const IntegerPolyhedron &symbolDomain)
      : SymbolicLexSimplex(constraints, symbolDomain,
                           getSubrangeBitVector(constraints.getNumVars(),
                                                symbolOffset,
                                                symbolDomain.getNumVars())) {}

  /// An overload to select the symbols of `constraints` as symbols for lexopt.
  SymbolicLexSimplex(const IntegerRelation &constraints,
                     const IntegerPolyhedron &symbolDomain)
      : SymbolicLexSimplex(constraints,
                           constraints.getVarKindOffset(VarKind::Symbol),
                           symbolDomain) {
    assert(constraints.getNumSymbolVars() == symbolDomain.getNumVars() &&
           "symbolDomain must have as many vars as constraints has symbols!");
  }

  /// The lexmin will be stored as a function `lexopt` from symbols to
  /// non-symbols in the result.
  ///
  /// For some values of the symbols, the lexmin may be unbounded.
  /// These parts of the symbol domain will be stored in `unboundedDomain`.
  ///
  /// The spaces of the sets in the result are compatible with the symbolDomain
  /// passed in the SymbolicLexSimplex constructor.
  SymbolicLexOpt computeSymbolicIntegerLexMin();

private:
  /// Perform all pivots that do not require branching.
  ///
  /// Return failure if the tableau became empty, indicating that the polytope
  /// is always integer empty in the current symbol domain.
  /// Return success otherwise.
  LogicalResult doNonBranchingPivots();

  /// Get a row that is always violated in the current domain, if one exists.
  std::optional<unsigned> maybeGetAlwaysViolatedRow();

  /// Get a row corresponding to a variable with non-integral sample value, if
  /// one exists.
  std::optional<unsigned> maybeGetNonIntegralVarRow();

  /// Given a row that has a non-integer sample value, cut away this fractional
  /// sample value witahout removing any integer points, i.e., the integer
  /// lexmin, if it exists, remains the same after a call to this function. This
  /// may add constraints or local variables to the tableau, as well as to the
  /// domain.
  ///
  /// Returns whether the cut constraint could be enforced, i.e. failure if the
  /// cut made the polytope empty, and success if it didn't. Failure status
  /// indicates that the polytope is always integer empty in the symbol domain
  /// at the time of the call. (This function may modify the symbol domain, but
  /// failure statu indicates that the polytope was empty for all symbol values
  /// in the initial domain.)
  LogicalResult addSymbolicCut(unsigned row);

  /// Get the numerator of the symbolic sample of the specific row.
  /// This is an affine expression in the symbols with integer coefficients.
  /// The last element is the constant term. This ignores the big M coefficient.
  SmallVector<DynamicAPInt, 8> getSymbolicSampleNumerator(unsigned row) const;

  /// Get an affine inequality in the symbols with integer coefficients that
  /// holds iff the symbolic sample of the specified row is non-negative.
  SmallVector<DynamicAPInt, 8> getSymbolicSampleIneq(unsigned row) const;

  /// Return whether all the coefficients of the symbolic sample are integers.
  ///
  /// This does not consult the domain to check if the specified expression
  /// is always integral despite coefficients being fractional.
  bool isSymbolicSampleIntegral(unsigned row) const;

  /// Record a lexmin. The tableau must be consistent with all variables
  /// having symbolic samples with integer coefficients.
  void recordOutput(SymbolicLexOpt &result) const;

  /// The symbol domain.
  IntegerPolyhedron domainPoly;
  /// Simplex corresponding to the symbol domain.
  LexSimplex domainSimplex;
};

/// The Simplex class uses the Normal pivot rule and supports integer emptiness
/// checks as well as detecting redundancies.
///
/// The Simplex class supports redundancy checking via detectRedundant and
/// isMarkedRedundant. A redundant constraint is one which is never violated as
/// long as the other constraints are not violated, i.e., removing a redundant
/// constraint does not change the set of solutions to the constraints. As a
/// heuristic, constraints that have been marked redundant can be ignored for
/// most operations. Therefore, these constraints are kept in rows 0 to
/// nRedundant - 1, where nRedundant is a member variable that tracks the number
/// of constraints that have been marked redundant.
///
/// Finding an integer sample is done with the Generalized Basis Reduction
/// algorithm. See the documentation for findIntegerSample and reduceBasis.
class Simplex : public SimplexBase {
public:
  enum class Direction { Up, Down };

  Simplex() = delete;
  explicit Simplex(unsigned nVar) : SimplexBase(nVar, /*mustUseBigM=*/false) {}
  explicit Simplex(const IntegerRelation &constraints)
      : Simplex(constraints.getNumVars()) {
    intersectIntegerRelation(constraints);
  }
  ~Simplex() override = default;

  /// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  ///
  /// This also tries to restore the tableau configuration to a consistent
  /// state and marks the Simplex empty if this is not possible.
  void addInequality(ArrayRef<DynamicAPInt> coeffs) final;

  /// Compute the maximum or minimum value of the given row, depending on
  /// direction. The specified row is never pivoted. On return, the row may
  /// have a negative sample value if the direction is down.
  ///
  /// Returns a Fraction denoting the optimum, or a null value if no optimum
  /// exists, i.e., if the expression is unbounded in this direction.
  MaybeOptimum<Fraction> computeRowOptimum(Direction direction, unsigned row);

  /// Compute the maximum or minimum value of the given expression, depending on
  /// direction. Should not be called when the Simplex is empty.
  ///
  /// Returns a Fraction denoting the optimum, or a null value if no optimum
  /// exists, i.e., if the expression is unbounded in this direction.
  MaybeOptimum<Fraction> computeOptimum(Direction direction,
                                        ArrayRef<DynamicAPInt> coeffs);

  /// Returns whether the perpendicular of the specified constraint is a
  /// is a direction along which the polytope is bounded.
  bool isBoundedAlongConstraint(unsigned constraintIndex);

  /// Returns whether the specified constraint has been marked as redundant.
  /// Constraints are numbered from 0 starting at the first added inequality.
  /// Equalities are added as a pair of inequalities and so correspond to two
  /// inequalities with successive indices.
  bool isMarkedRedundant(unsigned constraintIndex) const;

  /// Finds a subset of constraints that is redundant, i.e., such that
  /// the set of solutions does not change if these constraints are removed.
  /// Marks these constraints as redundant. Whether a specific constraint has
  /// been marked redundant can be queried using isMarkedRedundant.
  ///
  /// The first overload only tries to find redundant constraints with indices
  /// in the range [offset, offset + count), by scanning constraints from left
  /// to right in this range. If `count` is not provided, all constraints
  /// starting at `offset` are scanned, and if neither are provided, all
  /// constraints are scanned, starting from 0 and going to the last constraint.
  ///
  /// As an example, in the set (x) : (x >= 0, x >= 0, x >= 0), calling
  /// `detectRedundant` with no parameters will result in the first two
  /// constraints being marked redundant. All copies cannot be marked redundant
  /// because removing all the constraints changes the set. The first two are
  /// the ones marked redundant because we scan from left to right. Thus, when
  /// there is some preference among the constraints as to which should be
  /// marked redundant with priority when there are multiple possibilities, this
  /// could be accomplished by succesive calls to detectRedundant(offset,
  /// count).
  void detectRedundant(unsigned offset, unsigned count);
  void detectRedundant(unsigned offset) {
    assert(offset <= con.size() && "invalid offset!");
    detectRedundant(offset, con.size() - offset);
  }
  void detectRedundant() { detectRedundant(0, con.size()); }

  /// Returns a (min, max) pair denoting the minimum and maximum integer values
  /// of the given expression. If no integer value exists, both results will be
  /// of kind Empty.
  std::pair<MaybeOptimum<DynamicAPInt>, MaybeOptimum<DynamicAPInt>>
  computeIntegerBounds(ArrayRef<DynamicAPInt> coeffs);

  /// Check if the simplex takes only one rational value along the
  /// direction of `coeffs`.
  ///
  /// `this` must be nonempty.
  bool isFlatAlong(ArrayRef<DynamicAPInt> coeffs);

  /// Returns true if the polytope is unbounded, i.e., extends to infinity in
  /// some direction. Otherwise, returns false.
  bool isUnbounded();

  /// Make a tableau to represent a pair of points in the given tableaus, one in
  /// tableau A and one in B.
  static Simplex makeProduct(const Simplex &a, const Simplex &b);

  /// Returns an integer sample point if one exists, or std::nullopt
  /// otherwise. This should only be called for bounded sets.
  std::optional<SmallVector<DynamicAPInt, 8>> findIntegerSample();

  enum class IneqType { Redundant, Cut, Separate };

  /// Returns the type of the inequality with coefficients `coeffs`.
  ///
  /// Possible types are:
  /// Redundant   The inequality is satisfied in the polytope
  /// Cut         The inequality is satisfied by some points, but not by others
  /// Separate    The inequality is not satisfied by any point
  IneqType findIneqType(ArrayRef<DynamicAPInt> coeffs);

  /// Check if the specified inequality already holds in the polytope.
  bool isRedundantInequality(ArrayRef<DynamicAPInt> coeffs);

  /// Check if the specified equality already holds in the polytope.
  bool isRedundantEquality(ArrayRef<DynamicAPInt> coeffs);

  /// Returns true if this Simplex's polytope is a rational subset of `rel`.
  /// Otherwise, returns false.
  bool isRationalSubsetOf(const IntegerRelation &rel);

  /// Returns the current sample point if it is integral. Otherwise, returns
  /// std::nullopt.
  std::optional<SmallVector<DynamicAPInt, 8>> getSamplePointIfIntegral() const;

  /// Returns the current sample point, which may contain non-integer (rational)
  /// coordinates. Returns an empty optional when the tableau is empty.
  std::optional<SmallVector<Fraction, 8>> getRationalSample() const;

private:
  friend class GBRSimplex;

  /// Restore the unknown to a non-negative sample value.
  ///
  /// Returns success if the unknown was successfully restored to a non-negative
  /// sample value, failure otherwise.
  LogicalResult restoreRow(Unknown &u);

  /// Find a pivot to change the sample value of row in the specified
  /// direction while preserving tableau consistency, except that if the
  /// direction is down then the pivot may make the specified row take a
  /// negative value. The returned pivot row will be row if and only if the
  /// unknown is unbounded in the specified direction.
  ///
  /// Returns a (row, col) pair denoting a pivot, or an empty Optional if
  /// no valid pivot exists.
  std::optional<Pivot> findPivot(int row, Direction direction) const;

  /// Find a row that can be used to pivot the column in the specified
  /// direction. If skipRow is not null, then this row is excluded
  /// from consideration. The returned pivot will maintain all constraints
  /// except the column itself and skipRow, if it is set. (if these unknowns
  /// are restricted).
  ///
  /// Returns the row to pivot to, or an empty Optional if the column
  /// is unbounded in the specified direction.
  std::optional<unsigned> findPivotRow(std::optional<unsigned> skipRow,
                                       Direction direction, unsigned col) const;

  /// Undo the addition of the last constraint while preserving tableau
  /// consistency.
  void undoLastConstraint() final;

  /// Compute the maximum or minimum of the specified Unknown, depending on
  /// direction. The specified unknown may be pivoted. If the unknown is
  /// restricted, it will have a non-negative sample value on return.
  /// Should not be called if the Simplex is empty.
  ///
  /// Returns a Fraction denoting the optimum, or a null value if no optimum
  /// exists, i.e., if the expression is unbounded in this direction.
  MaybeOptimum<Fraction> computeOptimum(Direction direction, Unknown &u);

  /// Mark the specified unknown redundant. This operation is added to the undo
  /// log and will be undone by rollbacks. The specified unknown must be in row
  /// orientation.
  void markRowRedundant(Unknown &u);

  /// Reduce the given basis, starting at the specified level, using general
  /// basis reduction.
  void reduceBasis(IntMatrix &basis, unsigned level);
};

/// Takes a snapshot of the simplex state on construction and rolls back to the
/// snapshot on destruction.
///
/// Useful for performing operations in a "transient context", all changes from
/// which get rolled back on scope exit.
class SimplexRollbackScopeExit {
public:
  SimplexRollbackScopeExit(SimplexBase &simplex) : simplex(simplex) {
    snapshot = simplex.getSnapshot();
  };
  ~SimplexRollbackScopeExit() { simplex.rollback(snapshot); }

private:
  SimplexBase &simplex;
  unsigned snapshot;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
