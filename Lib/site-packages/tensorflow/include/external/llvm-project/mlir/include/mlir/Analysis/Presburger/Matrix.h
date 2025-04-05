//===- Matrix.h - MLIR Matrix Class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple 2D matrix class that supports reading, writing, resizing,
// swapping rows, and swapping columns. It can hold integers (DynamicAPInt) or
// rational numbers (Fraction).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MATRIX_H
#define MLIR_ANALYSIS_PRESBURGER_MATRIX_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace mlir {
namespace presburger {
using llvm::ArrayRef;
using llvm::MutableArrayRef;
using llvm::raw_ostream;
using llvm::SmallVector;

/// This is a class to represent a resizable matrix.
///
/// More columns and rows can be reserved than are currently used. The data is
/// stored as a single 1D array, viewed as a 2D matrix with nRows rows and
/// nReservedColumns columns, stored in row major form. Thus the element at
/// (i, j) is stored at data[i*nReservedColumns + j]. The reserved but unused
/// columns always have all zero values. The reserved rows are just reserved
/// space in the underlying SmallVector's capacity.
/// This class only works for the types DynamicAPInt and Fraction, since the
/// method implementations are in the Matrix.cpp file. Only these two types have
/// been explicitly instantiated there.
template <typename T>
class Matrix {
  static_assert(std::is_same_v<T, DynamicAPInt> || std::is_same_v<T, Fraction>,
                "T must be DynamicAPInt or Fraction.");

public:
  Matrix() = delete;

  /// Construct a matrix with the specified number of rows and columns.
  /// The number of reserved rows and columns will be at least the number
  /// specified, and will always be sufficient to accomodate the number of rows
  /// and columns specified.
  ///
  /// Initially, the entries are initialized to ero.
  Matrix(unsigned rows, unsigned columns, unsigned reservedRows = 0,
         unsigned reservedColumns = 0);

  /// Return the identity matrix of the specified dimension.
  static Matrix identity(unsigned dimension);

  /// Access the element at the specified row and column.
  T &at(unsigned row, unsigned column) {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nReservedColumns + column];
  }

  T at(unsigned row, unsigned column) const {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nReservedColumns + column];
  }

  T &operator()(unsigned row, unsigned column) { return at(row, column); }

  T operator()(unsigned row, unsigned column) const { return at(row, column); }

  bool operator==(const Matrix<T> &m) const;

  /// Swap the given columns.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap the given rows.
  void swapRows(unsigned row, unsigned otherRow);

  unsigned getNumRows() const { return nRows; }

  unsigned getNumColumns() const { return nColumns; }

  /// Return the maximum number of rows/columns that can be added without
  /// incurring a reallocation.
  unsigned getNumReservedRows() const;
  unsigned getNumReservedColumns() const { return nReservedColumns; }

  /// Reserve enough space to resize to the specified number of rows without
  /// reallocations.
  void reserveRows(unsigned rows);

  /// Get a [Mutable]ArrayRef corresponding to the specified row.
  MutableArrayRef<T> getRow(unsigned row);
  ArrayRef<T> getRow(unsigned row) const;

  /// Set the specified row to `elems`.
  void setRow(unsigned row, ArrayRef<T> elems);

  /// Insert columns having positions pos, pos + 1, ... pos + count - 1.
  /// Columns that were at positions 0 to pos - 1 will stay where they are;
  /// columns that were at positions pos to nColumns - 1 will be pushed to the
  /// right. pos should be at most nColumns.
  void insertColumns(unsigned pos, unsigned count);
  void insertColumn(unsigned pos);

  /// Insert rows having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// rows that were at positions pos to nColumns - 1 will be pushed to the
  /// right. pos should be at most nRows.
  void insertRows(unsigned pos, unsigned count);
  void insertRow(unsigned pos);

  /// Remove the columns having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// columns that were at positions pos + count - 1 or later will be pushed to
  /// the right. The columns to be deleted must be valid rows: pos + count - 1
  /// must be at most nColumns - 1.
  void removeColumns(unsigned pos, unsigned count);
  void removeColumn(unsigned pos);

  /// Remove the rows having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// rows that were at positions pos + count - 1 or later will be pushed to the
  /// right. The rows to be deleted must be valid rows: pos + count - 1 must be
  /// at most nRows - 1.
  void removeRows(unsigned pos, unsigned count);
  void removeRow(unsigned pos);

  void copyRow(unsigned sourceRow, unsigned targetRow);

  void fillRow(unsigned row, const T &value);
  void fillRow(unsigned row, int64_t value) { fillRow(row, T(value)); }

  /// Add `scale` multiples of the source row to the target row.
  void addToRow(unsigned sourceRow, unsigned targetRow, const T &scale);
  void addToRow(unsigned sourceRow, unsigned targetRow, int64_t scale) {
    addToRow(sourceRow, targetRow, T(scale));
  }
  /// Add `scale` multiples of the rowVec row to the specified row.
  void addToRow(unsigned row, ArrayRef<T> rowVec, const T &scale);

  /// Multiply the specified row by a factor of `scale`.
  void scaleRow(unsigned row, const T &scale);

  /// Add `scale` multiples of the source column to the target column.
  void addToColumn(unsigned sourceColumn, unsigned targetColumn,
                   const T &scale);
  void addToColumn(unsigned sourceColumn, unsigned targetColumn,
                   int64_t scale) {
    addToColumn(sourceColumn, targetColumn, T(scale));
  }

  /// Negate the specified column.
  void negateColumn(unsigned column);

  /// Negate the specified row.
  void negateRow(unsigned row);

  /// Negate the entire matrix.
  void negateMatrix();

  /// The given vector is interpreted as a row vector v. Post-multiply v with
  /// this matrix, say M, and return vM.
  SmallVector<T, 8> preMultiplyWithRow(ArrayRef<T> rowVec) const;

  /// The given vector is interpreted as a column vector v. Pre-multiply v with
  /// this matrix, say M, and return Mv.
  SmallVector<T, 8> postMultiplyWithColumn(ArrayRef<T> colVec) const;

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are initialized
  /// to zero.
  ///
  /// Due to the representation of the matrix, resizing vertically (adding rows)
  /// is less expensive than increasing the number of columns beyond
  /// nReservedColumns.
  void resize(unsigned newNRows, unsigned newNColumns);
  void resizeHorizontally(unsigned newNColumns);
  void resizeVertically(unsigned newNRows);

  /// Add an extra row at the bottom of the matrix and return its position.
  unsigned appendExtraRow();
  /// Same as above, but copy the given elements into the row. The length of
  /// `elems` must be equal to the number of columns.
  unsigned appendExtraRow(ArrayRef<T> elems);

  // Transpose the matrix without modifying it.
  Matrix<T> transpose() const;

  // Copy the cells in the intersection of
  // the rows between `fromRows` and `toRows` and
  // the columns between `fromColumns` and `toColumns`, both inclusive.
  Matrix<T> getSubMatrix(unsigned fromRow, unsigned toRow, unsigned fromColumn,
                         unsigned toColumn) const;

  /// Split the rows of a matrix into two matrices according to which bits are
  /// 1 and which are 0 in a given bitset.
  ///
  /// The first matrix returned has the rows corresponding to 1 and the second
  /// corresponding to 2.
  std::pair<Matrix<T>, Matrix<T>> splitByBitset(ArrayRef<int> indicator);

  /// Print the matrix.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Return whether the Matrix is in a consistent state with all its
  /// invariants satisfied.
  bool hasConsistentState() const;

  /// Move the columns in the source range [srcPos, srcPos + num) to the
  /// specified destination [dstPos, dstPos + num), while moving the columns
  /// adjacent to the source range to the left/right of the shifted columns.
  ///
  /// When moving the source columns right (i.e. dstPos > srcPos), columns that
  /// were at positions [0, srcPos) and [dstPos + num, nCols) will stay where
  /// they are; columns that were at positions [srcPos, srcPos + num) will be
  /// moved to [dstPos, dstPos + num); and columns that were at positions
  /// [srcPos + num, dstPos + num) will be moved to [srcPos, dstPos).
  /// Equivalently, the columns [srcPos + num, dstPos + num) are interchanged
  /// with [srcPos, srcPos + num).
  /// For example, if m = |0 1 2 3 4 5| then:
  /// m.moveColumns(1, 3, 2) will result in m = |0 4 1 2 3 5|; or
  /// m.moveColumns(1, 2, 4) will result in m = |0 3 4 5 1 2|.
  ///
  /// The left shift operation (i.e. dstPos < srcPos) works in a similar way.
  void moveColumns(unsigned srcPos, unsigned num, unsigned dstPos);

protected:
  /// The current number of rows, columns, and reserved columns. The underlying
  /// data vector is viewed as an nRows x nReservedColumns matrix, of which the
  /// first nColumns columns are currently in use, and the remaining are
  /// reserved columns filled with zeros.
  unsigned nRows, nColumns, nReservedColumns;

  /// Stores the data. data.size() is equal to nRows * nReservedColumns.
  /// data.capacity() / nReservedColumns is the number of reserved rows.
  SmallVector<T, 16> data;
};

extern template class Matrix<DynamicAPInt>;
extern template class Matrix<Fraction>;

// An inherited class for integer matrices, with no new data attributes.
// This is only used for the matrix-related methods which apply only
// to integers (hermite normal form computation and row normalisation).
class IntMatrix : public Matrix<DynamicAPInt> {
public:
  IntMatrix(unsigned rows, unsigned columns, unsigned reservedRows = 0,
            unsigned reservedColumns = 0)
      : Matrix<DynamicAPInt>(rows, columns, reservedRows, reservedColumns) {}

  IntMatrix(Matrix<DynamicAPInt> m) : Matrix<DynamicAPInt>(std::move(m)) {}

  /// Return the identity matrix of the specified dimension.
  static IntMatrix identity(unsigned dimension);

  /// Given the current matrix M, returns the matrices H, U such that H is the
  /// column hermite normal form of M, i.e. H = M * U, where U is unimodular and
  /// the matrix H has the following restrictions:
  ///  - H is lower triangular.
  ///  - The leading coefficient (the first non-zero entry from the top, called
  ///    the pivot) of a non-zero column is always strictly below of the leading
  ///    coefficient of the column before it; moreover, it is positive.
  ///  - The elements to the right of the pivots are zero and the elements to
  ///    the left of the pivots are nonnegative and strictly smaller than the
  ///    pivot.
  std::pair<IntMatrix, IntMatrix> computeHermiteNormalForm() const;

  /// Divide the first `nCols` of the specified row by their GCD.
  /// Returns the GCD of the first `nCols` of the specified row.
  DynamicAPInt normalizeRow(unsigned row, unsigned nCols);
  /// Divide the columns of the specified row by their GCD.
  /// Returns the GCD of the columns of the specified row.
  DynamicAPInt normalizeRow(unsigned row);

  // Compute the determinant of the matrix (cubic time).
  // Stores the integer inverse of the matrix in the pointer
  // passed (if any). The pointer is unchanged if the inverse
  // does not exist, which happens iff det = 0.
  // For a matrix M, the integer inverse is the matrix M' such that
  // M x M' = M'  M = det(M) x I.
  // Assert-fails if the matrix is not square.
  DynamicAPInt determinant(IntMatrix *inverse = nullptr) const;
};

// An inherited class for rational matrices, with no new data attributes.
// This class is for functionality that only applies to matrices of fractions.
class FracMatrix : public Matrix<Fraction> {
public:
  FracMatrix(unsigned rows, unsigned columns, unsigned reservedRows = 0,
             unsigned reservedColumns = 0)
      : Matrix<Fraction>(rows, columns, reservedRows, reservedColumns){};

  FracMatrix(Matrix<Fraction> m) : Matrix<Fraction>(std::move(m)){};

  explicit FracMatrix(IntMatrix m);

  /// Return the identity matrix of the specified dimension.
  static FracMatrix identity(unsigned dimension);

  // Compute the determinant of the matrix (cubic time).
  // Stores the inverse of the matrix in the pointer
  // passed (if any). The pointer is unchanged if the inverse
  // does not exist, which happens iff det = 0.
  // Assert-fails if the matrix is not square.
  Fraction determinant(FracMatrix *inverse = nullptr) const;

  // Computes the Gram-Schmidt orthogonalisation
  // of the rows of matrix (cubic time).
  // The rows of the matrix must be linearly independent.
  FracMatrix gramSchmidt() const;

  // Run LLL basis reduction on the matrix, modifying it in-place.
  // The parameter is what [the original
  // paper](https://www.cs.cmu.edu/~avrim/451f11/lectures/lect1129_LLL.pdf)
  // calls `y`, usually 3/4.
  void LLL(Fraction delta);

  // Multiply each row of the matrix by the LCM of the denominators, thereby
  // converting it to an integer matrix.
  IntMatrix normalizeRows() const;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_H
