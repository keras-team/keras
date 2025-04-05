// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEMATRIX_H
#define EIGEN_SPARSEMATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup SparseCore_Module
 *
 * \class SparseMatrix
 *
 * \brief A versatible sparse matrix representation
 *
 * This class implements a more versatile variants of the common \em compressed row/column storage format.
 * Each colmun's (resp. row) non zeros are stored as a pair of value with associated row (resp. colmiun) index.
 * All the non zeros are stored in a single large buffer. Unlike the \em compressed format, there might be extra
 * space in between the nonzeros of two successive colmuns (resp. rows) such that insertion of new non-zero
 * can be done with limited memory reallocation and copies.
 *
 * A call to the function makeCompressed() turns the matrix into the standard \em compressed format
 * compatible with many library.
 *
 * More details on this storage sceheme are given in the \ref TutorialSparse "manual pages".
 *
 * \tparam Scalar_ the scalar type, i.e. the type of the coefficients
 * \tparam Options_ Union of bit flags controlling the storage scheme. Currently the only possibility
 *                 is ColMajor or RowMajor. The default is 0 which means column-major.
 * \tparam StorageIndex_ the type of the indices. It has to be a \b signed type (e.g., short, int, std::ptrdiff_t).
 * Default is \c int.
 *
 * \warning In %Eigen 3.2, the undocumented type \c SparseMatrix::Index was improperly defined as the storage index type
 * (e.g., int), whereas it is now (starting from %Eigen 3.3) deprecated and always defined as Eigen::Index. Codes making
 * use of \c SparseMatrix::Index, might thus likely have to be changed to use \c SparseMatrix::StorageIndex instead.
 *
 * This class can be extended with the help of the plugin mechanism described on the page
 * \ref TopicCustomizing_Plugins by defining the preprocessor symbol \c EIGEN_SPARSEMATRIX_PLUGIN.
 */

namespace internal {
template <typename Scalar_, int Options_, typename StorageIndex_>
struct traits<SparseMatrix<Scalar_, Options_, StorageIndex_>> {
  typedef Scalar_ Scalar;
  typedef StorageIndex_ StorageIndex;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Options = Options_,
    Flags = Options_ | NestByRefBit | LvalueBit | CompressedAccessBit,
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};

template <typename Scalar_, int Options_, typename StorageIndex_, int DiagIndex>
struct traits<Diagonal<SparseMatrix<Scalar_, Options_, StorageIndex_>, DiagIndex>> {
  typedef SparseMatrix<Scalar_, Options_, StorageIndex_> MatrixType;
  typedef typename ref_selector<MatrixType>::type MatrixTypeNested;
  typedef std::remove_reference_t<MatrixTypeNested> MatrixTypeNested_;

  typedef Scalar_ Scalar;
  typedef Dense StorageKind;
  typedef StorageIndex_ StorageIndex;
  typedef MatrixXpr XprKind;

  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = 1,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = 1,
    Flags = LvalueBit
  };
};

template <typename Scalar_, int Options_, typename StorageIndex_, int DiagIndex>
struct traits<Diagonal<const SparseMatrix<Scalar_, Options_, StorageIndex_>, DiagIndex>>
    : public traits<Diagonal<SparseMatrix<Scalar_, Options_, StorageIndex_>, DiagIndex>> {
  enum { Flags = 0 };
};

template <typename StorageIndex>
struct sparse_reserve_op {
  EIGEN_DEVICE_FUNC sparse_reserve_op(Index begin, Index end, Index size) {
    Index range = numext::mini(end - begin, size);
    m_begin = begin;
    m_end = begin + range;
    m_val = StorageIndex(size / range);
    m_remainder = StorageIndex(size % range);
  }
  template <typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE StorageIndex operator()(IndexType i) const {
    if ((i >= m_begin) && (i < m_end))
      return m_val + ((i - m_begin) < m_remainder ? 1 : 0);
    else
      return 0;
  }
  StorageIndex m_val, m_remainder;
  Index m_begin, m_end;
};

template <typename Scalar>
struct functor_traits<sparse_reserve_op<Scalar>> {
  enum { Cost = 1, PacketAccess = false, IsRepeatable = true };
};

}  // end namespace internal

template <typename Scalar_, int Options_, typename StorageIndex_>
class SparseMatrix : public SparseCompressedBase<SparseMatrix<Scalar_, Options_, StorageIndex_>> {
  typedef SparseCompressedBase<SparseMatrix> Base;
  using Base::convert_index;
  friend class SparseVector<Scalar_, 0, StorageIndex_>;
  template <typename, typename, typename, typename, typename>
  friend struct internal::Assignment;

 public:
  using Base::isCompressed;
  using Base::nonZeros;
  EIGEN_SPARSE_PUBLIC_INTERFACE(SparseMatrix)
  using Base::operator+=;
  using Base::operator-=;

  typedef Eigen::Map<SparseMatrix<Scalar, Options_, StorageIndex>> Map;
  typedef Diagonal<SparseMatrix> DiagonalReturnType;
  typedef Diagonal<const SparseMatrix> ConstDiagonalReturnType;
  typedef typename Base::InnerIterator InnerIterator;
  typedef typename Base::ReverseInnerIterator ReverseInnerIterator;

  using Base::IsRowMajor;
  typedef internal::CompressedStorage<Scalar, StorageIndex> Storage;
  enum { Options = Options_ };

  typedef typename Base::IndexVector IndexVector;
  typedef typename Base::ScalarVector ScalarVector;

 protected:
  typedef SparseMatrix<Scalar, IsRowMajor ? ColMajor : RowMajor, StorageIndex> TransposedSparseMatrix;

  Index m_outerSize;
  Index m_innerSize;
  StorageIndex* m_outerIndex;
  StorageIndex* m_innerNonZeros;  // optional, if null then the data is compressed
  Storage m_data;

 public:
  /** \returns the number of rows of the matrix */
  inline Index rows() const { return IsRowMajor ? m_outerSize : m_innerSize; }
  /** \returns the number of columns of the matrix */
  inline Index cols() const { return IsRowMajor ? m_innerSize : m_outerSize; }

  /** \returns the number of rows (resp. columns) of the matrix if the storage order column major (resp. row major) */
  inline Index innerSize() const { return m_innerSize; }
  /** \returns the number of columns (resp. rows) of the matrix if the storage order column major (resp. row major) */
  inline Index outerSize() const { return m_outerSize; }

  /** \returns a const pointer to the array of values.
   * This function is aimed at interoperability with other libraries.
   * \sa innerIndexPtr(), outerIndexPtr() */
  inline const Scalar* valuePtr() const { return m_data.valuePtr(); }
  /** \returns a non-const pointer to the array of values.
   * This function is aimed at interoperability with other libraries.
   * \sa innerIndexPtr(), outerIndexPtr() */
  inline Scalar* valuePtr() { return m_data.valuePtr(); }

  /** \returns a const pointer to the array of inner indices.
   * This function is aimed at interoperability with other libraries.
   * \sa valuePtr(), outerIndexPtr() */
  inline const StorageIndex* innerIndexPtr() const { return m_data.indexPtr(); }
  /** \returns a non-const pointer to the array of inner indices.
   * This function is aimed at interoperability with other libraries.
   * \sa valuePtr(), outerIndexPtr() */
  inline StorageIndex* innerIndexPtr() { return m_data.indexPtr(); }

  /** \returns a const pointer to the array of the starting positions of the inner vectors.
   * This function is aimed at interoperability with other libraries.
   * \sa valuePtr(), innerIndexPtr() */
  inline const StorageIndex* outerIndexPtr() const { return m_outerIndex; }
  /** \returns a non-const pointer to the array of the starting positions of the inner vectors.
   * This function is aimed at interoperability with other libraries.
   * \sa valuePtr(), innerIndexPtr() */
  inline StorageIndex* outerIndexPtr() { return m_outerIndex; }

  /** \returns a const pointer to the array of the number of non zeros of the inner vectors.
   * This function is aimed at interoperability with other libraries.
   * \warning it returns the null pointer 0 in compressed mode */
  inline const StorageIndex* innerNonZeroPtr() const { return m_innerNonZeros; }
  /** \returns a non-const pointer to the array of the number of non zeros of the inner vectors.
   * This function is aimed at interoperability with other libraries.
   * \warning it returns the null pointer 0 in compressed mode */
  inline StorageIndex* innerNonZeroPtr() { return m_innerNonZeros; }

  /** \internal */
  inline Storage& data() { return m_data; }
  /** \internal */
  inline const Storage& data() const { return m_data; }

  /** \returns the value of the matrix at position \a i, \a j
   * This function returns Scalar(0) if the element is an explicit \em zero */
  inline Scalar coeff(Index row, Index col) const {
    eigen_assert(row >= 0 && row < rows() && col >= 0 && col < cols());

    const Index outer = IsRowMajor ? row : col;
    const Index inner = IsRowMajor ? col : row;
    Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer + 1];
    return m_data.atInRange(m_outerIndex[outer], end, inner);
  }

  /** \returns a non-const reference to the value of the matrix at position \a i, \a j.
   *
   * If the element does not exist then it is inserted via the insert(Index,Index) function
   * which itself turns the matrix into a non compressed form if that was not the case.
   * The output parameter `inserted` is set to true.
   *
   * Otherwise, if the element does exist, `inserted` will be set to false.
   *
   * This is a O(log(nnz_j)) operation (binary search) plus the cost of insert(Index,Index)
   * function if the element does not already exist.
   */
  inline Scalar& findOrInsertCoeff(Index row, Index col, bool* inserted) {
    eigen_assert(row >= 0 && row < rows() && col >= 0 && col < cols());
    const Index outer = IsRowMajor ? row : col;
    const Index inner = IsRowMajor ? col : row;
    Index start = m_outerIndex[outer];
    Index end = isCompressed() ? m_outerIndex[outer + 1] : m_outerIndex[outer] + m_innerNonZeros[outer];
    eigen_assert(end >= start && "you probably called coeffRef on a non finalized matrix");
    Index dst = start == end ? end : m_data.searchLowerIndex(start, end, inner);
    if (dst == end) {
      Index capacity = m_outerIndex[outer + 1] - end;
      if (capacity > 0) {
        // implies uncompressed: push to back of vector
        m_innerNonZeros[outer]++;
        m_data.index(end) = StorageIndex(inner);
        m_data.value(end) = Scalar(0);
        if (inserted != nullptr) {
          *inserted = true;
        }
        return m_data.value(end);
      }
    }
    if ((dst < end) && (m_data.index(dst) == inner)) {
      // this coefficient exists, return a refernece to it
      if (inserted != nullptr) {
        *inserted = false;
      }
      return m_data.value(dst);
    } else {
      if (inserted != nullptr) {
        *inserted = true;
      }
      // insertion will require reconfiguring the buffer
      return insertAtByOuterInner(outer, inner, dst);
    }
  }

  /** \returns a non-const reference to the value of the matrix at position \a i, \a j
   *
   * If the element does not exist then it is inserted via the insert(Index,Index) function
   * which itself turns the matrix into a non compressed form if that was not the case.
   *
   * This is a O(log(nnz_j)) operation (binary search) plus the cost of insert(Index,Index)
   * function if the element does not already exist.
   */
  inline Scalar& coeffRef(Index row, Index col) { return findOrInsertCoeff(row, col, nullptr); }

  /** \returns a reference to a novel non zero coefficient with coordinates \a row x \a col.
   * The non zero coefficient must \b not already exist.
   *
   * If the matrix \c *this is in compressed mode, then \c *this is turned into uncompressed
   * mode while reserving room for 2 x this->innerSize() non zeros if reserve(Index) has not been called earlier.
   * In this case, the insertion procedure is optimized for a \e sequential insertion mode where elements are assumed to
   * be inserted by increasing outer-indices.
   *
   * If that's not the case, then it is strongly recommended to either use a triplet-list to assemble the matrix, or to
   * first call reserve(const SizesType &) to reserve the appropriate number of non-zero elements per inner vector.
   *
   * Assuming memory has been appropriately reserved, this function performs a sorted insertion in O(1)
   * if the elements of each inner vector are inserted in increasing inner index order, and in O(nnz_j) for a random
   * insertion.
   *
   */
  inline Scalar& insert(Index row, Index col);

 public:
  /** Removes all non zeros but keep allocated memory
   *
   * This function does not free the currently allocated memory. To release as much as memory as possible,
   * call \code mat.data().squeeze(); \endcode after resizing it.
   *
   * \sa resize(Index,Index), data()
   */
  inline void setZero() {
    m_data.clear();
    std::fill_n(m_outerIndex, m_outerSize + 1, StorageIndex(0));
    if (m_innerNonZeros) {
      std::fill_n(m_innerNonZeros, m_outerSize, StorageIndex(0));
    }
  }

  /** Preallocates \a reserveSize non zeros.
   *
   * Precondition: the matrix must be in compressed mode. */
  inline void reserve(Index reserveSize) {
    eigen_assert(isCompressed() && "This function does not make sense in non compressed mode.");
    m_data.reserve(reserveSize);
  }

#ifdef EIGEN_PARSED_BY_DOXYGEN
  /** Preallocates \a reserveSize[\c j] non zeros for each column (resp. row) \c j.
    *
    * This function turns the matrix in non-compressed mode.
    *
    * The type \c SizesType must expose the following interface:
      \code
      typedef value_type;
      const value_type& operator[](i) const;
      \endcode
    * for \c i in the [0,this->outerSize()[ range.
    * Typical choices include std::vector<int>, Eigen::VectorXi, Eigen::VectorXi::Constant, etc.
    */
  template <class SizesType>
  inline void reserve(const SizesType& reserveSizes);
#else
  template <class SizesType>
  inline void reserve(const SizesType& reserveSizes,
                      const typename SizesType::value_type& enableif = typename SizesType::value_type()) {
    EIGEN_UNUSED_VARIABLE(enableif);
    reserveInnerVectors(reserveSizes);
  }
#endif  // EIGEN_PARSED_BY_DOXYGEN
 protected:
  template <class SizesType>
  inline void reserveInnerVectors(const SizesType& reserveSizes) {
    if (isCompressed()) {
      Index totalReserveSize = 0;
      for (Index j = 0; j < m_outerSize; ++j) totalReserveSize += internal::convert_index<Index>(reserveSizes[j]);

      // if reserveSizes is empty, don't do anything!
      if (totalReserveSize == 0) return;

      // turn the matrix into non-compressed mode
      m_innerNonZeros = internal::conditional_aligned_new_auto<StorageIndex, true>(m_outerSize);

      // temporarily use m_innerSizes to hold the new starting points.
      StorageIndex* newOuterIndex = m_innerNonZeros;

      Index count = 0;
      for (Index j = 0; j < m_outerSize; ++j) {
        newOuterIndex[j] = internal::convert_index<StorageIndex>(count);
        Index reserveSize = internal::convert_index<Index>(reserveSizes[j]);
        count += reserveSize + internal::convert_index<Index>(m_outerIndex[j + 1] - m_outerIndex[j]);
      }

      m_data.reserve(totalReserveSize);
      StorageIndex previousOuterIndex = m_outerIndex[m_outerSize];
      for (Index j = m_outerSize - 1; j >= 0; --j) {
        StorageIndex innerNNZ = previousOuterIndex - m_outerIndex[j];
        StorageIndex begin = m_outerIndex[j];
        StorageIndex end = begin + innerNNZ;
        StorageIndex target = newOuterIndex[j];
        internal::smart_memmove(innerIndexPtr() + begin, innerIndexPtr() + end, innerIndexPtr() + target);
        internal::smart_memmove(valuePtr() + begin, valuePtr() + end, valuePtr() + target);
        previousOuterIndex = m_outerIndex[j];
        m_outerIndex[j] = newOuterIndex[j];
        m_innerNonZeros[j] = innerNNZ;
      }
      if (m_outerSize > 0)
        m_outerIndex[m_outerSize] = m_outerIndex[m_outerSize - 1] + m_innerNonZeros[m_outerSize - 1] +
                                    internal::convert_index<StorageIndex>(reserveSizes[m_outerSize - 1]);

      m_data.resize(m_outerIndex[m_outerSize]);
    } else {
      StorageIndex* newOuterIndex = internal::conditional_aligned_new_auto<StorageIndex, true>(m_outerSize + 1);

      Index count = 0;
      for (Index j = 0; j < m_outerSize; ++j) {
        newOuterIndex[j] = internal::convert_index<StorageIndex>(count);
        Index alreadyReserved =
            internal::convert_index<Index>(m_outerIndex[j + 1] - m_outerIndex[j] - m_innerNonZeros[j]);
        Index reserveSize = internal::convert_index<Index>(reserveSizes[j]);
        Index toReserve = numext::maxi(reserveSize, alreadyReserved);
        count += toReserve + internal::convert_index<Index>(m_innerNonZeros[j]);
      }
      newOuterIndex[m_outerSize] = internal::convert_index<StorageIndex>(count);

      m_data.resize(count);
      for (Index j = m_outerSize - 1; j >= 0; --j) {
        StorageIndex innerNNZ = m_innerNonZeros[j];
        StorageIndex begin = m_outerIndex[j];
        StorageIndex target = newOuterIndex[j];
        m_data.moveChunk(begin, target, innerNNZ);
      }

      std::swap(m_outerIndex, newOuterIndex);
      internal::conditional_aligned_delete_auto<StorageIndex, true>(newOuterIndex, m_outerSize + 1);
    }
  }

 public:
  //--- low level purely coherent filling ---

  /** \internal
   * \returns a reference to the non zero coefficient at position \a row, \a col assuming that:
   * - the nonzero does not already exist
   * - the new coefficient is the last one according to the storage order
   *
   * Before filling a given inner vector you must call the statVec(Index) function.
   *
   * After an insertion session, you should call the finalize() function.
   *
   * \sa insert, insertBackByOuterInner, startVec */
  inline Scalar& insertBack(Index row, Index col) {
    return insertBackByOuterInner(IsRowMajor ? row : col, IsRowMajor ? col : row);
  }

  /** \internal
   * \sa insertBack, startVec */
  inline Scalar& insertBackByOuterInner(Index outer, Index inner) {
    eigen_assert(Index(m_outerIndex[outer + 1]) == m_data.size() && "Invalid ordered insertion (invalid outer index)");
    eigen_assert((m_outerIndex[outer + 1] - m_outerIndex[outer] == 0 || m_data.index(m_data.size() - 1) < inner) &&
                 "Invalid ordered insertion (invalid inner index)");
    StorageIndex p = m_outerIndex[outer + 1];
    ++m_outerIndex[outer + 1];
    m_data.append(Scalar(0), inner);
    return m_data.value(p);
  }

  /** \internal
   * \warning use it only if you know what you are doing */
  inline Scalar& insertBackByOuterInnerUnordered(Index outer, Index inner) {
    StorageIndex p = m_outerIndex[outer + 1];
    ++m_outerIndex[outer + 1];
    m_data.append(Scalar(0), inner);
    return m_data.value(p);
  }

  /** \internal
   * \sa insertBack, insertBackByOuterInner */
  inline void startVec(Index outer) {
    eigen_assert(m_outerIndex[outer] == Index(m_data.size()) &&
                 "You must call startVec for each inner vector sequentially");
    eigen_assert(m_outerIndex[outer + 1] == 0 && "You must call startVec for each inner vector sequentially");
    m_outerIndex[outer + 1] = m_outerIndex[outer];
  }

  /** \internal
   * Must be called after inserting a set of non zero entries using the low level compressed API.
   */
  inline void finalize() {
    if (isCompressed()) {
      StorageIndex size = internal::convert_index<StorageIndex>(m_data.size());
      Index i = m_outerSize;
      // find the last filled column
      while (i >= 0 && m_outerIndex[i] == 0) --i;
      ++i;
      while (i <= m_outerSize) {
        m_outerIndex[i] = size;
        ++i;
      }
    }
  }

  // remove outer vectors j, j+1 ... j+num-1 and resize the matrix
  void removeOuterVectors(Index j, Index num = 1) {
    eigen_assert(num >= 0 && j >= 0 && j + num <= m_outerSize && "Invalid parameters");

    const Index newRows = IsRowMajor ? m_outerSize - num : rows();
    const Index newCols = IsRowMajor ? cols() : m_outerSize - num;

    const Index begin = j + num;
    const Index end = m_outerSize;
    const Index target = j;

    // if the removed vectors are not empty, uncompress the matrix
    if (m_outerIndex[j + num] > m_outerIndex[j]) uncompress();

    // shift m_outerIndex and m_innerNonZeros [num] to the left
    internal::smart_memmove(m_outerIndex + begin, m_outerIndex + end + 1, m_outerIndex + target);
    if (!isCompressed())
      internal::smart_memmove(m_innerNonZeros + begin, m_innerNonZeros + end, m_innerNonZeros + target);

    // if m_outerIndex[0] > 0, shift the data within the first vector while it is easy to do so
    if (m_outerIndex[0] > StorageIndex(0)) {
      uncompress();
      const Index from = internal::convert_index<Index>(m_outerIndex[0]);
      const Index to = Index(0);
      const Index chunkSize = internal::convert_index<Index>(m_innerNonZeros[0]);
      m_data.moveChunk(from, to, chunkSize);
      m_outerIndex[0] = StorageIndex(0);
    }

    // truncate the matrix to the smaller size
    conservativeResize(newRows, newCols);
  }

  // insert empty outer vectors at indices j, j+1 ... j+num-1 and resize the matrix
  void insertEmptyOuterVectors(Index j, Index num = 1) {
    EIGEN_USING_STD(fill_n);
    eigen_assert(num >= 0 && j >= 0 && j < m_outerSize && "Invalid parameters");

    const Index newRows = IsRowMajor ? m_outerSize + num : rows();
    const Index newCols = IsRowMajor ? cols() : m_outerSize + num;

    const Index begin = j;
    const Index end = m_outerSize;
    const Index target = j + num;

    // expand the matrix to the larger size
    conservativeResize(newRows, newCols);

    // shift m_outerIndex and m_innerNonZeros [num] to the right
    internal::smart_memmove(m_outerIndex + begin, m_outerIndex + end + 1, m_outerIndex + target);
    // m_outerIndex[begin] == m_outerIndex[target], set all indices in this range to same value
    fill_n(m_outerIndex + begin, num, m_outerIndex[begin]);

    if (!isCompressed()) {
      internal::smart_memmove(m_innerNonZeros + begin, m_innerNonZeros + end, m_innerNonZeros + target);
      // set the nonzeros of the newly inserted vectors to 0
      fill_n(m_innerNonZeros + begin, num, StorageIndex(0));
    }
  }

  template <typename InputIterators>
  void setFromTriplets(const InputIterators& begin, const InputIterators& end);

  template <typename InputIterators, typename DupFunctor>
  void setFromTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func);

  template <typename Derived, typename DupFunctor>
  void collapseDuplicates(DenseBase<Derived>& wi, DupFunctor dup_func = DupFunctor());

  template <typename InputIterators>
  void setFromSortedTriplets(const InputIterators& begin, const InputIterators& end);

  template <typename InputIterators, typename DupFunctor>
  void setFromSortedTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func);

  template <typename InputIterators>
  void insertFromTriplets(const InputIterators& begin, const InputIterators& end);

  template <typename InputIterators, typename DupFunctor>
  void insertFromTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func);

  template <typename InputIterators>
  void insertFromSortedTriplets(const InputIterators& begin, const InputIterators& end);

  template <typename InputIterators, typename DupFunctor>
  void insertFromSortedTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func);

  //---

  /** \internal
   * same as insert(Index,Index) except that the indices are given relative to the storage order */
  Scalar& insertByOuterInner(Index j, Index i) {
    eigen_assert(j >= 0 && j < m_outerSize && "invalid outer index");
    eigen_assert(i >= 0 && i < m_innerSize && "invalid inner index");
    Index start = m_outerIndex[j];
    Index end = isCompressed() ? m_outerIndex[j + 1] : start + m_innerNonZeros[j];
    Index dst = start == end ? end : m_data.searchLowerIndex(start, end, i);
    if (dst == end) {
      Index capacity = m_outerIndex[j + 1] - end;
      if (capacity > 0) {
        // implies uncompressed: push to back of vector
        m_innerNonZeros[j]++;
        m_data.index(end) = StorageIndex(i);
        m_data.value(end) = Scalar(0);
        return m_data.value(end);
      }
    }
    eigen_assert((dst == end || m_data.index(dst) != i) &&
                 "you cannot insert an element that already exists, you must call coeffRef to this end");
    return insertAtByOuterInner(j, i, dst);
  }

  /** Turns the matrix into the \em compressed format.
   */
  void makeCompressed() {
    if (isCompressed()) return;

    eigen_internal_assert(m_outerIndex != 0 && m_outerSize > 0);

    StorageIndex start = m_outerIndex[1];
    m_outerIndex[1] = m_innerNonZeros[0];
    // try to move fewer, larger contiguous chunks
    Index copyStart = start;
    Index copyTarget = m_innerNonZeros[0];
    for (Index j = 1; j < m_outerSize; j++) {
      StorageIndex end = start + m_innerNonZeros[j];
      StorageIndex nextStart = m_outerIndex[j + 1];
      // dont forget to move the last chunk!
      bool breakUpCopy = (end != nextStart) || (j == m_outerSize - 1);
      if (breakUpCopy) {
        Index chunkSize = end - copyStart;
        if (chunkSize > 0) m_data.moveChunk(copyStart, copyTarget, chunkSize);
        copyStart = nextStart;
        copyTarget += chunkSize;
      }
      start = nextStart;
      m_outerIndex[j + 1] = m_outerIndex[j] + m_innerNonZeros[j];
    }
    m_data.resize(m_outerIndex[m_outerSize]);

    // release as much memory as possible
    internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
    m_innerNonZeros = 0;
    m_data.squeeze();
  }

  /** Turns the matrix into the uncompressed mode */
  void uncompress() {
    if (!isCompressed()) return;
    m_innerNonZeros = internal::conditional_aligned_new_auto<StorageIndex, true>(m_outerSize);
    if (m_outerIndex[m_outerSize] == 0)
      std::fill_n(m_innerNonZeros, m_outerSize, StorageIndex(0));
    else
      for (Index j = 0; j < m_outerSize; j++) m_innerNonZeros[j] = m_outerIndex[j + 1] - m_outerIndex[j];
  }

  /** Suppresses all nonzeros which are \b much \b smaller \b than \a reference under the tolerance \a epsilon */
  void prune(const Scalar& reference, const RealScalar& epsilon = NumTraits<RealScalar>::dummy_precision()) {
    prune(default_prunning_func(reference, epsilon));
  }

  /** Turns the matrix into compressed format, and suppresses all nonzeros which do not satisfy the predicate \a keep.
   * The functor type \a KeepFunc must implement the following function:
   * \code
   * bool operator() (const Index& row, const Index& col, const Scalar& value) const;
   * \endcode
   * \sa prune(Scalar,RealScalar)
   */
  template <typename KeepFunc>
  void prune(const KeepFunc& keep = KeepFunc()) {
    StorageIndex k = 0;
    for (Index j = 0; j < m_outerSize; ++j) {
      StorageIndex previousStart = m_outerIndex[j];
      if (isCompressed())
        m_outerIndex[j] = k;
      else
        k = m_outerIndex[j];
      StorageIndex end = isCompressed() ? m_outerIndex[j + 1] : previousStart + m_innerNonZeros[j];
      for (StorageIndex i = previousStart; i < end; ++i) {
        StorageIndex row = IsRowMajor ? StorageIndex(j) : m_data.index(i);
        StorageIndex col = IsRowMajor ? m_data.index(i) : StorageIndex(j);
        bool keepEntry = keep(row, col, m_data.value(i));
        if (keepEntry) {
          m_data.value(k) = m_data.value(i);
          m_data.index(k) = m_data.index(i);
          ++k;
        } else if (!isCompressed())
          m_innerNonZeros[j]--;
      }
    }
    if (isCompressed()) {
      m_outerIndex[m_outerSize] = k;
      m_data.resize(k, 0);
    }
  }

  /** Resizes the matrix to a \a rows x \a cols matrix leaving old values untouched.
   *
   * If the sizes of the matrix are decreased, then the matrix is turned to \b uncompressed-mode
   * and the storage of the out of bounds coefficients is kept and reserved.
   * Call makeCompressed() to pack the entries and squeeze extra memory.
   *
   * \sa reserve(), setZero(), makeCompressed()
   */
  void conservativeResize(Index rows, Index cols) {
    // If one dimension is null, then there is nothing to be preserved
    if (rows == 0 || cols == 0) return resize(rows, cols);

    Index newOuterSize = IsRowMajor ? rows : cols;
    Index newInnerSize = IsRowMajor ? cols : rows;

    Index innerChange = newInnerSize - m_innerSize;
    Index outerChange = newOuterSize - m_outerSize;

    if (outerChange != 0) {
      m_outerIndex = internal::conditional_aligned_realloc_new_auto<StorageIndex, true>(m_outerIndex, newOuterSize + 1,
                                                                                        m_outerSize + 1);

      if (!isCompressed())
        m_innerNonZeros = internal::conditional_aligned_realloc_new_auto<StorageIndex, true>(m_innerNonZeros,
                                                                                             newOuterSize, m_outerSize);

      if (outerChange > 0) {
        StorageIndex lastIdx = m_outerSize == 0 ? StorageIndex(0) : m_outerIndex[m_outerSize];
        std::fill_n(m_outerIndex + m_outerSize, outerChange + 1, lastIdx);

        if (!isCompressed()) std::fill_n(m_innerNonZeros + m_outerSize, outerChange, StorageIndex(0));
      }
    }
    m_outerSize = newOuterSize;

    if (innerChange < 0) {
      for (Index j = 0; j < m_outerSize; j++) {
        Index start = m_outerIndex[j];
        Index end = isCompressed() ? m_outerIndex[j + 1] : start + m_innerNonZeros[j];
        Index lb = m_data.searchLowerIndex(start, end, newInnerSize);
        if (lb != end) {
          uncompress();
          m_innerNonZeros[j] = StorageIndex(lb - start);
        }
      }
    }
    m_innerSize = newInnerSize;

    Index newSize = m_outerIndex[m_outerSize];
    eigen_assert(newSize <= m_data.size());
    m_data.resize(newSize);
  }

  /** Resizes the matrix to a \a rows x \a cols matrix and initializes it to zero.
   *
   * This function does not free the currently allocated memory. To release as much as memory as possible,
   * call \code mat.data().squeeze(); \endcode after resizing it.
   *
   * \sa reserve(), setZero()
   */
  void resize(Index rows, Index cols) {
    const Index outerSize = IsRowMajor ? rows : cols;
    m_innerSize = IsRowMajor ? cols : rows;
    m_data.clear();

    if ((m_outerIndex == 0) || (m_outerSize != outerSize)) {
      m_outerIndex = internal::conditional_aligned_realloc_new_auto<StorageIndex, true>(m_outerIndex, outerSize + 1,
                                                                                        m_outerSize + 1);
      m_outerSize = outerSize;
    }

    internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
    m_innerNonZeros = 0;

    std::fill_n(m_outerIndex, m_outerSize + 1, StorageIndex(0));
  }

  /** \internal
   * Resize the nonzero vector to \a size */
  void resizeNonZeros(Index size) { m_data.resize(size); }

  /** \returns a const expression of the diagonal coefficients. */
  const ConstDiagonalReturnType diagonal() const { return ConstDiagonalReturnType(*this); }

  /** \returns a read-write expression of the diagonal coefficients.
   * \warning If the diagonal entries are written, then all diagonal
   * entries \b must already exist, otherwise an assertion will be raised.
   */
  DiagonalReturnType diagonal() { return DiagonalReturnType(*this); }

  /** Default constructor yielding an empty \c 0 \c x \c 0 matrix */
  inline SparseMatrix() : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0) { resize(0, 0); }

  /** Constructs a \a rows \c x \a cols empty matrix */
  inline SparseMatrix(Index rows, Index cols) : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0) {
    resize(rows, cols);
  }

  /** Constructs a sparse matrix from the sparse expression \a other */
  template <typename OtherDerived>
  inline SparseMatrix(const SparseMatrixBase<OtherDerived>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0) {
    EIGEN_STATIC_ASSERT(
        (internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    const bool needToTranspose = (Flags & RowMajorBit) != (internal::evaluator<OtherDerived>::Flags & RowMajorBit);
    if (needToTranspose)
      *this = other.derived();
    else {
#ifdef EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
      EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
#endif
      internal::call_assignment_no_alias(*this, other.derived());
    }
  }

  /** Constructs a sparse matrix from the sparse selfadjoint view \a other */
  template <typename OtherDerived, unsigned int UpLo>
  inline SparseMatrix(const SparseSelfAdjointView<OtherDerived, UpLo>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0) {
    Base::operator=(other);
  }

  /** Move constructor */
  inline SparseMatrix(SparseMatrix&& other) : SparseMatrix() { this->swap(other); }

  template <typename OtherDerived>
  inline SparseMatrix(SparseCompressedBase<OtherDerived>&& other) : SparseMatrix() {
    *this = other.derived().markAsRValue();
  }

  /** Copy constructor (it performs a deep copy) */
  inline SparseMatrix(const SparseMatrix& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0) {
    *this = other.derived();
  }

  /** \brief Copy constructor with in-place evaluation */
  template <typename OtherDerived>
  SparseMatrix(const ReturnByValue<OtherDerived>& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0) {
    initAssignment(other);
    other.evalTo(*this);
  }

  /** \brief Copy constructor with in-place evaluation */
  template <typename OtherDerived>
  explicit SparseMatrix(const DiagonalBase<OtherDerived>& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0) {
    *this = other.derived();
  }

  /** Swaps the content of two sparse matrices of the same type.
   * This is a fast operation that simply swaps the underlying pointers and parameters. */
  inline void swap(SparseMatrix& other) {
    // EIGEN_DBG_SPARSE(std::cout << "SparseMatrix:: swap\n");
    std::swap(m_outerIndex, other.m_outerIndex);
    std::swap(m_innerSize, other.m_innerSize);
    std::swap(m_outerSize, other.m_outerSize);
    std::swap(m_innerNonZeros, other.m_innerNonZeros);
    m_data.swap(other.m_data);
  }

  /** Sets *this to the identity matrix.
   * This function also turns the matrix into compressed mode, and drop any reserved memory. */
  inline void setIdentity() {
    eigen_assert(m_outerSize == m_innerSize && "ONLY FOR SQUARED MATRICES");
    internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
    m_innerNonZeros = 0;
    m_data.resize(m_outerSize);
    // is it necessary to squeeze?
    m_data.squeeze();
    std::iota(m_outerIndex, m_outerIndex + m_outerSize + 1, StorageIndex(0));
    std::iota(innerIndexPtr(), innerIndexPtr() + m_outerSize, StorageIndex(0));
    std::fill_n(valuePtr(), m_outerSize, Scalar(1));
  }

  inline SparseMatrix& operator=(const SparseMatrix& other) {
    if (other.isRValue()) {
      swap(other.const_cast_derived());
    } else if (this != &other) {
#ifdef EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
      EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
#endif
      initAssignment(other);
      if (other.isCompressed()) {
        internal::smart_copy(other.m_outerIndex, other.m_outerIndex + m_outerSize + 1, m_outerIndex);
        m_data = other.m_data;
      } else {
        Base::operator=(other);
      }
    }
    return *this;
  }

  inline SparseMatrix& operator=(SparseMatrix&& other) {
    this->swap(other);
    return *this;
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename OtherDerived>
  inline SparseMatrix& operator=(const EigenBase<OtherDerived>& other) {
    return Base::operator=(other.derived());
  }

  template <typename Lhs, typename Rhs>
  inline SparseMatrix& operator=(const Product<Lhs, Rhs, AliasFreeProduct>& other);
#endif  // EIGEN_PARSED_BY_DOXYGEN

  template <typename OtherDerived>
  EIGEN_DONT_INLINE SparseMatrix& operator=(const SparseMatrixBase<OtherDerived>& other);

  template <typename OtherDerived>
  inline SparseMatrix& operator=(SparseCompressedBase<OtherDerived>&& other) {
    *this = other.derived().markAsRValue();
    return *this;
  }

#ifndef EIGEN_NO_IO
  friend std::ostream& operator<<(std::ostream& s, const SparseMatrix& m) {
    EIGEN_DBG_SPARSE(
        s << "Nonzero entries:\n"; if (m.isCompressed()) {
          for (Index i = 0; i < m.nonZeros(); ++i) s << "(" << m.m_data.value(i) << "," << m.m_data.index(i) << ") ";
        } else {
          for (Index i = 0; i < m.outerSize(); ++i) {
            Index p = m.m_outerIndex[i];
            Index pe = m.m_outerIndex[i] + m.m_innerNonZeros[i];
            Index k = p;
            for (; k < pe; ++k) {
              s << "(" << m.m_data.value(k) << "," << m.m_data.index(k) << ") ";
            }
            for (; k < m.m_outerIndex[i + 1]; ++k) {
              s << "(_,_) ";
            }
          }
        } s << std::endl;
        s << std::endl; s << "Outer pointers:\n";
        for (Index i = 0; i < m.outerSize(); ++i) { s << m.m_outerIndex[i] << " "; } s << " $" << std::endl;
        if (!m.isCompressed()) {
          s << "Inner non zeros:\n";
          for (Index i = 0; i < m.outerSize(); ++i) {
            s << m.m_innerNonZeros[i] << " ";
          }
          s << " $" << std::endl;
        } s
        << std::endl;);
    s << static_cast<const SparseMatrixBase<SparseMatrix>&>(m);
    return s;
  }
#endif

  /** Destructor */
  inline ~SparseMatrix() {
    internal::conditional_aligned_delete_auto<StorageIndex, true>(m_outerIndex, m_outerSize + 1);
    internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
  }

  /** Overloaded for performance */
  Scalar sum() const;

#ifdef EIGEN_SPARSEMATRIX_PLUGIN
#include EIGEN_SPARSEMATRIX_PLUGIN
#endif

 protected:
  template <typename Other>
  void initAssignment(const Other& other) {
    resize(other.rows(), other.cols());
    internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
    m_innerNonZeros = 0;
  }

  /** \internal
   * \sa insert(Index,Index) */
  EIGEN_DEPRECATED EIGEN_DONT_INLINE Scalar& insertCompressed(Index row, Index col);

  /** \internal
   * A vector object that is equal to 0 everywhere but v at the position i */
  class SingletonVector {
    StorageIndex m_index;
    StorageIndex m_value;

   public:
    typedef StorageIndex value_type;
    SingletonVector(Index i, Index v) : m_index(convert_index(i)), m_value(convert_index(v)) {}

    StorageIndex operator[](Index i) const { return i == m_index ? m_value : 0; }
  };

  /** \internal
   * \sa insert(Index,Index) */
  EIGEN_DEPRECATED EIGEN_DONT_INLINE Scalar& insertUncompressed(Index row, Index col);

 public:
  /** \internal
   * \sa insert(Index,Index) */
  EIGEN_STRONG_INLINE Scalar& insertBackUncompressed(Index row, Index col) {
    const Index outer = IsRowMajor ? row : col;
    const Index inner = IsRowMajor ? col : row;

    eigen_assert(!isCompressed());
    eigen_assert(m_innerNonZeros[outer] <= (m_outerIndex[outer + 1] - m_outerIndex[outer]));

    Index p = m_outerIndex[outer] + m_innerNonZeros[outer]++;
    m_data.index(p) = StorageIndex(inner);
    m_data.value(p) = Scalar(0);
    return m_data.value(p);
  }

 protected:
  struct IndexPosPair {
    IndexPosPair(Index a_i, Index a_p) : i(a_i), p(a_p) {}
    Index i;
    Index p;
  };

  /** \internal assign \a diagXpr to the diagonal of \c *this
   * There are different strategies:
   *   1 - if *this is overwritten (Func==assign_op) or *this is empty, then we can work treat *this as a dense vector
   * expression. 2 - otherwise, for each diagonal coeff, 2.a - if it already exists, then we update it, 2.b - if the
   * correct position is at the end of the vector, and there is capacity, push to back 2.b - otherwise, the insertion
   * requires a data move, record insertion locations and handle in a second pass 3 - at the end, if some entries failed
   * to be updated in-place, then we alloc a new buffer, copy each chunk at the right position, and insert the new
   * elements.
   */
  template <typename DiagXpr, typename Func>
  void assignDiagonal(const DiagXpr diagXpr, const Func& assignFunc) {
    constexpr StorageIndex kEmptyIndexVal(-1);
    typedef typename ScalarVector::AlignedMapType ValueMap;

    Index n = diagXpr.size();

    const bool overwrite = internal::is_same<Func, internal::assign_op<Scalar, Scalar>>::value;
    if (overwrite) {
      if ((m_outerSize != n) || (m_innerSize != n)) resize(n, n);
    }

    if (m_data.size() == 0 || overwrite) {
      internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
      m_innerNonZeros = 0;
      resizeNonZeros(n);
      ValueMap valueMap(valuePtr(), n);
      std::iota(m_outerIndex, m_outerIndex + n + 1, StorageIndex(0));
      std::iota(innerIndexPtr(), innerIndexPtr() + n, StorageIndex(0));
      valueMap.setZero();
      internal::call_assignment_no_alias(valueMap, diagXpr, assignFunc);
    } else {
      internal::evaluator<DiagXpr> diaEval(diagXpr);

      ei_declare_aligned_stack_constructed_variable(StorageIndex, tmp, n, 0);
      typename IndexVector::AlignedMapType insertionLocations(tmp, n);
      insertionLocations.setConstant(kEmptyIndexVal);

      Index deferredInsertions = 0;
      Index shift = 0;

      for (Index j = 0; j < n; j++) {
        Index begin = m_outerIndex[j];
        Index end = isCompressed() ? m_outerIndex[j + 1] : begin + m_innerNonZeros[j];
        Index capacity = m_outerIndex[j + 1] - end;
        Index dst = m_data.searchLowerIndex(begin, end, j);
        // the entry exists: update it now
        if (dst != end && m_data.index(dst) == StorageIndex(j))
          assignFunc.assignCoeff(m_data.value(dst), diaEval.coeff(j));
        // the entry belongs at the back of the vector: push to back
        else if (dst == end && capacity > 0)
          assignFunc.assignCoeff(insertBackUncompressed(j, j), diaEval.coeff(j));
        // the insertion requires a data move, record insertion location and handle in second pass
        else {
          insertionLocations.coeffRef(j) = StorageIndex(dst);
          deferredInsertions++;
          // if there is no capacity, all vectors to the right of this are shifted
          if (capacity == 0) shift++;
        }
      }

      if (deferredInsertions > 0) {
        m_data.resize(m_data.size() + shift);
        Index copyEnd = isCompressed() ? m_outerIndex[m_outerSize]
                                       : m_outerIndex[m_outerSize - 1] + m_innerNonZeros[m_outerSize - 1];
        for (Index j = m_outerSize - 1; deferredInsertions > 0; j--) {
          Index begin = m_outerIndex[j];
          Index end = isCompressed() ? m_outerIndex[j + 1] : begin + m_innerNonZeros[j];
          Index capacity = m_outerIndex[j + 1] - end;

          bool doInsertion = insertionLocations(j) >= 0;
          bool breakUpCopy = doInsertion && (capacity > 0);
          // break up copy for sorted insertion into inactive nonzeros
          // optionally, add another criterium, i.e. 'breakUpCopy || (capacity > threhsold)'
          // where `threshold >= 0` to skip inactive nonzeros in each vector
          // this reduces the total number of copied elements, but requires more moveChunk calls
          if (breakUpCopy) {
            Index copyBegin = m_outerIndex[j + 1];
            Index to = copyBegin + shift;
            Index chunkSize = copyEnd - copyBegin;
            m_data.moveChunk(copyBegin, to, chunkSize);
            copyEnd = end;
          }

          m_outerIndex[j + 1] += shift;

          if (doInsertion) {
            // if there is capacity, shift into the inactive nonzeros
            if (capacity > 0) shift++;
            Index copyBegin = insertionLocations(j);
            Index to = copyBegin + shift;
            Index chunkSize = copyEnd - copyBegin;
            m_data.moveChunk(copyBegin, to, chunkSize);
            Index dst = to - 1;
            m_data.index(dst) = StorageIndex(j);
            m_data.value(dst) = Scalar(0);
            assignFunc.assignCoeff(m_data.value(dst), diaEval.coeff(j));
            if (!isCompressed()) m_innerNonZeros[j]++;
            shift--;
            deferredInsertions--;
            copyEnd = copyBegin;
          }
        }
      }
      eigen_assert((shift == 0) && (deferredInsertions == 0));
    }
  }

  /* These functions are used to avoid a redundant binary search operation in functions such as coeffRef() and assume
   * `dst` is the appropriate sorted insertion point */
  EIGEN_STRONG_INLINE Scalar& insertAtByOuterInner(Index outer, Index inner, Index dst);
  Scalar& insertCompressedAtByOuterInner(Index outer, Index inner, Index dst);
  Scalar& insertUncompressedAtByOuterInner(Index outer, Index inner, Index dst);

 private:
  EIGEN_STATIC_ASSERT(NumTraits<StorageIndex>::IsSigned, THE_INDEX_TYPE_MUST_BE_A_SIGNED_TYPE)
  EIGEN_STATIC_ASSERT((Options & (ColMajor | RowMajor)) == Options, INVALID_MATRIX_TEMPLATE_PARAMETERS)

  struct default_prunning_func {
    default_prunning_func(const Scalar& ref, const RealScalar& eps) : reference(ref), epsilon(eps) {}
    inline bool operator()(const Index&, const Index&, const Scalar& value) const {
      return !internal::isMuchSmallerThan(value, reference, epsilon);
    }
    Scalar reference;
    RealScalar epsilon;
  };
};

namespace internal {

// Creates a compressed sparse matrix from a range of unsorted triplets
// Requires temporary storage to handle duplicate entries
template <typename InputIterator, typename SparseMatrixType, typename DupFunctor>
void set_from_triplets(const InputIterator& begin, const InputIterator& end, SparseMatrixType& mat,
                       DupFunctor dup_func) {
  constexpr bool IsRowMajor = SparseMatrixType::IsRowMajor;
  using StorageIndex = typename SparseMatrixType::StorageIndex;
  using IndexMap = typename VectorX<StorageIndex>::AlignedMapType;
  using TransposedSparseMatrix =
      SparseMatrix<typename SparseMatrixType::Scalar, IsRowMajor ? ColMajor : RowMajor, StorageIndex>;

  if (begin == end) return;

  // There are two strategies to consider for constructing a matrix from unordered triplets:
  // A) construct the 'mat' in its native storage order and sort in-place (less memory); or,
  // B) construct the transposed matrix and use an implicit sort upon assignment to `mat` (less time).
  // This routine uses B) for faster execution time.
  TransposedSparseMatrix trmat(mat.rows(), mat.cols());

  // scan triplets to determine allocation size before constructing matrix
  Index nonZeros = 0;
  for (InputIterator it(begin); it != end; ++it) {
    eigen_assert(it->row() >= 0 && it->row() < mat.rows() && it->col() >= 0 && it->col() < mat.cols());
    StorageIndex j = convert_index<StorageIndex>(IsRowMajor ? it->col() : it->row());
    if (nonZeros == NumTraits<StorageIndex>::highest()) internal::throw_std_bad_alloc();
    trmat.outerIndexPtr()[j + 1]++;
    nonZeros++;
  }

  std::partial_sum(trmat.outerIndexPtr(), trmat.outerIndexPtr() + trmat.outerSize() + 1, trmat.outerIndexPtr());
  eigen_assert(nonZeros == trmat.outerIndexPtr()[trmat.outerSize()]);
  trmat.resizeNonZeros(nonZeros);

  // construct temporary array to track insertions (outersize) and collapse duplicates (innersize)
  ei_declare_aligned_stack_constructed_variable(StorageIndex, tmp, numext::maxi(mat.innerSize(), mat.outerSize()), 0);
  smart_copy(trmat.outerIndexPtr(), trmat.outerIndexPtr() + trmat.outerSize(), tmp);

  // push triplets to back of each vector
  for (InputIterator it(begin); it != end; ++it) {
    StorageIndex j = convert_index<StorageIndex>(IsRowMajor ? it->col() : it->row());
    StorageIndex i = convert_index<StorageIndex>(IsRowMajor ? it->row() : it->col());
    StorageIndex k = tmp[j];
    trmat.data().index(k) = i;
    trmat.data().value(k) = it->value();
    tmp[j]++;
  }

  IndexMap wi(tmp, trmat.innerSize());
  trmat.collapseDuplicates(wi, dup_func);
  // implicit sorting
  mat = trmat;
}

// Creates a compressed sparse matrix from a sorted range of triplets
template <typename InputIterator, typename SparseMatrixType, typename DupFunctor>
void set_from_triplets_sorted(const InputIterator& begin, const InputIterator& end, SparseMatrixType& mat,
                              DupFunctor dup_func) {
  constexpr bool IsRowMajor = SparseMatrixType::IsRowMajor;
  using StorageIndex = typename SparseMatrixType::StorageIndex;

  if (begin == end) return;

  constexpr StorageIndex kEmptyIndexValue(-1);
  // deallocate inner nonzeros if present and zero outerIndexPtr
  mat.resize(mat.rows(), mat.cols());
  // use outer indices to count non zero entries (excluding duplicate entries)
  StorageIndex previous_j = kEmptyIndexValue;
  StorageIndex previous_i = kEmptyIndexValue;
  // scan triplets to determine allocation size before constructing matrix
  Index nonZeros = 0;
  for (InputIterator it(begin); it != end; ++it) {
    eigen_assert(it->row() >= 0 && it->row() < mat.rows() && it->col() >= 0 && it->col() < mat.cols());
    StorageIndex j = convert_index<StorageIndex>(IsRowMajor ? it->row() : it->col());
    StorageIndex i = convert_index<StorageIndex>(IsRowMajor ? it->col() : it->row());
    eigen_assert(j > previous_j || (j == previous_j && i >= previous_i));
    // identify duplicates by examining previous location
    bool duplicate = (previous_j == j) && (previous_i == i);
    if (!duplicate) {
      if (nonZeros == NumTraits<StorageIndex>::highest()) internal::throw_std_bad_alloc();
      nonZeros++;
      mat.outerIndexPtr()[j + 1]++;
      previous_j = j;
      previous_i = i;
    }
  }

  // finalize outer indices and allocate memory
  std::partial_sum(mat.outerIndexPtr(), mat.outerIndexPtr() + mat.outerSize() + 1, mat.outerIndexPtr());
  eigen_assert(nonZeros == mat.outerIndexPtr()[mat.outerSize()]);
  mat.resizeNonZeros(nonZeros);

  previous_i = kEmptyIndexValue;
  previous_j = kEmptyIndexValue;
  Index back = 0;
  for (InputIterator it(begin); it != end; ++it) {
    StorageIndex j = convert_index<StorageIndex>(IsRowMajor ? it->row() : it->col());
    StorageIndex i = convert_index<StorageIndex>(IsRowMajor ? it->col() : it->row());
    bool duplicate = (previous_j == j) && (previous_i == i);
    if (duplicate) {
      mat.data().value(back - 1) = dup_func(mat.data().value(back - 1), it->value());
    } else {
      // push triplets to back
      mat.data().index(back) = i;
      mat.data().value(back) = it->value();
      previous_j = j;
      previous_i = i;
      back++;
    }
  }
  eigen_assert(back == nonZeros);
  // matrix is finalized
}

// thin wrapper around a generic binary functor to use the sparse disjunction evaulator instead of the default
// "arithmetic" evaulator
template <typename DupFunctor, typename LhsScalar, typename RhsScalar = LhsScalar>
struct scalar_disjunction_op {
  using result_type = typename result_of<DupFunctor(LhsScalar, RhsScalar)>::type;
  scalar_disjunction_op(const DupFunctor& op) : m_functor(op) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return m_functor(a, b);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const DupFunctor& functor() const { return m_functor; }
  const DupFunctor& m_functor;
};

template <typename DupFunctor, typename LhsScalar, typename RhsScalar>
struct functor_traits<scalar_disjunction_op<DupFunctor, LhsScalar, RhsScalar>> : public functor_traits<DupFunctor> {};

// Creates a compressed sparse matrix from its existing entries and those from an unsorted range of triplets
template <typename InputIterator, typename SparseMatrixType, typename DupFunctor>
void insert_from_triplets(const InputIterator& begin, const InputIterator& end, SparseMatrixType& mat,
                          DupFunctor dup_func) {
  using Scalar = typename SparseMatrixType::Scalar;
  using SrcXprType =
      CwiseBinaryOp<scalar_disjunction_op<DupFunctor, Scalar>, const SparseMatrixType, const SparseMatrixType>;

  // set_from_triplets is necessary to sort the inner indices and remove the duplicate entries
  SparseMatrixType trips(mat.rows(), mat.cols());
  set_from_triplets(begin, end, trips, dup_func);

  SrcXprType src = mat.binaryExpr(trips, scalar_disjunction_op<DupFunctor, Scalar>(dup_func));
  // the sparse assignment procedure creates a temporary matrix and swaps the final result
  assign_sparse_to_sparse<SparseMatrixType, SrcXprType>(mat, src);
}

// Creates a compressed sparse matrix from its existing entries and those from an sorted range of triplets
template <typename InputIterator, typename SparseMatrixType, typename DupFunctor>
void insert_from_triplets_sorted(const InputIterator& begin, const InputIterator& end, SparseMatrixType& mat,
                                 DupFunctor dup_func) {
  using Scalar = typename SparseMatrixType::Scalar;
  using SrcXprType =
      CwiseBinaryOp<scalar_disjunction_op<DupFunctor, Scalar>, const SparseMatrixType, const SparseMatrixType>;

  // TODO: process triplets without making a copy
  SparseMatrixType trips(mat.rows(), mat.cols());
  set_from_triplets_sorted(begin, end, trips, dup_func);

  SrcXprType src = mat.binaryExpr(trips, scalar_disjunction_op<DupFunctor, Scalar>(dup_func));
  // the sparse assignment procedure creates a temporary matrix and swaps the final result
  assign_sparse_to_sparse<SparseMatrixType, SrcXprType>(mat, src);
}

}  // namespace internal

/** Fill the matrix \c *this with the list of \em triplets defined in the half-open range from \a begin to \a end.
  *
  * A \em triplet is a tuple (i,j,value) defining a non-zero element.
  * The input list of triplets does not have to be sorted, and may contain duplicated elements.
  * In any case, the result is a \b sorted and \b compressed sparse matrix where the duplicates have been summed up.
  * This is a \em O(n) operation, with \em n the number of triplet elements.
  * The initial contents of \c *this are destroyed.
  * The matrix \c *this must be properly resized beforehand using the SparseMatrix(Index,Index) constructor,
  * or the resize(Index,Index) method. The sizes are not extracted from the triplet list.
  *
  * The \a InputIterators value_type must provide the following interface:
  * \code
  * Scalar value() const; // the value
  * IndexType row() const;   // the row index i
  * IndexType col() const;   // the column index j
  * \endcode
  * See for instance the Eigen::Triplet template class.
  *
  * Here is a typical usage example:
  * \code
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(estimation_of_entries);
    for(...)
    {
      // ...
      tripletList.push_back(T(i,j,v_ij));
    }
    SparseMatrixType m(rows,cols);
    m.setFromTriplets(tripletList.begin(), tripletList.end());
    // m is ready to go!
  * \endcode
  *
  * \warning The list of triplets is read multiple times (at least twice). Therefore, it is not recommended to define
  * an abstract iterator over a complex data-structure that would be expensive to evaluate. The triplets should rather
  * be explicitly stored into a std::vector for instance.
  */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename InputIterators>
void SparseMatrix<Scalar, Options_, StorageIndex_>::setFromTriplets(const InputIterators& begin,
                                                                    const InputIterators& end) {
  internal::set_from_triplets<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>>(
      begin, end, *this, internal::scalar_sum_op<Scalar, Scalar>());
}

/** The same as setFromTriplets but when duplicates are met the functor \a dup_func is applied:
 * \code
 * value = dup_func(OldValue, NewValue)
 * \endcode
 * Here is a C++11 example keeping the latest entry only:
 * \code
 * mat.setFromTriplets(triplets.begin(), triplets.end(), [] (const Scalar&,const Scalar &b) { return b; });
 * \endcode
 */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename InputIterators, typename DupFunctor>
void SparseMatrix<Scalar, Options_, StorageIndex_>::setFromTriplets(const InputIterators& begin,
                                                                    const InputIterators& end, DupFunctor dup_func) {
  internal::set_from_triplets<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>, DupFunctor>(
      begin, end, *this, dup_func);
}

/** The same as setFromTriplets but triplets are assumed to be pre-sorted. This is faster and requires less temporary
 * storage. Two triplets `a` and `b` are appropriately ordered if: \code ColMajor: ((a.col() != b.col()) ? (a.col() <
 * b.col()) : (a.row() < b.row()) RowMajor: ((a.row() != b.row()) ? (a.row() < b.row()) : (a.col() < b.col()) \endcode
 */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename InputIterators>
void SparseMatrix<Scalar, Options_, StorageIndex_>::setFromSortedTriplets(const InputIterators& begin,
                                                                          const InputIterators& end) {
  internal::set_from_triplets_sorted<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>>(
      begin, end, *this, internal::scalar_sum_op<Scalar, Scalar>());
}

/** The same as setFromSortedTriplets but when duplicates are met the functor \a dup_func is applied:
 * \code
 * value = dup_func(OldValue, NewValue)
 * \endcode
 * Here is a C++11 example keeping the latest entry only:
 * \code
 * mat.setFromSortedTriplets(triplets.begin(), triplets.end(), [] (const Scalar&,const Scalar &b) { return b; });
 * \endcode
 */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename InputIterators, typename DupFunctor>
void SparseMatrix<Scalar, Options_, StorageIndex_>::setFromSortedTriplets(const InputIterators& begin,
                                                                          const InputIterators& end,
                                                                          DupFunctor dup_func) {
  internal::set_from_triplets_sorted<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>, DupFunctor>(
      begin, end, *this, dup_func);
}

/** Insert a batch of elements into the matrix \c *this with the list of \em triplets defined in the half-open range
  from \a begin to \a end.
  *
  * A \em triplet is a tuple (i,j,value) defining a non-zero element.
  * The input list of triplets does not have to be sorted, and may contain duplicated elements.
  * In any case, the result is a \b sorted and \b compressed sparse matrix where the duplicates have been summed up.
  * This is a \em O(n) operation, with \em n the number of triplet elements.
  * The initial contents of \c *this are preserved (except for the summation of duplicate elements).
  * The matrix \c *this must be properly sized beforehand. The sizes are not extracted from the triplet list.
  *
  * The \a InputIterators value_type must provide the following interface:
  * \code
  * Scalar value() const; // the value
  * IndexType row() const;   // the row index i
  * IndexType col() const;   // the column index j
  * \endcode
  * See for instance the Eigen::Triplet template class.
  *
  * Here is a typical usage example:
  * \code
    SparseMatrixType m(rows,cols); // m contains nonzero entries
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(estimation_of_entries);
    for(...)
    {
      // ...
      tripletList.push_back(T(i,j,v_ij));
    }

    m.insertFromTriplets(tripletList.begin(), tripletList.end());
    // m is ready to go!
  * \endcode
  *
  * \warning The list of triplets is read multiple times (at least twice). Therefore, it is not recommended to define
  * an abstract iterator over a complex data-structure that would be expensive to evaluate. The triplets should rather
  * be explicitly stored into a std::vector for instance.
  */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename InputIterators>
void SparseMatrix<Scalar, Options_, StorageIndex_>::insertFromTriplets(const InputIterators& begin,
                                                                       const InputIterators& end) {
  internal::insert_from_triplets<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>>(
      begin, end, *this, internal::scalar_sum_op<Scalar, Scalar>());
}

/** The same as insertFromTriplets but when duplicates are met the functor \a dup_func is applied:
 * \code
 * value = dup_func(OldValue, NewValue)
 * \endcode
 * Here is a C++11 example keeping the latest entry only:
 * \code
 * mat.insertFromTriplets(triplets.begin(), triplets.end(), [] (const Scalar&,const Scalar &b) { return b; });
 * \endcode
 */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename InputIterators, typename DupFunctor>
void SparseMatrix<Scalar, Options_, StorageIndex_>::insertFromTriplets(const InputIterators& begin,
                                                                       const InputIterators& end, DupFunctor dup_func) {
  internal::insert_from_triplets<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>, DupFunctor>(
      begin, end, *this, dup_func);
}

/** The same as insertFromTriplets but triplets are assumed to be pre-sorted. This is faster and requires less temporary
 * storage. Two triplets `a` and `b` are appropriately ordered if: \code ColMajor: ((a.col() != b.col()) ? (a.col() <
 * b.col()) : (a.row() < b.row()) RowMajor: ((a.row() != b.row()) ? (a.row() < b.row()) : (a.col() < b.col()) \endcode
 */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename InputIterators>
void SparseMatrix<Scalar, Options_, StorageIndex_>::insertFromSortedTriplets(const InputIterators& begin,
                                                                             const InputIterators& end) {
  internal::insert_from_triplets_sorted<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>>(
      begin, end, *this, internal::scalar_sum_op<Scalar, Scalar>());
}

/** The same as insertFromSortedTriplets but when duplicates are met the functor \a dup_func is applied:
 * \code
 * value = dup_func(OldValue, NewValue)
 * \endcode
 * Here is a C++11 example keeping the latest entry only:
 * \code
 * mat.insertFromSortedTriplets(triplets.begin(), triplets.end(), [] (const Scalar&,const Scalar &b) { return b; });
 * \endcode
 */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename InputIterators, typename DupFunctor>
void SparseMatrix<Scalar, Options_, StorageIndex_>::insertFromSortedTriplets(const InputIterators& begin,
                                                                             const InputIterators& end,
                                                                             DupFunctor dup_func) {
  internal::insert_from_triplets_sorted<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>, DupFunctor>(
      begin, end, *this, dup_func);
}

/** \internal */
template <typename Scalar_, int Options_, typename StorageIndex_>
template <typename Derived, typename DupFunctor>
void SparseMatrix<Scalar_, Options_, StorageIndex_>::collapseDuplicates(DenseBase<Derived>& wi, DupFunctor dup_func) {
  // removes duplicate entries and compresses the matrix
  // the excess allocated memory is not released
  // the inner indices do not need to be sorted, nor is the matrix returned in a sorted state
  eigen_assert(wi.size() == m_innerSize);
  constexpr StorageIndex kEmptyIndexValue(-1);
  wi.setConstant(kEmptyIndexValue);
  StorageIndex count = 0;
  const bool is_compressed = isCompressed();
  // for each inner-vector, wi[inner_index] will hold the position of first element into the index/value buffers
  for (Index j = 0; j < m_outerSize; ++j) {
    const StorageIndex newBegin = count;
    const StorageIndex end = is_compressed ? m_outerIndex[j + 1] : m_outerIndex[j] + m_innerNonZeros[j];
    for (StorageIndex k = m_outerIndex[j]; k < end; ++k) {
      StorageIndex i = m_data.index(k);
      if (wi(i) >= newBegin) {
        // entry at k is a duplicate
        // accumulate it into the primary entry located at wi(i)
        m_data.value(wi(i)) = dup_func(m_data.value(wi(i)), m_data.value(k));
      } else {
        // k is the primary entry in j with inner index i
        // shift it to the left and record its location at wi(i)
        m_data.index(count) = i;
        m_data.value(count) = m_data.value(k);
        wi(i) = count;
        ++count;
      }
    }
    m_outerIndex[j] = newBegin;
  }
  m_outerIndex[m_outerSize] = count;
  m_data.resize(count);

  // turn the matrix into compressed form (if it is not already)
  internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
  m_innerNonZeros = 0;
}

/** \internal */
template <typename Scalar, int Options_, typename StorageIndex_>
template <typename OtherDerived>
EIGEN_DONT_INLINE SparseMatrix<Scalar, Options_, StorageIndex_>&
SparseMatrix<Scalar, Options_, StorageIndex_>::operator=(const SparseMatrixBase<OtherDerived>& other) {
  EIGEN_STATIC_ASSERT(
      (internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
      YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

#ifdef EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
  EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
#endif

  const bool needToTranspose = (Flags & RowMajorBit) != (internal::evaluator<OtherDerived>::Flags & RowMajorBit);
  if (needToTranspose) {
#ifdef EIGEN_SPARSE_TRANSPOSED_COPY_PLUGIN
    EIGEN_SPARSE_TRANSPOSED_COPY_PLUGIN
#endif
    // two passes algorithm:
    //  1 - compute the number of coeffs per dest inner vector
    //  2 - do the actual copy/eval
    // Since each coeff of the rhs has to be evaluated twice, let's evaluate it if needed
    typedef
        typename internal::nested_eval<OtherDerived, 2, typename internal::plain_matrix_type<OtherDerived>::type>::type
            OtherCopy;
    typedef internal::remove_all_t<OtherCopy> OtherCopy_;
    typedef internal::evaluator<OtherCopy_> OtherCopyEval;
    OtherCopy otherCopy(other.derived());
    OtherCopyEval otherCopyEval(otherCopy);

    SparseMatrix dest(other.rows(), other.cols());
    Eigen::Map<IndexVector>(dest.m_outerIndex, dest.outerSize()).setZero();

    // pass 1
    // FIXME the above copy could be merged with that pass
    for (Index j = 0; j < otherCopy.outerSize(); ++j)
      for (typename OtherCopyEval::InnerIterator it(otherCopyEval, j); it; ++it) ++dest.m_outerIndex[it.index()];

    // prefix sum
    StorageIndex count = 0;
    IndexVector positions(dest.outerSize());
    for (Index j = 0; j < dest.outerSize(); ++j) {
      StorageIndex tmp = dest.m_outerIndex[j];
      dest.m_outerIndex[j] = count;
      positions[j] = count;
      count += tmp;
    }
    dest.m_outerIndex[dest.outerSize()] = count;
    // alloc
    dest.m_data.resize(count);
    // pass 2
    for (StorageIndex j = 0; j < otherCopy.outerSize(); ++j) {
      for (typename OtherCopyEval::InnerIterator it(otherCopyEval, j); it; ++it) {
        Index pos = positions[it.index()]++;
        dest.m_data.index(pos) = j;
        dest.m_data.value(pos) = it.value();
      }
    }
    this->swap(dest);
    return *this;
  } else {
    if (other.isRValue()) {
      initAssignment(other.derived());
    }
    // there is no special optimization
    return Base::operator=(other.derived());
  }
}

template <typename Scalar_, int Options_, typename StorageIndex_>
inline typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insert(Index row, Index col) {
  return insertByOuterInner(IsRowMajor ? row : col, IsRowMajor ? col : row);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
EIGEN_STRONG_INLINE typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertAtByOuterInner(Index outer, Index inner, Index dst) {
  // random insertion into compressed matrix is very slow
  uncompress();
  return insertUncompressedAtByOuterInner(outer, inner, dst);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
EIGEN_DEPRECATED EIGEN_DONT_INLINE typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertUncompressed(Index row, Index col) {
  eigen_assert(!isCompressed());
  Index outer = IsRowMajor ? row : col;
  Index inner = IsRowMajor ? col : row;
  Index start = m_outerIndex[outer];
  Index end = start + m_innerNonZeros[outer];
  Index dst = start == end ? end : m_data.searchLowerIndex(start, end, inner);
  if (dst == end) {
    Index capacity = m_outerIndex[outer + 1] - end;
    if (capacity > 0) {
      // implies uncompressed: push to back of vector
      m_innerNonZeros[outer]++;
      m_data.index(end) = StorageIndex(inner);
      m_data.value(end) = Scalar(0);
      return m_data.value(end);
    }
  }
  eigen_assert((dst == end || m_data.index(dst) != inner) &&
               "you cannot insert an element that already exists, you must call coeffRef to this end");
  return insertUncompressedAtByOuterInner(outer, inner, dst);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
EIGEN_DEPRECATED EIGEN_DONT_INLINE typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertCompressed(Index row, Index col) {
  eigen_assert(isCompressed());
  Index outer = IsRowMajor ? row : col;
  Index inner = IsRowMajor ? col : row;
  Index start = m_outerIndex[outer];
  Index end = m_outerIndex[outer + 1];
  Index dst = start == end ? end : m_data.searchLowerIndex(start, end, inner);
  eigen_assert((dst == end || m_data.index(dst) != inner) &&
               "you cannot insert an element that already exists, you must call coeffRef to this end");
  return insertCompressedAtByOuterInner(outer, inner, dst);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertCompressedAtByOuterInner(Index outer, Index inner, Index dst) {
  eigen_assert(isCompressed());
  // compressed insertion always requires expanding the buffer
  // first, check if there is adequate allocated memory
  if (m_data.allocatedSize() <= m_data.size()) {
    // if there is no capacity for a single insertion, double the capacity
    // increase capacity by a mininum of 32
    Index minReserve = 32;
    Index reserveSize = numext::maxi(minReserve, m_data.allocatedSize());
    m_data.reserve(reserveSize);
  }
  m_data.resize(m_data.size() + 1);
  Index chunkSize = m_outerIndex[m_outerSize] - dst;
  // shift the existing data to the right if necessary
  m_data.moveChunk(dst, dst + 1, chunkSize);
  // update nonzero counts
  // potentially O(outerSize) bottleneck!
  for (Index j = outer; j < m_outerSize; j++) m_outerIndex[j + 1]++;
  // initialize the coefficient
  m_data.index(dst) = StorageIndex(inner);
  m_data.value(dst) = Scalar(0);
  // return a reference to the coefficient
  return m_data.value(dst);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertUncompressedAtByOuterInner(Index outer, Index inner, Index dst) {
  eigen_assert(!isCompressed());
  // find a vector with capacity, starting at `outer` and searching to the left and right
  for (Index leftTarget = outer - 1, rightTarget = outer; (leftTarget >= 0) || (rightTarget < m_outerSize);) {
    if (rightTarget < m_outerSize) {
      Index start = m_outerIndex[rightTarget];
      Index end = start + m_innerNonZeros[rightTarget];
      Index nextStart = m_outerIndex[rightTarget + 1];
      Index capacity = nextStart - end;
      if (capacity > 0) {
        // move [dst, end) to dst+1 and insert at dst
        Index chunkSize = end - dst;
        if (chunkSize > 0) m_data.moveChunk(dst, dst + 1, chunkSize);
        m_innerNonZeros[outer]++;
        for (Index j = outer; j < rightTarget; j++) m_outerIndex[j + 1]++;
        m_data.index(dst) = StorageIndex(inner);
        m_data.value(dst) = Scalar(0);
        return m_data.value(dst);
      }
      rightTarget++;
    }
    if (leftTarget >= 0) {
      Index start = m_outerIndex[leftTarget];
      Index end = start + m_innerNonZeros[leftTarget];
      Index nextStart = m_outerIndex[leftTarget + 1];
      Index capacity = nextStart - end;
      if (capacity > 0) {
        // tricky: dst is a lower bound, so we must insert at dst-1 when shifting left
        // move [nextStart, dst) to nextStart-1 and insert at dst-1
        Index chunkSize = dst - nextStart;
        if (chunkSize > 0) m_data.moveChunk(nextStart, nextStart - 1, chunkSize);
        m_innerNonZeros[outer]++;
        for (Index j = leftTarget; j < outer; j++) m_outerIndex[j + 1]--;
        m_data.index(dst - 1) = StorageIndex(inner);
        m_data.value(dst - 1) = Scalar(0);
        return m_data.value(dst - 1);
      }
      leftTarget--;
    }
  }

  // no room for interior insertion
  // nonZeros() == m_data.size()
  // record offset as outerIndxPtr will change
  Index dst_offset = dst - m_outerIndex[outer];
  // allocate space for random insertion
  if (m_data.allocatedSize() == 0) {
    // fast method to allocate space for one element per vector in empty matrix
    m_data.resize(m_outerSize);
    std::iota(m_outerIndex, m_outerIndex + m_outerSize + 1, StorageIndex(0));
  } else {
    // check for integer overflow: if maxReserveSize == 0, insertion is not possible
    Index maxReserveSize = static_cast<Index>(NumTraits<StorageIndex>::highest()) - m_data.allocatedSize();
    eigen_assert(maxReserveSize > 0);
    if (m_outerSize <= maxReserveSize) {
      // allocate space for one additional element per vector
      reserveInnerVectors(IndexVector::Constant(m_outerSize, 1));
    } else {
      // handle the edge case where StorageIndex is insufficient to reserve outerSize additional elements
      // allocate space for one additional element in the interval [outer,maxReserveSize)
      typedef internal::sparse_reserve_op<StorageIndex> ReserveSizesOp;
      typedef CwiseNullaryOp<ReserveSizesOp, IndexVector> ReserveSizesXpr;
      ReserveSizesXpr reserveSizesXpr(m_outerSize, 1, ReserveSizesOp(outer, m_outerSize, maxReserveSize));
      reserveInnerVectors(reserveSizesXpr);
    }
  }
  // insert element at `dst` with new outer indices
  Index start = m_outerIndex[outer];
  Index end = start + m_innerNonZeros[outer];
  Index new_dst = start + dst_offset;
  Index chunkSize = end - new_dst;
  if (chunkSize > 0) m_data.moveChunk(new_dst, new_dst + 1, chunkSize);
  m_innerNonZeros[outer]++;
  m_data.index(new_dst) = StorageIndex(inner);
  m_data.value(new_dst) = Scalar(0);
  return m_data.value(new_dst);
}

namespace internal {

template <typename Scalar_, int Options_, typename StorageIndex_>
struct evaluator<SparseMatrix<Scalar_, Options_, StorageIndex_>>
    : evaluator<SparseCompressedBase<SparseMatrix<Scalar_, Options_, StorageIndex_>>> {
  typedef evaluator<SparseCompressedBase<SparseMatrix<Scalar_, Options_, StorageIndex_>>> Base;
  typedef SparseMatrix<Scalar_, Options_, StorageIndex_> SparseMatrixType;
  evaluator() : Base() {}
  explicit evaluator(const SparseMatrixType& mat) : Base(mat) {}
};

}  // namespace internal

// Specialization for SparseMatrix.
// Serializes [rows, cols, isCompressed, outerSize, innerBufferSize,
// innerNonZeros, outerIndices, innerIndices, values].
template <typename Scalar, int Options, typename StorageIndex>
class Serializer<SparseMatrix<Scalar, Options, StorageIndex>, void> {
 public:
  typedef SparseMatrix<Scalar, Options, StorageIndex> SparseMat;

  struct Header {
    typename SparseMat::Index rows;
    typename SparseMat::Index cols;
    bool compressed;
    Index outer_size;
    Index inner_buffer_size;
  };

  EIGEN_DEVICE_FUNC size_t size(const SparseMat& value) const {
    // innerNonZeros.
    std::size_t num_storage_indices = value.isCompressed() ? 0 : value.outerSize();
    // Outer indices.
    num_storage_indices += value.outerSize() + 1;
    // Inner indices.
    const StorageIndex inner_buffer_size = value.outerIndexPtr()[value.outerSize()];
    num_storage_indices += inner_buffer_size;
    // Values.
    std::size_t num_values = inner_buffer_size;
    return sizeof(Header) + sizeof(Scalar) * num_values + sizeof(StorageIndex) * num_storage_indices;
  }

  EIGEN_DEVICE_FUNC uint8_t* serialize(uint8_t* dest, uint8_t* end, const SparseMat& value) {
    if (EIGEN_PREDICT_FALSE(dest == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(dest + size(value) > end)) return nullptr;

    const size_t header_bytes = sizeof(Header);
    Header header = {value.rows(), value.cols(), value.isCompressed(), value.outerSize(),
                     value.outerIndexPtr()[value.outerSize()]};
    EIGEN_USING_STD(memcpy)
    memcpy(dest, &header, header_bytes);
    dest += header_bytes;

    // innerNonZeros.
    if (!header.compressed) {
      std::size_t data_bytes = sizeof(StorageIndex) * header.outer_size;
      memcpy(dest, value.innerNonZeroPtr(), data_bytes);
      dest += data_bytes;
    }

    // Outer indices.
    std::size_t data_bytes = sizeof(StorageIndex) * (header.outer_size + 1);
    memcpy(dest, value.outerIndexPtr(), data_bytes);
    dest += data_bytes;

    // Inner indices.
    data_bytes = sizeof(StorageIndex) * header.inner_buffer_size;
    memcpy(dest, value.innerIndexPtr(), data_bytes);
    dest += data_bytes;

    // Values.
    data_bytes = sizeof(Scalar) * header.inner_buffer_size;
    memcpy(dest, value.valuePtr(), data_bytes);
    dest += data_bytes;

    return dest;
  }

  EIGEN_DEVICE_FUNC const uint8_t* deserialize(const uint8_t* src, const uint8_t* end, SparseMat& value) const {
    if (EIGEN_PREDICT_FALSE(src == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(src + sizeof(Header) > end)) return nullptr;

    const size_t header_bytes = sizeof(Header);
    Header header;
    EIGEN_USING_STD(memcpy)
    memcpy(&header, src, header_bytes);
    src += header_bytes;

    value.setZero();
    value.resize(header.rows, header.cols);
    if (header.compressed) {
      value.makeCompressed();
    } else {
      value.uncompress();
    }

    // Adjust value ptr size.
    value.data().resize(header.inner_buffer_size);

    // Initialize compressed state and inner non-zeros.
    if (!header.compressed) {
      // Inner non-zero counts.
      std::size_t data_bytes = sizeof(StorageIndex) * header.outer_size;
      if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
      memcpy(value.innerNonZeroPtr(), src, data_bytes);
      src += data_bytes;
    }

    // Outer indices.
    std::size_t data_bytes = sizeof(StorageIndex) * (header.outer_size + 1);
    if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
    memcpy(value.outerIndexPtr(), src, data_bytes);
    src += data_bytes;

    // Inner indices.
    data_bytes = sizeof(StorageIndex) * header.inner_buffer_size;
    if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
    memcpy(value.innerIndexPtr(), src, data_bytes);
    src += data_bytes;

    // Values.
    data_bytes = sizeof(Scalar) * header.inner_buffer_size;
    if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
    memcpy(value.valuePtr(), src, data_bytes);
    src += data_bytes;
    return src;
  }
};

}  // end namespace Eigen

#endif  // EIGEN_SPARSEMATRIX_H
