// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if !defined(EIGEN_PARSED_BY_DOXYGEN)

public:
// SFINAE dummy types

template <typename RowIndices, typename ColIndices>
using EnableOverload = std::enable_if_t<
    internal::valid_indexed_view_overload<RowIndices, ColIndices>::value && internal::is_lvalue<Derived>::value, bool>;

template <typename RowIndices, typename ColIndices>
using EnableConstOverload =
    std::enable_if_t<internal::valid_indexed_view_overload<RowIndices, ColIndices>::value, bool>;

template <typename Indices>
using EnableVectorOverload =
    std::enable_if_t<!internal::is_valid_index_type<Indices>::value && internal::is_lvalue<Derived>::value, bool>;

template <typename Indices>
using EnableConstVectorOverload = std::enable_if_t<!internal::is_valid_index_type<Indices>::value, bool>;

public:
// Public API for 2D matrices/arrays

// non-const versions

 template <typename RowIndices, typename ColIndices>
 using IndexedViewType = typename internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::ReturnType;

 template <typename RowIndices, typename ColIndices, EnableOverload<RowIndices, ColIndices> = true>
 IndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices, const ColIndices& colIndices) {
   return internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::run(derived(), rowIndices, colIndices);
 }

template <typename RowType, size_t RowSize, typename ColIndices, typename RowIndices = Array<RowType, RowSize, 1>,
          EnableOverload<RowIndices, ColIndices> = true>
IndexedViewType<RowIndices, ColIndices> operator()(const RowType (&rowIndices)[RowSize], const ColIndices& colIndices) {
  return internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::run(derived(), RowIndices{rowIndices},
                                                                             colIndices);
}

template <typename RowIndices, typename ColType, size_t ColSize, typename ColIndices = Array<ColType, ColSize, 1>,
          EnableOverload<RowIndices, ColIndices> = true>
IndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices, const ColType (&colIndices)[ColSize]) {
  return internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::run(derived(), rowIndices,
                                                                             ColIndices{colIndices});
}

template <typename RowType, size_t RowSize, typename ColType, size_t ColSize,
          typename RowIndices = Array<RowType, RowSize, 1>, typename ColIndices = Array<ColType, ColSize, 1>,
          EnableOverload<RowIndices, ColIndices> = true>
IndexedViewType<RowIndices, ColIndices> operator()(const RowType (&rowIndices)[RowSize],
                                                   const ColType (&colIndices)[ColSize]) {
  return internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::run(derived(), RowIndices{rowIndices},
                                                                             ColIndices{colIndices});
}

// const versions

template <typename RowIndices, typename ColIndices>
using ConstIndexedViewType = typename internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::ConstReturnType;

template <typename RowIndices, typename ColIndices, EnableConstOverload<RowIndices, ColIndices> = true>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices,
                                                        const ColIndices& colIndices) const {
  return internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::run(derived(), rowIndices, colIndices);
}

template <typename RowType, size_t RowSize, typename ColIndices, typename RowIndices = Array<RowType, RowSize, 1>,
          EnableConstOverload<RowIndices, ColIndices> = true>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowType (&rowIndices)[RowSize],
                                                        const ColIndices& colIndices) const {
  return internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::run(derived(), RowIndices{rowIndices},
                                                                             colIndices);
}

template <typename RowIndices, typename ColType, size_t ColSize, typename ColIndices = Array<ColType, ColSize, 1>,
          EnableConstOverload<RowIndices, ColIndices> = true>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices,
                                                        const ColType (&colIndices)[ColSize]) const {
  return internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::run(derived(), rowIndices,
                                                                             ColIndices{colIndices});
}

template <typename RowType, size_t RowSize, typename ColType, size_t ColSize,
          typename RowIndices = Array<RowType, RowSize, 1>, typename ColIndices = Array<ColType, ColSize, 1>,
          EnableConstOverload<RowIndices, ColIndices> = true>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowType (&rowIndices)[RowSize],
                                                        const ColType (&colIndices)[ColSize]) const {
  return internal::IndexedViewSelector<Derived, RowIndices, ColIndices>::run(derived(), RowIndices{rowIndices},
                                                                             ColIndices{colIndices});
}

// Public API for 1D vectors/arrays

// non-const versions

template <typename Indices>
using VectorIndexedViewType = typename internal::VectorIndexedViewSelector<Derived, Indices>::ReturnType;

template <typename Indices, EnableVectorOverload<Indices> = true>
VectorIndexedViewType<Indices> operator()(const Indices& indices) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return internal::VectorIndexedViewSelector<Derived, Indices>::run(derived(), indices);
}

template <typename IndexType, size_t Size, typename Indices = Array<IndexType, Size, 1>,
          EnableVectorOverload<Indices> = true>
VectorIndexedViewType<Indices> operator()(const IndexType (&indices)[Size]) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return internal::VectorIndexedViewSelector<Derived, Indices>::run(derived(), Indices{indices});
}

// const versions

template <typename Indices>
using ConstVectorIndexedViewType = typename internal::VectorIndexedViewSelector<Derived, Indices>::ConstReturnType;

template <typename Indices, EnableConstVectorOverload<Indices> = true>
ConstVectorIndexedViewType<Indices> operator()(const Indices& indices) const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return internal::VectorIndexedViewSelector<Derived, Indices>::run(derived(), indices);
}

template <typename IndexType, size_t Size, typename Indices = Array<IndexType, Size, 1>,
          EnableConstVectorOverload<Indices> = true>
ConstVectorIndexedViewType<Indices> operator()(const IndexType (&indices)[Size]) const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return internal::VectorIndexedViewSelector<Derived, Indices>::run(derived(), Indices{indices});
}

#else  // EIGEN_PARSED_BY_DOXYGEN

/**
 * \returns a generic submatrix view defined by the rows and columns indexed \a rowIndices and \a colIndices
 * respectively.
 *
 * Each parameter must either be:
 *  - An integer indexing a single row or column
 *  - Eigen::placeholders::all indexing the full set of respective rows or columns in increasing order
 *  - An ArithmeticSequence as returned by the Eigen::seq and Eigen::seqN functions
 *  - Any %Eigen's vector/array of integers or expressions
 *  - Plain C arrays: \c int[N]
 *  - And more generally any type exposing the following two member functions:
 * \code
 * <integral type> operator[](<integral type>) const;
 * <integral type> size() const;
 * \endcode
 * where \c <integral \c type>  stands for any integer type compatible with Eigen::Index (i.e. \c std::ptrdiff_t).
 *
 * The last statement implies compatibility with \c std::vector, \c std::valarray, \c std::array, many of the Range-v3's
 * ranges, etc.
 *
 * If the submatrix can be represented using a starting position \c (i,j) and positive sizes \c (rows,columns), then
 * this method will returns a Block object after extraction of the relevant information from the passed arguments. This
 * is the case when all arguments are either:
 *  - An integer
 *  - Eigen::placeholders::all
 *  - An ArithmeticSequence with compile-time increment strictly equal to 1, as returned by Eigen::seq(a,b), and
 * Eigen::seqN(a,N).
 *
 * Otherwise a more general IndexedView<Derived,RowIndices',ColIndices'> object will be returned, after conversion of
 * the inputs to more suitable types \c RowIndices' and \c ColIndices'.
 *
 * For 1D vectors and arrays, you better use the operator()(const Indices&) overload, which behave the same way but
 * taking a single parameter.
 *
 * See also this <a
 * href="https://stackoverflow.com/questions/46110917/eigen-replicate-items-along-one-dimension-without-useless-allocations">question</a>
 * and its answer for an example of how to duplicate coefficients.
 *
 * \sa operator()(const Indices&), class Block, class IndexedView, DenseBase::block(Index,Index,Index,Index)
 */
template <typename RowIndices, typename ColIndices>
IndexedView_or_Block operator()(const RowIndices& rowIndices, const ColIndices& colIndices);

/** This is an overload of operator()(const RowIndices&, const ColIndices&) for 1D vectors or arrays
 *
 * \only_for_vectors
 */
template <typename Indices>
IndexedView_or_VectorBlock operator()(const Indices& indices);

#endif  // EIGEN_PARSED_BY_DOXYGEN
