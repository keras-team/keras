// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_STATIC_ASSERT_H
#define EIGEN_STATIC_ASSERT_H

/* Some notes on Eigen's static assertion mechanism:
 *
 *  - in EIGEN_STATIC_ASSERT(CONDITION,MSG) the parameter CONDITION must be a compile time boolean
 *    expression, and MSG an enum listed in struct internal::static_assertion<true>
 *
 *  - currently EIGEN_STATIC_ASSERT can only be used in function scope
 *
 */

#ifndef EIGEN_STATIC_ASSERT
#ifndef EIGEN_NO_STATIC_ASSERT

#define EIGEN_STATIC_ASSERT(X, MSG) static_assert(X, #MSG);

#else  // EIGEN_NO_STATIC_ASSERT

#define EIGEN_STATIC_ASSERT(CONDITION, MSG)

#endif  // EIGEN_NO_STATIC_ASSERT
#endif  // EIGEN_STATIC_ASSERT

// static assertion failing if the type \a TYPE is not a vector type
#define EIGEN_STATIC_ASSERT_VECTOR_ONLY(TYPE) \
  EIGEN_STATIC_ASSERT(TYPE::IsVectorAtCompileTime, YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX)

// static assertion failing if the type \a TYPE is not fixed-size
#define EIGEN_STATIC_ASSERT_FIXED_SIZE(TYPE)                     \
  EIGEN_STATIC_ASSERT(TYPE::SizeAtCompileTime != Eigen::Dynamic, \
                      YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_MATRIX_OR_VECTOR)

// static assertion failing if the type \a TYPE is not dynamic-size
#define EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(TYPE)                   \
  EIGEN_STATIC_ASSERT(TYPE::SizeAtCompileTime == Eigen::Dynamic, \
                      YOU_CALLED_A_DYNAMIC_SIZE_METHOD_ON_A_FIXED_SIZE_MATRIX_OR_VECTOR)

// static assertion failing if the type \a TYPE is not a vector type of the given size
#define EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TYPE, SIZE)                         \
  EIGEN_STATIC_ASSERT(TYPE::IsVectorAtCompileTime&& TYPE::SizeAtCompileTime == SIZE, \
                      THIS_METHOD_IS_ONLY_FOR_VECTORS_OF_A_SPECIFIC_SIZE)

// static assertion failing if the type \a TYPE is not a vector type of the given size
#define EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(TYPE, ROWS, COLS)                        \
  EIGEN_STATIC_ASSERT(TYPE::RowsAtCompileTime == ROWS && TYPE::ColsAtCompileTime == COLS, \
                      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE)

// static assertion failing if the two vector expression types are not compatible (same fixed-size or dynamic size)
#define EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(TYPE0, TYPE1)                                                   \
  EIGEN_STATIC_ASSERT(                                                                                       \
      (int(TYPE0::SizeAtCompileTime) == Eigen::Dynamic || int(TYPE1::SizeAtCompileTime) == Eigen::Dynamic || \
       int(TYPE0::SizeAtCompileTime) == int(TYPE1::SizeAtCompileTime)),                                      \
      YOU_MIXED_VECTORS_OF_DIFFERENT_SIZES)

#define EIGEN_PREDICATE_SAME_MATRIX_SIZE(TYPE0, TYPE1)                                                     \
  ((int(Eigen::internal::size_of_xpr_at_compile_time<TYPE0>::ret) == 0 &&                                  \
    int(Eigen::internal::size_of_xpr_at_compile_time<TYPE1>::ret) == 0) ||                                 \
   ((int(TYPE0::RowsAtCompileTime) == Eigen::Dynamic || int(TYPE1::RowsAtCompileTime) == Eigen::Dynamic || \
     int(TYPE0::RowsAtCompileTime) == int(TYPE1::RowsAtCompileTime)) &&                                    \
    (int(TYPE0::ColsAtCompileTime) == Eigen::Dynamic || int(TYPE1::ColsAtCompileTime) == Eigen::Dynamic || \
     int(TYPE0::ColsAtCompileTime) == int(TYPE1::ColsAtCompileTime))))

#define EIGEN_STATIC_ASSERT_NON_INTEGER(TYPE) \
  EIGEN_STATIC_ASSERT(!Eigen::NumTraits<TYPE>::IsInteger, THIS_FUNCTION_IS_NOT_FOR_INTEGER_NUMERIC_TYPES)

// static assertion failing if it is guaranteed at compile-time that the two matrix expression types have different
// sizes
#define EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(TYPE0, TYPE1) \
  EIGEN_STATIC_ASSERT(EIGEN_PREDICATE_SAME_MATRIX_SIZE(TYPE0, TYPE1), YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)

#define EIGEN_STATIC_ASSERT_SIZE_1x1(TYPE)                                                             \
  EIGEN_STATIC_ASSERT((TYPE::RowsAtCompileTime == 1 || TYPE::RowsAtCompileTime == Eigen::Dynamic) &&   \
                          (TYPE::ColsAtCompileTime == 1 || TYPE::ColsAtCompileTime == Eigen::Dynamic), \
                      THIS_METHOD_IS_ONLY_FOR_1x1_EXPRESSIONS)

#define EIGEN_STATIC_ASSERT_LVALUE(Derived) \
  EIGEN_STATIC_ASSERT(Eigen::internal::is_lvalue<Derived>::value, THIS_EXPRESSION_IS_NOT_A_LVALUE__IT_IS_READ_ONLY)

#define EIGEN_STATIC_ASSERT_ARRAYXPR(Derived)                                                                          \
  EIGEN_STATIC_ASSERT((Eigen::internal::is_same<typename Eigen::internal::traits<Derived>::XprKind, ArrayXpr>::value), \
                      THIS_METHOD_IS_ONLY_FOR_ARRAYS_NOT_MATRICES)

#define EIGEN_STATIC_ASSERT_SAME_XPR_KIND(Derived1, Derived2)                                                 \
  EIGEN_STATIC_ASSERT((Eigen::internal::is_same<typename Eigen::internal::traits<Derived1>::XprKind,          \
                                                typename Eigen::internal::traits<Derived2>::XprKind>::value), \
                      YOU_CANNOT_MIX_ARRAYS_AND_MATRICES)

// Check that a cost value is positive, and that is stay within a reasonable range
// TODO this check could be enabled for internal debugging only
#define EIGEN_INTERNAL_CHECK_COST_VALUE(C)                    \
  EIGEN_STATIC_ASSERT((C) >= 0 && (C) <= HugeCost * HugeCost, \
                      EIGEN_INTERNAL_ERROR_PLEASE_FILE_A_BUG_REPORT__INVALID_COST_VALUE);

#endif  // EIGEN_STATIC_ASSERT_H
