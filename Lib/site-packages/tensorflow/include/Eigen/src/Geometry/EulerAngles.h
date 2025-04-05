// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2023 Juraj Oršulić, University of Zagreb <juraj.orsulic@fer.hr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_EULERANGLES_H
#define EIGEN_EULERANGLES_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \geometry_module \ingroup Geometry_Module
 *
 *
 * \returns the canonical Euler-angles of the rotation matrix \c *this using the convention defined by the triplet (\a
 * a0,\a a1,\a a2)
 *
 * Each of the three parameters \a a0,\a a1,\a a2 represents the respective rotation axis as an integer in {0,1,2}.
 * For instance, in:
 * \code Vector3f ea = mat.eulerAngles(2, 0, 2); \endcode
 * "2" represents the z axis and "0" the x axis, etc. The returned angles are such that
 * we have the following equality:
 * \code
 * mat == AngleAxisf(ea[0], Vector3f::UnitZ())
 *      * AngleAxisf(ea[1], Vector3f::UnitX())
 *      * AngleAxisf(ea[2], Vector3f::UnitZ()); \endcode
 * This corresponds to the right-multiply conventions (with right hand side frames).
 *
 * For Tait-Bryan angle configurations (a0 != a2), the returned angles are in the ranges [-pi:pi]x[-pi/2:pi/2]x[-pi:pi].
 * For proper Euler angle configurations (a0 == a2), the returned angles are in the ranges [-pi:pi]x[0:pi]x[-pi:pi].
 *
 * The approach used is also described here:
 * https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles.pdf
 *
 * \sa class AngleAxis
 */
template <typename Derived>
EIGEN_DEVICE_FUNC inline Matrix<typename MatrixBase<Derived>::Scalar, 3, 1> MatrixBase<Derived>::canonicalEulerAngles(
    Index a0, Index a1, Index a2) const {
  /* Implemented from Graphics Gems IV */
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3)

  Matrix<Scalar, 3, 1> res;

  const Index odd = ((a0 + 1) % 3 == a1) ? 0 : 1;
  const Index i = a0;
  const Index j = (a0 + 1 + odd) % 3;
  const Index k = (a0 + 2 - odd) % 3;

  if (a0 == a2) {
    // Proper Euler angles (same first and last axis).
    // The i, j, k indices enable addressing the input matrix as the XYX archetype matrix (see Graphics Gems IV),
    // where e.g. coeff(k, i) means third column, first row in the XYX archetype matrix:
    //  c2      s2s1              s2c1
    //  s2s3   -c2s1s3 + c1c3    -c2c1s3 - s1c3
    // -s2c3    c2s1c3 + c1s3     c2c1c3 - s1s3

    // Note: s2 is always positive.
    Scalar s2 = numext::hypot(coeff(j, i), coeff(k, i));
    if (odd) {
      res[0] = numext::atan2(coeff(j, i), coeff(k, i));
      // s2 is always positive, so res[1] will be within the canonical [0, pi] range
      res[1] = numext::atan2(s2, coeff(i, i));
    } else {
      // In the !odd case, signs of all three angles are flipped at the very end. To keep the solution within the
      // canonical range, we flip the solution and make res[1] always negative here (since s2 is always positive,
      // -atan2(s2, c2) will always be negative). The final flip at the end due to !odd will thus make res[1] positive
      // and canonical. NB: in the general case, there are two correct solutions, but only one is canonical. For proper
      // Euler angles, flipping from one solution to the other involves flipping the sign of the second angle res[1] and
      // adding/subtracting pi to the first and third angles. The addition/subtraction of pi to the first angle res[0]
      // is handled here by flipping the signs of arguments to atan2, while the calculation of the third angle does not
      // need special adjustment since it uses the adjusted res[0] as the input and produces a correct result.
      res[0] = numext::atan2(-coeff(j, i), -coeff(k, i));
      res[1] = -numext::atan2(s2, coeff(i, i));
    }

    // With a=(0,1,0), we have i=0; j=1; k=2, and after computing the first two angles,
    // we can compute their respective rotation, and apply its inverse to M. Since the result must
    // be a rotation around x, we have:
    //
    //  c2  s1.s2 c1.s2                   1  0   0
    //  0   c1    -s1       *    M    =   0  c3  s3
    //  -s2 s1.c2 c1.c2                   0 -s3  c3
    //
    //  Thus:  m11.c1 - m21.s1 = c3  &   m12.c1 - m22.s1 = s3

    Scalar s1 = numext::sin(res[0]);
    Scalar c1 = numext::cos(res[0]);
    res[2] = numext::atan2(c1 * coeff(j, k) - s1 * coeff(k, k), c1 * coeff(j, j) - s1 * coeff(k, j));
  } else {
    // Tait-Bryan angles (all three axes are different; typically used for yaw-pitch-roll calculations).
    // The i, j, k indices enable addressing the input matrix as the XYZ archetype matrix (see Graphics Gems IV),
    // where e.g. coeff(k, i) means third column, first row in the XYZ archetype matrix:
    //  c2c3    s2s1c3 - c1s3     s2c1c3 + s1s3
    //  c2s3    s2s1s3 + c1c3     s2c1s3 - s1c3
    // -s2      c2s1              c2c1

    res[0] = numext::atan2(coeff(j, k), coeff(k, k));

    Scalar c2 = numext::hypot(coeff(i, i), coeff(i, j));
    // c2 is always positive, so the following atan2 will always return a result in the correct canonical middle angle
    // range [-pi/2, pi/2]
    res[1] = numext::atan2(-coeff(i, k), c2);

    Scalar s1 = numext::sin(res[0]);
    Scalar c1 = numext::cos(res[0]);
    res[2] = numext::atan2(s1 * coeff(k, i) - c1 * coeff(j, i), c1 * coeff(j, j) - s1 * coeff(k, j));
  }
  if (!odd) {
    res = -res;
  }

  return res;
}

/** \geometry_module \ingroup Geometry_Module
 *
 *
 * \returns the Euler-angles of the rotation matrix \c *this using the convention defined by the triplet (\a a0,\a a1,\a
 * a2)
 *
 * NB: The returned angles are in non-canonical ranges [0:pi]x[-pi:pi]x[-pi:pi]. For canonical Tait-Bryan/proper Euler
 * ranges, use canonicalEulerAngles.
 *
 * \sa MatrixBase::canonicalEulerAngles
 * \sa class AngleAxis
 */
template <typename Derived>
EIGEN_DEPRECATED EIGEN_DEVICE_FUNC inline Matrix<typename MatrixBase<Derived>::Scalar, 3, 1>
MatrixBase<Derived>::eulerAngles(Index a0, Index a1, Index a2) const {
  /* Implemented from Graphics Gems IV */
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3)

  Matrix<Scalar, 3, 1> res;

  const Index odd = ((a0 + 1) % 3 == a1) ? 0 : 1;
  const Index i = a0;
  const Index j = (a0 + 1 + odd) % 3;
  const Index k = (a0 + 2 - odd) % 3;

  if (a0 == a2) {
    res[0] = numext::atan2(coeff(j, i), coeff(k, i));
    if ((odd && res[0] < Scalar(0)) || ((!odd) && res[0] > Scalar(0))) {
      if (res[0] > Scalar(0)) {
        res[0] -= Scalar(EIGEN_PI);
      } else {
        res[0] += Scalar(EIGEN_PI);
      }

      Scalar s2 = numext::hypot(coeff(j, i), coeff(k, i));
      res[1] = -numext::atan2(s2, coeff(i, i));
    } else {
      Scalar s2 = numext::hypot(coeff(j, i), coeff(k, i));
      res[1] = numext::atan2(s2, coeff(i, i));
    }

    // With a=(0,1,0), we have i=0; j=1; k=2, and after computing the first two angles,
    // we can compute their respective rotation, and apply its inverse to M. Since the result must
    // be a rotation around x, we have:
    //
    //  c2  s1.s2 c1.s2                   1  0   0
    //  0   c1    -s1       *    M    =   0  c3  s3
    //  -s2 s1.c2 c1.c2                   0 -s3  c3
    //
    //  Thus:  m11.c1 - m21.s1 = c3  &   m12.c1 - m22.s1 = s3

    Scalar s1 = numext::sin(res[0]);
    Scalar c1 = numext::cos(res[0]);
    res[2] = numext::atan2(c1 * coeff(j, k) - s1 * coeff(k, k), c1 * coeff(j, j) - s1 * coeff(k, j));
  } else {
    res[0] = numext::atan2(coeff(j, k), coeff(k, k));
    Scalar c2 = numext::hypot(coeff(i, i), coeff(i, j));
    if ((odd && res[0] < Scalar(0)) || ((!odd) && res[0] > Scalar(0))) {
      if (res[0] > Scalar(0)) {
        res[0] -= Scalar(EIGEN_PI);
      } else {
        res[0] += Scalar(EIGEN_PI);
      }
      res[1] = numext::atan2(-coeff(i, k), -c2);
    } else {
      res[1] = numext::atan2(-coeff(i, k), c2);
    }
    Scalar s1 = numext::sin(res[0]);
    Scalar c1 = numext::cos(res[0]);
    res[2] = numext::atan2(s1 * coeff(k, i) - c1 * coeff(j, i), c1 * coeff(j, j) - s1 * coeff(k, j));
  }
  if (!odd) {
    res = -res;
  }

  return res;
}

}  // end namespace Eigen

#endif  // EIGEN_EULERANGLES_H
