// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_QR_MODULE_H
#define EIGEN_QR_MODULE_H

#include "Core"

#include "Cholesky"
#include "Jacobi"
#include "Householder"

#include "src/Core/util/DisableStupidWarnings.h"

/** \defgroup QR_Module QR module
 *
 *
 *
 * This module provides various QR decompositions
 * This module also provides some MatrixBase methods, including:
 *  - MatrixBase::householderQr()
 *  - MatrixBase::colPivHouseholderQr()
 *  - MatrixBase::fullPivHouseholderQr()
 *
 * \code
 * #include <Eigen/QR>
 * \endcode
 */

// IWYU pragma: begin_exports
#include "src/QR/HouseholderQR.h"
#include "src/QR/FullPivHouseholderQR.h"
#include "src/QR/ColPivHouseholderQR.h"
#include "src/QR/CompleteOrthogonalDecomposition.h"
#ifdef EIGEN_USE_LAPACKE
#include "src/misc/lapacke_helpers.h"
#include "src/QR/HouseholderQR_LAPACKE.h"
#include "src/QR/ColPivHouseholderQR_LAPACKE.h"
#endif
// IWYU pragma: end_exports

#include "src/Core/util/ReenableStupidWarnings.h"

#endif  // EIGEN_QR_MODULE_H
