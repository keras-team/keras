// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ACCELERATESUPPORT_MODULE_H
#define EIGEN_ACCELERATESUPPORT_MODULE_H

#include "SparseCore"

#include "src/Core/util/DisableStupidWarnings.h"

/** \ingroup Support_modules
 * \defgroup AccelerateSupport_Module AccelerateSupport module
 *
 * This module provides an interface to the Apple Accelerate library.
 * It provides the seven following main factorization classes:
 * - class AccelerateLLT: a Cholesky (LL^T) factorization.
 * - class AccelerateLDLT: the default LDL^T factorization.
 * - class AccelerateLDLTUnpivoted: a Cholesky-like LDL^T factorization with only 1x1 pivots and no pivoting
 * - class AccelerateLDLTSBK: an LDL^T factorization with Supernode Bunch-Kaufman and static pivoting
 * - class AccelerateLDLTTPP: an LDL^T factorization with full threshold partial pivoting
 * - class AccelerateQR: a QR factorization
 * - class AccelerateCholeskyAtA: a QR factorization without storing Q (equivalent to A^TA = R^T R)
 *
 * \code
 * #include <Eigen/AccelerateSupport>
 * \endcode
 *
 * In order to use this module, the Accelerate headers must be accessible from
 * the include paths, and your binary must be linked to the Accelerate framework.
 * The Accelerate library is only available on Apple hardware.
 *
 * Note that many of the algorithms can be influenced by the UpLo template
 * argument. All matrices are assumed to be symmetric. For example, the following
 * creates an LDLT factorization where your matrix is symmetric (implicit) and
 * uses the lower triangle:
 *
 * \code
 * AccelerateLDLT<SparseMatrix<float>, Lower> ldlt;
 * \endcode
 */

// IWYU pragma: begin_exports
#include "src/AccelerateSupport/AccelerateSupport.h"
// IWYU pragma: end_exports

#include "src/Core/util/ReenableStupidWarnings.h"

#endif  // EIGEN_ACCELERATESUPPORT_MODULE_H
