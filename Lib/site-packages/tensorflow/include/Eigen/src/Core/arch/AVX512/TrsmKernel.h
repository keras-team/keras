// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Intel Corporation
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CORE_ARCH_AVX512_TRSM_KERNEL_H
#define EIGEN_CORE_ARCH_AVX512_TRSM_KERNEL_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

#if !defined(EIGEN_USE_AVX512_TRSM_KERNELS)
#define EIGEN_USE_AVX512_TRSM_KERNELS 1
#endif

// TRSM kernels currently unconditionally rely on malloc with AVX512.
// Disable them if malloc is explicitly disabled at compile-time.
#ifdef EIGEN_NO_MALLOC
#undef EIGEN_USE_AVX512_TRSM_KERNELS
#define EIGEN_USE_AVX512_TRSM_KERNELS 0
#endif

#if EIGEN_USE_AVX512_TRSM_KERNELS
#if !defined(EIGEN_USE_AVX512_TRSM_R_KERNELS)
#define EIGEN_USE_AVX512_TRSM_R_KERNELS 1
#endif
#if !defined(EIGEN_USE_AVX512_TRSM_L_KERNELS)
#define EIGEN_USE_AVX512_TRSM_L_KERNELS 1
#endif
#else  // EIGEN_USE_AVX512_TRSM_KERNELS == 0
#define EIGEN_USE_AVX512_TRSM_R_KERNELS 0
#define EIGEN_USE_AVX512_TRSM_L_KERNELS 0
#endif

// Need this for some std::min calls.
#ifdef min
#undef min
#endif

namespace Eigen {
namespace internal {

#define EIGEN_AVX_MAX_NUM_ACC (int64_t(24))
#define EIGEN_AVX_MAX_NUM_ROW (int64_t(8))  // Denoted L in code.
#define EIGEN_AVX_MAX_K_UNROL (int64_t(4))
#define EIGEN_AVX_B_LOAD_SETS (int64_t(2))
#define EIGEN_AVX_MAX_A_BCAST (int64_t(2))
typedef Packet16f vecFullFloat;
typedef Packet8d vecFullDouble;
typedef Packet8f vecHalfFloat;
typedef Packet4d vecHalfDouble;

// Compile-time unrolls are implemented here.
// Note: this depends on macros and typedefs above.
#include "TrsmUnrolls.inc"

#if (EIGEN_USE_AVX512_TRSM_KERNELS) && (EIGEN_COMP_CLANG != 0)
/**
 * For smaller problem sizes, and certain compilers, using the optimized kernels trsmKernelL/R directly
 * is faster than the packed versions in TriangularSolverMatrix.h.
 *
 * The current heuristic is based on having having all arrays used in the largest gemm-update
 * in triSolve fit in roughly L2Cap (percentage) of the L2 cache. These cutoffs are a bit conservative and could be
 * larger for some trsm cases.
 * The formula:
 *
 *   (L*M + M*N + L*N)*sizeof(Scalar) < L2Cache*L2Cap
 *
 *  L = number of rows to solve at a time
 *  N = number of rhs
 *  M = Dimension of triangular matrix
 *
 */
#if !defined(EIGEN_ENABLE_AVX512_NOCOPY_TRSM_CUTOFFS)
#define EIGEN_ENABLE_AVX512_NOCOPY_TRSM_CUTOFFS 1
#endif

#if EIGEN_ENABLE_AVX512_NOCOPY_TRSM_CUTOFFS

#if EIGEN_USE_AVX512_TRSM_R_KERNELS
#if !defined(EIGEN_ENABLE_AVX512_NOCOPY_TRSM_R_CUTOFFS)
#define EIGEN_ENABLE_AVX512_NOCOPY_TRSM_R_CUTOFFS 1
#endif  // !defined(EIGEN_ENABLE_AVX512_NOCOPY_TRSM_R_CUTOFFS)
#endif

#if EIGEN_USE_AVX512_TRSM_L_KERNELS
#if !defined(EIGEN_ENABLE_AVX512_NOCOPY_TRSM_L_CUTOFFS)
#define EIGEN_ENABLE_AVX512_NOCOPY_TRSM_L_CUTOFFS 1
#endif
#endif  // EIGEN_USE_AVX512_TRSM_L_KERNELS

#else  // EIGEN_ENABLE_AVX512_NOCOPY_TRSM_CUTOFFS == 0
#define EIGEN_ENABLE_AVX512_NOCOPY_TRSM_R_CUTOFFS 0
#define EIGEN_ENABLE_AVX512_NOCOPY_TRSM_L_CUTOFFS 0
#endif  // EIGEN_ENABLE_AVX512_NOCOPY_TRSM_CUTOFFS

template <typename Scalar>
int64_t avx512_trsm_cutoff(int64_t L2Size, int64_t N, double L2Cap) {
  const int64_t U3 = 3 * packet_traits<Scalar>::size;
  const int64_t MaxNb = 5 * U3;
  int64_t Nb = std::min(MaxNb, N);
  double cutoff_d =
      (((L2Size * L2Cap) / (sizeof(Scalar))) - (EIGEN_AVX_MAX_NUM_ROW)*Nb) / ((EIGEN_AVX_MAX_NUM_ROW) + Nb);
  int64_t cutoff_l = static_cast<int64_t>(cutoff_d);
  return (cutoff_l / EIGEN_AVX_MAX_NUM_ROW) * EIGEN_AVX_MAX_NUM_ROW;
}
#else  // !(EIGEN_USE_AVX512_TRSM_KERNELS) || !(EIGEN_COMP_CLANG != 0)
#define EIGEN_ENABLE_AVX512_NOCOPY_TRSM_CUTOFFS 0
#define EIGEN_ENABLE_AVX512_NOCOPY_TRSM_R_CUTOFFS 0
#define EIGEN_ENABLE_AVX512_NOCOPY_TRSM_L_CUTOFFS 0
#endif

/**
 * Used by gemmKernel for the case A/B row-major and C col-major.
 */
template <typename Scalar, typename vec, int64_t unrollM, int64_t unrollN, bool remM, bool remN>
EIGEN_ALWAYS_INLINE void transStoreC(PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, Scalar *C_arr,
                                     int64_t LDC, int64_t remM_ = 0, int64_t remN_ = 0) {
  EIGEN_UNUSED_VARIABLE(remN_);
  EIGEN_UNUSED_VARIABLE(remM_);
  using urolls = unrolls::trans<Scalar>;

  constexpr int64_t U3 = urolls::PacketSize * 3;
  constexpr int64_t U2 = urolls::PacketSize * 2;
  constexpr int64_t U1 = urolls::PacketSize * 1;

  static_assert(unrollN == U1 || unrollN == U2 || unrollN == U3, "unrollN should be a multiple of PacketSize");
  static_assert(unrollM == EIGEN_AVX_MAX_NUM_ROW, "unrollM should be equal to EIGEN_AVX_MAX_NUM_ROW");

  urolls::template transpose<unrollN, 0>(zmm);
  EIGEN_IF_CONSTEXPR(unrollN > U2) urolls::template transpose<unrollN, 2>(zmm);
  EIGEN_IF_CONSTEXPR(unrollN > U1) urolls::template transpose<unrollN, 1>(zmm);

  static_assert((remN && unrollN == U1) || !remN, "When handling N remainder set unrollN=U1");
  EIGEN_IF_CONSTEXPR(!remN) {
    urolls::template storeC<std::min(unrollN, U1), unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
    EIGEN_IF_CONSTEXPR(unrollN > U1) {
      constexpr int64_t unrollN_ = std::min(unrollN - U1, U1);
      urolls::template storeC<unrollN_, unrollN, 1, remM>(C_arr + U1 * LDC, LDC, zmm, remM_);
    }
    EIGEN_IF_CONSTEXPR(unrollN > U2) {
      constexpr int64_t unrollN_ = std::min(unrollN - U2, U1);
      urolls::template storeC<unrollN_, unrollN, 2, remM>(C_arr + U2 * LDC, LDC, zmm, remM_);
    }
  }
  else {
    EIGEN_IF_CONSTEXPR((std::is_same<Scalar, float>::value)) {
      // Note: without "if constexpr" this section of code will also be
      // parsed by the compiler so each of the storeC will still be instantiated.
      // We use enable_if in aux_storeC to set it to an empty function for
      // these cases.
      if (remN_ == 15)
        urolls::template storeC<15, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 14)
        urolls::template storeC<14, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 13)
        urolls::template storeC<13, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 12)
        urolls::template storeC<12, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 11)
        urolls::template storeC<11, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 10)
        urolls::template storeC<10, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 9)
        urolls::template storeC<9, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 8)
        urolls::template storeC<8, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 7)
        urolls::template storeC<7, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 6)
        urolls::template storeC<6, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 5)
        urolls::template storeC<5, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 4)
        urolls::template storeC<4, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 3)
        urolls::template storeC<3, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 2)
        urolls::template storeC<2, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 1)
        urolls::template storeC<1, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
    }
    else {
      if (remN_ == 7)
        urolls::template storeC<7, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 6)
        urolls::template storeC<6, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 5)
        urolls::template storeC<5, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 4)
        urolls::template storeC<4, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 3)
        urolls::template storeC<3, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 2)
        urolls::template storeC<2, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
      else if (remN_ == 1)
        urolls::template storeC<1, unrollN, 0, remM>(C_arr, LDC, zmm, remM_);
    }
  }
}

/**
 * GEMM like operation for trsm panel updates.
 * Computes: C -= A*B
 * K must be multipe of 4.
 *
 * Unrolls used are {1,2,4,8}x{U1,U2,U3};
 * For good performance we want K to be large with M/N relatively small, but also large enough
 * to use the {8,U3} unroll block.
 *
 * isARowMajor: is A_arr row-major?
 * isCRowMajor: is C_arr row-major? (B_arr is assumed to be row-major).
 * isAdd: C += A*B or C -= A*B (used by trsm)
 * handleKRem: Handle arbitrary K? This is not needed for trsm.
 */
template <typename Scalar, bool isARowMajor, bool isCRowMajor, bool isAdd, bool handleKRem>
void gemmKernel(Scalar *A_arr, Scalar *B_arr, Scalar *C_arr, int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB,
                int64_t LDC) {
  using urolls = unrolls::gemm<Scalar, isAdd>;
  constexpr int64_t U3 = urolls::PacketSize * 3;
  constexpr int64_t U2 = urolls::PacketSize * 2;
  constexpr int64_t U1 = urolls::PacketSize * 1;
  using vec = typename std::conditional<std::is_same<Scalar, float>::value, vecFullFloat, vecFullDouble>::type;
  int64_t N_ = (N / U3) * U3;
  int64_t M_ = (M / EIGEN_AVX_MAX_NUM_ROW) * EIGEN_AVX_MAX_NUM_ROW;
  int64_t K_ = (K / EIGEN_AVX_MAX_K_UNROL) * EIGEN_AVX_MAX_K_UNROL;
  int64_t j = 0;
  for (; j < N_; j += U3) {
    constexpr int64_t EIGEN_AVX_MAX_B_LOAD = EIGEN_AVX_B_LOAD_SETS * 3;
    int64_t i = 0;
    for (; i < M_; i += EIGEN_AVX_MAX_NUM_ROW) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)], *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<3, EIGEN_AVX_MAX_NUM_ROW>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 3, EIGEN_AVX_MAX_NUM_ROW, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 3, EIGEN_AVX_MAX_NUM_ROW, 1, EIGEN_AVX_B_LOAD_SETS * 3,
                                       EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<3, EIGEN_AVX_MAX_NUM_ROW>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<3, EIGEN_AVX_MAX_NUM_ROW>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U3, false, false>(zmm, &C_arr[i + j * LDC], LDC);
      }
    }
    if (M - i >= 4) {  // Note: this block assumes EIGEN_AVX_MAX_NUM_ROW = 8. Should be removed otherwise
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<3, 4>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 3, 4, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_B_LOAD_SETS * 3,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 3, 4, 1, EIGEN_AVX_B_LOAD_SETS * 3, EIGEN_AVX_MAX_A_BCAST>(
              B_t, A_t, LDB, LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<3, 4>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<3, 4>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U3, true, false>(zmm, &C_arr[i + j * LDC], LDC, 4);
      }
      i += 4;
    }
    if (M - i >= 2) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<3, 2>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 3, 2, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_B_LOAD_SETS * 3,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 3, 2, 1, EIGEN_AVX_B_LOAD_SETS * 3, EIGEN_AVX_MAX_A_BCAST>(
              B_t, A_t, LDB, LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<3, 2>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<3, 2>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U3, true, false>(zmm, &C_arr[i + j * LDC], LDC, 2);
      }
      i += 2;
    }
    if (M - i > 0) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<3, 1>(zmm);
      {
        for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
          urolls::template microKernel<isARowMajor, 3, 1, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_B_LOAD_SETS * 3, 1>(
              B_t, A_t, LDB, LDA, zmm);
          B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
          else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
        }
        EIGEN_IF_CONSTEXPR(handleKRem) {
          for (int64_t k = K_; k < K; k++) {
            urolls::template microKernel<isARowMajor, 3, 1, 1, EIGEN_AVX_B_LOAD_SETS * 3, 1>(B_t, A_t, LDB, LDA, zmm);
            B_t += LDB;
            EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
            else A_t += LDA;
          }
        }
        EIGEN_IF_CONSTEXPR(isCRowMajor) {
          urolls::template updateC<3, 1>(&C_arr[i * LDC + j], LDC, zmm);
          urolls::template storeC<3, 1>(&C_arr[i * LDC + j], LDC, zmm);
        }
        else {
          transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U3, true, false>(zmm, &C_arr[i + j * LDC], LDC, 1);
        }
      }
    }
  }
  if (N - j >= U2) {
    constexpr int64_t EIGEN_AVX_MAX_B_LOAD = EIGEN_AVX_B_LOAD_SETS * 2;
    int64_t i = 0;
    for (; i < M_; i += EIGEN_AVX_MAX_NUM_ROW) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)], *B_t = &B_arr[0 * LDB + j];
      EIGEN_IF_CONSTEXPR(isCRowMajor) B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<2, EIGEN_AVX_MAX_NUM_ROW>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 2, EIGEN_AVX_MAX_NUM_ROW, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 2, EIGEN_AVX_MAX_NUM_ROW, 1, EIGEN_AVX_MAX_B_LOAD,
                                       EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<2, EIGEN_AVX_MAX_NUM_ROW>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<2, EIGEN_AVX_MAX_NUM_ROW>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U2, false, false>(zmm, &C_arr[i + j * LDC], LDC);
      }
    }
    if (M - i >= 4) {  // Note: this block assumes EIGEN_AVX_MAX_NUM_ROW = 8. Should be removed otherwise
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<2, 4>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 2, 4, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 2, 4, 1, EIGEN_AVX_MAX_B_LOAD, EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB,
                                                                                                          LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<2, 4>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<2, 4>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U2, true, false>(zmm, &C_arr[i + j * LDC], LDC, 4);
      }
      i += 4;
    }
    if (M - i >= 2) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<2, 2>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 2, 2, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 2, 2, 1, EIGEN_AVX_MAX_B_LOAD, EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB,
                                                                                                          LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<2, 2>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<2, 2>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U2, true, false>(zmm, &C_arr[i + j * LDC], LDC, 2);
      }
      i += 2;
    }
    if (M - i > 0) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<2, 1>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 2, 1, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD, 1>(B_t, A_t, LDB,
                                                                                                        LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 2, 1, 1, EIGEN_AVX_MAX_B_LOAD, 1>(B_t, A_t, LDB, LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<2, 1>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<2, 1>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U2, true, false>(zmm, &C_arr[i + j * LDC], LDC, 1);
      }
    }
    j += U2;
  }
  if (N - j >= U1) {
    constexpr int64_t EIGEN_AVX_MAX_B_LOAD = EIGEN_AVX_B_LOAD_SETS * 1;
    int64_t i = 0;
    for (; i < M_; i += EIGEN_AVX_MAX_NUM_ROW) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)], *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<1, EIGEN_AVX_MAX_NUM_ROW>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 1, EIGEN_AVX_MAX_NUM_ROW, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 1, EIGEN_AVX_MAX_NUM_ROW, 1, EIGEN_AVX_B_LOAD_SETS * 1,
                                       EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<1, EIGEN_AVX_MAX_NUM_ROW>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<1, EIGEN_AVX_MAX_NUM_ROW>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U1, false, false>(zmm, &C_arr[i + j * LDC], LDC);
      }
    }
    if (M - i >= 4) {  // Note: this block assumes EIGEN_AVX_MAX_NUM_ROW = 8. Should be removed otherwise
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<1, 4>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 1, 4, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 1, 4, 1, EIGEN_AVX_MAX_B_LOAD, EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB,
                                                                                                          LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<1, 4>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<1, 4>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U1, true, false>(zmm, &C_arr[i + j * LDC], LDC, 4);
      }
      i += 4;
    }
    if (M - i >= 2) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<1, 2>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 1, 2, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB, LDA, zmm);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 1, 2, 1, EIGEN_AVX_MAX_B_LOAD, EIGEN_AVX_MAX_A_BCAST>(B_t, A_t, LDB,
                                                                                                          LDA, zmm);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<1, 2>(&C_arr[i * LDC + j], LDC, zmm);
        urolls::template storeC<1, 2>(&C_arr[i * LDC + j], LDC, zmm);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U1, true, false>(zmm, &C_arr[i + j * LDC], LDC, 2);
      }
      i += 2;
    }
    if (M - i > 0) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<1, 1>(zmm);
      {
        for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
          urolls::template microKernel<isARowMajor, 1, 1, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD, 1>(B_t, A_t, LDB,
                                                                                                          LDA, zmm);
          B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
          else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
        }
        EIGEN_IF_CONSTEXPR(handleKRem) {
          for (int64_t k = K_; k < K; k++) {
            urolls::template microKernel<isARowMajor, 1, 1, 1, EIGEN_AVX_B_LOAD_SETS * 1, 1>(B_t, A_t, LDB, LDA, zmm);
            B_t += LDB;
            EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
            else A_t += LDA;
          }
        }
        EIGEN_IF_CONSTEXPR(isCRowMajor) {
          urolls::template updateC<1, 1>(&C_arr[i * LDC + j], LDC, zmm);
          urolls::template storeC<1, 1>(&C_arr[i * LDC + j], LDC, zmm);
        }
        else {
          transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U1, true, false>(zmm, &C_arr[i + j * LDC], LDC, 1);
        }
      }
    }
    j += U1;
  }
  if (N - j > 0) {
    constexpr int64_t EIGEN_AVX_MAX_B_LOAD = EIGEN_AVX_B_LOAD_SETS * 1;
    int64_t i = 0;
    for (; i < M_; i += EIGEN_AVX_MAX_NUM_ROW) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<1, EIGEN_AVX_MAX_NUM_ROW>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 1, EIGEN_AVX_MAX_NUM_ROW, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST, true>(B_t, A_t, LDB, LDA, zmm, N - j);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 1, EIGEN_AVX_MAX_NUM_ROW, 1, EIGEN_AVX_MAX_B_LOAD,
                                       EIGEN_AVX_MAX_A_BCAST, true>(B_t, A_t, LDB, LDA, zmm, N - j);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<1, EIGEN_AVX_MAX_NUM_ROW, true>(&C_arr[i * LDC + j], LDC, zmm, N - j);
        urolls::template storeC<1, EIGEN_AVX_MAX_NUM_ROW, true>(&C_arr[i * LDC + j], LDC, zmm, N - j);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U1, false, true>(zmm, &C_arr[i + j * LDC], LDC, 0, N - j);
      }
    }
    if (M - i >= 4) {  // Note: this block assumes EIGEN_AVX_MAX_NUM_ROW = 8. Should be removed otherwise
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<1, 4>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 1, 4, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST, true>(B_t, A_t, LDB, LDA, zmm, N - j);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 1, 4, 1, EIGEN_AVX_MAX_B_LOAD, EIGEN_AVX_MAX_A_BCAST, true>(
              B_t, A_t, LDB, LDA, zmm, N - j);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<1, 4, true>(&C_arr[i * LDC + j], LDC, zmm, N - j);
        urolls::template storeC<1, 4, true>(&C_arr[i * LDC + j], LDC, zmm, N - j);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U1, true, true>(zmm, &C_arr[i + j * LDC], LDC, 4, N - j);
      }
      i += 4;
    }
    if (M - i >= 2) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<1, 2>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 1, 2, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD,
                                     EIGEN_AVX_MAX_A_BCAST, true>(B_t, A_t, LDB, LDA, zmm, N - j);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 1, 2, 1, EIGEN_AVX_MAX_B_LOAD, EIGEN_AVX_MAX_A_BCAST, true>(
              B_t, A_t, LDB, LDA, zmm, N - j);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<1, 2, true>(&C_arr[i * LDC + j], LDC, zmm, N - j);
        urolls::template storeC<1, 2, true>(&C_arr[i * LDC + j], LDC, zmm, N - j);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U1, true, true>(zmm, &C_arr[i + j * LDC], LDC, 2, N - j);
      }
      i += 2;
    }
    if (M - i > 0) {
      Scalar *A_t = &A_arr[idA<isARowMajor>(i, 0, LDA)];
      Scalar *B_t = &B_arr[0 * LDB + j];
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> zmm;
      urolls::template setzero<1, 1>(zmm);
      for (int64_t k = 0; k < K_; k += EIGEN_AVX_MAX_K_UNROL) {
        urolls::template microKernel<isARowMajor, 1, 1, EIGEN_AVX_MAX_K_UNROL, EIGEN_AVX_MAX_B_LOAD, 1, true>(
            B_t, A_t, LDB, LDA, zmm, N - j);
        B_t += EIGEN_AVX_MAX_K_UNROL * LDB;
        EIGEN_IF_CONSTEXPR(isARowMajor) A_t += EIGEN_AVX_MAX_K_UNROL;
        else A_t += EIGEN_AVX_MAX_K_UNROL * LDA;
      }
      EIGEN_IF_CONSTEXPR(handleKRem) {
        for (int64_t k = K_; k < K; k++) {
          urolls::template microKernel<isARowMajor, 1, 1, 1, EIGEN_AVX_MAX_B_LOAD, 1, true>(B_t, A_t, LDB, LDA, zmm,
                                                                                            N - j);
          B_t += LDB;
          EIGEN_IF_CONSTEXPR(isARowMajor) A_t++;
          else A_t += LDA;
        }
      }
      EIGEN_IF_CONSTEXPR(isCRowMajor) {
        urolls::template updateC<1, 1, true>(&C_arr[i * LDC + j], LDC, zmm, N - j);
        urolls::template storeC<1, 1, true>(&C_arr[i * LDC + j], LDC, zmm, N - j);
      }
      else {
        transStoreC<Scalar, vec, EIGEN_AVX_MAX_NUM_ROW, U1, true, true>(zmm, &C_arr[i + j * LDC], LDC, 1, N - j);
      }
    }
  }
}

/**
 * Triangular solve kernel with A on left with K number of rhs. dim(A) = unrollM
 *
 * unrollM: dimension of A matrix (triangular matrix). unrollM should be <= EIGEN_AVX_MAX_NUM_ROW
 * isFWDSolve: is forward solve?
 * isUnitDiag: is the diagonal of A all ones?
 * The B matrix (RHS) is assumed to be row-major
 */
template <typename Scalar, typename vec, int64_t unrollM, bool isARowMajor, bool isFWDSolve, bool isUnitDiag>
EIGEN_ALWAYS_INLINE void triSolveKernel(Scalar *A_arr, Scalar *B_arr, int64_t K, int64_t LDA, int64_t LDB) {
  static_assert(unrollM <= EIGEN_AVX_MAX_NUM_ROW, "unrollM should be equal to EIGEN_AVX_MAX_NUM_ROW");
  using urolls = unrolls::trsm<Scalar>;
  constexpr int64_t U3 = urolls::PacketSize * 3;
  constexpr int64_t U2 = urolls::PacketSize * 2;
  constexpr int64_t U1 = urolls::PacketSize * 1;

  PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> RHSInPacket;
  PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> AInPacket;

  int64_t k = 0;
  while (K - k >= U3) {
    urolls::template loadRHS<isFWDSolve, unrollM, 3>(B_arr + k, LDB, RHSInPacket);
    urolls::template triSolveMicroKernel<isARowMajor, isFWDSolve, isUnitDiag, unrollM, 3>(A_arr, LDA, RHSInPacket,
                                                                                          AInPacket);
    urolls::template storeRHS<isFWDSolve, unrollM, 3>(B_arr + k, LDB, RHSInPacket);
    k += U3;
  }
  if (K - k >= U2) {
    urolls::template loadRHS<isFWDSolve, unrollM, 2>(B_arr + k, LDB, RHSInPacket);
    urolls::template triSolveMicroKernel<isARowMajor, isFWDSolve, isUnitDiag, unrollM, 2>(A_arr, LDA, RHSInPacket,
                                                                                          AInPacket);
    urolls::template storeRHS<isFWDSolve, unrollM, 2>(B_arr + k, LDB, RHSInPacket);
    k += U2;
  }
  if (K - k >= U1) {
    urolls::template loadRHS<isFWDSolve, unrollM, 1>(B_arr + k, LDB, RHSInPacket);
    urolls::template triSolveMicroKernel<isARowMajor, isFWDSolve, isUnitDiag, unrollM, 1>(A_arr, LDA, RHSInPacket,
                                                                                          AInPacket);
    urolls::template storeRHS<isFWDSolve, unrollM, 1>(B_arr + k, LDB, RHSInPacket);
    k += U1;
  }
  if (K - k > 0) {
    // Handle remaining number of RHS
    urolls::template loadRHS<isFWDSolve, unrollM, 1, true>(B_arr + k, LDB, RHSInPacket, K - k);
    urolls::template triSolveMicroKernel<isARowMajor, isFWDSolve, isUnitDiag, unrollM, 1>(A_arr, LDA, RHSInPacket,
                                                                                          AInPacket);
    urolls::template storeRHS<isFWDSolve, unrollM, 1, true>(B_arr + k, LDB, RHSInPacket, K - k);
  }
}

/**
 * Triangular solve routine with A on left and dimension of at most L with K number of rhs. This is essentially
 * a wrapper for triSolveMicrokernel for M = {1,2,3,4,5,6,7,8}.
 *
 * isFWDSolve: is forward solve?
 * isUnitDiag: is the diagonal of A all ones?
 * The B matrix (RHS) is assumed to be row-major
 */
template <typename Scalar, bool isARowMajor, bool isFWDSolve, bool isUnitDiag>
void triSolveKernelLxK(Scalar *A_arr, Scalar *B_arr, int64_t M, int64_t K, int64_t LDA, int64_t LDB) {
  // Note: this assumes EIGEN_AVX_MAX_NUM_ROW = 8. Unrolls should be adjusted
  // accordingly if EIGEN_AVX_MAX_NUM_ROW is smaller.
  using vec = typename std::conditional<std::is_same<Scalar, float>::value, vecFullFloat, vecFullDouble>::type;
  if (M == 8)
    triSolveKernel<Scalar, vec, 8, isARowMajor, isFWDSolve, isUnitDiag>(A_arr, B_arr, K, LDA, LDB);
  else if (M == 7)
    triSolveKernel<Scalar, vec, 7, isARowMajor, isFWDSolve, isUnitDiag>(A_arr, B_arr, K, LDA, LDB);
  else if (M == 6)
    triSolveKernel<Scalar, vec, 6, isARowMajor, isFWDSolve, isUnitDiag>(A_arr, B_arr, K, LDA, LDB);
  else if (M == 5)
    triSolveKernel<Scalar, vec, 5, isARowMajor, isFWDSolve, isUnitDiag>(A_arr, B_arr, K, LDA, LDB);
  else if (M == 4)
    triSolveKernel<Scalar, vec, 4, isARowMajor, isFWDSolve, isUnitDiag>(A_arr, B_arr, K, LDA, LDB);
  else if (M == 3)
    triSolveKernel<Scalar, vec, 3, isARowMajor, isFWDSolve, isUnitDiag>(A_arr, B_arr, K, LDA, LDB);
  else if (M == 2)
    triSolveKernel<Scalar, vec, 2, isARowMajor, isFWDSolve, isUnitDiag>(A_arr, B_arr, K, LDA, LDB);
  else if (M == 1)
    triSolveKernel<Scalar, vec, 1, isARowMajor, isFWDSolve, isUnitDiag>(A_arr, B_arr, K, LDA, LDB);
  return;
}

/**
 * This routine is used to copy B to/from a temporary array (row-major) for cases where B is column-major.
 *
 * toTemp: true => copy to temporary array, false => copy from temporary array
 * remM: true = need to handle remainder values for M (M < EIGEN_AVX_MAX_NUM_ROW)
 *
 */
template <typename Scalar, bool toTemp = true, bool remM = false>
EIGEN_ALWAYS_INLINE void copyBToRowMajor(Scalar *B_arr, int64_t LDB, int64_t K, Scalar *B_temp, int64_t LDB_,
                                         int64_t remM_ = 0) {
  EIGEN_UNUSED_VARIABLE(remM_);
  using urolls = unrolls::transB<Scalar>;
  using vecHalf = typename std::conditional<std::is_same<Scalar, float>::value, vecHalfFloat, vecFullDouble>::type;
  PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> ymm;
  constexpr int64_t U3 = urolls::PacketSize * 3;
  constexpr int64_t U2 = urolls::PacketSize * 2;
  constexpr int64_t U1 = urolls::PacketSize * 1;
  int64_t K_ = K / U3 * U3;
  int64_t k = 0;

  for (; k < K_; k += U3) {
    urolls::template transB_kernel<U3, toTemp, remM>(B_arr + k * LDB, LDB, B_temp, LDB_, ymm, remM_);
    B_temp += U3;
  }
  if (K - k >= U2) {
    urolls::template transB_kernel<U2, toTemp, remM>(B_arr + k * LDB, LDB, B_temp, LDB_, ymm, remM_);
    B_temp += U2;
    k += U2;
  }
  if (K - k >= U1) {
    urolls::template transB_kernel<U1, toTemp, remM>(B_arr + k * LDB, LDB, B_temp, LDB_, ymm, remM_);
    B_temp += U1;
    k += U1;
  }
  EIGEN_IF_CONSTEXPR(U1 > 8) {
    // Note: without "if constexpr" this section of code will also be
    // parsed by the compiler so there is an additional check in {load/store}BBlock
    // to make sure the counter is not non-negative.
    if (K - k >= 8) {
      urolls::template transB_kernel<8, toTemp, remM>(B_arr + k * LDB, LDB, B_temp, LDB_, ymm, remM_);
      B_temp += 8;
      k += 8;
    }
  }
  EIGEN_IF_CONSTEXPR(U1 > 4) {
    // Note: without "if constexpr" this section of code will also be
    // parsed by the compiler so there is an additional check in {load/store}BBlock
    // to make sure the counter is not non-negative.
    if (K - k >= 4) {
      urolls::template transB_kernel<4, toTemp, remM>(B_arr + k * LDB, LDB, B_temp, LDB_, ymm, remM_);
      B_temp += 4;
      k += 4;
    }
  }
  if (K - k >= 2) {
    urolls::template transB_kernel<2, toTemp, remM>(B_arr + k * LDB, LDB, B_temp, LDB_, ymm, remM_);
    B_temp += 2;
    k += 2;
  }
  if (K - k >= 1) {
    urolls::template transB_kernel<1, toTemp, remM>(B_arr + k * LDB, LDB, B_temp, LDB_, ymm, remM_);
    B_temp += 1;
    k += 1;
  }
}

/**
 * Main triangular solve driver
 *
 * Triangular solve with A on the left.
 * Scalar:    Scalar precision, only float/double is supported.
 * isARowMajor:  is A row-major?
 * isBRowMajor:  is B row-major?
 * isFWDSolve:   is this forward solve or backward (true => forward)?
 * isUnitDiag: is diagonal of A unit or nonunit (true => A has unit diagonal)?
 *
 * M: dimension of A
 * numRHS: number of right hand sides (coincides with K dimension for gemm updates)
 *
 * Here are the mapping between the different TRSM cases (col-major) and triSolve:
 *
 * LLN (left , lower, A non-transposed) ::  isARowMajor=false, isBRowMajor=false, isFWDSolve=true
 * LUT (left , upper, A transposed)     ::  isARowMajor=true,  isBRowMajor=false, isFWDSolve=true
 * LUN (left , upper, A non-transposed) ::  isARowMajor=false, isBRowMajor=false, isFWDSolve=false
 * LLT (left , lower, A transposed)     ::  isARowMajor=true,  isBRowMajor=false, isFWDSolve=false
 * RUN (right, upper, A non-transposed) ::  isARowMajor=true,  isBRowMajor=true,  isFWDSolve=true
 * RLT (right, lower, A transposed)     ::  isARowMajor=false, isBRowMajor=true,  isFWDSolve=true
 * RUT (right, upper, A transposed)     ::  isARowMajor=false, isBRowMajor=true,  isFWDSolve=false
 * RLN (right, lower, A non-transposed) ::  isARowMajor=true,  isBRowMajor=true,  isFWDSolve=false
 *
 * Note: For RXX cases M,numRHS should be swapped.
 *
 */
template <typename Scalar, bool isARowMajor = true, bool isBRowMajor = true, bool isFWDSolve = true,
          bool isUnitDiag = false>
void triSolve(Scalar *A_arr, Scalar *B_arr, int64_t M, int64_t numRHS, int64_t LDA, int64_t LDB) {
  constexpr int64_t psize = packet_traits<Scalar>::size;
  /**
   * The values for kB, numM were determined experimentally.
   * kB: Number of RHS we process at a time.
   * numM: number of rows of B we will store in a temporary array (see below.) This should be a multiple of L.
   *
   * kB was determined by initially setting kB = numRHS and benchmarking triSolve (TRSM-RUN case)
   * performance with M=numRHS.
   * It was observed that performance started to drop around M=numRHS=240. This is likely machine dependent.
   *
   * numM was chosen "arbitrarily". It should be relatively small so B_temp is not too large, but it should be
   * large enough to allow GEMM updates to have larger "K"s (see below.) No benchmarking has been done so far to
   * determine optimal values for numM.
   */
  constexpr int64_t kB = (3 * psize) * 5;  // 5*U3
  constexpr int64_t numM = 8 * EIGEN_AVX_MAX_NUM_ROW;

  int64_t sizeBTemp = 0;
  Scalar *B_temp = NULL;
  EIGEN_IF_CONSTEXPR(!isBRowMajor) {
    /**
     * If B is col-major, we copy it to a fixed-size temporary array of size at most ~numM*kB and
     * transpose it to row-major. Call the solve routine, and copy+transpose it back to the original array.
     * The updated row-major copy of B is reused in the GEMM updates.
     */
    sizeBTemp = (((std::min(kB, numRHS) + psize - 1) / psize + 4) * psize) * numM;
  }

  EIGEN_IF_CONSTEXPR(!isBRowMajor) B_temp = (Scalar *)handmade_aligned_malloc(sizeof(Scalar) * sizeBTemp, 64);

  for (int64_t k = 0; k < numRHS; k += kB) {
    int64_t bK = numRHS - k > kB ? kB : numRHS - k;
    int64_t M_ = (M / EIGEN_AVX_MAX_NUM_ROW) * EIGEN_AVX_MAX_NUM_ROW, gemmOff = 0;

    // bK rounded up to next multiple of L=EIGEN_AVX_MAX_NUM_ROW. When B_temp is used, we solve for bkL RHS
    // instead of bK RHS in triSolveKernelLxK.
    int64_t bkL = ((bK + (EIGEN_AVX_MAX_NUM_ROW - 1)) / EIGEN_AVX_MAX_NUM_ROW) * EIGEN_AVX_MAX_NUM_ROW;
    const int64_t numScalarPerCache = 64 / sizeof(Scalar);
    // Leading dimension of B_temp, will be a multiple of the cache line size.
    int64_t LDT = ((bkL + (numScalarPerCache - 1)) / numScalarPerCache) * numScalarPerCache;
    int64_t offsetBTemp = 0;
    for (int64_t i = 0; i < M_; i += EIGEN_AVX_MAX_NUM_ROW) {
      EIGEN_IF_CONSTEXPR(!isBRowMajor) {
        int64_t indA_i = isFWDSolve ? i : M - 1 - i;
        int64_t indB_i = isFWDSolve ? i : M - (i + EIGEN_AVX_MAX_NUM_ROW);
        int64_t offB_1 = isFWDSolve ? offsetBTemp : sizeBTemp - EIGEN_AVX_MAX_NUM_ROW * LDT - offsetBTemp;
        int64_t offB_2 = isFWDSolve ? offsetBTemp : sizeBTemp - LDT - offsetBTemp;
        // Copy values from B to B_temp.
        copyBToRowMajor<Scalar, true, false>(B_arr + indB_i + k * LDB, LDB, bK, B_temp + offB_1, LDT);
        // Triangular solve with a small block of A and long horizontal blocks of B (or B_temp if B col-major)
        triSolveKernelLxK<Scalar, isARowMajor, isFWDSolve, isUnitDiag>(
            &A_arr[idA<isARowMajor>(indA_i, indA_i, LDA)], B_temp + offB_2, EIGEN_AVX_MAX_NUM_ROW, bkL, LDA, LDT);
        // Copy values from B_temp back to B. B_temp will be reused in gemm call below.
        copyBToRowMajor<Scalar, false, false>(B_arr + indB_i + k * LDB, LDB, bK, B_temp + offB_1, LDT);

        offsetBTemp += EIGEN_AVX_MAX_NUM_ROW * LDT;
      }
      else {
        int64_t ind = isFWDSolve ? i : M - 1 - i;
        triSolveKernelLxK<Scalar, isARowMajor, isFWDSolve, isUnitDiag>(
            &A_arr[idA<isARowMajor>(ind, ind, LDA)], B_arr + k + ind * LDB, EIGEN_AVX_MAX_NUM_ROW, bK, LDA, LDB);
      }
      if (i + EIGEN_AVX_MAX_NUM_ROW < M_) {
        /**
         * For the GEMM updates, we want "K" (K=i+8 in this case) to be large as soon as possible
         * to reuse the accumulators in GEMM as much as possible. So we only update 8xbK blocks of
         * B as follows:
         *
         *        A             B
         *     __
         *    |__|__           |__|
         *    |__|__|__        |__|
         *    |__|__|__|__     |__|
         *    |********|__|    |**|
         */
        EIGEN_IF_CONSTEXPR(isBRowMajor) {
          int64_t indA_i = isFWDSolve ? i + EIGEN_AVX_MAX_NUM_ROW : M - (i + 2 * EIGEN_AVX_MAX_NUM_ROW);
          int64_t indA_j = isFWDSolve ? 0 : M - (i + EIGEN_AVX_MAX_NUM_ROW);
          int64_t indB_i = isFWDSolve ? 0 : M - (i + EIGEN_AVX_MAX_NUM_ROW);
          int64_t indB_i2 = isFWDSolve ? i + EIGEN_AVX_MAX_NUM_ROW : M - (i + 2 * EIGEN_AVX_MAX_NUM_ROW);
          gemmKernel<Scalar, isARowMajor, isBRowMajor, false, false>(
              &A_arr[idA<isARowMajor>(indA_i, indA_j, LDA)], B_arr + k + indB_i * LDB, B_arr + k + indB_i2 * LDB,
              EIGEN_AVX_MAX_NUM_ROW, bK, i + EIGEN_AVX_MAX_NUM_ROW, LDA, LDB, LDB);
        }
        else {
          if (offsetBTemp + EIGEN_AVX_MAX_NUM_ROW * LDT > sizeBTemp) {
            /**
             * Similar idea as mentioned above, but here we are limited by the number of updated values of B
             * that can be stored (row-major) in B_temp.
             *
             * If there is not enough space to store the next batch of 8xbK of B in B_temp, we call GEMM
             * update and partially update the remaining old values of B which depends on the new values
             * of B stored in B_temp. These values are then no longer needed and can be overwritten.
             */
            int64_t indA_i = isFWDSolve ? i + EIGEN_AVX_MAX_NUM_ROW : 0;
            int64_t indA_j = isFWDSolve ? gemmOff : M - (i + EIGEN_AVX_MAX_NUM_ROW);
            int64_t indB_i = isFWDSolve ? i + EIGEN_AVX_MAX_NUM_ROW : 0;
            int64_t offB_1 = isFWDSolve ? 0 : sizeBTemp - offsetBTemp;
            gemmKernel<Scalar, isARowMajor, isBRowMajor, false, false>(
                &A_arr[idA<isARowMajor>(indA_i, indA_j, LDA)], B_temp + offB_1, B_arr + indB_i + (k)*LDB,
                M - (i + EIGEN_AVX_MAX_NUM_ROW), bK, i + EIGEN_AVX_MAX_NUM_ROW - gemmOff, LDA, LDT, LDB);
            offsetBTemp = 0;
            gemmOff = i + EIGEN_AVX_MAX_NUM_ROW;
          } else {
            /**
             * If there is enough space in B_temp, we only update the next 8xbK values of B.
             */
            int64_t indA_i = isFWDSolve ? i + EIGEN_AVX_MAX_NUM_ROW : M - (i + 2 * EIGEN_AVX_MAX_NUM_ROW);
            int64_t indA_j = isFWDSolve ? gemmOff : M - (i + EIGEN_AVX_MAX_NUM_ROW);
            int64_t indB_i = isFWDSolve ? i + EIGEN_AVX_MAX_NUM_ROW : M - (i + 2 * EIGEN_AVX_MAX_NUM_ROW);
            int64_t offB_1 = isFWDSolve ? 0 : sizeBTemp - offsetBTemp;
            gemmKernel<Scalar, isARowMajor, isBRowMajor, false, false>(
                &A_arr[idA<isARowMajor>(indA_i, indA_j, LDA)], B_temp + offB_1, B_arr + indB_i + (k)*LDB,
                EIGEN_AVX_MAX_NUM_ROW, bK, i + EIGEN_AVX_MAX_NUM_ROW - gemmOff, LDA, LDT, LDB);
          }
        }
      }
    }
    // Handle M remainder..
    int64_t bM = M - M_;
    if (bM > 0) {
      if (M_ > 0) {
        EIGEN_IF_CONSTEXPR(isBRowMajor) {
          int64_t indA_i = isFWDSolve ? M_ : 0;
          int64_t indA_j = isFWDSolve ? 0 : bM;
          int64_t indB_i = isFWDSolve ? 0 : bM;
          int64_t indB_i2 = isFWDSolve ? M_ : 0;
          gemmKernel<Scalar, isARowMajor, isBRowMajor, false, false>(
              &A_arr[idA<isARowMajor>(indA_i, indA_j, LDA)], B_arr + k + indB_i * LDB, B_arr + k + indB_i2 * LDB, bM,
              bK, M_, LDA, LDB, LDB);
        }
        else {
          int64_t indA_i = isFWDSolve ? M_ : 0;
          int64_t indA_j = isFWDSolve ? gemmOff : bM;
          int64_t indB_i = isFWDSolve ? M_ : 0;
          int64_t offB_1 = isFWDSolve ? 0 : sizeBTemp - offsetBTemp;
          gemmKernel<Scalar, isARowMajor, isBRowMajor, false, false>(&A_arr[idA<isARowMajor>(indA_i, indA_j, LDA)],
                                                                     B_temp + offB_1, B_arr + indB_i + (k)*LDB, bM, bK,
                                                                     M_ - gemmOff, LDA, LDT, LDB);
        }
      }
      EIGEN_IF_CONSTEXPR(!isBRowMajor) {
        int64_t indA_i = isFWDSolve ? M_ : M - 1 - M_;
        int64_t indB_i = isFWDSolve ? M_ : 0;
        int64_t offB_1 = isFWDSolve ? 0 : (bM - 1) * bkL;
        copyBToRowMajor<Scalar, true, true>(B_arr + indB_i + k * LDB, LDB, bK, B_temp, bkL, bM);
        triSolveKernelLxK<Scalar, isARowMajor, isFWDSolve, isUnitDiag>(&A_arr[idA<isARowMajor>(indA_i, indA_i, LDA)],
                                                                       B_temp + offB_1, bM, bkL, LDA, bkL);
        copyBToRowMajor<Scalar, false, true>(B_arr + indB_i + k * LDB, LDB, bK, B_temp, bkL, bM);
      }
      else {
        int64_t ind = isFWDSolve ? M_ : M - 1 - M_;
        triSolveKernelLxK<Scalar, isARowMajor, isFWDSolve, isUnitDiag>(&A_arr[idA<isARowMajor>(ind, ind, LDA)],
                                                                       B_arr + k + ind * LDB, bM, bK, LDA, LDB);
      }
    }
  }

  EIGEN_IF_CONSTEXPR(!isBRowMajor) handmade_aligned_free(B_temp);
}

// Template specializations of trsmKernelL/R for float/double and inner strides of 1.
#if (EIGEN_USE_AVX512_TRSM_KERNELS)
#if (EIGEN_USE_AVX512_TRSM_R_KERNELS)
template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride,
          bool Specialized>
struct trsmKernelR;

template <typename Index, int Mode, int TriStorageOrder>
struct trsmKernelR<float, Index, Mode, false, TriStorageOrder, 1, true> {
  static void kernel(Index size, Index otherSize, const float *_tri, Index triStride, float *_other, Index otherIncr,
                     Index otherStride);
};

template <typename Index, int Mode, int TriStorageOrder>
struct trsmKernelR<double, Index, Mode, false, TriStorageOrder, 1, true> {
  static void kernel(Index size, Index otherSize, const double *_tri, Index triStride, double *_other, Index otherIncr,
                     Index otherStride);
};

template <typename Index, int Mode, int TriStorageOrder>
EIGEN_DONT_INLINE void trsmKernelR<float, Index, Mode, false, TriStorageOrder, 1, true>::kernel(
    Index size, Index otherSize, const float *_tri, Index triStride, float *_other, Index otherIncr,
    Index otherStride) {
  EIGEN_UNUSED_VARIABLE(otherIncr);
#ifdef EIGEN_RUNTIME_NO_MALLOC
  if (!is_malloc_allowed()) {
    trsmKernelR<float, Index, Mode, false, TriStorageOrder, 1, /*Specialized=*/false>::kernel(
        size, otherSize, _tri, triStride, _other, otherIncr, otherStride);
    return;
  }
#endif
  triSolve<float, TriStorageOrder != RowMajor, true, (Mode & Lower) != Lower, (Mode & UnitDiag) != 0>(
      const_cast<float *>(_tri), _other, size, otherSize, triStride, otherStride);
}

template <typename Index, int Mode, int TriStorageOrder>
EIGEN_DONT_INLINE void trsmKernelR<double, Index, Mode, false, TriStorageOrder, 1, true>::kernel(
    Index size, Index otherSize, const double *_tri, Index triStride, double *_other, Index otherIncr,
    Index otherStride) {
  EIGEN_UNUSED_VARIABLE(otherIncr);
#ifdef EIGEN_RUNTIME_NO_MALLOC
  if (!is_malloc_allowed()) {
    trsmKernelR<double, Index, Mode, false, TriStorageOrder, 1, /*Specialized=*/false>::kernel(
        size, otherSize, _tri, triStride, _other, otherIncr, otherStride);
    return;
  }
#endif
  triSolve<double, TriStorageOrder != RowMajor, true, (Mode & Lower) != Lower, (Mode & UnitDiag) != 0>(
      const_cast<double *>(_tri), _other, size, otherSize, triStride, otherStride);
}
#endif  // (EIGEN_USE_AVX512_TRSM_R_KERNELS)

// These trsm kernels require temporary memory allocation
#if (EIGEN_USE_AVX512_TRSM_L_KERNELS)
template <typename Scalar, typename Index, int Mode, bool Conjugate, int TriStorageOrder, int OtherInnerStride,
          bool Specialized = true>
struct trsmKernelL;

template <typename Index, int Mode, int TriStorageOrder>
struct trsmKernelL<float, Index, Mode, false, TriStorageOrder, 1, true> {
  static void kernel(Index size, Index otherSize, const float *_tri, Index triStride, float *_other, Index otherIncr,
                     Index otherStride);
};

template <typename Index, int Mode, int TriStorageOrder>
struct trsmKernelL<double, Index, Mode, false, TriStorageOrder, 1, true> {
  static void kernel(Index size, Index otherSize, const double *_tri, Index triStride, double *_other, Index otherIncr,
                     Index otherStride);
};

template <typename Index, int Mode, int TriStorageOrder>
EIGEN_DONT_INLINE void trsmKernelL<float, Index, Mode, false, TriStorageOrder, 1, true>::kernel(
    Index size, Index otherSize, const float *_tri, Index triStride, float *_other, Index otherIncr,
    Index otherStride) {
  EIGEN_UNUSED_VARIABLE(otherIncr);
#ifdef EIGEN_RUNTIME_NO_MALLOC
  if (!is_malloc_allowed()) {
    trsmKernelL<float, Index, Mode, false, TriStorageOrder, 1, /*Specialized=*/false>::kernel(
        size, otherSize, _tri, triStride, _other, otherIncr, otherStride);
    return;
  }
#endif
  triSolve<float, TriStorageOrder == RowMajor, false, (Mode & Lower) == Lower, (Mode & UnitDiag) != 0>(
      const_cast<float *>(_tri), _other, size, otherSize, triStride, otherStride);
}

template <typename Index, int Mode, int TriStorageOrder>
EIGEN_DONT_INLINE void trsmKernelL<double, Index, Mode, false, TriStorageOrder, 1, true>::kernel(
    Index size, Index otherSize, const double *_tri, Index triStride, double *_other, Index otherIncr,
    Index otherStride) {
  EIGEN_UNUSED_VARIABLE(otherIncr);
#ifdef EIGEN_RUNTIME_NO_MALLOC
  if (!is_malloc_allowed()) {
    trsmKernelL<double, Index, Mode, false, TriStorageOrder, 1, /*Specialized=*/false>::kernel(
        size, otherSize, _tri, triStride, _other, otherIncr, otherStride);
    return;
  }
#endif
  triSolve<double, TriStorageOrder == RowMajor, false, (Mode & Lower) == Lower, (Mode & UnitDiag) != 0>(
      const_cast<double *>(_tri), _other, size, otherSize, triStride, otherStride);
}
#endif  // EIGEN_USE_AVX512_TRSM_L_KERNELS
#endif  // EIGEN_USE_AVX512_TRSM_KERNELS
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_CORE_ARCH_AVX512_TRSM_KERNEL_H
