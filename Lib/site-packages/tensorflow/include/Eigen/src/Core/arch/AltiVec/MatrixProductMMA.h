// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Everton Constantino (everton.constantino@ibm.com)
// Copyright (C) 2021 Chip Kerchner (chip.kerchner@ibm.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
#define EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H

// If using dynamic dispatch, set the CPU target.
#if defined(EIGEN_ALTIVEC_MMA_DYNAMIC_DISPATCH)
#pragma GCC push_options
#pragma GCC target("cpu=power10,htm")
#endif

#ifdef __has_builtin
#if !__has_builtin(__builtin_vsx_assemble_pair)
#define __builtin_vsx_assemble_pair __builtin_mma_assemble_pair
#endif
#if !__has_builtin(__builtin_vsx_disassemble_pair)
#define __builtin_vsx_disassemble_pair __builtin_mma_disassemble_pair
#endif
#endif

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

#include "MatrixProductMMAbfloat16.h"

namespace Eigen {

namespace internal {

#define accColsC (accCols / 2)

EIGEN_ALWAYS_INLINE void bsetzeroMMA(__vector_quad* acc) { __builtin_mma_xxsetaccz(acc); }

template <typename DataMapper, typename Packet, bool full>
EIGEN_ALWAYS_INLINE void storeAccumulator(Index i, const DataMapper& data, const Packet& alpha, const Index elements,
                                          __vector_quad* acc) {
  PacketBlock<Packet, 4> result;
  __builtin_mma_disassemble_acc(&result.packet, acc);

  PacketBlock<Packet, 4> tRes;
  if (full) {
    EIGEN_UNUSED_VARIABLE(elements);
    bload<DataMapper, Packet, 0, ColMajor, false, 4>(tRes, data, i, 0);
    bscale<Packet, 4>(tRes, result, alpha);
    bstore<DataMapper, Packet, 4>(tRes, data, i);
  } else {
    bload_partial<DataMapper, Packet, 0, false, 4>(tRes, data, i, elements);
    bscale<Packet, 4>(tRes, result, alpha);
    bstore_partial<DataMapper, Packet, 4>(tRes, data, i, elements);
  }
}

template <typename DataMapper, typename Packet, typename Packetc, const Index accCols, const Index accCols2>
EIGEN_ALWAYS_INLINE void storeComplexAccumulator(Index i, const DataMapper& data, const Packet& alphaReal,
                                                 const Packet& alphaImag, const Packet& pMask, __vector_quad* accReal,
                                                 __vector_quad* accImag) {
  constexpr bool full = (accCols2 > accColsC);
  PacketBlock<Packet, 4> resultReal, resultImag;
  __builtin_mma_disassemble_acc(&resultReal.packet, accReal);
  __builtin_mma_disassemble_acc(&resultImag.packet, accImag);

  PacketBlock<Packetc, 8> tRes;
  bload<DataMapper, Packetc, accColsC, ColMajor, true, 4, full>(tRes, data, i, 0);

  PacketBlock<Packet, 4> taccReal, taccImag;
  bscalec<Packet, 4, (accCols != accCols2)>(resultReal, resultImag, alphaReal, alphaImag, taccReal, taccImag, pMask);

  PacketBlock<Packetc, 4> acc1, acc2;
  bcouple<Packet, Packetc, 4, full>(taccReal, taccImag, tRes, acc1, acc2);

  bstore<DataMapper, Packetc, 4>(acc1, data, i);
  if (full) {
    bstore<DataMapper, Packetc, 4>(acc2, data, i + accColsC);
  }
}

// Defaults to float32, since Eigen still supports C++03 we can't use default template arguments
template <typename LhsPacket, typename RhsPacket, bool NegativeAccumulate>
EIGEN_ALWAYS_INLINE void pgerMMA(__vector_quad* acc, const RhsPacket& a, const LhsPacket& b) {
  if (NegativeAccumulate) {
    __builtin_mma_xvf32gernp(acc, (__vector unsigned char)a, (__vector unsigned char)b);
  } else {
    __builtin_mma_xvf32gerpp(acc, (__vector unsigned char)a, (__vector unsigned char)b);
  }
}

template <typename LhsPacket, typename RhsPacket, bool NegativeAccumulate>
EIGEN_ALWAYS_INLINE void pgerMMA(__vector_quad* acc, const __vector_pair& a, const Packet2d& b) {
  if (NegativeAccumulate) {
    __builtin_mma_xvf64gernp(acc, (__vector_pair)a, (__vector unsigned char)b);
  } else {
    __builtin_mma_xvf64gerpp(acc, (__vector_pair)a, (__vector unsigned char)b);
  }
}

template <typename Packet, typename RhsPacket, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void pgercMMA(__vector_quad* accReal, __vector_quad* accImag, const Packet& lhsV, Packet& lhsVi,
                                  const RhsPacket& rhsV, RhsPacket& rhsVi) {
  pgerMMA<Packet, RhsPacket, false>(accReal, rhsV, lhsV);
  if (LhsIsReal) {
    pgerMMA<Packet, RhsPacket, ConjugateRhs>(accImag, rhsVi, lhsV);
    EIGEN_UNUSED_VARIABLE(lhsVi);
  } else {
    if (!RhsIsReal) {
      pgerMMA<Packet, RhsPacket, ConjugateLhs == ConjugateRhs>(accReal, rhsVi, lhsVi);
      pgerMMA<Packet, RhsPacket, ConjugateRhs>(accImag, rhsVi, lhsV);
    } else {
      EIGEN_UNUSED_VARIABLE(rhsVi);
    }
    pgerMMA<Packet, RhsPacket, ConjugateLhs>(accImag, rhsV, lhsVi);
  }
}

// This is necessary because ploadRhs for double returns a pair of vectors when MMA is enabled.
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet ploadRhs(const __UNPACK_TYPE__(Packet) * rhs) {
  return ploadu<Packet>(rhs);
}

template <typename Scalar, typename Packet>
EIGEN_ALWAYS_INLINE void ploadRhsMMA(const Scalar* rhs, Packet& rhsV) {
  rhsV = ploadRhs<Packet>(rhs);
}

template <>
EIGEN_ALWAYS_INLINE void ploadRhsMMA(const double* rhs, __vector_pair& rhsV) {
#if EIGEN_COMP_LLVM
  __builtin_vsx_assemble_pair(
      &rhsV, reinterpret_cast<__vector unsigned char>(ploadRhs<Packet2d>(rhs + (sizeof(Packet2d) / sizeof(double)))),
      reinterpret_cast<__vector unsigned char>(ploadRhs<Packet2d>(rhs)));
#else
  rhsV = *reinterpret_cast<__vector_pair*>(const_cast<double*>(rhs));
#endif
}

EIGEN_ALWAYS_INLINE void ploadLhsMMA(const double* lhs, __vector_pair& lhsV) { ploadRhsMMA(lhs, lhsV); }

#define GEMM_MULTIPLE_COLS

// Disable in GCC until unnecessary register moves are fixed
// #if (EIGEN_COMP_LLVM || (__GNUC__ >= 11))
#if EIGEN_COMP_LLVM
#define VECTOR_PAIR_LOADS_LHS
#endif

// PEEL_MMA loop factor.
#ifdef GEMM_MULTIPLE_COLS
#define PEEL_MMA 8
#else
// Register spillage with GCC12+
#if EIGEN_COMP_LLVM || (__GNUC__ < 12) || defined(VECTOR_PAIR_LOADS_LHS)
#define PEEL_MMA 7
#else
#define PEEL_MMA 6
#endif
#endif

#define MICRO_MMA_UNROLL(func) func(0) func(1) func(2) func(3) func(4) func(5) func(6) func(7)

#define MICRO_MMA_WORK(func, type, peel)                                                                        \
  if (accItr == 1) {                                                                                            \
    func(0, type, peel, 0, 0) func(1, type, peel, 1, 0) func(2, type, peel, 2, 0) func(3, type, peel, 3, 0)     \
        func(4, type, peel, 4, 0) func(5, type, peel, 5, 0) func(6, type, peel, 6, 0) func(7, type, peel, 7, 0) \
  } else if (accItr == 2) {                                                                                     \
    func(0, type, peel, 0, 0) func(1, type, peel, 0, 1) func(2, type, peel, 1, 0) func(3, type, peel, 1, 1)     \
        func(4, type, peel, 2, 0) func(5, type, peel, 2, 1) func(6, type, peel, 3, 0) func(7, type, peel, 3, 1) \
  } else {                                                                                                      \
    func(0, type, peel, 0, 0) func(1, type, peel, 0, 1) func(2, type, peel, 0, 2) func(3, type, peel, 0, 3)     \
        func(4, type, peel, 1, 0) func(5, type, peel, 1, 1) func(6, type, peel, 1, 2) func(7, type, peel, 1, 3) \
  }

#define MICRO_MMA_WORK_ONE(iter, type, peel, left, right)                        \
  if (unroll_factor > left) {                                                    \
    pgerMMA<Packet, type, false>(&accZero##iter, rhsV##right[peel], lhsV##left); \
  }

#ifdef VECTOR_PAIR_LOADS_LHS
#define MICRO_MMA_WORK_TWO(iter, type, peel, left, right)                                          \
  if (unroll_factor > left) {                                                                      \
    pgerMMA<Packet, type, false>(&accZero##iter, rhsV##right[peel], lhsV2##left.packet[peel & 1]); \
  }

#define MICRO_MMA_LOAD1_TWO(lhs_ptr, left)                                                        \
  if (unroll_factor > left) {                                                                     \
    if (MICRO_NORMAL(left)) {                                                                     \
      ploadLhsMMA(reinterpret_cast<const double*>(lhs_ptr##left), plhsV##left);                   \
      __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(&lhsV2##left.packet), &plhsV##left); \
      lhs_ptr##left += accCols * 2;                                                               \
    } else {                                                                                      \
      lhsV2##left.packet[0] = ploadLhs<Packet>(lhs_ptr##left);                                    \
      lhsV2##left.packet[1] = ploadLhs<Packet>(lhs_ptr##left + accCols2);                         \
      lhs_ptr##left += accCols2 * 2;                                                              \
      EIGEN_UNUSED_VARIABLE(plhsV##left);                                                         \
    }                                                                                             \
  } else {                                                                                        \
    EIGEN_UNUSED_VARIABLE(lhsV2##left);                                                           \
    EIGEN_UNUSED_VARIABLE(plhsV##left);                                                           \
  }

#define MICRO_MMA_LOAD_TWO(left) MICRO_MMA_LOAD1_TWO(lhs_ptr, left)
#endif

#define MICRO_MMA_UNROLL_ITER(func, val)                       \
  func(val, 0) if (accItr > 1) {                               \
    func(val, 1) if (accItr > 2) { func(val, 2) func(val, 3) } \
  }

#define MICRO_MMA_LOAD_ONE_RHS1(peel, right) ploadRhsMMA(rhs_ptr##right + (accRows * peel), rhsV##right[peel]);

#define MICRO_MMA_LOAD_ONE_RHS(peel) MICRO_MMA_UNROLL_ITER(MICRO_MMA_LOAD_ONE_RHS1, peel)

#define MICRO_MMA_TYPE_PEEL(funcw, funcl, type, peel)              \
  if (PEEL_MMA > peel) {                                           \
    Packet lhsV0, lhsV1, lhsV2, lhsV3, lhsV4, lhsV5, lhsV6, lhsV7; \
    MICRO_MMA_LOAD_ONE_RHS(peel)                                   \
    MICRO_MMA_UNROLL(funcl)                                        \
    MICRO_MMA_WORK(funcw, type, peel)                              \
  }

#ifndef VECTOR_PAIR_LOADS_LHS
#define MICRO_MMA_UNROLL_TYPE_PEEL(funcw, funcl, type)                                                  \
  type rhsV0[8], rhsV1[(accItr > 1) ? 8 : 1], rhsV2[(accItr > 2) ? 8 : 1], rhsV3[(accItr > 2) ? 8 : 1]; \
  MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 0)                                                            \
  MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 1)                                                            \
  MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 2)                                                            \
  MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 3)                                                            \
  MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 4)                                                            \
  MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 5)                                                            \
  MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 6) MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 7)
#else
#define MICRO_MMA_LOAD_TWO_RHS(peel1, right)                                                      \
  ploadRhsMMA(reinterpret_cast<const double*>(rhs_ptr##right + (accRows * peel1)), prhsV##peel1); \
  __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(&rhsV##right[peel1]), &prhsV##peel1);

#define MICRO_MMA_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type, peel1, peel2)           \
  if (PEEL_MMA > peel2) {                                                                  \
    PacketBlock<Packet, 2> lhsV20, lhsV21, lhsV22, lhsV23, lhsV24, lhsV25, lhsV26, lhsV27; \
    __vector_pair plhsV0, plhsV1, plhsV2, plhsV3, plhsV4, plhsV5, plhsV6, plhsV7;          \
    if (sizeof(type) == 16) {                                                              \
      MICRO_MMA_UNROLL_ITER(MICRO_MMA_LOAD_TWO_RHS, peel1)                                 \
    } else {                                                                               \
      EIGEN_UNUSED_VARIABLE(prhsV##peel1);                                                 \
      MICRO_MMA_LOAD_ONE_RHS(peel1)                                                        \
      MICRO_MMA_LOAD_ONE_RHS(peel2)                                                        \
    }                                                                                      \
    MICRO_MMA_UNROLL(funcl2)                                                               \
    MICRO_MMA_WORK(funcw2, type, peel1)                                                    \
    MICRO_MMA_WORK(funcw2, type, peel2)                                                    \
  } else {                                                                                 \
    EIGEN_UNUSED_VARIABLE(prhsV##peel1);                                                   \
    MICRO_MMA_TYPE_PEEL(funcw1, funcl1, type, peel1)                                       \
  }

#define MICRO_MMA_UNROLL_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type)                               \
  type rhsV0[8], rhsV1[(accItr > 1) ? 8 : 1], rhsV2[(accItr > 2) ? 8 : 1], rhsV3[(accItr > 2) ? 8 : 1]; \
  __vector_pair prhsV0, prhsV2, prhsV4, prhsV6;                                                         \
  MICRO_MMA_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type, 0, 1)                                      \
  MICRO_MMA_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type, 2, 3)                                      \
  MICRO_MMA_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type, 4, 5)                                      \
  MICRO_MMA_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type, 6, 7)
#endif

#define MICRO_MMA_UNROLL_TYPE_ONE(funcw, funcl, type) \
  type rhsV0[1], rhsV1[1], rhsV2[1], rhsV3[1];        \
  MICRO_MMA_TYPE_PEEL(funcw, funcl, type, 0)

#define MICRO_MMA_UPDATE_RHS1(size, right) rhs_ptr##right += (accRows * size);

#define MICRO_MMA_UPDATE_RHS(size) MICRO_MMA_UNROLL_ITER(MICRO_MMA_UPDATE_RHS1, size)

#define MICRO_MMA_UNROLL_TYPE(MICRO_MMA_TYPE, size)             \
  MICRO_MMA_TYPE(MICRO_MMA_WORK_ONE, MICRO_LOAD_ONE, RhsPacket) \
  MICRO_MMA_UPDATE_RHS(size)

#ifndef VECTOR_PAIR_LOADS_LHS
#define MICRO_MMA_ONE_PEEL MICRO_MMA_UNROLL_TYPE(MICRO_MMA_UNROLL_TYPE_PEEL, PEEL_MMA)
#else
#define MICRO_MMA_UNROLL_TYPE2(MICRO_MMA_TYPE, size)                                                    \
  MICRO_MMA_TYPE(MICRO_MMA_WORK_ONE, MICRO_LOAD_ONE, MICRO_MMA_WORK_TWO, MICRO_MMA_LOAD_TWO, RhsPacket) \
  MICRO_MMA_UPDATE_RHS(size)

#define MICRO_MMA_ONE_PEEL MICRO_MMA_UNROLL_TYPE2(MICRO_MMA_UNROLL_TYPE_PEEL2, PEEL_MMA)
#endif

#define MICRO_MMA_ONE MICRO_MMA_UNROLL_TYPE(MICRO_MMA_UNROLL_TYPE_ONE, 1)

#define MICRO_MMA_DST_PTR_ONE(iter)       \
  if (unroll_factor * accItr > iter) {    \
    bsetzeroMMA(&accZero##iter);          \
  } else {                                \
    EIGEN_UNUSED_VARIABLE(accZero##iter); \
  }

#define MICRO_MMA_DST_PTR MICRO_MMA_UNROLL(MICRO_MMA_DST_PTR_ONE)

#define MICRO_MMA_SRC_PTR MICRO_MMA_UNROLL(MICRO_SRC_PTR_ONE)

#define MICRO_MMA_PREFETCH MICRO_MMA_UNROLL(MICRO_PREFETCH_ONE)

#define MICRO_MMA_STORE_ONE(iter, left, right)                                                                 \
  if (unroll_factor > left) {                                                                                  \
    storeAccumulator<DataMapper, Packet, MICRO_NORMAL_PARTIAL(left)>(row + left * accCols, res##right, pAlpha, \
                                                                     accCols2, &accZero##iter);                \
  }

#define MICRO_MMA_ITER_UNROLL(func)                                                                                 \
  if (accItr == 1) {                                                                                                \
    func(0, 0, 0) func(1, 1, 0) func(2, 2, 0) func(3, 3, 0) func(4, 4, 0) func(5, 5, 0) func(6, 6, 0) func(7, 7, 0) \
  } else if (accItr == 2) {                                                                                         \
    func(0, 0, 0) func(1, 0, 1) func(2, 1, 0) func(3, 1, 1) func(4, 2, 0) func(5, 2, 1) func(6, 3, 0) func(7, 3, 1) \
  } else {                                                                                                          \
    func(0, 0, 0) func(1, 0, 1) func(2, 0, 2) func(3, 0, 3) func(4, 1, 0) func(5, 1, 1) func(6, 1, 2) func(7, 1, 3) \
  }

#define MICRO_MMA_STORE MICRO_MMA_ITER_UNROLL(MICRO_MMA_STORE_ONE)

#define MICRO_MMA_EXTRA_ROWS(right)                                                                           \
  gemm_extra_row<Scalar, Packet, DataMapper, accRows, accCols>(                                               \
      res3##right, blockA, rhs_base + right * accRows * strideB, depth, strideA, offsetA, strideB, row, rows, \
      remaining_rows, pAlpha, pMask);

#define MICRO_MMA_EXTRA_ROWS1(val, right) MICRO_MMA_EXTRA_ROWS(right);

template <int unroll_factor, typename Scalar, typename Packet, typename RhsPacket, typename DataMapper,
          const Index accRows, const Index accCols, bool full, const Index accItr>
EIGEN_ALWAYS_INLINE void gemm_unrolled_MMA_iteration(const DataMapper& res0, const DataMapper& res1,
                                                     const DataMapper& res2, const DataMapper& res3,
                                                     const Scalar* lhs_base, const Scalar* rhs_base, Index depth,
                                                     Index strideA, Index strideB, Index offsetA, Index& row,
                                                     const Packet& pAlpha, Index accCols2) {
  const Scalar *rhs_ptr0 = rhs_base, *rhs_ptr1 = NULL, *rhs_ptr2 = NULL, *rhs_ptr3 = NULL;
  const Scalar *lhs_ptr0 = NULL, *lhs_ptr1 = NULL, *lhs_ptr2 = NULL, *lhs_ptr3 = NULL, *lhs_ptr4 = NULL,
               *lhs_ptr5 = NULL, *lhs_ptr6 = NULL, *lhs_ptr7 = NULL;
  __vector_quad accZero0, accZero1, accZero2, accZero3, accZero4, accZero5, accZero6, accZero7;

  if (accItr > 1) {
    rhs_ptr1 = rhs_base + (accRows * strideB);
  } else {
    EIGEN_UNUSED_VARIABLE(strideB);
    EIGEN_UNUSED_VARIABLE(rhs_ptr1);
    EIGEN_UNUSED_VARIABLE(res1);
  }
  if (accItr > 2) {
    rhs_ptr2 = rhs_base + (2 * accRows * strideB);
    rhs_ptr3 = rhs_base + (3 * accRows * strideB);
  } else {
    EIGEN_UNUSED_VARIABLE(rhs_ptr2);
    EIGEN_UNUSED_VARIABLE(rhs_ptr3);
    EIGEN_UNUSED_VARIABLE(res2);
    EIGEN_UNUSED_VARIABLE(res3);
  }

  MICRO_MMA_SRC_PTR
  MICRO_MMA_DST_PTR

  Index k = 0, depth2 = depth - PEEL_MMA;
  for (; k <= depth2; k += PEEL_MMA) {
    EIGEN_POWER_PREFETCH(rhs_ptr);
    MICRO_MMA_PREFETCH
    MICRO_MMA_ONE_PEEL
  }
  for (; k < depth; k++) {
    MICRO_MMA_ONE
  }
  MICRO_MMA_STORE

  MICRO_UPDATE
}

#define MICRO_MMA_UNROLL_ITER2(N, M)                                                                                 \
  gemm_unrolled_MMA_iteration<N + (M ? 1 : 0), Scalar, Packet, RhsPacket, DataMapper, accRows, accCols, !M, accItr>( \
      res30, res31, res32, res33, lhs_base, rhs_base, depth, strideA, strideB, offsetA, row, pAlpha,                 \
      M ? remaining_rows : accCols);                                                                                 \
  if (M) return;

#define MICRO_MMA_ROWS(n)             \
  while (row + n * accCols <= rows) { \
    MICRO_MMA_UNROLL_ITER2(n, 0);     \
  }

template <typename Scalar, typename Packet, typename RhsPacket, typename DataMapper, const Index accRows,
          const Index accCols, const Index accItr>
EIGEN_ALWAYS_INLINE void gemmMMA_cols(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index depth,
                                      Index strideA, Index offsetA, Index strideB, Index offsetB, Index col, Index rows,
                                      Index remaining_rows, const Packet& pAlpha, const Packet& pMask) {
  const DataMapper res30 = res.getSubMapper(0, col);
  const DataMapper res31 = (accItr > 1) ? res30.getSubMapper(0, accRows * 1) : res30;
  const DataMapper res32 = (accItr > 2) ? res30.getSubMapper(0, accRows * 2) : res30;
  const DataMapper res33 = (accItr > 2) ? res30.getSubMapper(0, accRows * 3) : res30;

  const Scalar* rhs_base = blockB + col * strideB + accRows * offsetB;
  const Scalar* lhs_base = blockA + accCols * offsetA;
  Index row = 0;

#define MAX_MMA_UNROLL 7

#if MAX_MMA_UNROLL < 2
  if (1) {
#elif MAX_MMA_UNROLL < 4
  if (accItr <= 2) {
#else
  if (accItr == 1) {
#endif
    MICRO_MMA_ROWS(MAX_MMA_UNROLL);
  } else if (accItr == 2) {
    MICRO_MMA_ROWS(4);
  } else {
    MICRO_MMA_ROWS(2);
  }
  switch ((rows - row) / accCols) {
#if MAX_MMA_UNROLL > 7
    case 7:
      if (accItr == 1) {
        MICRO_UNROLL_ITER(MICRO_MMA_UNROLL_ITER2, 7)
      }
      break;
#endif
#if MAX_MMA_UNROLL > 6
    case 6:
      if (accItr == 1) {
        MICRO_UNROLL_ITER(MICRO_MMA_UNROLL_ITER2, 6)
      }
      break;
#endif
#if MAX_MMA_UNROLL > 5
    case 5:
      if (accItr == 1) {
        MICRO_UNROLL_ITER(MICRO_MMA_UNROLL_ITER2, 5)
      }
      break;
#endif
#if MAX_MMA_UNROLL > 4
    case 4:
      if (accItr == 1) {
        MICRO_UNROLL_ITER(MICRO_MMA_UNROLL_ITER2, 4)
      }
      break;
#endif
#if MAX_MMA_UNROLL > 3
    case 3:
      if (accItr <= 2) {
        MICRO_UNROLL_ITER(MICRO_MMA_UNROLL_ITER2, 3)
      }
      break;
#endif
#if MAX_MMA_UNROLL > 2
    case 2:
      if (accItr <= 2) {
        MICRO_UNROLL_ITER(MICRO_MMA_UNROLL_ITER2, 2)
      }
      break;
#endif
#if MAX_MMA_UNROLL > 1
    case 1:
      MICRO_UNROLL_ITER(MICRO_MMA_UNROLL_ITER2, 1)
      break;
#endif
    default:
      break;
  }
#undef MAX_MMA_UNROLL

  if (remaining_rows > 0) {
    MICRO_MMA_UNROLL_ITER(MICRO_MMA_EXTRA_ROWS1, 0)
  }
}

#define MICRO_MMA_COLS(n)                                                                                          \
  for (; col + n * accRows <= cols; col += n * accRows) {                                                          \
    gemmMMA_cols<Scalar, Packet, RhsPacket2, DataMapper, accRows, accCols, n>(                                     \
        res, blockA, blockB, depth, strideA, offsetA, strideB, offsetB, col, rows, remaining_rows, pAlpha, pMask); \
  }

template <typename Scalar, typename Packet, typename RhsPacket, typename DataMapper, const Index accRows,
          const Index accCols>
void gemmMMA(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows, Index depth, Index cols,
             Scalar alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  const Index remaining_rows = rows % accCols;

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;

  const Packet pAlpha = pset1<Packet>(alpha);
  const Packet pMask = bmask<Packet>(remaining_rows);

  typedef typename std::conditional_t<(sizeof(Scalar) == sizeof(float)), RhsPacket, __vector_pair> RhsPacket2;

  Index col = 0;
#ifdef GEMM_MULTIPLE_COLS
  MICRO_MMA_COLS(4);
  MICRO_MMA_COLS(2);
#endif
  MICRO_MMA_COLS(1);

  if (col != cols) {
    gemm_extra_cols<Scalar, Packet, DataMapper, accCols>(res, blockA, blockB, depth, strideA, offsetA, strideB, offsetB,
                                                         col, rows, cols, remaining_rows, pAlpha, pMask);
  }
}

#define advanceRows ((LhsIsReal) ? 1 : 2)
#define advanceCols ((RhsIsReal) ? 1 : 2)

// PEEL_COMPLEX_MMA loop factor.
#ifdef GEMM_MULTIPLE_COLS
#define PEEL_COMPLEX_MMA 4
#else
#define PEEL_COMPLEX_MMA 3
#endif

#define MICRO_COMPLEX_MMA_UNROLL(func) func(0) func(1) func(2) func(3)

#define MICRO_COMPLEX_MMA_WORK(func, type, peel)                                                            \
  if (accItr == 1) {                                                                                        \
    func(0, type, peel, 0, 0) func(1, type, peel, 1, 0) func(2, type, peel, 2, 0) func(3, type, peel, 3, 0) \
  } else if (accItr == 2) {                                                                                 \
    func(0, type, peel, 0, 0) func(1, type, peel, 0, 1) func(2, type, peel, 1, 0) func(3, type, peel, 1, 1) \
  } else {                                                                                                  \
    func(0, type, peel, 0, 0) func(1, type, peel, 0, 1) func(2, type, peel, 0, 2) func(3, type, peel, 0, 3) \
  }

#define MICRO_COMPLEX_MMA_WORK_ONE(iter, type, peel, left, right)                                        \
  if (unroll_factor > left) {                                                                            \
    pgercMMA<Packet, type, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal>(                            \
        &accReal##iter, &accImag##iter, lhsV##left, lhsVi##left, rhsV##right[peel], rhsVi##right[peel]); \
  }

#ifdef VECTOR_PAIR_LOADS_LHS
#define MICRO_COMPLEX_MMA_WORK_TWO(iter, type, peel, left, right)                                    \
  if (unroll_factor > left) {                                                                        \
    pgercMMA<Packet, type, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal>(                        \
        &accReal##iter, &accImag##iter, lhsV2##left.packet[peel & 1], lhsVi2##left.packet[peel & 1], \
        rhsV##right[peel], rhsVi##right[peel]);                                                      \
  }

#define MICRO_COMPLEX_MMA_LOAD1_TWO(lhs_ptr, left)                                                  \
  if (!LhsIsReal && (unroll_factor > left)) {                                                       \
    if (MICRO_NORMAL(left)) {                                                                       \
      ploadLhsMMA(reinterpret_cast<const double*>(lhs_ptr_real##left + imag_delta), plhsVi##left);  \
      __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(&lhsVi2##left.packet), &plhsVi##left); \
    } else {                                                                                        \
      lhsVi2##left.packet[0] = ploadLhs<Packet>(lhs_ptr_real##left + imag_delta2);                  \
      lhsVi2##left.packet[1] = ploadLhs<Packet>(lhs_ptr_real##left + imag_delta2 + accCols2);       \
      EIGEN_UNUSED_VARIABLE(plhsVi##left);                                                          \
    }                                                                                               \
  } else {                                                                                          \
    EIGEN_UNUSED_VARIABLE(lhsVi2##left);                                                            \
    EIGEN_UNUSED_VARIABLE(plhsVi##left);                                                            \
  }                                                                                                 \
  MICRO_MMA_LOAD1_TWO(lhs_ptr_real, left)

#define MICRO_COMPLEX_MMA_LOAD_TWO(left) MICRO_COMPLEX_MMA_LOAD1_TWO(lhs_ptr, left)
#endif

#define MICRO_COMPLEX_MMA_LOAD_RHS1(peel, right)                             \
  ploadRhsMMA(rhs_ptr_real##right + (accRows * peel), rhsV##right[peel]);    \
  if (!RhsIsReal) {                                                          \
    ploadRhsMMA(rhs_ptr_imag##right + (accRows * peel), rhsVi##right[peel]); \
  }

#define MICRO_COMPLEX_MMA_LOAD_ONE_RHS(peel) MICRO_MMA_UNROLL_ITER(MICRO_COMPLEX_MMA_LOAD_RHS1, peel)

#define MICRO_COMPLEX_MMA_TYPE_PEEL(funcw, funcl, type, peel) \
  if (PEEL_COMPLEX_MMA > peel) {                              \
    Packet lhsV0, lhsV1, lhsV2, lhsV3;                        \
    Packet lhsVi0, lhsVi1, lhsVi2, lhsVi3;                    \
    MICRO_COMPLEX_MMA_LOAD_ONE_RHS(peel)                      \
    MICRO_COMPLEX_MMA_UNROLL(funcl)                           \
    MICRO_COMPLEX_MMA_WORK(funcw, type, peel)                 \
  }

#ifndef VECTOR_PAIR_LOADS_LHS
#define MICRO_COMPLEX_MMA_UNROLL_TYPE_PEEL(funcw, funcl, type)                                                      \
  type rhsV0[4], rhsVi0[4], rhsV1[(accItr > 1) ? 4 : 1], rhsVi1[(accItr > 1) ? 4 : 1], rhsV2[(accItr > 2) ? 4 : 1], \
      rhsVi2[(accItr > 2) ? 4 : 1], rhsV3[(accItr > 2) ? 4 : 1], rhsVi3[(accItr > 2) ? 4 : 1];                      \
  MICRO_COMPLEX_MMA_TYPE_PEEL(funcw, funcl, type, 0)                                                                \
  MICRO_COMPLEX_MMA_TYPE_PEEL(funcw, funcl, type, 1)                                                                \
  MICRO_COMPLEX_MMA_TYPE_PEEL(funcw, funcl, type, 2) MICRO_COMPLEX_MMA_TYPE_PEEL(funcw, funcl, type, 3)
#else
#define MICRO_COMPLEX_MMA_LOAD_TWO_RHS(peel1, right)                                                      \
  ploadRhsMMA(reinterpret_cast<const double*>(rhs_ptr_real##right + (accRows * peel1)), prhsV##peel1);    \
  __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(&rhsV##right[peel1]), &prhsV##peel1);            \
  if (!RhsIsReal) {                                                                                       \
    ploadRhsMMA(reinterpret_cast<const double*>(rhs_ptr_imag##right + (accRows * peel1)), prhsVi##peel1); \
    __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(&rhsVi##right[peel1]), &prhsVi##peel1);        \
  } else {                                                                                                \
    EIGEN_UNUSED_VARIABLE(prhsVi##peel1);                                                                 \
  }

#define MICRO_COMPLEX_MMA_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type, peel1, peel2) \
  if (PEEL_COMPLEX_MMA > peel2) {                                                        \
    PacketBlock<Packet, 2> lhsV20, lhsV21, lhsV22, lhsV23;                               \
    PacketBlock<Packet, 2> lhsVi20, lhsVi21, lhsVi22, lhsVi23;                           \
    __vector_pair plhsV0, plhsV1, plhsV2, plhsV3;                                        \
    __vector_pair plhsVi0, plhsVi1, plhsVi2, plhsVi3;                                    \
    if (sizeof(type) == 16) {                                                            \
      MICRO_MMA_UNROLL_ITER(MICRO_COMPLEX_MMA_LOAD_TWO_RHS, peel1)                       \
    } else {                                                                             \
      EIGEN_UNUSED_VARIABLE(prhsV##peel1);                                               \
      EIGEN_UNUSED_VARIABLE(prhsVi##peel1);                                              \
      MICRO_COMPLEX_MMA_LOAD_ONE_RHS(peel1);                                             \
      MICRO_COMPLEX_MMA_LOAD_ONE_RHS(peel2);                                             \
    }                                                                                    \
    MICRO_COMPLEX_MMA_UNROLL(funcl2)                                                     \
    MICRO_COMPLEX_MMA_WORK(funcw2, type, peel1)                                          \
    MICRO_COMPLEX_MMA_WORK(funcw2, type, peel2)                                          \
  } else {                                                                               \
    EIGEN_UNUSED_VARIABLE(prhsV##peel1);                                                 \
    EIGEN_UNUSED_VARIABLE(prhsVi##peel1);                                                \
    MICRO_COMPLEX_MMA_TYPE_PEEL(funcw1, funcl1, type, peel1)                             \
  }

#define MICRO_COMPLEX_MMA_UNROLL_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type)                                   \
  type rhsV0[4], rhsVi0[4], rhsV1[(accItr > 1) ? 4 : 1], rhsVi1[(accItr > 1) ? 4 : 1], rhsV2[(accItr > 2) ? 4 : 1], \
      rhsVi2[(accItr > 2) ? 4 : 1], rhsV3[(accItr > 2) ? 4 : 1], rhsVi3[(accItr > 2) ? 4 : 1];                      \
  __vector_pair prhsV0, prhsV2;                                                                                     \
  __vector_pair prhsVi0, prhsVi2;                                                                                   \
  MICRO_COMPLEX_MMA_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type, 0, 1)                                          \
  MICRO_COMPLEX_MMA_TYPE_PEEL2(funcw1, funcl1, funcw2, funcl2, type, 2, 3)
#endif

#define MICRO_COMPLEX_MMA_UNROLL_TYPE_ONE(funcw, funcl, type)                              \
  type rhsV0[1], rhsVi0[1], rhsV1[1], rhsVi1[1], rhsV2[1], rhsVi2[1], rhsV3[1], rhsVi3[1]; \
  MICRO_COMPLEX_MMA_TYPE_PEEL(funcw, funcl, type, 0)

#define MICRO_COMPLEX_MMA_UPDATE_RHS1(size, right) \
  rhs_ptr_real##right += (accRows * size);         \
  if (!RhsIsReal) rhs_ptr_imag##right += (accRows * size);

#define MICRO_COMPLEX_MMA_UPDATE_RHS(size) MICRO_MMA_UNROLL_ITER(MICRO_COMPLEX_MMA_UPDATE_RHS1, size)

#define MICRO_COMPLEX_MMA_UNROLL_TYPE(MICRO_COMPLEX_MMA_TYPE, size)                     \
  MICRO_COMPLEX_MMA_TYPE(MICRO_COMPLEX_MMA_WORK_ONE, MICRO_COMPLEX_LOAD_ONE, RhsPacket) \
  MICRO_COMPLEX_MMA_UPDATE_RHS(size);

#ifndef VECTOR_PAIR_LOADS_LHS
#define MICRO_COMPLEX_MMA_ONE_PEEL MICRO_COMPLEX_MMA_UNROLL_TYPE(MICRO_COMPLEX_MMA_UNROLL_TYPE_PEEL, PEEL_COMPLEX_MMA)
#else
#define MICRO_COMPLEX_MMA_UNROLL_TYPE2(MICRO_COMPLEX_MMA_TYPE, size)                                     \
  MICRO_COMPLEX_MMA_TYPE(MICRO_COMPLEX_MMA_WORK_ONE, MICRO_COMPLEX_LOAD_ONE, MICRO_COMPLEX_MMA_WORK_TWO, \
                         MICRO_COMPLEX_MMA_LOAD_TWO, RhsPacket)                                          \
  MICRO_COMPLEX_MMA_UPDATE_RHS(size);

#define MICRO_COMPLEX_MMA_ONE_PEEL MICRO_COMPLEX_MMA_UNROLL_TYPE2(MICRO_COMPLEX_MMA_UNROLL_TYPE_PEEL2, PEEL_COMPLEX_MMA)
#endif

#define MICRO_COMPLEX_MMA_ONE MICRO_COMPLEX_MMA_UNROLL_TYPE(MICRO_COMPLEX_MMA_UNROLL_TYPE_ONE, 1)

#define MICRO_COMPLEX_MMA_DST_PTR_ONE(iter) \
  if (unroll_factor * accItr > iter) {      \
    bsetzeroMMA(&accReal##iter);            \
    bsetzeroMMA(&accImag##iter);            \
  } else {                                  \
    EIGEN_UNUSED_VARIABLE(accReal##iter);   \
    EIGEN_UNUSED_VARIABLE(accImag##iter);   \
  }

#define MICRO_COMPLEX_MMA_DST_PTR MICRO_COMPLEX_MMA_UNROLL(MICRO_COMPLEX_MMA_DST_PTR_ONE)

#define MICRO_COMPLEX_MMA_SRC_PTR MICRO_COMPLEX_MMA_UNROLL(MICRO_COMPLEX_SRC_PTR_ONE)

#define MICRO_COMPLEX_MMA_PREFETCH MICRO_COMPLEX_MMA_UNROLL(MICRO_COMPLEX_PREFETCH_ONE)

#define MICRO_COMPLEX_MMA_STORE_ONE(iter, left, right)                                                                 \
  if (unroll_factor > left) {                                                                                          \
    storeComplexAccumulator<DataMapper, Packet, Packetc, accCols, (unroll_factor != (left + 1)) ? accCols : accCols2>( \
        row + left * accCols, res##right, pAlphaReal, pAlphaImag, pMask, &accReal##iter, &accImag##iter);              \
  }

#define MICRO_COMPLEX_MMA_ITER_UNROLL(func)                 \
  if (accItr == 1) {                                        \
    func(0, 0, 0) func(1, 1, 0) func(2, 2, 0) func(3, 3, 0) \
  } else if (accItr == 2) {                                 \
    func(0, 0, 0) func(1, 0, 1) func(2, 1, 0) func(3, 1, 1) \
  } else {                                                  \
    func(0, 0, 0) func(1, 0, 1) func(2, 0, 2) func(3, 0, 3) \
  }

#define MICRO_COMPLEX_MMA_STORE MICRO_COMPLEX_MMA_ITER_UNROLL(MICRO_COMPLEX_MMA_STORE_ONE)

#define MICRO_COMPLEX_MMA_EXTRA_ROWS(right)                                                                            \
  gemm_complex_extra_row<Scalar, Packet, Packetc, DataMapper, accRows, accCols, ConjugateLhs, ConjugateRhs, LhsIsReal, \
                         RhsIsReal>(res3##right, blockA, rhs_base + right * accRows * (RhsIsReal ? 1 : 2) * strideB,   \
                                    depth, strideA, offsetA, strideB, row, rows, remaining_rows, pAlphaReal,           \
                                    pAlphaImag, pMask);

#define MICRO_COMPLEX_MMA_EXTRA_ROWS1(val, right) MICRO_COMPLEX_MMA_EXTRA_ROWS(right);

template <int unroll_factor, typename Scalar, typename Packet, typename Packetc, typename RhsPacket,
          typename DataMapper, const Index accRows, const Index accCols, const Index accCols2, bool ConjugateLhs,
          bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal, const Index accItr>
EIGEN_ALWAYS_INLINE void gemm_complex_unrolled_MMA_iteration(const DataMapper& res0, const DataMapper& res1,
                                                             const DataMapper& res2, const DataMapper& res3,
                                                             const Scalar* lhs_base, const Scalar* rhs_base,
                                                             Index depth, Index strideA, Index offsetA, Index strideB,
                                                             Index& row, const Packet& pAlphaReal,
                                                             const Packet& pAlphaImag, const Packet& pMask) {
  const Scalar *rhs_ptr_real0 = rhs_base, *rhs_ptr_real1 = NULL, *rhs_ptr_real2 = NULL, *rhs_ptr_real3 = NULL;
  const Scalar *rhs_ptr_imag0 = NULL, *rhs_ptr_imag1 = NULL, *rhs_ptr_imag2 = NULL, *rhs_ptr_imag3 = NULL;
  const Index imag_delta = accCols * strideA;
  const Index imag_delta2 = accCols2 * strideA;

  if (!RhsIsReal) {
    rhs_ptr_imag0 = rhs_base + accRows * strideB;
  } else {
    EIGEN_UNUSED_VARIABLE(rhs_ptr_imag0);
  }
  if (accItr > 1) {
    if (!RhsIsReal) {
      rhs_ptr_real1 = rhs_base + (2 * accRows * strideB);
      rhs_ptr_imag1 = rhs_base + (3 * accRows * strideB);
    } else {
      rhs_ptr_real1 = rhs_base + accRows * strideB;
      EIGEN_UNUSED_VARIABLE(rhs_ptr_imag1);
    }
  } else {
    EIGEN_UNUSED_VARIABLE(rhs_ptr_real1);
    EIGEN_UNUSED_VARIABLE(rhs_ptr_imag1);
    EIGEN_UNUSED_VARIABLE(res1);
  }
  if (accItr > 2) {
    if (!RhsIsReal) {
      rhs_ptr_real2 = rhs_base + (4 * accRows * strideB);
      rhs_ptr_imag2 = rhs_base + (5 * accRows * strideB);
      rhs_ptr_real3 = rhs_base + (6 * accRows * strideB);
      rhs_ptr_imag3 = rhs_base + (7 * accRows * strideB);
    } else {
      rhs_ptr_real2 = rhs_base + (2 * accRows * strideB);
      rhs_ptr_real3 = rhs_base + (3 * accRows * strideB);
      EIGEN_UNUSED_VARIABLE(rhs_ptr_imag2);
      EIGEN_UNUSED_VARIABLE(rhs_ptr_imag3);
    }
  } else {
    EIGEN_UNUSED_VARIABLE(rhs_ptr_real2);
    EIGEN_UNUSED_VARIABLE(rhs_ptr_real3);
    EIGEN_UNUSED_VARIABLE(rhs_ptr_imag2);
    EIGEN_UNUSED_VARIABLE(rhs_ptr_imag3);
    EIGEN_UNUSED_VARIABLE(res2);
    EIGEN_UNUSED_VARIABLE(res3);
  }
  const Scalar *lhs_ptr_real0 = NULL, *lhs_ptr_real1 = NULL;
  const Scalar *lhs_ptr_real2 = NULL, *lhs_ptr_real3 = NULL;
  __vector_quad accReal0, accImag0, accReal1, accImag1, accReal2, accImag2, accReal3, accImag3;

  MICRO_COMPLEX_MMA_SRC_PTR
  MICRO_COMPLEX_MMA_DST_PTR

  Index k = 0, depth2 = depth - PEEL_COMPLEX_MMA;
  for (; k <= depth2; k += PEEL_COMPLEX_MMA) {
    EIGEN_POWER_PREFETCH(rhs_ptr_real);
    if (!RhsIsReal) {
      EIGEN_POWER_PREFETCH(rhs_ptr_imag);
    }
    MICRO_COMPLEX_MMA_PREFETCH
    MICRO_COMPLEX_MMA_ONE_PEEL
  }
  for (; k < depth; k++) {
    MICRO_COMPLEX_MMA_ONE
  }
  MICRO_COMPLEX_MMA_STORE

  MICRO_COMPLEX_UPDATE
}

#define MICRO_COMPLEX_MMA_UNROLL_ITER2(N, M)                                                                           \
  gemm_complex_unrolled_MMA_iteration<N + (M ? 1 : 0), Scalar, Packet, Packetc, RhsPacket, DataMapper, accRows,        \
                                      accCols, M ? M : accCols, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal,      \
                                      accItr>(res30, res31, res32, res33, lhs_base, rhs_base, depth, strideA, offsetA, \
                                              strideB, row, pAlphaReal, pAlphaImag, pMask);                            \
  if (M) return;

#define MICRO_COMPLEX_MMA_ROWS(n)         \
  while (row + n * accCols <= rows) {     \
    MICRO_COMPLEX_MMA_UNROLL_ITER2(n, 0); \
  }

template <typename Scalar, typename Packet, typename Packetc, typename RhsPacket, typename DataMapper,
          const Index accRows, const Index accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal,
          bool RhsIsReal, const Index accItr>
EIGEN_ALWAYS_INLINE void gemmMMA_complex_cols(const DataMapper& res, const Scalar* blockA, const Scalar* blockB,
                                              Index depth, Index strideA, Index offsetA, Index strideB, Index offsetB,
                                              Index col, Index rows, Index remaining_rows, const Packet& pAlphaReal,
                                              const Packet& pAlphaImag, const Packet& pMask) {
  const DataMapper res30 = res.getSubMapper(0, col);
  const DataMapper res31 = (accItr > 1) ? res30.getSubMapper(0, accRows * 1) : res30;
  const DataMapper res32 = (accItr > 2) ? res30.getSubMapper(0, accRows * 2) : res30;
  const DataMapper res33 = (accItr > 2) ? res30.getSubMapper(0, accRows * 3) : res30;

  const Scalar* rhs_base = blockB + advanceCols * col * strideB + accRows * offsetB;
  const Scalar* lhs_base = blockA + accCols * offsetA;
  Index row = 0;

#define MAX_COMPLEX_MMA_UNROLL 4

#if MAX_COMPLEX_MMA_UNROLL < 2
  if (1) {
#elif MAX_COMPLEX_MMA_UNROLL < 4
  if (accItr <= 2) {
#else
  if (accItr == 1) {
#endif
    MICRO_COMPLEX_MMA_ROWS(MAX_COMPLEX_MMA_UNROLL);
  } else if (accItr == 2) {
    MICRO_COMPLEX_MMA_ROWS(2);
  } else {
    MICRO_COMPLEX_MMA_ROWS(1);
  }
  switch ((rows - row) / accCols) {
#if MAX_COMPLEX_MMA_UNROLL > 3
    case 3:
      if (accItr == 1) {
        MICRO_COMPLEX_UNROLL_ITER(MICRO_COMPLEX_MMA_UNROLL_ITER2, 3)
      }
      break;
#endif
#if MAX_COMPLEX_MMA_UNROLL > 2
    case 2:
      if (accItr == 1) {
        MICRO_COMPLEX_UNROLL_ITER(MICRO_COMPLEX_MMA_UNROLL_ITER2, 2)
      }
      break;
#endif
#if MAX_COMPLEX_MMA_UNROLL > 1
    case 1:
      if (accItr <= 2) {
        MICRO_COMPLEX_UNROLL_ITER(MICRO_COMPLEX_MMA_UNROLL_ITER2, 1)
      }
      break;
#endif
    default:
      break;
  }
#undef MAX_COMPLEX_MMA_UNROLL

  if (remaining_rows > 0) {
    MICRO_MMA_UNROLL_ITER(MICRO_COMPLEX_MMA_EXTRA_ROWS1, 0)
  }
}

#define MICRO_COMPLEX_MMA_COLS(n)                                                                                      \
  for (; col + n * accRows <= cols; col += n * accRows) {                                                              \
    gemmMMA_complex_cols<Scalar, Packet, Packetc, RhsPacket2, DataMapper, accRows, accCols, ConjugateLhs,              \
                         ConjugateRhs, LhsIsReal, RhsIsReal, n>(res, blockA, blockB, depth, strideA, offsetA, strideB, \
                                                                offsetB, col, rows, remaining_rows, pAlphaReal,        \
                                                                pAlphaImag, pMask);                                    \
  }

template <typename LhsScalar, typename RhsScalar, typename Scalarc, typename Scalar, typename Packet, typename Packetc,
          typename RhsPacket, typename DataMapper, const Index accRows, const Index accCols, bool ConjugateLhs,
          bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
void gemm_complexMMA(const DataMapper& res, const LhsScalar* blockAc, const RhsScalar* blockBc, Index rows, Index depth,
                     Index cols, Scalarc alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  const Index remaining_rows = rows % accCols;

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;

  const Packet pAlphaReal = pset1<Packet>(alpha.real());
  const Packet pAlphaImag = pset1<Packet>(alpha.imag());
  const Packet pMask = bmask<Packet>(remaining_rows);

  const Scalar* blockA = (Scalar*)blockAc;
  const Scalar* blockB = (Scalar*)blockBc;

  typedef typename std::conditional_t<(sizeof(Scalar) == sizeof(float)), RhsPacket, __vector_pair> RhsPacket2;

  Index col = 0;
#ifdef GEMM_MULTIPLE_COLS
  MICRO_COMPLEX_MMA_COLS(4);
  MICRO_COMPLEX_MMA_COLS(2);
#endif
  MICRO_COMPLEX_MMA_COLS(1);

  if (col != cols) {
    gemm_complex_extra_cols<Scalar, Packet, Packetc, DataMapper, accCols, ConjugateLhs, ConjugateRhs, LhsIsReal,
                            RhsIsReal>(res, blockA, blockB, depth, strideA, offsetA, strideB, offsetB, col, rows, cols,
                                       remaining_rows, pAlphaReal, pAlphaImag, pMask);
  }
}

#undef accColsC
#undef advanceRows
#undef advanceCols

}  // end namespace internal

}  // end namespace Eigen

#if defined(EIGEN_ALTIVEC_MMA_DYNAMIC_DISPATCH)
#pragma GCC pop_options
#endif

#endif  // EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
