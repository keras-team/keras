// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Chip Kerchner (chip.kerchner@ibm.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_VECTOR_PRODUCT_ALTIVEC_H
#define EIGEN_MATRIX_VECTOR_PRODUCT_ALTIVEC_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

#if defined(__MMA__) && !EIGEN_ALTIVEC_DISABLE_MMA
#if EIGEN_COMP_LLVM || (__GNUC__ > 10 || __GNUC_MINOR__ >= 3)
#define USE_GEMV_MMA
#endif

#if !EIGEN_COMP_LLVM && (__GNUC__ < 11)
// Only allow one vector_pair in buggy gcc - gcc 10.x has a bug
#define GCC_ONE_VECTORPAIR_BUG
#endif
#endif

// #define USE_SLOWER_GEMV_MMA   // MMA is currently not as fast as VSX in complex double GEMV (revisit when gcc is
// improved)

// #define EIGEN_POWER_USE_GEMV_PREFETCH
#ifdef EIGEN_POWER_USE_GEMV_PREFETCH
#define EIGEN_POWER_GEMV_PREFETCH(p) prefetch(p)
#else
#define EIGEN_POWER_GEMV_PREFETCH(p)
#endif

#ifdef __has_builtin
#if !__has_builtin(__builtin_vsx_assemble_pair)
#define __builtin_vsx_assemble_pair __builtin_mma_assemble_pair
#endif
#if !__has_builtin(__builtin_vsx_disassemble_pair)
#define __builtin_vsx_disassemble_pair __builtin_mma_disassemble_pair
#endif
#endif

#if EIGEN_COMP_LLVM
#define GEMV_BUILDPAIR_MMA(dst, src1, src2) \
  __builtin_vsx_assemble_pair(&dst, (__vector unsigned char)src2, (__vector unsigned char)src1)
#else
#if (__GNUC__ <= 10)
#if (__GNUC_MINOR__ > 3)
#define GEMV_BUILDPAIR_MMA(dst, src1, src2) \
  __builtin_vsx_assemble_pair(&dst, (__vector unsigned char)src2, (__vector unsigned char)src1)
#else
#define GEMV_BUILDPAIR_MMA(dst, src1, src2) \
  __builtin_vsx_assemble_pair(&dst, (__vector unsigned char)src1, (__vector unsigned char)src2)
#endif
#else
#define GEMV_BUILDPAIR_MMA(dst, src1, src2) \
  __builtin_vsx_build_pair(&dst, (__vector unsigned char)src1, (__vector unsigned char)src2)
#endif
#endif

#define GEMV_IS_COMPLEX_COMPLEX ((sizeof(LhsPacket) == 16) && (sizeof(RhsPacket) == 16))
#define GEMV_IS_FLOAT (ResPacketSize == (16 / sizeof(float)))
#define GEMV_IS_SCALAR (sizeof(ResPacket) != 16)
#define GEMV_IS_COMPLEX_FLOAT (ResPacketSize == (16 / sizeof(std::complex<float>)))

/** \internal multiply and add and store results */
template <typename ResPacket, typename ResScalar>
EIGEN_ALWAYS_INLINE void storeMaddData(ResScalar* res, ResPacket& palpha, ResPacket& data) {
  pstoreu(res, pmadd(data, palpha, ploadu<ResPacket>(res)));
}

template <typename ResScalar>
EIGEN_ALWAYS_INLINE void storeMaddData(ResScalar* res, ResScalar& alpha, ResScalar& data) {
  *res += (alpha * data);
}

#define GEMV_UNROLL(func, N) func(0, N) func(1, N) func(2, N) func(3, N) func(4, N) func(5, N) func(6, N) func(7, N)

#define GEMV_UNROLL_HALF(func, N) func(0, 0, 1, N) func(1, 2, 3, N) func(2, 4, 5, N) func(3, 6, 7, N)

#define GEMV_GETN(N) (((N) * ResPacketSize) >> 2)

#define GEMV_LOADPACKET_COL(iter) lhs.template load<LhsPacket, LhsAlignment>(i + ((iter) * LhsPacketSize), j)

#ifdef USE_GEMV_MMA
#define GEMV_UNROLL3(func, N, which)                                                                          \
  func(0, N, which) func(1, N, which) func(2, N, which) func(3, N, which) func(4, N, which) func(5, N, which) \
      func(6, N, which) func(7, N, which)

#define GEMV_UNUSED_VAR(iter, N, which) \
  if (GEMV_GETN(N) <= iter) {           \
    EIGEN_UNUSED_VARIABLE(which##iter); \
  }

#define GEMV_UNUSED_EXTRA_VAR(iter, N, which) \
  if (N <= iter) {                            \
    EIGEN_UNUSED_VARIABLE(which##iter);       \
  }

#define GEMV_UNUSED_EXTRA(N, which) GEMV_UNROLL3(GEMV_UNUSED_EXTRA_VAR, N, which)

#define GEMV_UNUSED(N, which) GEMV_UNROLL3(GEMV_UNUSED_VAR, N, which)

#define GEMV_INIT_MMA(iter, N)         \
  if (GEMV_GETN(N) > iter) {           \
    __builtin_mma_xxsetaccz(&e##iter); \
  }

#if EIGEN_COMP_LLVM
#define GEMV_LOADPAIR_COL_MMA(iter1, iter2) \
  GEMV_BUILDPAIR_MMA(b##iter1, GEMV_LOADPACKET_COL(iter2), GEMV_LOADPACKET_COL((iter2) + 1));
#else
#define GEMV_LOADPAIR_COL_MMA(iter1, iter2)                                     \
  const LhsScalar& src##iter1 = lhs(i + ((iter1 * 32) / sizeof(LhsScalar)), j); \
  b##iter1 = *reinterpret_cast<__vector_pair*>(const_cast<LhsScalar*>(&src##iter1));
#endif

#define GEMV_LOAD1A_COL_MMA(iter, N)         \
  if (GEMV_GETN(N) > iter) {                 \
    if (GEMV_IS_FLOAT) {                     \
      g##iter = GEMV_LOADPACKET_COL(iter);   \
      EIGEN_UNUSED_VARIABLE(b##iter);        \
    } else {                                 \
      GEMV_LOADPAIR_COL_MMA(iter, iter << 1) \
      EIGEN_UNUSED_VARIABLE(g##iter);        \
    }                                        \
  } else {                                   \
    EIGEN_UNUSED_VARIABLE(b##iter);          \
    EIGEN_UNUSED_VARIABLE(g##iter);          \
  }

#define GEMV_WORK1A_COL_MMA(iter, N)                                      \
  if (GEMV_GETN(N) > iter) {                                              \
    if (GEMV_IS_FLOAT) {                                                  \
      pger_vecMMA_acc<LhsPacket, RhsPacket, true>(&e##iter, a0, g##iter); \
    } else {                                                              \
      pger_vecMMA_acc<LhsPacket, RhsPacket, true>(&e##iter, b##iter, a0); \
    }                                                                     \
  }

#define GEMV_LOAD1B_COL_MMA(iter1, iter2, iter3, N) \
  if (GEMV_GETN(N) > iter1) {                       \
    if (GEMV_IS_FLOAT) {                            \
      GEMV_LOADPAIR_COL_MMA(iter2, iter2)           \
      EIGEN_UNUSED_VARIABLE(b##iter3);              \
    } else {                                        \
      GEMV_LOADPAIR_COL_MMA(iter2, iter2 << 1)      \
      GEMV_LOADPAIR_COL_MMA(iter3, iter3 << 1)      \
    }                                               \
  } else {                                          \
    EIGEN_UNUSED_VARIABLE(b##iter2);                \
    EIGEN_UNUSED_VARIABLE(b##iter3);                \
  }                                                 \
  EIGEN_UNUSED_VARIABLE(g##iter2);                  \
  EIGEN_UNUSED_VARIABLE(g##iter3);

#define GEMV_WORK1B_COL_MMA(iter1, iter2, iter3, N)                          \
  if (GEMV_GETN(N) > iter1) {                                                \
    if (GEMV_IS_FLOAT) {                                                     \
      LhsPacket h[2];                                                        \
      __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(h), &b##iter2); \
      pger_vecMMA_acc<LhsPacket, RhsPacket, true>(&e##iter2, a0, h[0]);      \
      pger_vecMMA_acc<LhsPacket, RhsPacket, true>(&e##iter3, a0, h[1]);      \
    } else {                                                                 \
      pger_vecMMA_acc<LhsPacket, RhsPacket, true>(&e##iter2, b##iter2, a0);  \
      pger_vecMMA_acc<LhsPacket, RhsPacket, true>(&e##iter3, b##iter3, a0);  \
    }                                                                        \
  }

#if EIGEN_COMP_LLVM
#define GEMV_LOAD_COL_MMA(N)                        \
  if (GEMV_GETN(N) > 1) {                           \
    GEMV_UNROLL_HALF(GEMV_LOAD1B_COL_MMA, (N >> 1)) \
  } else {                                          \
    GEMV_UNROLL(GEMV_LOAD1A_COL_MMA, N)             \
  }

#define GEMV_WORK_COL_MMA(N)                        \
  if (GEMV_GETN(N) > 1) {                           \
    GEMV_UNROLL_HALF(GEMV_WORK1B_COL_MMA, (N >> 1)) \
  } else {                                          \
    GEMV_UNROLL(GEMV_WORK1A_COL_MMA, N)             \
  }
#else
#define GEMV_LOAD_COL_MMA(N) GEMV_UNROLL(GEMV_LOAD1A_COL_MMA, N)

#define GEMV_WORK_COL_MMA(N) GEMV_UNROLL(GEMV_WORK1A_COL_MMA, N)
#endif

#define GEMV_DISASSEMBLE_MMA(iter, N)                              \
  if (GEMV_GETN(N) > iter) {                                       \
    __builtin_mma_disassemble_acc(&result##iter.packet, &e##iter); \
    if (!GEMV_IS_FLOAT) {                                          \
      result##iter.packet[0][1] = result##iter.packet[1][0];       \
      result##iter.packet[2][1] = result##iter.packet[3][0];       \
    }                                                              \
  }

#define GEMV_LOADPAIR2_COL_MMA(iter1, iter2) \
  b##iter1 = *reinterpret_cast<__vector_pair*>(res + i + ((iter2) * ResPacketSize));

#define GEMV_LOAD2_COL_MMA(iter1, iter2, iter3, N) \
  if (GEMV_GETN(N) > iter1) {                      \
    if (GEMV_IS_FLOAT) {                           \
      GEMV_LOADPAIR2_COL_MMA(iter2, iter2);        \
      EIGEN_UNUSED_VARIABLE(b##iter3);             \
    } else {                                       \
      GEMV_LOADPAIR2_COL_MMA(iter2, iter2 << 1);   \
      GEMV_LOADPAIR2_COL_MMA(iter3, iter3 << 1);   \
    }                                              \
  } else {                                         \
    EIGEN_UNUSED_VARIABLE(b##iter2);               \
    EIGEN_UNUSED_VARIABLE(b##iter3);               \
  }

#if EIGEN_COMP_LLVM
#define GEMV_WORKPAIR2_COL_MMA(iter2, iter3, iter4)                                         \
  ResPacket f##iter2[2];                                                                    \
  __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(f##iter2), &b##iter2);             \
  f##iter2[0] = pmadd(result##iter2.packet[0], palpha, f##iter2[0]);                        \
  f##iter2[1] = pmadd(result##iter3.packet[(iter2 == iter3) ? 2 : 0], palpha, f##iter2[1]); \
  GEMV_BUILDPAIR_MMA(b##iter2, f##iter2[0], f##iter2[1]);
#else
#define GEMV_WORKPAIR2_COL_MMA(iter2, iter3, iter4)                                        \
  if (GEMV_IS_FLOAT) {                                                                     \
    __asm__("xvmaddasp %0,%x1,%x3\n\txvmaddasp %L0,%x2,%x3"                                \
            : "+&d"(b##iter2)                                                              \
            : "wa"(result##iter3.packet[0]), "wa"(result##iter2.packet[0]), "wa"(palpha)); \
  } else {                                                                                 \
    __asm__("xvmaddadp %0,%x1,%x3\n\txvmaddadp %L0,%x2,%x3"                                \
            : "+&d"(b##iter2)                                                              \
            : "wa"(result##iter2.packet[2]), "wa"(result##iter2.packet[0]), "wa"(palpha)); \
  }
#endif

#define GEMV_WORK2_COL_MMA(iter1, iter2, iter3, N)      \
  if (GEMV_GETN(N) > iter1) {                           \
    if (GEMV_IS_FLOAT) {                                \
      GEMV_WORKPAIR2_COL_MMA(iter2, iter3, iter2);      \
    } else {                                            \
      GEMV_WORKPAIR2_COL_MMA(iter2, iter2, iter2 << 1); \
      GEMV_WORKPAIR2_COL_MMA(iter3, iter3, iter3 << 1); \
    }                                                   \
  }

#define GEMV_STOREPAIR2_COL_MMA(iter1, iter2) \
  *reinterpret_cast<__vector_pair*>(res + i + ((iter2) * ResPacketSize)) = b##iter1;

#define GEMV_STORE_COL_MMA(iter, N)                                                                          \
  if (GEMV_GETN(N) > iter) {                                                                                 \
    if (GEMV_IS_FLOAT) {                                                                                     \
      storeMaddData<ResPacket, ResScalar>(res + i + (iter * ResPacketSize), palpha, result##iter.packet[0]); \
    } else {                                                                                                 \
      GEMV_LOADPAIR2_COL_MMA(iter, iter << 1)                                                                \
      GEMV_WORKPAIR2_COL_MMA(iter, iter, iter << 1)                                                          \
      GEMV_STOREPAIR2_COL_MMA(iter, iter << 1)                                                               \
    }                                                                                                        \
  }

#define GEMV_STORE2_COL_MMA(iter1, iter2, iter3, N) \
  if (GEMV_GETN(N) > iter1) {                       \
    if (GEMV_IS_FLOAT) {                            \
      GEMV_STOREPAIR2_COL_MMA(iter2, iter2);        \
    } else {                                        \
      GEMV_STOREPAIR2_COL_MMA(iter2, iter2 << 1)    \
      GEMV_STOREPAIR2_COL_MMA(iter3, iter3 << 1)    \
    }                                               \
  }

#define GEMV_PROCESS_COL_ONE_MMA(N)                 \
  GEMV_UNROLL(GEMV_INIT_MMA, N)                     \
  Index j = j2;                                     \
  __vector_pair b0, b1, b2, b3, b4, b5, b6, b7;     \
  do {                                              \
    LhsPacket g0, g1, g2, g3, g4, g5, g6, g7;       \
    RhsPacket a0 = pset1<RhsPacket>(rhs2(j, 0));    \
    GEMV_UNROLL(GEMV_PREFETCH, N)                   \
    GEMV_LOAD_COL_MMA(N)                            \
    GEMV_WORK_COL_MMA(N)                            \
  } while (++j < jend);                             \
  GEMV_UNROLL(GEMV_DISASSEMBLE_MMA, N)              \
  if (GEMV_GETN(N) <= 1) {                          \
    GEMV_UNROLL(GEMV_STORE_COL_MMA, N)              \
  } else {                                          \
    GEMV_UNROLL_HALF(GEMV_LOAD2_COL_MMA, (N >> 1))  \
    GEMV_UNROLL_HALF(GEMV_WORK2_COL_MMA, (N >> 1))  \
    GEMV_UNROLL_HALF(GEMV_STORE2_COL_MMA, (N >> 1)) \
  }                                                 \
  i += (ResPacketSize * N);
#endif

#define GEMV_INIT(iter, N)                    \
  if (N > iter) {                             \
    c##iter = pset1<ResPacket>(ResScalar(0)); \
  } else {                                    \
    EIGEN_UNUSED_VARIABLE(c##iter);           \
  }

#ifdef EIGEN_POWER_USE_GEMV_PREFETCH
#define GEMV_PREFETCH(iter, N)                                   \
  if (GEMV_GETN(N) > ((iter >> 1) + ((N >> 1) * (iter & 1)))) {  \
    lhs.prefetch(i + (iter * LhsPacketSize) + prefetch_dist, j); \
  }
#else
#define GEMV_PREFETCH(iter, N)
#endif

#define GEMV_WORK_COL(iter, N)                                   \
  if (N > iter) {                                                \
    c##iter = pcj.pmadd(GEMV_LOADPACKET_COL(iter), a0, c##iter); \
  }

#define GEMV_STORE_COL(iter, N)                                                           \
  if (N > iter) {                                                                         \
    pstoreu(res + i + (iter * ResPacketSize),                                             \
            pmadd(c##iter, palpha, ploadu<ResPacket>(res + i + (iter * ResPacketSize)))); \
  }

/** \internal main macro for gemv_col - initialize accumulators, multiply and add inputs, and store results */
#define GEMV_PROCESS_COL_ONE(N)                  \
  GEMV_UNROLL(GEMV_INIT, N)                      \
  Index j = j2;                                  \
  do {                                           \
    RhsPacket a0 = pset1<RhsPacket>(rhs2(j, 0)); \
    GEMV_UNROLL(GEMV_PREFETCH, N)                \
    GEMV_UNROLL(GEMV_WORK_COL, N)                \
  } while (++j < jend);                          \
  GEMV_UNROLL(GEMV_STORE_COL, N)                 \
  i += (ResPacketSize * N);

#ifdef USE_GEMV_MMA
#define GEMV_PROCESS_COL(N) GEMV_PROCESS_COL_ONE_MMA(N)
#else
#define GEMV_PROCESS_COL(N) GEMV_PROCESS_COL_ONE(N)
#endif

/** \internal perform a matrix multiply and accumulate of packet a and packet b */
#ifdef USE_GEMV_MMA
template <typename LhsPacket, typename RhsPacket, bool accumulate>
EIGEN_ALWAYS_INLINE void pger_vecMMA_acc(__vector_quad* acc, const RhsPacket& a, const LhsPacket& b) {
  if (accumulate) {
    __builtin_mma_xvf32gerpp(acc, (__vector unsigned char)a, (__vector unsigned char)b);
  } else {
    __builtin_mma_xvf32ger(acc, (__vector unsigned char)a, (__vector unsigned char)b);
  }
}

/** \internal perform a matrix multiply and accumulate of vector_pair a and packet b */
template <typename LhsPacket, typename RhsPacket, bool accumulate>
EIGEN_ALWAYS_INLINE void pger_vecMMA_acc(__vector_quad* acc, __vector_pair& a, const LhsPacket& b) {
  if (accumulate) {
    __builtin_mma_xvf64gerpp(acc, a, (__vector unsigned char)b);
  } else {
    __builtin_mma_xvf64ger(acc, a, (__vector unsigned char)b);
  }
}
#endif

template <typename LhsScalar, typename LhsMapper, typename RhsScalar, typename RhsMapper, typename ResScalar>
EIGEN_STRONG_INLINE void gemv_col(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs, ResScalar* res,
                                  Index resIncr, ResScalar alpha) {
  typedef gemv_traits<LhsScalar, RhsScalar> Traits;

  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;

  EIGEN_UNUSED_VARIABLE(resIncr);
  eigen_internal_assert(resIncr == 1);

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);
  RhsMapper rhs2(rhs);

  conj_helper<LhsScalar, RhsScalar, false, false> cj;
  conj_helper<LhsPacket, RhsPacket, false, false> pcj;

  const Index lhsStride = lhs.stride();
  // TODO: for padded aligned inputs, we could enable aligned reads
  enum {
    LhsAlignment = Unaligned,
    ResPacketSize = Traits::ResPacketSize,
    LhsPacketSize = Traits::LhsPacketSize,
    RhsPacketSize = Traits::RhsPacketSize,
  };

#ifndef GCC_ONE_VECTORPAIR_BUG
  const Index n8 = rows - 8 * ResPacketSize + 1;
  const Index n4 = rows - 4 * ResPacketSize + 1;
  const Index n2 = rows - 2 * ResPacketSize + 1;
#endif
  const Index n1 = rows - 1 * ResPacketSize + 1;
#ifdef EIGEN_POWER_USE_GEMV_PREFETCH
  const Index prefetch_dist = 64 * LhsPacketSize;
#endif

  // TODO: improve the following heuristic:
  const Index block_cols = cols < 128 ? cols : (lhsStride * sizeof(LhsScalar) < 16000 ? 16 : 8);
  ResPacket palpha = pset1<ResPacket>(alpha);

  for (Index j2 = 0; j2 < cols; j2 += block_cols) {
    Index jend = numext::mini(j2 + block_cols, cols);
    Index i = 0;
    ResPacket c0, c1, c2, c3, c4, c5, c6, c7;
#ifdef USE_GEMV_MMA
    __vector_quad e0, e1, e2, e3, e4, e5, e6, e7;
    PacketBlock<ResPacket, 4> result0, result1, result2, result3, result4, result5, result6, result7;
    GEMV_UNUSED(8, e)
    GEMV_UNUSED(8, result)
    GEMV_UNUSED_EXTRA(1, c)
#endif
#ifndef GCC_ONE_VECTORPAIR_BUG
    while (i < n8) {
      GEMV_PROCESS_COL(8)
    }
    if (i < n4) {
      GEMV_PROCESS_COL(4)
    }
    if (i < n2) {
      GEMV_PROCESS_COL(2)
    }
    if (i < n1)
#else
    while (i < n1)
#endif
    {
      GEMV_PROCESS_COL_ONE(1)
    }
    for (; i < rows; ++i) {
      ResScalar d0(0);
      Index j = j2;
      do {
        d0 += cj.pmul(lhs(i, j), rhs2(j, 0));
      } while (++j < jend);
      res[i] += alpha * d0;
    }
  }
}

template <bool extraRows>
EIGEN_ALWAYS_INLINE void outputVecCol(Packet4f acc, float* result, Packet4f pAlpha, Index extra_rows) {
  Packet4f d0 = ploadu<Packet4f>(result);
  d0 = pmadd(acc, pAlpha, d0);
  if (extraRows) {
    pstoreu_partial(result, d0, extra_rows);
  } else {
    pstoreu(result, d0);
  }
}

template <Index num_acc, bool extraRows, Index size>
EIGEN_ALWAYS_INLINE void outputVecColResults(Packet4f (&acc)[num_acc][size], float* result, Packet4f pAlpha,
                                             Index extra_rows) {
  constexpr Index real_acc = (num_acc - (extraRows ? 1 : 0));
  for (Index k = 0; k < real_acc; k++) {
    outputVecCol<false>(acc[k][0], result + k * 4, pAlpha, extra_rows);
  }
  if (extraRows) {
    outputVecCol<true>(acc[real_acc][0], result + real_acc * 4, pAlpha, extra_rows);
  }
}

static Packet16uc p16uc_MERGE16_32_V1 = {0, 1, 16, 17, 0, 1, 16, 17, 0, 1, 16, 17, 0, 1, 16, 17};
static Packet16uc p16uc_MERGE16_32_V2 = {2, 3, 18, 19, 2, 3, 18, 19, 2, 3, 18, 19, 2, 3, 18, 19};

template <Index num_acc, typename LhsMapper, bool zero>
EIGEN_ALWAYS_INLINE void loadVecLoopVSX(Index k, LhsMapper& lhs, Packet4f (&a0)[num_acc][2]) {
  Packet8bf c0 = lhs.template loadPacket<Packet8bf>(k * 4, 0);
  Packet8bf b1;
  if (!zero) {
    b1 = lhs.template loadPacket<Packet8bf>(k * 4, 1);

    a0[k + 0][1] = oneConvertBF16Hi(b1.m_val);
  }
  a0[k + 0][0] = oneConvertBF16Hi(c0.m_val);

  if (num_acc > (k + 1)) {
    a0[k + 1][0] = oneConvertBF16Lo(c0.m_val);
    if (!zero) {
      a0[k + 1][1] = oneConvertBF16Lo(b1.m_val);
    }
  }
}

template <Index num_acc, bool zero>
EIGEN_ALWAYS_INLINE void multVecVSX(Packet4f (&acc)[num_acc][2], Packet4f (&a0)[num_acc][2], Packet4f (&b0)[2]) {
  for (Index k = 0; k < num_acc; k++) {
    for (Index i = 0; i < (zero ? 1 : 2); i++) {
      acc[k][i] = pmadd(b0[i], a0[k][i], acc[k][i]);
    }
  }
}

template <typename RhsMapper, bool linear>
struct loadColData_impl {
  // linear == false
  static EIGEN_ALWAYS_INLINE Packet8bf run(RhsMapper& rhs, Index j) {
    const Index n = unpacket_traits<Packet8bf>::size;
    EIGEN_ALIGN16 bfloat16 to[n];
    LOAD_STORE_UNROLL_16
    for (Index i = 0; i < n; i++) {
      to[i] = rhs(j + i, 0);
    }
    return pload<Packet8bf>(to);
  }
};

template <typename RhsMapper>
struct loadColData_impl<RhsMapper, true> {
  // linear == true
  static EIGEN_ALWAYS_INLINE Packet8bf run(RhsMapper& rhs, Index j) {
    return rhs.template loadPacket<Packet8bf>(j + 0, 0);
  }
};

template <typename RhsMapper, bool linear>
EIGEN_ALWAYS_INLINE Packet8bf loadColData(RhsMapper& rhs, Index j) {
  return loadColData_impl<RhsMapper, linear>::run(rhs, j);
}

template <Index num_acc, typename LhsMapper, typename RhsMapper, bool zero, bool linear>
EIGEN_ALWAYS_INLINE void vecColLoopVSX(Index j, LhsMapper& lhs, RhsMapper& rhs, Packet4f (&acc)[num_acc][2]) {
  Packet4f a0[num_acc][2], b0[2];
  Packet8bf b2 = loadColData<RhsMapper, linear>(rhs, j);

  b0[0] = oneConvertBF16Perm(b2.m_val, p16uc_MERGE16_32_V1);
  if (!zero) {
    b0[1] = oneConvertBF16Perm(b2.m_val, p16uc_MERGE16_32_V2);
  }

  using LhsSubMapper = typename LhsMapper::SubMapper;

  LhsSubMapper lhs2 = lhs.getSubMapper(0, j);
  for (Index k = 0; k < num_acc; k += 2) {
    loadVecLoopVSX<num_acc, LhsSubMapper, zero>(k, lhs2, a0);
  }

  multVecVSX<num_acc, zero>(acc, a0, b0);
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void addResultsVSX(Packet4f (&acc)[num_acc][2]) {
  for (Index i = 0; i < num_acc; i++) {
    acc[i][0] = acc[i][0] + acc[i][1];
  }
}

// Uses 2X the accumulators or 4X the number of VSX registers
#define MAX_BFLOAT16_VEC_ACC_VSX 8

template <const Index num_acc, typename LhsMapper, typename RhsMapper, bool extraRows, bool linear>
void colVSXVecColLoopBody(Index& row, Index cend, Index rows, LhsMapper& lhs, RhsMapper& rhs, const Packet4f pAlpha,
                          float* result) {
  constexpr Index step = (num_acc * 4);
  const Index extra_rows = (extraRows) ? (rows & 3) : 0;
  constexpr bool multiIters = !extraRows && (num_acc == MAX_BFLOAT16_VEC_ACC_VSX);

  do {
    Packet4f acc[num_acc][2];

    zeroAccumulators<num_acc, 2>(acc);

    using LhsSubMapper = typename LhsMapper::SubMapper;

    LhsSubMapper lhs2 = lhs.getSubMapper(row, 0);
    for (Index j = 0; j + 2 <= cend; j += 2) {
      vecColLoopVSX<num_acc, LhsSubMapper, RhsMapper, false, linear>(j, lhs2, rhs, acc);
    }
    if (cend & 1) {
      vecColLoopVSX<num_acc, LhsSubMapper, RhsMapper, true, linear>(cend - 1, lhs2, rhs, acc);
    }

    addResultsVSX<num_acc>(acc);

    outputVecColResults<num_acc, extraRows, 2>(acc, result, pAlpha, extra_rows);

    result += step;
  } while (multiIters && (step <= rows - (row += step)));
}

template <const Index num_acc, typename LhsMapper, typename RhsMapper, bool extraRows, bool linear>
EIGEN_ALWAYS_INLINE void colVSXVecColLoopBodyExtraN(Index& row, Index cend, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                                    const Packet4f pAlpha, float* result) {
  if (MAX_BFLOAT16_VEC_ACC_VSX > num_acc) {
    colVSXVecColLoopBody<num_acc + (extraRows ? 1 : 0), LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs,
                                                                                                 rhs, pAlpha, result);
  }
}

template <typename LhsMapper, typename RhsMapper, bool extraRows, bool linear>
EIGEN_ALWAYS_INLINE void colVSXVecColLoopBodyExtra(Index& row, Index cend, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                                   const Packet4f pAlpha, float* result) {
  switch ((rows - row) >> 2) {
    case 7:
      colVSXVecColLoopBodyExtraN<7, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 6:
      colVSXVecColLoopBodyExtraN<6, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 5:
      colVSXVecColLoopBodyExtraN<5, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 4:
      colVSXVecColLoopBodyExtraN<4, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 3:
      colVSXVecColLoopBodyExtraN<3, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 2:
      colVSXVecColLoopBodyExtraN<2, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 1:
      colVSXVecColLoopBodyExtraN<1, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    default:
      if (extraRows) {
        colVSXVecColLoopBody<1, LhsMapper, RhsMapper, true, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      }
      break;
  }
}

template <typename LhsMapper, typename RhsMapper, bool linear>
EIGEN_ALWAYS_INLINE void calcVSXVecColLoops(Index cend, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                            const Packet4f pAlpha, float* result) {
  Index row = 0;
  if (rows >= (MAX_BFLOAT16_VEC_ACC_VSX * 4)) {
    colVSXVecColLoopBody<MAX_BFLOAT16_VEC_ACC_VSX, LhsMapper, RhsMapper, false, linear>(row, cend, rows, lhs, rhs,
                                                                                        pAlpha, result);
    result += row;
  }
  if (rows & 3) {
    colVSXVecColLoopBodyExtra<LhsMapper, RhsMapper, true, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
  } else {
    colVSXVecColLoopBodyExtra<LhsMapper, RhsMapper, false, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
  }
}

template <const Index size, bool inc, Index delta>
EIGEN_ALWAYS_INLINE void storeBF16fromResult(bfloat16* dst, Packet8bf data, Index resInc, Index extra) {
  if (inc) {
    if (size < 8) {
      pscatter_partial(dst + delta * resInc, data, resInc, extra);
    } else {
      pscatter(dst + delta * resInc, data, resInc);
    }
  } else {
    if (size < 8) {
      pstoreu_partial(dst + delta, data, extra);
    } else {
      pstoreu(dst + delta, data);
    }
  }
}

template <const Index size, bool inc = false>
EIGEN_ALWAYS_INLINE void convertPointerF32toBF16VSX(Index& i, float* result, Index rows, bfloat16*& dst,
                                                    Index resInc = 1) {
  constexpr Index extra = ((size < 8) ? 8 : size);
  while (i + size <= rows) {
    PacketBlock<Packet8bf, (size + 7) / 8> r32;
    r32.packet[0] = convertF32toBF16VSX(result + i + 0);
    if (size >= 16) {
      r32.packet[1] = convertF32toBF16VSX(result + i + 8);
    }
    if (size >= 32) {
      r32.packet[2] = convertF32toBF16VSX(result + i + 16);
      r32.packet[3] = convertF32toBF16VSX(result + i + 24);
    }
    storeBF16fromResult<size, inc, 0>(dst, r32.packet[0], resInc, rows & 7);
    if (size >= 16) {
      storeBF16fromResult<size, inc, 8>(dst, r32.packet[1], resInc);
    }
    if (size >= 32) {
      storeBF16fromResult<size, inc, 16>(dst, r32.packet[2], resInc);
      storeBF16fromResult<size, inc, 24>(dst, r32.packet[3], resInc);
    }
    i += extra;
    dst += extra * resInc;
    if (size != 32) break;
  }
}

template <bool inc = false>
EIGEN_ALWAYS_INLINE void convertArrayPointerF32toBF16VSX(float* result, Index rows, bfloat16* dst, Index resInc = 1) {
  Index i = 0;
  convertPointerF32toBF16VSX<32, inc>(i, result, rows, dst, resInc);
  convertPointerF32toBF16VSX<16, inc>(i, result, rows, dst, resInc);
  convertPointerF32toBF16VSX<8, inc>(i, result, rows, dst, resInc);
  convertPointerF32toBF16VSX<1, inc>(i, result, rows, dst, resInc);
}

template <typename RhsMapper, typename LhsMapper, typename = void>
struct UseStride : std::false_type {
  static EIGEN_ALWAYS_INLINE void run(Index j2, Index jend, Index rows, LhsMapper& lhs, RhsMapper& rhs, Packet4f pAlpha,
                                      float* result) {
    using RhsSubMapper = typename RhsMapper::SubMapper;

    RhsSubMapper rhs2 = rhs.getSubMapper(j2, 0);
    calcVSXVecColLoops<LhsMapper, RhsSubMapper, false>(jend - j2, rows, lhs, rhs2, pAlpha, result);
  }
};

template <typename RhsMapper, typename LhsMapper>
struct UseStride<RhsMapper, LhsMapper,
                 std::enable_if_t<std::is_member_function_pointer<decltype(&RhsMapper::stride)>::value>>
    : std::true_type {
  static EIGEN_ALWAYS_INLINE void run(Index j2, Index jend, Index rows, LhsMapper& lhs, RhsMapper& rhs, Packet4f pAlpha,
                                      float* result) {
    using RhsSubMapper = typename RhsMapper::SubMapper;

    RhsSubMapper rhs2 = rhs.getSubMapper(j2, 0);
    if (rhs.stride() == 1) {
      calcVSXVecColLoops<LhsMapper, RhsSubMapper, true>(jend - j2, rows, lhs, rhs2, pAlpha, result);
    } else {
      calcVSXVecColLoops<LhsMapper, RhsSubMapper, false>(jend - j2, rows, lhs, rhs2, pAlpha, result);
    }
  }
};

template <typename LhsMapper, typename RhsMapper>
void gemv_bfloat16_col(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs, bfloat16* res,
                       Index resIncr, bfloat16 alpha) {
  EIGEN_UNUSED_VARIABLE(resIncr);
  eigen_internal_assert(resIncr == 1);

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);
  RhsMapper rhs2(rhs);

  const Index lhsStride = lhs.stride();

  // TODO: improve the following heuristic:
  const Index block_cols = cols < 128 ? cols : (lhsStride * sizeof(bfloat16) < 16000 ? 16 : 8);
  float falpha = Eigen::bfloat16_impl::bfloat16_to_float(alpha);
  Packet4f pAlpha = pset1<Packet4f>(falpha);

  ei_declare_aligned_stack_constructed_variable(float, result, rows, 0);

  convertArrayPointerBF16toF32(result, 1, rows, res);

  for (Index j2 = 0; j2 < cols; j2 += block_cols) {
    Index jend = numext::mini(j2 + block_cols, cols);

    using LhsSubMapper = typename LhsMapper::SubMapper;

    LhsSubMapper lhs2 = lhs.getSubMapper(0, j2);
    UseStride<RhsMapper, LhsSubMapper>::run(j2, jend, rows, lhs2, rhs2, pAlpha, result);
  }

  convertArrayPointerF32toBF16VSX(result, rows, res);
}

template <Index num_acc, Index size>
EIGEN_ALWAYS_INLINE void outputVecResults(Packet4f (&acc)[num_acc][size], float* result, Packet4f pAlpha) {
  constexpr Index extra = num_acc & 3;

  for (Index k = 0; k < num_acc; k += 4) {
    Packet4f d0 = ploadu<Packet4f>(result + k);
    d0 = pmadd(acc[k + 0][0], pAlpha, d0);

    if (num_acc > (k + 3)) {
      pstoreu(result + k, d0);
    } else {
      if (extra == 3) {
        pstoreu_partial(result + k, d0, extra);
      } else {
        memcpy((void*)(result + k), (void*)(&d0), sizeof(float) * extra);
      }
    }
  }
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void preduxVecResults2VSX(Packet4f (&acc)[num_acc][2], Index k) {
  if (num_acc > (k + 1)) {
    acc[k][1] = vec_mergel(acc[k + 0][0], acc[k + 1][0]);
    acc[k][0] = vec_mergeh(acc[k + 0][0], acc[k + 1][0]);
    acc[k][0] = acc[k][0] + acc[k][1];
    acc[k][0] += vec_sld(acc[k][0], acc[k][0], 8);
  } else {
    acc[k][0] += vec_sld(acc[k][0], acc[k][0], 8);
#ifdef _BIG_ENDIAN
    acc[k][0] += vec_sld(acc[k][0], acc[k][0], 12);
#else
    acc[k][0] += vec_sld(acc[k][0], acc[k][0], 4);
#endif
  }
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void preduxVecResultsVSX(Packet4f (&acc)[num_acc][2]) {
  for (Index k = 0; k < num_acc; k += 4) {
    preduxVecResults2VSX<num_acc>(acc, k + 0);
    if (num_acc > (k + 2)) {
      preduxVecResults2VSX<num_acc>(acc, k + 2);
#ifdef EIGEN_VECTORIZE_VSX
      acc[k + 0][0] = reinterpret_cast<Packet4f>(
          vec_mergeh(reinterpret_cast<Packet2ul>(acc[k + 0][0]), reinterpret_cast<Packet2ul>(acc[k + 2][0])));
#else
      acc[k + 0][0] = reinterpret_cast<Packet4f>(vec_perm(acc[k + 0][0], acc[k + 2][0], p16uc_TRANSPOSE64_HI));
#endif
    }
  }
}

#ifndef _ARCH_PWR9
EIGEN_ALWAYS_INLINE Packet8us loadPacketPartialZero(Packet8us data, Index extra_cols) {
  Packet16uc shift = pset1<Packet16uc>(8 * 2 * (8 - extra_cols));
#ifdef _BIG_ENDIAN
  return reinterpret_cast<Packet8us>(vec_slo(vec_sro(reinterpret_cast<Packet16uc>(data), shift), shift));
#else
  return reinterpret_cast<Packet8us>(vec_sro(vec_slo(reinterpret_cast<Packet16uc>(data), shift), shift));
#endif
}
#endif

template <Index num_acc, typename LhsMapper, typename RhsMapper, bool extra>
EIGEN_ALWAYS_INLINE void multVSXVecLoop(Packet4f (&acc)[num_acc][2], const LhsMapper& lhs, RhsMapper& rhs, Index j,
                                        Index extra_cols) {
  Packet4f a0[num_acc][2], b0[2];
  Packet8bf a1, b1;

  if (extra) {
    b1 = rhs.template loadPacketPartial<Packet8bf>(j, extra_cols);
#ifndef _ARCH_PWR9
    b1 = loadPacketPartialZero(b1.m_val, extra_cols);
#endif
  } else {
    b1 = rhs.template loadPacket<Packet8bf>(j);
  }
  b0[0] = oneConvertBF16Hi(b1.m_val);
  b0[1] = oneConvertBF16Lo(b1.m_val);

  const LhsMapper lhs2 = lhs.getSubMapper(0, j);
  for (Index k = 0; k < num_acc; k++) {
    if (extra) {
      a1 = lhs2.template loadPacketPartial<Packet8bf>(k, 0, extra_cols);
#ifndef _ARCH_PWR9
      a1 = loadPacketPartialZero(a1.m_val, extra_cols);
#endif
    } else {
      a1 = lhs2.template loadPacket<Packet8bf>(k, 0);
    }
    a0[k][0] = oneConvertBF16Hi(a1.m_val);
    a0[k][1] = oneConvertBF16Lo(a1.m_val);
  }

  multVecVSX<num_acc, false>(acc, a0, b0);
}

template <Index num_acc, typename LhsMapper, typename RhsMapper>
EIGEN_ALWAYS_INLINE void vecVSXLoop(Index cols, const LhsMapper& lhs, RhsMapper& rhs, Packet4f (&acc)[num_acc][2],
                                    Index extra_cols) {
  Index j = 0;
  for (; j + 8 <= cols; j += 8) {
    multVSXVecLoop<num_acc, LhsMapper, RhsMapper, false>(acc, lhs, rhs, j, extra_cols);
  }

  if (extra_cols) {
    multVSXVecLoop<num_acc, LhsMapper, RhsMapper, true>(acc, lhs, rhs, j, extra_cols);
  }
}

template <const Index num_acc, typename LhsMapper, typename RhsMapper>
void colVSXVecLoopBody(Index& row, Index cols, Index rows, LhsMapper& lhs, RhsMapper& rhs, const Packet4f pAlpha,
                       float* result) {
  constexpr bool multiIters = (num_acc == MAX_BFLOAT16_VEC_ACC_VSX);
  const Index extra_cols = (cols & 7);

  do {
    Packet4f acc[num_acc][2];

    zeroAccumulators<num_acc, 2>(acc);

    const LhsMapper lhs2 = lhs.getSubMapper(row, 0);
    vecVSXLoop<num_acc, LhsMapper, RhsMapper>(cols, lhs2, rhs, acc, extra_cols);

    addResultsVSX<num_acc>(acc);

    preduxVecResultsVSX<num_acc>(acc);

    outputVecResults<num_acc, 2>(acc, result, pAlpha);

    result += num_acc;
  } while (multiIters && (num_acc <= rows - (row += num_acc)));
}

template <const Index num_acc, typename LhsMapper, typename RhsMapper>
EIGEN_ALWAYS_INLINE void colVSXVecLoopBodyExtraN(Index& row, Index cols, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                                 const Packet4f pAlpha, float* result) {
  if (MAX_BFLOAT16_VEC_ACC_VSX > num_acc) {
    colVSXVecLoopBody<num_acc, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
  }
}

template <typename LhsMapper, typename RhsMapper>
EIGEN_ALWAYS_INLINE void colVSXVecLoopBodyExtra(Index& row, Index cols, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                                const Packet4f pAlpha, float* result) {
  switch (rows - row) {
    case 7:
      colVSXVecLoopBodyExtraN<7, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 6:
      colVSXVecLoopBodyExtraN<6, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 5:
      colVSXVecLoopBodyExtraN<5, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 4:
      colVSXVecLoopBodyExtraN<4, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 3:
      colVSXVecLoopBodyExtraN<3, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 2:
      colVSXVecLoopBodyExtraN<2, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 1:
      colVSXVecLoopBodyExtraN<1, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
  }
}

template <typename LhsMapper, typename RhsMapper>
EIGEN_ALWAYS_INLINE void calcVSXVecLoops(Index cols, Index rows, LhsMapper& lhs, RhsMapper& rhs, const Packet4f pAlpha,
                                         float* result) {
  Index row = 0;
  if (rows >= MAX_BFLOAT16_VEC_ACC_VSX) {
    colVSXVecLoopBody<MAX_BFLOAT16_VEC_ACC_VSX, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
    result += row;
  }
  colVSXVecLoopBodyExtra<LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
}

template <typename LhsMapper, typename RhsMapper>
EIGEN_STRONG_INLINE void gemv_bfloat16_row(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs,
                                           bfloat16* res, Index resIncr, bfloat16 alpha) {
  typedef typename RhsMapper::LinearMapper LinearMapper;

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);
  LinearMapper rhs2 = rhs.getLinearMapper(0, 0);

  eigen_internal_assert(rhs.stride() == 1);

  float falpha = Eigen::bfloat16_impl::bfloat16_to_float(alpha);
  const Packet4f pAlpha = pset1<Packet4f>(falpha);

  ei_declare_aligned_stack_constructed_variable(float, result, rows, 0);
  if (resIncr == 1) {
    convertArrayPointerBF16toF32(result, 1, rows, res);
  } else {
    convertArrayPointerBF16toF32<true>(result, 1, rows, res, resIncr);
  }
  calcVSXVecLoops<LhsMapper, LinearMapper>(cols, rows, lhs, rhs2, pAlpha, result);
  if (resIncr == 1) {
    convertArrayPointerF32toBF16VSX(result, rows, res);
  } else {
    convertArrayPointerF32toBF16VSX<true>(result, rows, res, resIncr);
  }
}

#undef MAX_BFLOAT16_VEC_ACC_VSX

const Packet16uc p16uc_COMPLEX32_XORFLIP = {0x44, 0x55, 0x66, 0x77, 0x00, 0x11, 0x22, 0x33,
                                            0xcc, 0xdd, 0xee, 0xff, 0x88, 0x99, 0xaa, 0xbb};
const Packet16uc p16uc_COMPLEX64_XORFLIP = {0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
                                            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};

#ifdef _BIG_ENDIAN
const Packet16uc p16uc_COMPLEX32_CONJ_XOR = {0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
                                             0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00};
const Packet16uc p16uc_COMPLEX64_CONJ_XOR = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                             0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const Packet16uc p16uc_COMPLEX32_CONJ_XOR2 = {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                              0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const Packet16uc p16uc_COMPLEX64_CONJ_XOR2 = {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                              0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const Packet16uc p16uc_COMPLEX32_NEGATE = {0x80, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
                                           0x80, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00};
const Packet16uc p16uc_COMPLEX64_NEGATE = {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
#else
const Packet16uc p16uc_COMPLEX32_CONJ_XOR = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
                                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80};
const Packet16uc p16uc_COMPLEX64_CONJ_XOR = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80};
const Packet16uc p16uc_COMPLEX32_CONJ_XOR2 = {0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00,
                                              0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00};
const Packet16uc p16uc_COMPLEX64_CONJ_XOR2 = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
                                              0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const Packet16uc p16uc_COMPLEX32_NEGATE = {0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x80,
                                           0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x80};
const Packet16uc p16uc_COMPLEX64_NEGATE = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
                                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80};
#endif

#ifdef _BIG_ENDIAN
#define COMPLEX_DELTA 0
#else
#define COMPLEX_DELTA 2
#endif

/** \internal packet conjugate (same as pconj but uses the constants in pcplxflipconj for better code generation) */
EIGEN_ALWAYS_INLINE Packet2cf pconj2(const Packet2cf& a) {
  return Packet2cf(pxor(a.v, reinterpret_cast<Packet4f>(p16uc_COMPLEX32_CONJ_XOR)));
}

EIGEN_ALWAYS_INLINE Packet1cd pconj2(const Packet1cd& a) {
  return Packet1cd(pxor(a.v, reinterpret_cast<Packet2d>(p16uc_COMPLEX64_CONJ_XOR)));
}

/** \internal packet conjugate with real & imaginary operation inverted */
EIGEN_ALWAYS_INLINE Packet2cf pconjinv(const Packet2cf& a) {
#ifdef __POWER8_VECTOR__
  return Packet2cf(Packet4f(vec_neg(Packet2d(a.v))));
#else
  return Packet2cf(pxor(a.v, reinterpret_cast<Packet4f>(p16uc_COMPLEX32_CONJ_XOR2)));
#endif
}

EIGEN_ALWAYS_INLINE Packet1cd pconjinv(const Packet1cd& a) {
  return Packet1cd(pxor(a.v, reinterpret_cast<Packet2d>(p16uc_COMPLEX64_CONJ_XOR2)));
}

#if defined(_ARCH_PWR8) && (!EIGEN_COMP_LLVM || __clang_major__ >= 12)
#define PERMXOR_GOOD  // Clang had a bug with vec_permxor and endianness prior to version 12
#endif

/** \internal flip the real & imaginary results and packet conjugate */
EIGEN_ALWAYS_INLINE Packet2cf pcplxflipconj(Packet2cf a) {
#ifdef PERMXOR_GOOD
  return Packet2cf(Packet4f(vec_permxor(Packet16uc(a.v), p16uc_COMPLEX32_CONJ_XOR, p16uc_COMPLEX32_XORFLIP)));
#else
  return pcplxflip(pconj2(a));
#endif
}

EIGEN_ALWAYS_INLINE Packet1cd pcplxflipconj(Packet1cd a) {
#ifdef PERMXOR_GOOD
  return Packet1cd(Packet2d(vec_permxor(Packet16uc(a.v), p16uc_COMPLEX64_CONJ_XOR, p16uc_COMPLEX64_XORFLIP)));
#else
  return pcplxflip(pconj2(a));
#endif
}

/** \internal packet conjugate and flip the real & imaginary results */
EIGEN_ALWAYS_INLINE Packet2cf pcplxconjflip(Packet2cf a) {
#ifdef PERMXOR_GOOD
  return Packet2cf(Packet4f(vec_permxor(Packet16uc(a.v), p16uc_COMPLEX32_CONJ_XOR2, p16uc_COMPLEX32_XORFLIP)));
#else
  return pconj2(pcplxflip(a));
#endif
}

EIGEN_ALWAYS_INLINE Packet1cd pcplxconjflip(Packet1cd a) {
#ifdef PERMXOR_GOOD
  return Packet1cd(Packet2d(vec_permxor(Packet16uc(a.v), p16uc_COMPLEX64_CONJ_XOR2, p16uc_COMPLEX64_XORFLIP)));
#else
  return pconj2(pcplxflip(a));
#endif
}

/** \internal packet negate */
EIGEN_ALWAYS_INLINE Packet2cf pnegate2(Packet2cf a) {
#ifdef __POWER8_VECTOR__
  return Packet2cf(vec_neg(a.v));
#else
  return Packet2cf(pxor(a.v, reinterpret_cast<Packet4f>(p16uc_COMPLEX32_NEGATE)));
#endif
}

EIGEN_ALWAYS_INLINE Packet1cd pnegate2(Packet1cd a) {
#ifdef __POWER8_VECTOR__
  return Packet1cd(vec_neg(a.v));
#else
  return Packet1cd(pxor(a.v, reinterpret_cast<Packet2d>(p16uc_COMPLEX64_NEGATE)));
#endif
}

/** \internal flip the real & imaginary results and negate */
EIGEN_ALWAYS_INLINE Packet2cf pcplxflipnegate(Packet2cf a) {
#ifdef PERMXOR_GOOD
  return Packet2cf(Packet4f(vec_permxor(Packet16uc(a.v), p16uc_COMPLEX32_NEGATE, p16uc_COMPLEX32_XORFLIP)));
#else
  return pcplxflip(pnegate2(a));
#endif
}

EIGEN_ALWAYS_INLINE Packet1cd pcplxflipnegate(Packet1cd a) {
#ifdef PERMXOR_GOOD
  return Packet1cd(Packet2d(vec_permxor(Packet16uc(a.v), p16uc_COMPLEX64_NEGATE, p16uc_COMPLEX64_XORFLIP)));
#else
  return pcplxflip(pnegate2(a));
#endif
}

/** \internal flip the real & imaginary results */
EIGEN_ALWAYS_INLINE Packet2cf pcplxflip2(Packet2cf a) {
  return Packet2cf(Packet4f(vec_perm(Packet16uc(a.v), Packet16uc(a.v), p16uc_COMPLEX32_XORFLIP)));
}

EIGEN_ALWAYS_INLINE Packet1cd pcplxflip2(Packet1cd a) {
#ifdef EIGEN_VECTORIZE_VSX
  return Packet1cd(__builtin_vsx_xxpermdi(a.v, a.v, 2));
#else
  return Packet1cd(Packet2d(vec_perm(Packet16uc(a.v), Packet16uc(a.v), p16uc_COMPLEX64_XORFLIP)));
#endif
}

/** \internal load half a vector with one complex value */
EIGEN_ALWAYS_INLINE Packet4f pload_complex_half(std::complex<float>* src) {
  Packet4f t;
#ifdef EIGEN_VECTORIZE_VSX
  // Load float64/two float32 (doubleword alignment)
  __asm__("lxsdx %x0,%y1" : "=wa"(t) : "Z"(*src));
#else
  *reinterpret_cast<std::complex<float>*>(reinterpret_cast<float*>(&t) + COMPLEX_DELTA) = *src;
#endif
  return t;
}

/** \internal load two vectors from the real and imaginary portions of a complex value */
template <typename RhsScalar>
EIGEN_ALWAYS_INLINE void pload_realimag(RhsScalar* src, Packet4f& r, Packet4f& i) {
#ifdef _ARCH_PWR9
  __asm__("lxvwsx %x0,%y1" : "=wa"(r) : "Z"(*(reinterpret_cast<float*>(src) + 0)));
  __asm__("lxvwsx %x0,%y1" : "=wa"(i) : "Z"(*(reinterpret_cast<float*>(src) + 1)));
#else
  Packet4f t = pload_complex_half(src);
  r = vec_splat(t, COMPLEX_DELTA + 0);
  i = vec_splat(t, COMPLEX_DELTA + 1);
#endif
}

template <typename RhsScalar>
EIGEN_ALWAYS_INLINE void pload_realimag(RhsScalar* src, Packet2d& r, Packet2d& i) {
#ifdef EIGEN_VECTORIZE_VSX
  __asm__("lxvdsx %x0,%y1" : "=wa"(r) : "Z"(*(reinterpret_cast<double*>(src) + 0)));
  __asm__("lxvdsx %x0,%y1" : "=wa"(i) : "Z"(*(reinterpret_cast<double*>(src) + 1)));
#else
  Packet2d t = ploadu<Packet2d>(reinterpret_cast<double*>(src));
  r = vec_splat(t, 0);
  i = vec_splat(t, 1);
#endif
}

#ifndef __POWER8_VECTOR__
const Packet16uc p16uc_MERGEE = {0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
                                 0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19, 0x1A, 0x1B};

const Packet16uc p16uc_MERGEO = {0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17,
                                 0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F};
#endif

/** \internal load two vectors from the interleaved real & imaginary values of src */
template <typename RhsScalar>
EIGEN_ALWAYS_INLINE void pload_realimag_row(RhsScalar* src, Packet4f& r, Packet4f& i) {
  Packet4f t = ploadu<Packet4f>(reinterpret_cast<float*>(src));
#ifdef __POWER8_VECTOR__
  r = vec_mergee(t, t);
  i = vec_mergeo(t, t);
#else
  r = vec_perm(t, t, p16uc_MERGEE);
  i = vec_perm(t, t, p16uc_MERGEO);
#endif
}

template <typename RhsScalar>
EIGEN_ALWAYS_INLINE void pload_realimag_row(RhsScalar* src, Packet2d& r, Packet2d& i) {
  return pload_realimag(src, r, i);
}

/** \internal load and splat a complex value into a vector - column-wise */
EIGEN_ALWAYS_INLINE Packet4f pload_realimag_combine(std::complex<float>* src) {
#ifdef EIGEN_VECTORIZE_VSX
  Packet4f ret;
  __asm__("lxvdsx %x0,%y1" : "=wa"(ret) : "Z"(*(reinterpret_cast<double*>(src) + 0)));
  return ret;
#else
  return Packet4f(ploaddup<Packet2d>(reinterpret_cast<double*>(src)));
#endif
}

EIGEN_ALWAYS_INLINE Packet2d pload_realimag_combine(std::complex<double>* src) { return ploadu<Packet1cd>(src).v; }

/** \internal load a complex value into a vector - row-wise */
EIGEN_ALWAYS_INLINE Packet4f pload_realimag_combine_row(std::complex<float>* src) { return ploadu<Packet2cf>(src).v; }

EIGEN_ALWAYS_INLINE Packet2d pload_realimag_combine_row(std::complex<double>* src) { return ploadu<Packet1cd>(src).v; }

/** \internal load a scalar or a vector from complex location */
template <typename ResPacket>
EIGEN_ALWAYS_INLINE Packet4f pload_complex(std::complex<float>* src) {
  if (GEMV_IS_SCALAR) {
    return pload_complex_half(src);
  } else {
    return ploadu<Packet4f>(reinterpret_cast<float*>(src));
  }
}

template <typename ResPacket>
EIGEN_ALWAYS_INLINE Packet2d pload_complex(std::complex<double>* src) {
  return ploadu<Packet2d>(reinterpret_cast<double*>(src));
}

/** \internal load from a complex vector and convert to a real vector */
template <typename ResPacket>
EIGEN_ALWAYS_INLINE Packet4f pload_complex(Packet2cf* src) {
  return src->v;
}

template <typename ResPacket>
EIGEN_ALWAYS_INLINE Packet2d pload_complex(Packet1cd* src) {
  return src->v;
}

/** \internal load a full vector from complex location - column-wise */
EIGEN_ALWAYS_INLINE Packet4f pload_complex_full(std::complex<float>* src) {
  return Packet4f(ploaddup<Packet2d>(reinterpret_cast<double*>(src)));
}

EIGEN_ALWAYS_INLINE Packet2d pload_complex_full(std::complex<double>* src) { return ploadu<Packet1cd>(src).v; }

/** \internal load a full vector from complex location - row-wise */
EIGEN_ALWAYS_INLINE Packet4f pload_complex_full_row(std::complex<float>* src) { return ploadu<Packet2cf>(src).v; }

EIGEN_ALWAYS_INLINE Packet2d pload_complex_full_row(std::complex<double>* src) { return pload_complex_full(src); }

/** \internal load a vector from a real-only scalar location - column-wise */
EIGEN_ALWAYS_INLINE Packet4f pload_real(float* src) { return pset1<Packet4f>(*src); }

EIGEN_ALWAYS_INLINE Packet2d pload_real(double* src) { return pset1<Packet2d>(*src); }

EIGEN_ALWAYS_INLINE Packet4f pload_real(Packet4f& src) { return src; }

EIGEN_ALWAYS_INLINE Packet2d pload_real(Packet2d& src) { return src; }

/** \internal load a vector from a real-only vector location */
EIGEN_ALWAYS_INLINE Packet4f pload_real_full(float* src) {
  Packet4f ret = ploadu<Packet4f>(src);
  return vec_mergeh(ret, ret);
}

EIGEN_ALWAYS_INLINE Packet2d pload_real_full(double* src) { return pload_real(src); }

EIGEN_ALWAYS_INLINE Packet4f pload_real_full(std::complex<float>* src) {
  return pload_complex_full(src);  // Just for compilation
}

EIGEN_ALWAYS_INLINE Packet2d pload_real_full(std::complex<double>* src) {
  return pload_complex_full(src);  // Just for compilation
}

/** \internal load a vector from a real-only scalar location - row-wise */
template <typename ResPacket>
EIGEN_ALWAYS_INLINE Packet4f pload_real_row(float* src) {
  if (GEMV_IS_SCALAR) {
    return pload_real_full(src);
  } else {
    return ploadu<Packet4f>(src);
  }
}

template <typename ResPacket>
EIGEN_ALWAYS_INLINE Packet2d pload_real_row(double* src) {
  return pload_real(src);
}

EIGEN_ALWAYS_INLINE Packet2cf padd(Packet2cf& a, std::complex<float>& b) {
  EIGEN_UNUSED_VARIABLE(b);
  return a;  // Just for compilation
}

EIGEN_ALWAYS_INLINE Packet1cd padd(Packet1cd& a, std::complex<double>& b) {
  EIGEN_UNUSED_VARIABLE(b);
  return a;  // Just for compilation
}

/** \internal set a scalar from complex location */
template <typename Scalar, typename ResScalar>
EIGEN_ALWAYS_INLINE Scalar pset1_realimag(ResScalar& alpha, int which, int conj) {
  return (which) ? ((conj) ? -alpha.real() : alpha.real()) : ((conj) ? -alpha.imag() : alpha.imag());
}

/** \internal set a vector from complex location */
template <typename Scalar, typename ResScalar, typename ResPacket, int which>
EIGEN_ALWAYS_INLINE Packet2cf pset1_complex(std::complex<float>& alpha) {
  Packet2cf ret;
  ret.v[COMPLEX_DELTA + 0] = pset1_realimag<Scalar, ResScalar>(alpha, (which & 0x01), (which & 0x04));
  ret.v[COMPLEX_DELTA + 1] = pset1_realimag<Scalar, ResScalar>(alpha, (which & 0x02), (which & 0x08));
  ret.v[2 - COMPLEX_DELTA] = ret.v[COMPLEX_DELTA + 0];
  ret.v[3 - COMPLEX_DELTA] = ret.v[COMPLEX_DELTA + 1];
  return ret;
}

template <typename Scalar, typename ResScalar, typename ResPacket, int which>
EIGEN_ALWAYS_INLINE Packet1cd pset1_complex(std::complex<double>& alpha) {
  Packet1cd ret;
  ret.v[0] = pset1_realimag<Scalar, ResScalar>(alpha, (which & 0x01), (which & 0x04));
  ret.v[1] = pset1_realimag<Scalar, ResScalar>(alpha, (which & 0x02), (which & 0x08));
  return ret;
}

/** \internal zero out a vector for real or complex forms */
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet pset_zero() {
  return pset1<Packet>(__UNPACK_TYPE__(Packet)(0));
}

template <>
EIGEN_ALWAYS_INLINE Packet2cf pset_zero<Packet2cf>() {
  return Packet2cf(pset1<Packet4f>(float(0)));
}

template <>
EIGEN_ALWAYS_INLINE Packet1cd pset_zero<Packet1cd>() {
  return Packet1cd(pset1<Packet2d>(double(0)));
}

/** \internal initialize a vector from another vector */
template <typename Packet, typename LhsPacket, typename RhsPacket>
EIGEN_ALWAYS_INLINE Packet pset_init(Packet& c1) {
  if (GEMV_IS_COMPLEX_COMPLEX) {
    EIGEN_UNUSED_VARIABLE(c1);
    return pset_zero<Packet>();
  } else {
    return c1;  // Intentionally left uninitialized
  }
}

template <typename PResPacket, typename ResPacket, typename ResScalar, typename Scalar>
struct alpha_store {
  alpha_store(ResScalar& alpha) {
    separate.r = pset1_complex<Scalar, ResScalar, ResPacket, 0x3>(alpha);
    separate.i = pset1_complex<Scalar, ResScalar, ResPacket, 0x0>(alpha);
  }
  struct ri {
    PResPacket r;
    PResPacket i;
  } separate;
};

/** \internal multiply and add for complex math */
template <typename ScalarPacket, typename AlphaData>
EIGEN_ALWAYS_INLINE ScalarPacket pmadd_complex(ScalarPacket& c0, ScalarPacket& c2, ScalarPacket& c4, AlphaData& b0) {
  return pmadd(c2, b0.separate.i.v, pmadd(c0, b0.separate.r.v, c4));
}

/** \internal store and madd for complex math */
template <typename Scalar, typename ScalarPacket, typename PResPacket, typename ResPacket, typename ResScalar,
          typename AlphaData>
EIGEN_ALWAYS_INLINE void pstoreu_pmadd_complex(PResPacket& c0, AlphaData& b0, ResScalar* res) {
  PResPacket c2 = pcplxflipconj(c0);
  if (GEMV_IS_SCALAR) {
    ScalarPacket c4 = ploadu<ScalarPacket>(reinterpret_cast<Scalar*>(res));
    ScalarPacket c3 = pmadd_complex<ScalarPacket, AlphaData>(c0.v, c2.v, c4, b0);
    pstoreu(reinterpret_cast<Scalar*>(res), c3);
  } else {
    ScalarPacket c4 = pload_complex<ResPacket>(res);
    PResPacket c3 = PResPacket(pmadd_complex<ScalarPacket, AlphaData>(c0.v, c2.v, c4, b0));
    pstoreu(res, c3);
  }
}

template <typename ScalarPacket, typename PResPacket, typename ResPacket, typename ResScalar, typename AlphaData,
          Index ResPacketSize, Index iter2>
EIGEN_ALWAYS_INLINE void pstoreu_pmadd_complex(PResPacket& c0, PResPacket& c1, AlphaData& b0, ResScalar* res) {
  PResPacket c2 = pcplxflipconj(c0);
  PResPacket c3 = pcplxflipconj(c1);
#if !defined(_ARCH_PWR10)
  ScalarPacket c4 = pload_complex<ResPacket>(res + (iter2 * ResPacketSize));
  ScalarPacket c5 = pload_complex<ResPacket>(res + ((iter2 + 1) * ResPacketSize));
  PResPacket c6 = PResPacket(pmadd_complex<ScalarPacket, AlphaData>(c0.v, c2.v, c4, b0));
  PResPacket c7 = PResPacket(pmadd_complex<ScalarPacket, AlphaData>(c1.v, c3.v, c5, b0));
  pstoreu(res + (iter2 * ResPacketSize), c6);
  pstoreu(res + ((iter2 + 1) * ResPacketSize), c7);
#else
  __vector_pair a = *reinterpret_cast<__vector_pair*>(res + (iter2 * ResPacketSize));
#if EIGEN_COMP_LLVM
  PResPacket c6[2];
  __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(c6), &a);
  c6[0] = PResPacket(pmadd_complex<ScalarPacket, AlphaData>(c0.v, c2.v, c6[0].v, b0));
  c6[1] = PResPacket(pmadd_complex<ScalarPacket, AlphaData>(c1.v, c3.v, c6[1].v, b0));
  GEMV_BUILDPAIR_MMA(a, c6[0].v, c6[1].v);
#else
  if (GEMV_IS_COMPLEX_FLOAT) {
    __asm__("xvmaddasp %L0,%x1,%x2\n\txvmaddasp %0,%x1,%x3" : "+&d"(a) : "wa"(b0.separate.r.v), "wa"(c0.v), "wa"(c1.v));
    __asm__("xvmaddasp %L0,%x1,%x2\n\txvmaddasp %0,%x1,%x3" : "+&d"(a) : "wa"(b0.separate.i.v), "wa"(c2.v), "wa"(c3.v));
  } else {
    __asm__("xvmaddadp %L0,%x1,%x2\n\txvmaddadp %0,%x1,%x3" : "+&d"(a) : "wa"(b0.separate.r.v), "wa"(c0.v), "wa"(c1.v));
    __asm__("xvmaddadp %L0,%x1,%x2\n\txvmaddadp %0,%x1,%x3" : "+&d"(a) : "wa"(b0.separate.i.v), "wa"(c2.v), "wa"(c3.v));
  }
#endif
  *reinterpret_cast<__vector_pair*>(res + (iter2 * ResPacketSize)) = a;
#endif
}

/** \internal load lhs packet */
template <typename Scalar, typename LhsScalar, typename LhsMapper, typename LhsPacket>
EIGEN_ALWAYS_INLINE LhsPacket loadLhsPacket(LhsMapper& lhs, Index i, Index j) {
  if (sizeof(Scalar) == sizeof(LhsScalar)) {
    const LhsScalar& src = lhs(i + 0, j);
    return LhsPacket(pload_real_full(const_cast<LhsScalar*>(&src)));
  }
  return lhs.template load<LhsPacket, Unaligned>(i + 0, j);
}

/** \internal madd for complex times complex */
template <typename ComplexPacket, typename RealPacket, bool ConjugateLhs, bool ConjugateRhs, bool Negate>
EIGEN_ALWAYS_INLINE RealPacket pmadd_complex_complex(RealPacket& a, RealPacket& b, RealPacket& c) {
  if (ConjugateLhs && ConjugateRhs) {
    return vec_madd(a, pconj2(ComplexPacket(b)).v, c);
  } else if (Negate && !ConjugateLhs && ConjugateRhs) {
    return vec_nmsub(a, b, c);
  } else {
    return vec_madd(a, b, c);
  }
}

/** \internal madd for complex times real */
template <typename ComplexPacket, typename RealPacket, bool Conjugate>
EIGEN_ALWAYS_INLINE RealPacket pmadd_complex_real(RealPacket& a, RealPacket& b, RealPacket& c) {
  if (Conjugate) {
    return vec_madd(a, pconj2(ComplexPacket(b)).v, c);
  } else {
    return vec_madd(a, b, c);
  }
}

template <typename LhsPacket, typename RhsScalar, typename RhsPacket, typename PResPacket, bool ConjugateLhs,
          bool ConjugateRhs, int StorageOrder>
EIGEN_ALWAYS_INLINE void gemv_mult_generic(LhsPacket& a0, RhsScalar* b, PResPacket& c0) {
  conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs> pcj;
  RhsPacket b0;
  if (StorageOrder == ColMajor) {
    b0 = pset1<RhsPacket>(*b);
  } else {
    b0 = ploadu<RhsPacket>(b);
  }
  c0 = pcj.pmadd(a0, b0, c0);
}

/** \internal core multiply operation for vectors - complex times complex */
template <typename ScalarPacket, typename LhsPacket, typename RhsScalar, typename RhsPacket, typename PResPacket,
          typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>
EIGEN_ALWAYS_INLINE void gemv_mult_complex_complex(LhsPacket& a0, RhsScalar* b, PResPacket& c0, ResPacket& c1) {
  ScalarPacket br, bi;
  if (StorageOrder == ColMajor) {
    pload_realimag<RhsScalar>(b, br, bi);
  } else {
    pload_realimag_row<RhsScalar>(b, br, bi);
  }
  if (ConjugateLhs && !ConjugateRhs) a0 = pconj2(a0);
  LhsPacket a1 = pcplxflipconj(a0);
  ScalarPacket cr = pmadd_complex_complex<LhsPacket, ScalarPacket, ConjugateLhs, ConjugateRhs, false>(a0.v, br, c0.v);
  ScalarPacket ci = pmadd_complex_complex<LhsPacket, ScalarPacket, ConjugateLhs, ConjugateRhs, true>(a1.v, bi, c1.v);
  c1 = ResPacket(ci);
  c0 = PResPacket(cr);
}

/** \internal core multiply operation for vectors - real times complex */
template <typename ScalarPacket, typename LhsPacket, typename RhsScalar, typename RhsPacket, typename PResPacket,
          typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>
EIGEN_ALWAYS_INLINE void gemv_mult_real_complex(LhsPacket& a0, RhsScalar* b, PResPacket& c0) {
  ScalarPacket b0;
  if (StorageOrder == ColMajor) {
    b0 = pload_complex_full(b);
  } else {
    b0 = pload_complex_full_row(b);
  }
  ScalarPacket cri = pmadd_complex_real<PResPacket, ScalarPacket, ConjugateRhs>(a0, b0, c0.v);
  c0 = PResPacket(cri);
}

/** \internal core multiply operation for vectors - complex times real */
template <typename ScalarPacket, typename LhsPacket, typename RhsScalar, typename RhsPacket, typename PResPacket,
          typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>
EIGEN_ALWAYS_INLINE void gemv_mult_complex_real(LhsPacket& a0, RhsScalar* b, PResPacket& c0) {
  ScalarPacket a1 = pload_complex<ResPacket>(&a0);
  ScalarPacket b0;
  if (StorageOrder == ColMajor) {
    b0 = pload_real(b);
  } else {
    b0 = pload_real_row<ResPacket>(b);
  }
  ScalarPacket cri = pmadd_complex_real<PResPacket, ScalarPacket, ConjugateLhs>(a1, b0, c0.v);
  c0 = PResPacket(cri);
}

#define GEMV_MULT_COMPLEX_COMPLEX(LhsType, RhsType, ResType)                                                        \
  template <typename ScalarPacket, typename LhsPacket, typename RhsScalar, typename RhsPacket, typename PResPacket, \
            typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>                             \
  EIGEN_ALWAYS_INLINE void gemv_mult_complex(LhsType& a0, RhsType* b, ResType& c0, ResType& c1) {                   \
    gemv_mult_complex_complex<ScalarPacket, LhsPacket, RhsScalar, RhsPacket, PResPacket, ResPacket, ConjugateLhs,   \
                              ConjugateRhs, StorageOrder>(a0, b, c0, c1);                                           \
  }

GEMV_MULT_COMPLEX_COMPLEX(Packet2cf, std::complex<float>, Packet2cf)
GEMV_MULT_COMPLEX_COMPLEX(Packet1cd, std::complex<double>, Packet1cd)

#define GEMV_MULT_REAL_COMPLEX(LhsType, RhsType, ResType)                                                           \
  template <typename ScalarPacket, typename LhsPacket, typename RhsScalar, typename RhsPacket, typename PResPacket, \
            typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>                             \
  EIGEN_ALWAYS_INLINE void gemv_mult_complex(LhsType& a0, RhsType* b, ResType& c0, RhsType&) {                      \
    gemv_mult_real_complex<ScalarPacket, LhsPacket, RhsScalar, RhsPacket, PResPacket, ResPacket, ConjugateLhs,      \
                           ConjugateRhs, StorageOrder>(a0, b, c0);                                                  \
  }

GEMV_MULT_REAL_COMPLEX(float, std::complex<float>, Packet2cf)
GEMV_MULT_REAL_COMPLEX(double, std::complex<double>, Packet1cd)
GEMV_MULT_REAL_COMPLEX(Packet4f, std::complex<float>, Packet2cf)
GEMV_MULT_REAL_COMPLEX(Packet2d, std::complex<double>, Packet1cd)

#define GEMV_MULT_COMPLEX_REAL(LhsType, RhsType, ResType1, ResType2)                                                \
  template <typename ScalarPacket, typename LhsPacket, typename RhsScalar, typename RhsPacket, typename PResPacket, \
            typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>                             \
  EIGEN_ALWAYS_INLINE void gemv_mult_complex(LhsType& a0, RhsType* b, ResType1& c0, ResType2&) {                    \
    gemv_mult_complex_real<ScalarPacket, LhsPacket, RhsScalar, RhsPacket, PResPacket, ResPacket, ConjugateLhs,      \
                           ConjugateRhs, StorageOrder>(a0, b, c0);                                                  \
  }

GEMV_MULT_COMPLEX_REAL(Packet2cf, float, Packet2cf, std::complex<float>)
GEMV_MULT_COMPLEX_REAL(Packet1cd, double, Packet1cd, std::complex<double>)
GEMV_MULT_COMPLEX_REAL(std::complex<float>, float, Packet2cf, std::complex<float>)
GEMV_MULT_COMPLEX_REAL(std::complex<double>, double, Packet1cd, std::complex<double>)

#ifdef USE_GEMV_MMA
/** \internal convert packet to real form */
template <typename T>
EIGEN_ALWAYS_INLINE T convertReal(T a) {
  return a;
}

EIGEN_ALWAYS_INLINE Packet4f convertReal(Packet2cf a) { return a.v; }

EIGEN_ALWAYS_INLINE Packet2d convertReal(Packet1cd a) { return a.v; }

/** \internal convert packet to complex form */
template <typename T>
EIGEN_ALWAYS_INLINE T convertComplex(T a) {
  return a;
}

EIGEN_ALWAYS_INLINE Packet2cf convertComplex(Packet4f a) { return Packet2cf(a); }

EIGEN_ALWAYS_INLINE Packet1cd convertComplex(Packet2d a) { return Packet1cd(a); }

/** \internal load a vector from a complex location (for MMA version) */
template <typename ScalarPacket, typename LhsPacket, typename SLhsPacket, typename ResPacket>
EIGEN_ALWAYS_INLINE void pload_complex_MMA(SLhsPacket& a) {
  a = SLhsPacket(pload_complex<ResPacket>(&a));
}

template <typename ScalarPacket, typename LhsPacket, typename SLhsPacket, typename ResPacket>
EIGEN_ALWAYS_INLINE void pload_complex_MMA(__vector_pair&) {
  // Pass thru
}

/** \internal perform a matrix multiply and accumulate (positive and negative) of packet a and packet b */
template <typename LhsPacket, typename RhsPacket, bool NegativeAccumulate>
EIGEN_ALWAYS_INLINE void pger_vecMMA(__vector_quad* acc, RhsPacket& a, LhsPacket& b) {
  if (NegativeAccumulate) {
    __builtin_mma_xvf32gernp(acc, (__vector unsigned char)a, (__vector unsigned char)b);
  } else {
    __builtin_mma_xvf32gerpp(acc, (__vector unsigned char)a, (__vector unsigned char)b);
  }
}

/** \internal perform a matrix multiply and accumulate (positive and negative) of vector_pair a and packet b */
template <typename LhsPacket, typename RhsPacket, bool NegativeAccumulate>
EIGEN_ALWAYS_INLINE void pger_vecMMA(__vector_quad* acc, __vector_pair& a, Packet2d& b) {
  if (NegativeAccumulate) {
    __builtin_mma_xvf64gernp(acc, (__vector_pair)a, (__vector unsigned char)b);
  } else {
    __builtin_mma_xvf64gerpp(acc, (__vector_pair)a, (__vector unsigned char)b);
  }
}

template <typename LhsPacket, typename RhsPacket, bool NegativeAccumulate>
EIGEN_ALWAYS_INLINE void pger_vecMMA(__vector_quad*, __vector_pair&, Packet4f&) {
  // Just for compilation
}

/** \internal madd for complex times complex (MMA version) */
template <typename RealPacket, typename LhsPacket, bool ConjugateLhs, bool ConjugateRhs, bool Negate>
EIGEN_ALWAYS_INLINE void pmadd_complex_complex_MMA(LhsPacket& a, RealPacket& b, __vector_quad* c) {
  if (ConjugateLhs && ConjugateRhs) {
    RealPacket b2 = pconj2(convertComplex(b)).v;
    return pger_vecMMA<RealPacket, RealPacket, false>(c, b2, a.v);
  } else if (Negate && !ConjugateLhs && ConjugateRhs) {
    return pger_vecMMA<RealPacket, RealPacket, true>(c, b, a.v);
  } else {
    return pger_vecMMA<RealPacket, RealPacket, false>(c, b, a.v);
  }
}

template <typename RealPacket, typename LhsPacket, bool ConjugateLhs, bool ConjugateRhs, bool Negate>
EIGEN_ALWAYS_INLINE void pmadd_complex_complex_MMA(__vector_pair& a, RealPacket& b, __vector_quad* c) {
  if (ConjugateLhs && ConjugateRhs) {
    RealPacket b2 = pconj2(convertComplex(b)).v;
    return pger_vecMMA<RealPacket, __vector_pair, false>(c, a, b2);
  } else if (Negate && !ConjugateLhs && ConjugateRhs) {
    return pger_vecMMA<RealPacket, __vector_pair, true>(c, a, b);
  } else {
    return pger_vecMMA<RealPacket, __vector_pair, false>(c, a, b);
  }
}

/** \internal madd for complex times real (MMA version) */
template <typename RealPacket, typename LhsPacket, bool Conjugate, int StorageOrder>
EIGEN_ALWAYS_INLINE void pmadd_complex_real_MMA(LhsPacket& a, RealPacket& b, __vector_quad* c) {
  RealPacket a2 = convertReal(a);
  if (Conjugate) {
    RealPacket b2 = pconj2(convertComplex(b)).v;
    if (StorageOrder == ColMajor) {
      return pger_vecMMA<RealPacket, RealPacket, false>(c, b2, a2);
    } else {
      return pger_vecMMA<RealPacket, RealPacket, false>(c, a2, b2);
    }
  } else {
    if (StorageOrder == ColMajor) {
      return pger_vecMMA<RealPacket, RealPacket, false>(c, b, a2);
    } else {
      return pger_vecMMA<RealPacket, RealPacket, false>(c, a2, b);
    }
  }
}

/** \internal madd for real times complex (MMA version) */
template <typename RealPacket, typename LhsPacket, bool Conjugate, int StorageOrder>
EIGEN_ALWAYS_INLINE void pmadd_complex_real_MMA(__vector_pair& a, RealPacket& b, __vector_quad* c) {
  if (Conjugate) {
    RealPacket b2 = pconj2(convertComplex(b)).v;
    return pger_vecMMA<RealPacket, __vector_pair, false>(c, a, b2);
  } else {
    return pger_vecMMA<RealPacket, __vector_pair, false>(c, a, b);
  }
}

/** \internal core multiply operation for vectors (MMA version) - complex times complex */
template <typename ScalarPacket, typename LhsPacket, typename SLhsPacket, typename RhsScalar, typename ResPacket,
          bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>
EIGEN_ALWAYS_INLINE void gemv_mult_complex_complex_MMA(SLhsPacket& a0, RhsScalar* b, __vector_quad* c0) {
  ScalarPacket b0;
  if (StorageOrder == ColMajor) {
    b0 = pload_realimag_combine(b);
  } else {
    b0 = pload_realimag_combine_row(b);
  }
  pmadd_complex_complex_MMA<ScalarPacket, LhsPacket, ConjugateLhs, ConjugateRhs, false>(a0, b0, c0);
}

/** \internal core multiply operation for vectors (MMA version) - complex times real */
template <typename ScalarPacket, typename LhsPacket, typename SLhsPacket, typename RhsScalar, typename ResPacket,
          bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>
EIGEN_ALWAYS_INLINE void gemv_mult_complex_real_MMA(SLhsPacket& a0, RhsScalar* b, __vector_quad* c0) {
  pload_complex_MMA<ScalarPacket, LhsPacket, SLhsPacket, ResPacket>(a0);
  ScalarPacket b0;
  if (StorageOrder == ColMajor) {
    b0 = pload_real(b);
  } else {
    b0 = pload_real_row<ResPacket>(b);
  }
  pmadd_complex_real_MMA<ScalarPacket, LhsPacket, ConjugateLhs, ColMajor>(a0, b0, c0);
}

/** \internal core multiply operation for vectors (MMA version) - real times complex */
template <typename ScalarPacket, typename LhsPacket, typename SLhsPacket, typename RhsScalar, typename ResPacket,
          bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>
EIGEN_ALWAYS_INLINE void gemv_mult_real_complex_MMA(SLhsPacket& a0, RhsScalar* b, __vector_quad* c0) {
  ScalarPacket b0;
  if (StorageOrder == ColMajor) {
    b0 = pload_complex_full(b);
  } else {
    b0 = pload_complex_full_row(b);
  }
  pmadd_complex_real_MMA<ScalarPacket, LhsPacket, ConjugateRhs,
                         (sizeof(RhsScalar) == sizeof(std::complex<float>)) ? StorageOrder : ColMajor>(a0, b0, c0);
}

#define GEMV_MULT_COMPLEX_COMPLEX_MMA(LhsType, RhsType)                                                             \
  template <typename ScalarPacket, typename LhsScalar, typename LhsPacket, typename SLhsPacket, typename RhsScalar, \
            typename RhsPacket, typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>         \
  EIGEN_ALWAYS_INLINE void gemv_mult_complex_MMA(LhsType& a0, RhsType* b, __vector_quad* c0) {                      \
    gemv_mult_complex_complex_MMA<ScalarPacket, LhsPacket, SLhsPacket, RhsScalar, ResPacket, ConjugateLhs,          \
                                  ConjugateRhs, StorageOrder>(a0, b, c0);                                           \
  }

GEMV_MULT_COMPLEX_COMPLEX_MMA(Packet2cf, std::complex<float>)
GEMV_MULT_COMPLEX_COMPLEX_MMA(__vector_pair, std::complex<float>)
GEMV_MULT_COMPLEX_COMPLEX_MMA(Packet1cd, std::complex<double>)

/** \internal core multiply operation for vectors (MMA version) - complex times complex */
template <typename ScalarPacket, typename LhsScalar, typename LhsPacket, typename SLhsPacket, typename RhsScalar,
          typename RhsPacket, typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>
EIGEN_ALWAYS_INLINE void gemv_mult_complex_MMA(__vector_pair& a0, std::complex<double>* b, __vector_quad* c0) {
  if (sizeof(LhsScalar) == 16) {
    gemv_mult_complex_complex_MMA<ScalarPacket, LhsPacket, SLhsPacket, RhsScalar, ResPacket, ConjugateLhs, ConjugateRhs,
                                  StorageOrder>(a0, b, c0);
  } else {
    gemv_mult_real_complex_MMA<ScalarPacket, LhsPacket, SLhsPacket, RhsScalar, ResPacket, ConjugateLhs, ConjugateRhs,
                               StorageOrder>(a0, b, c0);
  }
}

#define GEMV_MULT_REAL_COMPLEX_MMA(LhsType, RhsType)                                                                  \
  template <typename ScalarPacket, typename LhsScalar, typename LhsPacket, typename SLhsPacket, typename RhsScalar,   \
            typename RhsPacket, typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>           \
  EIGEN_ALWAYS_INLINE void gemv_mult_complex_MMA(LhsType& a0, RhsType* b, __vector_quad* c0) {                        \
    gemv_mult_real_complex_MMA<ScalarPacket, LhsPacket, SLhsPacket, RhsScalar, ResPacket, ConjugateLhs, ConjugateRhs, \
                               StorageOrder>(a0, b, c0);                                                              \
  }

GEMV_MULT_REAL_COMPLEX_MMA(Packet4f, std::complex<float>)
GEMV_MULT_REAL_COMPLEX_MMA(Packet2d, std::complex<double>)

#define GEMV_MULT_COMPLEX_REAL_MMA(LhsType, RhsType)                                                                  \
  template <typename ScalarPacket, typename LhsScalar, typename LhsPacket, typename SLhsPacket, typename RhsScalar,   \
            typename RhsPacket, typename ResPacket, bool ConjugateLhs, bool ConjugateRhs, int StorageOrder>           \
  EIGEN_ALWAYS_INLINE void gemv_mult_complex_MMA(LhsType& a0, RhsType* b, __vector_quad* c0) {                        \
    gemv_mult_complex_real_MMA<ScalarPacket, LhsPacket, SLhsPacket, RhsScalar, ResPacket, ConjugateLhs, ConjugateRhs, \
                               StorageOrder>(a0, b, c0);                                                              \
  }

GEMV_MULT_COMPLEX_REAL_MMA(Packet2cf, float)
GEMV_MULT_COMPLEX_REAL_MMA(Packet1cd, double)
GEMV_MULT_COMPLEX_REAL_MMA(__vector_pair, float)
GEMV_MULT_COMPLEX_REAL_MMA(__vector_pair, double)

/** \internal disassemble MMA accumulator results into packets */
template <typename Scalar, typename ScalarPacket, typename LhsPacket, typename RhsPacket, bool ConjugateLhs,
          bool ConjugateRhs>
EIGEN_ALWAYS_INLINE void disassembleResults2(__vector_quad* c0, PacketBlock<ScalarPacket, 4>& result0) {
  __builtin_mma_disassemble_acc(&result0.packet, c0);
  if (sizeof(LhsPacket) == 16) {
    if (sizeof(RhsPacket) == 16) {
      ScalarPacket tmp0, tmp2;
      tmp2 = vec_mergeh(result0.packet[2], result0.packet[3]);
      tmp0 = vec_mergeh(result0.packet[0], result0.packet[1]);
      result0.packet[3] = vec_mergel(result0.packet[3], result0.packet[2]);
      result0.packet[1] = vec_mergel(result0.packet[1], result0.packet[0]);
      result0.packet[2] = tmp2;
      result0.packet[0] = tmp0;

      if (ConjugateLhs) {
        result0.packet[0] = pconj2(convertComplex(result0.packet[0])).v;
        result0.packet[2] = pconj2(convertComplex(result0.packet[2])).v;
      } else if (ConjugateRhs) {
        result0.packet[1] = pconj2(convertComplex(result0.packet[1])).v;
        result0.packet[3] = pconj2(convertComplex(result0.packet[3])).v;
      } else {
        result0.packet[1] = pconjinv(convertComplex(result0.packet[1])).v;
        result0.packet[3] = pconjinv(convertComplex(result0.packet[3])).v;
      }
      result0.packet[0] = vec_add(result0.packet[0], result0.packet[1]);
      result0.packet[2] = vec_add(result0.packet[2], result0.packet[3]);
    } else {
      result0.packet[0][1] = result0.packet[1][1];
      result0.packet[2][1] = result0.packet[3][1];
    }
  }
}

template <typename Scalar, typename ScalarPacket, typename LhsPacket, typename RhsPacket, bool ConjugateLhs,
          bool ConjugateRhs>
EIGEN_ALWAYS_INLINE void disassembleResults4(__vector_quad* c0, PacketBlock<ScalarPacket, 4>& result0) {
  __builtin_mma_disassemble_acc(&result0.packet, c0);
  if (GEMV_IS_COMPLEX_COMPLEX) {
    if (ConjugateLhs) {
      result0.packet[0] = pconj2(convertComplex(result0.packet[0])).v;
      result0.packet[1] = pcplxflip2(convertComplex(result0.packet[1])).v;
    } else {
      if (ConjugateRhs) {
        result0.packet[1] = pcplxconjflip(convertComplex(result0.packet[1])).v;
      } else {
        result0.packet[1] = pcplxflipconj(convertComplex(result0.packet[1])).v;
      }
    }
    result0.packet[0] = vec_add(result0.packet[0], result0.packet[1]);
  } else if (sizeof(LhsPacket) == sizeof(std::complex<float>)) {
    if (ConjugateLhs) {
      result0.packet[0] = pconj2(convertComplex(result0.packet[0])).v;
    }
  } else {
    result0.packet[0] = vec_mergee(result0.packet[0], result0.packet[1]);
  }
}

template <typename Scalar, typename ScalarPacket, int ResPacketSize, typename LhsPacket, typename RhsPacket,
          bool ConjugateLhs, bool ConjugateRhs>
EIGEN_ALWAYS_INLINE void disassembleResults(__vector_quad* c0, PacketBlock<ScalarPacket, 4>& result0) {
  if (!GEMV_IS_COMPLEX_FLOAT) {
    disassembleResults2<Scalar, ScalarPacket, LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>(c0, result0);
  } else {
    disassembleResults4<Scalar, ScalarPacket, LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>(c0, result0);
  }
}
#endif

#define GEMV_GETN_COMPLEX(N) (((N) * ResPacketSize) >> 1)

#define GEMV_LOADPACKET_COL_COMPLEX(iter) \
  loadLhsPacket<Scalar, LhsScalar, LhsMapper, PLhsPacket>(lhs, i + ((iter) * ResPacketSize), j)

#define GEMV_LOADPACKET_COL_COMPLEX_DATA(iter) convertReal(GEMV_LOADPACKET_COL_COMPLEX(iter))

#ifdef USE_GEMV_MMA
#define GEMV_INIT_COL_COMPLEX_MMA(iter, N) \
  if (GEMV_GETN_COMPLEX(N) > iter) {       \
    __builtin_mma_xxsetaccz(&e0##iter);    \
  }

#if EIGEN_COMP_LLVM
#define GEMV_LOADPAIR_COL_COMPLEX_MMA(iter1, iter2)                     \
  GEMV_BUILDPAIR_MMA(a##iter1, GEMV_LOADPACKET_COL_COMPLEX_DATA(iter2), \
                     GEMV_LOADPACKET_COL_COMPLEX_DATA((iter2) + 1));    \
  EIGEN_UNUSED_VARIABLE(f##iter1);
#else
#define GEMV_LOADPAIR_COL_COMPLEX_MMA(iter1, iter2)                                                         \
  if (sizeof(LhsPacket) == 16) {                                                                            \
    const LhsScalar& src = lhs(i + ((32 * iter1) / sizeof(LhsScalar)), j);                                  \
    a##iter1 = *reinterpret_cast<__vector_pair*>(const_cast<LhsScalar*>(&src));                             \
    EIGEN_UNUSED_VARIABLE(f##iter1);                                                                        \
  } else {                                                                                                  \
    f##iter1 = lhs.template load<PLhsPacket, Unaligned>(i + ((iter2) * ResPacketSize), j);                  \
    GEMV_BUILDPAIR_MMA(a##iter1, vec_splat(convertReal(f##iter1), 0), vec_splat(convertReal(f##iter1), 1)); \
  }
#endif

#define GEMV_LOAD1_COL_COMPLEX_MMA(iter, N)          \
  if (GEMV_GETN_COMPLEX(N) > iter) {                 \
    if (GEMV_IS_COMPLEX_FLOAT) {                     \
      f##iter = GEMV_LOADPACKET_COL_COMPLEX(iter);   \
      EIGEN_UNUSED_VARIABLE(a##iter);                \
    } else {                                         \
      GEMV_LOADPAIR_COL_COMPLEX_MMA(iter, iter << 1) \
    }                                                \
  } else {                                           \
    EIGEN_UNUSED_VARIABLE(a##iter);                  \
    EIGEN_UNUSED_VARIABLE(f##iter);                  \
  }

#define GEMV_WORK1_COL_COMPLEX_MMA(iter, N)                                                                      \
  if (GEMV_GETN_COMPLEX(N) > iter) {                                                                             \
    if (GEMV_IS_COMPLEX_FLOAT) {                                                                                 \
      gemv_mult_complex_MMA<ScalarPacket, LhsScalar, PLhsPacket, PLhsPacket, RhsScalar, RhsPacket, ResPacket,    \
                            ConjugateLhs, ConjugateRhs, ColMajor>(f##iter, b, &e0##iter);                        \
    } else {                                                                                                     \
      gemv_mult_complex_MMA<ScalarPacket, LhsScalar, PLhsPacket, __vector_pair, RhsScalar, RhsPacket, ResPacket, \
                            ConjugateLhs, ConjugateRhs, ColMajor>(a##iter, b, &e0##iter);                        \
    }                                                                                                            \
  }

#define GEMV_LOADPAIR2_COL_COMPLEX_MMA(iter1, iter2) \
  GEMV_BUILDPAIR_MMA(a##iter1, GEMV_LOADPACKET_COL_COMPLEX_DATA(iter2), GEMV_LOADPACKET_COL_COMPLEX_DATA((iter2) + 1));

#define GEMV_LOAD2_COL_COMPLEX_MMA(iter1, iter2, iter3, N) \
  if (GEMV_GETN_COMPLEX(N) > iter1) {                      \
    if (GEMV_IS_COMPLEX_FLOAT) {                           \
      GEMV_LOADPAIR2_COL_COMPLEX_MMA(iter2, iter2);        \
      EIGEN_UNUSED_VARIABLE(a##iter3)                      \
    } else {                                               \
      GEMV_LOADPAIR2_COL_COMPLEX_MMA(iter2, iter2 << 1);   \
      GEMV_LOADPAIR2_COL_COMPLEX_MMA(iter3, iter3 << 1);   \
    }                                                      \
  } else {                                                 \
    EIGEN_UNUSED_VARIABLE(a##iter2);                       \
    EIGEN_UNUSED_VARIABLE(a##iter3);                       \
  }                                                        \
  EIGEN_UNUSED_VARIABLE(f##iter2);                         \
  EIGEN_UNUSED_VARIABLE(f##iter3);

#define GEMV_WORK2_COL_COMPLEX_MMA(iter1, iter2, iter3, N)                                                       \
  if (GEMV_GETN_COMPLEX(N) > iter1) {                                                                            \
    if (GEMV_IS_COMPLEX_FLOAT) {                                                                                 \
      PLhsPacket g[2];                                                                                           \
      __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(g), &a##iter2);                                     \
      gemv_mult_complex_MMA<ScalarPacket, LhsScalar, PLhsPacket, PLhsPacket, RhsScalar, RhsPacket, ResPacket,    \
                            ConjugateLhs, ConjugateRhs, ColMajor>(g[0], b, &e0##iter2);                          \
      gemv_mult_complex_MMA<ScalarPacket, LhsScalar, PLhsPacket, PLhsPacket, RhsScalar, RhsPacket, ResPacket,    \
                            ConjugateLhs, ConjugateRhs, ColMajor>(g[1], b, &e0##iter3);                          \
    } else {                                                                                                     \
      gemv_mult_complex_MMA<ScalarPacket, LhsScalar, PLhsPacket, __vector_pair, RhsScalar, RhsPacket, ResPacket, \
                            ConjugateLhs, ConjugateRhs, ColMajor>(a##iter2, b, &e0##iter2);                      \
      gemv_mult_complex_MMA<ScalarPacket, LhsScalar, PLhsPacket, __vector_pair, RhsScalar, RhsPacket, ResPacket, \
                            ConjugateLhs, ConjugateRhs, ColMajor>(a##iter3, b, &e0##iter3);                      \
    }                                                                                                            \
  }

#if EIGEN_COMP_LLVM
#define GEMV_LOAD_COL_COMPLEX_MMA(N)                       \
  if (GEMV_GETN_COMPLEX(N) > 1) {                          \
    GEMV_UNROLL_HALF(GEMV_LOAD2_COL_COMPLEX_MMA, (N >> 1)) \
  } else {                                                 \
    GEMV_UNROLL(GEMV_LOAD1_COL_COMPLEX_MMA, N)             \
  }

#define GEMV_WORK_COL_COMPLEX_MMA(N)                       \
  if (GEMV_GETN_COMPLEX(N) > 1) {                          \
    GEMV_UNROLL_HALF(GEMV_WORK2_COL_COMPLEX_MMA, (N >> 1)) \
  } else {                                                 \
    GEMV_UNROLL(GEMV_WORK1_COL_COMPLEX_MMA, N)             \
  }
#else
#define GEMV_LOAD_COL_COMPLEX_MMA(N) GEMV_UNROLL(GEMV_LOAD1_COL_COMPLEX_MMA, N)

#define GEMV_WORK_COL_COMPLEX_MMA(N) GEMV_UNROLL(GEMV_WORK1_COL_COMPLEX_MMA, N)
#endif

#define GEMV_DISASSEMBLE_COMPLEX_MMA(iter)                                                                   \
  disassembleResults<Scalar, ScalarPacket, ResPacketSize, LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>( \
      &e0##iter, result0##iter);

#define GEMV_STORE_COL_COMPLEX_MMA(iter, N)                                                     \
  if (GEMV_GETN_COMPLEX(N) > iter) {                                                            \
    GEMV_DISASSEMBLE_COMPLEX_MMA(iter);                                                         \
    c0##iter = PResPacket(result0##iter.packet[0]);                                             \
    if (GEMV_IS_COMPLEX_FLOAT) {                                                                \
      pstoreu_pmadd_complex<Scalar, ScalarPacket, PResPacket, ResPacket, ResScalar, AlphaData>( \
          c0##iter, alpha_data, res + i + (iter * ResPacketSize));                              \
    } else {                                                                                    \
      pstoreu_pmadd_complex<Scalar, ScalarPacket, PResPacket, ResPacket, ResScalar, AlphaData>( \
          c0##iter, alpha_data, res + i + ((iter << 1) * ResPacketSize));                       \
      c0##iter = PResPacket(result0##iter.packet[2]);                                           \
      pstoreu_pmadd_complex<Scalar, ScalarPacket, PResPacket, ResPacket, ResScalar, AlphaData>( \
          c0##iter, alpha_data, res + i + (((iter << 1) + 1) * ResPacketSize));                 \
    }                                                                                           \
  }

#define GEMV_STORE2_COL_COMPLEX_MMA(iter1, iter2, iter3, N)                                                        \
  if (GEMV_GETN_COMPLEX(N) > iter1) {                                                                              \
    GEMV_DISASSEMBLE_COMPLEX_MMA(iter2);                                                                           \
    GEMV_DISASSEMBLE_COMPLEX_MMA(iter3);                                                                           \
    c0##iter2 = PResPacket(result0##iter2.packet[0]);                                                              \
    if (GEMV_IS_COMPLEX_FLOAT) {                                                                                   \
      c0##iter3 = PResPacket(result0##iter3.packet[0]);                                                            \
      pstoreu_pmadd_complex<ScalarPacket, PResPacket, ResPacket, ResScalar, AlphaData, ResPacketSize, iter2>(      \
          c0##iter2, c0##iter3, alpha_data, res + i);                                                              \
    } else {                                                                                                       \
      c0##iter3 = PResPacket(result0##iter2.packet[2]);                                                            \
      pstoreu_pmadd_complex<ScalarPacket, PResPacket, ResPacket, ResScalar, AlphaData, ResPacketSize, iter2 << 1>( \
          c0##iter2, c0##iter3, alpha_data, res + i);                                                              \
      c0##iter2 = PResPacket(result0##iter3.packet[0]);                                                            \
      c0##iter3 = PResPacket(result0##iter3.packet[2]);                                                            \
      pstoreu_pmadd_complex<ScalarPacket, PResPacket, ResPacket, ResScalar, AlphaData, ResPacketSize, iter3 << 1>( \
          c0##iter2, c0##iter3, alpha_data, res + i);                                                              \
    }                                                                                                              \
  }

#define GEMV_PROCESS_COL_COMPLEX_ONE_MMA(N)                 \
  GEMV_UNROLL(GEMV_INIT_COL_COMPLEX_MMA, N)                 \
  Index j = j2;                                             \
  do {                                                      \
    const RhsScalar& b1 = rhs2(j, 0);                       \
    RhsScalar* b = const_cast<RhsScalar*>(&b1);             \
    GEMV_UNROLL(GEMV_PREFETCH, N)                           \
    GEMV_LOAD_COL_COMPLEX_MMA(N)                            \
    GEMV_WORK_COL_COMPLEX_MMA(N)                            \
  } while (++j < jend);                                     \
  if (GEMV_GETN(N) <= 2) {                                  \
    GEMV_UNROLL(GEMV_STORE_COL_COMPLEX_MMA, N)              \
  } else {                                                  \
    GEMV_UNROLL_HALF(GEMV_STORE2_COL_COMPLEX_MMA, (N >> 1)) \
  }                                                         \
  i += (ResPacketSize * N);
#endif

#define GEMV_INIT_COMPLEX(iter, N)                                   \
  if (N > iter) {                                                    \
    c0##iter = pset_zero<PResPacket>();                              \
    c1##iter = pset_init<ResPacket, LhsPacket, RhsPacket>(c1##iter); \
  } else {                                                           \
    EIGEN_UNUSED_VARIABLE(c0##iter);                                 \
    EIGEN_UNUSED_VARIABLE(c1##iter);                                 \
  }

#define GEMV_WORK_COL_COMPLEX(iter, N)                                                                     \
  if (N > iter) {                                                                                          \
    f##iter = GEMV_LOADPACKET_COL_COMPLEX(iter);                                                           \
    gemv_mult_complex<ScalarPacket, PLhsPacket, RhsScalar, RhsPacket, PResPacket, ResPacket, ConjugateLhs, \
                      ConjugateRhs, ColMajor>(f##iter, b, c0##iter, c1##iter);                             \
  } else {                                                                                                 \
    EIGEN_UNUSED_VARIABLE(f##iter);                                                                        \
  }

#define GEMV_STORE_COL_COMPLEX(iter, N)                                                       \
  if (N > iter) {                                                                             \
    if (GEMV_IS_COMPLEX_COMPLEX) {                                                            \
      c0##iter = padd(c0##iter, c1##iter);                                                    \
    }                                                                                         \
    pstoreu_pmadd_complex<Scalar, ScalarPacket, PResPacket, ResPacket, ResScalar, AlphaData>( \
        c0##iter, alpha_data, res + i + (iter * ResPacketSize));                              \
  }

/** \internal main macro for gemv_complex_col - initialize accumulators, multiply and add inputs, and store results */
#define GEMV_PROCESS_COL_COMPLEX_ONE(N)         \
  GEMV_UNROLL(GEMV_INIT_COMPLEX, N)             \
  Index j = j2;                                 \
  do {                                          \
    const RhsScalar& b1 = rhs2(j, 0);           \
    RhsScalar* b = const_cast<RhsScalar*>(&b1); \
    GEMV_UNROLL(GEMV_PREFETCH, N)               \
    GEMV_UNROLL(GEMV_WORK_COL_COMPLEX, N)       \
  } while (++j < jend);                         \
  GEMV_UNROLL(GEMV_STORE_COL_COMPLEX, N)        \
  i += (ResPacketSize * N);

#if defined(USE_GEMV_MMA) && (EIGEN_COMP_LLVM || defined(USE_SLOWER_GEMV_MMA))
#define USE_GEMV_COL_COMPLEX_MMA
#endif

#ifdef USE_GEMV_COL_COMPLEX_MMA
#define GEMV_PROCESS_COL_COMPLEX(N) GEMV_PROCESS_COL_COMPLEX_ONE_MMA(N)
#else
#if defined(USE_GEMV_MMA) && (__GNUC__ > 10)
#define GEMV_PROCESS_COL_COMPLEX(N)          \
  if (sizeof(Scalar) != sizeof(LhsPacket)) { \
    GEMV_PROCESS_COL_COMPLEX_ONE_MMA(N)      \
  } else {                                   \
    GEMV_PROCESS_COL_COMPLEX_ONE(N)          \
  }
#else
#define GEMV_PROCESS_COL_COMPLEX(N) GEMV_PROCESS_COL_COMPLEX_ONE(N)
#endif
#endif

template <typename Scalar, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, bool LhsIsReal,
          typename RhsScalar, typename RhsMapper, bool ConjugateRhs, bool RhsIsReal, typename ResScalar>
EIGEN_STRONG_INLINE void gemv_complex_col(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs,
                                          ResScalar* res, Index resIncr, ResScalar alpha) {
  typedef gemv_traits<LhsScalar, RhsScalar> Traits;

  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;

  typedef typename packet_traits<Scalar>::type ScalarPacket;
  typedef typename packet_traits<LhsScalar>::type PLhsPacket;
  typedef typename packet_traits<ResScalar>::type PResPacket;
  typedef gemv_traits<ResPacket, ResPacket> PTraits;

  EIGEN_UNUSED_VARIABLE(resIncr);
  eigen_internal_assert(resIncr == 1);

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);
  RhsMapper rhs2(rhs);

  conj_helper<LhsScalar, RhsScalar, ConjugateLhs, ConjugateRhs> cj;

  const Index lhsStride = lhs.stride();
  // TODO: for padded aligned inputs, we could enable aligned reads
  enum {
    LhsAlignment = Unaligned,
    ResPacketSize = PTraits::ResPacketSize,
    LhsPacketSize = PTraits::LhsPacketSize,
    RhsPacketSize = PTraits::RhsPacketSize,
  };
#ifdef EIGEN_POWER_USE_GEMV_PREFETCH
  const Index prefetch_dist = 64 * LhsPacketSize;
#endif

#ifndef GCC_ONE_VECTORPAIR_BUG
  const Index n8 = rows - 8 * ResPacketSize + 1;
  const Index n4 = rows - 4 * ResPacketSize + 1;
  const Index n2 = rows - 2 * ResPacketSize + 1;
#endif
  const Index n1 = rows - 1 * ResPacketSize + 1;

  // TODO: improve the following heuristic:
  const Index block_cols = cols < 128 ? cols : (lhsStride * sizeof(LhsScalar) < 16000 ? 16 : 8);

  typedef alpha_store<PResPacket, ResPacket, ResScalar, Scalar> AlphaData;
  AlphaData alpha_data(alpha);

  for (Index j2 = 0; j2 < cols; j2 += block_cols) {
    Index jend = numext::mini(j2 + block_cols, cols);
    Index i = 0;
    PResPacket c00, c01, c02, c03, c04, c05, c06, c07;
    ResPacket c10, c11, c12, c13, c14, c15, c16, c17;
    PLhsPacket f0, f1, f2, f3, f4, f5, f6, f7;
#ifdef USE_GEMV_MMA
    __vector_quad e00, e01, e02, e03, e04, e05, e06, e07;
    __vector_pair a0, a1, a2, a3, a4, a5, a6, a7;
    PacketBlock<ScalarPacket, 4> result00, result01, result02, result03, result04, result05, result06, result07;
    GEMV_UNUSED(8, e0)
    GEMV_UNUSED(8, result0)
    GEMV_UNUSED(8, a)
    GEMV_UNUSED(8, f)
#if !defined(GCC_ONE_VECTORPAIR_BUG) && defined(USE_GEMV_COL_COMPLEX_MMA)
    if (GEMV_IS_COMPLEX_COMPLEX || !GEMV_IS_COMPLEX_FLOAT)
#endif
#endif
#ifndef GCC_ONE_VECTORPAIR_BUG
    {
      while (i < n8) {
        GEMV_PROCESS_COL_COMPLEX(8)
      }
    }
    while (i < n4) {
      GEMV_PROCESS_COL_COMPLEX(4)
    }
    if (i < n2) {
      GEMV_PROCESS_COL_COMPLEX(2)
    }
    if (i < n1)
#else
    while (i < n1)
#endif
    {
      GEMV_PROCESS_COL_COMPLEX_ONE(1)
    }
    for (; i < rows; ++i) {
      ResScalar d0(0);
      Index j = j2;
      do {
        d0 += cj.pmul(lhs(i, j), rhs2(j, 0));
      } while (++j < jend);
      res[i] += alpha * d0;
    }
  }
}

template <typename Scalar, int N>
struct ScalarBlock {
  Scalar scalar[N];
};

#ifdef USE_GEMV_MMA
static Packet16uc p16uc_ELEMENT_3 = {0x0c, 0x0d, 0x0e, 0x0f, 0x1c, 0x1d, 0x1e, 0x1f,
                                     0x0c, 0x0d, 0x0e, 0x0f, 0x1c, 0x1d, 0x1e, 0x1f};

/** \internal predux (add elements of a vector) from a MMA accumulator - real results */
template <typename ResScalar, typename ResPacket>
EIGEN_ALWAYS_INLINE ScalarBlock<ResScalar, 2> predux_real(__vector_quad* acc0, __vector_quad* acc1) {
  PacketBlock<ResPacket, 4> result0, result1;
  __builtin_mma_disassemble_acc(&result0.packet, acc0);
  __builtin_mma_disassemble_acc(&result1.packet, acc1);
  result0.packet[0] = vec_mergeh(result0.packet[0], result1.packet[0]);
  result0.packet[1] = vec_mergeo(result0.packet[1], result1.packet[1]);
  result0.packet[2] = vec_mergel(result0.packet[2], result1.packet[2]);
  result0.packet[3] = vec_perm(result0.packet[3], result1.packet[3], p16uc_ELEMENT_3);
  result0.packet[0] =
      vec_add(vec_add(result0.packet[0], result0.packet[2]), vec_add(result0.packet[1], result0.packet[3]));
  return *reinterpret_cast<ScalarBlock<ResScalar, 2>*>(&result0.packet[0]);
}

template <>
EIGEN_ALWAYS_INLINE ScalarBlock<double, 2> predux_real<double, Packet2d>(__vector_quad* acc0, __vector_quad* acc1) {
  PacketBlock<Packet2d, 4> result0, result1;
  __builtin_mma_disassemble_acc(&result0.packet, acc0);
  __builtin_mma_disassemble_acc(&result1.packet, acc1);
  result0.packet[0] =
      vec_add(vec_mergeh(result0.packet[0], result1.packet[0]), vec_mergel(result0.packet[1], result1.packet[1]));
  return *reinterpret_cast<ScalarBlock<double, 2>*>(&result0.packet[0]);
}

/** \internal add complex results together */
template <typename LhsPacket, typename RhsPacket, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_ALWAYS_INLINE ScalarBlock<std::complex<float>, 2> addComplexResults(PacketBlock<Packet4f, 4>& result0,
                                                                          PacketBlock<Packet4f, 4>& result1) {
  ScalarBlock<std::complex<float>, 2> cc0;
  result0.packet[0] = reinterpret_cast<Packet4f>(
      vec_mergeh(reinterpret_cast<Packet2d>(result0.packet[0]), reinterpret_cast<Packet2d>(result1.packet[0])));
  result0.packet[2] = reinterpret_cast<Packet4f>(
      vec_mergel(reinterpret_cast<Packet2d>(result0.packet[2]), reinterpret_cast<Packet2d>(result1.packet[2])));
  result0.packet[0] = vec_add(result0.packet[0], result0.packet[2]);
  if (GEMV_IS_COMPLEX_COMPLEX) {
    result0.packet[1] = reinterpret_cast<Packet4f>(
        vec_mergeh(reinterpret_cast<Packet2d>(result0.packet[1]), reinterpret_cast<Packet2d>(result1.packet[1])));
    result0.packet[3] = reinterpret_cast<Packet4f>(
        vec_mergel(reinterpret_cast<Packet2d>(result0.packet[3]), reinterpret_cast<Packet2d>(result1.packet[3])));
    result0.packet[1] = vec_add(result0.packet[1], result0.packet[3]);
    if (ConjugateLhs) {
      result0.packet[0] = pconj2(convertComplex(result0.packet[0])).v;
      result0.packet[1] = pcplxflip2(convertComplex(result0.packet[1])).v;
    } else if (ConjugateRhs) {
      result0.packet[1] = pcplxconjflip(convertComplex(result0.packet[1])).v;
    } else {
      result0.packet[1] = pcplxflipconj(convertComplex(result0.packet[1])).v;
    }
    result0.packet[0] = vec_add(result0.packet[0], result0.packet[1]);
  } else {
    if (ConjugateLhs && (sizeof(LhsPacket) == sizeof(std::complex<float>))) {
      result0.packet[0] = pconj2(convertComplex(result0.packet[0])).v;
    }
  }
  cc0.scalar[0].real(result0.packet[0][0]);
  cc0.scalar[0].imag(result0.packet[0][1]);
  cc0.scalar[1].real(result0.packet[0][2]);
  cc0.scalar[1].imag(result0.packet[0][3]);
  return cc0;
}

template <typename LhsPacket, typename RhsPacket, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_ALWAYS_INLINE ScalarBlock<std::complex<double>, 2> addComplexResults(PacketBlock<Packet2d, 4>&,
                                                                           PacketBlock<Packet2d, 4>&) {
  ScalarBlock<std::complex<double>, 2> cc0;
  EIGEN_UNUSED_VARIABLE(cc0);
  return cc0;  // Just for compilation
}

/** \internal predux (add elements of a vector) from a MMA accumulator - complex results */
template <typename ResScalar, typename ResPacket, typename LhsPacket, typename RhsPacket, bool ConjugateLhs,
          bool ConjugateRhs>
EIGEN_ALWAYS_INLINE ScalarBlock<ResScalar, 2> predux_complex(__vector_quad* acc0, __vector_quad* acc1) {
  PacketBlock<ResPacket, 4> result0, result1;
  __builtin_mma_disassemble_acc(&result0.packet, acc0);
  __builtin_mma_disassemble_acc(&result1.packet, acc1);
  return addComplexResults<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>(result0, result1);
}

template <typename ResScalar, typename ResPacket>
EIGEN_ALWAYS_INLINE ScalarBlock<ResScalar, 2> predux_real(__vector_quad* acc0) {
  PacketBlock<ResPacket, 4> result0;
  __builtin_mma_disassemble_acc(&result0.packet, acc0);
  result0.packet[0] =
      vec_add(vec_mergeh(result0.packet[0], result0.packet[2]), vec_mergel(result0.packet[1], result0.packet[3]));
  return *reinterpret_cast<ScalarBlock<ResScalar, 2>*>(&result0.packet[0]);
}

template <typename ResScalar, typename ResPacket, typename LhsPacket, typename RhsPacket, bool ConjugateLhs,
          bool ConjugateRhs>
EIGEN_ALWAYS_INLINE ScalarBlock<ResScalar, 2> predux_complex(__vector_quad* acc0) {
  ScalarBlock<ResScalar, 2> cc0;
  PacketBlock<ResPacket, 4> result0;
  __builtin_mma_disassemble_acc(&result0.packet, acc0);
  if (GEMV_IS_COMPLEX_COMPLEX) {
    if (ConjugateLhs) {
      result0.packet[1] = pconjinv(convertComplex(result0.packet[1])).v;
      result0.packet[3] = pconjinv(convertComplex(result0.packet[3])).v;
    } else if (ConjugateRhs) {
      result0.packet[0] = pconj2(convertComplex(result0.packet[0])).v;
      result0.packet[2] = pconj2(convertComplex(result0.packet[2])).v;
    } else {
      result0.packet[1] = pconj2(convertComplex(result0.packet[1])).v;
      result0.packet[3] = pconj2(convertComplex(result0.packet[3])).v;
    }
    result0.packet[0] = vec_add(result0.packet[0], __builtin_vsx_xxpermdi(result0.packet[1], result0.packet[1], 2));
    result0.packet[2] = vec_add(result0.packet[2], __builtin_vsx_xxpermdi(result0.packet[3], result0.packet[3], 2));
  } else {
    result0.packet[0] = __builtin_vsx_xxpermdi(result0.packet[0], result0.packet[1], 1);
    result0.packet[2] = __builtin_vsx_xxpermdi(result0.packet[2], result0.packet[3], 1);
  }
  cc0.scalar[0].real(result0.packet[0][0]);
  cc0.scalar[0].imag(result0.packet[0][1]);
  cc0.scalar[1].real(result0.packet[2][0]);
  cc0.scalar[1].imag(result0.packet[2][1]);
  return cc0;
}
#endif

template <typename ResScalar, typename ResPacket>
EIGEN_ALWAYS_INLINE ScalarBlock<ResScalar, 2> predux_real(ResPacket& a, ResPacket& b) {
  ScalarBlock<ResScalar, 2> cc0;
  cc0.scalar[0] = predux(a);
  cc0.scalar[1] = predux(b);
  return cc0;
}

template <typename ResScalar, typename ResPacket>
EIGEN_ALWAYS_INLINE ScalarBlock<ResScalar, 2> predux_complex(ResPacket& a, ResPacket& b) {
  return predux_real<ResScalar, ResPacket>(a, b);
}

#define GEMV_UNROLL_ROW(func, N) func(0, N) func(1, N) func(2, N) func(3, N) func(4, N) func(5, N) func(6, N) func(7, N)

#define GEMV_UNROLL_ROW_HALF(func, N) func(0, 0, 1, N) func(1, 2, 3, N) func(2, 4, 5, N) func(3, 6, 7, N)

#define GEMV_LOADPACKET_ROW(iter) lhs.template load<LhsPacket, Unaligned>(i + (iter), j)

#ifdef USE_GEMV_MMA
#define GEMV_UNROLL3_ROW(func, N, which)                                                                      \
  func(0, N, which) func(1, N, which) func(2, N, which) func(3, N, which) func(4, N, which) func(5, N, which) \
      func(6, N, which) func(7, N, which)

#define GEMV_UNUSED_ROW(N, which) GEMV_UNROLL3_ROW(GEMV_UNUSED_VAR, N, which)

#define GEMV_INIT_ROW(iter, N)         \
  if (GEMV_GETN(N) > iter) {           \
    __builtin_mma_xxsetaccz(&c##iter); \
  }

#define GEMV_LOADPAIR_ROW(iter1, iter2) \
  GEMV_BUILDPAIR_MMA(b##iter1, GEMV_LOADPACKET_ROW(iter2), GEMV_LOADPACKET_ROW((iter2) + 1));

#define GEMV_WORK_ROW(iter, N)                                                              \
  if (GEMV_GETN(N) > iter) {                                                                \
    if (GEMV_IS_FLOAT) {                                                                    \
      pger_vecMMA_acc<LhsPacket, RhsPacket, true>(&c##iter, a0, GEMV_LOADPACKET_ROW(iter)); \
    } else {                                                                                \
      __vector_pair b##iter;                                                                \
      GEMV_LOADPAIR_ROW(iter, iter << 1)                                                    \
      pger_vecMMA_acc<LhsPacket, RhsPacket, true>(&c##iter, b##iter, a0);                   \
    }                                                                                       \
  }

#define GEMV_PREDUX2(iter1, iter2, iter3, N)                               \
  if (N > iter1) {                                                         \
    if (GEMV_IS_FLOAT) {                                                   \
      cc##iter1 = predux_real<ResScalar, ResPacket>(&c##iter2, &c##iter3); \
    } else {                                                               \
      cc##iter1 = predux_real<ResScalar, ResPacket>(&c##iter1);            \
    }                                                                      \
  } else {                                                                 \
    EIGEN_UNUSED_VARIABLE(cc##iter1);                                      \
  }
#else
#define GEMV_INIT_ROW(iter, N)                \
  if (N > iter) {                             \
    c##iter = pset1<ResPacket>(ResScalar(0)); \
  } else {                                    \
    EIGEN_UNUSED_VARIABLE(c##iter);           \
  }

#define GEMV_WORK_ROW(iter, N)                                   \
  if (N > iter) {                                                \
    c##iter = pcj.pmadd(GEMV_LOADPACKET_ROW(iter), a0, c##iter); \
  }

#define GEMV_PREDUX2(iter1, iter2, iter3, N)                           \
  if (N > iter1) {                                                     \
    cc##iter1 = predux_real<ResScalar, ResPacket>(c##iter2, c##iter3); \
  } else {                                                             \
    EIGEN_UNUSED_VARIABLE(cc##iter1);                                  \
  }
#endif

#define GEMV_MULT(iter1, iter2, iter3, N)                  \
  if (N > iter1) {                                         \
    cc##iter1.scalar[0] += cj.pmul(lhs(i + iter2, j), a0); \
    cc##iter1.scalar[1] += cj.pmul(lhs(i + iter3, j), a0); \
  }

#define GEMV_STORE_ROW(iter1, iter2, iter3, N)                                           \
  if (N > iter1) {                                                                       \
    storeMaddData<ResScalar>(res + ((i + iter2) * resIncr), alpha, cc##iter1.scalar[0]); \
    storeMaddData<ResScalar>(res + ((i + iter3) * resIncr), alpha, cc##iter1.scalar[1]); \
  }

/** \internal main macro for gemv_row - initialize accumulators, multiply and add inputs, predux and store results */
#define GEMV_PROCESS_ROW(N)                                       \
  for (; i < n##N; i += N) {                                      \
    GEMV_UNROLL_ROW(GEMV_INIT_ROW, N)                             \
    Index j = 0;                                                  \
    for (; j + LhsPacketSize <= cols; j += LhsPacketSize) {       \
      RhsPacket a0 = rhs2.template load<RhsPacket, Unaligned>(j); \
      GEMV_UNROLL_ROW(GEMV_WORK_ROW, N)                           \
    }                                                             \
    GEMV_UNROLL_ROW_HALF(GEMV_PREDUX2, (N >> 1))                  \
    for (; j < cols; ++j) {                                       \
      RhsScalar a0 = rhs2(j);                                     \
      GEMV_UNROLL_ROW_HALF(GEMV_MULT, (N >> 1))                   \
    }                                                             \
    GEMV_UNROLL_ROW_HALF(GEMV_STORE_ROW, (N >> 1))                \
  }

template <typename LhsScalar, typename LhsMapper, typename RhsScalar, typename RhsMapper, typename ResScalar>
EIGEN_STRONG_INLINE void gemv_row(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs, ResScalar* res,
                                  Index resIncr, ResScalar alpha) {
  typedef gemv_traits<LhsScalar, RhsScalar> Traits;

  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);
  typename RhsMapper::LinearMapper rhs2 = rhs.getLinearMapper(0, 0);

  eigen_internal_assert(rhs.stride() == 1);
  conj_helper<LhsScalar, RhsScalar, false, false> cj;
  conj_helper<LhsPacket, RhsPacket, false, false> pcj;

  // TODO: fine tune the following heuristic. The rationale is that if the matrix is very large,
  //       processing 8 rows at once might be counter productive wrt cache.
#ifndef GCC_ONE_VECTORPAIR_BUG
  const Index n8 = lhs.stride() * sizeof(LhsScalar) > 32000 ? (rows - 7) : (rows - 7);
  const Index n4 = rows - 3;
  const Index n2 = rows - 1;
#endif

  // TODO: for padded aligned inputs, we could enable aligned reads
  enum {
    LhsAlignment = Unaligned,
    ResPacketSize = Traits::ResPacketSize,
    LhsPacketSize = Traits::LhsPacketSize,
    RhsPacketSize = Traits::RhsPacketSize,
  };

  Index i = 0;
#ifdef USE_GEMV_MMA
  __vector_quad c0, c1, c2, c3, c4, c5, c6, c7;
  GEMV_UNUSED_ROW(8, c)
#else
  ResPacket c0, c1, c2, c3, c4, c5, c6, c7;
#endif
#ifndef GCC_ONE_VECTORPAIR_BUG
  ScalarBlock<ResScalar, 2> cc0, cc1, cc2, cc3;
  GEMV_PROCESS_ROW(8)
  GEMV_PROCESS_ROW(4)
  GEMV_PROCESS_ROW(2)
#endif
  for (; i < rows; ++i) {
    ResPacket d0 = pset1<ResPacket>(ResScalar(0));
    Index j = 0;
    for (; j + LhsPacketSize <= cols; j += LhsPacketSize) {
      RhsPacket b0 = rhs2.template load<RhsPacket, Unaligned>(j);

      d0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 0, j), b0, d0);
    }
    ResScalar dd0 = predux(d0);
    for (; j < cols; ++j) {
      dd0 += cj.pmul(lhs(i, j), rhs2(j));
    }
    res[i * resIncr] += alpha * dd0;
  }
}

#define EIGEN_POWER_GEMV_REAL_SPECIALIZE_COL(Scalar)                                                                   \
  template <typename Index, typename LhsMapper, bool ConjugateLhs, typename RhsMapper, bool ConjugateRhs, int Version> \
  struct general_matrix_vector_product<Index, Scalar, LhsMapper, ColMajor, ConjugateLhs, Scalar, RhsMapper,            \
                                       ConjugateRhs, Version> {                                                        \
    typedef typename ScalarBinaryOpTraits<Scalar, Scalar>::ReturnType ResScalar;                                       \
                                                                                                                       \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run(Index rows, Index cols, const LhsMapper& lhs,                  \
                                                        const RhsMapper& rhs, ResScalar* res, Index resIncr,           \
                                                        ResScalar alpha) {                                             \
      gemv_col<Scalar, LhsMapper, Scalar, RhsMapper, ResScalar>(rows, cols, lhs, rhs, res, resIncr, alpha);            \
    }                                                                                                                  \
  };

#define EIGEN_POWER_GEMV_REAL_SPECIALIZE_ROW(Scalar)                                                                   \
  template <typename Index, typename LhsMapper, bool ConjugateLhs, typename RhsMapper, bool ConjugateRhs, int Version> \
  struct general_matrix_vector_product<Index, Scalar, LhsMapper, RowMajor, ConjugateLhs, Scalar, RhsMapper,            \
                                       ConjugateRhs, Version> {                                                        \
    typedef typename ScalarBinaryOpTraits<Scalar, Scalar>::ReturnType ResScalar;                                       \
                                                                                                                       \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run(Index rows, Index cols, const LhsMapper& lhs,                  \
                                                        const RhsMapper& rhs, ResScalar* res, Index resIncr,           \
                                                        ResScalar alpha) {                                             \
      gemv_row<Scalar, LhsMapper, Scalar, RhsMapper, ResScalar>(rows, cols, lhs, rhs, res, resIncr, alpha);            \
    }                                                                                                                  \
  };

EIGEN_POWER_GEMV_REAL_SPECIALIZE_COL(float)
EIGEN_POWER_GEMV_REAL_SPECIALIZE_COL(double)
EIGEN_POWER_GEMV_REAL_SPECIALIZE_ROW(float)
EIGEN_POWER_GEMV_REAL_SPECIALIZE_ROW(double)

#ifdef USE_GEMV_MMA
#define gemv_bf16_col gemvMMA_bfloat16_col
#define gemv_bf16_row gemvMMA_bfloat16_row
#else
#define gemv_bf16_col gemv_bfloat16_col
#define gemv_bf16_row gemv_bfloat16_row
#endif

#define EIGEN_POWER_GEMV_REAL_SPECIALIZE_COL_BFLOAT16()                                                                \
  template <typename Index, typename LhsMapper, bool ConjugateLhs, typename RhsMapper, bool ConjugateRhs, int Version> \
  struct general_matrix_vector_product<Index, bfloat16, LhsMapper, ColMajor, ConjugateLhs, bfloat16, RhsMapper,        \
                                       ConjugateRhs, Version> {                                                        \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run(Index rows, Index cols, const LhsMapper& lhs,                  \
                                                        const RhsMapper& rhs, bfloat16* res, Index resIncr,            \
                                                        bfloat16 alpha) {                                              \
      gemv_bf16_col<LhsMapper, RhsMapper>(rows, cols, lhs, rhs, res, resIncr, alpha);                                  \
    }                                                                                                                  \
  };

#define EIGEN_POWER_GEMV_REAL_SPECIALIZE_ROW_BFLOAT16()                                                                \
  template <typename Index, typename LhsMapper, bool ConjugateLhs, typename RhsMapper, bool ConjugateRhs, int Version> \
  struct general_matrix_vector_product<Index, bfloat16, LhsMapper, RowMajor, ConjugateLhs, bfloat16, RhsMapper,        \
                                       ConjugateRhs, Version> {                                                        \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run(Index rows, Index cols, const LhsMapper& lhs,                  \
                                                        const RhsMapper& rhs, bfloat16* res, Index resIncr,            \
                                                        bfloat16 alpha) {                                              \
      gemv_bf16_row<LhsMapper, RhsMapper>(rows, cols, lhs, rhs, res, resIncr, alpha);                                  \
    }                                                                                                                  \
  };

EIGEN_POWER_GEMV_REAL_SPECIALIZE_COL_BFLOAT16()
EIGEN_POWER_GEMV_REAL_SPECIALIZE_ROW_BFLOAT16()

template <typename ResScalar, typename PResPacket, typename ResPacket, typename LhsPacket, typename RhsPacket>
EIGEN_ALWAYS_INLINE ScalarBlock<ResScalar, 2> predux_complex(PResPacket& a0, PResPacket& b0, ResPacket& a1,
                                                             ResPacket& b1) {
  if (GEMV_IS_COMPLEX_COMPLEX) {
    a0 = padd(a0, a1);
    b0 = padd(b0, b1);
  }
  return predux_complex<ResScalar, PResPacket>(a0, b0);
}

#define GEMV_LOADPACKET_ROW_COMPLEX(iter) loadLhsPacket<Scalar, LhsScalar, LhsMapper, PLhsPacket>(lhs, i + (iter), j)

#define GEMV_LOADPACKET_ROW_COMPLEX_DATA(iter) convertReal(GEMV_LOADPACKET_ROW_COMPLEX(iter))

#define GEMV_PROCESS_ROW_COMPLEX_SINGLE_WORK(which, N)    \
  j = 0;                                                  \
  for (; j + LhsPacketSize <= cols; j += LhsPacketSize) { \
    const RhsScalar& b1 = rhs2(j);                        \
    RhsScalar* b = const_cast<RhsScalar*>(&b1);           \
    GEMV_UNROLL_ROW(which, N)                             \
  }

#define GEMV_PROCESS_END_ROW_COMPLEX(N)               \
  for (; j < cols; ++j) {                             \
    RhsScalar b0 = rhs2(j);                           \
    GEMV_UNROLL_ROW_HALF(GEMV_MULT_COMPLEX, (N >> 1)) \
  }                                                   \
  GEMV_UNROLL_ROW_HALF(GEMV_STORE_ROW_COMPLEX, (N >> 1))

#ifdef USE_GEMV_MMA
#define GEMV_INIT_ROW_COMPLEX_MMA(iter, N) \
  if (GEMV_GETN_COMPLEX(N) > iter) {       \
    __builtin_mma_xxsetaccz(&e0##iter);    \
  }

#define GEMV_LOADPAIR_ROW_COMPLEX_MMA(iter1, iter2) \
  GEMV_BUILDPAIR_MMA(a##iter1, GEMV_LOADPACKET_ROW_COMPLEX_DATA(iter2), GEMV_LOADPACKET_ROW_COMPLEX_DATA((iter2) + 1));

#define GEMV_WORK_ROW_COMPLEX_MMA(iter, N)                                                                       \
  if (GEMV_GETN_COMPLEX(N) > iter) {                                                                             \
    if (GEMV_IS_COMPLEX_FLOAT) {                                                                                 \
      PLhsPacket a##iter = GEMV_LOADPACKET_ROW_COMPLEX(iter);                                                    \
      gemv_mult_complex_MMA<ScalarPacket, LhsScalar, PLhsPacket, PLhsPacket, RhsScalar, RhsPacket, ResPacket,    \
                            ConjugateLhs, ConjugateRhs, RowMajor>(a##iter, b, &e0##iter);                        \
    } else {                                                                                                     \
      __vector_pair a##iter;                                                                                     \
      GEMV_LOADPAIR_ROW_COMPLEX_MMA(iter, iter << 1)                                                             \
      gemv_mult_complex_MMA<ScalarPacket, LhsScalar, PLhsPacket, __vector_pair, RhsScalar, RhsPacket, ResPacket, \
                            ConjugateLhs, ConjugateRhs, RowMajor>(a##iter, b, &e0##iter);                        \
    }                                                                                                            \
  }

#define GEMV_PREDUX4_COMPLEX_MMA(iter1, iter2, iter3, N)                                                         \
  if (N > iter1) {                                                                                               \
    if (GEMV_IS_COMPLEX_FLOAT) {                                                                                 \
      cc##iter1 = predux_complex<ResScalar, ScalarPacket, LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>(     \
          &e0##iter2, &e0##iter3);                                                                               \
    } else {                                                                                                     \
      cc##iter1 =                                                                                                \
          predux_complex<ResScalar, ScalarPacket, LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs>(&e0##iter1); \
    }                                                                                                            \
  } else {                                                                                                       \
    EIGEN_UNUSED_VARIABLE(cc##iter1);                                                                            \
  }

#define GEMV_PROCESS_ROW_COMPLEX_SINGLE_MMA(N)  \
  GEMV_UNROLL_ROW(GEMV_INIT_ROW_COMPLEX_MMA, N) \
  GEMV_PROCESS_ROW_COMPLEX_SINGLE_WORK(GEMV_WORK_ROW_COMPLEX_MMA, N)

#define GEMV_PROCESS_ROW_COMPLEX_ONE_MMA(N)                  \
  for (; i < n##N; i += N) {                                 \
    GEMV_PROCESS_ROW_COMPLEX_SINGLE_MMA(N)                   \
    GEMV_UNROLL_ROW_HALF(GEMV_PREDUX4_COMPLEX_MMA, (N >> 1)) \
    GEMV_PROCESS_END_ROW_COMPLEX(N);                         \
  }
#endif

#define GEMV_WORK_ROW_COMPLEX(iter, N)                                                                     \
  if (N > iter) {                                                                                          \
    PLhsPacket a##iter = GEMV_LOADPACKET_ROW_COMPLEX(iter);                                                \
    gemv_mult_complex<ScalarPacket, PLhsPacket, RhsScalar, RhsPacket, PResPacket, ResPacket, ConjugateLhs, \
                      ConjugateRhs, RowMajor>(a##iter, b, c0##iter, c1##iter);                             \
  }

#define GEMV_PREDUX4_COMPLEX(iter1, iter2, iter3, N)                                                          \
  if (N > iter1) {                                                                                            \
    cc##iter1 = predux_complex<ResScalar, PResPacket, ResPacket, LhsPacket, RhsPacket>(c0##iter2, c0##iter3,  \
                                                                                       c1##iter2, c1##iter3); \
  } else {                                                                                                    \
    EIGEN_UNUSED_VARIABLE(cc##iter1);                                                                         \
  }

#define GEMV_MULT_COMPLEX(iter1, iter2, iter3, N)          \
  if (N > iter1) {                                         \
    cc##iter1.scalar[0] += cj.pmul(lhs(i + iter2, j), b0); \
    cc##iter1.scalar[1] += cj.pmul(lhs(i + iter3, j), b0); \
  }

#define GEMV_STORE_ROW_COMPLEX(iter1, iter2, iter3, N)                                   \
  if (N > iter1) {                                                                       \
    storeMaddData<ResScalar>(res + ((i + iter2) * resIncr), alpha, cc##iter1.scalar[0]); \
    storeMaddData<ResScalar>(res + ((i + iter3) * resIncr), alpha, cc##iter1.scalar[1]); \
  }

#define GEMV_PROCESS_ROW_COMPLEX_SINGLE_NEW(N) \
  GEMV_UNROLL_ROW(GEMV_INIT_COMPLEX, N)        \
  GEMV_PROCESS_ROW_COMPLEX_SINGLE_WORK(GEMV_WORK_ROW_COMPLEX, N)

/** \internal main macro for gemv_complex_row - initialize accumulators, multiply and add inputs, predux and store
 * results */
#define GEMV_PROCESS_ROW_COMPLEX_ONE_NEW(N)              \
  for (; i < n##N; i += N) {                             \
    GEMV_PROCESS_ROW_COMPLEX_SINGLE_NEW(N)               \
    GEMV_UNROLL_ROW_HALF(GEMV_PREDUX4_COMPLEX, (N >> 1)) \
    GEMV_PROCESS_END_ROW_COMPLEX(N);                     \
  }

#define GEMV_PROCESS_ROW_COMPLEX_PREDUX_NEW(iter) \
  if (GEMV_IS_COMPLEX_COMPLEX) {                  \
    c0##iter = padd(c0##iter, c1##iter);          \
  }                                               \
  dd0 = predux(c0##iter);

#if EIGEN_COMP_LLVM
#define GEMV_PROCESS_ROW_COMPLEX_SINGLE(N) GEMV_PROCESS_ROW_COMPLEX_SINGLE_NEW(N)

#define GEMV_PROCESS_ROW_COMPLEX_ONE(N) GEMV_PROCESS_ROW_COMPLEX_ONE_NEW(N)

#define GEMV_PROCESS_ROW_COMPLEX_PREDUX(iter) GEMV_PROCESS_ROW_COMPLEX_PREDUX_NEW(iter)
#else
// gcc seems to be reading and writing registers unnecessarily to memory.
// Use the old way for complex double until it is fixed.

#define GEMV_LOADPACKET_ROW_COMPLEX_OLD(iter) lhs.template load<LhsPacket, LhsAlignment>(i + (iter), j)

#define GEMV_INIT_COMPLEX_OLD(iter, N) \
  EIGEN_UNUSED_VARIABLE(c0##iter);     \
  if (N > iter) {                      \
    c1##iter = pset_zero<ResPacket>(); \
  } else {                             \
    EIGEN_UNUSED_VARIABLE(c1##iter);   \
  }

#define GEMV_WORK_ROW_COMPLEX_OLD(iter, N)                     \
  if (N > iter) {                                              \
    LhsPacket a##iter = GEMV_LOADPACKET_ROW_COMPLEX_OLD(iter); \
    c1##iter = pcj.pmadd(a##iter, b0, c1##iter);               \
  }

#define GEMV_PREDUX4_COMPLEX_OLD(iter1, iter2, iter3, N) \
  if (N > iter1) {                                       \
    cc##iter1.scalar[0] = predux(c1##iter2);             \
    cc##iter1.scalar[1] = predux(c1##iter3);             \
  } else {                                               \
    EIGEN_UNUSED_VARIABLE(cc##iter1);                    \
  }

#define GEMV_PROCESS_ROW_COMPLEX_SINGLE_OLD(N)                  \
  GEMV_UNROLL_ROW(GEMV_INIT_COMPLEX_OLD, N)                     \
  j = 0;                                                        \
  for (; j + LhsPacketSize <= cols; j += LhsPacketSize) {       \
    RhsPacket b0 = rhs2.template load<RhsPacket, Unaligned>(j); \
    GEMV_UNROLL_ROW(GEMV_WORK_ROW_COMPLEX_OLD, N)               \
  }

#define GEMV_PROCESS_ROW_COMPLEX_ONE_OLD(N)                  \
  for (; i < n##N; i += N) {                                 \
    GEMV_PROCESS_ROW_COMPLEX_SINGLE_OLD(N)                   \
    GEMV_UNROLL_ROW_HALF(GEMV_PREDUX4_COMPLEX_OLD, (N >> 1)) \
    GEMV_PROCESS_END_ROW_COMPLEX(N)                          \
  }

#define GEMV_PROCESS_ROW_COMPLEX_PREDUX_OLD(iter) dd0 = predux(c1##iter);

#if (__GNUC__ > 10)
#define GEMV_PROCESS_ROW_COMPLEX_IS_NEW 1
#else
#define GEMV_PROCESS_ROW_COMPLEX_IS_NEW (sizeof(Scalar) == sizeof(float)) || GEMV_IS_COMPLEX_COMPLEX
#endif

#define GEMV_PROCESS_ROW_COMPLEX_SINGLE(N) \
  if (GEMV_PROCESS_ROW_COMPLEX_IS_NEW) {   \
    GEMV_PROCESS_ROW_COMPLEX_SINGLE_NEW(N) \
  } else {                                 \
    GEMV_PROCESS_ROW_COMPLEX_SINGLE_OLD(N) \
  }

#define GEMV_PROCESS_ROW_COMPLEX_ONE(N)  \
  if (GEMV_PROCESS_ROW_COMPLEX_IS_NEW) { \
    GEMV_PROCESS_ROW_COMPLEX_ONE_NEW(N)  \
  } else {                               \
    GEMV_PROCESS_ROW_COMPLEX_ONE_OLD(N)  \
  }

#define GEMV_PROCESS_ROW_COMPLEX_PREDUX(iter) \
  if (GEMV_PROCESS_ROW_COMPLEX_IS_NEW) {      \
    GEMV_PROCESS_ROW_COMPLEX_PREDUX_NEW(iter) \
  } else {                                    \
    GEMV_PROCESS_ROW_COMPLEX_PREDUX_OLD(iter) \
  }
#endif

#ifdef USE_GEMV_MMA
#define GEMV_PROCESS_ROW_COMPLEX(N) GEMV_PROCESS_ROW_COMPLEX_ONE_MMA(N)
#else
#define GEMV_PROCESS_ROW_COMPLEX(N) GEMV_PROCESS_ROW_COMPLEX_ONE(N)
#endif

template <typename Scalar, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, bool LhsIsReal,
          typename RhsScalar, typename RhsMapper, bool ConjugateRhs, bool RhsIsReal, typename ResScalar>
EIGEN_STRONG_INLINE void gemv_complex_row(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs,
                                          ResScalar* res, Index resIncr, ResScalar alpha) {
  typedef gemv_traits<LhsScalar, RhsScalar> Traits;

  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;

  typedef typename packet_traits<Scalar>::type ScalarPacket;
  typedef typename packet_traits<LhsScalar>::type PLhsPacket;
  typedef typename packet_traits<ResScalar>::type PResPacket;
  typedef gemv_traits<ResPacket, ResPacket> PTraits;

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate proper code.
  LhsMapper lhs(alhs);
  typename RhsMapper::LinearMapper rhs2 = rhs.getLinearMapper(0, 0);

  eigen_internal_assert(rhs.stride() == 1);
  conj_helper<LhsScalar, RhsScalar, ConjugateLhs, ConjugateRhs> cj;
#if !EIGEN_COMP_LLVM
  conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs> pcj;
#endif

  // TODO: fine tune the following heuristic. The rationale is that if the matrix is very large,
  //       processing 8 rows at once might be counter productive wrt cache.
#ifndef GCC_ONE_VECTORPAIR_BUG
  const Index n8 = lhs.stride() * sizeof(LhsScalar) > 32000 ? (rows - 7) : (rows - 7);
  const Index n4 = rows - 3;
  const Index n2 = rows - 1;
#endif

  // TODO: for padded aligned inputs, we could enable aligned reads
  enum {
    LhsAlignment = Unaligned,
    ResPacketSize = PTraits::ResPacketSize,
    LhsPacketSize = PTraits::LhsPacketSize,
    RhsPacketSize = PTraits::RhsPacketSize,
  };

  Index i = 0, j;
  PResPacket c00, c01, c02, c03, c04, c05, c06, c07;
  ResPacket c10, c11, c12, c13, c14, c15, c16, c17;
#ifdef USE_GEMV_MMA
  __vector_quad e00, e01, e02, e03, e04, e05, e06, e07;
  GEMV_UNUSED_ROW(8, e0)
  GEMV_UNUSED_EXTRA(1, c0)
  GEMV_UNUSED_EXTRA(1, c1)
#endif
  ResScalar dd0;
#ifndef GCC_ONE_VECTORPAIR_BUG
  ScalarBlock<ResScalar, 2> cc0, cc1, cc2, cc3;
#ifdef USE_GEMV_MMA
  if (!GEMV_IS_COMPLEX_COMPLEX)
#endif
  {
    GEMV_PROCESS_ROW_COMPLEX(8)
  }
  GEMV_PROCESS_ROW_COMPLEX(4)
  GEMV_PROCESS_ROW_COMPLEX(2)
#endif
  for (; i < rows; ++i) {
    GEMV_PROCESS_ROW_COMPLEX_SINGLE(1)
    GEMV_PROCESS_ROW_COMPLEX_PREDUX(0)
    for (; j < cols; ++j) {
      dd0 += cj.pmul(lhs(i, j), rhs2(j));
    }
    res[i * resIncr] += alpha * dd0;
  }
}

#define EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_COL(Scalar, LhsScalar, RhsScalar)                                          \
  template <typename Index, typename LhsMapper, bool ConjugateLhs, typename RhsMapper, bool ConjugateRhs, int Version> \
  struct general_matrix_vector_product<Index, LhsScalar, LhsMapper, ColMajor, ConjugateLhs, RhsScalar, RhsMapper,      \
                                       ConjugateRhs, Version> {                                                        \
    typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;                                 \
                                                                                                                       \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run(Index rows, Index cols, const LhsMapper& lhs,                  \
                                                        const RhsMapper& rhs, ResScalar* res, Index resIncr,           \
                                                        ResScalar alpha) {                                             \
      gemv_complex_col<Scalar, LhsScalar, LhsMapper, ConjugateLhs, sizeof(Scalar) == sizeof(LhsScalar), RhsScalar,     \
                       RhsMapper, ConjugateRhs, sizeof(Scalar) == sizeof(RhsScalar), ResScalar>(rows, cols, lhs, rhs,  \
                                                                                                res, resIncr, alpha);  \
    }                                                                                                                  \
  };

#define EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_ROW(Scalar, LhsScalar, RhsScalar)                                          \
  template <typename Index, typename LhsMapper, bool ConjugateLhs, typename RhsMapper, bool ConjugateRhs, int Version> \
  struct general_matrix_vector_product<Index, LhsScalar, LhsMapper, RowMajor, ConjugateLhs, RhsScalar, RhsMapper,      \
                                       ConjugateRhs, Version> {                                                        \
    typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;                                 \
                                                                                                                       \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run(Index rows, Index cols, const LhsMapper& lhs,                  \
                                                        const RhsMapper& rhs, ResScalar* res, Index resIncr,           \
                                                        ResScalar alpha) {                                             \
      gemv_complex_row<Scalar, LhsScalar, LhsMapper, ConjugateLhs, sizeof(Scalar) == sizeof(LhsScalar), RhsScalar,     \
                       RhsMapper, ConjugateRhs, sizeof(Scalar) == sizeof(RhsScalar), ResScalar>(rows, cols, lhs, rhs,  \
                                                                                                res, resIncr, alpha);  \
    }                                                                                                                  \
  };

EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_COL(float, float, std::complex<float>)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_COL(float, std::complex<float>, float)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_COL(float, std::complex<float>, std::complex<float>)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_COL(double, double, std::complex<double>)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_COL(double, std::complex<double>, double)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_COL(double, std::complex<double>, std::complex<double>)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_ROW(float, float, std::complex<float>)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_ROW(float, std::complex<float>, float)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_ROW(float, std::complex<float>, std::complex<float>)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_ROW(double, double, std::complex<double>)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_ROW(double, std::complex<double>, double)
EIGEN_POWER_GEMV_COMPLEX_SPECIALIZE_ROW(double, std::complex<double>, std::complex<double>)

#endif  // EIGEN_MATRIX_VECTOR_PRODUCT_ALTIVEC_H
