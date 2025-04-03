// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Intel Corporation
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CORE_ARCH_AVX512_TRSM_UNROLLS_H
#define EIGEN_CORE_ARCH_AVX512_TRSM_UNROLLS_H

template <bool isARowMajor = true>
EIGEN_ALWAYS_INLINE int64_t idA(int64_t i, int64_t j, int64_t LDA) {
  EIGEN_IF_CONSTEXPR(isARowMajor) return i * LDA + j;
  else return i + j * LDA;
}

/**
 * This namespace contains various classes used to generate compile-time unrolls which are
 * used throughout the trsm/gemm kernels. The unrolls are characterized as for-loops (1-D), nested
 * for-loops (2-D), or triple nested for-loops (3-D). Unrolls are generated using template recursion
 *
 * Example, the 2-D for-loop is unrolled recursively by first flattening to a 1-D loop.
 *
 * for(startI = 0; startI < endI; startI++)             for(startC = 0; startC < endI*endJ; startC++)
 *   for(startJ = 0; startJ < endJ; startJ++)  ---->      startI = (startC)/(endJ)
 *     func(startI,startJ)                                startJ = (startC)%(endJ)
 *                                                        func(...)
 *
 * The 1-D loop can be unrolled recursively by using enable_if and defining an auxillary function
 * with a template parameter used as a counter.
 *
 * template <endI, endJ, counter>
 * std::enable_if_t<(counter <= 0)>  <---- tail case.
 * aux_func {}
 *
 * template <endI, endJ, counter>
 * std::enable_if_t<(counter > 0)>   <---- actual for-loop
 * aux_func {
 *   startC = endI*endJ - counter
 *   startI = (startC)/(endJ)
 *   startJ = (startC)%(endJ)
 *   func(startI, startJ)
 *   aux_func<endI, endJ, counter-1>()
 * }
 *
 * Note: Additional wrapper functions are provided for aux_func which hides the counter template
 * parameter since counter usually depends on endI, endJ, etc...
 *
 * Conventions:
 * 1) endX: specifies the terminal value for the for-loop, (ex: for(startX = 0; startX < endX; startX++))
 *
 * 2) rem, remM, remK template parameters are used for deciding whether to use masked operations for
 *    handling remaining tails (when sizes are not multiples of PacketSize or EIGEN_AVX_MAX_NUM_ROW)
 */
namespace unrolls {

template <int64_t N>
EIGEN_ALWAYS_INLINE auto remMask(int64_t m) {
  EIGEN_IF_CONSTEXPR(N == 16) { return 0xFFFF >> (16 - m); }
  else EIGEN_IF_CONSTEXPR(N == 8) {
    return 0xFF >> (8 - m);
  }
  else EIGEN_IF_CONSTEXPR(N == 4) {
    return 0x0F >> (4 - m);
  }
  return 0;
}

template <typename Packet>
EIGEN_ALWAYS_INLINE void trans8x8blocks(PacketBlock<Packet, 8> &kernel);

template <>
EIGEN_ALWAYS_INLINE void trans8x8blocks(PacketBlock<Packet16f, 8> &kernel) {
  __m512 T0 = _mm512_unpacklo_ps(kernel.packet[0], kernel.packet[1]);
  __m512 T1 = _mm512_unpackhi_ps(kernel.packet[0], kernel.packet[1]);
  __m512 T2 = _mm512_unpacklo_ps(kernel.packet[2], kernel.packet[3]);
  __m512 T3 = _mm512_unpackhi_ps(kernel.packet[2], kernel.packet[3]);
  __m512 T4 = _mm512_unpacklo_ps(kernel.packet[4], kernel.packet[5]);
  __m512 T5 = _mm512_unpackhi_ps(kernel.packet[4], kernel.packet[5]);
  __m512 T6 = _mm512_unpacklo_ps(kernel.packet[6], kernel.packet[7]);
  __m512 T7 = _mm512_unpackhi_ps(kernel.packet[6], kernel.packet[7]);

  kernel.packet[0] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(T0), _mm512_castps_pd(T2)));
  kernel.packet[1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(T0), _mm512_castps_pd(T2)));
  kernel.packet[2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(T1), _mm512_castps_pd(T3)));
  kernel.packet[3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(T1), _mm512_castps_pd(T3)));
  kernel.packet[4] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(T4), _mm512_castps_pd(T6)));
  kernel.packet[5] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(T4), _mm512_castps_pd(T6)));
  kernel.packet[6] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(T5), _mm512_castps_pd(T7)));
  kernel.packet[7] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(T5), _mm512_castps_pd(T7)));

  T0 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(kernel.packet[4]), 0x4E));
  T0 = _mm512_mask_blend_ps(0xF0F0, kernel.packet[0], T0);
  T4 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(kernel.packet[0]), 0x4E));
  T4 = _mm512_mask_blend_ps(0xF0F0, T4, kernel.packet[4]);
  T1 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(kernel.packet[5]), 0x4E));
  T1 = _mm512_mask_blend_ps(0xF0F0, kernel.packet[1], T1);
  T5 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(kernel.packet[1]), 0x4E));
  T5 = _mm512_mask_blend_ps(0xF0F0, T5, kernel.packet[5]);
  T2 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(kernel.packet[6]), 0x4E));
  T2 = _mm512_mask_blend_ps(0xF0F0, kernel.packet[2], T2);
  T6 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(kernel.packet[2]), 0x4E));
  T6 = _mm512_mask_blend_ps(0xF0F0, T6, kernel.packet[6]);
  T3 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(kernel.packet[7]), 0x4E));
  T3 = _mm512_mask_blend_ps(0xF0F0, kernel.packet[3], T3);
  T7 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(kernel.packet[3]), 0x4E));
  T7 = _mm512_mask_blend_ps(0xF0F0, T7, kernel.packet[7]);

  kernel.packet[0] = T0;
  kernel.packet[1] = T1;
  kernel.packet[2] = T2;
  kernel.packet[3] = T3;
  kernel.packet[4] = T4;
  kernel.packet[5] = T5;
  kernel.packet[6] = T6;
  kernel.packet[7] = T7;
}

template <>
EIGEN_ALWAYS_INLINE void trans8x8blocks(PacketBlock<Packet8d, 8> &kernel) {
  ptranspose(kernel);
}

/***
 * Unrolls for tranposed C stores
 */
template <typename Scalar>
class trans {
 public:
  using vec = typename std::conditional<std::is_same<Scalar, float>::value, vecFullFloat, vecFullDouble>::type;
  using vecHalf = typename std::conditional<std::is_same<Scalar, float>::value, vecHalfFloat, vecFullDouble>::type;
  static constexpr int64_t PacketSize = packet_traits<Scalar>::size;

  /***********************************
   * Auxillary Functions for:
   *  - storeC
   ***********************************
   */

  /**
   * aux_storeC
   *
   * 1-D unroll
   *      for(startN = 0; startN < endN; startN++)
   *
   * (endN <= PacketSize) is required to handle the fp32 case, see comments in transStoreC
   *
   **/
  template <int64_t endN, int64_t counter, int64_t unrollN, int64_t packetIndexOffset, bool remM>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0 && endN <= PacketSize)> aux_storeC(
      Scalar *C_arr, int64_t LDC, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t remM_ = 0) {
    constexpr int64_t counterReverse = endN - counter;
    constexpr int64_t startN = counterReverse;

    EIGEN_IF_CONSTEXPR(startN < EIGEN_AVX_MAX_NUM_ROW) {
      EIGEN_IF_CONSTEXPR(remM) {
        pstoreu<Scalar>(
            C_arr + LDC * startN,
            padd(ploadu<vecHalf>((const Scalar *)C_arr + LDC * startN, remMask<EIGEN_AVX_MAX_NUM_ROW>(remM_)),
                 preinterpret<vecHalf>(zmm.packet[packetIndexOffset + (unrollN / PacketSize) * startN]),
                 remMask<EIGEN_AVX_MAX_NUM_ROW>(remM_)),
            remMask<EIGEN_AVX_MAX_NUM_ROW>(remM_));
      }
      else {
        pstoreu<Scalar>(C_arr + LDC * startN,
                        padd(ploadu<vecHalf>((const Scalar *)C_arr + LDC * startN),
                             preinterpret<vecHalf>(zmm.packet[packetIndexOffset + (unrollN / PacketSize) * startN])));
      }
    }
    else {  // This block is only needed for fp32 case
      // Reinterpret as __m512 for _mm512_shuffle_f32x4
      vecFullFloat zmm2vecFullFloat = preinterpret<vecFullFloat>(
          zmm.packet[packetIndexOffset + (unrollN / PacketSize) * (startN - EIGEN_AVX_MAX_NUM_ROW)]);
      // Swap lower and upper half of avx register.
      zmm.packet[packetIndexOffset + (unrollN / PacketSize) * (startN - EIGEN_AVX_MAX_NUM_ROW)] =
          preinterpret<vec>(_mm512_shuffle_f32x4(zmm2vecFullFloat, zmm2vecFullFloat, 0b01001110));

      EIGEN_IF_CONSTEXPR(remM) {
        pstoreu<Scalar>(
            C_arr + LDC * startN,
            padd(ploadu<vecHalf>((const Scalar *)C_arr + LDC * startN, remMask<EIGEN_AVX_MAX_NUM_ROW>(remM_)),
                 preinterpret<vecHalf>(
                     zmm.packet[packetIndexOffset + (unrollN / PacketSize) * (startN - EIGEN_AVX_MAX_NUM_ROW)])),
            remMask<EIGEN_AVX_MAX_NUM_ROW>(remM_));
      }
      else {
        pstoreu<Scalar>(
            C_arr + LDC * startN,
            padd(ploadu<vecHalf>((const Scalar *)C_arr + LDC * startN),
                 preinterpret<vecHalf>(
                     zmm.packet[packetIndexOffset + (unrollN / PacketSize) * (startN - EIGEN_AVX_MAX_NUM_ROW)])));
      }
    }
    aux_storeC<endN, counter - 1, unrollN, packetIndexOffset, remM>(C_arr, LDC, zmm, remM_);
  }

  template <int64_t endN, int64_t counter, int64_t unrollN, int64_t packetIndexOffset, bool remM>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<!(counter > 0 && endN <= PacketSize)> aux_storeC(
      Scalar *C_arr, int64_t LDC, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t remM_ = 0) {
    EIGEN_UNUSED_VARIABLE(C_arr);
    EIGEN_UNUSED_VARIABLE(LDC);
    EIGEN_UNUSED_VARIABLE(zmm);
    EIGEN_UNUSED_VARIABLE(remM_);
  }

  template <int64_t endN, int64_t unrollN, int64_t packetIndexOffset, bool remM>
  static EIGEN_ALWAYS_INLINE void storeC(Scalar *C_arr, int64_t LDC,
                                         PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm,
                                         int64_t remM_ = 0) {
    aux_storeC<endN, endN, unrollN, packetIndexOffset, remM>(C_arr, LDC, zmm, remM_);
  }

  /**
   * Transposes LxunrollN row major block of matrices stored EIGEN_AVX_MAX_NUM_ACC zmm registers to
   * "unrollN"xL ymm registers to be stored col-major into C.
   *
   *  For 8x48, the 8x48 block (row-major) is stored in zmm as follows:
   *
   *  row0: zmm0 zmm1 zmm2
   *  row1: zmm3 zmm4 zmm5
   *    .
   *    .
   *  row7: zmm21 zmm22 zmm23
   *
   *  For 8x32, the 8x32 block (row-major) is stored in zmm as follows:
   *
   *  row0: zmm0 zmm1
   *  row1: zmm2 zmm3
   *    .
   *    .
   *  row7: zmm14 zmm15
   *
   *
   * In general we will have {1,2,3} groups of avx registers each of size
   * EIGEN_AVX_MAX_NUM_ROW. packetIndexOffset is used to select which "block" of
   * avx registers are being transposed.
   */
  template <int64_t unrollN, int64_t packetIndexOffset>
  static EIGEN_ALWAYS_INLINE void transpose(PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm) {
    // Note: this assumes EIGEN_AVX_MAX_NUM_ROW = 8. Unrolls should be adjusted
    // accordingly if EIGEN_AVX_MAX_NUM_ROW is smaller.
    constexpr int64_t zmmStride = unrollN / PacketSize;
    PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> r;
    r.packet[0] = zmm.packet[packetIndexOffset + zmmStride * 0];
    r.packet[1] = zmm.packet[packetIndexOffset + zmmStride * 1];
    r.packet[2] = zmm.packet[packetIndexOffset + zmmStride * 2];
    r.packet[3] = zmm.packet[packetIndexOffset + zmmStride * 3];
    r.packet[4] = zmm.packet[packetIndexOffset + zmmStride * 4];
    r.packet[5] = zmm.packet[packetIndexOffset + zmmStride * 5];
    r.packet[6] = zmm.packet[packetIndexOffset + zmmStride * 6];
    r.packet[7] = zmm.packet[packetIndexOffset + zmmStride * 7];
    trans8x8blocks(r);
    zmm.packet[packetIndexOffset + zmmStride * 0] = r.packet[0];
    zmm.packet[packetIndexOffset + zmmStride * 1] = r.packet[1];
    zmm.packet[packetIndexOffset + zmmStride * 2] = r.packet[2];
    zmm.packet[packetIndexOffset + zmmStride * 3] = r.packet[3];
    zmm.packet[packetIndexOffset + zmmStride * 4] = r.packet[4];
    zmm.packet[packetIndexOffset + zmmStride * 5] = r.packet[5];
    zmm.packet[packetIndexOffset + zmmStride * 6] = r.packet[6];
    zmm.packet[packetIndexOffset + zmmStride * 7] = r.packet[7];
  }
};

/**
 * Unrolls for copyBToRowMajor
 *
 * Idea:
 *  1) Load a block of right-hand sides to registers (using loadB).
 *  2) Convert the block from column-major to row-major (transposeLxL)
 *  3) Store the blocks from register either to a temp array (toTemp == true), or back to B (toTemp == false).
 *
 *  We use at most EIGEN_AVX_MAX_NUM_ACC avx registers to store the blocks of B. The remaining registers are
 *  used as temps for transposing.
 *
 *  Blocks will be of size Lx{U1,U2,U3}. packetIndexOffset is used to index between these subblocks
 *  For fp32, PacketSize = 2*EIGEN_AVX_MAX_NUM_ROW, so we reinterpret packets as packets half the size (zmm -> ymm).
 */
template <typename Scalar>
class transB {
 public:
  using vec = typename std::conditional<std::is_same<Scalar, float>::value, vecFullFloat, vecFullDouble>::type;
  using vecHalf = typename std::conditional<std::is_same<Scalar, float>::value, vecHalfFloat, vecFullDouble>::type;
  static constexpr int64_t PacketSize = packet_traits<Scalar>::size;

  /***********************************
   * Auxillary Functions for:
   *  - loadB
   *  - storeB
   *  - loadBBlock
   *  - storeBBlock
   ***********************************
   */

  /**
   * aux_loadB
   *
   * 1-D unroll
   *      for(startN = 0; startN < endN; startN++)
   **/
  template <int64_t endN, int64_t counter, int64_t packetIndexOffset, bool remM, int64_t remN_>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_loadB(
      Scalar *B_arr, int64_t LDB, PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm,
      int64_t remM_ = 0) {
    constexpr int64_t counterReverse = endN - counter;
    constexpr int64_t startN = counterReverse;

    EIGEN_IF_CONSTEXPR(remM) {
      ymm.packet[packetIndexOffset + startN] =
          ploadu<vecHalf>((const Scalar *)&B_arr[startN * LDB], remMask<EIGEN_AVX_MAX_NUM_ROW>(remM_));
    }
    else {
      EIGEN_IF_CONSTEXPR(remN_ == 0) {
        ymm.packet[packetIndexOffset + startN] = ploadu<vecHalf>((const Scalar *)&B_arr[startN * LDB]);
      }
      else ymm.packet[packetIndexOffset + startN] =
          ploadu<vecHalf>((const Scalar *)&B_arr[startN * LDB], remMask<EIGEN_AVX_MAX_NUM_ROW>(remN_));
    }

    aux_loadB<endN, counter - 1, packetIndexOffset, remM, remN_>(B_arr, LDB, ymm, remM_);
  }

  template <int64_t endN, int64_t counter, int64_t packetIndexOffset, bool remM, int64_t remN_>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_loadB(
      Scalar *B_arr, int64_t LDB, PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm,
      int64_t remM_ = 0) {
    EIGEN_UNUSED_VARIABLE(B_arr);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(ymm);
    EIGEN_UNUSED_VARIABLE(remM_);
  }

  /**
   * aux_storeB
   *
   * 1-D unroll
   *      for(startN = 0; startN < endN; startN++)
   **/
  template <int64_t endN, int64_t counter, int64_t packetIndexOffset, bool remK, bool remM>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_storeB(
      Scalar *B_arr, int64_t LDB, PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm, int64_t rem_ = 0) {
    constexpr int64_t counterReverse = endN - counter;
    constexpr int64_t startN = counterReverse;

    EIGEN_IF_CONSTEXPR(remK || remM) {
      pstoreu<Scalar>(&B_arr[startN * LDB], ymm.packet[packetIndexOffset + startN],
                      remMask<EIGEN_AVX_MAX_NUM_ROW>(rem_));
    }
    else {
      pstoreu<Scalar>(&B_arr[startN * LDB], ymm.packet[packetIndexOffset + startN]);
    }

    aux_storeB<endN, counter - 1, packetIndexOffset, remK, remM>(B_arr, LDB, ymm, rem_);
  }

  template <int64_t endN, int64_t counter, int64_t packetIndexOffset, bool remK, bool remM>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_storeB(
      Scalar *B_arr, int64_t LDB, PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(B_arr);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(ymm);
    EIGEN_UNUSED_VARIABLE(rem_);
  }

  /**
   * aux_loadBBlock
   *
   * 1-D unroll
   *      for(startN = 0; startN < endN; startN += EIGEN_AVX_MAX_NUM_ROW)
   **/
  template <int64_t endN, int64_t counter, bool toTemp, bool remM, int64_t remN_>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_loadBBlock(
      Scalar *B_arr, int64_t LDB, Scalar *B_temp, int64_t LDB_,
      PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm, int64_t remM_ = 0) {
    constexpr int64_t counterReverse = endN - counter;
    constexpr int64_t startN = counterReverse;
    transB::template loadB<EIGEN_AVX_MAX_NUM_ROW, startN, false, (toTemp ? 0 : remN_)>(&B_temp[startN], LDB_, ymm);
    aux_loadBBlock<endN, counter - EIGEN_AVX_MAX_NUM_ROW, toTemp, remM, remN_>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
  }

  template <int64_t endN, int64_t counter, bool toTemp, bool remM, int64_t remN_>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_loadBBlock(
      Scalar *B_arr, int64_t LDB, Scalar *B_temp, int64_t LDB_,
      PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm, int64_t remM_ = 0) {
    EIGEN_UNUSED_VARIABLE(B_arr);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(B_temp);
    EIGEN_UNUSED_VARIABLE(LDB_);
    EIGEN_UNUSED_VARIABLE(ymm);
    EIGEN_UNUSED_VARIABLE(remM_);
  }

  /**
   * aux_storeBBlock
   *
   * 1-D unroll
   *      for(startN = 0; startN < endN; startN += EIGEN_AVX_MAX_NUM_ROW)
   **/
  template <int64_t endN, int64_t counter, bool toTemp, bool remM, int64_t remK_>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_storeBBlock(
      Scalar *B_arr, int64_t LDB, Scalar *B_temp, int64_t LDB_,
      PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm, int64_t remM_ = 0) {
    constexpr int64_t counterReverse = endN - counter;
    constexpr int64_t startN = counterReverse;

    EIGEN_IF_CONSTEXPR(toTemp) {
      transB::template storeB<EIGEN_AVX_MAX_NUM_ROW, startN, remK_ != 0, false>(&B_temp[startN], LDB_, ymm, remK_);
    }
    else {
      transB::template storeB<std::min(EIGEN_AVX_MAX_NUM_ROW, endN), startN, false, remM>(&B_arr[0 + startN * LDB], LDB,
                                                                                          ymm, remM_);
    }
    aux_storeBBlock<endN, counter - EIGEN_AVX_MAX_NUM_ROW, toTemp, remM, remK_>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
  }

  template <int64_t endN, int64_t counter, bool toTemp, bool remM, int64_t remK_>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_storeBBlock(
      Scalar *B_arr, int64_t LDB, Scalar *B_temp, int64_t LDB_,
      PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm, int64_t remM_ = 0) {
    EIGEN_UNUSED_VARIABLE(B_arr);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(B_temp);
    EIGEN_UNUSED_VARIABLE(LDB_);
    EIGEN_UNUSED_VARIABLE(ymm);
    EIGEN_UNUSED_VARIABLE(remM_);
  }

  /********************************************************
   * Wrappers for aux_XXXX to hide counter parameter
   ********************************************************/

  template <int64_t endN, int64_t packetIndexOffset, bool remM, int64_t remN_>
  static EIGEN_ALWAYS_INLINE void loadB(Scalar *B_arr, int64_t LDB,
                                        PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm,
                                        int64_t remM_ = 0) {
    aux_loadB<endN, endN, packetIndexOffset, remM, remN_>(B_arr, LDB, ymm, remM_);
  }

  template <int64_t endN, int64_t packetIndexOffset, bool remK, bool remM>
  static EIGEN_ALWAYS_INLINE void storeB(Scalar *B_arr, int64_t LDB,
                                         PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm,
                                         int64_t rem_ = 0) {
    aux_storeB<endN, endN, packetIndexOffset, remK, remM>(B_arr, LDB, ymm, rem_);
  }

  template <int64_t unrollN, bool toTemp, bool remM, int64_t remN_ = 0>
  static EIGEN_ALWAYS_INLINE void loadBBlock(Scalar *B_arr, int64_t LDB, Scalar *B_temp, int64_t LDB_,
                                             PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm,
                                             int64_t remM_ = 0) {
    EIGEN_IF_CONSTEXPR(toTemp) { transB::template loadB<unrollN, 0, remM, 0>(&B_arr[0], LDB, ymm, remM_); }
    else {
      aux_loadBBlock<unrollN, unrollN, toTemp, remM, remN_>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
    }
  }

  template <int64_t unrollN, bool toTemp, bool remM, int64_t remK_>
  static EIGEN_ALWAYS_INLINE void storeBBlock(Scalar *B_arr, int64_t LDB, Scalar *B_temp, int64_t LDB_,
                                              PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm,
                                              int64_t remM_ = 0) {
    aux_storeBBlock<unrollN, unrollN, toTemp, remM, remK_>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
  }

  template <int64_t packetIndexOffset>
  static EIGEN_ALWAYS_INLINE void transposeLxL(PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm) {
    // Note: this assumes EIGEN_AVX_MAX_NUM_ROW = 8. Unrolls should be adjusted
    // accordingly if EIGEN_AVX_MAX_NUM_ROW is smaller.
    PacketBlock<vecHalf, EIGEN_AVX_MAX_NUM_ROW> r;
    r.packet[0] = ymm.packet[packetIndexOffset + 0];
    r.packet[1] = ymm.packet[packetIndexOffset + 1];
    r.packet[2] = ymm.packet[packetIndexOffset + 2];
    r.packet[3] = ymm.packet[packetIndexOffset + 3];
    r.packet[4] = ymm.packet[packetIndexOffset + 4];
    r.packet[5] = ymm.packet[packetIndexOffset + 5];
    r.packet[6] = ymm.packet[packetIndexOffset + 6];
    r.packet[7] = ymm.packet[packetIndexOffset + 7];
    ptranspose(r);
    ymm.packet[packetIndexOffset + 0] = r.packet[0];
    ymm.packet[packetIndexOffset + 1] = r.packet[1];
    ymm.packet[packetIndexOffset + 2] = r.packet[2];
    ymm.packet[packetIndexOffset + 3] = r.packet[3];
    ymm.packet[packetIndexOffset + 4] = r.packet[4];
    ymm.packet[packetIndexOffset + 5] = r.packet[5];
    ymm.packet[packetIndexOffset + 6] = r.packet[6];
    ymm.packet[packetIndexOffset + 7] = r.packet[7];
  }

  template <int64_t unrollN, bool toTemp, bool remM>
  static EIGEN_ALWAYS_INLINE void transB_kernel(Scalar *B_arr, int64_t LDB, Scalar *B_temp, int64_t LDB_,
                                                PacketBlock<vecHalf, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &ymm,
                                                int64_t remM_ = 0) {
    constexpr int64_t U3 = PacketSize * 3;
    constexpr int64_t U2 = PacketSize * 2;
    constexpr int64_t U1 = PacketSize * 1;
    /**
     *  Unrolls needed for each case:
     *   - AVX512 fp32 48 32 16 8 4 2 1
     *   - AVX512 fp64 24 16 8  4 2 1
     *
     *  For fp32 L and U1 are 1:2 so for U3/U2 cases the loads/stores need to be split up.
     */
    EIGEN_IF_CONSTEXPR(unrollN == U3) {
      // load LxU3 B col major, transpose LxU3 row major
      constexpr int64_t maxUBlock = std::min(3 * EIGEN_AVX_MAX_NUM_ROW, U3);
      transB::template loadBBlock<maxUBlock, toTemp, remM>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
      transB::template transposeLxL<0 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
      transB::template transposeLxL<1 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
      transB::template transposeLxL<2 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
      transB::template storeBBlock<maxUBlock, toTemp, remM, 0>(B_arr, LDB, B_temp, LDB_, ymm, remM_);

      EIGEN_IF_CONSTEXPR(maxUBlock < U3) {
        transB::template loadBBlock<maxUBlock, toTemp, remM>(&B_arr[maxUBlock * LDB], LDB, &B_temp[maxUBlock], LDB_,
                                                             ymm, remM_);
        transB::template transposeLxL<0 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
        transB::template transposeLxL<1 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
        transB::template transposeLxL<2 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
        transB::template storeBBlock<maxUBlock, toTemp, remM, 0>(&B_arr[maxUBlock * LDB], LDB, &B_temp[maxUBlock], LDB_,
                                                                 ymm, remM_);
      }
    }
    else EIGEN_IF_CONSTEXPR(unrollN == U2) {
      // load LxU2 B col major, transpose LxU2 row major
      constexpr int64_t maxUBlock = std::min(3 * EIGEN_AVX_MAX_NUM_ROW, U2);
      transB::template loadBBlock<maxUBlock, toTemp, remM>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
      transB::template transposeLxL<0 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
      transB::template transposeLxL<1 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
      EIGEN_IF_CONSTEXPR(maxUBlock < U2) transB::template transposeLxL<2 * EIGEN_AVX_MAX_NUM_ROW>(ymm);
      transB::template storeBBlock<maxUBlock, toTemp, remM, 0>(B_arr, LDB, B_temp, LDB_, ymm, remM_);

      EIGEN_IF_CONSTEXPR(maxUBlock < U2) {
        transB::template loadBBlock<EIGEN_AVX_MAX_NUM_ROW, toTemp, remM>(&B_arr[maxUBlock * LDB], LDB,
                                                                         &B_temp[maxUBlock], LDB_, ymm, remM_);
        transB::template transposeLxL<0>(ymm);
        transB::template storeBBlock<EIGEN_AVX_MAX_NUM_ROW, toTemp, remM, 0>(&B_arr[maxUBlock * LDB], LDB,
                                                                             &B_temp[maxUBlock], LDB_, ymm, remM_);
      }
    }
    else EIGEN_IF_CONSTEXPR(unrollN == U1) {
      // load LxU1 B col major, transpose LxU1 row major
      transB::template loadBBlock<U1, toTemp, remM>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
      transB::template transposeLxL<0>(ymm);
      EIGEN_IF_CONSTEXPR(EIGEN_AVX_MAX_NUM_ROW < U1) { transB::template transposeLxL<1 * EIGEN_AVX_MAX_NUM_ROW>(ymm); }
      transB::template storeBBlock<U1, toTemp, remM, 0>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
    }
    else EIGEN_IF_CONSTEXPR(unrollN == 8 && U1 > 8) {
      // load Lx4 B col major, transpose Lx4 row major
      transB::template loadBBlock<8, toTemp, remM>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
      transB::template transposeLxL<0>(ymm);
      transB::template storeBBlock<8, toTemp, remM, 8>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
    }
    else EIGEN_IF_CONSTEXPR(unrollN == 4 && U1 > 4) {
      // load Lx4 B col major, transpose Lx4 row major
      transB::template loadBBlock<4, toTemp, remM>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
      transB::template transposeLxL<0>(ymm);
      transB::template storeBBlock<4, toTemp, remM, 4>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
    }
    else EIGEN_IF_CONSTEXPR(unrollN == 2) {
      // load Lx2 B col major, transpose Lx2 row major
      transB::template loadBBlock<2, toTemp, remM, 2>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
      transB::template transposeLxL<0>(ymm);
      transB::template storeBBlock<2, toTemp, remM, 2>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
    }
    else EIGEN_IF_CONSTEXPR(unrollN == 1) {
      // load Lx1 B col major, transpose Lx1 row major
      transB::template loadBBlock<1, toTemp, remM, 1>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
      transB::template transposeLxL<0>(ymm);
      transB::template storeBBlock<1, toTemp, remM, 1>(B_arr, LDB, B_temp, LDB_, ymm, remM_);
    }
  }
};

/**
 * Unrolls for triSolveKernel
 *
 * Idea:
 *  1) Load a block of right-hand sides to registers in RHSInPacket (using loadRHS).
 *  2) Do triangular solve with RHSInPacket and a small block of A (triangular matrix)
 *     stored in AInPacket (using triSolveMicroKernel).
 *  3) Store final results (in avx registers) back into memory (using storeRHS).
 *
 *  RHSInPacket uses at most EIGEN_AVX_MAX_NUM_ACC avx registers and AInPacket uses at most
 *  EIGEN_AVX_MAX_NUM_ROW registers.
 */
template <typename Scalar>
class trsm {
 public:
  using vec = typename std::conditional<std::is_same<Scalar, float>::value, vecFullFloat, vecFullDouble>::type;
  static constexpr int64_t PacketSize = packet_traits<Scalar>::size;

  /***********************************
   * Auxillary Functions for:
   *  - loadRHS
   *  - storeRHS
   *  - divRHSByDiag
   *  - updateRHS
   *  - triSolveMicroKernel
   ************************************/
  /**
   * aux_loadRHS
   *
   * 2-D unroll
   *      for(startM = 0; startM < endM; startM++)
   *        for(startK = 0; startK < endK; startK++)
   **/
  template <bool isFWDSolve, int64_t endM, int64_t endK, int64_t counter, bool krem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_loadRHS(
      Scalar *B_arr, int64_t LDB, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket, int64_t rem = 0) {
    constexpr int64_t counterReverse = endM * endK - counter;
    constexpr int64_t startM = counterReverse / (endK);
    constexpr int64_t startK = counterReverse % endK;

    constexpr int64_t packetIndex = startM * endK + startK;
    constexpr int64_t startM_ = isFWDSolve ? startM : -startM;
    const int64_t rhsIndex = (startK * PacketSize) + startM_ * LDB;
    EIGEN_IF_CONSTEXPR(krem) {
      RHSInPacket.packet[packetIndex] = ploadu<vec>(&B_arr[rhsIndex], remMask<PacketSize>(rem));
    }
    else {
      RHSInPacket.packet[packetIndex] = ploadu<vec>(&B_arr[rhsIndex]);
    }
    aux_loadRHS<isFWDSolve, endM, endK, counter - 1, krem>(B_arr, LDB, RHSInPacket, rem);
  }

  template <bool isFWDSolve, int64_t endM, int64_t endK, int64_t counter, bool krem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_loadRHS(
      Scalar *B_arr, int64_t LDB, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket, int64_t rem = 0) {
    EIGEN_UNUSED_VARIABLE(B_arr);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(RHSInPacket);
    EIGEN_UNUSED_VARIABLE(rem);
  }

  /**
   * aux_storeRHS
   *
   * 2-D unroll
   *      for(startM = 0; startM < endM; startM++)
   *        for(startK = 0; startK < endK; startK++)
   **/
  template <bool isFWDSolve, int64_t endM, int64_t endK, int64_t counter, bool krem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_storeRHS(
      Scalar *B_arr, int64_t LDB, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket, int64_t rem = 0) {
    constexpr int64_t counterReverse = endM * endK - counter;
    constexpr int64_t startM = counterReverse / (endK);
    constexpr int64_t startK = counterReverse % endK;

    constexpr int64_t packetIndex = startM * endK + startK;
    constexpr int64_t startM_ = isFWDSolve ? startM : -startM;
    const int64_t rhsIndex = (startK * PacketSize) + startM_ * LDB;
    EIGEN_IF_CONSTEXPR(krem) {
      pstoreu<Scalar>(&B_arr[rhsIndex], RHSInPacket.packet[packetIndex], remMask<PacketSize>(rem));
    }
    else {
      pstoreu<Scalar>(&B_arr[rhsIndex], RHSInPacket.packet[packetIndex]);
    }
    aux_storeRHS<isFWDSolve, endM, endK, counter - 1, krem>(B_arr, LDB, RHSInPacket, rem);
  }

  template <bool isFWDSolve, int64_t endM, int64_t endK, int64_t counter, bool krem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_storeRHS(
      Scalar *B_arr, int64_t LDB, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket, int64_t rem = 0) {
    EIGEN_UNUSED_VARIABLE(B_arr);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(RHSInPacket);
    EIGEN_UNUSED_VARIABLE(rem);
  }

  /**
   * aux_divRHSByDiag
   *
   * currM may be -1, (currM >=0) in enable_if checks for this
   *
   * 1-D unroll
   *      for(startK = 0; startK < endK; startK++)
   **/
  template <int64_t currM, int64_t endK, int64_t counter>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0 && currM >= 0)> aux_divRHSByDiag(
      PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    constexpr int64_t counterReverse = endK - counter;
    constexpr int64_t startK = counterReverse;

    constexpr int64_t packetIndex = currM * endK + startK;
    RHSInPacket.packet[packetIndex] = pmul(AInPacket.packet[currM], RHSInPacket.packet[packetIndex]);
    aux_divRHSByDiag<currM, endK, counter - 1>(RHSInPacket, AInPacket);
  }

  template <int64_t currM, int64_t endK, int64_t counter>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<!(counter > 0 && currM >= 0)> aux_divRHSByDiag(
      PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    EIGEN_UNUSED_VARIABLE(RHSInPacket);
    EIGEN_UNUSED_VARIABLE(AInPacket);
  }

  /**
   * aux_updateRHS
   *
   * 2-D unroll
   *      for(startM = initM; startM < endM; startM++)
   *        for(startK = 0; startK < endK; startK++)
   **/
  template <bool isARowMajor, bool isFWDSolve, bool isUnitDiag, int64_t initM, int64_t endM, int64_t endK,
            int64_t counter, int64_t currentM>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_updateRHS(
      Scalar *A_arr, int64_t LDA, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket,
      PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    constexpr int64_t counterReverse = (endM - initM) * endK - counter;
    constexpr int64_t startM = initM + counterReverse / (endK);
    constexpr int64_t startK = counterReverse % endK;

    // For each row of A, first update all corresponding RHS
    constexpr int64_t packetIndex = startM * endK + startK;
    EIGEN_IF_CONSTEXPR(currentM > 0) {
      RHSInPacket.packet[packetIndex] =
          pnmadd(AInPacket.packet[startM], RHSInPacket.packet[(currentM - 1) * endK + startK],
                 RHSInPacket.packet[packetIndex]);
    }

    EIGEN_IF_CONSTEXPR(startK == endK - 1) {
      // Once all RHS for previous row of A is updated, we broadcast the next element in the column A_{i, currentM}.
      EIGEN_IF_CONSTEXPR(startM == currentM && !isUnitDiag) {
        // If diagonal is not unit, we broadcast reciprocals of diagonals AinPacket.packet[currentM].
        // This will be used in divRHSByDiag
        EIGEN_IF_CONSTEXPR(isFWDSolve)
        AInPacket.packet[currentM] = pset1<vec>(Scalar(1) / A_arr[idA<isARowMajor>(currentM, currentM, LDA)]);
        else AInPacket.packet[currentM] = pset1<vec>(Scalar(1) / A_arr[idA<isARowMajor>(-currentM, -currentM, LDA)]);
      }
      else {
        // Broadcast next off diagonal element of A
        EIGEN_IF_CONSTEXPR(isFWDSolve)
        AInPacket.packet[startM] = pset1<vec>(A_arr[idA<isARowMajor>(startM, currentM, LDA)]);
        else AInPacket.packet[startM] = pset1<vec>(A_arr[idA<isARowMajor>(-startM, -currentM, LDA)]);
      }
    }

    aux_updateRHS<isARowMajor, isFWDSolve, isUnitDiag, initM, endM, endK, counter - 1, currentM>(
        A_arr, LDA, RHSInPacket, AInPacket);
  }

  template <bool isARowMajor, bool isFWDSolve, bool isUnitDiag, int64_t initM, int64_t endM, int64_t endK,
            int64_t counter, int64_t currentM>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_updateRHS(
      Scalar *A_arr, int64_t LDA, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket,
      PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    EIGEN_UNUSED_VARIABLE(A_arr);
    EIGEN_UNUSED_VARIABLE(LDA);
    EIGEN_UNUSED_VARIABLE(RHSInPacket);
    EIGEN_UNUSED_VARIABLE(AInPacket);
  }

  /**
   * aux_triSolverMicroKernel
   *
   * 1-D unroll
   *      for(startM = 0; startM < endM; startM++)
   **/
  template <bool isARowMajor, bool isFWDSolve, bool isUnitDiag, int64_t endM, int64_t counter, int64_t numK>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_triSolveMicroKernel(
      Scalar *A_arr, int64_t LDA, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket,
      PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    constexpr int64_t counterReverse = endM - counter;
    constexpr int64_t startM = counterReverse;

    constexpr int64_t currentM = startM;
    // Divides the right-hand side in row startM, by digonal value of A
    // broadcasted to AInPacket.packet[startM-1] in the previous iteration.
    //
    // Without "if constexpr" the compiler instantiates the case <-1, numK>
    // this is handled with enable_if to prevent out-of-bound warnings
    // from the compiler
    EIGEN_IF_CONSTEXPR(!isUnitDiag && startM > 0)
    trsm::template divRHSByDiag<startM - 1, numK>(RHSInPacket, AInPacket);

    // After division, the rhs corresponding to subsequent rows of A can be partially updated
    // We also broadcast the reciprocal of the next diagonal to AInPacket.packet[currentM] (if needed)
    // to be used in the next iteration.
    trsm::template updateRHS<isARowMajor, isFWDSolve, isUnitDiag, startM, endM, numK, currentM>(A_arr, LDA, RHSInPacket,
                                                                                                AInPacket);

    // Handle division for the RHS corresponding to the final row of A.
    EIGEN_IF_CONSTEXPR(!isUnitDiag && startM == endM - 1)
    trsm::template divRHSByDiag<startM, numK>(RHSInPacket, AInPacket);

    aux_triSolveMicroKernel<isARowMajor, isFWDSolve, isUnitDiag, endM, counter - 1, numK>(A_arr, LDA, RHSInPacket,
                                                                                          AInPacket);
  }

  template <bool isARowMajor, bool isFWDSolve, bool isUnitDiag, int64_t endM, int64_t counter, int64_t numK>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_triSolveMicroKernel(
      Scalar *A_arr, int64_t LDA, PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket,
      PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    EIGEN_UNUSED_VARIABLE(A_arr);
    EIGEN_UNUSED_VARIABLE(LDA);
    EIGEN_UNUSED_VARIABLE(RHSInPacket);
    EIGEN_UNUSED_VARIABLE(AInPacket);
  }

  /********************************************************
   * Wrappers for aux_XXXX to hide counter parameter
   ********************************************************/

  /**
   * Load endMxendK block of B to RHSInPacket
   * Masked loads are used for cases where endK is not a multiple of PacketSize
   */
  template <bool isFWDSolve, int64_t endM, int64_t endK, bool krem = false>
  static EIGEN_ALWAYS_INLINE void loadRHS(Scalar *B_arr, int64_t LDB,
                                          PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket, int64_t rem = 0) {
    aux_loadRHS<isFWDSolve, endM, endK, endM * endK, krem>(B_arr, LDB, RHSInPacket, rem);
  }

  /**
   * Load endMxendK block of B to RHSInPacket
   * Masked loads are used for cases where endK is not a multiple of PacketSize
   */
  template <bool isFWDSolve, int64_t endM, int64_t endK, bool krem = false>
  static EIGEN_ALWAYS_INLINE void storeRHS(Scalar *B_arr, int64_t LDB,
                                           PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket, int64_t rem = 0) {
    aux_storeRHS<isFWDSolve, endM, endK, endM * endK, krem>(B_arr, LDB, RHSInPacket, rem);
  }

  /**
   * Only used if Triangular matrix has non-unit diagonal values
   */
  template <int64_t currM, int64_t endK>
  static EIGEN_ALWAYS_INLINE void divRHSByDiag(PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket,
                                               PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    aux_divRHSByDiag<currM, endK, endK>(RHSInPacket, AInPacket);
  }

  /**
   * Update right-hand sides (stored in avx registers)
   * Traversing along the column A_{i,currentM}, where currentM <= i <= endM, and broadcasting each value to AInPacket.
   **/
  template <bool isARowMajor, bool isFWDSolve, bool isUnitDiag, int64_t startM, int64_t endM, int64_t endK,
            int64_t currentM>
  static EIGEN_ALWAYS_INLINE void updateRHS(Scalar *A_arr, int64_t LDA,
                                            PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket,
                                            PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    aux_updateRHS<isARowMajor, isFWDSolve, isUnitDiag, startM, endM, endK, (endM - startM) * endK, currentM>(
        A_arr, LDA, RHSInPacket, AInPacket);
  }

  /**
   * endM: dimension of A. 1 <= endM <= EIGEN_AVX_MAX_NUM_ROW
   * numK: number of avx registers to use for each row of B (ex fp32: 48 rhs => 3 avx reg used). 1 <= endK <= 3.
   * isFWDSolve: true => forward substitution, false => backwards substitution
   * isUnitDiag: true => triangular matrix has unit diagonal.
   */
  template <bool isARowMajor, bool isFWDSolve, bool isUnitDiag, int64_t endM, int64_t numK>
  static EIGEN_ALWAYS_INLINE void triSolveMicroKernel(Scalar *A_arr, int64_t LDA,
                                                      PacketBlock<vec, EIGEN_AVX_MAX_NUM_ACC> &RHSInPacket,
                                                      PacketBlock<vec, EIGEN_AVX_MAX_NUM_ROW> &AInPacket) {
    static_assert(numK >= 1 && numK <= 3, "numK out of range");
    aux_triSolveMicroKernel<isARowMajor, isFWDSolve, isUnitDiag, endM, endM, numK>(A_arr, LDA, RHSInPacket, AInPacket);
  }
};

/**
 * Unrolls for gemm kernel
 *
 * isAdd: true => C += A*B, false => C -= A*B
 */
template <typename Scalar, bool isAdd>
class gemm {
 public:
  using vec = typename std::conditional<std::is_same<Scalar, float>::value, vecFullFloat, vecFullDouble>::type;
  static constexpr int64_t PacketSize = packet_traits<Scalar>::size;

  /***********************************
   * Auxillary Functions for:
   *  - setzero
   *  - updateC
   *  - storeC
   *  - startLoadB
   *  - triSolveMicroKernel
   ************************************/

  /**
   * aux_setzero
   *
   * 2-D unroll
   *      for(startM = 0; startM < endM; startM++)
   *        for(startN = 0; startN < endN; startN++)
   **/
  template <int64_t endM, int64_t endN, int64_t counter>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_setzero(
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm) {
    constexpr int64_t counterReverse = endM * endN - counter;
    constexpr int64_t startM = counterReverse / (endN);
    constexpr int64_t startN = counterReverse % endN;

    zmm.packet[startN * endM + startM] = pzero(zmm.packet[startN * endM + startM]);
    aux_setzero<endM, endN, counter - 1>(zmm);
  }

  template <int64_t endM, int64_t endN, int64_t counter>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_setzero(
      PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm) {
    EIGEN_UNUSED_VARIABLE(zmm);
  }

  /**
   * aux_updateC
   *
   * 2-D unroll
   *      for(startM = 0; startM < endM; startM++)
   *        for(startN = 0; startN < endN; startN++)
   **/
  template <int64_t endM, int64_t endN, int64_t counter, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_updateC(
      Scalar *C_arr, int64_t LDC, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    constexpr int64_t counterReverse = endM * endN - counter;
    constexpr int64_t startM = counterReverse / (endN);
    constexpr int64_t startN = counterReverse % endN;

    EIGEN_IF_CONSTEXPR(rem)
    zmm.packet[startN * endM + startM] =
        padd(ploadu<vec>(&C_arr[(startN)*LDC + startM * PacketSize], remMask<PacketSize>(rem_)),
             zmm.packet[startN * endM + startM], remMask<PacketSize>(rem_));
    else zmm.packet[startN * endM + startM] =
        padd(ploadu<vec>(&C_arr[(startN)*LDC + startM * PacketSize]), zmm.packet[startN * endM + startM]);
    aux_updateC<endM, endN, counter - 1, rem>(C_arr, LDC, zmm, rem_);
  }

  template <int64_t endM, int64_t endN, int64_t counter, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_updateC(
      Scalar *C_arr, int64_t LDC, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(C_arr);
    EIGEN_UNUSED_VARIABLE(LDC);
    EIGEN_UNUSED_VARIABLE(zmm);
    EIGEN_UNUSED_VARIABLE(rem_);
  }

  /**
   * aux_storeC
   *
   * 2-D unroll
   *      for(startM = 0; startM < endM; startM++)
   *        for(startN = 0; startN < endN; startN++)
   **/
  template <int64_t endM, int64_t endN, int64_t counter, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_storeC(
      Scalar *C_arr, int64_t LDC, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    constexpr int64_t counterReverse = endM * endN - counter;
    constexpr int64_t startM = counterReverse / (endN);
    constexpr int64_t startN = counterReverse % endN;

    EIGEN_IF_CONSTEXPR(rem)
    pstoreu<Scalar>(&C_arr[(startN)*LDC + startM * PacketSize], zmm.packet[startN * endM + startM],
                    remMask<PacketSize>(rem_));
    else pstoreu<Scalar>(&C_arr[(startN)*LDC + startM * PacketSize], zmm.packet[startN * endM + startM]);
    aux_storeC<endM, endN, counter - 1, rem>(C_arr, LDC, zmm, rem_);
  }

  template <int64_t endM, int64_t endN, int64_t counter, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_storeC(
      Scalar *C_arr, int64_t LDC, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(C_arr);
    EIGEN_UNUSED_VARIABLE(LDC);
    EIGEN_UNUSED_VARIABLE(zmm);
    EIGEN_UNUSED_VARIABLE(rem_);
  }

  /**
   * aux_startLoadB
   *
   * 1-D unroll
   *      for(startL = 0; startL < endL; startL++)
   **/
  template <int64_t unrollM, int64_t unrollN, int64_t endL, int64_t counter, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_startLoadB(
      Scalar *B_t, int64_t LDB, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    constexpr int64_t counterReverse = endL - counter;
    constexpr int64_t startL = counterReverse;

    EIGEN_IF_CONSTEXPR(rem)
    zmm.packet[unrollM * unrollN + startL] =
        ploadu<vec>(&B_t[(startL / unrollM) * LDB + (startL % unrollM) * PacketSize], remMask<PacketSize>(rem_));
    else zmm.packet[unrollM * unrollN + startL] =
        ploadu<vec>(&B_t[(startL / unrollM) * LDB + (startL % unrollM) * PacketSize]);

    aux_startLoadB<unrollM, unrollN, endL, counter - 1, rem>(B_t, LDB, zmm, rem_);
  }

  template <int64_t unrollM, int64_t unrollN, int64_t endL, int64_t counter, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_startLoadB(
      Scalar *B_t, int64_t LDB, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(B_t);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(zmm);
    EIGEN_UNUSED_VARIABLE(rem_);
  }

  /**
   * aux_startBCastA
   *
   * 1-D unroll
   *      for(startB = 0; startB < endB; startB++)
   **/
  template <bool isARowMajor, int64_t unrollM, int64_t unrollN, int64_t endB, int64_t counter, int64_t numLoad>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_startBCastA(
      Scalar *A_t, int64_t LDA, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm) {
    constexpr int64_t counterReverse = endB - counter;
    constexpr int64_t startB = counterReverse;

    zmm.packet[unrollM * unrollN + numLoad + startB] = pload1<vec>(&A_t[idA<isARowMajor>(startB, 0, LDA)]);

    aux_startBCastA<isARowMajor, unrollM, unrollN, endB, counter - 1, numLoad>(A_t, LDA, zmm);
  }

  template <bool isARowMajor, int64_t unrollM, int64_t unrollN, int64_t endB, int64_t counter, int64_t numLoad>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_startBCastA(
      Scalar *A_t, int64_t LDA, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm) {
    EIGEN_UNUSED_VARIABLE(A_t);
    EIGEN_UNUSED_VARIABLE(LDA);
    EIGEN_UNUSED_VARIABLE(zmm);
  }

  /**
   * aux_loadB
   * currK: current K
   *
   * 1-D unroll
   *      for(startM = 0; startM < endM; startM++)
   **/
  template <int64_t endM, int64_t counter, int64_t unrollN, int64_t currK, int64_t unrollK, int64_t numLoad,
            int64_t numBCast, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_loadB(
      Scalar *B_t, int64_t LDB, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    if ((numLoad / endM + currK < unrollK)) {
      constexpr int64_t counterReverse = endM - counter;
      constexpr int64_t startM = counterReverse;

      EIGEN_IF_CONSTEXPR(rem) {
        zmm.packet[endM * unrollN + (startM + currK * endM) % numLoad] =
            ploadu<vec>(&B_t[(numLoad / endM + currK) * LDB + startM * PacketSize], remMask<PacketSize>(rem_));
      }
      else {
        zmm.packet[endM * unrollN + (startM + currK * endM) % numLoad] =
            ploadu<vec>(&B_t[(numLoad / endM + currK) * LDB + startM * PacketSize]);
      }

      aux_loadB<endM, counter - 1, unrollN, currK, unrollK, numLoad, numBCast, rem>(B_t, LDB, zmm, rem_);
    }
  }

  template <int64_t endM, int64_t counter, int64_t unrollN, int64_t currK, int64_t unrollK, int64_t numLoad,
            int64_t numBCast, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_loadB(
      Scalar *B_t, int64_t LDB, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm, int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(B_t);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(zmm);
    EIGEN_UNUSED_VARIABLE(rem_);
  }

  /**
   * aux_microKernel
   *
   * 3-D unroll
   *      for(startM = 0; startM < endM; startM++)
   *        for(startN = 0; startN < endN; startN++)
   *          for(startK = 0; startK < endK; startK++)
   **/
  template <bool isARowMajor, int64_t endM, int64_t endN, int64_t endK, int64_t counter, int64_t numLoad,
            int64_t numBCast, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter > 0)> aux_microKernel(
      Scalar *B_t, Scalar *A_t, int64_t LDB, int64_t LDA, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm,
      int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    constexpr int64_t counterReverse = endM * endN * endK - counter;
    constexpr int startK = counterReverse / (endM * endN);
    constexpr int startN = (counterReverse / (endM)) % endN;
    constexpr int startM = counterReverse % endM;

    EIGEN_IF_CONSTEXPR(startK == 0 && startM == 0 && startN == 0) {
      gemm::template startLoadB<endM, endN, numLoad, rem>(B_t, LDB, zmm, rem_);
      gemm::template startBCastA<isARowMajor, endM, endN, numBCast, numLoad>(A_t, LDA, zmm);
    }

    {
      // Interleave FMA and Bcast
      EIGEN_IF_CONSTEXPR(isAdd) {
        zmm.packet[startN * endM + startM] =
            pmadd(zmm.packet[endM * endN + numLoad + (startN + startK * endN) % numBCast],
                  zmm.packet[endM * endN + (startM + startK * endM) % numLoad], zmm.packet[startN * endM + startM]);
      }
      else {
        zmm.packet[startN * endM + startM] =
            pnmadd(zmm.packet[endM * endN + numLoad + (startN + startK * endN) % numBCast],
                   zmm.packet[endM * endN + (startM + startK * endM) % numLoad], zmm.packet[startN * endM + startM]);
      }
      // Bcast
      EIGEN_IF_CONSTEXPR(startM == endM - 1 && (numBCast + startN + startK * endN < endK * endN)) {
        zmm.packet[endM * endN + numLoad + (startN + startK * endN) % numBCast] = pload1<vec>(&A_t[idA<isARowMajor>(
            (numBCast + startN + startK * endN) % endN, (numBCast + startN + startK * endN) / endN, LDA)]);
      }
    }

    // We have updated all accumlators, time to load next set of B's
    EIGEN_IF_CONSTEXPR((startN == endN - 1) && (startM == endM - 1)) {
      gemm::template loadB<endM, endN, startK, endK, numLoad, numBCast, rem>(B_t, LDB, zmm, rem_);
    }
    aux_microKernel<isARowMajor, endM, endN, endK, counter - 1, numLoad, numBCast, rem>(B_t, A_t, LDB, LDA, zmm, rem_);
  }

  template <bool isARowMajor, int64_t endM, int64_t endN, int64_t endK, int64_t counter, int64_t numLoad,
            int64_t numBCast, bool rem>
  static EIGEN_ALWAYS_INLINE std::enable_if_t<(counter <= 0)> aux_microKernel(
      Scalar *B_t, Scalar *A_t, int64_t LDB, int64_t LDA, PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm,
      int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(B_t);
    EIGEN_UNUSED_VARIABLE(A_t);
    EIGEN_UNUSED_VARIABLE(LDB);
    EIGEN_UNUSED_VARIABLE(LDA);
    EIGEN_UNUSED_VARIABLE(zmm);
    EIGEN_UNUSED_VARIABLE(rem_);
  }

  /********************************************************
   * Wrappers for aux_XXXX to hide counter parameter
   ********************************************************/

  template <int64_t endM, int64_t endN>
  static EIGEN_ALWAYS_INLINE void setzero(PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm) {
    aux_setzero<endM, endN, endM * endN>(zmm);
  }

  /**
   * Ideally the compiler folds these into vaddp{s,d} with an embedded memory load.
   */
  template <int64_t endM, int64_t endN, bool rem = false>
  static EIGEN_ALWAYS_INLINE void updateC(Scalar *C_arr, int64_t LDC,
                                          PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm,
                                          int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    aux_updateC<endM, endN, endM * endN, rem>(C_arr, LDC, zmm, rem_);
  }

  template <int64_t endM, int64_t endN, bool rem = false>
  static EIGEN_ALWAYS_INLINE void storeC(Scalar *C_arr, int64_t LDC,
                                         PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm,
                                         int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    aux_storeC<endM, endN, endM * endN, rem>(C_arr, LDC, zmm, rem_);
  }

  /**
   * Use numLoad registers for loading B at start of microKernel
   */
  template <int64_t unrollM, int64_t unrollN, int64_t endL, bool rem>
  static EIGEN_ALWAYS_INLINE void startLoadB(Scalar *B_t, int64_t LDB,
                                             PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm,
                                             int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    aux_startLoadB<unrollM, unrollN, endL, endL, rem>(B_t, LDB, zmm, rem_);
  }

  /**
   * Use numBCast registers for broadcasting A at start of microKernel
   */
  template <bool isARowMajor, int64_t unrollM, int64_t unrollN, int64_t endB, int64_t numLoad>
  static EIGEN_ALWAYS_INLINE void startBCastA(Scalar *A_t, int64_t LDA,
                                              PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm) {
    aux_startBCastA<isARowMajor, unrollM, unrollN, endB, endB, numLoad>(A_t, LDA, zmm);
  }

  /**
   * Loads next set of B into vector registers between each K unroll.
   */
  template <int64_t endM, int64_t unrollN, int64_t currK, int64_t unrollK, int64_t numLoad, int64_t numBCast, bool rem>
  static EIGEN_ALWAYS_INLINE void loadB(Scalar *B_t, int64_t LDB,
                                        PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm,
                                        int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    aux_loadB<endM, endM, unrollN, currK, unrollK, numLoad, numBCast, rem>(B_t, LDB, zmm, rem_);
  }

  /**
   * Generates a microkernel for gemm (row-major) with unrolls {1,2,4,8}x{U1,U2,U3} to compute C -= A*B.
   * A matrix can be row/col-major. B matrix is assumed row-major.
   *
   * isARowMajor: is A row major
   * endM: Number registers per row
   * endN: Number of rows
   * endK: Loop unroll for K.
   * numLoad: Number of registers for loading B.
   * numBCast: Number of registers for broadcasting A.
   *
   * Ex: microkernel<isARowMajor,0,3,0,4,0,4,6,2>: 8x48 unroll (24 accumulators), k unrolled 4 times,
   * 6 register for loading B, 2 for broadcasting A.
   *
   * Note: Ideally the microkernel should not have any register spilling.
   * The avx instruction counts should be:
   *   - endK*endN vbroadcasts{s,d}
   *   - endK*endM vmovup{s,d}
   *   - endK*endN*endM FMAs
   *
   * From testing, there are no register spills with clang. There are register spills with GNU, which
   * causes a performance hit.
   */
  template <bool isARowMajor, int64_t endM, int64_t endN, int64_t endK, int64_t numLoad, int64_t numBCast,
            bool rem = false>
  static EIGEN_ALWAYS_INLINE void microKernel(Scalar *B_t, Scalar *A_t, int64_t LDB, int64_t LDA,
                                              PacketBlock<vec, EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS> &zmm,
                                              int64_t rem_ = 0) {
    EIGEN_UNUSED_VARIABLE(rem_);
    aux_microKernel<isARowMajor, endM, endN, endK, endM * endN * endK, numLoad, numBCast, rem>(B_t, A_t, LDB, LDA, zmm,
                                                                                               rem_);
  }
};
}  // namespace unrolls

#endif  // EIGEN_CORE_ARCH_AVX512_TRSM_UNROLLS_H
