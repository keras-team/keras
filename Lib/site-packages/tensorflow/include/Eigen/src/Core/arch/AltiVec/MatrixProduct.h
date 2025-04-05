// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Everton Constantino (everton.constantino@ibm.com)
// Copyright (C) 2021 Chip Kerchner (chip.kerchner@ibm.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_PRODUCT_ALTIVEC_H
#define EIGEN_MATRIX_PRODUCT_ALTIVEC_H

#ifndef EIGEN_ALTIVEC_USE_CUSTOM_PACK
#define EIGEN_ALTIVEC_USE_CUSTOM_PACK 1
#endif

#if !defined(EIGEN_ALTIVEC_DISABLE_MMA)
#define EIGEN_ALTIVEC_DISABLE_MMA 0
#endif

// Check for MMA builtin support.
#if !EIGEN_ALTIVEC_DISABLE_MMA && defined(__has_builtin)
#if __has_builtin(__builtin_mma_assemble_acc)
#define EIGEN_ALTIVEC_MMA_SUPPORT
#endif
#endif

// Check if and how we should actually use MMA if supported.
#if defined(EIGEN_ALTIVEC_MMA_SUPPORT)

#if !defined(EIGEN_ALTIVEC_ENABLE_MMA_DYNAMIC_DISPATCH)
#define EIGEN_ALTIVEC_ENABLE_MMA_DYNAMIC_DISPATCH 0
#endif

// Check if we want to enable dynamic dispatch. Not supported by LLVM.
#if EIGEN_ALTIVEC_ENABLE_MMA_DYNAMIC_DISPATCH && !EIGEN_COMP_LLVM
#define EIGEN_ALTIVEC_MMA_DYNAMIC_DISPATCH 1
// Otherwise, use MMA by default if available.
#elif defined(__MMA__)
#define EIGEN_ALTIVEC_MMA_ONLY 1
#endif

#endif  // EIGEN_ALTIVEC_MMA_SUPPORT

#include "MatrixProductCommon.h"

#if defined(EIGEN_ALTIVEC_MMA_ONLY) || defined(EIGEN_ALTIVEC_MMA_DYNAMIC_DISPATCH)
#include "MatrixProductMMA.h"
#endif

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/**************************
 * Constants and typedefs *
 **************************/
template <typename Scalar>
struct quad_traits {
  typedef typename packet_traits<Scalar>::type vectortype;
  typedef PacketBlock<vectortype, 4> type;
  typedef vectortype rhstype;
  enum { vectorsize = packet_traits<Scalar>::size, size = 4, rows = 4 };
};

template <>
struct quad_traits<double> {
  typedef Packet2d vectortype;
  typedef PacketBlock<vectortype, 4> type;
  typedef PacketBlock<Packet2d, 2> rhstype;
  enum { vectorsize = packet_traits<double>::size, size = 2, rows = 4 };
};

template <>
struct quad_traits<bfloat16> {
  typedef Packet8bf vectortype;
  typedef PacketBlock<vectortype, 4> type;
  typedef vectortype rhstype;
  enum { vectorsize = packet_traits<bfloat16>::size, size = 8, rows = 4 };
};

// MatrixProduct decomposes real/imaginary vectors into a real vector and an imaginary vector, this turned out
// to be faster than Eigen's usual approach of having real/imaginary pairs on a single vector. This constants then
// are responsible to extract from convert between Eigen's and MatrixProduct approach.

const static Packet16uc p16uc_GETREAL32 = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};

const static Packet16uc p16uc_GETIMAG32 = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

const static Packet16uc p16uc_GETREAL32b = {0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27};

const static Packet16uc p16uc_GETIMAG32b = {4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31};

/*********************************************
 * Single precision real and complex packing *
 * *******************************************/

/**
 * Symm packing is related to packing of symmetric adjoint blocks, as expected the packing leaves
 * the diagonal real, whatever is below it is copied from the respective upper diagonal element and
 * conjugated. There's no PanelMode available for symm packing.
 *
 * Packing in general is supposed to leave the lhs block and the rhs block easy to be read by gemm using
 * its respective rank-update instructions. The float32/64 versions are different because at this moment
 * the size of the accumulator is fixed at 512-bits so you can't have a 4x4 accumulator of 64-bit elements.
 *
 * As mentioned earlier MatrixProduct breaks complex numbers into a real vector and a complex vector so packing has
 * to take that into account, at the moment, we run pack the real part and then the imaginary part, this is the main
 * reason why packing for complex is broken down into several different parts, also the reason why we endup having a
 * float32/64 and complex float32/64 version.
 **/
template <typename Scalar, int StorageOrder>
EIGEN_ALWAYS_INLINE std::complex<Scalar> getAdjointVal(
    Index i, Index j, const_blas_data_mapper<std::complex<Scalar>, Index, StorageOrder>& dt) {
  std::complex<Scalar> v;
  if (i < j) {
    v.real(dt(j, i).real());
    v.imag(-dt(j, i).imag());
  } else if (i > j) {
    v.real(dt(i, j).real());
    v.imag(dt(i, j).imag());
  } else {
    v.real(dt(i, j).real());
    v.imag((Scalar)0.0);
  }
  return v;
}

template <typename Scalar, int StorageOrder, int N>
EIGEN_STRONG_INLINE void symm_pack_complex_rhs_helper(std::complex<Scalar>* blockB, const std::complex<Scalar>* _rhs,
                                                      Index rhsStride, Index rows, Index cols, Index k2) {
  const Index depth = k2 + rows;
  const_blas_data_mapper<std::complex<Scalar>, Index, StorageOrder> rhs(_rhs, rhsStride);
  const Index vectorSize = N * quad_traits<Scalar>::vectorsize;
  const Index vectorDelta = vectorSize * rows;
  Scalar* blockBf = reinterpret_cast<Scalar*>(blockB);

  Index rir = 0, rii, j = 0;
  for (; j + vectorSize <= cols; j += vectorSize) {
    rii = rir + vectorDelta;

    for (Index i = k2; i < depth; i++) {
      for (Index k = 0; k < vectorSize; k++) {
        std::complex<Scalar> v = getAdjointVal<Scalar, StorageOrder>(i, j + k, rhs);

        blockBf[rir + k] = v.real();
        blockBf[rii + k] = v.imag();
      }
      rir += vectorSize;
      rii += vectorSize;
    }

    rir += vectorDelta;
  }

  for (; j < cols; j++) {
    rii = rir + rows;

    for (Index i = k2; i < depth; i++) {
      std::complex<Scalar> v = getAdjointVal<Scalar, StorageOrder>(i, j, rhs);

      blockBf[rir] = v.real();
      blockBf[rii] = v.imag();

      rir += 1;
      rii += 1;
    }

    rir += rows;
  }
}

template <typename Scalar, int StorageOrder>
EIGEN_STRONG_INLINE void symm_pack_complex_lhs_helper(std::complex<Scalar>* blockA, const std::complex<Scalar>* _lhs,
                                                      Index lhsStride, Index cols, Index rows) {
  const Index depth = cols;
  const_blas_data_mapper<std::complex<Scalar>, Index, StorageOrder> lhs(_lhs, lhsStride);
  const Index vectorSize = quad_traits<Scalar>::vectorsize;
  const Index vectorDelta = vectorSize * depth;
  Scalar* blockAf = reinterpret_cast<Scalar*>(blockA);

  Index rir = 0, rii, j = 0;
  for (; j + vectorSize <= rows; j += vectorSize) {
    rii = rir + vectorDelta;

    for (Index i = 0; i < depth; i++) {
      for (Index k = 0; k < vectorSize; k++) {
        std::complex<Scalar> v = getAdjointVal<Scalar, StorageOrder>(j + k, i, lhs);

        blockAf[rir + k] = v.real();
        blockAf[rii + k] = v.imag();
      }
      rir += vectorSize;
      rii += vectorSize;
    }

    rir += vectorDelta;
  }

  if (j < rows) {
    rii = rir + ((rows - j) * depth);

    for (Index i = 0; i < depth; i++) {
      Index k = j;
      for (; k < rows; k++) {
        std::complex<Scalar> v = getAdjointVal<Scalar, StorageOrder>(k, i, lhs);

        blockAf[rir] = v.real();
        blockAf[rii] = v.imag();

        rir += 1;
        rii += 1;
      }
    }
  }
}

template <typename Scalar, int StorageOrder, int N>
EIGEN_STRONG_INLINE void symm_pack_rhs_helper(Scalar* blockB, const Scalar* _rhs, Index rhsStride, Index rows,
                                              Index cols, Index k2) {
  const Index depth = k2 + rows;
  const_blas_data_mapper<Scalar, Index, StorageOrder> rhs(_rhs, rhsStride);
  const Index vectorSize = quad_traits<Scalar>::vectorsize;

  Index ri = 0, j = 0;
  for (; j + N * vectorSize <= cols; j += N * vectorSize) {
    Index i = k2;
    for (; i < depth; i++) {
      for (Index k = 0; k < N * vectorSize; k++) {
        if (i <= j + k)
          blockB[ri + k] = rhs(j + k, i);
        else
          blockB[ri + k] = rhs(i, j + k);
      }
      ri += N * vectorSize;
    }
  }

  for (; j < cols; j++) {
    for (Index i = k2; i < depth; i++) {
      if (j <= i)
        blockB[ri] = rhs(i, j);
      else
        blockB[ri] = rhs(j, i);
      ri += 1;
    }
  }
}

template <typename Scalar, int StorageOrder>
EIGEN_STRONG_INLINE void symm_pack_lhs_helper(Scalar* blockA, const Scalar* _lhs, Index lhsStride, Index cols,
                                              Index rows) {
  const Index depth = cols;
  const_blas_data_mapper<Scalar, Index, StorageOrder> lhs(_lhs, lhsStride);
  const Index vectorSize = quad_traits<Scalar>::vectorsize;

  Index ri = 0, j = 0;
  for (; j + vectorSize <= rows; j += vectorSize) {
    Index i = 0;

    for (; i < depth; i++) {
      for (Index k = 0; k < vectorSize; k++) {
        if (i <= j + k)
          blockA[ri + k] = lhs(j + k, i);
        else
          blockA[ri + k] = lhs(i, j + k);
      }
      ri += vectorSize;
    }
  }

  if (j < rows) {
    for (Index i = 0; i < depth; i++) {
      Index k = j;
      for (; k < rows; k++) {
        if (i <= k)
          blockA[ri] = lhs(k, i);
        else
          blockA[ri] = lhs(i, k);
        ri += 1;
      }
    }
  }
}

template <typename Index, int nr, int StorageOrder>
struct symm_pack_rhs<std::complex<float>, Index, nr, StorageOrder> {
  void operator()(std::complex<float>* blockB, const std::complex<float>* _rhs, Index rhsStride, Index rows, Index cols,
                  Index k2) {
    symm_pack_complex_rhs_helper<float, StorageOrder, 1>(blockB, _rhs, rhsStride, rows, cols, k2);
  }
};

template <typename Index, int Pack1, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<std::complex<float>, Index, Pack1, Pack2_dummy, StorageOrder> {
  void operator()(std::complex<float>* blockA, const std::complex<float>* _lhs, Index lhsStride, Index cols,
                  Index rows) {
    symm_pack_complex_lhs_helper<float, StorageOrder>(blockA, _lhs, lhsStride, cols, rows);
  }
};

// *********** symm_pack std::complex<float64> ***********

template <typename Index, int nr, int StorageOrder>
struct symm_pack_rhs<std::complex<double>, Index, nr, StorageOrder> {
  void operator()(std::complex<double>* blockB, const std::complex<double>* _rhs, Index rhsStride, Index rows,
                  Index cols, Index k2) {
    symm_pack_complex_rhs_helper<double, StorageOrder, 2>(blockB, _rhs, rhsStride, rows, cols, k2);
  }
};

template <typename Index, int Pack1, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<std::complex<double>, Index, Pack1, Pack2_dummy, StorageOrder> {
  void operator()(std::complex<double>* blockA, const std::complex<double>* _lhs, Index lhsStride, Index cols,
                  Index rows) {
    symm_pack_complex_lhs_helper<double, StorageOrder>(blockA, _lhs, lhsStride, cols, rows);
  }
};

// *********** symm_pack float32 ***********
template <typename Index, int nr, int StorageOrder>
struct symm_pack_rhs<float, Index, nr, StorageOrder> {
  void operator()(float* blockB, const float* _rhs, Index rhsStride, Index rows, Index cols, Index k2) {
    symm_pack_rhs_helper<float, StorageOrder, 1>(blockB, _rhs, rhsStride, rows, cols, k2);
  }
};

template <typename Index, int Pack1, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<float, Index, Pack1, Pack2_dummy, StorageOrder> {
  void operator()(float* blockA, const float* _lhs, Index lhsStride, Index cols, Index rows) {
    symm_pack_lhs_helper<float, StorageOrder>(blockA, _lhs, lhsStride, cols, rows);
  }
};

// *********** symm_pack float64 ***********
template <typename Index, int nr, int StorageOrder>
struct symm_pack_rhs<double, Index, nr, StorageOrder> {
  void operator()(double* blockB, const double* _rhs, Index rhsStride, Index rows, Index cols, Index k2) {
    symm_pack_rhs_helper<double, StorageOrder, 2>(blockB, _rhs, rhsStride, rows, cols, k2);
  }
};

template <typename Index, int Pack1, int Pack2_dummy, int StorageOrder>
struct symm_pack_lhs<double, Index, Pack1, Pack2_dummy, StorageOrder> {
  void operator()(double* blockA, const double* _lhs, Index lhsStride, Index cols, Index rows) {
    symm_pack_lhs_helper<double, StorageOrder>(blockA, _lhs, lhsStride, cols, rows);
  }
};

/**
 * PanelMode
 * Packing might be called several times before being multiplied by gebp_kernel, this happens because
 * on special occasions it fills part of block with other parts of the matrix. Two variables control
 * how PanelMode should behave: offset and stride. The idea is that those variables represent whatever
 * is going to be the real offset and stride in the future and this is what you should obey. The process
 * is to behave as you would with normal packing but leave the start of each part with the correct offset
 * and the end as well respecting the real stride the block will have. Gebp is aware of both blocks stride
 * and offset and behaves accordingly.
 **/

template <typename Scalar, typename Packet, int N>
EIGEN_ALWAYS_INLINE void storeBlock(Scalar* to, PacketBlock<Packet, N>& block) {
  const Index size = 16 / sizeof(Scalar);
  pstore<Scalar>(to + (0 * size), block.packet[0]);
  pstore<Scalar>(to + (1 * size), block.packet[1]);
  if (N > 2) {
    pstore<Scalar>(to + (2 * size), block.packet[2]);
  }
  if (N > 3) {
    pstore<Scalar>(to + (3 * size), block.packet[3]);
  }
}

// General template for lhs & rhs complex packing.
template <typename Scalar, typename DataMapper, typename Packet, typename PacketC, int StorageOrder, bool Conjugate,
          bool PanelMode, bool UseLhs>
struct dhs_cpack {
  template <bool transpose>
  EIGEN_ALWAYS_INLINE void dhs_cblock(PacketBlock<PacketC, 8>& cblock, PacketBlock<Packet, 4>& block,
                                      Packet16uc permute) {
    if (transpose) {
      block.packet[0] = vec_perm(cblock.packet[0].v, cblock.packet[1].v, permute);
      block.packet[1] = vec_perm(cblock.packet[2].v, cblock.packet[3].v, permute);
      block.packet[2] = vec_perm(cblock.packet[4].v, cblock.packet[5].v, permute);
      block.packet[3] = vec_perm(cblock.packet[6].v, cblock.packet[7].v, permute);

      Packet4f t0, t1, t2, t3;
#ifdef EIGEN_VECTORIZE_VSX
      t0 = reinterpret_cast<Packet>(
          vec_mergeh(reinterpret_cast<Packet2ul>(block.packet[0]), reinterpret_cast<Packet2ul>(block.packet[1])));
      t1 = reinterpret_cast<Packet>(
          vec_mergel(reinterpret_cast<Packet2ul>(block.packet[0]), reinterpret_cast<Packet2ul>(block.packet[1])));
      t2 = reinterpret_cast<Packet>(
          vec_mergeh(reinterpret_cast<Packet2ul>(block.packet[2]), reinterpret_cast<Packet2ul>(block.packet[3])));
      t3 = reinterpret_cast<Packet>(
          vec_mergel(reinterpret_cast<Packet2ul>(block.packet[2]), reinterpret_cast<Packet2ul>(block.packet[3])));
#else
      t0 = reinterpret_cast<Packet>(vec_perm(block.packet[0], block.packet[1], p16uc_TRANSPOSE64_HI));
      t1 = reinterpret_cast<Packet>(vec_perm(block.packet[0], block.packet[1], p16uc_TRANSPOSE64_LO));
      t2 = reinterpret_cast<Packet>(vec_perm(block.packet[2], block.packet[3], p16uc_TRANSPOSE64_HI));
      t3 = reinterpret_cast<Packet>(vec_perm(block.packet[2], block.packet[3], p16uc_TRANSPOSE64_LO));
#endif

      block.packet[0] = t0;
      block.packet[1] = t1;
      block.packet[2] = t2;
      block.packet[3] = t3;
    } else {
      block.packet[0] = vec_perm(cblock.packet[0].v, cblock.packet[4].v, permute);
      block.packet[1] = vec_perm(cblock.packet[1].v, cblock.packet[5].v, permute);
      block.packet[2] = vec_perm(cblock.packet[2].v, cblock.packet[6].v, permute);
      block.packet[3] = vec_perm(cblock.packet[3].v, cblock.packet[7].v, permute);
    }
  }

  EIGEN_ALWAYS_INLINE void dhs_ccopy(Scalar* blockAt, const DataMapper& lhs2, Index& i, Index& rir, Index& rii,
                                     Index depth, const Index vectorSize) {
    PacketBlock<Packet, 4> blockr, blocki;
    PacketBlock<PacketC, 8> cblock;

    for (; i + vectorSize <= depth; i += vectorSize) {
      if (UseLhs) {
        bload<DataMapper, PacketC, 2, StorageOrder, true, 4>(cblock, lhs2, 0, i);
      } else {
        bload<DataMapper, PacketC, 2, StorageOrder, true, 4>(cblock, lhs2, i, 0);
      }

      if (((StorageOrder == RowMajor) && UseLhs) || (((StorageOrder == ColMajor) && !UseLhs))) {
        dhs_cblock<true>(cblock, blockr, p16uc_GETREAL32b);
        dhs_cblock<true>(cblock, blocki, p16uc_GETIMAG32b);
      } else {
        dhs_cblock<false>(cblock, blockr, p16uc_GETREAL32);
        dhs_cblock<false>(cblock, blocki, p16uc_GETIMAG32);
      }

      if (Conjugate) {
        blocki.packet[0] = -blocki.packet[0];
        blocki.packet[1] = -blocki.packet[1];
        blocki.packet[2] = -blocki.packet[2];
        blocki.packet[3] = -blocki.packet[3];
      }

      storeBlock<Scalar, Packet, 4>(blockAt + rir, blockr);
      storeBlock<Scalar, Packet, 4>(blockAt + rii, blocki);

      rir += 4 * vectorSize;
      rii += 4 * vectorSize;
    }
  }

  EIGEN_STRONG_INLINE void operator()(std::complex<Scalar>* blockA, const DataMapper& lhs, Index depth, Index rows,
                                      Index stride, Index offset) {
    const Index vectorSize = quad_traits<Scalar>::vectorsize;
    const Index vectorDelta = vectorSize * ((PanelMode) ? stride : depth);
    Index rir = ((PanelMode) ? (vectorSize * offset) : 0), rii;
    Scalar* blockAt = reinterpret_cast<Scalar*>(blockA);
    Index j = 0;

    for (; j + vectorSize <= rows; j += vectorSize) {
      const DataMapper lhs2 = UseLhs ? lhs.getSubMapper(j, 0) : lhs.getSubMapper(0, j);
      Index i = 0;

      rii = rir + vectorDelta;

      dhs_ccopy(blockAt, lhs2, i, rir, rii, depth, vectorSize);

      for (; i < depth; i++) {
        PacketBlock<Packet, 1> blockr, blocki;
        PacketBlock<PacketC, 2> cblock;

        if (((StorageOrder == ColMajor) && UseLhs) || (((StorageOrder == RowMajor) && !UseLhs))) {
          if (UseLhs) {
            cblock.packet[0] = lhs2.template loadPacket<PacketC>(0, i);
            cblock.packet[1] = lhs2.template loadPacket<PacketC>(2, i);
          } else {
            cblock.packet[0] = lhs2.template loadPacket<PacketC>(i, 0);
            cblock.packet[1] = lhs2.template loadPacket<PacketC>(i, 2);
          }
        } else {
          if (UseLhs) {
            cblock.packet[0] = pload2(lhs2(0, i), lhs2(1, i));
            cblock.packet[1] = pload2(lhs2(2, i), lhs2(3, i));
          } else {
            cblock.packet[0] = pload2(lhs2(i, 0), lhs2(i, 1));
            cblock.packet[1] = pload2(lhs2(i, 2), lhs2(i, 3));
          }
        }

        blockr.packet[0] = vec_perm(cblock.packet[0].v, cblock.packet[1].v, p16uc_GETREAL32);
        blocki.packet[0] = vec_perm(cblock.packet[0].v, cblock.packet[1].v, p16uc_GETIMAG32);

        if (Conjugate) {
          blocki.packet[0] = -blocki.packet[0];
        }

        pstore<Scalar>(blockAt + rir, blockr.packet[0]);
        pstore<Scalar>(blockAt + rii, blocki.packet[0]);

        rir += vectorSize;
        rii += vectorSize;
      }

      rir += ((PanelMode) ? (vectorSize * (2 * stride - depth)) : vectorDelta);
    }

    if (!UseLhs) {
      if (PanelMode) rir -= (offset * (vectorSize - 1));

      for (; j < rows; j++) {
        const DataMapper lhs2 = lhs.getSubMapper(0, j);
        rii = rir + ((PanelMode) ? stride : depth);

        for (Index i = 0; i < depth; i++) {
          blockAt[rir] = lhs2(i, 0).real();

          if (Conjugate)
            blockAt[rii] = -lhs2(i, 0).imag();
          else
            blockAt[rii] = lhs2(i, 0).imag();

          rir += 1;
          rii += 1;
        }

        rir += ((PanelMode) ? (2 * stride - depth) : depth);
      }
    } else {
      if (j < rows) {
        if (PanelMode) rir += (offset * (rows - j - vectorSize));
        rii = rir + (((PanelMode) ? stride : depth) * (rows - j));

        for (Index i = 0; i < depth; i++) {
          Index k = j;
          for (; k < rows; k++) {
            blockAt[rir] = lhs(k, i).real();

            if (Conjugate)
              blockAt[rii] = -lhs(k, i).imag();
            else
              blockAt[rii] = lhs(k, i).imag();

            rir += 1;
            rii += 1;
          }
        }
      }
    }
  }
};

// General template for lhs & rhs packing.
template <typename Scalar, typename DataMapper, typename Packet, int StorageOrder, bool PanelMode, bool UseLhs>
struct dhs_pack {
  template <Index n>
  EIGEN_ALWAYS_INLINE void dhs_copy(Scalar* blockA, const DataMapper& lhs2, Index& i, Index& ri, Index depth,
                                    const Index vectorSize) {
    PacketBlock<Packet, 4> block[n];

    for (; i + n * vectorSize <= depth; i += n * vectorSize) {
      for (Index k = 0; k < n; k++) {
        if (UseLhs) {
          bload<DataMapper, Packet, 4, StorageOrder, false, 4>(block[k], lhs2, 0, i + k * vectorSize);
        } else {
          bload<DataMapper, Packet, 4, StorageOrder, false, 4>(block[k], lhs2, i + k * vectorSize, 0);
        }
      }

      if (((StorageOrder == RowMajor) && UseLhs) || ((StorageOrder == ColMajor) && !UseLhs)) {
        for (Index k = 0; k < n; k++) {
          ptranspose(block[k]);
        }
      }

      for (Index k = 0; k < n; k++) {
        storeBlock<Scalar, Packet, 4>(blockA + ri + k * 4 * vectorSize, block[k]);
      }

      ri += n * 4 * vectorSize;
    }
  }

  EIGEN_STRONG_INLINE void operator()(Scalar* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride,
                                      Index offset) {
    const Index vectorSize = quad_traits<Scalar>::vectorsize;
    Index ri = 0, j = 0;

    for (; j + vectorSize <= rows; j += vectorSize) {
      const DataMapper lhs2 = UseLhs ? lhs.getSubMapper(j, 0) : lhs.getSubMapper(0, j);
      Index i = 0;

      if (PanelMode) ri += vectorSize * offset;

      dhs_copy<4>(blockA, lhs2, i, ri, depth, vectorSize);
      dhs_copy<2>(blockA, lhs2, i, ri, depth, vectorSize);
      dhs_copy<1>(blockA, lhs2, i, ri, depth, vectorSize);

      for (; i < depth; i++) {
        if (((StorageOrder == RowMajor) && UseLhs) || ((StorageOrder == ColMajor) && !UseLhs)) {
          if (UseLhs) {
            blockA[ri + 0] = lhs2(0, i);
            blockA[ri + 1] = lhs2(1, i);
            blockA[ri + 2] = lhs2(2, i);
            blockA[ri + 3] = lhs2(3, i);
          } else {
            blockA[ri + 0] = lhs2(i, 0);
            blockA[ri + 1] = lhs2(i, 1);
            blockA[ri + 2] = lhs2(i, 2);
            blockA[ri + 3] = lhs2(i, 3);
          }
        } else {
          Packet lhsV;
          if (UseLhs) {
            lhsV = lhs2.template loadPacket<Packet>(0, i);
          } else {
            lhsV = lhs2.template loadPacket<Packet>(i, 0);
          }
          pstore<Scalar>(blockA + ri, lhsV);
        }

        ri += vectorSize;
      }

      if (PanelMode) ri += vectorSize * (stride - offset - depth);
    }

    if (!UseLhs) {
      if (PanelMode) ri += offset;

      for (; j < rows; j++) {
        const DataMapper lhs2 = lhs.getSubMapper(0, j);
        for (Index i = 0; i < depth; i++) {
          blockA[ri] = lhs2(i, 0);
          ri += 1;
        }

        if (PanelMode) ri += stride - depth;
      }
    } else {
      if (j < rows) {
        if (PanelMode) ri += offset * (rows - j);

        for (Index i = 0; i < depth; i++) {
          Index k = j;
          for (; k < rows; k++) {
            blockA[ri] = lhs(k, i);
            ri += 1;
          }
        }
      }
    }
  }
};

// General template for lhs packing, float64 specialization.
template <typename DataMapper, int StorageOrder, bool PanelMode>
struct dhs_pack<double, DataMapper, Packet2d, StorageOrder, PanelMode, true> {
  template <Index n>
  EIGEN_ALWAYS_INLINE void dhs_copy(double* blockA, const DataMapper& lhs2, Index& i, Index& ri, Index depth,
                                    const Index vectorSize) {
    PacketBlock<Packet2d, 2> block[n];

    for (; i + n * vectorSize <= depth; i += n * vectorSize) {
      for (Index k = 0; k < n; k++) {
        if (StorageOrder == RowMajor) {
          block[k].packet[0] = lhs2.template loadPacket<Packet2d>(0, i + k * vectorSize);
          block[k].packet[1] = lhs2.template loadPacket<Packet2d>(1, i + k * vectorSize);
        } else {
          block[k].packet[0] = lhs2.template loadPacket<Packet2d>(0, i + k * vectorSize + 0);
          block[k].packet[1] = lhs2.template loadPacket<Packet2d>(0, i + k * vectorSize + 1);
        }
      }

      if (StorageOrder == RowMajor) {
        for (Index k = 0; k < n; k++) {
          ptranspose(block[k]);
        }
      }

      for (Index k = 0; k < n; k++) {
        storeBlock<double, Packet2d, 2>(blockA + ri + k * 2 * vectorSize, block[k]);
      }

      ri += n * 2 * vectorSize;
    }
  }

  EIGEN_STRONG_INLINE void operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride,
                                      Index offset) {
    const Index vectorSize = quad_traits<double>::vectorsize;
    Index ri = 0, j = 0;

    for (; j + vectorSize <= rows; j += vectorSize) {
      const DataMapper lhs2 = lhs.getSubMapper(j, 0);
      Index i = 0;

      if (PanelMode) ri += vectorSize * offset;

      dhs_copy<4>(blockA, lhs2, i, ri, depth, vectorSize);
      dhs_copy<2>(blockA, lhs2, i, ri, depth, vectorSize);
      dhs_copy<1>(blockA, lhs2, i, ri, depth, vectorSize);

      for (; i < depth; i++) {
        if (StorageOrder == RowMajor) {
          blockA[ri + 0] = lhs2(0, i);
          blockA[ri + 1] = lhs2(1, i);
        } else {
          Packet2d lhsV = lhs2.template loadPacket<Packet2d>(0, i);
          pstore<double>(blockA + ri, lhsV);
        }

        ri += vectorSize;
      }

      if (PanelMode) ri += vectorSize * (stride - offset - depth);
    }

    if (j < rows) {
      if (PanelMode) ri += offset * (rows - j);

      for (Index i = 0; i < depth; i++) {
        Index k = j;
        for (; k < rows; k++) {
          blockA[ri] = lhs(k, i);
          ri += 1;
        }
      }
    }
  }
};

// General template for rhs packing, float64 specialization.
template <typename DataMapper, int StorageOrder, bool PanelMode>
struct dhs_pack<double, DataMapper, Packet2d, StorageOrder, PanelMode, false> {
  template <Index n>
  EIGEN_ALWAYS_INLINE void dhs_copy(double* blockB, const DataMapper& rhs2, Index& i, Index& ri, Index depth,
                                    const Index vectorSize) {
    PacketBlock<Packet2d, 2> block1[n], block2[n];
    PacketBlock<Packet2d, 4> block3[n];

    for (; i + n * vectorSize <= depth; i += n * vectorSize) {
      for (Index k = 0; k < n; k++) {
        if (StorageOrder == ColMajor) {
          block1[k].packet[0] = rhs2.template loadPacket<Packet2d>(i + k * vectorSize, 0);
          block1[k].packet[1] = rhs2.template loadPacket<Packet2d>(i + k * vectorSize, 1);
          block2[k].packet[0] = rhs2.template loadPacket<Packet2d>(i + k * vectorSize, 2);
          block2[k].packet[1] = rhs2.template loadPacket<Packet2d>(i + k * vectorSize, 3);
        } else {
          block3[k].packet[0] = rhs2.template loadPacket<Packet2d>(i + k * vectorSize + 0, 0);  //[a1 a2]
          block3[k].packet[1] = rhs2.template loadPacket<Packet2d>(i + k * vectorSize + 0, 2);  //[a3 a4]
          block3[k].packet[2] = rhs2.template loadPacket<Packet2d>(i + k * vectorSize + 1, 0);  //[b1 b2]
          block3[k].packet[3] = rhs2.template loadPacket<Packet2d>(i + k * vectorSize + 1, 2);  //[b3 b4]
        }
      }

      if (StorageOrder == ColMajor) {
        for (Index k = 0; k < n; k++) {
          ptranspose(block1[k]);
          ptranspose(block2[k]);
        }
      }

      for (Index k = 0; k < n; k++) {
        if (StorageOrder == ColMajor) {
          pstore<double>(blockB + ri + k * 4 * vectorSize, block1[k].packet[0]);
          pstore<double>(blockB + ri + k * 4 * vectorSize + 2, block2[k].packet[0]);
          pstore<double>(blockB + ri + k * 4 * vectorSize + 4, block1[k].packet[1]);
          pstore<double>(blockB + ri + k * 4 * vectorSize + 6, block2[k].packet[1]);
        } else {
          storeBlock<double, Packet2d, 4>(blockB + ri + k * 4 * vectorSize, block3[k]);
        }
      }

      ri += n * 4 * vectorSize;
    }
  }

  EIGEN_STRONG_INLINE void operator()(double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride,
                                      Index offset) {
    const Index vectorSize = quad_traits<double>::vectorsize;
    Index ri = 0, j = 0;

    for (; j + 2 * vectorSize <= cols; j += 2 * vectorSize) {
      const DataMapper rhs2 = rhs.getSubMapper(0, j);
      Index i = 0;

      if (PanelMode) ri += offset * (2 * vectorSize);

      dhs_copy<4>(blockB, rhs2, i, ri, depth, vectorSize);
      dhs_copy<2>(blockB, rhs2, i, ri, depth, vectorSize);
      dhs_copy<1>(blockB, rhs2, i, ri, depth, vectorSize);

      for (; i < depth; i++) {
        if (StorageOrder == ColMajor) {
          blockB[ri + 0] = rhs2(i, 0);
          blockB[ri + 1] = rhs2(i, 1);

          ri += vectorSize;

          blockB[ri + 0] = rhs2(i, 2);
          blockB[ri + 1] = rhs2(i, 3);
        } else {
          Packet2d rhsV = rhs2.template loadPacket<Packet2d>(i, 0);
          pstore<double>(blockB + ri, rhsV);

          ri += vectorSize;

          rhsV = rhs2.template loadPacket<Packet2d>(i, 2);
          pstore<double>(blockB + ri, rhsV);
        }
        ri += vectorSize;
      }

      if (PanelMode) ri += (2 * vectorSize) * (stride - offset - depth);
    }

    if (PanelMode) ri += offset;

    for (; j < cols; j++) {
      const DataMapper rhs2 = rhs.getSubMapper(0, j);
      for (Index i = 0; i < depth; i++) {
        blockB[ri] = rhs2(i, 0);
        ri += 1;
      }

      if (PanelMode) ri += stride - depth;
    }
  }
};

// General template for lhs packing, bfloat16 specialization.
template <typename DataMapper, int StorageOrder, bool PanelMode>
struct dhs_pack<bfloat16, DataMapper, Packet8bf, StorageOrder, PanelMode, true> {
  EIGEN_STRONG_INLINE void operator()(bfloat16* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride,
                                      Index offset) {
    const Index vectorSize = quad_traits<bfloat16>::vectorsize;
    Index ri = 0, j = 0;

    for (; j + 2 * vectorSize <= rows; j += 2 * vectorSize) {
      const DataMapper lhs2 = lhs.getSubMapper(j, 0);
      Index i = 0;

      if (PanelMode) ri += 2 * vectorSize * offset;

      if (StorageOrder == ColMajor) {
        for (; i + 2 <= depth; i += 2) {
          PacketBlock<Packet8bf, 4> block;

          block.packet[0] = lhs2.template loadPacket<Packet8bf>(0 * vectorSize, i + 0);
          block.packet[1] = lhs2.template loadPacket<Packet8bf>(1 * vectorSize, i + 0);
          block.packet[2] = lhs2.template loadPacket<Packet8bf>(0 * vectorSize, i + 1);
          block.packet[3] = lhs2.template loadPacket<Packet8bf>(1 * vectorSize, i + 1);

          Packet8bf t0, t1;
          t0 = vec_mergeh(block.packet[0].m_val, block.packet[2].m_val);
          t1 = vec_mergel(block.packet[0].m_val, block.packet[2].m_val);
          block.packet[2] = vec_mergeh(block.packet[1].m_val, block.packet[3].m_val);
          block.packet[3] = vec_mergel(block.packet[1].m_val, block.packet[3].m_val);
          block.packet[0] = t0;
          block.packet[1] = t1;

          storeBlock<bfloat16, Packet8bf, 4>(blockA + ri, block);

          ri += 2 * 2 * vectorSize;
        }
        if (depth & 1) {
          PacketBlock<Packet8bf, 2> block;

          block.packet[0] = lhs2.template loadPacket<Packet8bf>(0 * vectorSize, i + 0);
          block.packet[1] = lhs2.template loadPacket<Packet8bf>(1 * vectorSize, i + 0);

          storeBlock<bfloat16, Packet8bf, 2>(blockA + ri, block);

          ri += 2 * vectorSize;
        }
      } else {
        for (; i + vectorSize <= depth; i += vectorSize) {
          PacketBlock<Packet8bf, 8> block1, block2;

          bload<DataMapper, Packet8bf, 8, StorageOrder, false, 8>(block1, lhs2, 0 * vectorSize, i);
          bload<DataMapper, Packet8bf, 8, StorageOrder, false, 8>(block2, lhs2, 1 * vectorSize, i);

          Packet4ui v1[8], v2[8];

          v1[0] = vec_mergeh(reinterpret_cast<Packet4ui>(block1.packet[0].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[1].m_val));
          v1[1] = vec_mergel(reinterpret_cast<Packet4ui>(block1.packet[0].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[1].m_val));
          v1[2] = vec_mergeh(reinterpret_cast<Packet4ui>(block1.packet[2].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[3].m_val));
          v1[3] = vec_mergel(reinterpret_cast<Packet4ui>(block1.packet[2].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[3].m_val));
          v1[4] = vec_mergeh(reinterpret_cast<Packet4ui>(block1.packet[4].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[5].m_val));
          v1[5] = vec_mergel(reinterpret_cast<Packet4ui>(block1.packet[4].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[5].m_val));
          v1[6] = vec_mergeh(reinterpret_cast<Packet4ui>(block1.packet[6].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[7].m_val));
          v1[7] = vec_mergel(reinterpret_cast<Packet4ui>(block1.packet[6].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[7].m_val));
          v2[0] = vec_mergeh(reinterpret_cast<Packet4ui>(block2.packet[0].m_val),
                             reinterpret_cast<Packet4ui>(block2.packet[1].m_val));
          v2[1] = vec_mergel(reinterpret_cast<Packet4ui>(block2.packet[0].m_val),
                             reinterpret_cast<Packet4ui>(block2.packet[1].m_val));
          v2[2] = vec_mergeh(reinterpret_cast<Packet4ui>(block2.packet[2].m_val),
                             reinterpret_cast<Packet4ui>(block2.packet[3].m_val));
          v2[3] = vec_mergel(reinterpret_cast<Packet4ui>(block2.packet[2].m_val),
                             reinterpret_cast<Packet4ui>(block2.packet[3].m_val));
          v2[4] = vec_mergeh(reinterpret_cast<Packet4ui>(block2.packet[4].m_val),
                             reinterpret_cast<Packet4ui>(block2.packet[5].m_val));
          v2[5] = vec_mergel(reinterpret_cast<Packet4ui>(block2.packet[4].m_val),
                             reinterpret_cast<Packet4ui>(block2.packet[5].m_val));
          v2[6] = vec_mergeh(reinterpret_cast<Packet4ui>(block2.packet[6].m_val),
                             reinterpret_cast<Packet4ui>(block2.packet[7].m_val));
          v2[7] = vec_mergel(reinterpret_cast<Packet4ui>(block2.packet[6].m_val),
                             reinterpret_cast<Packet4ui>(block2.packet[7].m_val));

#ifdef EIGEN_VECTORIZE_VSX
          block1.packet[0] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v1[0]), reinterpret_cast<Packet2ul>(v1[2])));
          block1.packet[2] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v1[0]), reinterpret_cast<Packet2ul>(v1[2])));
          block1.packet[4] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v1[1]), reinterpret_cast<Packet2ul>(v1[3])));
          block1.packet[6] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v1[1]), reinterpret_cast<Packet2ul>(v1[3])));
          block1.packet[1] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v1[4]), reinterpret_cast<Packet2ul>(v1[6])));
          block1.packet[3] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v1[4]), reinterpret_cast<Packet2ul>(v1[6])));
          block1.packet[5] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v1[5]), reinterpret_cast<Packet2ul>(v1[7])));
          block1.packet[7] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v1[5]), reinterpret_cast<Packet2ul>(v1[7])));
          block2.packet[0] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v2[0]), reinterpret_cast<Packet2ul>(v2[2])));
          block2.packet[2] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v2[0]), reinterpret_cast<Packet2ul>(v2[2])));
          block2.packet[4] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v2[1]), reinterpret_cast<Packet2ul>(v2[3])));
          block2.packet[6] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v2[1]), reinterpret_cast<Packet2ul>(v2[3])));
          block2.packet[1] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v2[4]), reinterpret_cast<Packet2ul>(v2[6])));
          block2.packet[3] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v2[4]), reinterpret_cast<Packet2ul>(v2[6])));
          block2.packet[5] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v2[5]), reinterpret_cast<Packet2ul>(v2[7])));
          block2.packet[7] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v2[5]), reinterpret_cast<Packet2ul>(v2[7])));
#else
          block1.packet[0] = reinterpret_cast<Packet8us>(vec_perm(v1[0], v1[2], p16uc_TRANSPOSE64_HI));
          block1.packet[2] = reinterpret_cast<Packet8us>(vec_perm(v1[0], v1[2], p16uc_TRANSPOSE64_LO));
          block1.packet[4] = reinterpret_cast<Packet8us>(vec_perm(v1[1], v1[3], p16uc_TRANSPOSE64_HI));
          block1.packet[6] = reinterpret_cast<Packet8us>(vec_perm(v1[1], v1[3], p16uc_TRANSPOSE64_LO));
          block1.packet[1] = reinterpret_cast<Packet8us>(vec_perm(v1[4], v1[6], p16uc_TRANSPOSE64_HI));
          block1.packet[3] = reinterpret_cast<Packet8us>(vec_perm(v1[4], v1[6], p16uc_TRANSPOSE64_LO));
          block1.packet[5] = reinterpret_cast<Packet8us>(vec_perm(v1[5], v1[7], p16uc_TRANSPOSE64_HI));
          block1.packet[7] = reinterpret_cast<Packet8us>(vec_perm(v1[5], v1[7], p16uc_TRANSPOSE64_LO));
          block2.packet[0] = reinterpret_cast<Packet8us>(vec_perm(v2[0], v2[2], p16uc_TRANSPOSE64_HI));
          block2.packet[2] = reinterpret_cast<Packet8us>(vec_perm(v2[0], v2[2], p16uc_TRANSPOSE64_LO));
          block2.packet[4] = reinterpret_cast<Packet8us>(vec_perm(v2[1], v2[3], p16uc_TRANSPOSE64_HI));
          block2.packet[6] = reinterpret_cast<Packet8us>(vec_perm(v2[1], v2[3], p16uc_TRANSPOSE64_LO));
          block2.packet[1] = reinterpret_cast<Packet8us>(vec_perm(v2[4], v2[6], p16uc_TRANSPOSE64_HI));
          block2.packet[3] = reinterpret_cast<Packet8us>(vec_perm(v2[4], v2[6], p16uc_TRANSPOSE64_LO));
          block2.packet[5] = reinterpret_cast<Packet8us>(vec_perm(v2[5], v2[7], p16uc_TRANSPOSE64_HI));
          block2.packet[7] = reinterpret_cast<Packet8us>(vec_perm(v2[5], v2[7], p16uc_TRANSPOSE64_LO));
#endif

          for (Index M = 0; M < 8; M += 2) {
            pstore<bfloat16>(blockA + ri + (0 * vectorSize) + (2 * vectorSize * M), block1.packet[M + 0]);
            pstore<bfloat16>(blockA + ri + (1 * vectorSize) + (2 * vectorSize * M), block1.packet[M + 1]);
            pstore<bfloat16>(blockA + ri + (2 * vectorSize) + (2 * vectorSize * M), block2.packet[M + 0]);
            pstore<bfloat16>(blockA + ri + (3 * vectorSize) + (2 * vectorSize * M), block2.packet[M + 1]);
          }

          ri += 2 * vectorSize * vectorSize;
        }
        for (; i + 2 <= depth; i += 2) {
          for (Index M = 0; M < 2 * vectorSize; M++) {
            blockA[ri + (M * 2) + 0] = lhs2(M, i + 0);
            blockA[ri + (M * 2) + 1] = lhs2(M, i + 1);
          }

          ri += 2 * 2 * vectorSize;
        }
        if (depth & 1) {
          for (Index M = 0; M < 2 * vectorSize; M++) {
            blockA[ri + M] = lhs2(M, i);
          }
          ri += 2 * vectorSize;
        }
      }

      if (PanelMode) ri += 2 * vectorSize * (stride - offset - depth);
    }
    for (; j + vectorSize <= rows; j += vectorSize) {
      const DataMapper lhs2 = lhs.getSubMapper(j, 0);
      Index i = 0;

      if (PanelMode) ri += vectorSize * offset;

      if (StorageOrder == ColMajor) {
        for (; i + 2 <= depth; i += 2) {
          PacketBlock<Packet8bf, 2> block;

          block.packet[0] = lhs2.template loadPacket<Packet8bf>(0 * vectorSize, i + 0);
          block.packet[1] = lhs2.template loadPacket<Packet8bf>(0 * vectorSize, i + 1);

          Packet8bf t0;
          t0 = vec_mergeh(block.packet[0].m_val, block.packet[1].m_val);
          block.packet[1] = vec_mergel(block.packet[0].m_val, block.packet[1].m_val);
          block.packet[0] = t0;

          storeBlock<bfloat16, Packet8bf, 2>(blockA + ri, block);

          ri += 2 * vectorSize;
        }
        if (depth & 1) {
          Packet8bf lhsV = lhs2.template loadPacket<Packet8bf>(0 * vectorSize, i + 0);
          pstore<bfloat16>(blockA + ri, lhsV);

          ri += vectorSize;
        }
      } else {
        for (; i + vectorSize <= depth; i += vectorSize) {
          PacketBlock<Packet8bf, 8> block1;

          bload<DataMapper, Packet8bf, 8, StorageOrder, false, 8>(block1, lhs2, 0 * vectorSize, i);

          Packet4ui v1[8];

          // This is transposing and interleaving data
          v1[0] = vec_mergeh(reinterpret_cast<Packet4ui>(block1.packet[0].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[1].m_val));
          v1[1] = vec_mergel(reinterpret_cast<Packet4ui>(block1.packet[0].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[1].m_val));
          v1[2] = vec_mergeh(reinterpret_cast<Packet4ui>(block1.packet[2].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[3].m_val));
          v1[3] = vec_mergel(reinterpret_cast<Packet4ui>(block1.packet[2].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[3].m_val));
          v1[4] = vec_mergeh(reinterpret_cast<Packet4ui>(block1.packet[4].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[5].m_val));
          v1[5] = vec_mergel(reinterpret_cast<Packet4ui>(block1.packet[4].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[5].m_val));
          v1[6] = vec_mergeh(reinterpret_cast<Packet4ui>(block1.packet[6].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[7].m_val));
          v1[7] = vec_mergel(reinterpret_cast<Packet4ui>(block1.packet[6].m_val),
                             reinterpret_cast<Packet4ui>(block1.packet[7].m_val));

#ifdef EIGEN_VECTORIZE_VSX
          block1.packet[0] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v1[0]), reinterpret_cast<Packet2ul>(v1[2])));
          block1.packet[2] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v1[0]), reinterpret_cast<Packet2ul>(v1[2])));
          block1.packet[4] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v1[1]), reinterpret_cast<Packet2ul>(v1[3])));
          block1.packet[6] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v1[1]), reinterpret_cast<Packet2ul>(v1[3])));
          block1.packet[1] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v1[4]), reinterpret_cast<Packet2ul>(v1[6])));
          block1.packet[3] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v1[4]), reinterpret_cast<Packet2ul>(v1[6])));
          block1.packet[5] = reinterpret_cast<Packet8us>(
              vec_mergeh(reinterpret_cast<Packet2ul>(v1[5]), reinterpret_cast<Packet2ul>(v1[7])));
          block1.packet[7] = reinterpret_cast<Packet8us>(
              vec_mergel(reinterpret_cast<Packet2ul>(v1[5]), reinterpret_cast<Packet2ul>(v1[7])));
#else
          block1.packet[0] = reinterpret_cast<Packet8us>(vec_perm(v1[0], v1[2], p16uc_TRANSPOSE64_HI));
          block1.packet[2] = reinterpret_cast<Packet8us>(vec_perm(v1[0], v1[2], p16uc_TRANSPOSE64_LO));
          block1.packet[4] = reinterpret_cast<Packet8us>(vec_perm(v1[1], v1[3], p16uc_TRANSPOSE64_HI));
          block1.packet[6] = reinterpret_cast<Packet8us>(vec_perm(v1[1], v1[3], p16uc_TRANSPOSE64_LO));
          block1.packet[1] = reinterpret_cast<Packet8us>(vec_perm(v1[4], v1[6], p16uc_TRANSPOSE64_HI));
          block1.packet[3] = reinterpret_cast<Packet8us>(vec_perm(v1[4], v1[6], p16uc_TRANSPOSE64_LO));
          block1.packet[5] = reinterpret_cast<Packet8us>(vec_perm(v1[5], v1[7], p16uc_TRANSPOSE64_HI));
          block1.packet[7] = reinterpret_cast<Packet8us>(vec_perm(v1[5], v1[7], p16uc_TRANSPOSE64_LO));
#endif

          for (Index M = 0; M < 8; M++) {
            pstore<bfloat16>(blockA + ri + (vectorSize * M), block1.packet[M]);
          }

          ri += vectorSize * vectorSize;
        }
        for (; i + 2 <= depth; i += 2) {
          for (Index M = 0; M < vectorSize; M++) {
            blockA[ri + (M * 2) + 0] = lhs2(M, i + 0);
            blockA[ri + (M * 2) + 1] = lhs2(M, i + 1);
          }

          ri += 2 * vectorSize;
        }
        if (depth & 1) {
          for (Index M = 0; M < vectorSize; M++) {
            blockA[ri + M] = lhs2(M, i);
          }

          ri += vectorSize;
        }
      }

      if (PanelMode) ri += vectorSize * (stride - offset - depth);
    }
    if (j + 4 <= rows) {
      const DataMapper lhs2 = lhs.getSubMapper(j, 0);
      Index i = 0;

      if (PanelMode) ri += 4 * offset;

      for (; i + 2 <= depth; i += 2) {
        if (StorageOrder == ColMajor) {
          PacketBlock<Packet8bf, 2> block;

          block.packet[0] = lhs2.template loadPacketPartial<Packet8bf>(0, i + 0, 4);
          block.packet[1] = lhs2.template loadPacketPartial<Packet8bf>(0, i + 1, 4);

          block.packet[0] = vec_mergeh(block.packet[0].m_val, block.packet[1].m_val);

          pstore<bfloat16>(blockA + ri, block.packet[0]);
        } else {
          blockA[ri + 0] = lhs2(0, i + 0);
          blockA[ri + 1] = lhs2(0, i + 1);
          blockA[ri + 2] = lhs2(1, i + 0);
          blockA[ri + 3] = lhs2(1, i + 1);
          blockA[ri + 4] = lhs2(2, i + 0);
          blockA[ri + 5] = lhs2(2, i + 1);
          blockA[ri + 6] = lhs2(3, i + 0);
          blockA[ri + 7] = lhs2(3, i + 1);
        }

        ri += 2 * 4;
      }
      if (depth & 1) {
        if (StorageOrder == ColMajor) {
          Packet8bf lhsV = lhs2.template loadPacketPartial<Packet8bf>(0, i + 0, 4);

          pstore_partial<bfloat16>(blockA + ri, lhsV, 4);
        } else {
          blockA[ri + 0] = lhs2(0, i);
          blockA[ri + 1] = lhs2(1, i);
          blockA[ri + 2] = lhs2(2, i);
          blockA[ri + 3] = lhs2(3, i);
        }

        ri += 4;
      }

      if (PanelMode) ri += 4 * (stride - offset - depth);
      j += 4;
    }

    if (j < rows) {
      if (PanelMode) ri += offset * (rows - j);

      Index i = 0;
      for (; i + 2 <= depth; i += 2) {
        Index k = j;
        for (; k < rows; k++) {
          blockA[ri + 0] = lhs(k, i + 0);
          blockA[ri + 1] = lhs(k, i + 1);
          ri += 2;
        }
      }
      if (depth & 1) {
        for (; j < rows; j++) {
          blockA[ri] = lhs(j, i);
          ri += 1;
        }
      }
    }
  }
};

// General template for rhs packing, bfloat16 specialization.
template <typename DataMapper, int StorageOrder, bool PanelMode>
struct dhs_pack<bfloat16, DataMapper, Packet8bf, StorageOrder, PanelMode, false> {
  EIGEN_STRONG_INLINE void operator()(bfloat16* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride,
                                      Index offset) {
    const Index vectorSize = quad_traits<bfloat16>::vectorsize;
    Index ri = 0, j = 0;

    for (; j + 4 <= cols; j += 4) {
      const DataMapper rhs2 = rhs.getSubMapper(0, j);
      Index i = 0;

      if (PanelMode) ri += 4 * offset;

      for (; i + vectorSize <= depth; i += vectorSize) {
        if (StorageOrder == ColMajor) {
          PacketBlock<Packet8bf, 4> block;

          bload<DataMapper, Packet8bf, 4, StorageOrder, false, 4>(block, rhs2, i, 0);

          Packet4ui t0, t1, t2, t3;

          t0 = vec_mergeh(reinterpret_cast<Packet4ui>(block.packet[0].m_val),
                          reinterpret_cast<Packet4ui>(block.packet[1].m_val));
          t1 = vec_mergel(reinterpret_cast<Packet4ui>(block.packet[0].m_val),
                          reinterpret_cast<Packet4ui>(block.packet[1].m_val));
          t2 = vec_mergeh(reinterpret_cast<Packet4ui>(block.packet[2].m_val),
                          reinterpret_cast<Packet4ui>(block.packet[3].m_val));
          t3 = vec_mergel(reinterpret_cast<Packet4ui>(block.packet[2].m_val),
                          reinterpret_cast<Packet4ui>(block.packet[3].m_val));

#ifdef EIGEN_VECTORIZE_VSX
          block.packet[0] =
              reinterpret_cast<Packet8us>(vec_mergeh(reinterpret_cast<Packet2ul>(t0), reinterpret_cast<Packet2ul>(t2)));
          block.packet[1] =
              reinterpret_cast<Packet8us>(vec_mergel(reinterpret_cast<Packet2ul>(t0), reinterpret_cast<Packet2ul>(t2)));
          block.packet[2] =
              reinterpret_cast<Packet8us>(vec_mergeh(reinterpret_cast<Packet2ul>(t1), reinterpret_cast<Packet2ul>(t3)));
          block.packet[3] =
              reinterpret_cast<Packet8us>(vec_mergel(reinterpret_cast<Packet2ul>(t1), reinterpret_cast<Packet2ul>(t3)));
#else
          block.packet[0] = reinterpret_cast<Packet8us>(vec_perm(t0, t2, p16uc_TRANSPOSE64_HI));
          block.packet[1] = reinterpret_cast<Packet8us>(vec_perm(t0, t2, p16uc_TRANSPOSE64_LO));
          block.packet[2] = reinterpret_cast<Packet8us>(vec_perm(t1, t3, p16uc_TRANSPOSE64_HI));
          block.packet[3] = reinterpret_cast<Packet8us>(vec_perm(t1, t3, p16uc_TRANSPOSE64_LO));
#endif

          storeBlock<bfloat16, Packet8bf, 4>(blockB + ri, block);
        } else {
          PacketBlock<Packet8bf, 8> block;

          for (int M = 0; M < 8; M++) {
            block.packet[M] = rhs2.template loadPacketPartial<Packet8bf>(i + M, 0, 4);
          }

          block.packet[0] = vec_mergeh(block.packet[0].m_val, block.packet[1].m_val);
          block.packet[1] = vec_mergeh(block.packet[2].m_val, block.packet[3].m_val);
          block.packet[2] = vec_mergeh(block.packet[4].m_val, block.packet[5].m_val);
          block.packet[3] = vec_mergeh(block.packet[6].m_val, block.packet[7].m_val);

          const Index size = 16 / sizeof(bfloat16);

          for (int M = 0; M < 4; M++) {
            pstore<bfloat16>(blockB + ri + (M * size), block.packet[M]);
          }
        }

        ri += 4 * vectorSize;
      }
      for (; i + 2 <= depth; i += 2) {
        if (StorageOrder == ColMajor) {
          blockB[ri + 0] = rhs2(i + 0, 0);
          blockB[ri + 1] = rhs2(i + 1, 0);
          blockB[ri + 2] = rhs2(i + 0, 1);
          blockB[ri + 3] = rhs2(i + 1, 1);
          blockB[ri + 4] = rhs2(i + 0, 2);
          blockB[ri + 5] = rhs2(i + 1, 2);
          blockB[ri + 6] = rhs2(i + 0, 3);
          blockB[ri + 7] = rhs2(i + 1, 3);
        } else {
          PacketBlock<Packet8bf, 2> block;

          for (int M = 0; M < 2; M++) {
            block.packet[M] = rhs2.template loadPacketPartial<Packet8bf>(i + M, 0, 4);
          }

          block.packet[0] = vec_mergeh(block.packet[0].m_val, block.packet[1].m_val);

          pstore<bfloat16>(blockB + ri, block.packet[0]);
        }

        ri += 4 * 2;
      }
      if (depth & 1) {
        blockB[ri + 0] = rhs2(i, 0);
        blockB[ri + 1] = rhs2(i, 1);
        blockB[ri + 2] = rhs2(i, 2);
        blockB[ri + 3] = rhs2(i, 3);

        ri += 4;
      }

      if (PanelMode) ri += 4 * (stride - offset - depth);
    }

    if (j < cols) {
      if (PanelMode) ri += offset * (cols - j);

      Index i = 0;
      for (; i + 2 <= depth; i += 2) {
        Index k = j;
        for (; k < cols; k++) {
          blockB[ri + 0] = rhs(i + 0, k);
          blockB[ri + 1] = rhs(i + 1, k);
          ri += 2;
        }
      }
      if (depth & 1) {
        for (; j < cols; j++) {
          blockB[ri] = rhs(i, j);
          ri += 1;
        }
      }
    }
  }
};

// General template for lhs complex packing, float64 specialization.
template <typename DataMapper, typename Packet, typename PacketC, int StorageOrder, bool Conjugate, bool PanelMode>
struct dhs_cpack<double, DataMapper, Packet, PacketC, StorageOrder, Conjugate, PanelMode, true> {
  EIGEN_ALWAYS_INLINE void dhs_ccopy(double* blockAt, const DataMapper& lhs2, Index& i, Index& rir, Index& rii,
                                     Index depth, const Index vectorSize) {
    PacketBlock<Packet, 2> blockr, blocki;
    PacketBlock<PacketC, 4> cblock;

    for (; i + vectorSize <= depth; i += vectorSize) {
      if (StorageOrder == ColMajor) {
        cblock.packet[0] = lhs2.template loadPacket<PacketC>(0, i + 0);  //[a1 a1i]
        cblock.packet[1] = lhs2.template loadPacket<PacketC>(0, i + 1);  //[b1 b1i]

        cblock.packet[2] = lhs2.template loadPacket<PacketC>(1, i + 0);  //[a2 a2i]
        cblock.packet[3] = lhs2.template loadPacket<PacketC>(1, i + 1);  //[b2 b2i]

        blockr.packet[0] = vec_mergeh(cblock.packet[0].v, cblock.packet[2].v);  //[a1 a2]
        blockr.packet[1] = vec_mergeh(cblock.packet[1].v, cblock.packet[3].v);  //[b1 b2]

        blocki.packet[0] = vec_mergel(cblock.packet[0].v, cblock.packet[2].v);
        blocki.packet[1] = vec_mergel(cblock.packet[1].v, cblock.packet[3].v);
      } else {
        cblock.packet[0] = lhs2.template loadPacket<PacketC>(0, i);  //[a1 a1i]
        cblock.packet[1] = lhs2.template loadPacket<PacketC>(1, i);  //[a2 a2i]

        cblock.packet[2] = lhs2.template loadPacket<PacketC>(0, i + 1);  //[b1 b1i]
        cblock.packet[3] = lhs2.template loadPacket<PacketC>(1, i + 1);  //[b2 b2i

        blockr.packet[0] = vec_mergeh(cblock.packet[0].v, cblock.packet[1].v);  //[a1 a2]
        blockr.packet[1] = vec_mergeh(cblock.packet[2].v, cblock.packet[3].v);  //[b1 b2]

        blocki.packet[0] = vec_mergel(cblock.packet[0].v, cblock.packet[1].v);
        blocki.packet[1] = vec_mergel(cblock.packet[2].v, cblock.packet[3].v);
      }

      if (Conjugate) {
        blocki.packet[0] = -blocki.packet[0];
        blocki.packet[1] = -blocki.packet[1];
      }

      storeBlock<double, Packet, 2>(blockAt + rir, blockr);
      storeBlock<double, Packet, 2>(blockAt + rii, blocki);

      rir += 2 * vectorSize;
      rii += 2 * vectorSize;
    }
  }

  EIGEN_STRONG_INLINE void operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows,
                                      Index stride, Index offset) {
    const Index vectorSize = quad_traits<double>::vectorsize;
    const Index vectorDelta = vectorSize * ((PanelMode) ? stride : depth);
    Index rir = ((PanelMode) ? (vectorSize * offset) : 0), rii;
    double* blockAt = reinterpret_cast<double*>(blockA);
    Index j = 0;

    for (; j + vectorSize <= rows; j += vectorSize) {
      const DataMapper lhs2 = lhs.getSubMapper(j, 0);
      Index i = 0;

      rii = rir + vectorDelta;

      dhs_ccopy(blockAt, lhs2, i, rir, rii, depth, vectorSize);

      for (; i < depth; i++) {
        PacketBlock<Packet, 1> blockr, blocki;
        PacketBlock<PacketC, 2> cblock;

        cblock.packet[0] = lhs2.template loadPacket<PacketC>(0, i);
        cblock.packet[1] = lhs2.template loadPacket<PacketC>(1, i);

        blockr.packet[0] = vec_mergeh(cblock.packet[0].v, cblock.packet[1].v);
        blocki.packet[0] = vec_mergel(cblock.packet[0].v, cblock.packet[1].v);

        if (Conjugate) {
          blocki.packet[0] = -blocki.packet[0];
        }

        pstore<double>(blockAt + rir, blockr.packet[0]);
        pstore<double>(blockAt + rii, blocki.packet[0]);

        rir += vectorSize;
        rii += vectorSize;
      }

      rir += ((PanelMode) ? (vectorSize * (2 * stride - depth)) : vectorDelta);
    }

    if (j < rows) {
      if (PanelMode) rir += (offset * (rows - j - vectorSize));
      rii = rir + (((PanelMode) ? stride : depth) * (rows - j));

      for (Index i = 0; i < depth; i++) {
        Index k = j;
        for (; k < rows; k++) {
          blockAt[rir] = lhs(k, i).real();

          if (Conjugate)
            blockAt[rii] = -lhs(k, i).imag();
          else
            blockAt[rii] = lhs(k, i).imag();

          rir += 1;
          rii += 1;
        }
      }
    }
  }
};

// General template for rhs complex packing, float64 specialization.
template <typename DataMapper, typename Packet, typename PacketC, int StorageOrder, bool Conjugate, bool PanelMode>
struct dhs_cpack<double, DataMapper, Packet, PacketC, StorageOrder, Conjugate, PanelMode, false> {
  EIGEN_ALWAYS_INLINE void dhs_ccopy(double* blockBt, const DataMapper& rhs2, Index& i, Index& rir, Index& rii,
                                     Index depth, const Index vectorSize) {
    for (; i < depth; i++) {
      PacketBlock<PacketC, 4> cblock;
      PacketBlock<Packet, 2> blockr, blocki;

      bload<DataMapper, PacketC, 2, ColMajor, false, 4>(cblock, rhs2, i, 0);

      blockr.packet[0] = vec_mergeh(cblock.packet[0].v, cblock.packet[1].v);
      blockr.packet[1] = vec_mergeh(cblock.packet[2].v, cblock.packet[3].v);

      blocki.packet[0] = vec_mergel(cblock.packet[0].v, cblock.packet[1].v);
      blocki.packet[1] = vec_mergel(cblock.packet[2].v, cblock.packet[3].v);

      if (Conjugate) {
        blocki.packet[0] = -blocki.packet[0];
        blocki.packet[1] = -blocki.packet[1];
      }

      storeBlock<double, Packet, 2>(blockBt + rir, blockr);
      storeBlock<double, Packet, 2>(blockBt + rii, blocki);

      rir += 2 * vectorSize;
      rii += 2 * vectorSize;
    }
  }

  EIGEN_STRONG_INLINE void operator()(std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols,
                                      Index stride, Index offset) {
    const Index vectorSize = quad_traits<double>::vectorsize;
    const Index vectorDelta = 2 * vectorSize * ((PanelMode) ? stride : depth);
    Index rir = ((PanelMode) ? (2 * vectorSize * offset) : 0), rii;
    double* blockBt = reinterpret_cast<double*>(blockB);
    Index j = 0;

    for (; j + 2 * vectorSize <= cols; j += 2 * vectorSize) {
      const DataMapper rhs2 = rhs.getSubMapper(0, j);
      Index i = 0;

      rii = rir + vectorDelta;

      dhs_ccopy(blockBt, rhs2, i, rir, rii, depth, vectorSize);

      rir += ((PanelMode) ? (2 * vectorSize * (2 * stride - depth)) : vectorDelta);
    }

    if (PanelMode) rir -= (offset * (2 * vectorSize - 1));

    for (; j < cols; j++) {
      const DataMapper rhs2 = rhs.getSubMapper(0, j);
      rii = rir + ((PanelMode) ? stride : depth);

      for (Index i = 0; i < depth; i++) {
        blockBt[rir] = rhs2(i, 0).real();

        if (Conjugate)
          blockBt[rii] = -rhs2(i, 0).imag();
        else
          blockBt[rii] = rhs2(i, 0).imag();

        rir += 1;
        rii += 1;
      }

      rir += ((PanelMode) ? (2 * stride - depth) : depth);
    }
  }
};

/**************
 * GEMM utils *
 **************/

// 512-bits rank1-update of acc. It can either positive or negative accumulate (useful for complex gemm).
template <typename Packet, bool NegativeAccumulate, int N>
EIGEN_ALWAYS_INLINE void pger_common(PacketBlock<Packet, N>* acc, const Packet& lhsV, const Packet* rhsV) {
  if (NegativeAccumulate) {
    for (int M = 0; M < N; M++) {
      acc->packet[M] = vec_nmsub(lhsV, rhsV[M], acc->packet[M]);
    }
  } else {
    for (int M = 0; M < N; M++) {
      acc->packet[M] = vec_madd(lhsV, rhsV[M], acc->packet[M]);
    }
  }
}

template <int N, typename Scalar, typename Packet, bool NegativeAccumulate>
EIGEN_ALWAYS_INLINE void pger(PacketBlock<Packet, N>* acc, const Scalar* lhs, const Packet* rhsV) {
  Packet lhsV = pload<Packet>(lhs);

  pger_common<Packet, NegativeAccumulate, N>(acc, lhsV, rhsV);
}

// 512-bits rank1-update of complex acc. It takes decoupled accumulators as entries. It also takes cares of mixed types
// real * complex and complex * real.
template <int N, typename Packet, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void pgerc_common(PacketBlock<Packet, N>* accReal, PacketBlock<Packet, N>* accImag,
                                      const Packet& lhsV, Packet& lhsVi, const Packet* rhsV, const Packet* rhsVi) {
  pger_common<Packet, false, N>(accReal, lhsV, rhsV);
  if (LhsIsReal) {
    pger_common<Packet, ConjugateRhs, N>(accImag, lhsV, rhsVi);
    EIGEN_UNUSED_VARIABLE(lhsVi);
  } else {
    if (!RhsIsReal) {
      pger_common<Packet, ConjugateLhs == ConjugateRhs, N>(accReal, lhsVi, rhsVi);
      pger_common<Packet, ConjugateRhs, N>(accImag, lhsV, rhsVi);
    } else {
      EIGEN_UNUSED_VARIABLE(rhsVi);
    }
    pger_common<Packet, ConjugateLhs, N>(accImag, lhsVi, rhsV);
  }
}

template <int N, typename Scalar, typename Packet, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void pgerc(PacketBlock<Packet, N>* accReal, PacketBlock<Packet, N>* accImag, const Scalar* lhs_ptr,
                               const Scalar* lhs_ptr_imag, const Packet* rhsV, const Packet* rhsVi) {
  Packet lhsV = ploadLhs<Packet>(lhs_ptr);
  Packet lhsVi;
  if (!LhsIsReal)
    lhsVi = ploadLhs<Packet>(lhs_ptr_imag);
  else
    EIGEN_UNUSED_VARIABLE(lhs_ptr_imag);

  pgerc_common<N, Packet, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal>(accReal, accImag, lhsV, lhsVi, rhsV, rhsVi);
}

template <typename Packet>
EIGEN_ALWAYS_INLINE Packet ploadLhs(const __UNPACK_TYPE__(Packet) * lhs) {
  return ploadu<Packet>(lhs);
}

// Zero the accumulator on PacketBlock.
template <typename Packet, int N>
EIGEN_ALWAYS_INLINE void bsetzero(PacketBlock<Packet, N>& acc) {
  for (int M = 0; M < N; M++) {
    acc.packet[M] = pset1<Packet>((__UNPACK_TYPE__(Packet))0);
  }
}

template <typename Packet, int N>
EIGEN_ALWAYS_INLINE void bscalec_common(PacketBlock<Packet, N>& acc, PacketBlock<Packet, N>& accZ,
                                        const Packet& pAlpha) {
  for (int M = 0; M < N; M++) {
    acc.packet[M] = vec_mul(accZ.packet[M], pAlpha);
  }
}

template <typename Packet, int N>
EIGEN_ALWAYS_INLINE void band(PacketBlock<Packet, N>& acc, const Packet& pMask) {
  for (int M = 0; M < N; M++) {
    acc.packet[M] = pand<Packet>(acc.packet[M], pMask);
  }
}

// Complex version of PacketBlock scaling.
template <typename Packet, int N, bool mask>
EIGEN_ALWAYS_INLINE void bscalec(PacketBlock<Packet, N>& aReal, PacketBlock<Packet, N>& aImag, const Packet& bReal,
                                 const Packet& bImag, PacketBlock<Packet, N>& cReal, PacketBlock<Packet, N>& cImag,
                                 const Packet& pMask) {
  if (mask && (sizeof(__UNPACK_TYPE__(Packet)) == sizeof(float))) {
    band<Packet, N>(aReal, pMask);
    band<Packet, N>(aImag, pMask);
  } else {
    EIGEN_UNUSED_VARIABLE(pMask);
  }

  bscalec_common<Packet, N>(cReal, aReal, bReal);

  bscalec_common<Packet, N>(cImag, aImag, bReal);

  pger_common<Packet, true, N>(&cReal, bImag, aImag.packet);

  pger_common<Packet, false, N>(&cImag, bImag, aReal.packet);
}

// Load a PacketBlock, the N parameters make tunning gemm easier so we can add more accumulators as needed.
//
// full = operate (load) on the entire PacketBlock or only half
template <typename DataMapper, typename Packet, const Index accCols, int StorageOrder, bool Complex, int N, bool full>
EIGEN_ALWAYS_INLINE void bload(PacketBlock<Packet, N*(Complex ? 2 : 1)>& acc, const DataMapper& res, Index row,
                               Index col) {
  if (StorageOrder == RowMajor) {
    for (int M = 0; M < N; M++) {
      acc.packet[M] = res.template loadPacket<Packet>(row + M, col);
    }
    if (Complex) {
      for (int M = 0; M < N; M++) {
        acc.packet[M + N] = res.template loadPacket<Packet>(row + M, col + accCols);
      }
    }
  } else {
    for (int M = 0; M < N; M++) {
      acc.packet[M] = res.template loadPacket<Packet>(row, col + M);
    }
    if (Complex && full) {
      for (int M = 0; M < N; M++) {
        acc.packet[M + N] = res.template loadPacket<Packet>(row + accCols, col + M);
      }
    }
  }
}

template <typename DataMapper, typename Packet, int N>
EIGEN_ALWAYS_INLINE void bstore(PacketBlock<Packet, N>& acc, const DataMapper& res, Index row) {
  for (int M = 0; M < N; M++) {
    res.template storePacket<Packet>(row, M, acc.packet[M]);
  }
}

#ifdef USE_PARTIAL_PACKETS
template <typename DataMapper, typename Packet, const Index accCols, bool Complex, Index N, bool full>
EIGEN_ALWAYS_INLINE void bload_partial(PacketBlock<Packet, N*(Complex ? 2 : 1)>& acc, const DataMapper& res, Index row,
                                       Index elements) {
  for (Index M = 0; M < N; M++) {
    acc.packet[M] = res.template loadPacketPartial<Packet>(row, M, elements);
  }
  if (Complex && full) {
    for (Index M = 0; M < N; M++) {
      acc.packet[M + N] = res.template loadPacketPartial<Packet>(row + accCols, M, elements);
    }
  }
}

template <typename DataMapper, typename Packet, Index N>
EIGEN_ALWAYS_INLINE void bstore_partial(PacketBlock<Packet, N>& acc, const DataMapper& res, Index row, Index elements) {
  for (Index M = 0; M < N; M++) {
    res.template storePacketPartial<Packet>(row, M, acc.packet[M], elements);
  }
}
#endif

#ifdef _ARCH_PWR10
#define USE_P10_AND_PVIPR2_0 (EIGEN_COMP_LLVM || (__GNUC__ >= 11))
#else
#define USE_P10_AND_PVIPR2_0 0
#endif

#if !USE_P10_AND_PVIPR2_0
const static Packet4i mask4[4] = {{0, 0, 0, 0}, {-1, 0, 0, 0}, {-1, -1, 0, 0}, {-1, -1, -1, 0}};
#endif

template <typename Packet>
EIGEN_ALWAYS_INLINE Packet bmask(const Index remaining_rows) {
#if USE_P10_AND_PVIPR2_0
#ifdef _BIG_ENDIAN
  return Packet(vec_reve(vec_genwm((1 << remaining_rows) - 1)));
#else
  return Packet(vec_genwm((1 << remaining_rows) - 1));
#endif
#else
  return Packet(mask4[remaining_rows]);
#endif
}

template <>
EIGEN_ALWAYS_INLINE Packet2d bmask<Packet2d>(const Index remaining_rows) {
#if USE_P10_AND_PVIPR2_0
  Packet2d mask2 = Packet2d(vec_gendm(remaining_rows));
#ifdef _BIG_ENDIAN
  return preverse(mask2);
#else
  return mask2;
#endif
#else
  Packet2l ret = {-remaining_rows, 0};
  return Packet2d(ret);
#endif
}

template <typename Packet, int N>
EIGEN_ALWAYS_INLINE void bscale(PacketBlock<Packet, N>& acc, PacketBlock<Packet, N>& accZ, const Packet& pAlpha) {
  for (int M = 0; M < N; M++) {
    acc.packet[M] = pmadd<Packet>(pAlpha, accZ.packet[M], acc.packet[M]);
  }
}

// Scale the PacketBlock vectors by alpha.
template <typename Packet, int N, bool mask>
EIGEN_ALWAYS_INLINE void bscale(PacketBlock<Packet, N>& acc, PacketBlock<Packet, N>& accZ, const Packet& pAlpha,
                                const Packet& pMask) {
  if (mask) {
    band<Packet, N>(accZ, pMask);
  } else {
    EIGEN_UNUSED_VARIABLE(pMask);
  }

  bscale<Packet, N>(acc, accZ, pAlpha);
}

template <typename Packet, int N, bool real>
EIGEN_ALWAYS_INLINE void pbroadcastN(const __UNPACK_TYPE__(Packet) * ap0, const __UNPACK_TYPE__(Packet) * ap1,
                                     const __UNPACK_TYPE__(Packet) * ap2, Packet& a0, Packet& a1, Packet& a2,
                                     Packet& a3) {
  a0 = pset1<Packet>(ap0[0]);
  if (N == 4) {
    a1 = pset1<Packet>(ap0[1]);
    a2 = pset1<Packet>(ap0[2]);
    a3 = pset1<Packet>(ap0[3]);
    EIGEN_UNUSED_VARIABLE(ap1);
    EIGEN_UNUSED_VARIABLE(ap2);
  } else {
    if (N > 1) {
      a1 = pset1<Packet>(ap1[0]);
    } else {
      EIGEN_UNUSED_VARIABLE(a1);
      EIGEN_UNUSED_VARIABLE(ap1);
    }
    if (N > 2) {
      a2 = pset1<Packet>(ap2[0]);
    } else {
      EIGEN_UNUSED_VARIABLE(a2);
      EIGEN_UNUSED_VARIABLE(ap2);
    }
  }
}

template <>
EIGEN_ALWAYS_INLINE void pbroadcastN<Packet4f, 4, true>(const float* ap0, const float*, const float*, Packet4f& a0,
                                                        Packet4f& a1, Packet4f& a2, Packet4f& a3) {
  pbroadcast4<Packet4f>(ap0, a0, a1, a2, a3);
}

template <>
EIGEN_ALWAYS_INLINE void pbroadcastN<Packet4f, 4, false>(const float* ap0, const float* ap1, const float* ap2,
                                                         Packet4f& a0, Packet4f& a1, Packet4f& a2, Packet4f& a3) {
  pbroadcastN<Packet4f, 4, true>(ap0, ap1, ap2, a0, a1, a2, a3);
}

template <>
EIGEN_ALWAYS_INLINE void pbroadcastN<Packet2d, 4, false>(const double* ap0, const double*, const double*, Packet2d& a0,
                                                         Packet2d& a1, Packet2d& a2, Packet2d& a3) {
  a1 = pload<Packet2d>(ap0);
  a3 = pload<Packet2d>(ap0 + 2);
  a0 = vec_splat(a1, 0);
  a1 = vec_splat(a1, 1);
  a2 = vec_splat(a3, 0);
  a3 = vec_splat(a3, 1);
}

// Grab two decouples real/imaginary PacketBlocks and return two coupled (real/imaginary pairs) PacketBlocks.
template <typename Packet, typename Packetc, int N, bool full>
EIGEN_ALWAYS_INLINE void bcouple_common(PacketBlock<Packet, N>& taccReal, PacketBlock<Packet, N>& taccImag,
                                        PacketBlock<Packetc, N>& acc1, PacketBlock<Packetc, N>& acc2) {
  for (int M = 0; M < N; M++) {
    acc1.packet[M].v = vec_mergeh(taccReal.packet[M], taccImag.packet[M]);
  }

  if (full) {
    for (int M = 0; M < N; M++) {
      acc2.packet[M].v = vec_mergel(taccReal.packet[M], taccImag.packet[M]);
    }
  }
}

template <typename Packet, typename Packetc, int N, bool full>
EIGEN_ALWAYS_INLINE void bcouple(PacketBlock<Packet, N>& taccReal, PacketBlock<Packet, N>& taccImag,
                                 PacketBlock<Packetc, N * 2>& tRes, PacketBlock<Packetc, N>& acc1,
                                 PacketBlock<Packetc, N>& acc2) {
  bcouple_common<Packet, Packetc, N, full>(taccReal, taccImag, acc1, acc2);

  for (int M = 0; M < N; M++) {
    acc1.packet[M] = padd<Packetc>(tRes.packet[M], acc1.packet[M]);
  }

  if (full) {
    for (int M = 0; M < N; M++) {
      acc2.packet[M] = padd<Packetc>(tRes.packet[M + N], acc2.packet[M]);
    }
  }
}

// PEEL loop factor.
#define PEEL 7
#define PEEL_ROW 7

#define MICRO_UNROLL(func) func(0) func(1) func(2) func(3) func(4) func(5) func(6) func(7)

#define MICRO_NORMAL_ROWS accRows == quad_traits<Scalar>::rows || accRows == 1

#define MICRO_NEW_ROWS ((MICRO_NORMAL_ROWS) ? accRows : 1)

#define MICRO_RHS(ptr, N) rhs_##ptr##N

#define MICRO_ZERO_PEEL(peel)                 \
  if ((PEEL_ROW > peel) && (peel != 0)) {     \
    bsetzero<Packet, accRows>(accZero##peel); \
  } else {                                    \
    EIGEN_UNUSED_VARIABLE(accZero##peel);     \
  }

#define MICRO_ADD(ptr, N)               \
  if (MICRO_NORMAL_ROWS) {              \
    MICRO_RHS(ptr, 0) += (accRows * N); \
  } else {                              \
    MICRO_RHS(ptr, 0) += N;             \
    MICRO_RHS(ptr, 1) += N;             \
    if (accRows == 3) {                 \
      MICRO_RHS(ptr, 2) += N;           \
    }                                   \
  }

#define MICRO_ADD_ROWS(N) MICRO_ADD(ptr, N)

#define MICRO_BROADCAST1(peel, ptr, rhsV, real)                                                                      \
  if (MICRO_NORMAL_ROWS) {                                                                                           \
    pbroadcastN<Packet, accRows, real>(MICRO_RHS(ptr, 0) + (accRows * peel), MICRO_RHS(ptr, 0), MICRO_RHS(ptr, 0),   \
                                       rhsV##peel[0], rhsV##peel[1], rhsV##peel[2], rhsV##peel[3]);                  \
  } else {                                                                                                           \
    pbroadcastN<Packet, accRows, real>(MICRO_RHS(ptr, 0) + peel, MICRO_RHS(ptr, 1) + peel, MICRO_RHS(ptr, 2) + peel, \
                                       rhsV##peel[0], rhsV##peel[1], rhsV##peel[2], rhsV##peel[3]);                  \
  }

#define MICRO_BROADCAST(peel) MICRO_BROADCAST1(peel, ptr, rhsV, true)

#define MICRO_BROADCAST_EXTRA1(ptr, rhsV, real)                                                                 \
  pbroadcastN<Packet, accRows, real>(MICRO_RHS(ptr, 0), MICRO_RHS(ptr, 1), MICRO_RHS(ptr, 2), rhsV[0], rhsV[1], \
                                     rhsV[2], rhsV[3]);

#define MICRO_BROADCAST_EXTRA             \
  Packet rhsV[4];                         \
  MICRO_BROADCAST_EXTRA1(ptr, rhsV, true) \
  MICRO_ADD_ROWS(1)

#define MICRO_SRC2(ptr, N, M)                   \
  if (MICRO_NORMAL_ROWS) {                      \
    EIGEN_UNUSED_VARIABLE(strideB);             \
    EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr, 1));   \
    EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr, 2));   \
  } else {                                      \
    MICRO_RHS(ptr, 1) = rhs_base + N + M;       \
    if (accRows == 3) {                         \
      MICRO_RHS(ptr, 2) = rhs_base + N * 2 + M; \
    } else {                                    \
      EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr, 2)); \
    }                                           \
  }

#define MICRO_SRC2_PTR MICRO_SRC2(ptr, strideB, 0)

#define MICRO_ZERO_PEEL_ROW MICRO_UNROLL(MICRO_ZERO_PEEL)

#define MICRO_WORK_PEEL(peel)                                                                            \
  if (PEEL_ROW > peel) {                                                                                 \
    MICRO_BROADCAST(peel)                                                                                \
    pger<accRows, Scalar, Packet, false>(&accZero##peel, lhs_ptr + (remaining_rows * peel), rhsV##peel); \
  } else {                                                                                               \
    EIGEN_UNUSED_VARIABLE(rhsV##peel);                                                                   \
  }

#define MICRO_WORK_PEEL_ROW                                                              \
  Packet rhsV0[4], rhsV1[4], rhsV2[4], rhsV3[4], rhsV4[4], rhsV5[4], rhsV6[4], rhsV7[4]; \
  MICRO_UNROLL(MICRO_WORK_PEEL)                                                          \
  lhs_ptr += (remaining_rows * PEEL_ROW);                                                \
  MICRO_ADD_ROWS(PEEL_ROW)

#define MICRO_ADD_PEEL(peel, sum)                        \
  if (PEEL_ROW > peel) {                                 \
    for (Index i = 0; i < accRows; i++) {                \
      accZero##sum.packet[i] += accZero##peel.packet[i]; \
    }                                                    \
  }

#define MICRO_ADD_PEEL_ROW \
  MICRO_ADD_PEEL(4, 0)     \
  MICRO_ADD_PEEL(5, 1)     \
  MICRO_ADD_PEEL(6, 2) MICRO_ADD_PEEL(7, 3) MICRO_ADD_PEEL(2, 0) MICRO_ADD_PEEL(3, 1) MICRO_ADD_PEEL(1, 0)

#define MICRO_PREFETCHN1(ptr, N)               \
  EIGEN_POWER_PREFETCH(MICRO_RHS(ptr, 0));     \
  if (N == 2 || N == 3) {                      \
    EIGEN_POWER_PREFETCH(MICRO_RHS(ptr, 1));   \
    if (N == 3) {                              \
      EIGEN_POWER_PREFETCH(MICRO_RHS(ptr, 2)); \
    }                                          \
  }

#define MICRO_PREFETCHN(N) MICRO_PREFETCHN1(ptr, N)

#define MICRO_COMPLEX_PREFETCHN(N) \
  MICRO_PREFETCHN1(ptr_real, N);   \
  if (!RhsIsReal) {                \
    MICRO_PREFETCHN1(ptr_imag, N); \
  }

template <typename Scalar, typename Packet, const Index accRows, const Index remaining_rows>
EIGEN_ALWAYS_INLINE void MICRO_EXTRA_ROW(const Scalar*& lhs_ptr, const Scalar*& rhs_ptr0, const Scalar*& rhs_ptr1,
                                         const Scalar*& rhs_ptr2, PacketBlock<Packet, accRows>& accZero) {
  MICRO_BROADCAST_EXTRA
  pger<accRows, Scalar, Packet, false>(&accZero, lhs_ptr, rhsV);
  lhs_ptr += remaining_rows;
}

template <typename Scalar, typename Packet, typename DataMapper, const Index accRows, const Index accCols,
          const Index remaining_rows>
EIGEN_ALWAYS_INLINE void gemm_unrolled_row_iteration(const DataMapper& res, const Scalar* lhs_base,
                                                     const Scalar* rhs_base, Index depth, Index strideA, Index offsetA,
                                                     Index strideB, Index row, Index rows, const Packet& pAlpha,
                                                     const Packet& pMask) {
  const Scalar *rhs_ptr0 = rhs_base, *rhs_ptr1 = NULL, *rhs_ptr2 = NULL;
  const Scalar* lhs_ptr = lhs_base + row * strideA + remaining_rows * offsetA;
  PacketBlock<Packet, accRows> accZero0, accZero1, accZero2, accZero3, accZero4, accZero5, accZero6, accZero7, acc;

  MICRO_SRC2_PTR
  bsetzero<Packet, accRows>(accZero0);

  Index remaining_depth = depth & -quad_traits<Scalar>::rows;
  Index k = 0;
  if (remaining_depth >= PEEL_ROW) {
    MICRO_ZERO_PEEL_ROW
    do {
      MICRO_PREFETCHN(accRows)
      EIGEN_POWER_PREFETCH(lhs_ptr);
      MICRO_WORK_PEEL_ROW
    } while ((k += PEEL_ROW) + PEEL_ROW <= remaining_depth);
    MICRO_ADD_PEEL_ROW
  }
  for (; k < depth; k++) {
    MICRO_EXTRA_ROW<Scalar, Packet, accRows, remaining_rows>(lhs_ptr, rhs_ptr0, rhs_ptr1, rhs_ptr2, accZero0);
  }

#ifdef USE_PARTIAL_PACKETS
  EIGEN_UNUSED_VARIABLE(rows);
  EIGEN_UNUSED_VARIABLE(pMask);
  bload_partial<DataMapper, Packet, 0, false, accRows>(acc, res, row, remaining_rows);
  bscale<Packet, accRows>(acc, accZero0, pAlpha);
  bstore_partial<DataMapper, Packet, accRows>(acc, res, row, remaining_rows);
#else
  bload<DataMapper, Packet, 0, ColMajor, false, accRows>(acc, res, row, 0);
  if ((accRows == 1) || (rows >= accCols)) {
    bscale<Packet, accRows, true>(acc, accZero0, pAlpha, pMask);
    bstore<DataMapper, Packet, accRows>(acc, res, row);
  } else {
    bscale<Packet, accRows, false>(acc, accZero0, pAlpha, pMask);
    for (Index j = 0; j < accRows; j++) {
      for (Index i = 0; i < remaining_rows; i++) {
        res(row + i, j) = acc.packet[j][i];
      }
    }
  }
#endif
}

#define MICRO_EXTRA(MICRO_EXTRA_UNROLL, value, is_col)   \
  switch (value) {                                       \
    default:                                             \
      MICRO_EXTRA_UNROLL(1)                              \
      break;                                             \
    case 2:                                              \
      if (is_col || (sizeof(Scalar) == sizeof(float))) { \
        MICRO_EXTRA_UNROLL(2)                            \
      }                                                  \
      break;                                             \
    case 3:                                              \
      if (is_col || (sizeof(Scalar) == sizeof(float))) { \
        MICRO_EXTRA_UNROLL(3)                            \
      }                                                  \
      break;                                             \
  }

#define MICRO_EXTRA_ROWS(N)                                                     \
  gemm_unrolled_row_iteration<Scalar, Packet, DataMapper, accRows, accCols, N>( \
      res, lhs_base, rhs_base, depth, strideA, offsetA, strideB, row, rows, pAlpha, pMask);

template <typename Scalar, typename Packet, typename DataMapper, const Index accRows, const Index accCols>
EIGEN_ALWAYS_INLINE void gemm_extra_row(const DataMapper& res, const Scalar* lhs_base, const Scalar* rhs_base,
                                        Index depth, Index strideA, Index offsetA, Index strideB, Index row, Index rows,
                                        Index remaining_rows, const Packet& pAlpha, const Packet& pMask) {
  MICRO_EXTRA(MICRO_EXTRA_ROWS, remaining_rows, false)
}

#define MICRO_UNROLL_WORK(func, func2, peel) \
  MICRO_UNROLL(func2);                       \
  func(0, peel) func(1, peel) func(2, peel) func(3, peel) func(4, peel) func(5, peel) func(6, peel) func(7, peel)

#define MICRO_WORK_ONE(iter, peel)                                               \
  if (unroll_factor > iter) {                                                    \
    pger_common<Packet, false, accRows>(&accZero##iter, lhsV##iter, rhsV##peel); \
  }

#define MICRO_TYPE_PEEL4(func, func2, peel)                        \
  if (PEEL > peel) {                                               \
    Packet lhsV0, lhsV1, lhsV2, lhsV3, lhsV4, lhsV5, lhsV6, lhsV7; \
    MICRO_BROADCAST(peel)                                          \
    MICRO_UNROLL_WORK(func, func2, peel)                           \
  } else {                                                         \
    EIGEN_UNUSED_VARIABLE(rhsV##peel);                             \
  }

#define MICRO_UNROLL_TYPE_PEEL(M, func, func1, func2)                                                           \
  Packet rhsV0[M], rhsV1[M], rhsV2[M], rhsV3[M], rhsV4[M], rhsV5[M], rhsV6[M], rhsV7[M];                        \
  func(func1, func2, 0) func(func1, func2, 1) func(func1, func2, 2) func(func1, func2, 3) func(func1, func2, 4) \
      func(func1, func2, 5) func(func1, func2, 6) func(func1, func2, 7)

#define MICRO_UNROLL_TYPE_ONE(M, func, func1, func2) \
  Packet rhsV0[M];                                   \
  func(func1, func2, 0)

#define MICRO_UNROLL_TYPE(MICRO_TYPE, size)                       \
  MICRO_TYPE(4, MICRO_TYPE_PEEL4, MICRO_WORK_ONE, MICRO_LOAD_ONE) \
  MICRO_ADD_ROWS(size)

#define MICRO_ONE_PEEL4 MICRO_UNROLL_TYPE(MICRO_UNROLL_TYPE_PEEL, PEEL)

#define MICRO_ONE4 MICRO_UNROLL_TYPE(MICRO_UNROLL_TYPE_ONE, 1)

#define MICRO_DST_PTR_ONE(iter)               \
  if (unroll_factor > iter) {                 \
    bsetzero<Packet, accRows>(accZero##iter); \
  } else {                                    \
    EIGEN_UNUSED_VARIABLE(accZero##iter);     \
  }

#define MICRO_DST_PTR MICRO_UNROLL(MICRO_DST_PTR_ONE)

#define MICRO_SRC_PTR MICRO_UNROLL(MICRO_SRC_PTR_ONE)

#define MICRO_PREFETCH MICRO_UNROLL(MICRO_PREFETCH_ONE)

#ifdef USE_PARTIAL_PACKETS
#define MICRO_STORE_ONE(iter)                                                                         \
  if (unroll_factor > iter) {                                                                         \
    if (MICRO_NORMAL_PARTIAL(iter)) {                                                                 \
      bload<DataMapper, Packet, 0, ColMajor, false, accRows>(acc, res, row + iter * accCols, 0);      \
      bscale<Packet, accRows>(acc, accZero##iter, pAlpha);                                            \
      bstore<DataMapper, Packet, accRows>(acc, res, row + iter * accCols);                            \
    } else {                                                                                          \
      bload_partial<DataMapper, Packet, 0, false, accRows>(acc, res, row + iter * accCols, accCols2); \
      bscale<Packet, accRows>(acc, accZero##iter, pAlpha);                                            \
      bstore_partial<DataMapper, Packet, accRows>(acc, res, row + iter * accCols, accCols2);          \
    }                                                                                                 \
  }
#else
#define MICRO_STORE_ONE(iter)                                                                  \
  if (unroll_factor > iter) {                                                                  \
    bload<DataMapper, Packet, 0, ColMajor, false, accRows>(acc, res, row + iter * accCols, 0); \
    bscale<Packet, accRows, !(MICRO_NORMAL(iter))>(acc, accZero##iter, pAlpha, pMask);         \
    bstore<DataMapper, Packet, accRows>(acc, res, row + iter * accCols);                       \
  }
#endif

#define MICRO_STORE MICRO_UNROLL(MICRO_STORE_ONE)

#ifdef USE_PARTIAL_PACKETS
template <int unroll_factor, typename Scalar, typename Packet, typename DataMapper, const Index accRows,
          const Index accCols, bool full>
#else
template <int unroll_factor, typename Scalar, typename Packet, typename DataMapper, const Index accRows,
          const Index accCols, const Index accCols2>
#endif
EIGEN_ALWAYS_INLINE void gemm_unrolled_iteration(const DataMapper& res, const Scalar* lhs_base, const Scalar* rhs_base,
                                                 Index depth, Index strideA, Index offsetA, Index strideB, Index& row,
                                                 const Packet& pAlpha,
#ifdef USE_PARTIAL_PACKETS
                                                 Index accCols2
#else
                                                 const Packet& pMask
#endif
) {
  const Scalar *rhs_ptr0 = rhs_base, *rhs_ptr1 = NULL, *rhs_ptr2 = NULL;
  const Scalar *lhs_ptr0 = NULL, *lhs_ptr1 = NULL, *lhs_ptr2 = NULL, *lhs_ptr3 = NULL, *lhs_ptr4 = NULL,
               *lhs_ptr5 = NULL, *lhs_ptr6 = NULL, *lhs_ptr7 = NULL;
  PacketBlock<Packet, accRows> accZero0, accZero1, accZero2, accZero3, accZero4, accZero5, accZero6, accZero7;
  PacketBlock<Packet, accRows> acc;

  MICRO_SRC2_PTR
  MICRO_SRC_PTR
  MICRO_DST_PTR

  Index k = 0;
  for (; k + PEEL <= depth; k += PEEL) {
    MICRO_PREFETCHN(accRows)
    MICRO_PREFETCH
    MICRO_ONE_PEEL4
  }
  for (; k < depth; k++) {
    MICRO_ONE4
  }
  MICRO_STORE

  MICRO_UPDATE
}

#ifdef USE_PARTIAL_PACKETS
#define MICRO_UNROLL_ITER2(N, M)                                                                              \
  gemm_unrolled_iteration<N + ((M) ? 1 : 0), Scalar, Packet, DataMapper, accRows, accCols, !M>(               \
      res3, lhs_base, rhs_base, depth, strideA, offsetA, strideB, row, pAlpha, M ? remaining_rows : accCols); \
  if (M) return;
#else
#define MICRO_UNROLL_ITER2(N, M)                                                                             \
  gemm_unrolled_iteration<N + ((M) ? 1 : 0), Scalar, Packet, DataMapper, accRows, accCols, M ? M : accCols>( \
      res3, lhs_base, rhs_base, depth, strideA, offsetA, strideB, row, pAlpha, pMask);                       \
  if (M) return;
#endif

template <typename Scalar, typename Packet, typename DataMapper, const Index accRows, const Index accCols>
EIGEN_ALWAYS_INLINE void gemm_cols(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index depth,
                                   Index strideA, Index offsetA, Index strideB, Index offsetB, Index col, Index rows,
                                   Index remaining_rows, const Packet& pAlpha, const Packet& pMask) {
  const DataMapper res3 = res.getSubMapper(0, col);

  const Scalar* rhs_base = blockB + col * strideB + MICRO_NEW_ROWS * offsetB;
  const Scalar* lhs_base = blockA + accCols * offsetA;
  Index row = 0;

#define MAX_UNROLL 7
  while (row + MAX_UNROLL * accCols <= rows) {
    MICRO_UNROLL_ITER2(MAX_UNROLL, 0);
  }
  switch ((rows - row) / accCols) {
#if MAX_UNROLL > 7
    case 7:
      MICRO_UNROLL_ITER(MICRO_UNROLL_ITER2, 7)
      break;
#endif
#if MAX_UNROLL > 6
    case 6:
      MICRO_UNROLL_ITER(MICRO_UNROLL_ITER2, 6)
      break;
#endif
#if MAX_UNROLL > 5
    case 5:
      MICRO_UNROLL_ITER(MICRO_UNROLL_ITER2, 5)
      break;
#endif
#if MAX_UNROLL > 4
    case 4:
      MICRO_UNROLL_ITER(MICRO_UNROLL_ITER2, 4)
      break;
#endif
#if MAX_UNROLL > 3
    case 3:
      MICRO_UNROLL_ITER(MICRO_UNROLL_ITER2, 3)
      break;
#endif
#if MAX_UNROLL > 2
    case 2:
      MICRO_UNROLL_ITER(MICRO_UNROLL_ITER2, 2)
      break;
#endif
#if MAX_UNROLL > 1
    case 1:
      MICRO_UNROLL_ITER(MICRO_UNROLL_ITER2, 1)
      break;
#endif
    default:
      break;
  }
#undef MAX_UNROLL

  if (remaining_rows > 0) {
    gemm_extra_row<Scalar, Packet, DataMapper, accRows, accCols>(res3, blockA, rhs_base, depth, strideA, offsetA,
                                                                 strideB, row, rows, remaining_rows, pAlpha, pMask);
  }
}

#define MICRO_EXTRA_COLS(N)                                                                                         \
  gemm_cols<Scalar, Packet, DataMapper, N, accCols>(res, blockA, blockB, depth, strideA, offsetA, strideB, offsetB, \
                                                    col, rows, remaining_rows, pAlpha, pMask);

template <typename Scalar, typename Packet, typename DataMapper, const Index accCols>
EIGEN_ALWAYS_INLINE void gemm_extra_cols(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index depth,
                                         Index strideA, Index offsetA, Index strideB, Index offsetB, Index col,
                                         Index rows, Index cols, Index remaining_rows, const Packet& pAlpha,
                                         const Packet& pMask) {
  MICRO_EXTRA(MICRO_EXTRA_COLS, cols - col, true)
}

/****************
 * GEMM kernels *
 * **************/
template <typename Scalar, typename Packet, typename RhsPacket, typename DataMapper, const Index accRows,
          const Index accCols>
EIGEN_STRONG_INLINE void gemm(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index rows,
                              Index depth, Index cols, Scalar alpha, Index strideA, Index strideB, Index offsetA,
                              Index offsetB) {
  const Index remaining_rows = rows % accCols;

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;

  const Packet pAlpha = pset1<Packet>(alpha);
  const Packet pMask = bmask<Packet>(remaining_rows);

  Index col = 0;
  for (; col + accRows <= cols; col += accRows) {
    gemm_cols<Scalar, Packet, DataMapper, accRows, accCols>(res, blockA, blockB, depth, strideA, offsetA, strideB,
                                                            offsetB, col, rows, remaining_rows, pAlpha, pMask);
  }

  if (col != cols) {
    gemm_extra_cols<Scalar, Packet, DataMapper, accCols>(res, blockA, blockB, depth, strideA, offsetA, strideB, offsetB,
                                                         col, rows, cols, remaining_rows, pAlpha, pMask);
  }
}

#define accColsC (accCols / 2)
#define advanceRows ((LhsIsReal) ? 1 : 2)
#define advanceCols ((RhsIsReal) ? 1 : 2)

// PEEL_COMPLEX loop factor.
#define PEEL_COMPLEX 3
#define PEEL_COMPLEX_ROW 3

#define MICRO_COMPLEX_UNROLL(func) func(0) func(1) func(2) func(3)

#define MICRO_COMPLEX_ZERO_PEEL(peel)             \
  if ((PEEL_COMPLEX_ROW > peel) && (peel != 0)) { \
    bsetzero<Packet, accRows>(accReal##peel);     \
    bsetzero<Packet, accRows>(accImag##peel);     \
  } else {                                        \
    EIGEN_UNUSED_VARIABLE(accReal##peel);         \
    EIGEN_UNUSED_VARIABLE(accImag##peel);         \
  }

#define MICRO_COMPLEX_ADD_ROWS(N, used)            \
  MICRO_ADD(ptr_real, N)                           \
  if (!RhsIsReal) {                                \
    MICRO_ADD(ptr_imag, N)                         \
  } else if (used) {                               \
    EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr_imag, 0)); \
    EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr_imag, 1)); \
    EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr_imag, 2)); \
  }

#define MICRO_COMPLEX_BROADCAST(peel)              \
  MICRO_BROADCAST1(peel, ptr_real, rhsV, false)    \
  if (!RhsIsReal) {                                \
    MICRO_BROADCAST1(peel, ptr_imag, rhsVi, false) \
  } else {                                         \
    EIGEN_UNUSED_VARIABLE(rhsVi##peel);            \
  }

#define MICRO_COMPLEX_BROADCAST_EXTRA              \
  Packet rhsV[4], rhsVi[4];                        \
  MICRO_BROADCAST_EXTRA1(ptr_real, rhsV, false)    \
  if (!RhsIsReal) {                                \
    MICRO_BROADCAST_EXTRA1(ptr_imag, rhsVi, false) \
  } else {                                         \
    EIGEN_UNUSED_VARIABLE(rhsVi);                  \
  }                                                \
  MICRO_COMPLEX_ADD_ROWS(1, true)

#define MICRO_COMPLEX_SRC2_PTR                                    \
  MICRO_SRC2(ptr_real, strideB* advanceCols, 0)                   \
  if (!RhsIsReal) {                                               \
    MICRO_RHS(ptr_imag, 0) = rhs_base + MICRO_NEW_ROWS * strideB; \
    MICRO_SRC2(ptr_imag, strideB* advanceCols, strideB)           \
  } else {                                                        \
    EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr_imag, 0));                \
    EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr_imag, 1));                \
    EIGEN_UNUSED_VARIABLE(MICRO_RHS(ptr_imag, 2));                \
  }

#define MICRO_COMPLEX_ZERO_PEEL_ROW MICRO_COMPLEX_UNROLL(MICRO_COMPLEX_ZERO_PEEL)

#define MICRO_COMPLEX_WORK_PEEL(peel)                                                 \
  if (PEEL_COMPLEX_ROW > peel) {                                                      \
    MICRO_COMPLEX_BROADCAST(peel)                                                     \
    pgerc<accRows, Scalar, Packet, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal>( \
        &accReal##peel, &accImag##peel, lhs_ptr_real + (remaining_rows * peel),       \
        lhs_ptr_imag + (remaining_rows * peel), rhsV##peel, rhsVi##peel);             \
  } else {                                                                            \
    EIGEN_UNUSED_VARIABLE(rhsV##peel);                                                \
    EIGEN_UNUSED_VARIABLE(rhsVi##peel);                                               \
  }

#define MICRO_COMPLEX_ADD_COLS(size)         \
  lhs_ptr_real += (remaining_rows * size);   \
  if (!LhsIsReal)                            \
    lhs_ptr_imag += (remaining_rows * size); \
  else                                       \
    EIGEN_UNUSED_VARIABLE(lhs_ptr_imag);

#define MICRO_COMPLEX_WORK_PEEL_ROW                  \
  Packet rhsV0[4], rhsV1[4], rhsV2[4], rhsV3[4];     \
  Packet rhsVi0[4], rhsVi1[4], rhsVi2[4], rhsVi3[4]; \
  MICRO_COMPLEX_UNROLL(MICRO_COMPLEX_WORK_PEEL)      \
  MICRO_COMPLEX_ADD_COLS(PEEL_COMPLEX_ROW)           \
  MICRO_COMPLEX_ADD_ROWS(PEEL_COMPLEX_ROW, false)

#define MICRO_COMPLEX_ADD_PEEL(peel, sum)                \
  if (PEEL_COMPLEX_ROW > peel) {                         \
    for (Index i = 0; i < accRows; i++) {                \
      accReal##sum.packet[i] += accReal##peel.packet[i]; \
      accImag##sum.packet[i] += accImag##peel.packet[i]; \
    }                                                    \
  }

#define MICRO_COMPLEX_ADD_PEEL_ROW \
  MICRO_COMPLEX_ADD_PEEL(2, 0) MICRO_COMPLEX_ADD_PEEL(3, 1) MICRO_COMPLEX_ADD_PEEL(1, 0)

template <typename Scalar, typename Packet, const Index accRows, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal,
          bool RhsIsReal, const Index remaining_rows>
EIGEN_ALWAYS_INLINE void MICRO_COMPLEX_EXTRA_ROW(const Scalar*& lhs_ptr_real, const Scalar*& lhs_ptr_imag,
                                                 const Scalar*& rhs_ptr_real0, const Scalar*& rhs_ptr_real1,
                                                 const Scalar*& rhs_ptr_real2, const Scalar*& rhs_ptr_imag0,
                                                 const Scalar*& rhs_ptr_imag1, const Scalar*& rhs_ptr_imag2,
                                                 PacketBlock<Packet, accRows>& accReal,
                                                 PacketBlock<Packet, accRows>& accImag) {
  MICRO_COMPLEX_BROADCAST_EXTRA
  pgerc<accRows, Scalar, Packet, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal>(&accReal, &accImag, lhs_ptr_real,
                                                                                   lhs_ptr_imag, rhsV, rhsVi);
  MICRO_COMPLEX_ADD_COLS(1)
}

template <typename Scalar, typename Packet, typename Packetc, typename DataMapper, const Index accRows,
          const Index accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal,
          const Index remaining_rows>
EIGEN_ALWAYS_INLINE void gemm_unrolled_complex_row_iteration(const DataMapper& res, const Scalar* lhs_base,
                                                             const Scalar* rhs_base, Index depth, Index strideA,
                                                             Index offsetA, Index strideB, Index row, Index rows,
                                                             const Packet& pAlphaReal, const Packet& pAlphaImag,
                                                             const Packet& pMask) {
  const Scalar *rhs_ptr_real0 = rhs_base, *rhs_ptr_real1 = NULL, *rhs_ptr_real2 = NULL;
  const Scalar *rhs_ptr_imag0 = NULL, *rhs_ptr_imag1 = NULL, *rhs_ptr_imag2 = NULL;
  const Scalar* lhs_ptr_real = lhs_base + advanceRows * row * strideA + remaining_rows * offsetA;
  const Scalar* lhs_ptr_imag = NULL;
  if (!LhsIsReal)
    lhs_ptr_imag = lhs_ptr_real + remaining_rows * strideA;
  else
    EIGEN_UNUSED_VARIABLE(lhs_ptr_imag);
  PacketBlock<Packet, accRows> accReal0, accImag0, accReal1, accImag1, accReal2, accImag2, accReal3, accImag3;
  PacketBlock<Packet, accRows> taccReal, taccImag;
  PacketBlock<Packetc, accRows> acc0, acc1;
  PacketBlock<Packetc, accRows * 2> tRes;

  MICRO_COMPLEX_SRC2_PTR

  bsetzero<Packet, accRows>(accReal0);
  bsetzero<Packet, accRows>(accImag0);

  Index remaining_depth = depth & -quad_traits<Scalar>::rows;
  Index k = 0;
  if (remaining_depth >= PEEL_COMPLEX_ROW) {
    MICRO_COMPLEX_ZERO_PEEL_ROW
    do {
      MICRO_COMPLEX_PREFETCHN(accRows)
      EIGEN_POWER_PREFETCH(lhs_ptr_real);
      if (!LhsIsReal) {
        EIGEN_POWER_PREFETCH(lhs_ptr_imag);
      }
      MICRO_COMPLEX_WORK_PEEL_ROW
    } while ((k += PEEL_COMPLEX_ROW) + PEEL_COMPLEX_ROW <= remaining_depth);
    MICRO_COMPLEX_ADD_PEEL_ROW
  }
  for (; k < depth; k++) {
    MICRO_COMPLEX_EXTRA_ROW<Scalar, Packet, accRows, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal, remaining_rows>(
        lhs_ptr_real, lhs_ptr_imag, rhs_ptr_real0, rhs_ptr_real1, rhs_ptr_real2, rhs_ptr_imag0, rhs_ptr_imag1,
        rhs_ptr_imag2, accReal0, accImag0);
  }

  constexpr bool full = (remaining_rows > accColsC);
  bload<DataMapper, Packetc, accColsC, ColMajor, true, accRows, full>(tRes, res, row, 0);
  if ((accRows == 1) || (rows >= accCols)) {
    bscalec<Packet, accRows, true>(accReal0, accImag0, pAlphaReal, pAlphaImag, taccReal, taccImag, pMask);
    bcouple<Packet, Packetc, accRows, full>(taccReal, taccImag, tRes, acc0, acc1);
    bstore<DataMapper, Packetc, accRows>(acc0, res, row + 0);
    if (full) {
      bstore<DataMapper, Packetc, accRows>(acc1, res, row + accColsC);
    }
  } else {
    bscalec<Packet, accRows, false>(accReal0, accImag0, pAlphaReal, pAlphaImag, taccReal, taccImag, pMask);
    bcouple<Packet, Packetc, accRows, full>(taccReal, taccImag, tRes, acc0, acc1);

    if ((sizeof(Scalar) == sizeof(float)) && (remaining_rows == 1)) {
      for (Index j = 0; j < accRows; j++) {
        res(row + 0, j) = pfirst<Packetc>(acc0.packet[j]);
      }
    } else {
      bstore<DataMapper, Packetc, accRows>(acc0, res, row + 0);
      if (full) {
        for (Index j = 0; j < accRows; j++) {
          res(row + accColsC, j) = pfirst<Packetc>(acc1.packet[j]);
        }
      }
    }
  }
}

#define MICRO_COMPLEX_EXTRA_ROWS(N)                                                                        \
  gemm_unrolled_complex_row_iteration<Scalar, Packet, Packetc, DataMapper, accRows, accCols, ConjugateLhs, \
                                      ConjugateRhs, LhsIsReal, RhsIsReal, N>(                              \
      res, lhs_base, rhs_base, depth, strideA, offsetA, strideB, row, rows, pAlphaReal, pAlphaImag, pMask);

template <typename Scalar, typename Packet, typename Packetc, typename DataMapper, const Index accRows,
          const Index accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void gemm_complex_extra_row(const DataMapper& res, const Scalar* lhs_base, const Scalar* rhs_base,
                                                Index depth, Index strideA, Index offsetA, Index strideB, Index row,
                                                Index rows, Index remaining_rows, const Packet& pAlphaReal,
                                                const Packet& pAlphaImag, const Packet& pMask) {
  MICRO_EXTRA(MICRO_COMPLEX_EXTRA_ROWS, remaining_rows, false)
}

#define MICRO_COMPLEX_UNROLL_WORK(func, func2, peel) \
  MICRO_COMPLEX_UNROLL(func2);                       \
  func(0, peel) func(1, peel) func(2, peel) func(3, peel)

#define MICRO_COMPLEX_WORK_ONE4(iter, peel)                                                \
  if (unroll_factor > iter) {                                                              \
    pgerc_common<accRows, Packet, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal>(       \
        &accReal##iter, &accImag##iter, lhsV##iter, lhsVi##iter, rhsV##peel, rhsVi##peel); \
  }

#define MICRO_COMPLEX_TYPE_PEEL4(func, func2, peel) \
  if (PEEL_COMPLEX > peel) {                        \
    Packet lhsV0, lhsV1, lhsV2, lhsV3;              \
    Packet lhsVi0, lhsVi1, lhsVi2, lhsVi3;          \
    MICRO_COMPLEX_BROADCAST(peel)                   \
    MICRO_COMPLEX_UNROLL_WORK(func, func2, peel)    \
  } else {                                          \
    EIGEN_UNUSED_VARIABLE(rhsV##peel);              \
    EIGEN_UNUSED_VARIABLE(rhsVi##peel);             \
  }

#define MICRO_COMPLEX_UNROLL_TYPE_PEEL(M, func, func1, func2) \
  Packet rhsV0[M], rhsV1[M], rhsV2[M], rhsV3[M];              \
  Packet rhsVi0[M], rhsVi1[M], rhsVi2[M], rhsVi3[M];          \
  func(func1, func2, 0) func(func1, func2, 1) func(func1, func2, 2) func(func1, func2, 3)

#define MICRO_COMPLEX_UNROLL_TYPE_ONE(M, func, func1, func2) \
  Packet rhsV0[M], rhsVi0[M];                                \
  func(func1, func2, 0)

#define MICRO_COMPLEX_UNROLL_TYPE(MICRO_COMPLEX_TYPE, size)                                        \
  MICRO_COMPLEX_TYPE(4, MICRO_COMPLEX_TYPE_PEEL4, MICRO_COMPLEX_WORK_ONE4, MICRO_COMPLEX_LOAD_ONE) \
  MICRO_COMPLEX_ADD_ROWS(size, false)

#define MICRO_COMPLEX_ONE_PEEL4 MICRO_COMPLEX_UNROLL_TYPE(MICRO_COMPLEX_UNROLL_TYPE_PEEL, PEEL_COMPLEX)

#define MICRO_COMPLEX_ONE4 MICRO_COMPLEX_UNROLL_TYPE(MICRO_COMPLEX_UNROLL_TYPE_ONE, 1)

#define MICRO_COMPLEX_DST_PTR_ONE(iter)       \
  if (unroll_factor > iter) {                 \
    bsetzero<Packet, accRows>(accReal##iter); \
    bsetzero<Packet, accRows>(accImag##iter); \
  } else {                                    \
    EIGEN_UNUSED_VARIABLE(accReal##iter);     \
    EIGEN_UNUSED_VARIABLE(accImag##iter);     \
  }

#define MICRO_COMPLEX_DST_PTR MICRO_COMPLEX_UNROLL(MICRO_COMPLEX_DST_PTR_ONE)

#define MICRO_COMPLEX_SRC_PTR MICRO_COMPLEX_UNROLL(MICRO_COMPLEX_SRC_PTR_ONE)

#define MICRO_COMPLEX_PREFETCH MICRO_COMPLEX_UNROLL(MICRO_COMPLEX_PREFETCH_ONE)

#define MICRO_COMPLEX_STORE_ONE(iter)                                                                               \
  if (unroll_factor > iter) {                                                                                       \
    constexpr bool full = ((MICRO_NORMAL(iter)) || (accCols2 > accColsC));                                          \
    bload<DataMapper, Packetc, accColsC, ColMajor, true, accRows, full>(tRes, res, row + iter * accCols, 0);        \
    bscalec<Packet, accRows, !(MICRO_NORMAL(iter))>(accReal##iter, accImag##iter, pAlphaReal, pAlphaImag, taccReal, \
                                                    taccImag, pMask);                                               \
    bcouple<Packet, Packetc, accRows, full>(taccReal, taccImag, tRes, acc0, acc1);                                  \
    bstore<DataMapper, Packetc, accRows>(acc0, res, row + iter * accCols + 0);                                      \
    if (full) {                                                                                                     \
      bstore<DataMapper, Packetc, accRows>(acc1, res, row + iter * accCols + accColsC);                             \
    }                                                                                                               \
  }

#define MICRO_COMPLEX_STORE MICRO_COMPLEX_UNROLL(MICRO_COMPLEX_STORE_ONE)

template <int unroll_factor, typename Scalar, typename Packet, typename Packetc, typename DataMapper,
          const Index accRows, const Index accCols, const Index accCols2, bool ConjugateLhs, bool ConjugateRhs,
          bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void gemm_complex_unrolled_iteration(const DataMapper& res, const Scalar* lhs_base,
                                                         const Scalar* rhs_base, Index depth, Index strideA,
                                                         Index offsetA, Index strideB, Index& row,
                                                         const Packet& pAlphaReal, const Packet& pAlphaImag,
                                                         const Packet& pMask) {
  const Scalar *rhs_ptr_real0 = rhs_base, *rhs_ptr_real1 = NULL, *rhs_ptr_real2 = NULL;
  const Scalar *rhs_ptr_imag0 = NULL, *rhs_ptr_imag1 = NULL, *rhs_ptr_imag2 = NULL;
  const Index imag_delta = accCols * strideA;
  const Index imag_delta2 = accCols2 * strideA;
  const Scalar *lhs_ptr_real0 = NULL, *lhs_ptr_real1 = NULL;
  const Scalar *lhs_ptr_real2 = NULL, *lhs_ptr_real3 = NULL;
  PacketBlock<Packet, accRows> accReal0, accImag0, accReal1, accImag1;
  PacketBlock<Packet, accRows> accReal2, accImag2, accReal3, accImag3;
  PacketBlock<Packet, accRows> taccReal, taccImag;
  PacketBlock<Packetc, accRows> acc0, acc1;
  PacketBlock<Packetc, accRows * 2> tRes;

  MICRO_COMPLEX_SRC2_PTR
  MICRO_COMPLEX_SRC_PTR
  MICRO_COMPLEX_DST_PTR

  Index k = 0;
  for (; k + PEEL_COMPLEX <= depth; k += PEEL_COMPLEX) {
    MICRO_COMPLEX_PREFETCHN(accRows)
    MICRO_COMPLEX_PREFETCH
    MICRO_COMPLEX_ONE_PEEL4
  }
  for (; k < depth; k++) {
    MICRO_COMPLEX_ONE4
  }
  MICRO_COMPLEX_STORE

  MICRO_COMPLEX_UPDATE
}

#define MICRO_COMPLEX_UNROLL_ITER2(N, M)                                                                  \
  gemm_complex_unrolled_iteration<N + (M ? 1 : 0), Scalar, Packet, Packetc, DataMapper, accRows, accCols, \
                                  M ? M : accCols, ConjugateLhs, ConjugateRhs, LhsIsReal, RhsIsReal>(     \
      res3, lhs_base, rhs_base, depth, strideA, offsetA, strideB, row, pAlphaReal, pAlphaImag, pMask);    \
  if (M) return;

template <typename Scalar, typename Packet, typename Packetc, typename DataMapper, const Index accRows,
          const Index accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void gemm_complex_cols(const DataMapper& res, const Scalar* blockA, const Scalar* blockB,
                                           Index depth, Index strideA, Index offsetA, Index strideB, Index offsetB,
                                           Index col, Index rows, Index remaining_rows, const Packet& pAlphaReal,
                                           const Packet& pAlphaImag, const Packet& pMask) {
  const DataMapper res3 = res.getSubMapper(0, col);

  const Scalar* rhs_base = blockB + advanceCols * col * strideB + MICRO_NEW_ROWS * offsetB;
  const Scalar* lhs_base = blockA + accCols * offsetA;
  Index row = 0;

#define MAX_COMPLEX_UNROLL 4
  while (row + MAX_COMPLEX_UNROLL * accCols <= rows) {
    MICRO_COMPLEX_UNROLL_ITER2(MAX_COMPLEX_UNROLL, 0);
  }
  switch ((rows - row) / accCols) {
#if MAX_COMPLEX_UNROLL > 4
    case 4:
      MICRO_COMPLEX_UNROLL_ITER(MICRO_COMPLEX_UNROLL_ITER2, 4)
      break;
#endif
#if MAX_COMPLEX_UNROLL > 3
    case 3:
      MICRO_COMPLEX_UNROLL_ITER(MICRO_COMPLEX_UNROLL_ITER2, 3)
      break;
#endif
#if MAX_COMPLEX_UNROLL > 2
    case 2:
      MICRO_COMPLEX_UNROLL_ITER(MICRO_COMPLEX_UNROLL_ITER2, 2)
      break;
#endif
#if MAX_COMPLEX_UNROLL > 1
    case 1:
      MICRO_COMPLEX_UNROLL_ITER(MICRO_COMPLEX_UNROLL_ITER2, 1)
      break;
#endif
    default:
      break;
  }
#undef MAX_COMPLEX_UNROLL

  if (remaining_rows > 0) {
    gemm_complex_extra_row<Scalar, Packet, Packetc, DataMapper, accRows, accCols, ConjugateLhs, ConjugateRhs, LhsIsReal,
                           RhsIsReal>(res3, blockA, rhs_base, depth, strideA, offsetA, strideB, row, rows,
                                      remaining_rows, pAlphaReal, pAlphaImag, pMask);
  }
}

#define MICRO_COMPLEX_EXTRA_COLS(N)                                                                         \
  gemm_complex_cols<Scalar, Packet, Packetc, DataMapper, N, accCols, ConjugateLhs, ConjugateRhs, LhsIsReal, \
                    RhsIsReal>(res, blockA, blockB, depth, strideA, offsetA, strideB, offsetB, col, rows,   \
                               remaining_rows, pAlphaReal, pAlphaImag, pMask);

template <typename Scalar, typename Packet, typename Packetc, typename DataMapper, const Index accCols,
          bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void gemm_complex_extra_cols(const DataMapper& res, const Scalar* blockA, const Scalar* blockB,
                                                 Index depth, Index strideA, Index offsetA, Index strideB,
                                                 Index offsetB, Index col, Index rows, Index cols, Index remaining_rows,
                                                 const Packet& pAlphaReal, const Packet& pAlphaImag,
                                                 const Packet& pMask) {
  MICRO_EXTRA(MICRO_COMPLEX_EXTRA_COLS, cols - col, true)
}

template <typename LhsScalar, typename RhsScalar, typename Scalarc, typename Scalar, typename Packet, typename Packetc,
          typename RhsPacket, typename DataMapper, const Index accRows, const Index accCols, bool ConjugateLhs,
          bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_STRONG_INLINE void gemm_complex(const DataMapper& res, const LhsScalar* blockAc, const RhsScalar* blockBc,
                                      Index rows, Index depth, Index cols, Scalarc alpha, Index strideA, Index strideB,
                                      Index offsetA, Index offsetB) {
  const Index remaining_rows = rows % accCols;

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;

  const Packet pAlphaReal = pset1<Packet>(alpha.real());
  const Packet pAlphaImag = pset1<Packet>(alpha.imag());
  const Packet pMask = bmask<Packet>(remaining_rows);

  const Scalar* blockA = (Scalar*)blockAc;
  const Scalar* blockB = (Scalar*)blockBc;

  Index col = 0;
  for (; col + accRows <= cols; col += accRows) {
    gemm_complex_cols<Scalar, Packet, Packetc, DataMapper, accRows, accCols, ConjugateLhs, ConjugateRhs, LhsIsReal,
                      RhsIsReal>(res, blockA, blockB, depth, strideA, offsetA, strideB, offsetB, col, rows,
                                 remaining_rows, pAlphaReal, pAlphaImag, pMask);
  }

  if (col != cols) {
    gemm_complex_extra_cols<Scalar, Packet, Packetc, DataMapper, accCols, ConjugateLhs, ConjugateRhs, LhsIsReal,
                            RhsIsReal>(res, blockA, blockB, depth, strideA, offsetA, strideB, offsetB, col, rows, cols,
                                       remaining_rows, pAlphaReal, pAlphaImag, pMask);
  }
}

#undef accColsC
#undef advanceCols
#undef advanceRows

EIGEN_ALWAYS_INLINE bool supportsMMA() {
#if defined(EIGEN_ALTIVEC_MMA_ONLY)
  return true;
#elif defined(EIGEN_ALTIVEC_MMA_DYNAMIC_DISPATCH) && defined(__BUILTIN_CPU_SUPPORTS__)
  return __builtin_cpu_supports("arch_3_1") && __builtin_cpu_supports("mma");
#else
  return false;  // No dynamic dispatch for LLVM or older GCC
#endif
}

EIGEN_ALWAYS_INLINE Packet4f loadAndMultiplyF32(Packet4f acc, const Packet4f pAlpha, float* result) {
  Packet4f result_block = ploadu<Packet4f>(result);
  return pmadd(acc, pAlpha, result_block);
}

template <bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void storeF32(float*& result, Packet4f result_block, Index rows, Index extra_rows) {
  if (lhsExtraRows) {
    pstoreu_partial(result, result_block, extra_rows);
  } else {
    pstoreu(result, result_block);
  }
  result += rows;
}

template <bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void storeResults(Packet4f (&acc)[4], Index rows, const Packet4f pAlpha, float* result,
                                      Index extra_cols, Index extra_rows) {
  Index x = 0;
  if (rhsExtraCols) {
    do {
      Packet4f result_block = loadAndMultiplyF32(acc[x], pAlpha, result);
      storeF32<lhsExtraRows>(result, result_block, rows, extra_rows);
    } while (++x < extra_cols);
  } else {
    Packet4f result_block[4];
    float* result2 = result;
    do {
      result_block[x] = loadAndMultiplyF32(acc[x], pAlpha, result);
      result += rows;
    } while (++x < 4);
    x = 0;
    do {
      storeF32<lhsExtraRows>(result2, result_block[x], rows, extra_rows);
    } while (++x < 4);
  }
}

EIGEN_ALWAYS_INLINE Packet4f oneConvertBF16Hi(Packet8us data) {
  Packet8us z = pset1<Packet8us>(0);
#ifdef _BIG_ENDIAN
  return reinterpret_cast<Packet4f>(vec_mergeh(data, z));
#else
  return reinterpret_cast<Packet4f>(vec_mergeh(z, data));
#endif
}

EIGEN_ALWAYS_INLINE Packet4f oneConvertBF16Lo(Packet8us data) {
  Packet8us z = pset1<Packet8us>(0);
#ifdef _BIG_ENDIAN
  return reinterpret_cast<Packet4f>(vec_mergel(data, z));
#else
  return reinterpret_cast<Packet4f>(vec_mergel(z, data));
#endif
}

template <Index N, Index M>
EIGEN_ALWAYS_INLINE void storeConvertTwoBF16(float* to, PacketBlock<Packet8bf, (N + 7) / 8>& block, Index extra = 0) {
  if (N < 4) {
    pstoreu_partial(to + 0, oneConvertBF16Hi(block.packet[0].m_val), extra);
  } else if (N >= (M * 8 + 4)) {
    pstoreu(to + 0, oneConvertBF16Hi(block.packet[M].m_val));
    if (N >= 8) {
      pstoreu(to + 4, oneConvertBF16Lo(block.packet[M].m_val));
    }
  }
}

template <Index N>
EIGEN_ALWAYS_INLINE void storeConvertBlockBF16(float* to, PacketBlock<Packet8bf, (N + 7) / 8>& block, Index extra) {
  storeConvertTwoBF16<N, 0>(to + 0, block, extra);
  if (N >= 16) {
    storeConvertTwoBF16<N, 1>(to + 8, block);
  }
  if (N >= 32) {
    storeConvertTwoBF16<N, 2>(to + 16, block);
    storeConvertTwoBF16<N, 3>(to + 24, block);
  }
}

template <bool non_unit_stride, Index delta>
EIGEN_ALWAYS_INLINE Packet8bf loadBF16fromResult(bfloat16* src, Index resInc) {
  if (non_unit_stride) {
    return pgather<bfloat16, Packet8bf>(src + delta * resInc, resInc);
  } else {
    return ploadu<Packet8bf>(src + delta);
  }
}

static Packet16uc p16uc_MERGE16_32_1 = {0, 1, 16, 17, 2, 3, 18, 19, 0, 1, 16, 17, 2, 3, 18, 19};
static Packet16uc p16uc_MERGE16_32_2 = {4, 5, 20, 21, 6, 7, 22, 23, 4, 5, 20, 21, 6, 7, 22, 23};
static Packet16uc p16uc_MERGE16_32_3 = {8, 9, 24, 25, 10, 11, 26, 27, 8, 9, 24, 25, 10, 11, 26, 27};
static Packet16uc p16uc_MERGE16_32_4 = {12, 13, 28, 29, 14, 15, 30, 31, 12, 13, 28, 29, 14, 15, 30, 31};

static Packet16uc p16uc_MERGE16_32_5 = {0, 1, 16, 17, 16, 17, 16, 17, 0, 1, 16, 17, 16, 17, 16, 17};
static Packet16uc p16uc_MERGE16_32_6 = {2, 3, 18, 19, 18, 19, 18, 19, 2, 3, 18, 19, 18, 19, 18, 19};
static Packet16uc p16uc_MERGE16_32_7 = {4, 5, 20, 21, 20, 21, 20, 21, 4, 5, 20, 21, 20, 21, 20, 21};
static Packet16uc p16uc_MERGE16_32_8 = {6, 7, 22, 23, 22, 23, 22, 23, 6, 7, 22, 23, 22, 23, 22, 23};

EIGEN_ALWAYS_INLINE Packet4f oneConvertBF16Perm(Packet8us data, Packet16uc mask) {
  Packet8us z = pset1<Packet8us>(0);
#ifdef _BIG_ENDIAN
  return reinterpret_cast<Packet4f>(vec_perm(data, z, mask));
#else
  return reinterpret_cast<Packet4f>(vec_perm(z, data, mask));
#endif
}

template <bool lhsExtraRows, bool odd, Index size>
EIGEN_ALWAYS_INLINE void convertArrayPointerBF16toF32DupOne(float* result, Index rows, const bfloat16* src,
                                                            Index extra_rows) {
  Packet4f dup[4 * 4];
  Packet8bf data[4];

  for (Index i = 0; i < size; i++) {
    data[i] = ploadu<Packet8bf>(src + rows * i);
  }

  for (Index i = 0, j = 0; i < size; i++, j += 4) {
    dup[j + 0] = oneConvertBF16Perm(data[i].m_val, odd ? p16uc_MERGE16_32_5 : p16uc_MERGE16_32_1);
    dup[j + 1] = oneConvertBF16Perm(data[i].m_val, odd ? p16uc_MERGE16_32_6 : p16uc_MERGE16_32_2);
    dup[j + 2] = oneConvertBF16Perm(data[i].m_val, odd ? p16uc_MERGE16_32_7 : p16uc_MERGE16_32_3);
    dup[j + 3] = oneConvertBF16Perm(data[i].m_val, odd ? p16uc_MERGE16_32_8 : p16uc_MERGE16_32_4);
  }

  for (Index j = 0; j < 4 * size; j += 4) {
    if (lhsExtraRows) {
      Packet4f z = pset1<Packet4f>(float(0));
      Index i = 0;
      do {
        pstoreu(result + (j + i) * 4, dup[j + i]);
      } while (++i < extra_rows);
      do {
        pstoreu(result + (j + i) * 4, z);
      } while (++i < 4);
    } else {
      for (Index i = 0; i < 4; i++) {
        pstoreu(result + (j + i) * 4, dup[j + i]);
      }
    }
  }
}

template <bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void convertArrayPointerBF16toF32Dup(float* result, Index cols, Index rows, const bfloat16* src,
                                                         Index delta, Index extra_rows) {
  Index col = 0;
  src += delta * 2;
  for (; col + 4 * 2 <= cols; col += 4 * 2, result += 4 * 4 * 4, src += 4 * rows) {
    convertArrayPointerBF16toF32DupOne<lhsExtraRows, false, 4>(result, rows, src, extra_rows);
  }
  for (; col + 2 <= cols; col += 2, result += 4 * 4, src += rows) {
    convertArrayPointerBF16toF32DupOne<lhsExtraRows, false, 1>(result, rows, src, extra_rows);
  }
  if (cols & 1) {
    convertArrayPointerBF16toF32DupOne<lhsExtraRows, true, 1>(result, rows, src - delta, extra_rows);
  }
}

template <const Index size, bool non_unit_stride>
EIGEN_ALWAYS_INLINE void convertPointerBF16toF32(Index& i, float* result, Index rows, bfloat16*& src, Index resInc) {
  constexpr Index extra = ((size < 4) ? 4 : size);
  while (i + size <= rows) {
    PacketBlock<Packet8bf, (size + 7) / 8> r32;
    r32.packet[0] = loadBF16fromResult<non_unit_stride, 0>(src, resInc);
    if (size >= 16) {
      r32.packet[1] = loadBF16fromResult<non_unit_stride, 8>(src, resInc);
    }
    if (size >= 32) {
      r32.packet[2] = loadBF16fromResult<non_unit_stride, 16>(src, resInc);
      r32.packet[3] = loadBF16fromResult<non_unit_stride, 24>(src, resInc);
    }
    storeConvertBlockBF16<size>(result + i, r32, rows & 3);
    i += extra;
    src += extra * resInc;
    if (size != 32) break;
  }
}

template <bool non_unit_stride>
EIGEN_ALWAYS_INLINE void convertArrayPointerBF16toF32(float* result, Index cols, Index rows, bfloat16* src,
                                                      Index resInc) {
  for (Index col = 0; col < cols; col++, src += (rows * resInc), result += rows) {
    Index i = 0;
    bfloat16* src2 = src;
    convertPointerBF16toF32<32, non_unit_stride>(i, result, rows, src2, resInc);
    convertPointerBF16toF32<16, non_unit_stride>(i, result, rows, src2, resInc);
    convertPointerBF16toF32<8, non_unit_stride>(i, result, rows, src2, resInc);
    convertPointerBF16toF32<4, non_unit_stride>(i, result, rows, src2, resInc);
    convertPointerBF16toF32<1, non_unit_stride>(i, result, rows, src2, resInc);
  }
}

template <Index num_acc, Index size = 4>
EIGEN_ALWAYS_INLINE void zeroAccumulators(Packet4f (&acc)[num_acc][size]) {
  Packet4f z = pset1<Packet4f>(float(0));

  for (Index k = 0; k < num_acc; k++) {
    for (Index j = 0; j < size; j++) {
      acc[k][j] = z;
    }
  }
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void tranposeResults(Packet4f (&acc)[num_acc][4]) {
  for (Index i = 0; i < num_acc; i++) {
    Packet4ui t0, t1, t2, t3;
    t0 = vec_mergeh(reinterpret_cast<Packet4ui>(acc[i][0]), reinterpret_cast<Packet4ui>(acc[i][2]));
    t1 = vec_mergel(reinterpret_cast<Packet4ui>(acc[i][0]), reinterpret_cast<Packet4ui>(acc[i][2]));
    t2 = vec_mergeh(reinterpret_cast<Packet4ui>(acc[i][1]), reinterpret_cast<Packet4ui>(acc[i][3]));
    t3 = vec_mergel(reinterpret_cast<Packet4ui>(acc[i][1]), reinterpret_cast<Packet4ui>(acc[i][3]));
    acc[i][0] = reinterpret_cast<Packet4f>(vec_mergeh(t0, t2));
    acc[i][1] = reinterpret_cast<Packet4f>(vec_mergel(t0, t2));
    acc[i][2] = reinterpret_cast<Packet4f>(vec_mergeh(t1, t3));
    acc[i][3] = reinterpret_cast<Packet4f>(vec_mergel(t1, t3));
  }
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void addResults(Packet4f (&acc)[num_acc][4]) {
  for (Index i = 0, j = 0; j < num_acc; i++, j += 2) {
    for (Index x = 0, y = 0; x < 2; x++, y += 2) {
      for (Index w = 0, z = 0; w < 2; w++, z += 2) {
        acc[i][y + w] = acc[j + x][z + 0] + acc[j + x][z + 1];
      }
    }
  }
}

template <Index num_acc, bool rhsExtraCols, bool lhsExtraRows, Index num_rhs>
EIGEN_ALWAYS_INLINE void outputResultsVSX(Packet4f (&acc)[num_acc][4], Index rows, const Packet4f pAlpha, float* result,
                                          const Index extra_cols, Index extra_rows) {
  tranposeResults<num_acc>(acc);
  addResults<num_acc>(acc);

  constexpr Index real_rhs = ((num_rhs / 2) - (rhsExtraCols ? 1 : 0));
  Index k = 0;
  for (Index i = 0; i < real_rhs; i++, result += 4 * rows, k++) {
    storeResults<false, lhsExtraRows>(acc[k], rows, pAlpha, result, extra_cols, extra_rows);
  }
  if (rhsExtraCols) {
    storeResults<rhsExtraCols, lhsExtraRows>(acc[k], rows, pAlpha, result, extra_cols, extra_rows);
  }
}

template <bool zero>
EIGEN_ALWAYS_INLINE void loadTwoRhsFloat32(const float* block, Index strideB, Index i, Packet4f& dhs0, Packet4f& dhs1) {
  dhs0 = ploadu<Packet4f>(block + strideB * i + 0);
  if (zero) {
    Packet4f dhs2 = pset1<Packet4f>(float(0));
    dhs1 = vec_mergel(dhs0, dhs2);
    dhs0 = vec_mergeh(dhs0, dhs2);
  } else {
    dhs1 = ploadu<Packet4f>(block + strideB * i + 4);
  }
}

template <Index num_acc, bool zero, bool rhsExtraCols, Index num_rhs>
EIGEN_ALWAYS_INLINE void KLoop(const float* indexA, const float* indexB, Packet4f (&acc)[num_acc][4], Index strideB,
                               Index k, Index offsetB, Index extra_cols) {
  constexpr Index num_lhs = 4;
  Packet4f lhs[num_lhs], rhs[num_rhs];

  constexpr Index real_rhs = (num_rhs - (rhsExtraCols ? 2 : 0));
  for (Index i = 0; i < real_rhs; i += 2) {
    loadTwoRhsFloat32<zero>(indexB + k * 4, strideB, i, rhs[i + 0], rhs[i + 1]);
  }
  if (rhsExtraCols) {
    loadTwoRhsFloat32<zero>(indexB + k * extra_cols - offsetB, strideB, real_rhs, rhs[real_rhs + 0], rhs[real_rhs + 1]);
  }

  indexA += 2 * k * 4;
  for (Index j = 0; j < num_lhs; j++) {
    lhs[j] = ploadu<Packet4f>(indexA + j * 4);
  }

  for (Index j = 0; j < num_rhs; j++) {
    for (Index i = 0; i < num_lhs; i++) {
      acc[j][i] = pmadd(rhs[j], lhs[i], acc[j][i]);
    }
  }
}

template <const Index num_acc, bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void colVSXLoopBodyIter(Index depth, Index rows, const Packet4f pAlpha, const float* indexA,
                                            const float* indexB, Index strideB, Index offsetB, float* result,
                                            const Index extra_cols, const Index extra_rows) {
  constexpr Index num_rhs = num_acc;

  Packet4f acc[num_acc][4];

  zeroAccumulators<num_acc>(acc);

  Index k;
  for (k = 0; k + 2 <= depth; k += 2) {
    KLoop<num_acc, false, rhsExtraCols, num_rhs>(indexA, indexB, acc, strideB, k, offsetB, extra_cols);
  }
  if (depth & 1) {
    KLoop<num_acc, true, rhsExtraCols, num_rhs>(indexA, indexB, acc, strideB, k, offsetB, extra_cols);
  }

  outputResultsVSX<num_acc, rhsExtraCols, lhsExtraRows, num_rhs>(acc, rows, pAlpha, result, extra_cols, extra_rows);
}

// No more than 4 (uses 2X the accumulators or 8X the number of VSX registers)
#define MAX_BFLOAT16_ACC_VSX 4

template <const Index num_acc, bool rhsExtraCols, bool lhsExtraRows>
void colVSXLoopBody(Index& col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const float* indexA,
                    const float* indexB, Index strideB, Index offsetB, float* result) {
  constexpr Index step = (num_acc * 4);  // each accumulator has 4 elements
  const Index extra_cols = (rhsExtraCols) ? (cols & 3) : 0;
  const Index extra_rows = (lhsExtraRows) ? (rows & 3) : 0;
  constexpr bool multiIters = !rhsExtraCols && (num_acc == MAX_BFLOAT16_ACC_VSX);

  do {
    colVSXLoopBodyIter<num_acc * 2, rhsExtraCols, lhsExtraRows>(depth, rows, pAlpha, indexA, indexB, strideB, offsetB,
                                                                result, extra_cols, extra_rows);

    indexB += strideB * (num_acc * 2);
    result += rows * step;
  } while (multiIters && (step <= cols - (col += step)));
}

template <const Index num_acc, bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void colVSXLoopBodyExtraN(Index col, Index depth, Index cols, Index rows, const Packet4f pAlpha,
                                              const float* indexA, const float* blockB, Index strideB, Index offsetB,
                                              float* result) {
  if (MAX_BFLOAT16_ACC_VSX > num_acc) {
    colVSXLoopBody<num_acc + (rhsExtraCols ? 1 : 0), rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA,
                                                                                 blockB, strideB, offsetB, result);
  }
}

template <bool rhsExtraCols, bool lhsExtraRows>
void colVSXLoopBodyExtra(Index col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const float* indexA,
                         const float* blockB, Index strideB, Index offsetB, float* result) {
  switch ((cols - col) >> 2) {
    case 3:
      colVSXLoopBodyExtraN<3, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB,
                                                          offsetB, result);
      break;
    case 2:
      colVSXLoopBodyExtraN<2, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB,
                                                          offsetB, result);
      break;
    case 1:
      colVSXLoopBodyExtraN<1, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB,
                                                          offsetB, result);
      break;
    default:
      if (rhsExtraCols) {
        colVSXLoopBody<1, true, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result);
      }
      break;
  }
}

template <Index size, bool lhsExtraRows = false>
EIGEN_ALWAYS_INLINE void colVSXLoops(Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA,
                                     const float* indexA2, const float* blockB2, Index strideA, Index strideB,
                                     Index offsetB, float* result2) {
  Index delta_rows = 2 * (lhsExtraRows ? (rows & 3) : size);
  for (Index row = 0; row < size; row += 4) {
    convertArrayPointerBF16toF32Dup<lhsExtraRows>(const_cast<float*>(indexA2), strideA, delta_rows, indexA, row,
                                                  rows & 3);

    const float* blockB = blockB2;
    float* result = result2 + row;

    Index col = 0;
    if (cols >= (MAX_BFLOAT16_ACC_VSX * 4)) {
      colVSXLoopBody<MAX_BFLOAT16_ACC_VSX, false, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA2, blockB,
                                                                strideB, 0, result);
      blockB += (strideB >> 1) * col;
      result += rows * col;
    }
    if (cols & 3) {
      colVSXLoopBodyExtra<true, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA2, blockB, strideB, offsetB,
                                              result);
    } else {
      colVSXLoopBodyExtra<false, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA2, blockB, strideB, 0, result);
    }
  }
}

template <Index size>
EIGEN_ALWAYS_INLINE void calcVSXColLoops(const bfloat16*& indexA, const float* indexA2, Index& row, Index depth,
                                         Index cols, Index rows, const Packet4f pAlpha, const float* indexB,
                                         Index strideA, Index strideB, Index offsetA, Index offsetB, Index bigSuffix,
                                         float* result) {
  if ((size == 16) || (rows & size)) {
    indexA += size * offsetA;
    colVSXLoops<size>(depth, cols, rows, pAlpha, indexA, indexA2, indexB, strideA, strideB, offsetB, result + row);
    row += size;
    indexA += bigSuffix * size / 16;
  }
}

template <const Index size, typename DataMapper>
EIGEN_ALWAYS_INLINE void convertBF16toF32(Index& i, float* result, Index rows, const DataMapper& src) {
  constexpr Index extra = ((size < 4) ? 4 : size);
  while (i + size <= rows) {
    PacketBlock<Packet8bf, (size + 7) / 8> r32;
    r32.packet[0] = src.template loadPacket<Packet8bf>(i + 0);
    if (size >= 16) {
      r32.packet[1] = src.template loadPacket<Packet8bf>(i + 8);
    }
    if (size >= 32) {
      r32.packet[2] = src.template loadPacket<Packet8bf>(i + 16);
      r32.packet[3] = src.template loadPacket<Packet8bf>(i + 24);
    }
    storeConvertBlockBF16<size>(result + i, r32, rows & 3);
    i += extra;
    if (size != 32) break;
  }
}

template <typename DataMapper>
EIGEN_ALWAYS_INLINE void convertArrayBF16toF32(float* result, Index cols, Index rows, const DataMapper& src) {
  typedef typename DataMapper::LinearMapper LinearMapper;
  for (Index j = 0; j < cols; j++, result += rows) {
    const LinearMapper src2 = src.getLinearMapper(0, j);
    Index i = 0;
    convertBF16toF32<32, LinearMapper>(i, result, rows, src2);
    convertBF16toF32<16, LinearMapper>(i, result, rows, src2);
    convertBF16toF32<8, LinearMapper>(i, result, rows, src2);
    convertBF16toF32<4, LinearMapper>(i, result, rows, src2);
    convertBF16toF32<1, LinearMapper>(i, result, rows, src2);
  }
}

EIGEN_ALWAYS_INLINE Packet8bf convertF32toBF16VSX(const float* res) {
  return F32ToBf16Both(ploadu<Packet4f>(res + 0), ploadu<Packet4f>(res + 4));
}

template <typename DataMapper, const Index size>
EIGEN_ALWAYS_INLINE void convertArrayF32toBF16ColVSX(float* result, Index col, Index rows, const DataMapper& res) {
  const DataMapper res2 = res.getSubMapper(0, col);
  Index row;
  float* result2 = result + col * rows;
  for (row = 0; row + 8 <= rows; row += 8, result2 += 8) {
    // get and save block
    PacketBlock<Packet8bf, size> block;
    for (Index j = 0; j < size; j++) {
      block.packet[j] = convertF32toBF16VSX(result2 + j * rows);
    }
    res2.template storePacketBlock<Packet8bf, size>(row, 0, block);
  }
  // extra rows
  if (row < rows) {
    for (Index j = 0; j < size; j++) {
      Packet8bf fp16 = convertF32toBF16VSX(result2 + j * rows);
      res2.template storePacketPartial<Packet8bf>(row, j, fp16, rows & 7);
    }
  }
}

template <typename DataMapper>
EIGEN_ALWAYS_INLINE void convertArrayF32toBF16VSX(float* result, Index cols, Index rows, const DataMapper& res) {
  Index col;
  for (col = 0; col + 4 <= cols; col += 4) {
    convertArrayF32toBF16ColVSX<DataMapper, 4>(result, col, rows, res);
  }
  // extra cols
  switch (cols - col) {
    case 1:
      convertArrayF32toBF16ColVSX<DataMapper, 1>(result, col, rows, res);
      break;
    case 2:
      convertArrayF32toBF16ColVSX<DataMapper, 2>(result, col, rows, res);
      break;
    case 3:
      convertArrayF32toBF16ColVSX<DataMapper, 3>(result, col, rows, res);
      break;
  }
}

template <typename DataMapper>
void gemmbfloat16(const DataMapper& res, const bfloat16* indexA, const bfloat16* indexB, Index rows, Index depth,
                  Index cols, bfloat16 alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  float falpha = Eigen::bfloat16_impl::bfloat16_to_float(alpha);
  const Packet4f pAlpha = pset1<Packet4f>(falpha);

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;

  ei_declare_aligned_stack_constructed_variable(float, result, cols* rows, 0);
  ei_declare_aligned_stack_constructed_variable(float, indexB2, strideB* cols, 0);
  ei_declare_aligned_stack_constructed_variable(float, indexA2, ((strideA + 1) & -2) * 4 * 2, 0);

  convertArrayBF16toF32<DataMapper>(result, cols, rows, res);
  convertArrayPointerBF16toF32(indexB2, cols, strideB, const_cast<bfloat16*>(indexB));

  Index bigSuffix = 2 * 8 * (strideA - offsetA);
  float* indexBF32 = indexB2 + 4 * offsetB;
  offsetB *= 3;
  strideB *= 2;

  Index row = 0;
  // LHS (8x16) block
  while (row + 16 <= rows) {
    calcVSXColLoops<16>(indexA, indexA2, row, depth, cols, rows, pAlpha, indexBF32, strideA, strideB, offsetA, offsetB,
                        bigSuffix, result);
  }
  // LHS (8x8) block
  calcVSXColLoops<8>(indexA, indexA2, row, depth, cols, rows, pAlpha, indexBF32, strideA, strideB, offsetA, offsetB,
                     bigSuffix, result);
  // LHS (8x4) block
  calcVSXColLoops<4>(indexA, indexA2, row, depth, cols, rows, pAlpha, indexBF32, strideA, strideB, offsetA, offsetB,
                     bigSuffix, result);
  // extra rows
  if (rows & 3) {
    // This index is the beginning of remaining block.
    colVSXLoops<4, true>(depth, cols, rows, pAlpha, indexA, indexA2, indexBF32, strideA, strideB, offsetB,
                         result + row);
  }

  // Convert back to bfloat16
  convertArrayF32toBF16VSX<DataMapper>(result, cols, rows, res);
}

#undef MAX_BFLOAT16_ACC_VSX

#include "MatrixVectorProduct.h"

/************************************
 * ppc64le template specializations *
 * **********************************/
template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<double, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode> {
  void operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<double, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>::operator()(
    double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset) {
  dhs_pack<double, DataMapper, Packet2d, ColMajor, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<double, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode> {
  void operator()(double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<double, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>::operator()(
    double* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset) {
  dhs_pack<double, DataMapper, Packet2d, RowMajor, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

#if EIGEN_ALTIVEC_USE_CUSTOM_PACK
template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<double, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode> {
  void operator()(double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<double, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>::operator()(
    double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_pack<double, DataMapper, Packet2d, ColMajor, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<double, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode> {
  void operator()(double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<double, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>::operator()(
    double* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_pack<double, DataMapper, Packet2d, RowMajor, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<bfloat16, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode> {
  void operator()(bfloat16* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<bfloat16, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>::operator()(
    bfloat16* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_pack<bfloat16, DataMapper, Packet8bf, ColMajor, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<bfloat16, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode> {
  void operator()(bfloat16* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<bfloat16, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>::operator()(
    bfloat16* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_pack<bfloat16, DataMapper, Packet8bf, RowMajor, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}
#endif

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<bfloat16, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode> {
  void operator()(bfloat16* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<bfloat16, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>::operator()(
    bfloat16* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset) {
  dhs_pack<bfloat16, DataMapper, Packet8bf, ColMajor, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<bfloat16, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode> {
  void operator()(bfloat16* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<bfloat16, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>::operator()(
    bfloat16* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset) {
  dhs_pack<bfloat16, DataMapper, Packet8bf, RowMajor, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode> {
  void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode>::operator()(
    float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset) {
  dhs_pack<float, DataMapper, Packet4f, RowMajor, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode> {
  void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>::operator()(
    float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset) {
  dhs_pack<float, DataMapper, Packet4f, ColMajor, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<std::complex<float>, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode> {
  void operator()(std::complex<float>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                  Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<std::complex<float>, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate,
                   PanelMode>::operator()(std::complex<float>* blockA, const DataMapper& lhs, Index depth, Index rows,
                                          Index stride, Index offset) {
  dhs_cpack<float, DataMapper, Packet4f, Packet2cf, RowMajor, Conjugate, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<std::complex<float>, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode> {
  void operator()(std::complex<float>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                  Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<std::complex<float>, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate,
                   PanelMode>::operator()(std::complex<float>* blockA, const DataMapper& lhs, Index depth, Index rows,
                                          Index stride, Index offset) {
  dhs_cpack<float, DataMapper, Packet4f, Packet2cf, ColMajor, Conjugate, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

#if EIGEN_ALTIVEC_USE_CUSTOM_PACK
template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode> {
  void operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>::operator()(
    float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_pack<float, DataMapper, Packet4f, ColMajor, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode> {
  void operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0, Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<float, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>::operator()(
    float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_pack<float, DataMapper, Packet4f, RowMajor, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}
#endif

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<std::complex<float>, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode> {
  void operator()(std::complex<float>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                  Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<std::complex<float>, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>::operator()(
    std::complex<float>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_cpack<float, DataMapper, Packet4f, Packet2cf, ColMajor, Conjugate, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<std::complex<float>, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode> {
  void operator()(std::complex<float>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                  Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<std::complex<float>, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>::operator()(
    std::complex<float>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_cpack<float, DataMapper, Packet4f, Packet2cf, RowMajor, Conjugate, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<std::complex<double>, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate, PanelMode> {
  void operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                  Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<std::complex<double>, Index, DataMapper, Pack1, Pack2, Packet, RowMajor, Conjugate,
                   PanelMode>::operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows,
                                          Index stride, Index offset) {
  dhs_cpack<double, DataMapper, Packet2d, Packet1cd, RowMajor, Conjugate, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<std::complex<double>, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode> {
  void operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0,
                  Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<std::complex<double>, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate,
                   PanelMode>::operator()(std::complex<double>* blockA, const DataMapper& lhs, Index depth, Index rows,
                                          Index stride, Index offset) {
  dhs_cpack<double, DataMapper, Packet2d, Packet1cd, ColMajor, Conjugate, PanelMode, true> pack;
  pack(blockA, lhs, depth, rows, stride, offset);
}

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<std::complex<double>, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode> {
  void operator()(std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                  Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<std::complex<double>, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>::operator()(
    std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_cpack<double, DataMapper, Packet2d, Packet1cd, ColMajor, Conjugate, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<std::complex<double>, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode> {
  void operator()(std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0,
                  Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<std::complex<double>, Index, DataMapper, nr, RowMajor, Conjugate, PanelMode>::operator()(
    std::complex<double>* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  dhs_cpack<double, DataMapper, Packet2d, Packet1cd, RowMajor, Conjugate, PanelMode, false> pack;
  pack(blockB, rhs, depth, cols, stride, offset);
}

// ********* gebp specializations *********
template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef typename quad_traits<float>::vectortype Packet;
  typedef typename quad_traits<float>::rhstype RhsPacket;

  void operator()(const DataMapper& res, const float* blockA, const float* blockB, Index rows, Index depth, Index cols,
                  float alpha, Index strideA = -1, Index strideB = -1, Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<float, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const float* blockA, const float* blockB, Index rows, Index depth, Index cols, float alpha,
    Index strideA, Index strideB, Index offsetA, Index offsetB) {
  const Index accRows = quad_traits<float>::rows;
  const Index accCols = quad_traits<float>::size;
  static void (*gemm_function)(const DataMapper&, const float*, const float*, Index, Index, Index, float, Index, Index,
                               Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA()) ? &Eigen::internal::gemmMMA<float, Packet, RhsPacket, DataMapper, accRows, accCols> :
#endif
                      &Eigen::internal::gemm<float, Packet, RhsPacket, DataMapper, accRows, accCols>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<float>, std::complex<float>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef Packet4f Packet;
  typedef Packet2cf Packetc;
  typedef Packet4f RhsPacket;

  void operator()(const DataMapper& res, const std::complex<float>* blockA, const std::complex<float>* blockB,
                  Index rows, Index depth, Index cols, std::complex<float> alpha, Index strideA = -1,
                  Index strideB = -1, Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<std::complex<float>, std::complex<float>, Index, DataMapper, mr, nr, ConjugateLhs,
                 ConjugateRhs>::operator()(const DataMapper& res, const std::complex<float>* blockA,
                                           const std::complex<float>* blockB, Index rows, Index depth, Index cols,
                                           std::complex<float> alpha, Index strideA, Index strideB, Index offsetA,
                                           Index offsetB) {
  const Index accRows = quad_traits<float>::rows;
  const Index accCols = quad_traits<float>::size;
  static void (*gemm_function)(const DataMapper&, const std::complex<float>*, const std::complex<float>*, Index, Index,
                               Index, std::complex<float>, Index, Index, Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA()) ? &Eigen::internal::gemm_complexMMA<std::complex<float>, std::complex<float>, std::complex<float>,
                                                          float, Packet, Packetc, RhsPacket, DataMapper, accRows,
                                                          accCols, ConjugateLhs, ConjugateRhs, false, false>
                      :
#endif
                      &Eigen::internal::gemm_complex<std::complex<float>, std::complex<float>, std::complex<float>,
                                                     float, Packet, Packetc, RhsPacket, DataMapper, accRows, accCols,
                                                     ConjugateLhs, ConjugateRhs, false, false>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, std::complex<float>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef Packet4f Packet;
  typedef Packet2cf Packetc;
  typedef Packet4f RhsPacket;

  void operator()(const DataMapper& res, const float* blockA, const std::complex<float>* blockB, Index rows,
                  Index depth, Index cols, std::complex<float> alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<float, std::complex<float>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const float* blockA, const std::complex<float>* blockB, Index rows, Index depth, Index cols,
    std::complex<float> alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  const Index accRows = quad_traits<float>::rows;
  const Index accCols = quad_traits<float>::size;
  static void (*gemm_function)(const DataMapper&, const float*, const std::complex<float>*, Index, Index, Index,
                               std::complex<float>, Index, Index, Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA()) ? &Eigen::internal::gemm_complexMMA<float, std::complex<float>, std::complex<float>, float,
                                                          Packet, Packetc, RhsPacket, DataMapper, accRows, accCols,
                                                          ConjugateLhs, ConjugateRhs, true, false>
                      :
#endif
                      &Eigen::internal::gemm_complex<float, std::complex<float>, std::complex<float>, float, Packet,
                                                     Packetc, RhsPacket, DataMapper, accRows, accCols, ConjugateLhs,
                                                     ConjugateRhs, true, false>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<float>, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef Packet4f Packet;
  typedef Packet2cf Packetc;
  typedef Packet4f RhsPacket;

  void operator()(const DataMapper& res, const std::complex<float>* blockA, const float* blockB, Index rows,
                  Index depth, Index cols, std::complex<float> alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<std::complex<float>, float, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const std::complex<float>* blockA, const float* blockB, Index rows, Index depth, Index cols,
    std::complex<float> alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  const Index accRows = quad_traits<float>::rows;
  const Index accCols = quad_traits<float>::size;
  static void (*gemm_function)(const DataMapper&, const std::complex<float>*, const float*, Index, Index, Index,
                               std::complex<float>, Index, Index, Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA()) ? &Eigen::internal::gemm_complexMMA<std::complex<float>, float, std::complex<float>, float,
                                                          Packet, Packetc, RhsPacket, DataMapper, accRows, accCols,
                                                          ConjugateLhs, ConjugateRhs, false, true>
                      :
#endif
                      &Eigen::internal::gemm_complex<std::complex<float>, float, std::complex<float>, float, Packet,
                                                     Packetc, RhsPacket, DataMapper, accRows, accCols, ConjugateLhs,
                                                     ConjugateRhs, false, true>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<double, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef typename quad_traits<double>::vectortype Packet;
  typedef typename quad_traits<double>::rhstype RhsPacket;

  void operator()(const DataMapper& res, const double* blockA, const double* blockB, Index rows, Index depth,
                  Index cols, double alpha, Index strideA = -1, Index strideB = -1, Index offsetA = 0,
                  Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<double, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const double* blockA, const double* blockB, Index rows, Index depth, Index cols,
    double alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  const Index accRows = quad_traits<double>::rows;
  const Index accCols = quad_traits<double>::size;
  static void (*gemm_function)(const DataMapper&, const double*, const double*, Index, Index, Index, double, Index,
                               Index, Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA()) ? &Eigen::internal::gemmMMA<double, Packet, RhsPacket, DataMapper, accRows, accCols> :
#endif
                      &Eigen::internal::gemm<double, Packet, RhsPacket, DataMapper, accRows, accCols>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<double>, std::complex<double>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef quad_traits<double>::vectortype Packet;
  typedef Packet1cd Packetc;
  typedef quad_traits<double>::rhstype RhsPacket;

  void operator()(const DataMapper& res, const std::complex<double>* blockA, const std::complex<double>* blockB,
                  Index rows, Index depth, Index cols, std::complex<double> alpha, Index strideA = -1,
                  Index strideB = -1, Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<std::complex<double>, std::complex<double>, Index, DataMapper, mr, nr, ConjugateLhs,
                 ConjugateRhs>::operator()(const DataMapper& res, const std::complex<double>* blockA,
                                           const std::complex<double>* blockB, Index rows, Index depth, Index cols,
                                           std::complex<double> alpha, Index strideA, Index strideB, Index offsetA,
                                           Index offsetB) {
  const Index accRows = quad_traits<double>::rows;
  const Index accCols = quad_traits<double>::size;
  static void (*gemm_function)(const DataMapper&, const std::complex<double>*, const std::complex<double>*, Index,
                               Index, Index, std::complex<double>, Index, Index, Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA())
          ? &Eigen::internal::gemm_complexMMA<std::complex<double>, std::complex<double>, std::complex<double>, double,
                                              Packet, Packetc, RhsPacket, DataMapper, accRows, accCols, ConjugateLhs,
                                              ConjugateRhs, false, false>
          :
#endif
          &Eigen::internal::gemm_complex<std::complex<double>, std::complex<double>, std::complex<double>, double,
                                         Packet, Packetc, RhsPacket, DataMapper, accRows, accCols, ConjugateLhs,
                                         ConjugateRhs, false, false>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<std::complex<double>, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef quad_traits<double>::vectortype Packet;
  typedef Packet1cd Packetc;
  typedef quad_traits<double>::rhstype RhsPacket;

  void operator()(const DataMapper& res, const std::complex<double>* blockA, const double* blockB, Index rows,
                  Index depth, Index cols, std::complex<double> alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<std::complex<double>, double, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const std::complex<double>* blockA, const double* blockB, Index rows, Index depth,
    Index cols, std::complex<double> alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  const Index accRows = quad_traits<double>::rows;
  const Index accCols = quad_traits<double>::size;
  static void (*gemm_function)(const DataMapper&, const std::complex<double>*, const double*, Index, Index, Index,
                               std::complex<double>, Index, Index, Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA()) ? &Eigen::internal::gemm_complexMMA<std::complex<double>, double, std::complex<double>, double,
                                                          Packet, Packetc, RhsPacket, DataMapper, accRows, accCols,
                                                          ConjugateLhs, ConjugateRhs, false, true>
                      :
#endif
                      &Eigen::internal::gemm_complex<std::complex<double>, double, std::complex<double>, double, Packet,
                                                     Packetc, RhsPacket, DataMapper, accRows, accCols, ConjugateLhs,
                                                     ConjugateRhs, false, true>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<double, std::complex<double>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef quad_traits<double>::vectortype Packet;
  typedef Packet1cd Packetc;
  typedef quad_traits<double>::rhstype RhsPacket;

  void operator()(const DataMapper& res, const double* blockA, const std::complex<double>* blockB, Index rows,
                  Index depth, Index cols, std::complex<double> alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<double, std::complex<double>, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const double* blockA, const std::complex<double>* blockB, Index rows, Index depth,
    Index cols, std::complex<double> alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  const Index accRows = quad_traits<double>::rows;
  const Index accCols = quad_traits<double>::size;
  static void (*gemm_function)(const DataMapper&, const double*, const std::complex<double>*, Index, Index, Index,
                               std::complex<double>, Index, Index, Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA()) ? &Eigen::internal::gemm_complexMMA<double, std::complex<double>, std::complex<double>, double,
                                                          Packet, Packetc, RhsPacket, DataMapper, accRows, accCols,
                                                          ConjugateLhs, ConjugateRhs, true, false>
                      :
#endif
                      &Eigen::internal::gemm_complex<double, std::complex<double>, std::complex<double>, double, Packet,
                                                     Packetc, RhsPacket, DataMapper, accRows, accCols, ConjugateLhs,
                                                     ConjugateRhs, true, false>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<bfloat16, bfloat16, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> {
  typedef typename quad_traits<bfloat16>::vectortype Packet;
  typedef typename quad_traits<bfloat16>::rhstype RhsPacket;

  void operator()(const DataMapper& res, const bfloat16* blockA, const bfloat16* blockB, Index rows, Index depth,
                  Index cols, bfloat16 alpha, Index strideA = -1, Index strideB = -1, Index offsetA = 0,
                  Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<bfloat16, bfloat16, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>::operator()(
    const DataMapper& res, const bfloat16* blockA, const bfloat16* blockB, Index rows, Index depth, Index cols,
    bfloat16 alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  static void (*gemm_function)(const DataMapper&, const bfloat16*, const bfloat16*, Index, Index, Index, bfloat16,
                               Index, Index, Index, Index) =
#ifdef EIGEN_MATRIX_PRODUCT_MMA_ALTIVEC_H
      (supportsMMA()) ? &Eigen::internal::gemmMMAbfloat16<DataMapper> :
#endif
                      &Eigen::internal::gemmbfloat16<DataMapper>;
  gemm_function(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
}
}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATRIX_PRODUCT_ALTIVEC_H
