#ifndef EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H
#define EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H

#if EIGEN_COMP_LLVM
#define BFLOAT16_UNROLL _Pragma("unroll 8")
#else
#define BFLOAT16_UNROLL _Pragma("GCC unroll(8)")
#endif

namespace Eigen {

namespace internal {

template <bool zero>
EIGEN_ALWAYS_INLINE Packet8bf loadBfloat16(const bfloat16* indexA) {
  Packet8bf lhs1 = ploadu<Packet8bf>(indexA);
  if (zero) {
    Packet8bf lhs2 = pset1<Packet8bf>(Eigen::bfloat16(0));
    return vec_mergeh(lhs1.m_val, lhs2.m_val);
  } else {
    return lhs1;
  }
}

template <bool zero>
EIGEN_ALWAYS_INLINE Packet8bf loadRhsBfloat16(const bfloat16* blockB, Index strideB, Index i) {
  return loadBfloat16<zero>(blockB + strideB * i);
}

template <Index num_acc, Index num_packets, bool zero, bool rhsExtraCols, bool lhsExtraRows, Index num_rhs,
          Index num_lhs>
EIGEN_ALWAYS_INLINE void KLoop(const bfloat16* indexA, const bfloat16* indexB, __vector_quad (&quad_acc)[num_acc],
                               Index strideB, Index k, Index offsetB, Index extra_cols, Index extra_rows) {
  Packet8bf lhs[num_lhs], rhs[num_rhs];

  BFLOAT16_UNROLL
  for (Index i = 0; i < (num_rhs - (rhsExtraCols ? 1 : 0)); i++) {
    rhs[i] = loadRhsBfloat16<zero>(indexB + k * 4, strideB, i);
  }
  if (rhsExtraCols) {
    rhs[num_rhs - 1] = loadRhsBfloat16<zero>(indexB + k * extra_cols - offsetB, strideB, num_rhs - 1);
  }

  indexA += k * (lhsExtraRows ? extra_rows : num_packets);
  if (num_lhs == 1) {
    lhs[0] = loadBfloat16<zero>(indexA);
  } else {
    BFLOAT16_UNROLL
    for (Index j = 0; j < num_lhs; j += 2) {
      Packet8bf lhs1 = ploadu<Packet8bf>(indexA + (j + 0) * (zero ? 4 : 8));
      if (zero) {
        Packet8bf lhs2 = pset1<Packet8bf>(Eigen::bfloat16(0));
        lhs[j + 0] = vec_mergeh(lhs1.m_val, lhs2.m_val);
        lhs[j + 1] = vec_mergel(lhs1.m_val, lhs2.m_val);
      } else {
        lhs[j + 0] = lhs1;
        lhs[j + 1] = ploadu<Packet8bf>(indexA + (j + 1) * 8);
      }
    }
  }

  BFLOAT16_UNROLL
  for (Index i = 0, x = 0; i < num_rhs; i++) {
    BFLOAT16_UNROLL
    for (Index j = 0; j < num_lhs; j++, x++) {
      __builtin_mma_xvbf16ger2pp(&(quad_acc[x]), reinterpret_cast<Packet16uc>(rhs[i].m_val),
                                 reinterpret_cast<Packet16uc>(lhs[j].m_val));
    }
  }
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void zeroAccumulators(__vector_quad (&quad_acc)[num_acc]) {
  BFLOAT16_UNROLL
  for (Index k = 0; k < num_acc; k++) __builtin_mma_xxsetaccz(&(quad_acc[k]));
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void disassembleAccumulators(__vector_quad (&quad_acc)[num_acc], Packet4f (&acc)[num_acc][4]) {
  BFLOAT16_UNROLL
  for (Index k = 0; k < num_acc; k++) __builtin_mma_disassemble_acc((void*)acc[k], &(quad_acc[k]));
}

template <Index num_acc, bool rhsExtraCols, bool lhsExtraRows, Index num_rhs, Index num_lhs>
EIGEN_ALWAYS_INLINE void outputResults(Packet4f (&acc)[num_acc][4], Index rows, const Packet4f pAlpha, float* result,
                                       const Index extra_cols, Index extra_rows) {
  BFLOAT16_UNROLL
  for (Index i = 0, k = 0; i < num_rhs - (rhsExtraCols ? 1 : 0); i++, result += 4 * rows) {
    BFLOAT16_UNROLL
    for (Index j = 0; j < num_lhs; j++, k++) {
      storeResults<false, lhsExtraRows>(acc[k], rows, pAlpha, result + j * 4, extra_cols, extra_rows);
    }
  }
  if (rhsExtraCols) {
    storeResults<rhsExtraCols, lhsExtraRows>(acc[num_acc - 1], rows, pAlpha, result, extra_cols, extra_rows);
  }
}

template <const Index num_acc, const Index num_packets, bool rhsExtraCols, bool lhsExtraRows, bool multiIter = false>
EIGEN_ALWAYS_INLINE void colLoopBodyIter(Index depth, Index rows, const Packet4f pAlpha, const bfloat16* indexA,
                                         const bfloat16* indexB, Index strideB, Index offsetB, float* result,
                                         const Index extra_cols, const Index extra_rows) {
  constexpr Index num_lhs = multiIter ? (num_packets / 4) : 1;
  constexpr Index num_rhs = (num_acc + num_lhs - 1) / num_lhs;

  for (Index offset_row = 0; offset_row < num_packets; offset_row += 4, indexA += (multiIter ? 0 : 8),
             indexB += (multiIter ? (num_rhs * strideB) : 0), result += (multiIter ? (4 * rows * num_rhs) : 4)) {
    Packet4f acc[num_acc][4];
    __vector_quad quad_acc[num_acc];

    zeroAccumulators<num_acc>(quad_acc);

    Index k;
    for (k = 0; k + 2 <= depth; k += 2) {
      KLoop<num_acc, num_packets, false, rhsExtraCols, lhsExtraRows, num_rhs, num_lhs>(
          indexA, indexB, quad_acc, strideB, k, offsetB, extra_cols, extra_rows);
    }
    if (depth & 1) {
      KLoop<num_acc, num_packets, true, rhsExtraCols, lhsExtraRows, num_rhs, num_lhs>(
          indexA - (multiIter ? 0 : offset_row), indexB, quad_acc, strideB, k, offsetB, extra_cols, extra_rows);
    }

    disassembleAccumulators<num_acc>(quad_acc, acc);

    outputResults<num_acc, rhsExtraCols, lhsExtraRows, num_rhs, num_lhs>(acc, rows, pAlpha, result, extra_cols,
                                                                         extra_rows);
  }
}

#define MAX_BFLOAT16_ACC 8

template <const Index num_acc, const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
void colLoopBody(Index& col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA,
                 const bfloat16* indexB, Index strideB, Index offsetB, float* result) {
  constexpr Index step = (num_acc * 4);  // each accumulator has 4 elements
  const Index extra_cols = (rhsExtraCols) ? (cols & 3) : 0;
  const Index extra_rows = (lhsExtraRows) ? (rows & 3) : 0;
  constexpr bool multiIters = !rhsExtraCols && (num_acc == MAX_BFLOAT16_ACC);
  constexpr bool normIters = multiIters && ((num_acc % (num_packets / 4)) == 0);

  do {
    colLoopBodyIter<num_acc, num_packets, rhsExtraCols, lhsExtraRows, normIters>(
        depth, rows, pAlpha, indexA, indexB, strideB, offsetB, result, extra_cols, extra_rows);

    indexB += strideB * num_acc;
    result += rows * step;
  } while (multiIters && (step <= cols - (col += step)));
}

template <const Index num_acc, const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void colLoopBodyExtraN(Index col, Index depth, Index cols, Index rows, const Packet4f pAlpha,
                                           const bfloat16* indexA, const bfloat16* blockB, Index strideB, Index offsetB,
                                           float* result) {
  if (MAX_BFLOAT16_ACC > num_acc) {
    colLoopBody<num_acc + (rhsExtraCols ? 1 : 0), num_packets, rhsExtraCols, lhsExtraRows>(
        col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB, result);
  }
}

template <const Index num_packets, bool rhsExtraCols, bool lhsExtraRows>
void colLoopBodyExtra(Index col, Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA,
                      const bfloat16* blockB, Index strideB, Index offsetB, float* result) {
  switch ((cols - col) >> 2) {
    case 7:
      colLoopBodyExtraN<7, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB,
                                                                    strideB, offsetB, result);
      break;
    case 6:
      colLoopBodyExtraN<6, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB,
                                                                    strideB, offsetB, result);
      break;
    case 5:
      colLoopBodyExtraN<5, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB,
                                                                    strideB, offsetB, result);
      break;
    case 4:
      colLoopBodyExtraN<4, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB,
                                                                    strideB, offsetB, result);
      break;
    case 3:
      colLoopBodyExtraN<3, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB,
                                                                    strideB, offsetB, result);
      break;
    case 2:
      colLoopBodyExtraN<2, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB,
                                                                    strideB, offsetB, result);
      break;
    case 1:
      colLoopBodyExtraN<1, num_packets, rhsExtraCols, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB,
                                                                    strideB, offsetB, result);
      break;
    default:
      if (rhsExtraCols) {
        colLoopBody<1, num_packets, true, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB,
                                                        offsetB, result);
      }
      break;
  }
}

template <const Index num_packets, bool lhsExtraRows = false>
EIGEN_ALWAYS_INLINE void colLoops(Index depth, Index cols, Index rows, const Packet4f pAlpha, const bfloat16* indexA,
                                  const bfloat16* blockB, Index strideB, Index offsetB, float* result) {
  Index col = 0;
  if (cols >= (MAX_BFLOAT16_ACC * 4)) {
    colLoopBody<MAX_BFLOAT16_ACC, num_packets, false, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB,
                                                                    strideB, 0, result);
    blockB += (strideB >> 2) * col;
    result += rows * col;
  }
  if (cols & 3) {
    colLoopBodyExtra<num_packets, true, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, offsetB,
                                                      result);
  } else {
    colLoopBodyExtra<num_packets, false, lhsExtraRows>(col, depth, cols, rows, pAlpha, indexA, blockB, strideB, 0,
                                                       result);
  }
}

EIGEN_ALWAYS_INLINE Packet8bf convertF32toBF16(const float* res) {
  Packet16uc fp16[2];
  __vector_pair fp16_vp = *reinterpret_cast<__vector_pair*>(const_cast<float*>(res));
  __builtin_vsx_disassemble_pair(reinterpret_cast<void*>(fp16), &fp16_vp);
  fp16[0] = __builtin_vsx_xvcvspbf16(fp16[0]);
  fp16[1] = __builtin_vsx_xvcvspbf16(fp16[1]);
  return vec_pack(reinterpret_cast<Packet4ui>(fp16[0]), reinterpret_cast<Packet4ui>(fp16[1]));
}

template <typename DataMapper, const Index size>
EIGEN_ALWAYS_INLINE void convertArrayF32toBF16Col(float* result, Index col, Index rows, const DataMapper& res) {
  const DataMapper res2 = res.getSubMapper(0, col);
  Index row;
  float* result2 = result + col * rows;
  for (row = 0; row + 8 <= rows; row += 8, result2 += 8) {
    // get and save block
    PacketBlock<Packet8bf, size> block;
    BFLOAT16_UNROLL
    for (Index j = 0; j < size; j++) {
      block.packet[j] = convertF32toBF16(result2 + j * rows);
    }
    res2.template storePacketBlock<Packet8bf, size>(row, 0, block);
  }
  // extra rows
  if (row < rows) {
    BFLOAT16_UNROLL
    for (Index j = 0; j < size; j++) {
      Packet8bf fp16 = convertF32toBF16(result2 + j * rows);
      res2.template storePacketPartial<Packet8bf>(row, j, fp16, rows & 7);
    }
  }
}

template <const Index size, bool non_unit_stride = false>
EIGEN_ALWAYS_INLINE void convertPointerF32toBF16(Index& i, float* result, Index rows, bfloat16*& dst,
                                                 Index resInc = 1) {
  constexpr Index extra = ((size < 8) ? 8 : size);
  while (i + size <= rows) {
    PacketBlock<Packet8bf, (size + 7) / 8> r32;
    r32.packet[0] = convertF32toBF16(result + i + 0);
    if (size >= 16) {
      r32.packet[1] = convertF32toBF16(result + i + 8);
    }
    if (size >= 32) {
      r32.packet[2] = convertF32toBF16(result + i + 16);
      r32.packet[3] = convertF32toBF16(result + i + 24);
    }
    storeBF16fromResult<size, non_unit_stride, 0>(dst, r32.packet[0], resInc, rows & 7);
    if (size >= 16) {
      storeBF16fromResult<size, non_unit_stride, 8>(dst, r32.packet[1], resInc);
    }
    if (size >= 32) {
      storeBF16fromResult<size, non_unit_stride, 16>(dst, r32.packet[2], resInc);
      storeBF16fromResult<size, non_unit_stride, 24>(dst, r32.packet[3], resInc);
    }
    i += extra;
    dst += extra * resInc;
    if (size != 32) break;
  }
}

template <bool non_unit_stride = false>
EIGEN_ALWAYS_INLINE void convertArrayPointerF32toBF16(float* result, Index rows, bfloat16* dst, Index resInc = 1) {
  Index i = 0;
  convertPointerF32toBF16<32, non_unit_stride>(i, result, rows, dst, resInc);
  convertPointerF32toBF16<16, non_unit_stride>(i, result, rows, dst, resInc);
  convertPointerF32toBF16<8, non_unit_stride>(i, result, rows, dst, resInc);
  convertPointerF32toBF16<1, non_unit_stride>(i, result, rows, dst, resInc);
}

template <typename DataMapper>
EIGEN_ALWAYS_INLINE void convertArrayF32toBF16(float* result, Index cols, Index rows, const DataMapper& res) {
  Index col;
  for (col = 0; col + 4 <= cols; col += 4) {
    convertArrayF32toBF16Col<DataMapper, 4>(result, col, rows, res);
  }
  // extra cols
  switch (cols - col) {
    case 1:
      convertArrayF32toBF16Col<DataMapper, 1>(result, col, rows, res);
      break;
    case 2:
      convertArrayF32toBF16Col<DataMapper, 2>(result, col, rows, res);
      break;
    case 3:
      convertArrayF32toBF16Col<DataMapper, 3>(result, col, rows, res);
      break;
  }
}

template <Index size>
EIGEN_ALWAYS_INLINE void calcColLoops(const bfloat16*& indexA, Index& row, Index depth, Index cols, Index rows,
                                      const Packet4f pAlpha, const bfloat16* indexB, Index strideB, Index offsetA,
                                      Index offsetB, Index bigSuffix, float* result) {
  if ((size == 16) || (rows & size)) {
    indexA += size * offsetA;
    colLoops<size>(depth, cols, rows, pAlpha, indexA, indexB, strideB, offsetB, result + row);
    row += size;
    indexA += bigSuffix * size / 16;
  }
}

template <typename DataMapper>
void gemmMMAbfloat16(const DataMapper& res, const bfloat16* indexA, const bfloat16* indexB, Index rows, Index depth,
                     Index cols, bfloat16 alpha, Index strideA, Index strideB, Index offsetA, Index offsetB) {
  float falpha = Eigen::bfloat16_impl::bfloat16_to_float(alpha);
  const Packet4f pAlpha = pset1<Packet4f>(falpha);
  ei_declare_aligned_stack_constructed_variable(float, result, cols* rows, 0);

  convertArrayBF16toF32<DataMapper>(result, cols, rows, res);

  if (strideA == -1) strideA = depth;
  if (strideB == -1) strideB = depth;
  // Packing is done in blocks.
  // There's 4 possible sizes of blocks
  // Blocks of 8 columns with 16 elements (8x16)
  // Blocks of 8 columns with 8 elements (8x8). This happens when there's 16 > rows >= 8
  // Blocks of 8 columns with 4 elements (8x4). This happens when there's 8 > rows >= 4
  // Blocks of 8 columns with < 4 elements. This happens when there's less than 4 remaining rows

  // Loop for LHS standard block (8x16)
  Index bigSuffix = (2 * 8) * (strideA - offsetA);
  indexB += 4 * offsetB;
  strideB *= 4;
  offsetB *= 3;

  Index row = 0;
  while (row + 16 <= rows) {
    calcColLoops<16>(indexA, row, depth, cols, rows, pAlpha, indexB, strideB, offsetA, offsetB, bigSuffix, result);
  }
  // LHS (8x8) block
  calcColLoops<8>(indexA, row, depth, cols, rows, pAlpha, indexB, strideB, offsetA, offsetB, bigSuffix, result);
  // LHS (8x4) block
  calcColLoops<4>(indexA, row, depth, cols, rows, pAlpha, indexB, strideB, offsetA, offsetB, bigSuffix, result);
  // extra rows
  if (rows & 3) {
    // This index is the beginning of remaining block.
    colLoops<4, true>(depth, cols, rows, pAlpha, indexA, indexB, strideB, offsetB, result + row);
  }

  // Convert back to bfloat16
  convertArrayF32toBF16<DataMapper>(result, cols, rows, res);
}

#undef MAX_BFLOAT16_ACC

#if !EIGEN_ALTIVEC_DISABLE_MMA
template <Index num_acc, typename LhsMapper, bool zero>
EIGEN_ALWAYS_INLINE void loadVecLoop(Index k, LhsMapper& lhs, Packet8bf (&a0)[num_acc], Packet8bf b1) {
  a0[k + 0] = lhs.template loadPacket<Packet8bf>(k * 4, 0);
  if (!zero) {
    b1 = lhs.template loadPacket<Packet8bf>(k * 4, 1);
  }
  if (num_acc > (k + 1)) {
    a0[k + 1] = vec_mergel(a0[k + 0].m_val, b1.m_val);
  }
  a0[k + 0] = vec_mergeh(a0[k + 0].m_val, b1.m_val);
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void multVec(__vector_quad (&quad_acc)[num_acc], Packet8bf (&a0)[num_acc], Packet8bf b0) {
  BFLOAT16_UNROLL
  for (Index k = 0; k < num_acc; k++) {
    __builtin_mma_xvbf16ger2pp(&(quad_acc[k]), reinterpret_cast<Packet16uc>(b0.m_val),
                               reinterpret_cast<Packet16uc>(a0[k].m_val));
  }
}

template <Index num_acc, typename LhsMapper, typename RhsMapper, bool zero, bool linear>
EIGEN_ALWAYS_INLINE void vecColLoop(Index j, LhsMapper& lhs, RhsMapper& rhs, __vector_quad (&quad_acc)[num_acc]) {
  Packet8bf a0[num_acc];
  Packet8bf b1 = pset1<Packet8bf>(Eigen::bfloat16(0));
  Packet8bf b0 = loadColData<RhsMapper, linear>(rhs, j);

  if (zero) {
    b0 = vec_mergeh(b0.m_val, b1.m_val);
  }

  using LhsSubMapper = typename LhsMapper::SubMapper;

  LhsSubMapper lhs2 = lhs.getSubMapper(0, j);
  BFLOAT16_UNROLL
  for (Index k = 0; k < num_acc; k += 2) {
    loadVecLoop<num_acc, LhsSubMapper, zero>(k, lhs2, a0, b1);
  }

  multVec<num_acc>(quad_acc, a0, b0);
}

#define MAX_BFLOAT16_VEC_ACC 8

template <const Index num_acc, typename LhsMapper, typename RhsMapper, bool extraRows, bool linear>
void colVecColLoopBody(Index& row, Index cend, Index rows, LhsMapper& lhs, RhsMapper& rhs, const Packet4f pAlpha,
                       float* result) {
  constexpr Index step = (num_acc * 4);
  const Index extra_rows = (extraRows) ? (rows & 3) : 0;
  constexpr bool multiIters = !extraRows && (num_acc == MAX_BFLOAT16_VEC_ACC);

  do {
    Packet4f acc[num_acc][4];
    __vector_quad quad_acc[num_acc];

    zeroAccumulators<num_acc>(quad_acc);

    using LhsSubMapper = typename LhsMapper::SubMapper;

    LhsSubMapper lhs2 = lhs.getSubMapper(row, 0);
    for (Index j = 0; j + 2 <= cend; j += 2) {
      vecColLoop<num_acc, LhsSubMapper, RhsMapper, false, linear>(j, lhs2, rhs, quad_acc);
    }
    if (cend & 1) {
      vecColLoop<num_acc, LhsSubMapper, RhsMapper, true, linear>(cend - 1, lhs2, rhs, quad_acc);
    }

    disassembleAccumulators<num_acc>(quad_acc, acc);

    outputVecColResults<num_acc, extraRows>(acc, result, pAlpha, extra_rows);

    result += step;
  } while (multiIters && (step <= rows - (row += step)));
}

template <const Index num_acc, typename LhsMapper, typename RhsMapper, bool extraRows, bool linear>
EIGEN_ALWAYS_INLINE void colVecColLoopBodyExtraN(Index& row, Index cend, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                                 const Packet4f pAlpha, float* result) {
  if (MAX_BFLOAT16_VEC_ACC > num_acc) {
    colVecColLoopBody<num_acc + (extraRows ? 1 : 0), LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs,
                                                                                              pAlpha, result);
  }
}

template <typename LhsMapper, typename RhsMapper, bool extraRows, bool linear>
EIGEN_ALWAYS_INLINE void colVecColLoopBodyExtra(Index& row, Index cend, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                                const Packet4f pAlpha, float* result) {
  switch ((rows - row) >> 2) {
    case 7:
      colVecColLoopBodyExtraN<7, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 6:
      colVecColLoopBodyExtraN<6, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 5:
      colVecColLoopBodyExtraN<5, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 4:
      colVecColLoopBodyExtraN<4, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 3:
      colVecColLoopBodyExtraN<3, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 2:
      colVecColLoopBodyExtraN<2, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    case 1:
      colVecColLoopBodyExtraN<1, LhsMapper, RhsMapper, extraRows, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      break;
    default:
      if (extraRows) {
        colVecColLoopBody<1, LhsMapper, RhsMapper, true, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
      }
      break;
  }
}

template <typename LhsMapper, typename RhsMapper, bool linear>
EIGEN_ALWAYS_INLINE void calcVecColLoops(Index cend, Index rows, LhsMapper& lhs, RhsMapper& rhs, const Packet4f pAlpha,
                                         float* result) {
  Index row = 0;
  if (rows >= (MAX_BFLOAT16_VEC_ACC * 4)) {
    colVecColLoopBody<MAX_BFLOAT16_VEC_ACC, LhsMapper, RhsMapper, false, linear>(row, cend, rows, lhs, rhs, pAlpha,
                                                                                 result);
    result += row;
  }
  if (rows & 3) {
    colVecColLoopBodyExtra<LhsMapper, RhsMapper, true, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
  } else {
    colVecColLoopBodyExtra<LhsMapper, RhsMapper, false, linear>(row, cend, rows, lhs, rhs, pAlpha, result);
  }
}

template <typename RhsMapper, typename LhsMapper, typename = void>
struct UseMMAStride : std::false_type {
  static EIGEN_ALWAYS_INLINE void run(Index j2, Index jend, Index rows, LhsMapper& lhs, RhsMapper& rhs, Packet4f pAlpha,
                                      float* result) {
    using RhsSubMapper = typename RhsMapper::SubMapper;

    RhsSubMapper rhs2 = rhs.getSubMapper(j2, 0);
    calcVecColLoops<LhsMapper, RhsSubMapper, false>(jend - j2, rows, lhs, rhs2, pAlpha, result);
  }
};

template <typename RhsMapper, typename LhsMapper>
struct UseMMAStride<RhsMapper, LhsMapper,
                    std::enable_if_t<std::is_member_function_pointer<decltype(&RhsMapper::stride)>::value>>
    : std::true_type {
  static EIGEN_ALWAYS_INLINE void run(Index j2, Index jend, Index rows, LhsMapper& lhs, RhsMapper& rhs, Packet4f pAlpha,
                                      float* result) {
    using RhsSubMapper = typename RhsMapper::SubMapper;

    RhsSubMapper rhs2 = rhs.getSubMapper(j2, 0);
    if (rhs.stride() == 1) {
      calcVecColLoops<LhsMapper, RhsSubMapper, true>(jend - j2, rows, lhs, rhs2, pAlpha, result);
    } else {
      calcVecColLoops<LhsMapper, RhsSubMapper, false>(jend - j2, rows, lhs, rhs2, pAlpha, result);
    }
  }
};

template <typename LhsMapper, typename RhsMapper>
void gemvMMA_bfloat16_col(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs, bfloat16* res,
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
    UseMMAStride<RhsMapper, LhsSubMapper>::run(j2, jend, rows, lhs2, rhs2, pAlpha, result);
  }

  convertArrayPointerF32toBF16(result, rows, res);
}

static Packet16uc p16uc_ELEMENT_VEC3 = {0x0c, 0x0d, 0x0e, 0x0f, 0x1c, 0x1d, 0x1e, 0x1f,
                                        0x0c, 0x0d, 0x0e, 0x0f, 0x1c, 0x1d, 0x1e, 0x1f};

template <Index num_acc>
EIGEN_ALWAYS_INLINE void preduxVecResults2(Packet4f (&acc)[num_acc][4], Index k) {
  if (num_acc > (k + 1)) {
    acc[k][0] = vec_mergeh(acc[k][0], acc[k + 1][0]);
    acc[k][1] = vec_mergeo(acc[k][1], acc[k + 1][1]);
    acc[k][2] = vec_mergel(acc[k][2], acc[k + 1][2]);
    acc[k][3] = vec_perm(acc[k][3], acc[k + 1][3], p16uc_ELEMENT_VEC3);

    acc[k][0] = (acc[k][0] + acc[k][2]) + (acc[k][1] + acc[k][3]);
  } else {
    acc[k][0] = vec_mergeh(acc[k][0], acc[k][1]);
    acc[k][0] += vec_mergel(acc[k][2], acc[k][3]);
#ifdef _BIG_ENDIAN
    acc[k][0] += vec_sld(acc[k][0], acc[k][0], 12);
#else
    acc[k][0] += vec_sld(acc[k][0], acc[k][0], 4);
#endif
  }
}

template <Index num_acc>
EIGEN_ALWAYS_INLINE void preduxVecResults(Packet4f (&acc)[num_acc][4]) {
  BFLOAT16_UNROLL
  for (Index k = 0; k < num_acc; k += 4) {
    preduxVecResults2<num_acc>(acc, k + 0);
    if (num_acc > (k + 2)) {
      preduxVecResults2<num_acc>(acc, k + 2);
      acc[k + 0][0] = reinterpret_cast<Packet4f>(
          vec_mergeh(reinterpret_cast<Packet2ul>(acc[k + 0][0]), reinterpret_cast<Packet2ul>(acc[k + 2][0])));
    }
  }
}

template <Index num_acc, typename LhsMapper, typename RhsMapper, bool extra>
EIGEN_ALWAYS_INLINE void multVecLoop(__vector_quad (&quad_acc)[num_acc], const LhsMapper& lhs, RhsMapper& rhs, Index j,
                                     Index extra_cols) {
  Packet8bf a0[num_acc], b0;

  if (extra) {
    b0 = rhs.template loadPacketPartial<Packet8bf>(j, extra_cols);
  } else {
    b0 = rhs.template loadPacket<Packet8bf>(j);
  }

  const LhsMapper lhs2 = lhs.getSubMapper(0, j);
  BFLOAT16_UNROLL
  for (Index k = 0; k < num_acc; k++) {
    if (extra) {
      a0[k] = lhs2.template loadPacketPartial<Packet8bf>(k, 0, extra_cols);
    } else {
      a0[k] = lhs2.template loadPacket<Packet8bf>(k, 0);
    }
  }

  multVec<num_acc>(quad_acc, a0, b0);
}

template <Index num_acc, typename LhsMapper, typename RhsMapper>
EIGEN_ALWAYS_INLINE void vecLoop(Index cols, const LhsMapper& lhs, RhsMapper& rhs, __vector_quad (&quad_acc)[num_acc],
                                 Index extra_cols) {
  Index j = 0;
  for (; j + 8 <= cols; j += 8) {
    multVecLoop<num_acc, LhsMapper, RhsMapper, false>(quad_acc, lhs, rhs, j, extra_cols);
  }

  if (extra_cols) {
    multVecLoop<num_acc, LhsMapper, RhsMapper, true>(quad_acc, lhs, rhs, j, extra_cols);
  }
}

template <const Index num_acc, typename LhsMapper, typename RhsMapper>
void colVecLoopBody(Index& row, Index cols, Index rows, LhsMapper& lhs, RhsMapper& rhs, const Packet4f pAlpha,
                    float* result) {
  constexpr bool multiIters = (num_acc == MAX_BFLOAT16_VEC_ACC);
  const Index extra_cols = (cols & 7);

  do {
    Packet4f acc[num_acc][4];
    __vector_quad quad_acc[num_acc];

    zeroAccumulators<num_acc>(quad_acc);

    const LhsMapper lhs2 = lhs.getSubMapper(row, 0);
    vecLoop<num_acc, LhsMapper, RhsMapper>(cols, lhs2, rhs, quad_acc, extra_cols);

    disassembleAccumulators<num_acc>(quad_acc, acc);

    preduxVecResults<num_acc>(acc);

    outputVecResults<num_acc>(acc, result, pAlpha);

    result += num_acc;
  } while (multiIters && (num_acc <= rows - (row += num_acc)));
}

template <const Index num_acc, typename LhsMapper, typename RhsMapper>
EIGEN_ALWAYS_INLINE void colVecLoopBodyExtraN(Index& row, Index cols, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                              const Packet4f pAlpha, float* result) {
  if (MAX_BFLOAT16_VEC_ACC > num_acc) {
    colVecLoopBody<num_acc, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
  }
}

template <typename LhsMapper, typename RhsMapper>
EIGEN_ALWAYS_INLINE void colVecLoopBodyExtra(Index& row, Index cols, Index rows, LhsMapper& lhs, RhsMapper& rhs,
                                             const Packet4f pAlpha, float* result) {
  switch (rows - row) {
    case 7:
      colVecLoopBodyExtraN<7, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 6:
      colVecLoopBodyExtraN<6, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 5:
      colVecLoopBodyExtraN<5, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 4:
      colVecLoopBodyExtraN<4, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 3:
      colVecLoopBodyExtraN<3, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 2:
      colVecLoopBodyExtraN<2, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
    case 1:
      colVecLoopBodyExtraN<1, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
      break;
  }
}

template <typename LhsMapper, typename RhsMapper>
EIGEN_ALWAYS_INLINE void calcVecLoops(Index cols, Index rows, LhsMapper& lhs, RhsMapper& rhs, const Packet4f pAlpha,
                                      float* result) {
  Index row = 0;
  if (rows >= MAX_BFLOAT16_VEC_ACC) {
    colVecLoopBody<MAX_BFLOAT16_VEC_ACC, LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
    result += row;
  }
  colVecLoopBodyExtra<LhsMapper, RhsMapper>(row, cols, rows, lhs, rhs, pAlpha, result);
}

template <typename LhsMapper, typename RhsMapper>
EIGEN_STRONG_INLINE void gemvMMA_bfloat16_row(Index rows, Index cols, const LhsMapper& alhs, const RhsMapper& rhs,
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
  calcVecLoops<LhsMapper, LinearMapper>(cols, rows, lhs, rhs2, pAlpha, result);
  if (resIncr == 1) {
    convertArrayPointerF32toBF16(result, rows, res);
  } else {
    convertArrayPointerF32toBF16<true>(result, rows, res, resIncr);
  }
}
#endif

#undef MAX_BFLOAT16_VEC_ACC
#undef BFLOAT16_UNROLL

}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_MATRIX_PRODUCT_MMA_BFLOAT16_ALTIVEC_H
