// #define EIGEN_POWER_USE_PREFETCH  // Use prefetching in gemm routines
#ifdef EIGEN_POWER_USE_PREFETCH
#define EIGEN_POWER_PREFETCH(p) prefetch(p)
#else
#define EIGEN_POWER_PREFETCH(p)
#endif

#if defined(_ARCH_PWR9) || defined(EIGEN_ALTIVEC_MMA_DYNAMIC_DISPATCH)
#define USE_PARTIAL_PACKETS
#endif

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Scalar, typename Packet, typename DataMapper, const Index accRows, const Index accCols>
EIGEN_ALWAYS_INLINE void gemm_extra_row(const DataMapper& res, const Scalar* lhs_base, const Scalar* rhs_base,
                                        Index depth, Index strideA, Index offsetA, Index strideB, Index row, Index rows,
                                        Index remaining_rows, const Packet& pAlpha, const Packet& pMask);

template <typename Scalar, typename Packet, typename DataMapper, const Index accCols>
EIGEN_ALWAYS_INLINE void gemm_extra_cols(const DataMapper& res, const Scalar* blockA, const Scalar* blockB, Index depth,
                                         Index strideA, Index offsetA, Index strideB, Index offsetB, Index col,
                                         Index rows, Index cols, Index remaining_rows, const Packet& pAlpha,
                                         const Packet& pMask);

template <typename Packet>
EIGEN_ALWAYS_INLINE Packet bmask(const Index remaining_rows);

template <typename Scalar, typename Packet, typename Packetc, typename DataMapper, const Index accRows,
          const Index accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void gemm_complex_extra_row(const DataMapper& res, const Scalar* lhs_base, const Scalar* rhs_base,
                                                Index depth, Index strideA, Index offsetA, Index strideB, Index row,
                                                Index rows, Index remaining_rows, const Packet& pAlphaReal,
                                                const Packet& pAlphaImag, const Packet& pMask);

template <typename Scalar, typename Packet, typename Packetc, typename DataMapper, const Index accCols,
          bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void gemm_complex_extra_cols(const DataMapper& res, const Scalar* blockA, const Scalar* blockB,
                                                 Index depth, Index strideA, Index offsetA, Index strideB,
                                                 Index offsetB, Index col, Index rows, Index cols, Index remaining_rows,
                                                 const Packet& pAlphaReal, const Packet& pAlphaImag,
                                                 const Packet& pMask);

template <typename DataMapper>
EIGEN_ALWAYS_INLINE void convertArrayBF16toF32(float* result, Index cols, Index rows, const DataMapper& src);

template <const Index size, bool non_unit_stride, Index delta>
EIGEN_ALWAYS_INLINE void storeBF16fromResult(bfloat16* dst, Packet8bf data, Index resInc, Index extra = 0);

template <bool non_unit_stride = false>
EIGEN_ALWAYS_INLINE void convertArrayPointerBF16toF32(float* result, Index cols, Index rows, bfloat16* src,
                                                      Index resInc = 1);

template <bool rhsExtraCols, bool lhsExtraRows>
EIGEN_ALWAYS_INLINE void storeResults(Packet4f (&acc)[4], Index rows, const Packet4f pAlpha, float* result,
                                      Index extra_cols, Index extra_rows);

template <Index num_acc, bool extraRows, Index size = 4>
EIGEN_ALWAYS_INLINE void outputVecColResults(Packet4f (&acc)[num_acc][size], float* result, Packet4f pAlpha,
                                             Index extra_rows);

template <Index num_acc, Index size = 4>
EIGEN_ALWAYS_INLINE void outputVecResults(Packet4f (&acc)[num_acc][size], float* result, Packet4f pAlpha);

template <typename RhsMapper, bool linear>
EIGEN_ALWAYS_INLINE Packet8bf loadColData(RhsMapper& rhs, Index j);

template <typename Packet>
EIGEN_ALWAYS_INLINE Packet ploadLhs(const __UNPACK_TYPE__(Packet) * lhs);

template <typename DataMapper, typename Packet, const Index accCols, int StorageOrder, bool Complex, int N,
          bool full = true>
EIGEN_ALWAYS_INLINE void bload(PacketBlock<Packet, N*(Complex ? 2 : 1)>& acc, const DataMapper& res, Index row,
                               Index col);

template <typename DataMapper, typename Packet, int N>
EIGEN_ALWAYS_INLINE void bstore(PacketBlock<Packet, N>& acc, const DataMapper& res, Index row);

#ifdef USE_PARTIAL_PACKETS
template <typename DataMapper, typename Packet, const Index accCols, bool Complex, Index N, bool full = true>
EIGEN_ALWAYS_INLINE void bload_partial(PacketBlock<Packet, N*(Complex ? 2 : 1)>& acc, const DataMapper& res, Index row,
                                       Index elements);

template <typename DataMapper, typename Packet, Index N>
EIGEN_ALWAYS_INLINE void bstore_partial(PacketBlock<Packet, N>& acc, const DataMapper& res, Index row, Index elements);
#endif

template <typename Packet, int N>
EIGEN_ALWAYS_INLINE void bscale(PacketBlock<Packet, N>& acc, PacketBlock<Packet, N>& accZ, const Packet& pAlpha);

template <typename Packet, int N, bool mask>
EIGEN_ALWAYS_INLINE void bscale(PacketBlock<Packet, N>& acc, PacketBlock<Packet, N>& accZ, const Packet& pAlpha,
                                const Packet& pMask);

template <typename Packet, int N, bool mask>
EIGEN_ALWAYS_INLINE void bscalec(PacketBlock<Packet, N>& aReal, PacketBlock<Packet, N>& aImag, const Packet& bReal,
                                 const Packet& bImag, PacketBlock<Packet, N>& cReal, PacketBlock<Packet, N>& cImag,
                                 const Packet& pMask);

template <typename Packet, typename Packetc, int N, bool full>
EIGEN_ALWAYS_INLINE void bcouple(PacketBlock<Packet, N>& taccReal, PacketBlock<Packet, N>& taccImag,
                                 PacketBlock<Packetc, N * 2>& tRes, PacketBlock<Packetc, N>& acc1,
                                 PacketBlock<Packetc, N>& acc2);

#define MICRO_NORMAL(iter) (accCols == accCols2) || (unroll_factor != (iter + 1))

#define MICRO_UNROLL_ITER1(func, N)          \
  switch (remaining_rows) {                  \
    default:                                 \
      func(N, 0) break;                      \
    case 1:                                  \
      func(N, 1) break;                      \
    case 2:                                  \
      if (sizeof(Scalar) == sizeof(float)) { \
        func(N, 2)                           \
      }                                      \
      break;                                 \
    case 3:                                  \
      if (sizeof(Scalar) == sizeof(float)) { \
        func(N, 3)                           \
      }                                      \
      break;                                 \
  }

#ifdef USE_PARTIAL_PACKETS
#define MICRO_UNROLL_ITER(func, N) \
  if (remaining_rows) {            \
    func(N, true);                 \
  } else {                         \
    func(N, false);                \
  }

#define MICRO_NORMAL_PARTIAL(iter) full || (unroll_factor != (iter + 1))
#else
#define MICRO_UNROLL_ITER(func, N) MICRO_UNROLL_ITER1(func, N)
#endif

#define MICRO_COMPLEX_UNROLL_ITER(func, N) MICRO_UNROLL_ITER1(func, N)

#define MICRO_NORMAL_COLS(iter, a, b) ((MICRO_NORMAL(iter)) ? a : b)

#define MICRO_LOAD1(lhs_ptr, iter)                               \
  if (unroll_factor > iter) {                                    \
    lhsV##iter = ploadLhs<Packet>(lhs_ptr##iter);                \
    lhs_ptr##iter += MICRO_NORMAL_COLS(iter, accCols, accCols2); \
  } else {                                                       \
    EIGEN_UNUSED_VARIABLE(lhsV##iter);                           \
  }

#define MICRO_LOAD_ONE(iter) MICRO_LOAD1(lhs_ptr, iter)

#define MICRO_COMPLEX_LOAD_ONE(iter)                                                                       \
  if (!LhsIsReal && (unroll_factor > iter)) {                                                              \
    lhsVi##iter = ploadLhs<Packet>(lhs_ptr_real##iter + MICRO_NORMAL_COLS(iter, imag_delta, imag_delta2)); \
  } else {                                                                                                 \
    EIGEN_UNUSED_VARIABLE(lhsVi##iter);                                                                    \
  }                                                                                                        \
  MICRO_LOAD1(lhs_ptr_real, iter)

#define MICRO_SRC_PTR1(lhs_ptr, advRows, iter)                                  \
  if (unroll_factor > iter) {                                                   \
    lhs_ptr##iter = lhs_base + (row + (iter * accCols)) * strideA * advRows -   \
                    MICRO_NORMAL_COLS(iter, 0, (accCols - accCols2) * offsetA); \
  } else {                                                                      \
    EIGEN_UNUSED_VARIABLE(lhs_ptr##iter);                                       \
  }

#define MICRO_SRC_PTR_ONE(iter) MICRO_SRC_PTR1(lhs_ptr, 1, iter)

#define MICRO_COMPLEX_SRC_PTR_ONE(iter) MICRO_SRC_PTR1(lhs_ptr_real, advanceRows, iter)

#define MICRO_PREFETCH1(lhs_ptr, iter)   \
  if (unroll_factor > iter) {            \
    EIGEN_POWER_PREFETCH(lhs_ptr##iter); \
  }

#define MICRO_PREFETCH_ONE(iter) MICRO_PREFETCH1(lhs_ptr, iter)

#define MICRO_COMPLEX_PREFETCH_ONE(iter) MICRO_PREFETCH1(lhs_ptr_real, iter)

#ifdef USE_PARTIAL_PACKETS
#define MICRO_UPDATE_MASK
#else
#define MICRO_UPDATE_MASK EIGEN_UNUSED_VARIABLE(pMask);
#endif

#define MICRO_UPDATE                \
  if (accCols == accCols2) {        \
    MICRO_UPDATE_MASK               \
    EIGEN_UNUSED_VARIABLE(offsetA); \
    row += unroll_factor * accCols; \
  }

#define MICRO_COMPLEX_UPDATE                \
  MICRO_UPDATE                              \
  if (LhsIsReal || (accCols == accCols2)) { \
    EIGEN_UNUSED_VARIABLE(imag_delta2);     \
  }

}  // end namespace internal
}  // end namespace Eigen
