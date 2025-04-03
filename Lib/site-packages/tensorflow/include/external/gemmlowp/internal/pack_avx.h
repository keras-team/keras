// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// pack_avx.h: optimized AVX specializations of the templates in pack.h.

#ifndef GEMMLOWP_INTERNAL_PACK_AVX_H_
#define GEMMLOWP_INTERNAL_PACK_AVX_H_

#include <immintrin.h>
#include "pack.h"

namespace gemmlowp {

// TODO: Add DepthMajorUint8SideMap

typedef SideMap<const std::uint8_t, SideMapOrder::WidthMajor>
    WidthMajorUint8SideMap;

template <int Cells>
using WidthMajorSideFormatNCells4x2 =
    KernelSideFormat<CellFormat<8, 2, CellOrder::WidthMajor>, Cells>;

template <int Cells>
class PackingRegisterBlock<
    WidthMajorUint8SideMap,
    PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells>>>
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells>>> {
 public:
  typedef WidthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat> *dst, int start_width) {
    std::uint8_t *dst_ptr = dst->current_data();
    const int width_stride = this->complete_src_.width_stride();
    int depth_step = 16;

    __m256i one = _mm256_set1_epi16(1);
    for (int cell_start_depth = 0; cell_start_depth < kRegisterSize;
         cell_start_depth += depth_step) {
      for (int cell_start_width = 0; cell_start_width < kKernelWidth;
           cell_start_width += kCellWidth) {
        std::int32_t *cell_sums_of_each_slice_ptr =
            dst->sums_of_each_slice() + start_width + cell_start_width;
        const std::uint8_t *src_data =
            this->complete_src_.data(cell_start_width, cell_start_depth);

        __m128i xmm1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(&src_data[0]));
        __m128i xmm2 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&src_data[1 * width_stride]));
        __m128i xmm3 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&src_data[2 * width_stride]));
        __m128i xmm4 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&src_data[3 * width_stride]));
        __m128i xmm5 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&src_data[4 * width_stride]));
        __m128i xmm6 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&src_data[5 * width_stride]));
        __m128i xmm7 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&src_data[6 * width_stride]));
        __m128i xmm8 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&src_data[7 * width_stride]));

        __m256i ymm1 = _mm256_set_m128i(xmm5, xmm1);
        __m256i ymm2 = _mm256_set_m128i(xmm6, xmm2);
        __m256i ymm3 = _mm256_set_m128i(xmm7, xmm3);
        __m256i ymm4 = _mm256_set_m128i(xmm8, xmm4);

        __m256i ymm5 = _mm256_unpacklo_epi16(ymm1, ymm2);
        __m256i ymm6 = _mm256_unpacklo_epi16(ymm3, ymm4);

        __m256i ymm9 = _mm256_unpackhi_epi16(ymm1, ymm2);
        __m256i ymm10 = _mm256_unpackhi_epi16(ymm3, ymm4);

        __m256i ymm7 = _mm256_unpacklo_epi32(ymm5, ymm6);
        __m256i ymm8 = _mm256_unpackhi_epi32(ymm5, ymm6);

        __m256i ymm13 = _mm256_unpacklo_epi32(ymm9, ymm10);
        __m256i ymm14 = _mm256_unpackhi_epi32(ymm9, ymm10);

        __m256i ymm11 = _mm256_permute4x64_epi64(ymm7, 0xd8);
        __m256i ymm12 = _mm256_permute4x64_epi64(ymm8, 0xd8);

        __m256i ymm15 = _mm256_permute4x64_epi64(ymm13, 0xd8);
        __m256i ymm16 = _mm256_permute4x64_epi64(ymm14, 0xd8);

        __m128i xmm9 = _mm256_castsi256_si128(ymm11);
        __m128i xmm10 = _mm256_castsi256_si128(ymm12);
        __m128i xmm11 = _mm256_extracti128_si256(ymm11, 1);
        __m128i xmm12 = _mm256_extracti128_si256(ymm12, 1);

        xmm1 = _mm256_castsi256_si128(ymm15);
        xmm2 = _mm256_castsi256_si128(ymm16);
        xmm3 = _mm256_extracti128_si256(ymm15, 1);
        xmm4 = _mm256_extracti128_si256(ymm16, 1);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(&dst_ptr[0]), xmm9);
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&dst_ptr[kCellSize * kCells]), xmm11);
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&dst_ptr[2 * kCellSize * kCells]),
            xmm10);
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&dst_ptr[3 * kCellSize * kCells]),
            xmm12);
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&dst_ptr[4 * kCellSize * kCells]),
            xmm1);
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&dst_ptr[5 * kCellSize * kCells]),
            xmm3);

        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&dst_ptr[6 * kCellSize * kCells]),
            xmm2);
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&dst_ptr[7 * kCellSize * kCells]),
            xmm4);

        ymm6 = _mm256_cvtepu8_epi16(xmm9);
        ymm7 = _mm256_madd_epi16(ymm6, one);
        __m256i sums_of_each_slice_xmm = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(&cell_sums_of_each_slice_ptr[0]));
        sums_of_each_slice_xmm = _mm256_add_epi32(sums_of_each_slice_xmm, ymm7);

        ymm6 = _mm256_cvtepu8_epi16(xmm11);
        ymm7 = _mm256_madd_epi16(ymm6, one);
        sums_of_each_slice_xmm = _mm256_add_epi32(sums_of_each_slice_xmm, ymm7);

        ymm6 = _mm256_cvtepu8_epi16(xmm10);
        ymm7 = _mm256_madd_epi16(ymm6, one);
        sums_of_each_slice_xmm = _mm256_add_epi32(sums_of_each_slice_xmm, ymm7);

        ymm6 = _mm256_cvtepu8_epi16(xmm12);
        ymm7 = _mm256_madd_epi16(ymm6, one);
        sums_of_each_slice_xmm = _mm256_add_epi32(sums_of_each_slice_xmm, ymm7);

        ymm6 = _mm256_cvtepu8_epi16(xmm1);
        ymm7 = _mm256_madd_epi16(ymm6, one);
        sums_of_each_slice_xmm = _mm256_add_epi32(sums_of_each_slice_xmm, ymm7);

        ymm6 = _mm256_cvtepu8_epi16(xmm3);
        ymm7 = _mm256_madd_epi16(ymm6, one);
        sums_of_each_slice_xmm = _mm256_add_epi32(sums_of_each_slice_xmm, ymm7);

        ymm6 = _mm256_cvtepu8_epi16(xmm2);
        ymm7 = _mm256_madd_epi16(ymm6, one);
        sums_of_each_slice_xmm = _mm256_add_epi32(sums_of_each_slice_xmm, ymm7);

        ymm6 = _mm256_cvtepu8_epi16(xmm4);
        ymm7 = _mm256_madd_epi16(ymm6, one);
        sums_of_each_slice_xmm = _mm256_add_epi32(sums_of_each_slice_xmm, ymm7);

        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(&cell_sums_of_each_slice_ptr[0]),
            sums_of_each_slice_xmm);
        dst_ptr += kCellSize;
      }
      dst_ptr += 7 * kCellSize * kCells;
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

// Pack format for 4x2 rhs format
template <int Cells>
using RhsWidthMajorSideFormatNCells4x2 =
    KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, Cells>;

template <int Cells>
class PackingRegisterBlock<
    WidthMajorUint8SideMap,
    PackedSideBlock<RhsWidthMajorSideFormatNCells4x2<Cells>>>
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<RhsWidthMajorSideFormatNCells4x2<Cells>>> {
 public:
  typedef RhsWidthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat> *dst, int start_width) {
    std::uint8_t *dst_ptr = dst->current_data();
    const int width_stride = this->complete_src_.width_stride();
    int depth_step = 8;

    __m128i one = _mm_set1_epi16(1);
    for (int cell_start_depth = 0; cell_start_depth < kRegisterSize;
         cell_start_depth += depth_step) {
      for (int cell_start_width = 0; cell_start_width < kKernelWidth;
           cell_start_width += kCellWidth) {
        std::int32_t *cell_sums_of_each_slice_ptr =
            dst->sums_of_each_slice() + start_width + cell_start_width;
        const std::uint8_t *src_data =
            this->complete_src_.data(cell_start_width, cell_start_depth);

        __m128i xmm1 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&src_data[0]));
        __m128i xmm2 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(&src_data[1 * width_stride]));
        __m128i xmm3 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(&src_data[2 * width_stride]));
        __m128i xmm4 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(&src_data[3 * width_stride]));

        __m128i xmm5 = _mm_unpacklo_epi16(xmm1, xmm2);
        __m128i xmm8 = _mm_shuffle_epi32(xmm5, 0x31);

        __m128i xmm6 = _mm_unpacklo_epi16(xmm3, xmm4);
        __m128i xmm7 = _mm_shuffle_epi32(xmm6, 0x80);

        __m128i xmm9 = _mm_blend_epi16(xmm5, xmm7, 0xcc);
        __m128i xmm10 = _mm_blend_epi16(xmm8, xmm6, 0xcc);

        _mm_storel_epi64(reinterpret_cast<__m128i *>(&dst_ptr[0]), xmm9);
        _mm_storel_epi64(
            reinterpret_cast<__m128i *>(&dst_ptr[kCellSize * kCells]), xmm10);

        __m128i xmm11 = _mm_shuffle_epi32(xmm9, 0xee);
        __m128i xmm12 = _mm_shuffle_epi32(xmm10, 0xee);

        _mm_storel_epi64(
            reinterpret_cast<__m128i *>(&dst_ptr[2 * kCellSize * kCells]),
            xmm11);
        _mm_storel_epi64(
            reinterpret_cast<__m128i *>(&dst_ptr[3 * kCellSize * kCells]),
            xmm12);

        xmm1 = _mm_cvtepu8_epi16(xmm9);
        xmm2 = _mm_madd_epi16(xmm1, one);
        __m128i sums_of_each_slice_xmm = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&cell_sums_of_each_slice_ptr[0]));
        sums_of_each_slice_xmm = _mm_add_epi32(sums_of_each_slice_xmm, xmm2);

        xmm1 = _mm_cvtepu8_epi16(xmm10);
        xmm2 = _mm_madd_epi16(xmm1, one);
        sums_of_each_slice_xmm = _mm_add_epi32(sums_of_each_slice_xmm, xmm2);

        xmm1 = _mm_cvtepu8_epi16(xmm11);
        xmm2 = _mm_madd_epi16(xmm1, one);
        sums_of_each_slice_xmm = _mm_add_epi32(sums_of_each_slice_xmm, xmm2);

        xmm1 = _mm_cvtepu8_epi16(xmm12);
        xmm2 = _mm_madd_epi16(xmm1, one);
        sums_of_each_slice_xmm = _mm_add_epi32(sums_of_each_slice_xmm, xmm2);

        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&cell_sums_of_each_slice_ptr[0]),
            sums_of_each_slice_xmm);
        dst_ptr += kCellSize;
      }
      dst_ptr += 3 * kCellSize * kCells;
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_AVX_H_
