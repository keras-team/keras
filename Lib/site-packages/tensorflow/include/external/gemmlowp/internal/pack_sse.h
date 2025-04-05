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

// pack_SSE.h: optimized SSE specializations of the templates in pack.h.

#ifndef GEMMLOWP_INTERNAL_PACK_SSE_H_
#define GEMMLOWP_INTERNAL_PACK_SSE_H_

#include <smmintrin.h>
#include "pack.h"

namespace gemmlowp {

// TODO: Add DepthMajorUint8SideMap

typedef SideMap<const std::uint8_t, SideMapOrder::WidthMajor>
    WidthMajorUint8SideMap;

template <int Cells>
using WidthMajorSideFormatNCells4x2 =
    KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, Cells>;

template <int Cells>
class PackingRegisterBlock<
    WidthMajorUint8SideMap,
    PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells> > >
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<WidthMajorSideFormatNCells4x2<Cells> > > {
 public:
  typedef WidthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static constexpr int kCells = KernelSideFormat::kCells;
  static constexpr int kCellWidth = CellFormat::kWidth;
  static constexpr int kKernelWidth = CellFormat::kWidth * kCells;
  static constexpr int kCellDepth = CellFormat::kDepth;
  static constexpr int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    std::uint8_t* dst_ptr = dst->current_data();
    const int width_stride = this->complete_src_.width_stride();
    int depth_step = 8;

    __m128i one = _mm_set1_epi16(1);
    for (int cell_start_depth = 0; cell_start_depth < kRegisterSize;
         cell_start_depth += depth_step) {
      for (int cell_start_width = 0; cell_start_width < kKernelWidth;
           cell_start_width += kCellWidth) {
        std::int32_t* cell_sums_of_each_slice_ptr =
            dst->sums_of_each_slice() + start_width + cell_start_width;
        const std::uint8_t* src_data =
            this->complete_src_.data(cell_start_width, cell_start_depth);

        __m128i xmm1 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src_data[0]));
        __m128i xmm2 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(&src_data[1 * width_stride]));
        __m128i xmm3 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(&src_data[2 * width_stride]));
        __m128i xmm4 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(&src_data[3 * width_stride]));

        __m128i xmm5 = _mm_unpacklo_epi16(xmm1, xmm2);
        __m128i xmm8 = _mm_shuffle_epi32(xmm5, 0x31);

        __m128i xmm6 = _mm_unpacklo_epi16(xmm3, xmm4);
        __m128i xmm7 = _mm_shuffle_epi32(xmm6, 0x80);

        __m128i xmm9 = _mm_blend_epi16(xmm5, xmm7, 0xcc);
        __m128i xmm10 = _mm_blend_epi16(xmm8, xmm6, 0xcc);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst_ptr[0]), xmm9);
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(&dst_ptr[kCellSize * kCells]), xmm10);

        __m128i xmm11 = _mm_shuffle_epi32(xmm9, 0xee);
        __m128i xmm12 = _mm_shuffle_epi32(xmm10, 0xee);

        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(&dst_ptr[2 * kCellSize * kCells]),
            xmm11);
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(&dst_ptr[3 * kCellSize * kCells]),
            xmm12);

        xmm1 = _mm_cvtepu8_epi16(xmm9);
        xmm2 = _mm_madd_epi16(xmm1, one);
        __m128i sums_of_each_slice_xmm = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&cell_sums_of_each_slice_ptr[0]));
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
            reinterpret_cast<__m128i*>(&cell_sums_of_each_slice_ptr[0]),
            sums_of_each_slice_xmm);
        dst_ptr += kCellSize;
      }
      dst_ptr += 3 * kCellSize * kCells;
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_SSE_H_
