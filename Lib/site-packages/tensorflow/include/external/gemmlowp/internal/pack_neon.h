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

// pack_neon.h: optimized NEON specializations of the templates in pack.h.

#ifndef GEMMLOWP_INTERNAL_PACK_NEON_H_
#define GEMMLOWP_INTERNAL_PACK_NEON_H_

#include "pack.h"

#include <arm_neon.h>

namespace gemmlowp {

typedef SideMap<const std::uint8_t, SideMapOrder::WidthMajor>
    WidthMajorUint8SideMap;

typedef SideMap<const std::int8_t, SideMapOrder::WidthMajor>
    WidthMajorInt8SideMap;

template <int Cells>
using DepthMajorSideFormatNCells4x2 = KernelSideFormat<CellFormat<4, 2>, Cells>;

template <int Cells>
class PackingRegisterBlock<
    WidthMajorUint8SideMap,
    PackedSideBlock<DepthMajorSideFormatNCells4x2<Cells>>>
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<DepthMajorSideFormatNCells4x2<Cells>>> {
 public:
  typedef DepthMajorSideFormatNCells4x2<Cells> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* const src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load source WidthMajor data
    uint8x16_t src_lines[4 * kCells];
    for (int i = 0; i < 4 * kCells; i++) {
      src_lines[i] = vld1q_u8(src_ptr + i * stride);
    }
    // Reorder the data within registers to make DepthMajor 4x2 cells
    uint8x16x2_t src_lines_intertwined_2x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_2x[2 * i] =
          vzipq_u8(src_lines[4 * i], src_lines[4 * i + 2]);
      src_lines_intertwined_2x[2 * i + 1] =
          vzipq_u8(src_lines[4 * i + 1], src_lines[4 * i + 3]);
    }
    uint8x16x2_t src_lines_intertwined_4x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_4x[2 * i] =
          vzipq_u8(src_lines_intertwined_2x[2 * i].val[0],
                   src_lines_intertwined_2x[2 * i + 1].val[0]);
      src_lines_intertwined_4x[2 * i + 1] =
          vzipq_u8(src_lines_intertwined_2x[2 * i].val[1],
                   src_lines_intertwined_2x[2 * i + 1].val[1]);
    }
    // Store the resulting DepthMajor 4x2 cells in the destination packed block
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        for (int cell = 0; cell < kCells; cell++) {
          uint8x8_t value = vget_low_u8(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]);
          vst1_u8(dst_ptr, value);
          dst_ptr += 8;
        }
        for (int cell = 0; cell < kCells; cell++) {
          uint8x8_t value = vget_high_u8(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]);
          vst1_u8(dst_ptr, value);
          dst_ptr += 8;
        }
      }
    }
    // Compute sums across the depth dimension
    uint16x8_t sums_of_2_cells[kCells][4];
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        int i = 2 * outer + inner;
        for (int cell = 0; cell < kCells; cell++) {
          sums_of_2_cells[cell][i] = vaddl_u8(
              vget_low_u8(
                  src_lines_intertwined_4x[2 * cell + outer].val[inner]),
              vget_high_u8(
                  src_lines_intertwined_4x[2 * cell + outer].val[inner]));
        }
      }
    }
    int32x4_t sums_of_4_cells[kCells][4];
    for (int i = 0; i < 4; i++) {
      for (int cell = 0; cell < kCells; cell++) {
        sums_of_4_cells[cell][i] = vreinterpretq_s32_u32(
            vaddl_u16(vget_low_u16(sums_of_2_cells[cell][i]),
                      vget_high_u16(sums_of_2_cells[cell][i])));
      }
    }
    // Update the sums_of_each_slice vector
    for (int cell = 0; cell < kCells; cell++) {
      int32x4_t s01 =
          vaddq_s32(sums_of_4_cells[cell][0], sums_of_4_cells[cell][1]);
      int32x4_t s23 =
          vaddq_s32(sums_of_4_cells[cell][2], sums_of_4_cells[cell][3]);
      int32x4_t s = vaddq_s32(s01, s23);
      std::int32_t* sums_of_each_slice_ptr =
          dst->sums_of_each_slice() + start_width + 4 * cell;
      vst1q_s32(sums_of_each_slice_ptr,
                vaddq_s32(s, vld1q_s32(sums_of_each_slice_ptr)));
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

template <int Cells>
using WidthMajorSideFormatNCells4x2 =
    KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, Cells>;

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

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load source WidthMajor data
    uint16x8_t src_lines[kCells * 4];
    for (int i = 0; i < kCells; i++) {
      // This packing path is used with our current
      // less-than-8-bit kernel, and the partial unrolling of this loop
      // results in substantially faster code (thanks to better
      // register allocation) on Nexus 5.

#define GEMMLOWP_UNROLLED_LOOP_ITER(k)                            \
  src_lines[4 * i + k] = vreinterpretq_u16_u8(vld1q_u8(src_ptr)); \
  src_ptr += stride;

      GEMMLOWP_UNROLLED_LOOP_ITER(0)
      GEMMLOWP_UNROLLED_LOOP_ITER(1)
      GEMMLOWP_UNROLLED_LOOP_ITER(2)
      GEMMLOWP_UNROLLED_LOOP_ITER(3)

#undef GEMMLOWP_UNROLLED_LOOP_ITER
    }
    // Reorder the data within registers to make WidthMajor 4x2 cells
    uint16x8x2_t src_lines_intertwined_2x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_2x[2 * i] =
          vzipq_u16(src_lines[4 * i], src_lines[4 * i + 2]);
      src_lines_intertwined_2x[2 * i + 1] =
          vzipq_u16(src_lines[4 * i + 1], src_lines[4 * i + 3]);
    }
    uint16x8x2_t src_lines_intertwined_4x[2 * kCells];
    for (int i = 0; i < kCells; i++) {
      src_lines_intertwined_4x[2 * i] =
          vzipq_u16(src_lines_intertwined_2x[2 * i].val[0],
                    src_lines_intertwined_2x[2 * i + 1].val[0]);
      src_lines_intertwined_4x[2 * i + 1] =
          vzipq_u16(src_lines_intertwined_2x[2 * i].val[1],
                    src_lines_intertwined_2x[2 * i + 1].val[1]);
    }
    // Store the resulting WidthMajor 4x2 cells in the destination packed block
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        for (int cell = 0; cell < kCells; cell++) {
          uint8x8_t value = vreinterpret_u8_u16(vget_low_u16(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]));
          vst1_u8(dst_ptr, value);
          dst_ptr += 8;
        }
        for (int cell = 0; cell < kCells; cell++) {
          uint8x8_t value = vreinterpret_u8_u16(vget_high_u16(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]));
          vst1_u8(dst_ptr, value);
          dst_ptr += 8;
        }
      }
    }
    // Compute sums across the depth dimension
    uint16x8_t sums_of_2[kCells][4];
    for (int outer = 0; outer < 2; outer++) {
      for (int inner = 0; inner < 2; inner++) {
        int i = 2 * outer + inner;
        for (int cell = 0; cell < kCells; cell++) {
          sums_of_2[cell][i] = vpaddlq_u8(vreinterpretq_u8_u16(
              src_lines_intertwined_4x[2 * cell + outer].val[inner]));
        }
      }
    }
    uint16x8_t sums_of_4[kCells][2];
    for (int i = 0; i < 2; i++) {
      for (int cell = 0; cell < kCells; cell++) {
        sums_of_4[cell][i] =
            vaddq_u16(sums_of_2[cell][2 * i], sums_of_2[cell][2 * i + 1]);
      }
    }
    uint16x8_t sums_of_8[kCells];
    for (int cell = 0; cell < kCells; cell++) {
      sums_of_8[cell] = vaddq_u16(sums_of_4[cell][0], sums_of_4[cell][1]);
    }

    uint16x4_t sums_of_16[kCells];
    for (int cell = 0; cell < kCells; cell++) {
      sums_of_16[cell] = vadd_u16(vget_low_u16(sums_of_8[cell]),
                                  vget_high_u16(sums_of_8[cell]));
    }
    // Update the sums_of_each_slice vector
    for (int cell = 0; cell < kCells; cell++) {
      int32x4_t s = vreinterpretq_s32_u32(vmovl_u16(sums_of_16[cell]));
      std::int32_t* sums_of_each_slice_ptr =
          dst->sums_of_each_slice() + start_width + 4 * cell;
      vst1q_s32(sums_of_each_slice_ptr,
                vaddq_s32(s, vld1q_s32(sums_of_each_slice_ptr)));
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

#ifdef GEMMLOWP_NEON_32
inline int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
  const int16x4_t c = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
  const int16x4_t d = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
  return vcombine_s16(c, d);
}
#endif

template <int Width>
using Int8FastKernelFormat =
    KernelSideFormatInt8<CellFormat<Width, 16, CellOrder::WidthMajor>, 1>;

template <int Width>
class PackingRegisterBlock<WidthMajorUint8SideMap,
                           PackedSideBlock<Int8FastKernelFormat<Width>>>
    : public PackingRegisterBlockBase<
          WidthMajorUint8SideMap,
          PackedSideBlock<Int8FastKernelFormat<Width>>> {
 public:
  static_assert(Width == 2 || Width == 4, "");
  typedef Int8FastKernelFormat<Width> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    std::int32_t* sums_ptr = dst->sums_of_each_slice() + start_width;
    std::uint8_t* dst_ptr = dst->current_data();
    const std::uint8_t* const src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load source WidthMajor data
    uint8x16_t src_lines[Width];
    for (int i = 0; i < Width; i++) {
      src_lines[i] = vld1q_u8(src_ptr + i * stride);
    }
    const uint8x16_t sign_bit_dup = vdupq_n_u8(0x80);
    for (int i = 0; i < Width; i++) {
      src_lines[i] = veorq_u8(src_lines[i], sign_bit_dup);
    }
    for (int i = 0; i < Width; i++) {
      vst1q_u8(dst_ptr + 16 * i, src_lines[i]);
    }
    int16x8_t sums2[Width];
    for (int i = 0; i < Width; i++) {
      const int8x8_t lo = vreinterpret_s8_u8(vget_low_u8(src_lines[i]));
      const int8x8_t hi = vreinterpret_s8_u8(vget_high_u8(src_lines[i]));
      sums2[i] = vaddl_s8(lo, hi);
    }
    int16x8_t sums4[Width / 2];
    for (int i = 0; i < Width / 2; i++) {
      sums4[i] = vpaddq_s16(sums2[2 * i], sums2[2 * i + 1]);
    }
    if (Width == 4) {
      int32x4_t sum = vld1q_s32(sums_ptr);
      int16x8_t sums8 = vpaddq_s16(sums4[0], sums4[1]);
      sum = vpadalq_s16(sum, sums8);
      vst1q_s32(sums_ptr, sum);
    } else {
      assert(Width == 2);
      int32x2_t sum = vld1_s32(sums_ptr);
      int16x4_t sums8 =
          vpadd_s16(vget_low_s16(sums4[0]), vget_high_s16(sums4[0]));
      sum = vpadal_s16(sum, sums8);
      vst1_s32(sums_ptr, sum);
    }
    dst->seek_forward_n_cells(1);
  }
};

template <int Width>
using Int8InputsFastKernelFormat =
    KernelSideFormatInt8Inputs<CellFormat<Width, 16, CellOrder::WidthMajor>, 1>;

// Same as above, but for int8 inputs, avoiding the uint8 -> int8 conversion.
template <int Width>
class PackingRegisterBlock<WidthMajorInt8SideMap,
                           PackedSideBlock<Int8InputsFastKernelFormat<Width>>>
    : public PackingRegisterBlockBase<
          WidthMajorInt8SideMap,
          PackedSideBlock<Int8InputsFastKernelFormat<Width>>> {
 public:
  static_assert(Width == 2 || Width == 4, "");
  typedef Int8InputsFastKernelFormat<Width> KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static const int kCells = KernelSideFormat::kCells;
  static const int kCellWidth = CellFormat::kWidth;
  static const int kKernelWidth = CellFormat::kWidth * kCells;
  static const int kCellDepth = CellFormat::kDepth;
  static const int kCellSize = CellFormat::kSize;

  void Pack(PackedSideBlock<KernelSideFormat>* dst, int start_width) {
    std::int32_t* sums_ptr = dst->sums_of_each_slice() + start_width;
    std::int8_t* dst_ptr = reinterpret_cast<std::int8_t*>(dst->current_data());
    const std::int8_t* const src_ptr = this->complete_src_.data();
    const int stride = this->complete_src_.stride();
    // Load source WidthMajor data
    int8x16_t src_lines[Width];
    for (int i = 0; i < Width; i++) {
      src_lines[i] = vld1q_s8(src_ptr + i * stride);
    }
    for (int i = 0; i < Width; i++) {
      vst1q_s8(dst_ptr + 16 * i, src_lines[i]);
    }
    int16x8_t sums2[Width];
    for (int i = 0; i < Width; i++) {
      const int8x8_t lo = vget_low_s8(src_lines[i]);
      const int8x8_t hi = vget_high_s8(src_lines[i]);
      sums2[i] = vaddl_s8(lo, hi);
    }
    int16x8_t sums4[Width / 2];
    for (int i = 0; i < Width / 2; i++) {
      sums4[i] = vpaddq_s16(sums2[2 * i], sums2[2 * i + 1]);
    }
    if (Width == 4) {
      int32x4_t sum = vld1q_s32(sums_ptr);
      int16x8_t sums8 = vpaddq_s16(sums4[0], sums4[1]);
      sum = vpadalq_s16(sum, sums8);
      vst1q_s32(sums_ptr, sum);
    } else {
      assert(Width == 2);
      int32x2_t sum = vld1_s32(sums_ptr);
      int16x4_t sums8 =
          vpadd_s16(vget_low_s16(sums4[0]), vget_high_s16(sums4[0]));
      sum = vpadal_s16(sum, sums8);
      vst1_s32(sums_ptr, sum);
    }
    dst->seek_forward_n_cells(1);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_PACK_NEON_H_
