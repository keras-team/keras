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

// kernel_reference.h: a reference kernel for CPU architectures where we don't
// have optimized kernels yet. Also useful for testing, as it's templatized
// to have any arbitrary format, allowing tests to cover all sorts of corner
// cases.

#ifndef GEMMLOWP_INTERNAL_KERNEL_REFERENCE_H_
#define GEMMLOWP_INTERNAL_KERNEL_REFERENCE_H_

#include "kernel.h"

#include <cstdio>
#include <cstring>

namespace gemmlowp {

// This kernel is templatized in an arbitrary Format template parameter,
// allowing it to have any arbitrary format.
template <typename tFormat>
struct ReferenceKernel : KernelBase {
  typedef tFormat Format;

  const char* Name() const override {
    static char buf[256];
    snprintf(buf, sizeof(buf),
             "reference(Lhs: %d cells %dx%d %s, Rhs: %d cells %dx%d %s)",
             Format::Lhs::kCells, Format::Lhs::Cell::kWidth,
             Format::Lhs::Cell::kDepth,
             CellOrderName(Format::Lhs::Cell::kOrder), Format::Rhs::kCells,
             Format::Rhs::Cell::kDepth, Format::Rhs::Cell::kWidth,
             CellOrderName(Format::Rhs::Cell::kOrder));
    return buf;
  }

  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    std::int32_t accumulator[Format::kRows * Format::kCols];
    memset(accumulator, 0, sizeof(accumulator));

    const int run_depth_cells = static_cast<int>(run_depth / Format::kDepth);

    // The outer loop is over the depth dimension.
    for (int dc = 0; dc < run_depth_cells; dc++) {
      // The next two loops are over cells of the Lhs (stacked vertically),
      // and over cells of the Rhs (stacked horizontally).
      for (int rc = 0; rc < Format::Lhs::kCells; rc++) {
        const std::uint8_t* lhs_cell_ptr =
            lhs_ptr + (dc * Format::Lhs::kCells + rc) *
                          Format::Lhs::Cell::kWidth * Format::kDepth;
        for (int cc = 0; cc < Format::Rhs::kCells; cc++) {
          const std::uint8_t* rhs_cell_ptr =
              rhs_ptr + (dc * Format::Rhs::kCells + cc) *
                            Format::Rhs::Cell::kWidth * Format::kDepth;

          // Now we are inside one cell of the Lhs and inside one cell
          // of the Rhs, so the remaining inner loops are just
          // traditional three loops of matrix multiplication.
          for (int di = 0; di < Format::kDepth; di++) {
            for (int ri = 0; ri < Format::Lhs::Cell::kWidth; ri++) {
              for (int ci = 0; ci < Format::Rhs::Cell::kWidth; ci++) {
                const std::uint8_t* lhs_coeff_ptr =
                    lhs_cell_ptr +
                    OffsetIntoCell<typename Format::Lhs::Cell>(ri, di);
                const std::uint8_t* rhs_coeff_ptr =
                    rhs_cell_ptr +
                    OffsetIntoCell<typename Format::Rhs::Cell>(ci, di);
                std::int32_t* accumulator_coeff_ptr =
                    accumulator + (ri + rc * Format::Lhs::Cell::kWidth) +
                    (ci + cc * Format::Rhs::Cell::kWidth) * Format::kRows;
                *accumulator_coeff_ptr +=
                    std::int32_t(*lhs_coeff_ptr) * std::int32_t(*rhs_coeff_ptr);
              }
            }
          }
        }
      }
    }

    if (start_depth == 0) {
      // start_depth == 0 means we haven't accumulated anything yet, so we need
      // to overwrite the accumulator, as it hasn't been initialized to zero.
      for (int r = 0; r < Format::kRows; r++) {
        for (int c = 0; c < Format::kCols; c++) {
          dst_ptr[r * dst_row_stride + c * dst_col_stride] =
              accumulator[r + c * Format::kRows];
        }
      }
    } else {
      // We have already accumulated stuff, so we need to continue accumulating
      // instead of just overwriting.
      for (int r = 0; r < Format::kRows; r++) {
        for (int c = 0; c < Format::kCols; c++) {
          dst_ptr[r * dst_row_stride + c * dst_col_stride] +=
              accumulator[r + c * Format::kRows];
        }
      }
    }
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_REFERENCE_H_
