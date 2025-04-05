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

#ifndef GEMMLOWP_META_LEGACY_SINGLE_THREAD_GEMM_H_
#define GEMMLOWP_META_LEGACY_SINGLE_THREAD_GEMM_H_

#include "../internal/common.h"

#ifdef GEMMLOWP_NEON

#include "quantized_mul_kernels.h"
#include "single_thread_gemm.h"
#include "streams.h"

namespace gemmlowp {
namespace meta {

void gemm_q8_strided(std::uint8_t* scratch, const std::uint8_t* lhs,
                     const std::uint8_t* rhs, std::int32_t m, std::int32_t n,
                     std::int32_t k, std::int32_t lhs_offset,
                     std::int32_t rhs_offset, std::int32_t result_offset,
                     std::int32_t multiplicative_offset, std::int32_t shift,
                     std::uint8_t* result, std::int32_t result_stride) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_LEGACY_VERBOSE
  std::cout << "Legacy::GemmQ8." << std::endl;
#endif
#endif
  typedef GemmParams<std::uint8_t, std::uint8_t, RowMajorWithSum,
                     RowMajorWithSum, QuantizedStaticPreprocessed, RowMajor>
      Params;
  Params params;

  params.m = m;
  params.n = n;
  params.k = k;

  params.lhs = lhs;
  params.rhs = rhs;
  params.result = result;
  params.scratch = scratch;

  params.left_stream.count = k;
  params.left_stream.stride = k;
  params.left_stream.multiplicative_sum_offset = rhs_offset;
  params.left_stream.additive_sum_offset =
      result_offset + k * lhs_offset * rhs_offset;

  params.right_stream.count = k;
  params.right_stream.stride = k;
  params.right_stream.multiplicative_sum_offset = lhs_offset;
  params.right_stream.additive_sum_offset = 0;

  params.fused_kernel.kernel.multiplicative_offset = multiplicative_offset;
  params.fused_kernel.kernel.rounding_offset = (1 << (shift - 1));
  params.fused_kernel.kernel.shift = -shift;
  params.fused_kernel.kernel.count = k;
  params.fused_kernel.output_stream.stride = result_stride;

  Gemm<GemmExecutorPackRHS, Params, 2, 4, 8>(params);
}

void gemv_q8(std::uint8_t* scratch, const std::uint8_t* lhs,
             const std::uint8_t* rhs, std::int32_t n, std::int32_t k,
             std::int32_t lhs_offset, std::int32_t rhs_offset,
             std::int32_t result_offset, std::int32_t multiplicative_offset,
             std::int32_t shift, std::uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_LEGACY_VERBOSE
  std::cout << "Legacy::GemvQ8." << std::endl;
#endif
#endif
  typedef GemmParams<std::uint8_t, std::uint8_t, RowMajorWithSum,
                     RowMajorWithSum, QuantizedStaticPreprocessed, RowMajor>
      Params;
  Params params;

  params.m = 1;
  params.n = n;
  params.k = k;

  params.lhs = lhs;
  params.rhs = rhs;
  params.result = result;
  params.scratch = scratch;

  params.left_stream.count = k;
  params.left_stream.stride = k;
  params.left_stream.multiplicative_sum_offset = rhs_offset;
  params.left_stream.additive_sum_offset =
      result_offset + k * lhs_offset * rhs_offset;

  params.right_stream.count = k;
  params.right_stream.stride = k;
  params.right_stream.multiplicative_sum_offset = lhs_offset;
  params.right_stream.additive_sum_offset = 0;

  params.fused_kernel.kernel.multiplicative_offset = multiplicative_offset;
  params.fused_kernel.kernel.rounding_offset = (1 << (shift - 1));
  params.fused_kernel.kernel.shift = -shift;
  params.fused_kernel.kernel.count = k;
  params.fused_kernel.output_stream.stride = n;

  if (k < 1536) {
    Gemm<GemmExecutorPackLHS, Params, 1, 8, 8>(params);
  } else {
    Gemm<GemmExecutorPackLHS, Params, 2, 4, 8>(params);
  }
}

void gemm_i32_strided(std::uint8_t* scratch, const std::uint8_t* lhs,
                      const std::uint8_t* rhs, std::int32_t m, std::int32_t n,
                      std::int32_t k, std::int32_t lhs_offset,
                      std::int32_t rhs_offset, std::int32_t* result,
                      std::int32_t result_stride) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_LEGACY_VERBOSE
  std::cout << "Legacy::GemmI32." << std::endl;
#endif
#endif
  typedef GemmParams<std::uint8_t, std::int32_t, RowMajorWithSum,
                     RowMajorWithSum, QuantizedStaticPreprocessedAsInt32,
                     RowMajor>
      Params;
  Params params;

  params.m = m;
  params.n = n;
  params.k = k;

  params.lhs = lhs;
  params.rhs = rhs;
  params.result = result;
  params.scratch = scratch;

  params.left_stream.count = k;
  params.left_stream.stride = k;
  params.left_stream.multiplicative_sum_offset = rhs_offset;
  params.left_stream.additive_sum_offset = k * lhs_offset * rhs_offset;

  params.right_stream.count = k;
  params.right_stream.stride = k;
  params.right_stream.multiplicative_sum_offset = lhs_offset;
  params.right_stream.additive_sum_offset = 0;

  params.fused_kernel.kernel.count = k;
  params.fused_kernel.output_stream.stride = result_stride * 4;

  Gemm<GemmExecutorPackRHS, Params, 2, 4, 8>(params);
}

void gemv_i32(std::uint8_t* scratch, const std::uint8_t* lhs,
              const std::uint8_t* rhs, std::int32_t n, std::int32_t k,
              std::int32_t lhs_offset, std::int32_t rhs_offset,
              std::int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_LEGACY_VERBOSE
  std::cout << "Legacy::GemvI32." << std::endl;
#endif
#endif
  typedef GemmParams<std::uint8_t, std::int32_t, RowMajorWithSum,
                     RowMajorWithSum, QuantizedStaticPreprocessedAsInt32,
                     RowMajor>
      Params;
  Params params;

  params.m = 1;
  params.n = n;
  params.k = k;

  params.lhs = lhs;
  params.rhs = rhs;
  params.result = result;
  params.scratch = scratch;

  params.left_stream.count = k;
  params.left_stream.stride = k;
  params.left_stream.multiplicative_sum_offset = rhs_offset;
  params.left_stream.additive_sum_offset = k * lhs_offset * rhs_offset;

  params.right_stream.count = k;
  params.right_stream.stride = k;
  params.right_stream.multiplicative_sum_offset = lhs_offset;
  params.right_stream.additive_sum_offset = 0;

  params.fused_kernel.kernel.count = k;
  params.fused_kernel.output_stream.stride = 0;

  if (k < 1664) {
    Gemm<GemmExecutorPackLHS, Params, 1, 8, 8>(params);
  } else {
    Gemm<GemmExecutorPackLHS, Params, 1, 6, 8>(params);
  }
}

void gemm_f_strided(std::uint8_t* scratch, const std::uint8_t* lhs,
                    const std::uint8_t* rhs, std::int32_t m, std::int32_t n,
                    std::int32_t k, std::int32_t lhs_offset,
                    std::int32_t rhs_offset, float result_offset, float* result,
                    std::int32_t result_stride) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_LEGACY_VERBOSE
  std::cout << "Legacy::GemmF." << std::endl;
#endif
#endif
  typedef GemmParams<std::uint8_t, float, RowMajorWithSum, RowMajorWithSum,
                     QuantizedStaticPreprocessedAsFloat, RowMajor>
      Params;
  Params params;

  params.m = m;
  params.n = n;
  params.k = k;

  params.lhs = lhs;
  params.rhs = rhs;
  params.result = result;
  params.scratch = scratch;

  params.left_stream.count = k;
  params.left_stream.stride = k;
  params.left_stream.multiplicative_sum_offset = rhs_offset;
  params.left_stream.additive_sum_offset = k * lhs_offset * rhs_offset;

  params.right_stream.count = k;
  params.right_stream.stride = k;
  params.right_stream.multiplicative_sum_offset = lhs_offset;
  params.right_stream.additive_sum_offset = 0;

  params.fused_kernel.kernel.count = k;
  params.fused_kernel.kernel.scale = result_offset;
  params.fused_kernel.output_stream.stride = result_stride * 4;

  Gemm<GemmExecutorPackRHS, Params, 2, 4, 8>(params);
}

void gemv_f(std::uint8_t* scratch, const std::uint8_t* lhs,
            const std::uint8_t* rhs, std::int32_t n, std::int32_t k,
            std::int32_t lhs_offset, std::int32_t rhs_offset,
            float result_offset, float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_LEGACY_VERBOSE
  std::cout << "Legacy::GemvF." << std::endl;
#endif
#endif
  typedef GemmParams<std::uint8_t, float, RowMajorWithSum, RowMajorWithSum,
                     QuantizedStaticPreprocessedAsFloat, RowMajor>
      Params;
  Params params;

  params.m = 1;
  params.n = n;
  params.k = k;

  params.lhs = lhs;
  params.rhs = rhs;
  params.result = result;
  params.scratch = scratch;

  params.left_stream.count = k;
  params.left_stream.stride = k;
  params.left_stream.multiplicative_sum_offset = rhs_offset;
  params.left_stream.additive_sum_offset = k * lhs_offset * rhs_offset;

  params.right_stream.count = k;
  params.right_stream.stride = k;
  params.right_stream.multiplicative_sum_offset = lhs_offset;
  params.right_stream.additive_sum_offset = 0;

  params.fused_kernel.kernel.count = k;
  params.fused_kernel.kernel.scale = result_offset;
  params.fused_kernel.output_stream.stride = 0;

  if (k < 1664) {
    Gemm<GemmExecutorPackLHS, Params, 1, 8, 8>(params);
  } else {
    Gemm<GemmExecutorPackLHS, Params, 1, 6, 8>(params);
  }
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm fast-path requires GEMMLOWP_NEON_(32|64)!"
#endif

#endif  // GEMMLOWP_META_LEGACY_SINGLE_THREAD_GEMM_H_
