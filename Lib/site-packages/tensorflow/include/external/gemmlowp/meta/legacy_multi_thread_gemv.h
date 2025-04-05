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

// multi_thread_gemv.h: Entry point to the multithreaded version of the
// generated (meta) gemv library.

#ifndef GEMMLOWP_META_MULTI_THREAD_GEMV_H_
#define GEMMLOWP_META_MULTI_THREAD_GEMV_H_

#ifdef GEMMLOWP_NEON

#include "legacy_multi_thread_common.h"
#include "legacy_operations_common.h"
#include "legacy_single_thread_gemm.h"

namespace gemmlowp {
namespace meta {
namespace internal {

class GemvQuantized8BitOperation : public Quantized8BitOperation {
 public:
  GemvQuantized8BitOperation(std::int32_t lhs_offset, std::int32_t rhs_offset,
                             std::int32_t sum_offset, std::int32_t multiplier,
                             std::int32_t shift)
      : Quantized8BitOperation(lhs_offset, rhs_offset, sum_offset, multiplier,
                               shift) {}

  void ExecuteMatrixMatrix(std::uint8_t* scratch, const std::uint8_t* lhs,
                           const std::uint8_t* rhs, std::int32_t m,
                           std::int32_t n, std::int32_t k, std::uint8_t* result,
                           std::int32_t result_stride) const {
    gemv_q8(scratch, lhs, rhs, n, k, lhs_offset, rhs_offset, sum_offset,
            multiplier, shift, result);
  }

  static std::int32_t ScratchPerThread(std::int32_t m, std::int32_t n,
                                       std::int32_t k) {
    return 128 * 1024;
  }
};

class GemvFloatOperation : public FloatOperation {
 public:
  GemvFloatOperation(std::int32_t lhs_offset, std::int32_t rhs_offset,
                     float result_offset)
      : FloatOperation(lhs_offset, rhs_offset, result_offset) {}

  void ExecuteMatrixMatrix(std::uint8_t* scratch, const std::uint8_t* lhs,
                           const std::uint8_t* rhs, std::int32_t m,
                           std::int32_t n, std::int32_t k, float* result,
                           std::int32_t result_stride) const {
    gemv_f(scratch, lhs, rhs, n, k, lhs_offset, rhs_offset, result_offset,
           result);
  }

  static std::int32_t ScratchPerThread(std::int32_t m, std::int32_t n,
                                       std::int32_t k) {
    return 128 * 1024;
  }
};

class GemvInt32Operation : public Int32Operation {
 public:
  GemvInt32Operation(std::int32_t lhs_offset, std::int32_t rhs_offset)
      : Int32Operation(lhs_offset, rhs_offset) {}

  void ExecuteMatrixMatrix(std::uint8_t* scratch, const std::uint8_t* lhs,
                           const std::uint8_t* rhs, std::int32_t m,
                           std::int32_t n, std::int32_t k, std::int32_t* result,
                           std::int32_t result_stride) const {
    gemv_i32(scratch, lhs, rhs, n, k, lhs_offset, rhs_offset, result);
  }

  static std::int32_t ScratchPerThread(std::int32_t m, std::int32_t n,
                                       std::int32_t k) {
    return 128 * 1024;
  }
};

}  // namespace internal

std::int32_t gemv_q8_scratch(std::int32_t m, std::int32_t n, std::int32_t k,
                             std::int32_t max_threads) {
  return internal::ResolveMaxThreads(max_threads) *
         internal::GemvQuantized8BitOperation::ScratchPerThread(m, n, k);
}

void multi_thread_gemv_q8(gemmlowp::WorkersPool* pool, std::int32_t max_threads,
                          std::uint8_t* scratch, const std::uint8_t* lhs,
                          const std::uint8_t* rhs, std::int32_t n,
                          std::int32_t k, std::int32_t lhs_offset,
                          std::int32_t rhs_offset, std::int32_t sum_offset,
                          std::int32_t multiplier, std::int32_t shift,
                          std::uint8_t* result) {
  max_threads = internal::ResolveMaxThreads(max_threads);
  internal::GemvQuantized8BitOperation operation(lhs_offset, rhs_offset,
                                                 sum_offset, multiplier, shift);
  if (max_threads == 1) {
    operation.ExecuteMatrixMatrix(scratch, lhs, rhs, 1, n, k, result, n);
  } else {
    internal::MultiThreadedMatrixMatrix(pool, max_threads, scratch, lhs, rhs, 1,
                                        n, k, result, n, operation);
  }
}

std::int32_t gemv_f_scratch(std::int32_t m, std::int32_t n, std::int32_t k,
                            std::int32_t max_threads) {
  return internal::ResolveMaxThreads(max_threads) *
         internal::GemvFloatOperation::ScratchPerThread(m, n, k);
}

void multi_thread_gemv_f(gemmlowp::WorkersPool* pool, std::int32_t max_threads,
                         std::uint8_t* scratch, const std::uint8_t* lhs,
                         const std::uint8_t* rhs, std::int32_t n,
                         std::int32_t k, std::int32_t lhs_offset,
                         std::int32_t rhs_offset, float result_offset,
                         float* result) {
  max_threads = internal::ResolveMaxThreads(max_threads);
  internal::GemvFloatOperation operation(lhs_offset, rhs_offset, result_offset);
  if (max_threads == 1) {
    operation.ExecuteMatrixMatrix(scratch, lhs, rhs, 1, n, k, result, n);
  } else {
    internal::MultiThreadedMatrixMatrix(pool, max_threads, scratch, lhs, rhs, 1,
                                        n, k, result, n, operation);
  }
}

std::int32_t gemv_i32_scratch(std::int32_t m, std::int32_t n, std::int32_t k,
                              std::int32_t max_threads) {
  return internal::ResolveMaxThreads(max_threads) *
         internal::GemvInt32Operation::ScratchPerThread(m, n, k);
}

void multi_thread_gemv_i32(gemmlowp::WorkersPool* pool,
                           std::int32_t max_threads, std::uint8_t* scratch,
                           const std::uint8_t* lhs, const std::uint8_t* rhs,
                           std::int32_t n, std::int32_t k,
                           std::int32_t lhs_offset, std::int32_t rhs_offset,
                           std::int32_t* result) {
  max_threads = internal::ResolveMaxThreads(max_threads);
  internal::GemvInt32Operation operation(lhs_offset, rhs_offset);
  if (max_threads == 1) {
    operation.ExecuteMatrixMatrix(scratch, lhs, rhs, 1, n, k, result, n);
  } else {
    internal::MultiThreadedMatrixMatrix(pool, max_threads, scratch, lhs, rhs, 1,
                                        n, k, result, n, operation);
  }
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm fast-path requires GEMMLOWP_NEON_(32|64)!"
#endif

#endif  // GEMMLOWP_META_MULTI_THREAD_GEMV_H_
