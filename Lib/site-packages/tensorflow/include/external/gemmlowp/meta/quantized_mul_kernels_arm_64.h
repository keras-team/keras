// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
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

#ifndef GEMMLOWP_META_QUANTIZED_MUL_KERNELS_ARM_64_H_
#define GEMMLOWP_META_QUANTIZED_MUL_KERNELS_ARM_64_H_

#ifdef GEMMLOWP_NEON_64

#include <cassert>
#include <cstdint>

namespace gemmlowp {
namespace meta {

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 1,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 1, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v1.2s}, [%x[lhs]], #8\n"
      "ld1 {v2.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v3.8h, v2.8b, v1.8b\n"
      "uadalp v0.4s, v3.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[multiplicative_offset]\n"
      "dup v7.4s, %w[rounding_offset]\n"
      "dup v8.4s, %w[shift]\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "mul v0.4s, v0.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "sshl v0.4s, v0.4s, v8.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtun v0.8b, v0.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.b}[0], [%x[result]], #1\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 2,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 2, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.2s}, [%x[lhs]], #8\n"
      "ld1 {v3.2s, v4.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v3.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v2.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[multiplicative_offset]\n"
      "dup v7.4s, %w[rounding_offset]\n"
      "dup v8.4s, %w[shift]\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "mul v0.4s, v0.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "sshl v0.4s, v0.4s, v8.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtun v0.8b, v0.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.h}[0], [%x[result]], #2\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 3,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 3, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.2s}, [%x[lhs]], #8\n"
      "ld1 {v4.2s, v5.2s, v6.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v4.8b, v3.8b\n"
      "umull v8.8h, v5.8b, v3.8b\n"
      "umull v9.8h, v6.8b, v3.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[multiplicative_offset]\n"
      "dup v7.4s, %w[rounding_offset]\n"
      "dup v8.4s, %w[shift]\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "mul v0.4s, v0.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "sshl v0.4s, v0.4s, v8.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtun v0.8b, v0.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.h}[0], [%x[result]], #2\n"
      "st1 {v0.b}[2], [%x[result]], #1\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc",
        "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 4,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 4, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.2s}, [%x[lhs]], #8\n"
      "ld1 {v5.2s, v6.2s, v7.2s, v8.2s}, [%x[rhs]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v9.8h, v5.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v4.8b\n"
      "umull v11.8h, v7.8b, v4.8b\n"
      "umull v12.8h, v8.8b, v4.8b\n"
      "uadalp v0.4s, v9.8h\n"
      "uadalp v1.4s, v10.8h\n"
      "uadalp v2.4s, v11.8h\n"
      "uadalp v3.4s, v12.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[multiplicative_offset]\n"
      "dup v7.4s, %w[rounding_offset]\n"
      "dup v8.4s, %w[shift]\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "mul v0.4s, v0.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "sshl v0.4s, v0.4s, v8.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtun v0.8b, v0.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 5,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 5, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v5.2s, v6.2s, v7.2s, v8.2s}, [%x[rhs]], #32\n"
      "ld1 {v9.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "umull v11.8h, v6.8b, v9.8b\n"
      "umull v12.8h, v7.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "ld1 {v5.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v10.8h\n"
      "uadalp v1.4s, v11.8h\n"
      "uadalp v2.4s, v12.8h\n"
      "uadalp v3.4s, v13.8h\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "uadalp v4.4s, v10.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v5.4s}, [%x[lhs]], #16\n"
      "ld1 {v6.4s, v7.4s}, [%x[rhs]], #32\n"
      "dup v8.4s, %w[multiplicative_offset]\n"
      "dup v9.4s, %w[rounding_offset]\n"
      "dup v10.4s, %w[shift]\n"
      "dup v5.4s, v5.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v4.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "add v0.4s, v0.4s, v6.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "mul v0.4s, v0.4s, v8.4s\n"
      "mul v1.4s, v1.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v1.4s, v1.4s, v9.4s\n"
      "sshl v0.4s, v0.4s, v10.4s\n"
      "sshl v1.4s, v1.4s, v10.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn2 v0.8h, v1.4s\n"
      "sqxtun v0.8b, v0.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      "st1 {v0.b}[4], [%x[result]], #1\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 6,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 6, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s, v8.2s, v9.2s}, [%x[rhs]], #32\n"
      "ld1 {v10.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v9.8b, v10.8b\n"
      "ld1 {v6.2s, v7.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "uadalp v4.4s, v11.8h\n"
      "uadalp v5.4s, v12.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s, v8.4s}, [%x[rhs]], #32\n"
      "dup v9.4s, %w[multiplicative_offset]\n"
      "dup v10.4s, %w[rounding_offset]\n"
      "dup v11.4s, %w[shift]\n"
      "dup v6.4s, v6.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v4.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v6.4s\n"
      "add v1.4s, v1.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v1.4s, v1.4s, v8.4s\n"
      "mul v0.4s, v0.4s, v9.4s\n"
      "mul v1.4s, v1.4s, v9.4s\n"
      "add v0.4s, v0.4s, v10.4s\n"
      "add v1.4s, v1.4s, v10.4s\n"
      "sshl v0.4s, v0.4s, v11.4s\n"
      "sshl v1.4s, v1.4s, v11.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn2 v0.8h, v1.4s\n"
      "sqxtun v0.8b, v0.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      "st1 {v0.h}[2], [%x[result]], #2\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 7,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 7, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v7.2s, v8.2s, v9.2s, v10.2s}, [%x[rhs]], #32\n"
      "ld1 {v11.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v10.8b, v11.8b\n"
      "ld1 {v7.2s, v8.2s, v9.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v12.8h\n"
      "uadalp v1.4s, v13.8h\n"
      "uadalp v2.4s, v14.8h\n"
      "uadalp v3.4s, v15.8h\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "uadalp v4.4s, v12.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v7.4s}, [%x[lhs]], #16\n"
      "ld1 {v8.4s, v9.4s}, [%x[rhs]], #32\n"
      "dup v10.4s, %w[multiplicative_offset]\n"
      "dup v11.4s, %w[rounding_offset]\n"
      "dup v12.4s, %w[shift]\n"
      "dup v7.4s, v7.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v6.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v7.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "add v0.4s, v0.4s, v8.4s\n"
      "add v1.4s, v1.4s, v9.4s\n"
      "mul v0.4s, v0.4s, v10.4s\n"
      "mul v1.4s, v1.4s, v10.4s\n"
      "add v0.4s, v0.4s, v11.4s\n"
      "add v1.4s, v1.4s, v11.4s\n"
      "sshl v0.4s, v0.4s, v12.4s\n"
      "sshl v1.4s, v1.4s, v12.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn2 v0.8h, v1.4s\n"
      "sqxtun v0.8b, v0.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      "st1 {v0.h}[2], [%x[result]], #2\n"
      "st1 {v0.b}[6], [%x[result]], #1\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 1, 8,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 1, 8, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // 1x8 lanes loop.
      "1:"

      "ld1 {v9.2s, v10.2s, v11.2s, v12.2s}, [%x[rhs]], #32\n"
      "ld1 {v8.2s}, [%x[lhs]], #8\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "umull v14.8h, v8.8b, v10.8b\n"
      "umull v15.8h, v8.8b, v11.8b\n"
      "umull v16.8h, v8.8b, v12.8b\n"
      "ld1 {v9.2s, v10.2s, v11.2s, v12.2s}, [%x[rhs]], #32\n"
      "uadalp v0.4s, v13.8h\n"
      "uadalp v1.4s, v14.8h\n"
      "uadalp v2.4s, v15.8h\n"
      "uadalp v3.4s, v16.8h\n"
      "prfm pldl1keep, [%x[rhs], #256]\n"
      "umull v17.8h, v8.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v8.8b, v11.8b\n"
      "umull v15.8h, v8.8b, v12.8b\n"
      "prfm pldl1keep, [%x[lhs], #32]\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "uadalp v4.4s, v17.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"
      "uadalp v7.4s, v15.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v8.4s}, [%x[lhs]], #16\n"
      "ld1 {v9.4s, v10.4s}, [%x[rhs]], #32\n"
      "dup v11.4s, %w[multiplicative_offset]\n"
      "dup v12.4s, %w[rounding_offset]\n"
      "dup v13.4s, %w[shift]\n"
      "dup v8.4s, v8.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v6.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v8.4s\n"
      "add v1.4s, v1.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v1.4s, v1.4s, v10.4s\n"
      "mul v0.4s, v0.4s, v11.4s\n"
      "mul v1.4s, v1.4s, v11.4s\n"
      "add v0.4s, v0.4s, v12.4s\n"
      "add v1.4s, v1.4s, v12.4s\n"
      "sshl v0.4s, v0.4s, v13.4s\n"
      "sshl v1.4s, v1.4s, v13.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn2 v0.8h, v1.4s\n"
      "sqxtun v0.8b, v0.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 2, 1,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 2, 1, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.2s, v3.2s}, [%x[lhs]], #16\n"
      "ld1 {v4.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v4.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v3.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[multiplicative_offset]\n"
      "dup v7.4s, %w[rounding_offset]\n"
      "dup v8.4s, %w[shift]\n"
      "dup v2.4s, v4.s[0]\n"
      "dup v4.4s, v4.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v2.4s\n"
      "add v1.4s, v1.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "mul v0.4s, v0.4s, v6.4s\n"
      "mul v1.4s, v1.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "sshl v0.4s, v0.4s, v8.4s\n"
      "sshl v1.4s, v1.4s, v8.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn v1.4h, v1.4s\n"
      "sqxtun v0.8b, v0.8h\n"
      "sqxtun v1.8b, v1.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.b}[0], [%x[result]], #1\n"
      "st1 {v1.b}[0], [x0], #1\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 2, 2,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 2, 2, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.2s, v5.2s}, [%x[lhs]], #16\n"
      "ld1 {v6.2s, v7.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v7.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v5.8b\n"
      "umull v11.8h, v7.8b, v5.8b\n"
      "uadalp v0.4s, v8.8h\n"
      "uadalp v1.4s, v9.8h\n"
      "uadalp v2.4s, v10.8h\n"
      "uadalp v3.4s, v11.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[multiplicative_offset]\n"
      "dup v7.4s, %w[rounding_offset]\n"
      "dup v8.4s, %w[shift]\n"
      "dup v9.4s, v4.s[0]\n"
      "dup v4.4s, v4.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v9.4s\n"
      "add v2.4s, v2.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v2.4s, v2.4s, v5.4s\n"
      "mul v0.4s, v0.4s, v6.4s\n"
      "mul v2.4s, v2.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v2.4s, v2.4s, v7.4s\n"
      "sshl v0.4s, v0.4s, v8.4s\n"
      "sshl v2.4s, v2.4s, v8.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn v2.4h, v2.4s\n"
      "sqxtun v0.8b, v0.8h\n"
      "sqxtun v2.8b, v2.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.h}[0], [%x[result]], #2\n"
      "st1 {v2.h}[0], [x0], #2\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 2, 3,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 2, 3, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s}, [%x[lhs]], #16\n"
      "ld1 {v8.2s, v9.2s, v10.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v8.8b, v6.8b\n"
      "umull v12.8h, v9.8b, v6.8b\n"
      "umull v13.8h, v10.8b, v6.8b\n"
      "umull v14.8h, v8.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v7.8b\n"
      "umull v16.8h, v10.8b, v7.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s}, [%x[rhs]], #16\n"
      "dup v8.4s, %w[multiplicative_offset]\n"
      "dup v9.4s, %w[rounding_offset]\n"
      "dup v10.4s, %w[shift]\n"
      "dup v11.4s, v6.s[0]\n"
      "dup v6.4s, v6.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v11.4s\n"
      "add v3.4s, v3.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v3.4s, v3.4s, v7.4s\n"
      "mul v0.4s, v0.4s, v8.4s\n"
      "mul v3.4s, v3.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v3.4s, v3.4s, v9.4s\n"
      "sshl v0.4s, v0.4s, v10.4s\n"
      "sshl v3.4s, v3.4s, v10.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn v3.4h, v3.4s\n"
      "sqxtun v0.8b, v0.8h\n"
      "sqxtun v3.8b, v3.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.h}[0], [%x[result]], #2\n"
      "st1 {v0.b}[2], [%x[result]], #1\n"
      "st1 {v3.h}[0], [x0], #2\n"
      "st1 {v3.b}[2], [x0], #1\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 2, 4,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 2, 4, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // 2x4 lanes loop.
      "1:"

      "ld1 {v10.8b, v11.8b, v12.8b, v13.8b}, [%x[rhs]], #32\n"
      "ld1 {v8.8b}, [%x[lhs]], #8\n"
      "umull v14.8h, v8.8b, v10.8b\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v8.8b, v11.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v16.8h, v8.8b, v12.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v17.8h, v8.8b, v13.8b\n"
      "umull v18.8h, v9.8b, v10.8b\n"
      "uadalp v0.4s, v14.8h\n"
      "uadalp v1.4s, v15.8h\n"
      "uadalp v2.4s, v16.8h\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "umull v16.8h, v9.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "uadalp v3.4s, v17.8h\n"
      "uadalp v4.4s, v18.8h\n"
      "uadalp v5.4s, v14.8h\n"
      "uadalp v6.4s, v15.8h\n"
      "uadalp v7.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v8.4s}, [%x[lhs]], #16\n"
      "ld1 {v9.4s}, [%x[rhs]], #16\n"
      "dup v10.4s, %w[multiplicative_offset]\n"
      "dup v11.4s, %w[rounding_offset]\n"
      "dup v12.4s, %w[shift]\n"
      "dup v13.4s, v8.s[0]\n"
      "dup v8.4s, v8.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v4.4s, v4.4s, v6.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v13.4s\n"
      "add v4.4s, v4.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v4.4s, v4.4s, v9.4s\n"
      "mul v0.4s, v0.4s, v10.4s\n"
      "mul v4.4s, v4.4s, v10.4s\n"
      "add v0.4s, v0.4s, v11.4s\n"
      "add v4.4s, v4.4s, v11.4s\n"
      "sshl v0.4s, v0.4s, v12.4s\n"
      "sshl v4.4s, v4.4s, v12.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v0.8b, v0.8h\n"
      "sqxtun v4.8b, v4.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      "st1 {v4.s}[0], [x0], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 3, 1,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 3, 1, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.2s, v4.2s, v5.2s}, [%x[lhs]], #24\n"
      "ld1 {v6.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v6.8b, v3.8b\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v6.8b, v5.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[multiplicative_offset]\n"
      "dup v7.4s, %w[rounding_offset]\n"
      "dup v8.4s, %w[shift]\n"
      "dup v3.4s, v4.s[0]\n"
      "dup v9.4s, v4.s[1]\n"
      "dup v4.4s, v4.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v3.4s\n"
      "add v1.4s, v1.4s, v9.4s\n"
      "add v2.4s, v2.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "add v2.4s, v2.4s, v5.4s\n"
      "mul v0.4s, v0.4s, v6.4s\n"
      "mul v1.4s, v1.4s, v6.4s\n"
      "mul v2.4s, v2.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "add v2.4s, v2.4s, v7.4s\n"
      "sshl v0.4s, v0.4s, v8.4s\n"
      "sshl v1.4s, v1.4s, v8.4s\n"
      "sshl v2.4s, v2.4s, v8.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn v1.4h, v1.4s\n"
      "sqxtn v2.4h, v2.4s\n"
      "sqxtun v0.8b, v0.8h\n"
      "sqxtun v1.8b, v1.8h\n"
      "sqxtun v2.8b, v2.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.b}[0], [%x[result]], #1\n"
      "st1 {v1.b}[0], [x0], #1\n"
      "st1 {v2.b}[0], [x1], #1\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 3, 2,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 3, 2, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s, v8.2s}, [%x[lhs]], #24\n"
      "ld1 {v9.2s, v10.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v9.8b, v6.8b\n"
      "umull v12.8h, v10.8b, v6.8b\n"
      "umull v13.8h, v9.8b, v7.8b\n"
      "umull v14.8h, v10.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v8.8b\n"
      "umull v16.8h, v10.8b, v8.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s}, [%x[rhs]], #16\n"
      "dup v8.4s, %w[multiplicative_offset]\n"
      "dup v9.4s, %w[rounding_offset]\n"
      "dup v10.4s, %w[shift]\n"
      "dup v11.4s, v6.s[0]\n"
      "dup v12.4s, v6.s[1]\n"
      "dup v6.4s, v6.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v11.4s\n"
      "add v2.4s, v2.4s, v12.4s\n"
      "add v4.4s, v4.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v2.4s, v2.4s, v7.4s\n"
      "add v4.4s, v4.4s, v7.4s\n"
      "mul v0.4s, v0.4s, v8.4s\n"
      "mul v2.4s, v2.4s, v8.4s\n"
      "mul v4.4s, v4.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v2.4s, v2.4s, v9.4s\n"
      "add v4.4s, v4.4s, v9.4s\n"
      "sshl v0.4s, v0.4s, v10.4s\n"
      "sshl v2.4s, v2.4s, v10.4s\n"
      "sshl v4.4s, v4.4s, v10.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn v2.4h, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v0.8b, v0.8h\n"
      "sqxtun v2.8b, v2.8h\n"
      "sqxtun v4.8b, v4.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.h}[0], [%x[result]], #2\n"
      "st1 {v2.h}[0], [x0], #2\n"
      "st1 {v4.h}[0], [x1], #2\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

template <>
inline void
MulKernel<uint8_t, uint8_t, QuantizedStaticPreprocessed, RowMajor, 3, 3,
          8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                       const FusedKernelParams<QuantizedStaticPreprocessed,
                                               RowMajor>& params,
                       uint8_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedRowMajor<uint8_t, uint8_t, "
               "QuantizedStaticPreprocessed, RowMajor, 3, 3, 8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"
      "mov v8.16b, v5.16b\n"

      // 3x3 lanes loop.
      "1:"

      "ld1 {v12.8b, v13.8b, v14.8b}, [%x[rhs]], #24\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "umull v16.8h, v9.8b, v13.8b\n"
      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "umull v17.8h, v9.8b, v14.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v18.8h, v10.8b, v12.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "uadalp v0.4s, v15.8h\n"
      "uadalp v1.4s, v16.8h\n"
      "uadalp v2.4s, v17.8h\n"
      "uadalp v3.4s, v18.8h\n"
      "umull v15.8h, v10.8b, v13.8b\n"
      "umull v16.8h, v10.8b, v14.8b\n"
      "umull v17.8h, v11.8b, v12.8b\n"
      "umull v18.8h, v11.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "umull v9.8h, v11.8b, v14.8b\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"
      "uadalp v6.4s, v17.8h\n"
      "uadalp v7.4s, v18.8h\n"
      "uadalp v8.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantization::Prepare
      "ld1 {v9.4s}, [%x[lhs]], #16\n"
      "ld1 {v10.4s}, [%x[rhs]], #16\n"
      "dup v11.4s, %w[multiplicative_offset]\n"
      "dup v12.4s, %w[rounding_offset]\n"
      "dup v13.4s, %w[shift]\n"
      "dup v14.4s, v9.s[0]\n"
      "dup v15.4s, v9.s[1]\n"
      "dup v9.4s, v9.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v6.4s, v6.4s, v8.4s\n"

      // StaticQuantization::Transform
      "add v0.4s, v0.4s, v14.4s\n"
      "add v3.4s, v3.4s, v15.4s\n"
      "add v6.4s, v6.4s, v9.4s\n"
      "add v0.4s, v0.4s, v10.4s\n"
      "add v3.4s, v3.4s, v10.4s\n"
      "add v6.4s, v6.4s, v10.4s\n"
      "mul v0.4s, v0.4s, v11.4s\n"
      "mul v3.4s, v3.4s, v11.4s\n"
      "mul v6.4s, v6.4s, v11.4s\n"
      "add v0.4s, v0.4s, v12.4s\n"
      "add v3.4s, v3.4s, v12.4s\n"
      "add v6.4s, v6.4s, v12.4s\n"
      "sshl v0.4s, v0.4s, v13.4s\n"
      "sshl v3.4s, v3.4s, v13.4s\n"
      "sshl v6.4s, v6.4s, v13.4s\n"
      "sqxtn v0.4h, v0.4s\n"
      "sqxtn v3.4h, v3.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtun v0.8b, v0.8h\n"
      "sqxtun v3.8b, v3.8h\n"
      "sqxtun v6.8b, v6.8h\n"

      // RowMajorOutput::Output
      "st1 {v0.h}[0], [%x[result]], #2\n"
      "st1 {v0.b}[2], [%x[result]], #1\n"
      "st1 {v3.h}[0], [x0], #2\n"
      "st1 {v3.b}[2], [x0], #1\n"
      "st1 {v6.h}[0], [x1], #2\n"
      "st1 {v6.b}[2], [x1], #1\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [multiplicative_offset] "r"(params.kernel.multiplicative_offset),
        [shift] "r"(params.kernel.shift),
        [stride] "r"(params.output_stream.stride),
        [rounding_offset] "r"(params.kernel.rounding_offset)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v1.2s}, [%x[lhs]], #8\n"
      "ld1 {v2.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v3.8h, v2.8b, v1.8b\n"
      "uadalp v0.4s, v3.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.2s}, [%x[lhs]], #8\n"
      "ld1 {v3.2s, v4.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v3.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v2.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.2s}, [%x[lhs]], #8\n"
      "ld1 {v4.2s, v5.2s, v6.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v4.8b, v3.8b\n"
      "umull v8.8h, v5.8b, v3.8b\n"
      "umull v9.8h, v6.8b, v3.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 4,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 4, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.2s}, [%x[lhs]], #8\n"
      "ld1 {v5.2s, v6.2s, v7.2s, v8.2s}, [%x[rhs]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v9.8h, v5.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v4.8b\n"
      "umull v11.8h, v7.8b, v4.8b\n"
      "umull v12.8h, v8.8b, v4.8b\n"
      "uadalp v0.4s, v9.8h\n"
      "uadalp v1.4s, v10.8h\n"
      "uadalp v2.4s, v11.8h\n"
      "uadalp v3.4s, v12.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 5,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 5, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v5.2s, v6.2s, v7.2s, v8.2s}, [%x[rhs]], #32\n"
      "ld1 {v9.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "umull v11.8h, v6.8b, v9.8b\n"
      "umull v12.8h, v7.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "ld1 {v5.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v10.8h\n"
      "uadalp v1.4s, v11.8h\n"
      "uadalp v2.4s, v12.8h\n"
      "uadalp v3.4s, v13.8h\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "uadalp v4.4s, v10.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v5.4s}, [%x[lhs]], #16\n"
      "ld1 {v6.4s, v7.4s}, [%x[rhs]], #32\n"
      "dup v5.4s, v5.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v4.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "add v0.4s, v0.4s, v6.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      "st1 {v1.s}[0], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 6,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 6, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s, v8.2s, v9.2s}, [%x[rhs]], #32\n"
      "ld1 {v10.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v9.8b, v10.8b\n"
      "ld1 {v6.2s, v7.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "uadalp v4.4s, v11.8h\n"
      "uadalp v5.4s, v12.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s, v8.4s}, [%x[rhs]], #32\n"
      "dup v6.4s, v6.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v4.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v6.4s\n"
      "add v1.4s, v1.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v1.4s, v1.4s, v8.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      "st1 {v1.2s}, [%x[result]], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 7,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 7, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v7.2s, v8.2s, v9.2s, v10.2s}, [%x[rhs]], #32\n"
      "ld1 {v11.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v10.8b, v11.8b\n"
      "ld1 {v7.2s, v8.2s, v9.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v12.8h\n"
      "uadalp v1.4s, v13.8h\n"
      "uadalp v2.4s, v14.8h\n"
      "uadalp v3.4s, v15.8h\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "uadalp v4.4s, v12.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v7.4s}, [%x[lhs]], #16\n"
      "ld1 {v8.4s, v9.4s}, [%x[rhs]], #32\n"
      "dup v7.4s, v7.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v6.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v7.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "add v0.4s, v0.4s, v8.4s\n"
      "add v1.4s, v1.4s, v9.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      "st1 {v1.2s}, [%x[result]], #8\n"
      "st1 {v1.s}[2], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 8,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 1, 8, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // 1x8 lanes loop.
      "1:"

      "ld1 {v9.2s, v10.2s, v11.2s, v12.2s}, [%x[rhs]], #32\n"
      "ld1 {v8.2s}, [%x[lhs]], #8\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "umull v14.8h, v8.8b, v10.8b\n"
      "umull v15.8h, v8.8b, v11.8b\n"
      "umull v16.8h, v8.8b, v12.8b\n"
      "ld1 {v9.2s, v10.2s, v11.2s, v12.2s}, [%x[rhs]], #32\n"
      "uadalp v0.4s, v13.8h\n"
      "uadalp v1.4s, v14.8h\n"
      "uadalp v2.4s, v15.8h\n"
      "uadalp v3.4s, v16.8h\n"
      "prfm pldl1keep, [%x[rhs], #256]\n"
      "umull v17.8h, v8.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v8.8b, v11.8b\n"
      "umull v15.8h, v8.8b, v12.8b\n"
      "prfm pldl1keep, [%x[lhs], #32]\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "uadalp v4.4s, v17.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"
      "uadalp v7.4s, v15.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v8.4s}, [%x[lhs]], #16\n"
      "ld1 {v9.4s, v10.4s}, [%x[rhs]], #32\n"
      "dup v8.4s, v8.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v6.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v8.4s\n"
      "add v1.4s, v1.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v1.4s, v1.4s, v10.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s, v1.4s}, [%x[result]], #32\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.2s, v3.2s}, [%x[lhs]], #16\n"
      "ld1 {v4.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v4.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v3.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v2.4s, v4.s[0]\n"
      "dup v4.4s, v4.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v2.4s\n"
      "add v1.4s, v1.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      "st1 {v1.s}[0], [x0], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.2s, v5.2s}, [%x[lhs]], #16\n"
      "ld1 {v6.2s, v7.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v7.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v5.8b\n"
      "umull v11.8h, v7.8b, v5.8b\n"
      "uadalp v0.4s, v8.8h\n"
      "uadalp v1.4s, v9.8h\n"
      "uadalp v2.4s, v10.8h\n"
      "uadalp v3.4s, v11.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, v4.s[0]\n"
      "dup v4.4s, v4.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v6.4s\n"
      "add v2.4s, v2.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v2.4s, v2.4s, v5.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v2.2s}, [x0], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s}, [%x[lhs]], #16\n"
      "ld1 {v8.2s, v9.2s, v10.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v8.8b, v6.8b\n"
      "umull v12.8h, v9.8b, v6.8b\n"
      "umull v13.8h, v10.8b, v6.8b\n"
      "umull v14.8h, v8.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v7.8b\n"
      "umull v16.8h, v10.8b, v7.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s}, [%x[rhs]], #16\n"
      "dup v8.4s, v6.s[0]\n"
      "dup v6.4s, v6.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v8.4s\n"
      "add v3.4s, v3.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v3.4s, v3.4s, v7.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], #4\n"
      "st1 {v3.2s}, [x0], #8\n"
      "st1 {v3.s}[2], [x0], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 4,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 2, 4, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // 2x4 lanes loop.
      "1:"

      "ld1 {v10.8b, v11.8b, v12.8b, v13.8b}, [%x[rhs]], #32\n"
      "ld1 {v8.8b}, [%x[lhs]], #8\n"
      "umull v14.8h, v8.8b, v10.8b\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v8.8b, v11.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v16.8h, v8.8b, v12.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v17.8h, v8.8b, v13.8b\n"
      "umull v18.8h, v9.8b, v10.8b\n"
      "uadalp v0.4s, v14.8h\n"
      "uadalp v1.4s, v15.8h\n"
      "uadalp v2.4s, v16.8h\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "umull v16.8h, v9.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "uadalp v3.4s, v17.8h\n"
      "uadalp v4.4s, v18.8h\n"
      "uadalp v5.4s, v14.8h\n"
      "uadalp v6.4s, v15.8h\n"
      "uadalp v7.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v8.4s}, [%x[lhs]], #16\n"
      "ld1 {v9.4s}, [%x[rhs]], #16\n"
      "dup v10.4s, v8.s[0]\n"
      "dup v8.4s, v8.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v4.4s, v4.4s, v6.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v10.4s\n"
      "add v4.4s, v4.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v4.4s, v4.4s, v9.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      "st1 {v4.4s}, [x0], #16\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.2s, v4.2s, v5.2s}, [%x[lhs]], #24\n"
      "ld1 {v6.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v6.8b, v3.8b\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v6.8b, v5.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v3.4s, v4.s[0]\n"
      "dup v6.4s, v4.s[1]\n"
      "dup v4.4s, v4.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v3.4s\n"
      "add v1.4s, v1.4s, v6.4s\n"
      "add v2.4s, v2.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "add v2.4s, v2.4s, v5.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      "st1 {v1.s}[0], [x0], #4\n"
      "st1 {v2.s}[0], [x1], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s, v8.2s}, [%x[lhs]], #24\n"
      "ld1 {v9.2s, v10.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v9.8b, v6.8b\n"
      "umull v12.8h, v10.8b, v6.8b\n"
      "umull v13.8h, v9.8b, v7.8b\n"
      "umull v14.8h, v10.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v8.8b\n"
      "umull v16.8h, v10.8b, v8.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s}, [%x[rhs]], #16\n"
      "dup v8.4s, v6.s[0]\n"
      "dup v9.4s, v6.s[1]\n"
      "dup v6.4s, v6.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v8.4s\n"
      "add v2.4s, v2.4s, v9.4s\n"
      "add v4.4s, v4.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v2.4s, v2.4s, v7.4s\n"
      "add v4.4s, v4.4s, v7.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v2.2s}, [x0], #8\n"
      "st1 {v4.2s}, [x1], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, int32_t, QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsInt32,
                                         RowMajor>& params,
                 int32_t* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsInt32RowMajor<uint8_t, int32_t, "
               "QuantizedStaticPreprocessedAsInt32, RowMajor, 3, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"
      "mov v8.16b, v5.16b\n"

      // 3x3 lanes loop.
      "1:"

      "ld1 {v12.8b, v13.8b, v14.8b}, [%x[rhs]], #24\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "umull v16.8h, v9.8b, v13.8b\n"
      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "umull v17.8h, v9.8b, v14.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v18.8h, v10.8b, v12.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "uadalp v0.4s, v15.8h\n"
      "uadalp v1.4s, v16.8h\n"
      "uadalp v2.4s, v17.8h\n"
      "uadalp v3.4s, v18.8h\n"
      "umull v15.8h, v10.8b, v13.8b\n"
      "umull v16.8h, v10.8b, v14.8b\n"
      "umull v17.8h, v11.8b, v12.8b\n"
      "umull v18.8h, v11.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "umull v9.8h, v11.8b, v14.8b\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"
      "uadalp v6.4s, v17.8h\n"
      "uadalp v7.4s, v18.8h\n"
      "uadalp v8.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationInt32::Prepare
      "ld1 {v9.4s}, [%x[lhs]], #16\n"
      "ld1 {v10.4s}, [%x[rhs]], #16\n"
      "dup v11.4s, v9.s[0]\n"
      "dup v12.4s, v9.s[1]\n"
      "dup v9.4s, v9.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v6.4s, v6.4s, v8.4s\n"

      // StaticQuantizationInt32::Transform
      "add v0.4s, v0.4s, v11.4s\n"
      "add v3.4s, v3.4s, v12.4s\n"
      "add v6.4s, v6.4s, v9.4s\n"
      "add v0.4s, v0.4s, v10.4s\n"
      "add v3.4s, v3.4s, v10.4s\n"
      "add v6.4s, v6.4s, v10.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], #4\n"
      "st1 {v3.2s}, [x0], #8\n"
      "st1 {v3.s}[2], [x0], #4\n"
      "st1 {v6.2s}, [x1], #8\n"
      "st1 {v6.s}[2], [x1], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v1.2s}, [%x[lhs]], #8\n"
      "ld1 {v2.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v3.8h, v2.8b, v1.8b\n"
      "uadalp v0.4s, v3.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[scale]\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.2s}, [%x[lhs]], #8\n"
      "ld1 {v3.2s, v4.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v3.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v2.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[scale]\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.2s}, [%x[lhs]], #8\n"
      "ld1 {v4.2s, v5.2s, v6.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v4.8b, v3.8b\n"
      "umull v8.8h, v5.8b, v3.8b\n"
      "umull v9.8h, v6.8b, v3.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[scale]\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc",
        "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 4,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 4, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.2s}, [%x[lhs]], #8\n"
      "ld1 {v5.2s, v6.2s, v7.2s, v8.2s}, [%x[rhs]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v9.8h, v5.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v4.8b\n"
      "umull v11.8h, v7.8b, v4.8b\n"
      "umull v12.8h, v8.8b, v4.8b\n"
      "uadalp v0.4s, v9.8h\n"
      "uadalp v1.4s, v10.8h\n"
      "uadalp v2.4s, v11.8h\n"
      "uadalp v3.4s, v12.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[scale]\n"
      "dup v4.4s, v4.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 5,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 5, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v5.2s, v6.2s, v7.2s, v8.2s}, [%x[rhs]], #32\n"
      "ld1 {v9.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "umull v11.8h, v6.8b, v9.8b\n"
      "umull v12.8h, v7.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "ld1 {v5.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v10.8h\n"
      "uadalp v1.4s, v11.8h\n"
      "uadalp v2.4s, v12.8h\n"
      "uadalp v3.4s, v13.8h\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "uadalp v4.4s, v10.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v5.4s}, [%x[lhs]], #16\n"
      "ld1 {v6.4s, v7.4s}, [%x[rhs]], #32\n"
      "dup v8.4s, %w[scale]\n"
      "dup v5.4s, v5.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v4.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "add v0.4s, v0.4s, v6.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v1.4s, v1.4s\n"
      "fmul v0.4s, v0.4s, v8.4s\n"
      "fmul v1.4s, v1.4s, v8.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      "st1 {v1.s}[0], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 6,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 6, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s, v8.2s, v9.2s}, [%x[rhs]], #32\n"
      "ld1 {v10.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v9.8b, v10.8b\n"
      "ld1 {v6.2s, v7.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "uadalp v4.4s, v11.8h\n"
      "uadalp v5.4s, v12.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s, v8.4s}, [%x[rhs]], #32\n"
      "dup v9.4s, %w[scale]\n"
      "dup v6.4s, v6.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v4.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v6.4s\n"
      "add v1.4s, v1.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v1.4s, v1.4s, v8.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v1.4s, v1.4s\n"
      "fmul v0.4s, v0.4s, v9.4s\n"
      "fmul v1.4s, v1.4s, v9.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      "st1 {v1.2s}, [%x[result]], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 7,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 7, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v7.2s, v8.2s, v9.2s, v10.2s}, [%x[rhs]], #32\n"
      "ld1 {v11.2s}, [%x[lhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v10.8b, v11.8b\n"
      "ld1 {v7.2s, v8.2s, v9.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[rhs], #128]\n"
      "uadalp v0.4s, v12.8h\n"
      "uadalp v1.4s, v13.8h\n"
      "uadalp v2.4s, v14.8h\n"
      "uadalp v3.4s, v15.8h\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "uadalp v4.4s, v12.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v7.4s}, [%x[lhs]], #16\n"
      "ld1 {v8.4s, v9.4s}, [%x[rhs]], #32\n"
      "dup v10.4s, %w[scale]\n"
      "dup v7.4s, v7.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v6.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v7.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "add v0.4s, v0.4s, v8.4s\n"
      "add v1.4s, v1.4s, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v1.4s, v1.4s\n"
      "fmul v0.4s, v0.4s, v10.4s\n"
      "fmul v1.4s, v1.4s, v10.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      "st1 {v1.2s}, [%x[result]], #8\n"
      "st1 {v1.s}[2], [%x[result]], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 8,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 1, 8, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // 1x8 lanes loop.
      "1:"

      "ld1 {v9.2s, v10.2s, v11.2s, v12.2s}, [%x[rhs]], #32\n"
      "ld1 {v8.2s}, [%x[lhs]], #8\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "umull v14.8h, v8.8b, v10.8b\n"
      "umull v15.8h, v8.8b, v11.8b\n"
      "umull v16.8h, v8.8b, v12.8b\n"
      "ld1 {v9.2s, v10.2s, v11.2s, v12.2s}, [%x[rhs]], #32\n"
      "uadalp v0.4s, v13.8h\n"
      "uadalp v1.4s, v14.8h\n"
      "uadalp v2.4s, v15.8h\n"
      "uadalp v3.4s, v16.8h\n"
      "prfm pldl1keep, [%x[rhs], #256]\n"
      "umull v17.8h, v8.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v8.8b, v11.8b\n"
      "umull v15.8h, v8.8b, v12.8b\n"
      "prfm pldl1keep, [%x[lhs], #32]\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "uadalp v4.4s, v17.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"
      "uadalp v7.4s, v15.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v8.4s}, [%x[lhs]], #16\n"
      "ld1 {v9.4s, v10.4s}, [%x[rhs]], #32\n"
      "dup v11.4s, %w[scale]\n"
      "dup v8.4s, v8.s[0]\n"

      // RowMajorOutput::Prepare

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v1.4s, v4.4s, v6.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v8.4s\n"
      "add v1.4s, v1.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v1.4s, v1.4s, v10.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v1.4s, v1.4s\n"
      "fmul v0.4s, v0.4s, v11.4s\n"
      "fmul v1.4s, v1.4s, v11.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s, v1.4s}, [%x[result]], #32\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.2s, v3.2s}, [%x[lhs]], #16\n"
      "ld1 {v4.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v4.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v3.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[scale]\n"
      "dup v2.4s, v4.s[0]\n"
      "dup v4.4s, v4.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v2.4s\n"
      "add v1.4s, v1.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v1.4s, v1.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"
      "fmul v1.4s, v1.4s, v6.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      "st1 {v1.s}[0], [x0], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.2s, v5.2s}, [%x[lhs]], #16\n"
      "ld1 {v6.2s, v7.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v7.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v5.8b\n"
      "umull v11.8h, v7.8b, v5.8b\n"
      "uadalp v0.4s, v8.8h\n"
      "uadalp v1.4s, v9.8h\n"
      "uadalp v2.4s, v10.8h\n"
      "uadalp v3.4s, v11.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[scale]\n"
      "dup v7.4s, v4.s[0]\n"
      "dup v4.4s, v4.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v7.4s\n"
      "add v2.4s, v2.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v2.4s, v2.4s, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v2.4s, v2.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"
      "fmul v2.4s, v2.4s, v6.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v2.2s}, [x0], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s}, [%x[lhs]], #16\n"
      "ld1 {v8.2s, v9.2s, v10.2s}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v8.8b, v6.8b\n"
      "umull v12.8h, v9.8b, v6.8b\n"
      "umull v13.8h, v10.8b, v6.8b\n"
      "umull v14.8h, v8.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v7.8b\n"
      "umull v16.8h, v10.8b, v7.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s}, [%x[rhs]], #16\n"
      "dup v8.4s, %w[scale]\n"
      "dup v9.4s, v6.s[0]\n"
      "dup v6.4s, v6.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v9.4s\n"
      "add v3.4s, v3.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v3.4s, v3.4s, v7.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v3.4s, v3.4s\n"
      "fmul v0.4s, v0.4s, v8.4s\n"
      "fmul v3.4s, v3.4s, v8.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], #4\n"
      "st1 {v3.2s}, [x0], #8\n"
      "st1 {v3.s}[2], [x0], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 4,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 2, 4, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // 2x4 lanes loop.
      "1:"

      "ld1 {v10.8b, v11.8b, v12.8b, v13.8b}, [%x[rhs]], #32\n"
      "ld1 {v8.8b}, [%x[lhs]], #8\n"
      "umull v14.8h, v8.8b, v10.8b\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v8.8b, v11.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v16.8h, v8.8b, v12.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v17.8h, v8.8b, v13.8b\n"
      "umull v18.8h, v9.8b, v10.8b\n"
      "uadalp v0.4s, v14.8h\n"
      "uadalp v1.4s, v15.8h\n"
      "uadalp v2.4s, v16.8h\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "umull v16.8h, v9.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "uadalp v3.4s, v17.8h\n"
      "uadalp v4.4s, v18.8h\n"
      "uadalp v5.4s, v14.8h\n"
      "uadalp v6.4s, v15.8h\n"
      "uadalp v7.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v8.4s}, [%x[lhs]], #16\n"
      "ld1 {v9.4s}, [%x[rhs]], #16\n"
      "dup v10.4s, %w[scale]\n"
      "dup v11.4s, v8.s[0]\n"
      "dup v8.4s, v8.s[1]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v4.4s, v4.4s, v6.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v11.4s\n"
      "add v4.4s, v4.4s, v8.4s\n"
      "add v0.4s, v0.4s, v9.4s\n"
      "add v4.4s, v4.4s, v9.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v4.4s, v4.4s\n"
      "fmul v0.4s, v0.4s, v10.4s\n"
      "fmul v4.4s, v4.4s, v10.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.4s}, [%x[result]], #16\n"
      "st1 {v4.4s}, [x0], #16\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 1,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 1, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.2s, v4.2s, v5.2s}, [%x[lhs]], #24\n"
      "ld1 {v6.2s}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v6.8b, v3.8b\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v6.8b, v5.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v4.4s}, [%x[lhs]], #16\n"
      "ld1 {v5.4s}, [%x[rhs]], #16\n"
      "dup v6.4s, %w[scale]\n"
      "dup v3.4s, v4.s[0]\n"
      "dup v7.4s, v4.s[1]\n"
      "dup v4.4s, v4.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v3.4s\n"
      "add v1.4s, v1.4s, v7.4s\n"
      "add v2.4s, v2.4s, v4.4s\n"
      "add v0.4s, v0.4s, v5.4s\n"
      "add v1.4s, v1.4s, v5.4s\n"
      "add v2.4s, v2.4s, v5.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v1.4s, v1.4s\n"
      "scvtf v2.4s, v2.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"
      "fmul v1.4s, v1.4s, v6.4s\n"
      "fmul v2.4s, v2.4s, v6.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.s}[0], [%x[result]], #4\n"
      "st1 {v1.s}[0], [x0], #4\n"
      "st1 {v2.s}[0], [x1], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 2,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 2, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.2s, v7.2s, v8.2s}, [%x[lhs]], #24\n"
      "ld1 {v9.2s, v10.2s}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v9.8b, v6.8b\n"
      "umull v12.8h, v10.8b, v6.8b\n"
      "umull v13.8h, v9.8b, v7.8b\n"
      "umull v14.8h, v10.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v8.8b\n"
      "umull v16.8h, v10.8b, v8.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v6.4s}, [%x[lhs]], #16\n"
      "ld1 {v7.4s}, [%x[rhs]], #16\n"
      "dup v8.4s, %w[scale]\n"
      "dup v9.4s, v6.s[0]\n"
      "dup v10.4s, v6.s[1]\n"
      "dup v6.4s, v6.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v9.4s\n"
      "add v2.4s, v2.4s, v10.4s\n"
      "add v4.4s, v4.4s, v6.4s\n"
      "add v0.4s, v0.4s, v7.4s\n"
      "add v2.4s, v2.4s, v7.4s\n"
      "add v4.4s, v4.4s, v7.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v2.4s, v2.4s\n"
      "scvtf v4.4s, v4.4s\n"
      "fmul v0.4s, v0.4s, v8.4s\n"
      "fmul v2.4s, v2.4s, v8.4s\n"
      "fmul v4.4s, v4.4s, v8.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v2.2s}, [x0], #8\n"
      "st1 {v4.2s}, [x1], #8\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

template <>
inline void MulKernel<
    uint8_t, float, QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 3,
    8>::Multiply(const uint8_t* lhs, const uint8_t* rhs,
                 const FusedKernelParams<QuantizedStaticPreprocessedAsFloat,
                                         RowMajor>& params,
                 float* result) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") QuantizedStaticPreprocessedAsFloatRowMajor<uint8_t, float, "
               "QuantizedStaticPreprocessedAsFloat, RowMajor, 3, 3, "
               "8>::Multiply()"
            << std::endl
            << std::flush;
#endif
#endif
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"
      "mov v8.16b, v5.16b\n"

      // 3x3 lanes loop.
      "1:"

      "ld1 {v12.8b, v13.8b, v14.8b}, [%x[rhs]], #24\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "umull v16.8h, v9.8b, v13.8b\n"
      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "umull v17.8h, v9.8b, v14.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v18.8h, v10.8b, v12.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "uadalp v0.4s, v15.8h\n"
      "uadalp v1.4s, v16.8h\n"
      "uadalp v2.4s, v17.8h\n"
      "uadalp v3.4s, v18.8h\n"
      "umull v15.8h, v10.8b, v13.8b\n"
      "umull v16.8h, v10.8b, v14.8b\n"
      "umull v17.8h, v11.8b, v12.8b\n"
      "umull v18.8h, v11.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "umull v9.8h, v11.8b, v14.8b\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"
      "uadalp v6.4s, v17.8h\n"
      "uadalp v7.4s, v18.8h\n"
      "uadalp v8.4s, v9.8h\n"

      // Loop break.
      "bgt 1b\n"

      // StaticQuantizationFloat::Prepare
      "ld1 {v9.4s}, [%x[lhs]], #16\n"
      "ld1 {v10.4s}, [%x[rhs]], #16\n"
      "dup v11.4s, %w[scale]\n"
      "dup v12.4s, v9.s[0]\n"
      "dup v13.4s, v9.s[1]\n"
      "dup v9.4s, v9.s[2]\n"

      // RowMajorOutput::Prepare
      "add x0, %x[result], %x[stride]\n"
      "add x1, x0, %x[stride]\n"

      // Reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v6.4s, v6.4s, v8.4s\n"

      // StaticQuantizationFloat::Transform
      "add v0.4s, v0.4s, v12.4s\n"
      "add v3.4s, v3.4s, v13.4s\n"
      "add v6.4s, v6.4s, v9.4s\n"
      "add v0.4s, v0.4s, v10.4s\n"
      "add v3.4s, v3.4s, v10.4s\n"
      "add v6.4s, v6.4s, v10.4s\n"
      "scvtf v0.4s, v0.4s\n"
      "scvtf v3.4s, v3.4s\n"
      "scvtf v6.4s, v6.4s\n"
      "fmul v0.4s, v0.4s, v11.4s\n"
      "fmul v3.4s, v3.4s, v11.4s\n"
      "fmul v6.4s, v6.4s, v11.4s\n"

      // RowMajorOutput::Output
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], #4\n"
      "st1 {v3.2s}, [x0], #8\n"
      "st1 {v3.s}[2], [x0], #4\n"
      "st1 {v6.2s}, [x1], #8\n"
      "st1 {v6.s}[2], [x1], #4\n"
      : [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      : [count] "r"(params.kernel.count),
        [stride] "r"(params.output_stream.stride),
        [scale] "r"(params.kernel.scale)
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc",
        "memory");
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm for arm64 requires: GEMMLOWP_NEON_64!"
#endif

#endif  // GEMMLOWP_META_QUANTIZED_MUL_KERNELS_ARM_64_H_
