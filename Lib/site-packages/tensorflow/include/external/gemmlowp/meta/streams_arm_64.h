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

#ifndef GEMMLOWP_META_STREAMS_ARM_64_H_
#define GEMMLOWP_META_STREAMS_ARM_64_H_

#ifdef GEMMLOWP_NEON_64

#include <cassert>
#include <cstdint>

namespace gemmlowp {
namespace meta {

template <>
inline void Stream<uint8_t, 1, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "movi v8.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 1x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 1x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x1.
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 1x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x2.
      "movi v0.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 1x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x3.
      "movi v0.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v0.b}[2], [%x[in]], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 1x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x4.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 1x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x5.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.b}[4], [%x[in]], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 1x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x6.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 1, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 1x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 1x7.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v0.b}[6], [%x[in]], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 2x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "v8", "v9", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 2x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x1.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "v8", "v9", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 2x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x2.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "v8", "v9", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 2x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x3.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v0.b}[2], [%x[in]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "v8", "v9", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 2x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x4.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "v8", "v9", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 2x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x5.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.b}[4], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "v8", "v9", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 2x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x6.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "v8", "v9", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 2, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 2x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 2x7.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v0.b}[6], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "st1 {v0.2s, v1.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "v8", "v9", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 3x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 3x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x1.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 3x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x2.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 3x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x3.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v0.b}[2], [%x[in]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 3x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x4.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 3x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x5.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.b}[4], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 3x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x6.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 3, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 3x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 3x7.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v0.b}[6], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 4x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 4x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x1.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "ld1 {v3.b}[0], [x2], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 4x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x2.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 4x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x3.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v0.b}[2], [%x[in]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v3.b}[2], [x2], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 4x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x4.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 4x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x5.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.b}[4], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.b}[4], [x2], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 4x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x6.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 4, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 4x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 4x7.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v0.b}[6], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v3.b}[6], [x2], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 5x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 5x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x1.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "ld1 {v3.b}[0], [x2], #1\n"
      "ld1 {v4.b}[0], [x3], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 5x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x2.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v4.h}[0], [x3], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 5x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x3.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v0.b}[2], [%x[in]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v3.b}[2], [x2], #1\n"
      "ld1 {v4.h}[0], [x3], #2\n"
      "ld1 {v4.b}[2], [x3], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 5x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x4.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 5x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x5.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.b}[4], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.b}[4], [x2], #1\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.b}[4], [x3], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 5x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x6.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.h}[2], [x3], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 5, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 5x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 5x7.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v0.b}[6], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v3.b}[6], [x2], #1\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.h}[2], [x3], #2\n"
      "ld1 {v4.b}[6], [x3], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 6x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v8",
        "v9", "v10", "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 6x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x1.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "ld1 {v3.b}[0], [x2], #1\n"
      "ld1 {v4.b}[0], [x3], #1\n"
      "ld1 {v5.b}[0], [x4], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v8",
        "v9", "v10", "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 6x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x2.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v4.h}[0], [x3], #2\n"
      "ld1 {v5.h}[0], [x4], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v8",
        "v9", "v10", "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 6x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x3.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v0.b}[2], [%x[in]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v3.b}[2], [x2], #1\n"
      "ld1 {v4.h}[0], [x3], #2\n"
      "ld1 {v4.b}[2], [x3], #1\n"
      "ld1 {v5.h}[0], [x4], #2\n"
      "ld1 {v5.b}[2], [x4], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v8",
        "v9", "v10", "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 6x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x4.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v8",
        "v9", "v10", "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 6x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x5.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.b}[4], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.b}[4], [x2], #1\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.b}[4], [x3], #1\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.b}[4], [x4], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v8",
        "v9", "v10", "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 6x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x6.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.h}[2], [x3], #2\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.h}[2], [x4], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v8",
        "v9", "v10", "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 6, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 6x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 6x7.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v0.b}[6], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v3.b}[6], [x2], #1\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.h}[2], [x3], #2\n"
      "ld1 {v4.b}[6], [x3], #1\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.h}[2], [x4], #2\n"
      "ld1 {v5.b}[6], [x4], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset),
        [additive_sum_offset] "r"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v8",
        "v9", "v10", "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 7x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 7x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x1.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "ld1 {v3.b}[0], [x2], #1\n"
      "ld1 {v4.b}[0], [x3], #1\n"
      "ld1 {v5.b}[0], [x4], #1\n"
      "ld1 {v6.b}[0], [x5], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 7x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x2.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v4.h}[0], [x3], #2\n"
      "ld1 {v5.h}[0], [x4], #2\n"
      "ld1 {v6.h}[0], [x5], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 7x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x3.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v0.b}[2], [%x[in]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v3.b}[2], [x2], #1\n"
      "ld1 {v4.h}[0], [x3], #2\n"
      "ld1 {v4.b}[2], [x3], #1\n"
      "ld1 {v5.h}[0], [x4], #2\n"
      "ld1 {v5.b}[2], [x4], #1\n"
      "ld1 {v6.h}[0], [x5], #2\n"
      "ld1 {v6.b}[2], [x5], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 7x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x4.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v6.s}[0], [x5], #4\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 7x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x5.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.b}[4], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.b}[4], [x2], #1\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.b}[4], [x3], #1\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.b}[4], [x4], #1\n"
      "ld1 {v6.s}[0], [x5], #4\n"
      "ld1 {v6.b}[4], [x5], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 7x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x6.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.h}[2], [x3], #2\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.h}[2], [x4], #2\n"
      "ld1 {v6.s}[0], [x5], #4\n"
      "ld1 {v6.h}[2], [x5], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 7, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 7x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 7x7.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v0.b}[6], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v3.b}[6], [x2], #1\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.h}[2], [x3], #2\n"
      "ld1 {v4.b}[6], [x3], #1\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.h}[2], [x4], #2\n"
      "ld1 {v5.b}[6], [x4], #1\n"
      "ld1 {v6.s}[0], [x5], #4\n"
      "ld1 {v6.h}[2], [x5], #2\n"
      "ld1 {v6.b}[6], [x5], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 0, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 0, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "add x6, x5, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 8x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "ld1 {v7.2s}, [x6], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4",
        "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 1, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 1, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "add x6, x5, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 8x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "ld1 {v7.2s}, [x6], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x1.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "ld1 {v3.b}[0], [x2], #1\n"
      "ld1 {v4.b}[0], [x3], #1\n"
      "ld1 {v5.b}[0], [x4], #1\n"
      "ld1 {v6.b}[0], [x5], #1\n"
      "ld1 {v7.b}[0], [x6], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4",
        "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 2, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 2, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "add x6, x5, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 8x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "ld1 {v7.2s}, [x6], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x2.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v4.h}[0], [x3], #2\n"
      "ld1 {v5.h}[0], [x4], #2\n"
      "ld1 {v6.h}[0], [x5], #2\n"
      "ld1 {v7.h}[0], [x6], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4",
        "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 3, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 3, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "add x6, x5, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 8x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "ld1 {v7.2s}, [x6], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x3.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], #2\n"
      "ld1 {v0.b}[2], [%x[in]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v3.b}[2], [x2], #1\n"
      "ld1 {v4.h}[0], [x3], #2\n"
      "ld1 {v4.b}[2], [x3], #1\n"
      "ld1 {v5.h}[0], [x4], #2\n"
      "ld1 {v5.b}[2], [x4], #1\n"
      "ld1 {v6.h}[0], [x5], #2\n"
      "ld1 {v6.b}[2], [x5], #1\n"
      "ld1 {v7.h}[0], [x6], #2\n"
      "ld1 {v7.b}[2], [x6], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4",
        "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 4, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 4, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "add x6, x5, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 8x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "ld1 {v7.2s}, [x6], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x4.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v6.s}[0], [x5], #4\n"
      "ld1 {v7.s}[0], [x6], #4\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4",
        "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 5, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 5, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "add x6, x5, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 8x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "ld1 {v7.2s}, [x6], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x5.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.b}[4], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.b}[4], [x2], #1\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.b}[4], [x3], #1\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.b}[4], [x4], #1\n"
      "ld1 {v6.s}[0], [x5], #4\n"
      "ld1 {v6.b}[4], [x5], #1\n"
      "ld1 {v7.s}[0], [x6], #4\n"
      "ld1 {v7.b}[4], [x6], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4",
        "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 6, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 6, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "add x6, x5, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 8x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "ld1 {v7.2s}, [x6], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x6.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.h}[2], [x3], #2\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.h}[2], [x4], #2\n"
      "ld1 {v6.s}[0], [x5], #4\n"
      "ld1 {v6.h}[2], [x5], #2\n"
      "ld1 {v7.s}[0], [x6], #4\n"
      "ld1 {v7.h}[2], [x6], #2\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4",
        "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 7, RowMajorWithSum>::Pack(
    const uint8_t* in, const RowMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout << __FILE__ << "(" << __LINE__
            << ") RowMajorWithSum<uint8_t, 8, 8, 7, RowMajorWithSum>::Pack()"
            << std::endl
            << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  asm volatile(
      "add x0, %x[in], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "add x3, x2, %x[stride]\n"
      "add x4, x3, %x[stride]\n"
      "add x5, x4, %x[stride]\n"
      "add x6, x5, %x[stride]\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store: 8x8.
      "ld1 {v0.2s}, [%x[in]], #8\n"
      "ld1 {v1.2s}, [x0], #8\n"
      "ld1 {v2.2s}, [x1], #8\n"
      "ld1 {v3.2s}, [x2], #8\n"
      "ld1 {v4.2s}, [x3], #8\n"
      "ld1 {v5.2s}, [x4], #8\n"
      "ld1 {v6.2s}, [x5], #8\n"
      "ld1 {v7.2s}, [x6], #8\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store: 8x7.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v0.h}[2], [%x[in]], #2\n"
      "ld1 {v0.b}[6], [%x[in]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v3.b}[6], [x2], #1\n"
      "ld1 {v4.s}[0], [x3], #4\n"
      "ld1 {v4.h}[2], [x3], #2\n"
      "ld1 {v4.b}[6], [x3], #1\n"
      "ld1 {v5.s}[0], [x4], #4\n"
      "ld1 {v5.h}[2], [x4], #2\n"
      "ld1 {v5.b}[6], [x4], #1\n"
      "ld1 {v6.s}[0], [x5], #4\n"
      "ld1 {v6.h}[2], [x5], #2\n"
      "ld1 {v6.b}[6], [x5], #1\n"
      "ld1 {v7.s}[0], [x6], #4\n"
      "ld1 {v7.h}[2], [x6], #2\n"
      "ld1 {v7.b}[6], [x6], #1\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "uaddw v15.8h, v15.8h, v7.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s, v7.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "ldr w0, %[multiplicative_sum_offset]\n"
      "ldr w1, %[additive_sum_offset]\n"
      "mov v0.s[0], w0\n"
      "dup v1.4s, w1\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [in] "+r"(in), [out] "+r"(out)
      : [stride] "r"(params.stride),
        [multiplicative_sum_offset] "m"(params.multiplicative_sum_offset),
        [additive_sum_offset] "m"(params.additive_sum_offset)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4",
        "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x1
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x2
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x3
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x4
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x5
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x6
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 1, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 1, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 1x8
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 1x7
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v0.b}[6], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "st1 {v0.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v8", "v0", "v1", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v8", "v9", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x1
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v8", "v9", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x2
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v8", "v9", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x3
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v8", "v9", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x4
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v8", "v9", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x5
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v8", "v9", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x6
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v8", "v9", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 2, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 2, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 2x8
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 2x7
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v0.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.h}[2], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uzp1 v2.8b, v0.8b, v1.8b\n"
      "uzp2 v3.8b, v0.8b, v1.8b\n"
      "uaddw v8.8h, v8.8h, v2.8b\n"
      "uaddw v9.8h, v9.8h, v3.8b\n"
      "st1 {v2.2s, v3.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v8", "v9", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x1
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x2
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x3
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x4
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x5
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x6
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 3, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 3, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 3x8
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 3x7
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld3 {v0.b, v1.b, v2.b}[0], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[1], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[2], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[3], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[4], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[5], [%x[in]], %x[stride]\n"
      "ld3 {v0.b, v1.b, v2.b}[6], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v10.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v8", "v9", "v10", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x1
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x2
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x3
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x4
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x5
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x6
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 4, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 4, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 4x8
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 4x7
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v4.4h, v0.4h, v2.4h\n"
      "trn2 v6.4h, v0.4h, v2.4h\n"
      "trn1 v5.4h, v1.4h, v3.4h\n"
      "trn2 v7.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v4.8b, v5.8b\n"
      "trn2 v1.8b, v4.8b, v5.8b\n"
      "trn1 v2.8b, v6.8b, v7.8b\n"
      "trn2 v3.8b, v6.8b, v7.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "st1 {v8.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x1
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x2
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x3
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x4
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x5
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x6
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 5, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 5, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 5x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 5x7
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v4.b}[6], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v5.4h, v0.4h, v2.4h\n"
      "trn2 v7.4h, v0.4h, v2.4h\n"
      "trn1 v6.4h, v1.4h, v3.4h\n"
      "trn2 v13.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v5.8b, v6.8b\n"
      "trn2 v1.8b, v5.8b, v6.8b\n"
      "trn1 v2.8b, v7.8b, v13.8b\n"
      "trn2 v3.8b, v7.8b, v13.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s}, [%x[out]], #8\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v12.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x1
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x2
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x3
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x4
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x5
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x6
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 6, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 6, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 6x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 6x7
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld1 {v4.h}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld1 {v5.h}[2], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v6.4h, v0.4h, v2.4h\n"
      "trn2 v14.4h, v0.4h, v2.4h\n"
      "trn1 v7.4h, v1.4h, v3.4h\n"
      "trn2 v15.4h, v1.4h, v3.4h\n"
      "uzp1 v16.8b, v4.8b, v5.8b\n"
      "uzp2 v17.8b, v4.8b, v5.8b\n"
      "trn1 v0.8b, v6.8b, v7.8b\n"
      "trn2 v1.8b, v6.8b, v7.8b\n"
      "trn1 v2.8b, v14.8b, v15.8b\n"
      "trn2 v3.8b, v14.8b, v15.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v16.8b\n"
      "uaddw v13.8h, v13.8h, v17.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v16.2s, v17.2s}, [%x[out]], #16\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v12.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x1
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x2
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x3
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x4
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x5
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x6
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 7, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 7, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "sub %x[stride], %x[stride], #4\n"
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 7x8
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[7], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 7x7
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "ld1 {v0.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[0], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[1], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[2], [%x[in]], %x[stride]\n"
      "ld1 {v3.s}[0], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[3], [%x[in]], %x[stride]\n"
      "ld1 {v0.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[4], [%x[in]], %x[stride]\n"
      "ld1 {v1.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[5], [%x[in]], %x[stride]\n"
      "ld1 {v2.s}[1], [%x[in]], #4\n"
      "ld3 {v4.b, v5.b, v6.b}[6], [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v7.4h, v0.4h, v2.4h\n"
      "trn2 v16.4h, v0.4h, v2.4h\n"
      "trn1 v15.4h, v1.4h, v3.4h\n"
      "trn2 v17.4h, v1.4h, v3.4h\n"
      "trn1 v0.8b, v7.8b, v15.8b\n"
      "trn2 v1.8b, v7.8b, v15.8b\n"
      "trn1 v2.8b, v16.8b, v17.8b\n"
      "trn2 v3.8b, v16.8b, v17.8b\n"
      "uaddw v8.8h, v8.8h, v0.8b\n"
      "uaddw v9.8h, v9.8h, v1.8b\n"
      "uaddw v10.8h, v10.8h, v2.8b\n"
      "uaddw v11.8h, v11.8h, v3.8b\n"
      "uaddw v12.8h, v12.8h, v4.8b\n"
      "uaddw v13.8h, v13.8h, v5.8b\n"
      "uaddw v14.8h, v14.8h, v6.8b\n"
      "st1 {v0.2s, v1.2s, v2.2s, v3.2s}, [%x[out]], #32\n"
      "st1 {v4.2s, v5.2s, v6.2s}, [%x[out]], #24\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v14.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 0, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 0, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v7.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 1, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 1, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v7.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x1
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 2, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 2, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v7.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x2
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 3, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 3, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v7.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x3
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 4, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 4, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v7.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x4
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 5, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 5, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v7.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x5
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 6, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 6, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v7.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x6
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "cc", "memory");
}

template <>
inline void Stream<uint8_t, 8, 8, 7, ColumnMajorWithSum>::Pack(
    const uint8_t* in, const ColumnMajorWithSum& params, uint8_t* out) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
  std::cout
      << __FILE__ << "(" << __LINE__
      << ") ColumnMajorWithSum<uint8_t, 8, 8, 7, ColumnMajorWithSum>::Pack()"
      << std::endl
      << std::flush;
#endif
#endif
  int params_count_copy = params.count;
  int params_stride_copy = params.stride;
  asm volatile(
      "movi v8.8h, #0\n"
      "movi v9.8h, #0\n"
      "movi v10.8h, #0\n"
      "movi v11.8h, #0\n"
      "movi v12.8h, #0\n"
      "movi v13.8h, #0\n"
      "movi v14.8h, #0\n"
      "movi v15.8h, #0\n"

      // Reduce count by leftovers.
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store - column major 8x8
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v7.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      "bne 1b\n"

      "2:"

      // Load Aggregate Store - column major 8x7
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "movi v4.8b, #0\n"
      "movi v5.8b, #0\n"
      "movi v6.8b, #0\n"
      "movi v7.8b, #0\n"
      "ld1 {v0.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v1.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v2.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v3.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v4.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v5.2s}, [%x[in]], %x[stride]\n"
      "ld1 {v6.2s}, [%x[in]], %x[stride]\n"
      "prfm pldl1keep, [%x[in]]\n"
      "trn1 v16.8b, v0.8b, v1.8b\n"
      "trn2 v17.8b, v0.8b, v1.8b\n"
      "trn1 v18.8b, v2.8b, v3.8b\n"
      "trn2 v19.8b, v2.8b, v3.8b\n"
      "trn1 v20.8b, v4.8b, v5.8b\n"
      "trn2 v21.8b, v4.8b, v5.8b\n"
      "trn1 v22.8b, v6.8b, v7.8b\n"
      "trn2 v23.8b, v6.8b, v7.8b\n"
      "trn1 v0.4h, v16.4h, v18.4h\n"
      "trn2 v2.4h, v16.4h, v18.4h\n"
      "trn1 v1.4h, v17.4h, v19.4h\n"
      "trn2 v3.4h, v17.4h, v19.4h\n"
      "trn1 v4.4h, v20.4h, v22.4h\n"
      "trn2 v6.4h, v20.4h, v22.4h\n"
      "trn1 v5.4h, v21.4h, v23.4h\n"
      "trn2 v7.4h, v21.4h, v23.4h\n"
      "trn1 v16.2s, v0.2s, v4.2s\n"
      "trn2 v20.2s, v0.2s, v4.2s\n"
      "trn1 v17.2s, v1.2s, v5.2s\n"
      "trn2 v21.2s, v1.2s, v5.2s\n"
      "trn1 v18.2s, v2.2s, v6.2s\n"
      "trn2 v22.2s, v2.2s, v6.2s\n"
      "trn1 v19.2s, v3.2s, v7.2s\n"
      "trn2 v23.2s, v3.2s, v7.2s\n"
      "uaddw v8.8h, v8.8h, v16.8b\n"
      "uaddw v9.8h, v9.8h, v17.8b\n"
      "uaddw v10.8h, v10.8h, v18.8b\n"
      "uaddw v11.8h, v11.8h, v19.8b\n"
      "uaddw v12.8h, v12.8h, v20.8b\n"
      "uaddw v13.8h, v13.8h, v21.8b\n"
      "uaddw v14.8h, v14.8h, v22.8b\n"
      "uaddw v15.8h, v15.8h, v23.8b\n"
      "st1 {v16.2s, v17.2s, v18.2s, v19.2s}, [%x[out]], #32\n"
      "st1 {v20.2s, v21.2s, v22.2s, v23.2s}, [%x[out]], #32\n"

      // Aggregator Reduction.
      "mov v0.s[0], %w[multiplicative_sum_offset]\n"
      "dup v1.4s, %w[additive_sum_offset]\n"
      "uaddlp v8.4s, v8.8h\n"
      "uaddlp v9.4s, v9.8h\n"
      "uaddlp v10.4s, v10.8h\n"
      "uaddlp v11.4s, v11.8h\n"
      "uaddlp v12.4s, v12.8h\n"
      "uaddlp v13.4s, v13.8h\n"
      "uaddlp v14.4s, v14.8h\n"
      "uaddlp v15.4s, v15.8h\n"
      "addp v8.4s, v8.4s, v9.4s\n"
      "addp v10.4s, v10.4s, v11.4s\n"
      "addp v12.4s, v12.4s, v13.4s\n"
      "addp v14.4s, v14.4s, v15.4s\n"
      "addp v8.4s, v8.4s, v10.4s\n"
      "addp v9.4s, v12.4s, v14.4s\n"
      "mul v8.4s, v8.4s, v0.s[0]\n"
      "mul v9.4s, v9.4s, v0.s[0]\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "st1 {v8.4s, v9.4s}, [%x[out]]\n"
      : [count] "+r"(params_count_copy), [stride] "+r"(params_stride_copy),
        [out] "+r"(out), [in] "+r"(in)
      : [additive_sum_offset] "r"(params.additive_sum_offset),
        [multiplicative_sum_offset] "r"(params.multiplicative_sum_offset)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "cc", "memory");
}

}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm for arm64 requires: GEMMLOWP_NEON_64!"
#endif

#endif  // GEMMLOWP_META_STREAMS_ARM_64_H_
